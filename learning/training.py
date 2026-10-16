"""Time-budgeted CPU experiment, transactional run state and bounded artifacts."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
import fcntl

import torch

from catan.state import _integer
from learning.evaluation import EvaluationPolicy, compare_evaluations, evaluate_neural
from learning.model import ActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from rl.collector import collect_rollouts
from rl.env import REWARD_SCHEME

RUN_SCHEMA_VERSION = 2
TUNE_SEED_BASE = 1_000_000_000
TEST_SEED_BASE = 2_000_000_000
_BALANCED_ROLES = ((3, 0), (4, 0), (3, 1), (4, 1), (3, 2), (4, 2), (4, 3))


def balanced_assignment(update, schedule):
    """Every seven-role/slot cycle covers each seat/opponent slot exactly once."""
    update = _integer(update, "update")
    if update < 0 or not schedule:
        raise ValueError("nonnegative update and nonempty schedule required")
    cycle, position = divmod(update, 7 * len(schedule))
    slot, role = divmod(position, 7)
    players, seat = _BALANCED_ROLES[(role + cycle) % 7]
    return players, seat, schedule[(slot + role + cycle) % len(schedule)]


def _utc():
    return datetime.now(timezone.utc)


def _atomic_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class RunConfig:
    budget_seconds: float = 7200
    max_updates: int = 100000
    games_per_update: int = 16
    active_games: int = 16
    max_actions: int = 5000
    evaluation_max_actions: int = 5000
    evaluation_batch_size: int = 8
    tune_seeds: int = 8
    test_seeds: int = 32
    evaluate_every: int = 25
    checkpoint_every: int = 5
    hidden_size: int = 128
    model_seed: int = 0
    shuffle_seed: int = 1
    training_seed_base: int = 1_000_000
    policy_seed: int = 17
    evaluation_seed_offset: int = 0
    collection_threads: int = 1
    update_threads: int = 4
    evaluation_threads: int = 1
    opponent_schedule: tuple[str, ...] = ("self",)
    tuning_opponent: str = "all"
    balanced_schedule: bool = False
    milestones: tuple[int, ...] = ()

    def __post_init__(self):
        if (
            type(self.budget_seconds) not in (int, float)
            or not 0 < self.budget_seconds <= 86400
        ):
            raise ValueError(
                "budget_seconds must be finite, positive and at most one day"
            )
        for name in self.__dataclass_fields__:
            if name in (
                "budget_seconds",
                "opponent_schedule",
                "tuning_opponent",
                "balanced_schedule",
                "milestones",
            ):
                continue
            value = _integer(getattr(self, name), name)
            object.__setattr__(self, name, value)
            if (
                name
                not in (
                    "model_seed",
                    "shuffle_seed",
                    "policy_seed",
                    "training_seed_base",
                    "evaluation_seed_offset",
                )
                and value < 1
            ):
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.evaluation_seed_offset <= 100000000:
            raise ValueError("evaluation_seed_offset outside reserved ranges")
        if (
            self.training_seed_base < 0
            or self.training_seed_base + self.max_updates * self.games_per_update
            >= TUNE_SEED_BASE
        ):
            raise ValueError(
                "training seed range would overlap reserved evaluation seeds"
            )
        if (
            self.games_per_update * self.max_actions > 160000
            or self.evaluation_batch_size * self.evaluation_max_actions > 80000
        ):
            raise ValueError(
                "requested collector row capacity exceeds experiment memory bounds"
            )
        if (
            self.max_updates > 100000
            or self.tune_seeds > 10000
            or self.test_seeds > 10000
        ):
            raise ValueError("evaluation seed count too large")
        if not isinstance(self.opponent_schedule, (tuple, list)):
            raise ValueError("opponent_schedule must be a tuple/list")
        schedule = tuple(self.opponent_schedule)
        if not schedule or any(
            name not in ("self", "greedy", "random", "previous", "frozen_original")
            for name in schedule
        ):
            raise ValueError(
                "use self/greedy/random/previous/frozen_original opponent names"
            )
        object.__setattr__(self, "opponent_schedule", schedule)
        if schedule != ("self",) and self.max_actions < 8:
            raise ValueError(
                "mixed training needs max_actions >= 8 so every learner seat can act"
            )
        if self.tuning_opponent not in ("all", "greedy"):
            raise ValueError("tuning_opponent must be all or greedy")
        if type(self.balanced_schedule) is not bool:
            raise ValueError("balanced_schedule must be Boolean")
        if not isinstance(self.milestones, (tuple, list)):
            raise ValueError("milestones must be a tuple/list")
        milestones = tuple(_integer(u, "milestone") for u in self.milestones)
        if (
            len(milestones) > 32
            or tuple(sorted(set(milestones))) != milestones
            or any(not 1 <= u <= self.max_updates for u in milestones)
        ):
            raise ValueError(
                "milestones need at most32 ascending unique in-range updates"
            )
        object.__setattr__(self, "milestones", milestones)


class TrainingRun:
    """Application owns CPU threads. One writer via Linux container file lock.

    Model+manifest commits refer to a complete learner state. Interrupted work
    after the latest commit is replayed with the same seed schedule. The original
    absolute experiment deadline remains in force after resume.
    """

    def __init__(
        self,
        root,
        *,
        config=None,
        ppo_config=None,
        resume=False,
        initial_checkpoint=None,
        reward_scheme=REWARD_SCHEME,
        env_factory=None,
    ):
        if resume and initial_checkpoint is not None:
            raise ValueError(
                "resume restores committed weights, not a new initial checkpoint"
            )
        self.env_factory = env_factory
        self.root = Path(root)
        if resume:
            if not (self.root / "state.json").is_file():
                raise ValueError("no committed run state to resume")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        self._lock = (self.root / "writer.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("another process owns this training run") from None
        self.stopping = False
        self._opponents = {}
        try:
            if resume:
                self._restore()
                if self.updater.reward_scheme != reward_scheme:
                    raise ValueError("resume reward profile differs from stored run")
            else:
                self.config = RunConfig() if config is None else config
                if not isinstance(self.config, RunConfig):
                    raise ValueError("config must be RunConfig")
                self._threads(self.config.collection_threads)
                self.initial_source = None
                model = ActorCritic(
                    self.config.hidden_size, seed=self.config.model_seed
                )
                if initial_checkpoint is not None:
                    source = PPOUpdater.load(initial_checkpoint)
                    if (
                        source.model.hidden_size != self.config.hidden_size
                        or source.model.seed != self.config.model_seed
                    ):
                        raise ValueError(
                            "initial checkpoint architecture/seed differs from config"
                        )
                    model = source.model
                    self.initial_source = {
                        "sha256": sha256(
                            Path(initial_checkpoint).read_bytes()
                        ).hexdigest(),
                        "source_update": source.updates,
                        "optimizer_reset": True,
                    }
                    del source
                self.updater = PPOUpdater(
                    model,
                    PPOConfig(gae_lambda=1.0) if ppo_config is None else ppo_config,
                    shuffle_seed=self.config.shuffle_seed,
                    reward_scheme=reward_scheme,
                )
                started = _utc()
                self.started = started.isoformat()
                self.deadline = (
                    started + timedelta(seconds=self.config.budget_seconds)
                ).isoformat()
                self.totals = {
                    key: 0
                    for key in (
                        "episodes",
                        "wins",
                        "cutoffs",
                        "actions",
                        "discards",
                        "max_payload_bytes",
                        "learner_rows",
                        "learner_wins",
                    )
                }
                self.timings = {
                    key: 0.0
                    for key in ("collection", "update", "evaluation", "checkpoint")
                }
                self.best_update = 0
                self.best_wins = -1
                self._committed_update = -1
                self._checkpoint = self._previous = None
                (self.root / "checkpoints").mkdir()
                (self.root / "evaluation").mkdir()
                self.save(status="initializing")
            self._stop_at = perf_counter() + max(
                0, (datetime.fromisoformat(self.deadline) - _utc()).total_seconds()
            )
        except Exception:
            self.close()
            raise

    def close(self):
        if not self._lock.closed:
            self._lock.close()

    @staticmethod
    def _threads(count):
        if torch.get_num_threads() != count:
            torch.set_num_threads(count)

    def _checkpoint_name(self, update):
        return f"u{update:08d}.pt"

    def _restore(self):
        state = _read(self.root / "state.json")
        if state["schema"] not in (1, RUN_SCHEMA_VERSION):
            raise ValueError("incompatible training run schema")
        self.config = RunConfig(**state["config"])
        self.initial_source = state.get("initial_source")
        self.started, self.deadline = state["started_utc"], state["deadline_utc"]
        if datetime.fromisoformat(self.deadline) <= datetime.fromisoformat(
            self.started
        ):
            raise ValueError("invalid run deadline")
        self._checkpoint, self._previous = (
            state["checkpoint"],
            state["previous_checkpoint"],
        )
        if not re.fullmatch(r"u\d{8}\.pt", self._checkpoint):
            raise ValueError("invalid checkpoint filename")
        path = self.root / "checkpoints" / self._checkpoint
        if sha256(path.read_bytes()).hexdigest() != state["checkpoint_sha256"]:
            raise ValueError("run checkpoint digest mismatch")
        self._threads(self.config.collection_threads)
        self.updater = PPOUpdater.load(path)
        if self.updater.updates != state["update"]:
            raise ValueError("model/run version mismatch")
        self._committed_update = self.updater.updates
        self.best_update, self.best_wins = (
            state["best_update"],
            state["best_tuning_wins"],
        )
        if not 0 <= self.best_update <= self.updater.updates:
            raise ValueError("invalid selected checkpoint version")
        if self.updater.config != PPOConfig(**state["ppo_config"]):
            raise ValueError("model/run PPO configuration mismatch")
        self.totals, self.timings = state["totals"], state["timings_seconds"]
        self.totals.setdefault("learner_rows", self.totals["actions"])
        self.totals.setdefault("learner_wins", self.totals["wins"])
        if (
            self.totals["episodes"]
            != self.updater.updates * self.config.games_per_update
            or self.totals["wins"] + self.totals["cutoffs"] != self.totals["episodes"]
        ):
            raise ValueError("inconsistent training episode totals")
        # Retain only committed metrics. A killed writer can leave a partial line
        # or fully logged updates that have not reached the manifest commit yet.
        path = self.root / "metrics.jsonl"
        if path.exists():
            retained = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                if record["update"] <= self._committed_update:
                    retained.append(json.dumps(record, allow_nan=False))
            path.write_text("".join(line + "\n" for line in retained), encoding="utf-8")

    def _state(self, status):
        path = self.root / "checkpoints" / self._checkpoint
        return {
            "schema": RUN_SCHEMA_VERSION,
            "status": status,
            "started_utc": self.started,
            "deadline_utc": self.deadline,
            "updated_utc": _utc().isoformat(),
            "config": asdict(self.config),
            "initial_source": self.initial_source,
            "observation_schema": self.updater.model.schema[1],
            "reward_scheme": self.updater.reward_scheme,
            "ppo_config": asdict(self.updater.config),
            "update": self.updater.updates,
            "optimizer_steps": self.updater.optimizer_steps,
            "checkpoint": self._checkpoint,
            "previous_checkpoint": self._previous,
            "checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
            "best_update": self.best_update,
            "best_tuning_wins": self.best_wins,
            "totals": dict(self.totals),
            "timings_seconds": dict(self.timings),
        }

    def save(self, *, status="running"):
        start = perf_counter()
        update = self.updater.updates
        name = self._checkpoint_name(update)
        if update != self._committed_update:
            self.updater.save(self.root / "checkpoints" / name)
            self._previous, self._checkpoint = self._checkpoint, name
        curve_path = self.root / "evaluation" / f"milestone-u{update:08d}.json"
        if update in self.config.milestones and curve_path.exists():
            curve = _read(curve_path)
            curve["checkpoint_sha256"] = sha256(
                (self.root / "checkpoints" / name).read_bytes()
            ).hexdigest()
            _atomic_json(curve_path, curve)
        self.timings["checkpoint"] += perf_counter() - start
        _atomic_json(self.root / "state.json", self._state(status))
        self._committed_update = update
        keep = {
            self._checkpoint,
            self._previous,
            self._checkpoint_name(0),
            self._checkpoint_name(self.best_update),
        }
        keep.update(
            self._checkpoint_name(u) for u in self.config.milestones if u <= update
        )
        for path in (self.root / "checkpoints").iterdir():
            if re.fullmatch(r"u\d{8}\.pt", path.name) and path.name not in keep:
                path.unlink()

    def _evaluate(self, updater, seeds, path):
        self._threads(self.config.evaluation_threads)
        start = perf_counter()
        result = evaluate_neural(
            updater.policy(),
            seeds,
            max_actions=self.config.evaluation_max_actions,
            batch_size=self.config.evaluation_batch_size,
            policy_seed=self.config.policy_seed,
            env_factory=self.env_factory,
        )
        self.timings["evaluation"] += perf_counter() - start
        _atomic_json(path, result)
        return result

    def initialize_baselines(self):
        tune = self.root / "evaluation" / "tuning-initial.json"
        test = self.root / "evaluation" / "test-initial.json"
        if tune.exists() and test.exists():
            if self.best_wins < 0:
                self.best_wins = self._selection_wins(_read(tune))
                self.save()
            return
        if self.updater.updates != 0:
            raise ValueError("baseline missing after training started")
        self._evaluate(
            self.updater,
            range(
                TUNE_SEED_BASE + self.config.evaluation_seed_offset,
                TUNE_SEED_BASE
                + self.config.evaluation_seed_offset
                + self.config.tune_seeds,
            ),
            tune,
        )
        self._evaluate(
            self.updater,
            range(
                TEST_SEED_BASE + self.config.evaluation_seed_offset,
                TEST_SEED_BASE
                + self.config.evaluation_seed_offset
                + self.config.test_seeds,
            ),
            test,
        )
        self.best_wins = self._selection_wins(_read(tune))
        self.save()

    def _selection_wins(self, result):
        if self.config.tuning_opponent == "all":
            return result["summary"]["wins"]
        return sum(
            result["groups"][f"{players}p-vs-greedy"]["wins"] for players in (3, 4)
        )

    def _collection_policy(self, update, players):
        learner = self.updater.policy()
        schedule = self.config.opponent_schedule
        if self.config.balanced_schedule:
            expected, seat, name = balanced_assignment(update, schedule)
            if players != expected:
                raise ValueError("player count differs from balanced schedule")
        else:
            name = schedule[update % len(schedule)]
            seat = (update // 2) % players
        if schedule == ("self",):
            return learner, None, name, learner.tag
        if name == "self":
            return learner, (seat,), name, learner.tag
        opponent = name
        if name in ("previous", "frozen_original"):
            initial = self._checkpoint_name(0)
            filename = (
                initial
                if name == "frozen_original" or (update // len(schedule)) % 2 == 0
                else (self._previous or initial)
            )
            # Only two frozen CPU models: initial and one retained previous generation.
            self._opponents = {
                k: v for k, v in self._opponents.items() if k in (initial, filename)
            }
            if filename not in self._opponents:
                self._opponents[filename] = PPOUpdater.load(
                    self.root / "checkpoints" / filename
                ).policy()
            opponent = self._opponents[filename]
        callback = EvaluationPolicy(learner, opponent, seat)
        return callback, (seat,), name, callback.reference.tag

    def step(self):
        self._threads(self.config.collection_threads)
        update = self.updater.updates
        base = self.config.training_seed_base + update * self.config.games_per_update
        start = perf_counter()
        players = (
            balanced_assignment(update, self.config.opponent_schedule)[0]
            if self.config.balanced_schedule
            else 3 + update % 2
        )
        callback, recorded, opponent, opponent_tag = self._collection_policy(
            update, players
        )
        batch = collect_rollouts(
            callback,
            range(base, base + self.config.games_per_update),
            n_players=players,
            max_actions=self.config.max_actions,
            batch_size=self.config.active_games,
            policy_seed=self.config.policy_seed,
            record_players=recorded,
            policy_tag=self.updater.policy().tag,
            env_factory=self.env_factory,
        )
        collection = perf_counter() - start
        self._threads(self.config.update_threads)
        start = perf_counter()
        stats = self.updater.update(batch)
        updating = perf_counter() - start
        episodes = [asdict(e) for e in batch.episodes]
        wins = sum(e["terminated"] for e in episodes)
        count, discards, payload = (
            len(batch),
            len(batch.discards.rows),
            batch.payload_bytes,
        )
        del batch
        self.totals["episodes"] += len(episodes)
        self.totals["wins"] += wins
        self.totals["cutoffs"] += len(episodes) - wins
        self.totals["actions"] += sum(e["actions"] for e in episodes)
        self.totals["learner_rows"] += count
        self.totals["learner_wins"] += sum(
            e["terminated"] and (recorded is None or e["winner"] in recorded)
            for e in episodes
        )
        self.totals["discards"] += discards
        self.totals["max_payload_bytes"] = max(
            self.totals["max_payload_bytes"], payload
        )
        self.timings["collection"] += collection
        self.timings["update"] += updating
        record = {
            **asdict(stats),
            "players": players,
            "opponent": opponent,
            "opponent_tag": opponent_tag,
            "learner_seat": None if recorded is None else recorded[0],
            "engine_actions": sum(e["actions"] for e in episodes),
            "episodes": episodes,
            "collection_seconds": collection,
            "update_seconds": updating,
            "discards": discards,
            "payload_bytes": payload,
        }
        update = self.updater.updates
        if update % self.config.evaluate_every == 0 or update in self.config.milestones:
            self.progress(record, status="tuning_evaluation")
            result = self._evaluate(
                self.updater,
                range(
                    TUNE_SEED_BASE + self.config.evaluation_seed_offset,
                    TUNE_SEED_BASE
                    + self.config.evaluation_seed_offset
                    + self.config.tune_seeds,
                ),
                self.root / "evaluation" / "tuning-latest.json",
            )
            score = self._selection_wins(result)
            if update in self.config.milestones:
                _atomic_json(
                    self.root / "evaluation" / f"milestone-u{update:08d}.json",
                    {
                        "update": update,
                        "checkpoint": self._checkpoint_name(update),
                        "evaluation": result,
                    },
                )
            if score > self.best_wins:
                self.best_update, self.best_wins = update, score
                _atomic_json(self.root / "evaluation" / "tuning-best.json", result)
            record["tuning"] = result["summary"]
            record["tuning_groups"] = result["groups"]
        with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, allow_nan=False) + "\n")
        if (
            update % self.config.checkpoint_every == 0
            or update % self.config.evaluate_every == 0
            or update in self.config.milestones
        ):
            self.save()
        self.progress(record)
        return record

    def _progress_allowance(self):
        return {
            "deadline_utc": self.deadline,
            "remaining_seconds": max(
                0, (datetime.fromisoformat(self.deadline) - _utc()).total_seconds()
            ),
        }

    def progress(self, record=None, *, status="running"):
        data = {
            "status": status,
            "updated_utc": _utc().isoformat(),
            "started_utc": self.started,
            "update": self.updater.updates,
            "committed_update": self._committed_update,
            **self._progress_allowance(),
            "totals": dict(self.totals),
            "timings_seconds": dict(self.timings),
            "best_update": self.best_update,
            "best_tuning_wins": self.best_wins,
        }
        if record is not None:
            data["latest"] = {
                key: value for key, value in record.items() if key != "episodes"
            }
        _atomic_json(self.root / "progress.json", data)
        return data

    def finalize(self):
        self.save(status="evaluating_final")
        # Selection is fixed using tuning wins before reading any final results.
        best = PPOUpdater.load(
            self.root / "checkpoints" / self._checkpoint_name(self.best_update)
        )
        seeds = range(
            TEST_SEED_BASE + self.config.evaluation_seed_offset,
            TEST_SEED_BASE
            + self.config.evaluation_seed_offset
            + self.config.test_seeds,
        )
        selected = self._evaluate(
            best, seeds, self.root / "evaluation" / "test-selected.json"
        )
        latest = (
            selected
            if self.best_update == self.updater.updates
            else self._evaluate(
                self.updater, seeds, self.root / "evaluation" / "test-latest.json"
            )
        )
        if self.best_update == self.updater.updates:
            _atomic_json(self.root / "evaluation" / "test-latest.json", latest)
        baseline = _read(self.root / "evaluation" / "test-initial.json")
        result = {
            "schema": RUN_SCHEMA_VERSION,
            "status": "completed",
            "started_utc": self.started,
            "finished_utc": _utc().isoformat(),
            "deadline_utc": self.deadline,
            "elapsed_seconds": (
                _utc() - datetime.fromisoformat(self.started)
            ).total_seconds(),
            "update": self.updater.updates,
            "optimizer_steps": self.updater.optimizer_steps,
            "selected_update": self.best_update,
            "selected_tuning_wins": self.best_wins,
            "config": asdict(self.config),
            "ppo_config": asdict(self.updater.config),
            "initial_source": self.initial_source,
            "totals": dict(self.totals),
            "timings_seconds": dict(self.timings),
            "selected_checkpoint": self._checkpoint_name(self.best_update),
            "latest_checkpoint": self._checkpoint,
            "baseline_groups": baseline["groups"],
            "selected_groups": selected["groups"],
            "latest_groups": latest["groups"],
            "selected_comparison": compare_evaluations(baseline, selected),
            "latest_comparison": compare_evaluations(baseline, latest),
            "selected_greedy_comparison": compare_evaluations(
                baseline, selected, opponent="greedy"
            ),
            "latest_greedy_comparison": compare_evaluations(
                baseline, latest, opponent="greedy"
            ),
            "note": "Primary checkpoint selected on tuning seeds; latest is a predeclared secondary comparison. Test seeds never select checkpoints. Self-play wins measure completion, not strength.",
        }
        result["config"]["opponent_schedule"] = list(self.config.opponent_schedule)
        result["config"]["milestones"] = list(self.config.milestones)
        _atomic_json(self.root / "final-report.json", result)
        self.save(status="completed")
        self.progress(status="completed")
        return result

    def run(self):
        if (self.root / "final-report.json").exists():
            return _read(self.root / "final-report.json")
        self.progress(status="baseline_evaluation")
        self.initialize_baselines()
        while (
            not self.stopping
            and self.updater.updates < self.config.max_updates
            and perf_counter() < self._stop_at
            and _utc() < datetime.fromisoformat(self.deadline)
        ):
            record = self.step()
            print(
                json.dumps(
                    {
                        key: record[key]
                        for key in (
                            "update",
                            "players",
                            "rows",
                            "collection_seconds",
                            "update_seconds",
                            "approximate_kl",
                            "entropy",
                        )
                    }
                ),
                flush=True,
            )
        if self.stopping:
            self.save(status="paused")
            return self.progress(status="paused")
        return self.finalize()

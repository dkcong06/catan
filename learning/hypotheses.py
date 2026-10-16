"""Four-variant experiment support using existing PPO and transactional runs."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np

from catan.state import Phase, _integer
from learning.evaluation import EvaluationPolicy
from learning.imitation import Demonstrations
from learning.ppo import PPOConfig, PPOUpdater
from learning.training import RunConfig, TrainingRun, _atomic_json
from rl.collector import ReferencePolicy, collect_rollouts
from rl.hypotheses import HypothesisEnvFactory, PROGRESS_REWARD

ARMS = ("B", "P", "R", "K")
OPPONENTS = ("greedy", "frozen_original", "greedy", "random")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    training_seconds: float = 480
    replicates: int = 3
    test_seeds: int = 64
    teacher_segments: int = 512
    teacher_validation_segments: int = 64
    games_per_update: int = 8
    max_actions: int = 5000
    max_updates: int = 10000
    total_target_seconds: int = 10800

    def __post_init__(self):
        if (
            type(self.training_seconds) not in (float, int)
            or not isfinite(self.training_seconds)
            or not 0 < self.training_seconds <= 3600
        ):
            raise ValueError("training_seconds outside experiment bounds")
        for name in self.__dataclass_fields__:
            if name != "training_seconds":
                object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not 1 <= self.replicates <= 5 or not 1 <= self.test_seeds <= 256:
            raise ValueError("unsupported replicate/test count")
        if not 2 <= self.teacher_segments <= 2048 or self.teacher_segments % 2:
            raise ValueError("teacher segments must be even, from two to 2048")
        if (
            not 2 <= self.teacher_validation_segments <= 512
            or self.teacher_validation_segments % 2
        ):
            raise ValueError("validation segments must be even, from two to 512")
        if not 1 <= self.games_per_update <= 16 or not 16 <= self.max_actions <= 5000:
            raise ValueError("unsupported cohort/action cap")
        if (
            not 1 <= self.max_updates <= 10000
            or self.total_target_seconds <= self.training_seconds * self.replicates * 4
        ):
            raise ValueError("session target must reserve final evaluation time")


@dataclass(frozen=True, slots=True)
class CaseSpec:
    arm: str
    replicate: int
    setup_charge_seconds: float
    fixed_checkpoint_sha256: str

    def __post_init__(self):
        if self.arm not in ARMS or not 0 <= _integer(self.replicate, "replicate") < 5:
            raise ValueError("unknown arm/replicate")
        if not isfinite(self.setup_charge_seconds) or self.setup_charge_seconds < 0:
            raise ValueError("invalid preparation charge")


def generate_setup_teachers(root, config):
    root = Path(root)
    root.mkdir()
    started = perf_counter()
    shards = []
    for role, count, seed_base in (
        ("training", config.teacher_segments, 260000000),
        ("validation", config.teacher_validation_segments, 260100000),
        ("main_validation", 32, 260200000),
    ):
        for players in (3, 4):
            width = 4 if role == "main_validation" else 64
            for start in range(0, count // 2, width):
                seeds = tuple(
                    range(
                        seed_base + (players - 3) * 10000 + start,
                        seed_base
                        + (players - 3) * 10000
                        + min(start + width, count // 2),
                    )
                )
                cap = (
                    max(128, config.max_actions)
                    if role == "main_validation"
                    else 4 * players
                )
                batch = collect_rollouts(
                    ReferencePolicy("greedy"),
                    seeds,
                    n_players=players,
                    max_actions=cap,
                    batch_size=min(8, len(seeds)),
                    policy_seed=17,
                    env_factory=HypothesisEnvFactory(),
                )
                data = Demonstrations.from_rollouts(batch)
                if role == "main_validation":
                    rows = np.flatnonzero(
                        ~np.isin(
                            data.phases, (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)
                        )
                    )
                    data = data.select(rows)
                name = f"{role}-{players}p-{start:04d}.npz"
                data.save(root / name)
                shards.append(
                    dict(
                        file=name,
                        role=role,
                        players=players,
                        seeds=seeds,
                        max_actions=cap,
                        rows=len(data),
                        sha256=digest(root / name),
                        episodes=[asdict(e) for e in batch.episodes],
                    )
                )
                del data, batch
    result = dict(
        schema=1,
        teacher="reference:greedy-v1",
        generation_seconds=perf_counter() - started,
        observation_schema=3,
        shards=shards,
        note="Setup segments are deliberately capped; only main-validation segments play full games.",
    )
    _atomic_json(root / "manifest.json", result)
    return result


class HypothesisRun(TrainingRun):
    """Parent owns checkpoints/metrics/locks; this run uses charged active time.

    Every completed cohort commits a checkpoint. Explicit pause is cooperative:
    complete the active cohort, persist remaining time, then close the writer.
    The inherited deadline is informational only for this separately versioned run.
    """

    def __init__(
        self,
        root,
        experiment,
        spec,
        fixed_checkpoint,
        *,
        initial_checkpoint=None,
        resume=False,
    ):
        self.experiment, self.spec = experiment, spec
        self.remaining = experiment.training_seconds - spec.setup_charge_seconds
        self._active_since = perf_counter()
        self._profile = asdict(spec)
        self._experiment_profile = asdict(experiment)
        if self.remaining <= 0:
            raise ValueError("preparation exhausted the fixed training allowance")
        if digest(fixed_checkpoint) != spec.fixed_checkpoint_sha256:
            raise ValueError("fixed opponent checkpoint digest changed")
        self.fixed = PPOUpdater.load(fixed_checkpoint).policy()
        self.factory = HypothesisEnvFactory(
            knowledge=spec.arm == "K", point_progress=spec.arm == "R"
        )
        config = RunConfig(
            budget_seconds=self.remaining,
            max_updates=experiment.max_updates,
            games_per_update=experiment.games_per_update,
            active_games=experiment.games_per_update,
            max_actions=experiment.max_actions,
            evaluation_max_actions=experiment.max_actions,
            tune_seeds=1,
            test_seeds=experiment.test_seeds,
            evaluate_every=100000,
            checkpoint_every=1,
            hidden_size=self.fixed.model.hidden_size,
            model_seed=self.fixed.model.seed,
            shuffle_seed=501 + spec.replicate,
            training_seed_base=400000000 + spec.replicate * 1000000,
            evaluation_seed_offset=200000 + spec.replicate * 1000,
            opponent_schedule=OPPONENTS,
            tuning_opponent="greedy",
        )
        super().__init__(
            root,
            config=config,
            ppo_config=PPOConfig(gae_lambda=1),
            resume=resume,
            initial_checkpoint=initial_checkpoint,
            reward_scheme=PROGRESS_REWARD if spec.arm == "R" else "terminal_win_v1",
            env_factory=self.factory,
        )
        if self.config != config or self.updater.config != PPOConfig(gae_lambda=1):
            self.close()
            raise ValueError("stored training configuration differs from experiment")
        # Loading/initialization costs count too, without charging paused downtime.
        self.remaining = self._remaining()
        self._active_since = None
        if not resume:
            self.save(status="ready")

    def _remaining(self):
        return self.remaining - (
            perf_counter() - self._active_since if self._active_since is not None else 0
        )

    def _restore(self):
        saved = read(self.root / "state.json")
        profile = saved["hypotheses"]
        if (
            profile["schema"] != 1
            or profile["case"] != self._profile
            or profile["experiment"] != self._experiment_profile
        ):
            raise ValueError("resume experiment/case configuration changed")
        remaining = profile["remaining_training_seconds"]
        if (
            type(remaining) not in (int, float)
            or not isfinite(remaining)
            or remaining < 0
            or remaining
            > self.experiment.training_seconds - self.spec.setup_charge_seconds
        ):
            raise ValueError("invalid remaining training allowance")
        # Time spent loading the committed model is part of resumed active work.
        self.remaining = remaining
        super()._restore()

    def _state(self, status):
        state = super()._state(status)
        current = self._remaining()
        state["hypotheses"] = dict(
            schema=1,
            case=self._profile,
            experiment=self._experiment_profile,
            remaining_training_seconds=max(0.0, current),
            cohort_overrun_seconds=max(0.0, -current),
            clock="charged_active_v1",
            inherited_deadline_is_informational=True,
        )
        return state

    def _collection_policy(self, update, players):
        learner = self.updater.policy()
        name = OPPONENTS[update % len(OPPONENTS)]
        opponent = self.fixed if name == "frozen_original" else name
        seat = (update // 2) % players
        self.env_factory = self.factory.for_observer(seat)
        callback = EvaluationPolicy(learner, opponent, seat)
        return callback, (seat,), name, callback.reference.tag

    def progress(self, record=None, *, status="running"):
        from learning.training import _utc

        result = dict(
            status=status,
            updated_utc=_utc().isoformat(),
            arm=self.spec.arm,
            replicate=self.spec.replicate,
            update=self.updater.updates,
            committed_update=self._committed_update,
            remaining_training_seconds=max(0.0, self._remaining()),
            totals=dict(self.totals),
            timings_seconds=dict(self.timings),
        )
        if record is not None:
            result["latest"] = {k: v for k, v in record.items() if k != "episodes"}
        _atomic_json(self.root / "progress.json", result)
        return result

    def train(self, should_pause=lambda: False):
        if read(self.root / "state.json")["status"] == "trained":
            return self.progress(status="trained")
        self._active_since = perf_counter()
        while self._remaining() > 0 and self.updater.updates < self.config.max_updates:
            if should_pause():
                break
            record = self.step()
            print(
                json.dumps(
                    dict(
                        event="update",
                        arm=self.spec.arm,
                        replicate=self.spec.replicate,
                        update=record["update"],
                        rows=record["rows"],
                        remaining_seconds=max(0.0, self._remaining()),
                    )
                ),
                flush=True,
            )
        status = "paused" if should_pause() else "trained"
        self.remaining = self._remaining()
        self._active_since = None
        self.save(status=status)
        return self.progress(status=status)

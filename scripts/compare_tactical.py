"""Fresh corrected teaching and a finite matched PPO-retention pilot."""

import argparse
from dataclasses import asdict, dataclass
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch

from catan.validation import _fingerprint, assert_invariants
from learning.evaluation import (
    EvaluationPaused,
    EvaluationPolicy,
    compare_evaluations,
    evaluate_neural,
)
from learning.imitation import Demonstrations
from learning.model import ActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from learning.tactics import LessonTrainer, ReplayAnchor, TacticalLessons, score_lessons
from learning.training import _atomic_json, balanced_assignment
from rl.collector import Discards, _discard_logs, collect_rollouts
from rl.env import CatanEnv
from rl.policies import greedy_policy, tactical_policy
from rl.tactics import winning_actions


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): digest(path)
        for folder in ("catan", "rl", "learning", "scripts")
        for path in sorted((root / folder).glob("*.py"))
    }


def generate_lessons(directory, *, seed, games, ordinary_per_game=64, max_actions=5000):
    """Natural legal trajectories with bounded ordinary and winning row buckets.

    Alternating legacy/corrected behavior broadens visited locations. Labels
    always come from the corrected teacher; no inspected diagnostic suite is used.
    Hash-priority sampling uses no gameplay/policy RNG draws.
    """
    if any(
        type(v) is not int for v in (seed, games, ordinary_per_game, max_actions)
    ) or not (
        0 <= seed <= 2**32 - 129
        and 1 <= games <= 128
        and 1 <= ordinary_per_game <= 64
        and 16 <= max_actions <= 5000
    ):
        raise ValueError("invalid finite lesson collection")
    directory = Path(directory)
    rows, provenance, episodes = [], [], []
    for game in range(games):
        board_seed = seed + game
        players = 3 + game % 2
        env = CatanEnv(max_actions=max_actions)
        request = env.reset(board_seed, n_players=players)
        streams = [
            Random(f"tactical-lessons-v1:{board_seed}:{p}") for p in range(players)
        ]
        ordinary, tactical = [], []
        while request is not None:
            wins = winning_actions(request)
            label = (
                int(np.flatnonzero(wins)[0])
                if wins.any()
                else greedy_policy(request, Random(0))
            )
            if request.mask is None or np.count_nonzero(request.mask) > 1:
                record = (request, label, wins)
                info = dict(
                    seed=board_seed,
                    players=players,
                    step=env.actions,
                    player=request.player,
                    phase=request.phase.name,
                )
                if wins.any():
                    priority = sha256(repr(info).encode()).hexdigest()
                    tactical.append((priority, record, info))
                    if len(tactical) > 64:
                        tactical.sort(key=lambda item: item[0])
                        tactical.pop()
                else:
                    priority = sha256(repr(info).encode()).hexdigest()
                    ordinary.append((priority, record, info))
                    if len(ordinary) > ordinary_per_game:
                        ordinary.sort(key=lambda item: item[0])
                        ordinary.pop()
            behavior = greedy_policy if game % 2 == 0 else tactical_policy
            choice = behavior(request, streams[request.player])
            env.step(choice, player=request.player)
            request = env.decision()
        state = env.snapshot()
        assert_invariants(state)
        episodes.append(
            dict(
                seed=board_seed,
                players=players,
                actions=env.actions,
                terminated=env.terminated,
                truncated=env.truncated,
                fingerprint=sha256(
                    repr([_fingerprint(state, stream) for stream in streams]).encode()
                ).hexdigest(),
            )
        )
        selected = [(record, info) for _, record, info in ordinary + tactical]
        selected.sort(key=lambda pair: pair[1]["step"])
        for record, info in selected:
            rows.append(record)
            provenance.append(info)
    if not rows or len(rows) > 16000:
        raise ValueError("lesson collection outside finite row allowance")
    observations, masks, actions, phases, wins = [], [], [], [], []
    discard_rows, hands, counts, logs = [], [], [], []
    for i, (request, label, targets) in enumerate(rows):
        observations.append(request.observation)
        masks.append(
            request.mask if request.mask is not None else np.zeros(254, dtype=np.bool_)
        )
        actions.append(-1 if request.mask is None else label)
        phases.append(int(request.phase))
        wins.append(targets)
        if request.mask is None:
            discard_rows.append(i)
            hands.append(request.discard_hand)
            counts.append(label)
            logs.append(_discard_logs(request.discard_hand, label))
    data = Demonstrations(
        np.asarray(observations, dtype=np.float32),
        np.asarray(masks, dtype=np.bool_),
        np.asarray(actions, dtype=np.int16),
        np.asarray(phases, dtype=np.int8),
        Discards(
            np.asarray(discard_rows, dtype=np.int32),
            np.asarray(hands, dtype=np.int8).reshape(-1, 5),
            np.asarray(counts, dtype=np.int8).reshape(-1, 5),
            np.asarray(logs, dtype=np.float32).reshape(-1, 5),
        ),
        "reference:tactical-v2:corrected-lessons-v1",
    )
    lessons = TacticalLessons(data, np.asarray(wins, dtype=np.bool_))
    lessons.save(directory)
    _atomic_json(directory / "provenance.json", provenance)
    _atomic_json(
        directory / "collection.json",
        dict(
            schema=1,
            seed=seed,
            games=games,
            rows=len(lessons),
            tactical_rows=int(lessons.tactical.sum()),
            ordinary_per_game=ordinary_per_game,
            max_actions=max_actions,
            episodes=episodes,
        ),
    )
    return TacticalLessons.load(directory)


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    teacher_games: int = 64
    validation_games: int = 16
    warmup_passes: int = 16
    replicates: int = 2
    cohorts: int = 28
    games_per_cohort: int = 8
    final_seeds: int = 8
    max_actions: int = 5000
    anchor_coefficient: float = 0.1
    seed: int = 3_600_000_000

    def __post_init__(self):
        for name, low, high in (
            ("teacher_games", 1, 128),
            ("validation_games", 1, 32),
            ("warmup_passes", 1, 32),
            ("replicates", 1, 3),
            ("cohorts", 1, 56),
            ("games_per_cohort", 1, 8),
            ("final_seeds", 1, 32),
            ("max_actions", 16, 5000),
            ("seed", 0, 3_800_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if (
            type(self.anchor_coefficient) not in (int, float)
            or not 0 < self.anchor_coefficient <= 1
        ):
            raise ValueError("invalid anchor coefficient")


def _save_anchor(anchor, path):
    if anchor is not None:
        temporary = path.with_suffix(".pending")
        torch.save(anchor.state_dict(), temporary)
        temporary.replace(path)


def run_comparison(output, config=None, *, resume=False):
    output = Path(output)
    if resume:
        protocol = json.loads((output / "protocol.json").read_text())
        saved = ComparisonConfig(**protocol["config"])
        if config is not None and config != saved:
            raise ValueError("resume recipe changed")
        config = saved
        if source_hashes() != protocol["sources"]:
            raise ValueError("resume sources changed")
    else:
        config = ComparisonConfig() if config is None else config
        output.mkdir(parents=True, exist_ok=False)
        protocol = dict(
            schema=1,
            config=asdict(config),
            sources=source_hashes(),
            teacher="tactical-v2",
            original_models_modified=False,
            checkpoint_primary="last",
            anchor_batch=32,
            acceptance="Warmup success criterion: >=90% tactical validation argmax. Fixed retention pilot also runs below this threshold as a diagnostic, never as proof of successful teaching. No automatic promotion.",
            training_seed=config.seed + 10_000_000,
            validation_seed=config.seed + 1_000_000,
            final_seed=config.seed + 20_000_000,
            ppo=asdict(PPOConfig(gae_lambda=1)),
            warmup=dict(learning_rate=1e-4, batch_size=128),
            schedule=["tactical", "greedy", "random", "greedy"],
        )
        _atomic_json(output / "protocol.json", protocol)
    with (output / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = perf_counter()
        torch.set_num_threads(1)
        datasets = []
        for name, seed, games in (
            ("train", config.seed, config.teacher_games),
            ("validation", config.seed + 1_000_000, config.validation_games),
        ):
            path = output / name
            _atomic_json(
                output / "progress.json", dict(status="collecting lessons", split=name)
            )
            datasets.append(
                TacticalLessons.load(path)
                if (path / "collection.json").exists()
                else generate_lessons(
                    path, seed=seed, games=games, max_actions=config.max_actions
                )
            )
        train, validation = datasets
        if not train.tactical.any() or not validation.tactical.any():
            raise ValueError(
                "pilot needs fresh training and validation winning locations"
            )
        data_hashes = {
            f"{name}/{filename}": digest(output / name / filename)
            for name in ("train", "validation")
            for filename in (
                "demonstrations.npz",
                "winning-sets.npz",
                "provenance.json",
                "collection.json",
            )
        }
        manifest = output / "data-hashes.json"
        if manifest.exists() and json.loads(manifest.read_text()) != data_hashes:
            raise ValueError("lesson inputs changed")
        _atomic_json(manifest, data_hashes)
        all_cases, comparisons = [], []
        for rep in range(config.replicates):
            root = output / f"r{rep}"
            root.mkdir(exist_ok=True)
            taught_path = root / "taught.pt"
            if not taught_path.exists():
                model = ActorCritic(seed=9100 + rep)
                trainer = LessonTrainer(model, seed=9200 + rep)
                curve = [
                    dict(
                        passes=0,
                        train=score_lessons(model, train),
                        validation=score_lessons(model, validation),
                    )
                ]
                for iteration in range(1, config.warmup_passes + 1):
                    trainer.fit(train)
                    if iteration in (1, 2, 4, 8, 16, config.warmup_passes):
                        curve.append(
                            dict(
                                passes=iteration,
                                steps=trainer.steps,
                                train=score_lessons(model, train),
                                validation=score_lessons(model, validation),
                            )
                        )
                        _atomic_json(root / "learning-curve.json", curve)
                        _atomic_json(
                            output / "progress.json",
                            dict(status="teaching", replicate=rep, passes=iteration),
                        )
                # Persist warm-up recipe/Adam/shuffle, separate from fresh PPO Adam.
                torch.save(
                    dict(
                        optimizer=trainer.optimizer.state_dict(),
                        generator=trainer.generator.get_state(),
                        steps=trainer.steps,
                    ),
                    root / "warmup-state.pt",
                )
                _atomic_json(root / "learning-curve.json", curve)
                PPOUpdater(
                    model, PPOConfig(gae_lambda=1), shuffle_seed=9300 + rep
                ).save(taught_path)
            reports = {}
            for arm in ("taught", "ppo", "anchored"):
                path = root / arm
                path.mkdir(exist_ok=True)
                cursor = path / "state.json"
                state = json.loads(cursor.read_text()) if cursor.exists() else None
                updater = PPOUpdater.load(
                    path / state["checkpoint"] if state else taught_path
                )
                anchor = (
                    ReplayAnchor(
                        train,
                        coefficient=config.anchor_coefficient,
                        batch_size=32,
                        seed=9400 + rep,
                    )
                    if arm == "anchored"
                    else None
                )
                if state:
                    if digest(path / state["checkpoint"]) != state["checkpoint_hash"]:
                        raise ValueError("committed checkpoint changed")
                    if anchor is not None:
                        if digest(path / state["anchor"]) != state["anchor_hash"]:
                            raise ValueError("committed replay state changed")
                        anchor.load_state_dict(
                            torch.load(path / state["anchor"], weights_only=True)
                        )
                metrics = state["metrics"] if state else []
                if len(metrics) != updater.updates or [
                    m["update"] for m in metrics
                ] != list(range(1, updater.updates + 1)):
                    raise ValueError("committed metrics/checkpoint counters disagree")
                limit = 0 if arm == "taught" else config.cohorts
                while updater.updates < limit:
                    if (output / "pause.request").exists():
                        _atomic_json(
                            output / "progress.json",
                            dict(status="paused", replicate=rep, arm=arm),
                        )
                        return None
                    update = updater.updates
                    players, seat, opponent = balanced_assignment(
                        update, protocol["schedule"]
                    )
                    seeds = range(
                        config.seed
                        + 10_000_000
                        + rep * 100_000
                        + update * config.games_per_cohort,
                        config.seed
                        + 10_000_000
                        + rep * 100_000
                        + (update + 1) * config.games_per_cohort,
                    )
                    policy = updater.policy()
                    collect_start = perf_counter()
                    batch = collect_rollouts(
                        EvaluationPolicy(policy, opponent, seat),
                        seeds,
                        n_players=players,
                        policy_seed=17,
                        max_actions=config.max_actions,
                        batch_size=config.games_per_cohort,
                        record_players=(seat,),
                        policy_tag=policy.tag,
                    )
                    collection_seconds = perf_counter() - collect_start
                    update_start = perf_counter()
                    stats = updater.update(batch, anchor=anchor)
                    entry = dict(
                        update=updater.updates,
                        players=players,
                        seat=seat,
                        opponent=opponent,
                        rows=len(batch),
                        collection_seconds=collection_seconds,
                        update_seconds=perf_counter() - update_start,
                        stats=asdict(stats),
                        episodes=[asdict(e) for e in batch.episodes],
                    )
                    del batch
                    if updater.updates in (1, 2, 4, 8, 16, config.cohorts):
                        entry["validation"] = score_lessons(updater.model, validation)
                    metrics.append(entry)
                    checkpoint = f"u{updater.updates:04d}.pt"
                    updater.save(path / checkpoint)
                    anchor_file = f"a{updater.updates:04d}.pt"
                    _save_anchor(anchor, path / anchor_file)
                    state = dict(
                        checkpoint=checkpoint,
                        checkpoint_hash=digest(path / checkpoint),
                        metrics=metrics,
                    )
                    if anchor is not None:
                        state.update(
                            anchor=anchor_file, anchor_hash=digest(path / anchor_file)
                        )
                    _atomic_json(cursor, state)
                    _atomic_json(
                        output / "progress.json",
                        dict(
                            status="training",
                            replicate=rep,
                            arm=arm,
                            update=updater.updates,
                            total_cohorts=config.cohorts,
                        ),
                    )
                evaluation_path = path / "evaluation.json"
                _atomic_json(
                    output / "progress.json",
                    dict(
                        status="evaluating",
                        replicate=rep,
                        arm=arm,
                        update=updater.updates,
                    ),
                )
                if evaluation_path.exists():
                    report = json.loads(evaluation_path.read_text())
                else:
                    checkpoint_hash = (
                        digest(path / state["checkpoint"])
                        if state
                        else digest(taught_path)
                    )
                    if (output / "pause.request").exists():
                        _atomic_json(
                            output / "progress.json",
                            dict(status="paused", replicate=rep, arm=arm),
                        )
                        return None
                    try:
                        report = evaluate_neural(
                            updater.policy(),
                            range(
                                config.seed + 20_000_000 + rep * 1000,
                                config.seed
                                + 20_000_000
                                + rep * 1000
                                + config.final_seeds,
                            ),
                            max_actions=config.max_actions,
                            batch_size=8,
                            progress_path=path / "evaluation-progress.json",
                            progress_identity=checkpoint_hash,
                            should_pause=lambda: (output / "pause.request").exists(),
                        )
                    except EvaluationPaused:
                        _atomic_json(
                            output / "progress.json",
                            dict(status="paused", replicate=rep, arm=arm),
                        )
                        return None
                    _atomic_json(evaluation_path, report)
                reports[arm] = report
                final = score_lessons(updater.model, validation)
                summary = dict(
                    replicate=rep,
                    arm=arm,
                    validation=final,
                    games=report["summary"],
                    updates=updater.updates,
                    metrics=metrics,
                    taught_hash=digest(taught_path),
                )
                _atomic_json(path / "summary.json", summary)
                all_cases.append(summary)
            comparisons.append(
                dict(
                    replicate=rep,
                    retention=compare_evaluations(reports["ppo"], reports["anchored"]),
                    greedy=compare_evaluations(
                        reports["ppo"], reports["anchored"], opponent="greedy"
                    ),
                    ppo_vs_taught=compare_evaluations(
                        reports["taught"], reports["ppo"]
                    ),
                    anchor_vs_taught=compare_evaluations(
                        reports["taught"], reports["anchored"]
                    ),
                )
            )
        result = dict(
            schema=1,
            config=asdict(config),
            cases=all_cases,
            comparisons=comparisons,
            seconds_this_invocation=perf_counter() - started,
            data_hashes=data_hashes,
            no_promotion=True,
            source_hashes=source_hashes(),
        )
        _atomic_json(output / "comparison.json", result)
        _atomic_json(output / "progress.json", dict(status="complete"))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = (
        ComparisonConfig(
            teacher_games=4,
            validation_games=2,
            warmup_passes=1,
            replicates=1,
            cohorts=1,
            final_seeds=1,
            seed=3_400_000_000,
        )
        if args.smoke
        else None
    )
    run_comparison(args.output, config, resume=args.resume)


if __name__ == "__main__":
    main()

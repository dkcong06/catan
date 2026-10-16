"""Fixed-cohort control of direct critic gradients, with early development curves."""

import argparse
from dataclasses import asdict, dataclass
import fcntl
from pathlib import Path
import signal
from time import perf_counter
from learning.cohorts import FixedCohortRun as CriticRun

from learning.evaluation import (
    EvaluationPaused,
    _summary,
    compare_evaluations,
    evaluate_neural,
)
from learning.hypotheses import digest, read
from learning.ppo import PPOConfig, PPOUpdater
from learning.training import (
    RunConfig,
    TrainingRun,
    _atomic_json,
    TEST_SEED_BASE,
)

ARMS = ("coupled", "head_only")


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    updates: int = 56
    replicates: int = 3
    development_seeds: int = 4
    test_seeds: int = 16
    games_per_update: int = 8
    max_actions: int = 5000
    seed_offset: int = 7_000_000
    training_seed_base: int = 500_000_000

    def __post_init__(self):
        for name, low, high in (
            ("updates", 1, 256),
            ("replicates", 1, 5),
            ("development_seeds", 1, 32),
            ("test_seeds", 1, 128),
            ("games_per_update", 1, 16),
            ("max_actions", 16, 5000),
            ("seed_offset", 0, 99_000_000),
            ("training_seed_base", 0, 990_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} outside comparison bounds")

    @property
    def milestones(self):
        return tuple(
            sorted(
                {
                    u
                    for u in (1, 2, 4, 8, 16, 32, 64, 128, 256, self.updates)
                    if u <= self.updates
                }
            )
        )


def case_config(config, replicate, source):
    return RunConfig(
        budget_seconds=86400,  # Informational in fixed_cohorts_v1, never a time budget.
        max_updates=config.updates,
        games_per_update=config.games_per_update,
        active_games=config.games_per_update,
        max_actions=config.max_actions,
        evaluation_max_actions=config.max_actions,
        tune_seeds=config.development_seeds,
        test_seeds=config.test_seeds,
        evaluate_every=100000,
        checkpoint_every=1,
        hidden_size=source.model.hidden_size,
        model_seed=source.model.seed,
        shuffle_seed=901 + replicate,
        training_seed_base=config.training_seed_base + replicate * 1_000_000,
        evaluation_seed_offset=config.seed_offset + replicate * 1000,
        opponent_schedule=("greedy", "frozen_original", "greedy", "random"),
        tuning_opponent="greedy",
        balanced_schedule=True,
        milestones=config.milestones,
    )


def _pooled(reports):
    return {
        **reports[0],
        "seeds": [s for report in reports for s in report["seeds"]],
        "records": [r for report in reports for r in report["records"]],
    }


def run_comparison(output, initial_checkpoint, *, config=None, resume=False):
    output, initial_checkpoint = Path(output), Path(initial_checkpoint)
    if resume:
        protocol = read(output / "protocol.json")
        saved = ComparisonConfig(**protocol["config"])
        if config is not None and saved != config:
            raise ValueError("resume configuration changed")
        config = saved
    else:
        config = ComparisonConfig() if config is None else config
        if not isinstance(config, ComparisonConfig):
            raise ValueError("config must be ComparisonConfig")
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / "writer.lock").open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another writer owns comparison") from None
        source = PPOUpdater.load(initial_checkpoint)
        if (
            source.model.schema != (1, 2, 1)
            or source.reward_scheme != "terminal_win_v1"
        ):
            raise ValueError("comparison requires the original model1/win profile")
        expected = dict(
            schema=1,
            config=asdict(config),
            source_sha256=digest(initial_checkpoint),
            clock="fixed_cohorts_v1",
            arms=list(ARMS),
            final_checkpoint_primary=True,
            milestones=list(config.milestones),
            policy_seed=17,
        )
        if resume:
            if protocol != expected:
                raise ValueError("source/protocol changed")
        else:
            protocol = expected
            _atomic_json(output / "protocol.json", protocol)
        if (output / "comparison.json").exists():
            result = read(output / "comparison.json")
            if any(
                digest(output / path) != checksum
                for path, checksum in result["checkpoints"].items()
            ):
                raise ValueError("completed checkpoint changed")
            return result
        state = dict(status="running", trained_cases=0, evaluated_models=0)
        started = perf_counter()
        previous_seconds = (
            read(output / "state.json").get("active_seconds", 0) if resume else 0
        )

        def save_state(status="running"):
            state.update(
                status=status,
                active_seconds=previous_seconds + perf_counter() - started,
            )
            _atomic_json(output / "state.json", state)
            return state

        def pause():
            return (output / "pause.request").exists()

        save_state()
        for rep in range(config.replicates):
            for arm in ARMS[rep % 2 :] + ARMS[: rep % 2]:
                if pause():
                    return save_state("paused")
                root = output / f"{arm}-r{rep}"
                state["current"] = root.name
                save_state()
                run = CriticRun(
                    root,
                    config=case_config(config, rep, source),
                    ppo_config=PPOConfig(
                        gae_lambda=1, critic_updates_trunk=arm == "coupled"
                    ),
                    initial_checkpoint=None
                    if (root / "state.json").exists()
                    else initial_checkpoint,
                    resume=(root / "state.json").exists(),
                )
                try:
                    progress = run.train(pause)
                    if progress["status"] == "paused":
                        return save_state("paused")
                    committed = read(root / "state.json")
                    _atomic_json(root / "training-summary.json", committed)
                finally:
                    run.close()
                state["trained_cases"] += 1
                save_state()
        (output / "evaluation").mkdir(exist_ok=True)
        evaluations = {}
        for rep in range(config.replicates):
            seeds = range(
                TEST_SEED_BASE + config.seed_offset + rep * 1000,
                TEST_SEED_BASE + config.seed_offset + rep * 1000 + config.test_seeds,
            )
            for arm in ("original", *ARMS):
                if pause():
                    return save_state("paused")
                label = f"{arm}-r{rep}"
                checkpoint = initial_checkpoint
                if arm != "original":
                    root = output / label
                    checkpoint = (
                        root / "checkpoints" / read(root / "state.json")["checkpoint"]
                    )
                checksum = digest(checkpoint)
                path = output / "evaluation" / f"{label}.json"
                state["current"] = label
                save_state()
                if path.exists():
                    report = read(path)
                    if (
                        report["checkpoint_sha256"] != checksum
                        or report["seeds"] != list(seeds)
                        or report["max_actions"] != config.max_actions
                    ):
                        raise ValueError("cached final evaluation changed")
                else:
                    policy = PPOUpdater.load(checkpoint).policy()
                    TrainingRun._threads(1)
                    try:
                        report = evaluate_neural(
                            policy,
                            seeds,
                            max_actions=config.max_actions,
                            progress_path=path.with_suffix(".progress.json"),
                            progress_identity=checksum,
                            should_pause=pause,
                        )
                    except EvaluationPaused:
                        return save_state("paused")
                    report["checkpoint_sha256"] = checksum
                    _atomic_json(path, report)
                evaluations[label] = report
                state["evaluated_models"] += 1
                save_state()
        pooled = {
            arm: _pooled(
                [evaluations[f"{arm}-r{rep}"] for rep in range(config.replicates)]
            )
            for arm in ("original", *ARMS)
        }
        result = dict(
            status="completed",
            protocol=protocol,
            evaluations=len(evaluations),
            summary={
                arm: {
                    opp: _summary(
                        [r for r in report["records"] if r["opponent"] == opp]
                    )
                    for opp in ("greedy", "random")
                }
                for arm, report in pooled.items()
            },
            comparison={
                opp: compare_evaluations(
                    pooled["coupled"], pooled["head_only"], opponent=opp
                )
                for opp in ("greedy", "random")
            },
            versus_original={
                arm: {
                    opp: compare_evaluations(
                        pooled["original"], pooled[arm], opponent=opp
                    )
                    for opp in ("greedy", "random")
                }
                for arm in ARMS
            },
            per_replicate=[
                {
                    opp: compare_evaluations(
                        evaluations[f"coupled-r{rep}"],
                        evaluations[f"head_only-r{rep}"],
                        opponent=opp,
                    )
                    for opp in ("greedy", "random")
                }
                for rep in range(config.replicates)
            ],
            active_seconds=previous_seconds + perf_counter() - started,
            checkpoints={
                str(path.relative_to(output)): digest(path)
                for arm in ARMS
                for rep in range(config.replicates)
                for path in (output / f"{arm}-r{rep}" / "checkpoints").glob("*.pt")
            },
            note="Small fixed-cohort pilot, final checkpoint primary, no promotion. "
            "Intervals condition on trained models and whole board clusters.",
        )
        if digest(initial_checkpoint) != protocol["source_sha256"]:
            raise ValueError("source checkpoint changed")
        _atomic_json(output / "comparison.json", result)
        save_state("completed")
        return result
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    for name in ComparisonConfig.__dataclass_fields__:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    args = parser.parse_args()
    values = {
        name: getattr(args, name)
        for name in ComparisonConfig.__dataclass_fields__
        if getattr(args, name) is not None
    }
    if args.resume and values:
        parser.error("resume uses stored configuration")

    def request_pause(*_):
        (args.output / "pause.request").touch()

    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    run_comparison(
        args.output,
        args.initial_checkpoint,
        config=ComparisonConfig(**values) if not args.resume else None,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

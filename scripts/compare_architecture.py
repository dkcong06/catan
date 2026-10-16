"""Two predeclared comparisons: economic inputs, then fresh MLP versus graph."""

import argparse
from dataclasses import asdict
import fcntl
from pathlib import Path
import signal
from time import perf_counter

import torch

from learning.architecture import ArchitectureConfig, ArchitectureRun, teachers, warmup
from learning.evaluation import (
    EvaluationPaused,
    _summary,
    compare_evaluations,
    evaluate_neural,
)
from learning.graph import GraphActorCritic
from learning.hypotheses import digest, read
from learning.model import EconomicActorCritic
from learning.ppo import PPOUpdater
from learning.training import TrainingRun, TEST_SEED_BASE, _atomic_json
from rl.economics import EconomicEnvFactory
from rl.env import CatanEnv
from scripts.compare_training import combine_evaluations

STAGES = {"features": ("zero", "economic"), "graph": ("mlp", "graph")}


def run_comparison(
    output, original_checkpoint, economic_checkpoint, *, config=None, resume=False
):
    output, original_checkpoint, economic_checkpoint = map(
        Path, (output, original_checkpoint, economic_checkpoint)
    )
    if resume:
        protocol = read(output / "protocol.json")
        stored = ArchitectureConfig(**protocol["config"])
        if config is not None and config != stored:
            raise ValueError("stored comparison configuration changed")
        config = stored
    else:
        config = ArchitectureConfig() if config is None else config
        if not isinstance(config, ArchitectureConfig):
            raise ValueError("use ArchitectureConfig")
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / "writer.lock").open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another writer owns architecture comparison") from None
        TrainingRun._threads(1)
        original, economic = (
            PPOUpdater.load(original_checkpoint),
            PPOUpdater.load(economic_checkpoint),
        )
        if (
            original.model.schema != (1, 2, 1)
            or economic.model.schema != (3, 4, 1)
            or economic.updates
            or economic.optimizer_steps
            or economic.optimizer.state
            or original.config != economic.config
        ):
            raise ValueError(
                "requires preserved original and untrained economic migration"
            )
        migrated = EconomicActorCritic.from_base(original.model)
        if any(
            not torch.equal(value, economic.model.state_dict()[name])
            for name, value in migrated.state_dict().items()
        ):
            raise ValueError("economic initialization is not the exact zero migration")
        del migrated
        expected = dict(
            schema=1,
            config=asdict(config),
            clock="fixed_cohorts_v1",
            original_sha256=digest(original_checkpoint),
            economic_sha256=digest(economic_checkpoint),
            ppo_config=asdict(original.config),
            stages={k: list(v) for k, v in STAGES.items()},
            graph=dict(model=4, observation=4, node_width=12, rounds=2, hidden=112),
            mlp=dict(model=3, observation=4, hidden=128),
            common_teacher=dict(
                training_base=850000000 + config.seed_shift,
                validation_base=850100000 + config.seed_shift,
                player_offset=10000,
                policy_seed=17,
            ),
            warmup=dict(
                epochs=config.warmup_epochs,
                learning_rate=0.0001,
                minibatch=256,
                phase_balanced=True,
                critic_labels=False,
            ),
            feature_initialization="identical economic-u0 weights, zero suffix versus enabled suffix",
            graph_initialization="fresh seed1701+rep; common fixed teacher passes; fresh PPO Adam",
            final_checkpoint_primary=True,
            policy_seed=17,
            primary="greedy real-win difference, paired whole boards; no automatic promotion",
            confidence=0.95,
            adjusted_two_primary_confidence=0.975,
            time_note="approximately three-hour target; finite opportunities, no run-until-favorable",
        )
        if resume:
            if protocol != expected:
                raise ValueError("source/protocol changed")
        else:
            protocol = expected
            _atomic_json(output / "protocol.json", protocol)
        if (output / "comparison.json").exists():
            result = read(output / "comparison.json")
            if any(digest(output / p) != h for p, h in result["checkpoints"].items()):
                raise ValueError("completed checkpoint changed")
            return result
        state = dict(status="running", trained_cases=0, evaluated_models=0)
        previous = read(output / "state.json").get("active_seconds", 0) if resume else 0
        started = perf_counter()

        def save(status="running"):
            state.update(
                status=status, active_seconds=previous + perf_counter() - started
            )
            _atomic_json(output / "state.json", state)
            return state

        def pause():
            return (output / "pause.request").exists()

        save()
        results = {}
        for stage, arms in STAGES.items():
            stage_root = output / stage
            stage_root.mkdir(exist_ok=True)
            state["stage"] = stage
            if stage == "graph":
                state["current"] = "common-teacher"
                save()
                TrainingRun._threads(1)
                manifest = teachers(stage_root / "teacher", config, pause)
                if manifest is None:
                    return save("paused")
            for rep in range(config.replicates):
                for arm in arms[rep % 2 :] + arms[: rep % 2]:
                    if pause():
                        return save("paused")
                    label = f"{arm}-r{rep}"
                    state["current"] = label
                    save()
                    initial = economic_checkpoint
                    if stage == "graph":
                        inputs = stage_root / "inputs"
                        inputs.mkdir(exist_ok=True)
                        cold, initial = (
                            inputs / f"{label}-cold.pt",
                            inputs / f"{label}-warm.pt",
                        )
                        if not cold.exists():
                            TrainingRun._threads(1)
                            model = (
                                EconomicActorCritic(128, seed=1701 + rep)
                                if arm == "mlp"
                                else GraphActorCritic(112, seed=1701 + rep)
                            )
                            PPOUpdater(
                                model, original.config, shuffle_seed=1901 + rep
                            ).save(cold)
                        TrainingRun._threads(4)
                        warmed = warmup(
                            cold,
                            initial,
                            stage_root / "teacher",
                            manifest,
                            epochs=config.warmup_epochs,
                            shuffle_seed=1901 + rep,
                            should_pause=pause,
                        )
                        if warmed is None:
                            return save("paused")
                    root = stage_root / label
                    model = PPOUpdater.load(initial).model
                    run = ArchitectureRun(
                        root,
                        enabled=arm != "zero",
                        original_checkpoint=original_checkpoint,
                        config=config.run_config(stage, rep, model),
                        ppo_config=original.config,
                        initial_checkpoint=None
                        if (root / "state.json").exists()
                        else initial,
                        resume=(root / "state.json").exists(),
                        on_progress=lambda _: save(),
                    )
                    try:
                        progress = run.train(pause)
                        if progress["status"] == "paused":
                            return save("paused")
                        _atomic_json(
                            root / "training-summary.json", read(root / "state.json")
                        )
                    finally:
                        run.close()
                    state["trained_cases"] += 1
                    save()
            evaluations = {}
            (stage_root / "evaluation").mkdir(exist_ok=True)
            offset = (11000000 if stage == "features" else 12000000) + config.seed_shift
            for rep in range(config.replicates):
                seeds = list(
                    range(
                        TEST_SEED_BASE + offset + rep * 1000,
                        TEST_SEED_BASE + offset + rep * 1000 + config.test_seeds,
                    )
                )
                for arm in ("original", *arms):
                    if pause():
                        return save("paused")
                    label = f"{arm}-r{rep}"
                    state["current"] = "final-" + label
                    save()
                    checkpoint = original_checkpoint
                    if arm != "original":
                        root = stage_root / label
                        checkpoint = (
                            root
                            / "checkpoints"
                            / read(root / "state.json")["checkpoint"]
                        )
                    checksum = digest(checkpoint)
                    path = stage_root / "evaluation" / f"{label}.json"
                    factory = (
                        CatanEnv
                        if arm == "original"
                        else EconomicEnvFactory(arm != "zero")
                    )
                    identity = dict(
                        checkpoint_sha256=checksum,
                        stage=stage,
                        arm=arm,
                        economic_enabled=arm != "zero",
                        observation=2 if arm == "original" else 4,
                    )
                    if path.exists():
                        report = read(path)
                        if (
                            report["identity"] != identity
                            or report["seeds"] != seeds
                            or report["max_actions"] != config.max_actions
                        ):
                            raise ValueError("cached final evaluation changed")
                    else:
                        TrainingRun._threads(1)
                        try:
                            report = evaluate_neural(
                                PPOUpdater.load(checkpoint).policy(),
                                seeds,
                                max_actions=config.max_actions,
                                env_factory=factory,
                                progress_path=path.with_suffix(".progress.json"),
                                progress_identity=str(identity),
                                should_pause=pause,
                            )
                        except EvaluationPaused:
                            return save("paused")
                        report.update(identity=identity, checkpoint_sha256=checksum)
                        _atomic_json(path, report)
                    evaluations[label] = report
                    state["evaluated_models"] += 1
                    save()
            pooled = {
                arm: combine_evaluations(
                    [evaluations[f"{arm}-r{r}"] for r in range(config.replicates)]
                )
                for arm in ("original", *arms)
            }
            result = dict(
                summary={
                    arm: {
                        opp: _summary(
                            [r for r in report["records"] if r["opponent"] == opp]
                        )
                        for opp in ("greedy", "random")
                    }
                    for arm, report in pooled.items()
                },
                primary={
                    opp: compare_evaluations(
                        pooled[arms[0]], pooled[arms[1]], opponent=opp
                    )
                    for opp in ("greedy", "random")
                },
                adjusted_primary=compare_evaluations(
                    pooled[arms[0]],
                    pooled[arms[1]],
                    opponent="greedy",
                    confidence=0.975,
                ),
                versus_original={
                    arm: {
                        opp: compare_evaluations(
                            pooled["original"], pooled[arm], opponent=opp
                        )
                        for opp in ("greedy", "random")
                    }
                    for arm in arms
                },
                per_replicate=[
                    {
                        opp: compare_evaluations(
                            evaluations[f"{arms[0]}-r{rep}"],
                            evaluations[f"{arms[1]}-r{rep}"],
                            opponent=opp,
                        )
                        for opp in ("greedy", "random")
                    }
                    for rep in range(config.replicates)
                ],
            )
            results[stage] = result
            _atomic_json(stage_root / "comparison.json", result)
        result = dict(
            status="completed",
            protocol=protocol,
            stages=results,
            active_seconds=previous + perf_counter() - started,
            checkpoints={
                str(p.relative_to(output)): digest(p) for p in output.rglob("*.pt")
            },
            note="Final fixed checkpoint primary; intervals condition on three trained models and board clusters.",
        )
        if digest(original_checkpoint) != protocol["original_sha256"]:
            raise ValueError("original changed")
        _atomic_json(output / "comparison.json", result)
        save("completed")
        return result
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--economic-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    for name in ArchitectureConfig.__dataclass_fields__:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    args = parser.parse_args()
    values = {
        name: getattr(args, name)
        for name in ArchitectureConfig.__dataclass_fields__
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
        args.original_checkpoint,
        args.economic_checkpoint,
        config=ArchitectureConfig(**values) if not args.resume else None,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

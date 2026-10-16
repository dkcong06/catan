"""Run the predeclared placement/reward/resource-knowledge comparison on CPU."""

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
from time import perf_counter

import torch

from learning.evaluation import EvaluationPaused, compare_evaluations, evaluate_neural
from learning.hypotheses import (
    ARMS,
    OPPONENTS,
    CaseSpec,
    ExperimentConfig,
    HypothesisRun,
    digest,
    generate_setup_teachers,
    read,
)
from learning.imitation import ImitationConfig
from learning.model import KnowledgeActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from learning.training import _atomic_json, _utc
from rl.hypotheses import HypothesisEnvFactory
from scripts.compare_training import combine_evaluations, pretrain


def run_experiment(output, source=None, *, config=None, resume=False):
    conversion_started = perf_counter()
    output = Path(output)
    if resume:
        if config is not None:
            raise ValueError("resume uses the stored experiment configuration")
        if not (output / "protocol.json").is_file():
            raise ValueError("no committed experiment protocol")
    else:
        config = ExperimentConfig() if config is None else config
        if not isinstance(config, ExperimentConfig):
            raise ValueError("use ExperimentConfig")
        if source is None:
            raise ValueError("new experiment requires the original checkpoint")
        # Check the restricted checkpoint before creating output.
        torch.set_num_threads(1)
        original = PPOUpdater.load(source)
        migrated = KnowledgeActorCritic.from_base(original.model)
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / "experiment.writer.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("another writer owns this experiment") from None
    try:
        if resume:
            protocol = read(output / "protocol.json")
            config = ExperimentConfig(**protocol["config"])
            source = Path(protocol["source_path"]) if source is None else Path(source)
            if digest(source) != protocol["source_sha256"]:
                raise ValueError("original checkpoint changed")
            if digest(output / "initial.pt") != protocol["fixed_checkpoint_sha256"]:
                raise ValueError("fixed opponent checkpoint changed")
            if (output / "comparison.json").exists():
                return read(output / "comparison.json")
        else:
            PPOUpdater(migrated, PPOConfig(gae_lambda=1), shuffle_seed=1).save(
                output / "initial.pt"
            )
            protocol = dict(
                schema=1,
                started_utc=_utc().isoformat(),
                config=asdict(config),
                source_path=str(Path(source).absolute()),
                source_sha256=digest(source),
                source_update=original.updates,
                fixed_checkpoint_sha256=digest(output / "initial.pt"),
                conversion_charge_seconds=perf_counter() - conversion_started,
                arms=list(ARMS),
                opponents=list(OPPONENTS),
                observation_schema=3,
                model_schema=2,
                information_profile="public_bank_observer_bounds_v1",
                point_reward="terminal_win_net_vp_0.1_v1",
                final_checkpoint_primary=True,
                alpha_primary_family=0.05,
                practical_greedy_gain=0.02,
                random_noninferiority_margin=0.05,
                teacher_recipe=dict(
                    segments=config.teacher_segments,
                    validation_segments=config.teacher_validation_segments,
                    epochs=2,
                    learning_rate=0.0001,
                ),
                test_boards=[
                    list(
                        range(
                            2000200000 + r * 1000,
                            2000200000 + r * 1000 + config.test_seeds,
                        )
                    )
                    for r in range(config.replicates)
                ],
                note="Seed replicates share original weights. Final comparisons are conditional on the trained models; test outcomes do not select checkpoints.",
            )
            _atomic_json(output / "protocol.json", protocol)
            del migrated, original
        torch.set_num_threads(1)
        protocol_hash = digest(output / "protocol.json")
        previous = (
            read(output / "state.json") if (output / "state.json").exists() else {}
        )
        if previous and previous["protocol_sha256"] != protocol_hash:
            raise ValueError("experiment protocol digest changed")
        accrued = previous.get("session_active_seconds", 0.0)
        invocation_started = perf_counter()
        phase, current = "preparing", None
        reports = []
        evaluations = {}
        last_saved = 0.0

        def save_job(status=None):
            nonlocal last_saved
            last_saved = perf_counter()
            result = dict(
                schema=1,
                status=status or phase,
                current=current,
                protocol_sha256=protocol_hash,
                updated_utc=_utc().isoformat(),
                session_active_seconds=accrued + perf_counter() - invocation_started,
                trained_cases=len(reports),
                planned_cases=4 * config.replicates,
                evaluated_models=len(evaluations),
            )
            _atomic_json(output / "state.json", result)
            return result

        def paused():
            requested = (output / "pause.request").exists()
            if requested or perf_counter() - last_saved > 10:
                save_job()
            return requested

        if paused():
            return save_job("paused")
        manifest_path = output / "teacher/manifest.json"
        if manifest_path.exists():
            manifest = read(manifest_path)
            for shard in manifest["shards"]:
                if digest(output / "teacher" / shard["file"]) != shard["sha256"]:
                    raise ValueError("teacher shard changed")
        else:
            manifest = generate_setup_teachers(output / "teacher", config)
        order = [
            (arm, r)
            for r in range(config.replicates)
            for arm in ARMS[r % 4 :] + ARMS[: r % 4]
        ]
        for arm, replicate in order:
            current = dict(arm=arm, replicate=replicate)
            root = output / f"{arm}-r{replicate}"
            report_path = root / "training-summary.json"
            if report_path.exists():
                report = read(report_path)
                if (
                    digest(root / "checkpoints" / report["checkpoint"])
                    != report["checkpoint_sha256"]
                ):
                    raise ValueError("committed final checkpoint changed")
                reports.append(report)
                continue
            if paused():
                return save_job("paused")
            initial = output / "initial.pt"
            preparation_charge = protocol["conversion_charge_seconds"]
            warm = None
            if arm == "P":
                initial = output / f"placement-r{replicate}.pt"
                if initial.with_suffix(".json").exists():
                    warm = read(initial.with_suffix(".json"))
                    if digest(initial) != warm["checkpoint_sha256"]:
                        raise ValueError("placement checkpoint changed")
                else:
                    phase = "placement_teaching"
                    save_job()
                    torch.set_num_threads(4)
                    before = perf_counter()
                    warm = pretrain(
                        output / "initial.pt",
                        output / "teacher",
                        manifest,
                        initial,
                        epochs=2,
                        shuffle_seed=701 + replicate,
                        config=ImitationConfig(epochs=1, learning_rate=0.0001),
                    )
                    warm["charged_wall_seconds"] = perf_counter() - before
                    _atomic_json(initial.with_suffix(".json"), warm)
                preparation_charge += (
                    manifest["generation_seconds"] + warm["charged_wall_seconds"]
                )
            spec_path = output / f"{arm}-r{replicate}.spec.json"
            spec = CaseSpec(
                arm, replicate, preparation_charge, protocol["fixed_checkpoint_sha256"]
            )
            if spec_path.exists():
                if read(spec_path) != asdict(spec):
                    raise ValueError("case preparation/configuration changed")
            else:
                _atomic_json(spec_path, asdict(spec))
            phase = "training"
            save_job()
            case = HypothesisRun(
                root,
                config,
                spec,
                output / "initial.pt",
                initial_checkpoint=None if (root / "state.json").exists() else initial,
                resume=(root / "state.json").exists(),
            )
            try:
                result = case.train(paused)
                if result["status"] == "paused":
                    return save_job("paused")
                state = read(root / "state.json")
                report = dict(
                    arm=arm,
                    replicate=replicate,
                    directory=root.name,
                    checkpoint=state["checkpoint"],
                    checkpoint_sha256=state["checkpoint_sha256"],
                    update=state["update"],
                    optimizer_steps=state["optimizer_steps"],
                    config=state["config"],
                    ppo_config=state["ppo_config"],
                    observation_schema=state["observation_schema"],
                    reward_scheme=state["reward_scheme"],
                    totals=state["totals"],
                    timings_seconds=state["timings_seconds"],
                    budget=state["hypotheses"],
                    warm_start=warm,
                )
                _atomic_json(report_path, report)
                reports.append(report)
            finally:
                case.close()
            save_job()

        phase = "evaluating"
        (output / "evaluation").mkdir(exist_ok=True)
        evaluation_jobs = []
        for replicate in range(config.replicates):
            evaluation_jobs.append(("original", replicate, output / "initial.pt"))
            for arm in ARMS:
                report = next(
                    r
                    for r in reports
                    if r["arm"] == arm and r["replicate"] == replicate
                )
                evaluation_jobs.append(
                    (
                        arm,
                        replicate,
                        output
                        / report["directory"]
                        / "checkpoints"
                        / report["checkpoint"],
                    )
                )
        for arm, replicate, checkpoint in evaluation_jobs:
            current = dict(arm=arm, replicate=replicate)
            label = f"{arm}-r{replicate}"
            path = output / "evaluation" / f"{label}.json"
            key = digest(checkpoint)
            if path.exists():
                evaluation = read(path)
                if evaluation["checkpoint_sha256"] != key:
                    raise ValueError("evaluation checkpoint identity changed")
                evaluations[label] = evaluation
                continue
            if paused():
                return save_job("paused")
            save_job()
            torch.set_num_threads(1)
            updater = PPOUpdater.load(checkpoint)
            started = perf_counter()
            try:
                evaluation = evaluate_neural(
                    updater.policy(),
                    protocol["test_boards"][replicate],
                    max_actions=config.max_actions,
                    batch_size=8,
                    policy_seed=17,
                    env_factory=HypothesisEnvFactory(knowledge=arm == "K"),
                    progress_path=path.with_suffix(".partial.json"),
                    progress_identity=key,
                    should_pause=paused,
                )
            except EvaluationPaused:
                return save_job("paused")
            evaluation.update(
                checkpoint_sha256=key,
                evaluation_seconds_this_invocation=perf_counter() - started,
            )
            _atomic_json(path, evaluation)
            path.with_suffix(".partial.json").unlink(missing_ok=True)
            evaluations[label] = evaluation
            del updater
            save_job()

        pooled = {
            arm: combine_evaluations(
                [evaluations[f"{arm}-r{r}"] for r in range(config.replicates)]
            )
            for arm in ("original",) + ARMS
        }
        comparisons = {}
        for arm in ("P", "R", "K"):
            greedy = compare_evaluations(
                pooled["B"], pooled[arm], opponent="greedy", confidence=1 - 0.05 / 3
            )
            random = compare_evaluations(pooled["B"], pooled[arm], opponent="random")
            individual = [
                compare_evaluations(
                    evaluations[f"B-r{r}"],
                    evaluations[f"{arm}-r{r}"],
                    opponent="greedy",
                )
                for r in range(config.replicates)
            ]
            positive = sum(c["win_rate_all_change"] > 0 for c in individual)
            comparisons[arm] = dict(
                greedy=greedy,
                random=random,
                per_replicate=individual,
                clear_candidate=bool(
                    greedy["win_rate_all_change"] >= 0.02
                    and positive >= min(2, config.replicates)
                    and greedy["seed_cluster_bootstrap_interval"][0] > 0
                    and random["seed_cluster_bootstrap_95_percent_interval"][0] > -0.05
                ),
            )
        result = dict(
            schema=1,
            status="completed",
            finished_utc=_utc().isoformat(),
            protocol=protocol,
            cases=reports,
            summary={arm: report["summary"] for arm, report in pooled.items()},
            greedy={
                arm: compare_evaluations(report, report, opponent="greedy")["candidate"]
                for arm, report in pooled.items()
            },
            random={
                arm: compare_evaluations(report, report, opponent="random")["candidate"]
                for arm, report in pooled.items()
            },
            comparisons=comparisons,
            versus_original={
                arm: compare_evaluations(
                    pooled["original"], pooled[arm], opponent="greedy"
                )
                for arm in ARMS
            },
            source_unchanged=digest(source) == protocol["source_sha256"],
            session_active_seconds=accrued + perf_counter() - invocation_started,
            note="Final checkpoint is primary. Eligibility is exploratory and conditional on trained models; independent verification is still required before the user-facing completion report. No checkpoint is promoted.",
        )
        if not result["source_unchanged"]:
            raise RuntimeError("original checkpoint changed")
        _atomic_json(output / "comparison.json", result)
        save_job("completed")
        return result
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--training-seconds", type=float)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--test-seeds", type=int)
    parser.add_argument("--teacher-segments", type=int)
    parser.add_argument("--teacher-validation-segments", type=int)
    parser.add_argument("--games-per-update", type=int)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    options = {
        k: getattr(args, k)
        for k in ExperimentConfig.__dataclass_fields__
        if hasattr(args, k) and getattr(args, k) is not None
    }
    if args.resume and options:
        parser.error(
            "resume accepts only stored config, output and optional original path"
        )
    try:
        result = run_experiment(
            args.output,
            args.initial_checkpoint,
            config=None if args.resume else ExperimentConfig(**options),
            resume=args.resume,
        )
    except (ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("cases", "protocol")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Bounded self-play/mixed/imitation comparison from one preserved checkpoint."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from learning.evaluation import _summary, compare_evaluations
from learning.imitation import Demonstrations, ImitationConfig, ImitationTrainer
from learning.ppo import PPOConfig, PPOUpdater
from learning.training import RunConfig, TrainingRun, _atomic_json
from rl.collector import ReferencePolicy, collect_rollouts

MIXED_SCHEDULE = ("greedy", "self", "previous", "greedy", "random")
TEACHER_SEED_BASE = 250_000_000
TEACHER_VALIDATION_BASE = 250_100_000
TRAINING_SEED_BASE = 100_000_000
EVALUATION_OFFSET = 50_000


def generate_teachers(root, *, games=64, max_actions=5000, shard_games=8):
    """Persist bounded, observation-only shards; validation boards stay separate."""
    root = Path(root)
    root.mkdir()
    started = perf_counter()
    shards = []
    for role, count, seed_base in (
        ("training", games, TEACHER_SEED_BASE),
        ("validation", 8, TEACHER_VALIDATION_BASE),
    ):
        for players in (3, 4):
            for start in range(0, count // 2, shard_games):
                seeds = tuple(
                    range(
                        seed_base + (players - 3) * 10000 + start,
                        seed_base
                        + (players - 3) * 10000
                        + min(start + shard_games, count // 2),
                    )
                )
                batch = collect_rollouts(
                    ReferencePolicy("greedy"),
                    seeds,
                    n_players=players,
                    max_actions=max_actions,
                    batch_size=shard_games,
                    policy_seed=17,
                )
                data = Demonstrations.from_rollouts(batch)
                name = f"{role}-{players}p-{start:04d}.npz"
                data.save(root / name)
                shards.append(
                    {
                        "file": name,
                        "role": role,
                        "players": players,
                        "rows": len(data),
                        "sha256": sha256((root / name).read_bytes()).hexdigest(),
                        "episodes": [asdict(episode) for episode in batch.episodes],
                    }
                )
                del data, batch
    result = {
        "schema": 1,
        "shards": shards,
        "generation_seconds": perf_counter() - started,
        "teacher": ReferencePolicy("greedy").tag,
        "max_actions": max_actions,
    }
    _atomic_json(root / "manifest.json", result)
    return result


def _teacher_data(root, shard):
    path = root / shard["file"]
    if sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
        raise ValueError("teacher shard hash mismatch")
    return Demonstrations.load(path)


def _validation_loss(trainer, root, shards, role="validation"):
    total = rows = 0
    for shard in shards:
        if shard["role"] == role:
            data = _teacher_data(root, shard)
            total += trainer.loss(data) * len(data)
            rows += len(data)
    return total / rows


def pretrain(
    source,
    teacher_root,
    manifest,
    destination,
    *,
    epochs=3,
    shuffle_seed=1,
    config=None,
):
    """One shard resident at a time; imitation Adam never enters PPO."""
    loaded = PPOUpdater.load(source)
    trainer = ImitationTrainer(
        loaded.model,
        ImitationConfig(epochs=1) if config is None else config,
        shuffle_seed=shuffle_seed,
    )
    del loaded
    started = perf_counter()
    before = _validation_loss(trainer, teacher_root, manifest["shards"])
    has_main = any(s["role"] == "main_validation" for s in manifest["shards"])
    main_before = (
        _validation_loss(trainer, teacher_root, manifest["shards"], "main_validation")
        if has_main
        else None
    )
    records = []
    for epoch in range(epochs):
        # Alternate player-count order across passes, without changing source labels.
        shards = [s for s in manifest["shards"] if s["role"] == "training"]
        if epoch % 2:
            shards = list(reversed(shards))
        for shard in shards:
            data = _teacher_data(teacher_root, shard)
            result = trainer.fit(data)
            records.append({"epoch": epoch, "file": shard["file"], **result})
            del data
    after = _validation_loss(trainer, teacher_root, manifest["shards"])
    main_after = (
        _validation_loss(trainer, teacher_root, manifest["shards"], "main_validation")
        if has_main
        else None
    )
    PPOUpdater(trainer.model, PPOConfig(gae_lambda=1), shuffle_seed=shuffle_seed).save(
        destination
    )
    result = {
        "schema": 1,
        "epochs": epochs,
        "shuffle_seed": shuffle_seed,
        "validation_loss_before": before,
        "validation_loss_after": after,
        "optimizer_steps": trainer.optimizer_steps,
        "seconds": perf_counter() - started,
        "checkpoint_sha256": sha256(Path(destination).read_bytes()).hexdigest(),
        "shards": records,
        "note": "Actor-only imitation. Critic outputs are not supervised; PPO uses a fresh Adam.",
    }
    if has_main:
        result.update(
            main_validation_loss_before=main_before,
            main_validation_loss_after=main_after,
        )
    _atomic_json(Path(destination).with_suffix(".json"), result)
    return result


def combine_evaluations(reports):
    """Pool disjoint board clusters; preserve actual wins and cutoffs."""
    first = reports[0]
    if any(
        any(r[k] != first[k] for k in ("policy_seed", "max_actions", "deterministic"))
        for r in reports
    ):
        raise ValueError("replicate evaluation protocols differ")
    seeds = tuple(seed for report in reports for seed in report["seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("replicate test boards must be disjoint")
    records = [record for report in reports for record in report["records"]]
    return {
        **{k: first[k] for k in ("policy_seed", "max_actions", "deterministic")},
        "seeds": seeds,
        "records": records,
        "summary": _summary(records),
    }


def compare_training(
    output,
    source,
    *,
    seconds_per_arm=120,
    replicates=3,
    teacher_games=64,
    teacher_epochs=3,
    games_per_update=8,
    max_updates=10000,
    max_actions=5000,
    tune_seeds=4,
    test_seeds=16,
):
    """Equal training-time targets; charge all teacher setup to each imitation arm.

    Final evaluations are outside the training budget; one complete cohort may
    overrun a deadline. Selection uses greedy tuning wins only. Test outcomes do
    not promote a particular checkpoint or change the engine's default policy.
    """
    source = Path(source)
    source_hash = sha256(source.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    loaded = PPOUpdater.load(source)
    hidden_size, model_seed = loaded.model.hidden_size, loaded.model.seed
    del loaded
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": 1,
        "source_sha256": source_hash,
        "seconds_per_arm": seconds_per_arm,
        "replicates": replicates,
        "teacher_games": teacher_games,
        "teacher_epochs": teacher_epochs,
        "games_per_update": games_per_update,
        "max_updates": max_updates,
        "max_actions": max_actions,
        "tune_seeds": tune_seeds,
        "test_seeds": test_seeds,
        "mixed_schedule": MIXED_SCHEDULE,
        "teacher_seed_base": TEACHER_SEED_BASE,
        "teacher_validation_base": TEACHER_VALIDATION_BASE,
        "training_seed_base": TRAINING_SEED_BASE,
        "evaluation_offset": EVALUATION_OFFSET,
        "note": "Exploratory comparison. Board-cluster intervals are conditional on the selected models and board sets; they do not estimate all training-seed uncertainty.",
    }
    _atomic_json(output / "protocol.json", protocol)
    manifest = generate_teachers(
        output / "teacher", games=teacher_games, max_actions=max_actions
    )
    cases = []
    for replicate in range(replicates):
        for arm in ("selfplay", "mixed", "imitation_mixed"):
            started = perf_counter()
            charged_setup = 0.0
            initial = source
            warm = None
            if arm == "imitation_mixed":
                torch.set_num_threads(4)
                initial = output / f"imitation-r{replicate}.pt"
                warm = pretrain(
                    source,
                    output / "teacher",
                    manifest,
                    initial,
                    epochs=teacher_epochs,
                    shuffle_seed=101 + replicate,
                )
                # Charge the entire shared generation cost to every imitation run,
                # rather than granting its reusable dataset free extra compute.
                charged_setup = manifest["generation_seconds"] + (
                    perf_counter() - started
                )
            remaining = seconds_per_arm - charged_setup
            if remaining <= 0:
                raise ValueError(
                    "teacher setup consumed the arm budget; no extra PPO time authorized"
                )
            config = RunConfig(
                budget_seconds=remaining,
                max_updates=max_updates,
                games_per_update=games_per_update,
                active_games=games_per_update,
                max_actions=max_actions,
                evaluation_max_actions=max_actions,
                tune_seeds=tune_seeds,
                test_seeds=test_seeds,
                evaluate_every=5,
                checkpoint_every=5,
                hidden_size=hidden_size,
                model_seed=model_seed,
                shuffle_seed=201 + replicate,
                training_seed_base=TRAINING_SEED_BASE + replicate * 1000000,
                evaluation_seed_offset=EVALUATION_OFFSET + replicate * 128,
                opponent_schedule=("self",) if arm == "selfplay" else MIXED_SCHEDULE,
                tuning_opponent="greedy",
            )
            root = output / f"{arm}-r{replicate}"
            print(
                json.dumps(
                    {
                        "event": "arm_started",
                        "arm": arm,
                        "replicate": replicate,
                        "ppo_budget_seconds": remaining,
                    }
                ),
                flush=True,
            )
            run = TrainingRun(
                root,
                config=config,
                ppo_config=PPOConfig(gae_lambda=1),
                initial_checkpoint=initial,
            )
            try:
                report = run.run()
            finally:
                run.close()
            cases.append(
                {
                    "arm": arm,
                    "replicate": replicate,
                    "directory": root.name,
                    "charged_setup_seconds": charged_setup,
                    "wall_seconds": perf_counter() - started,
                    "warm_start": warm,
                    "result": report,
                }
            )
            _atomic_json(
                output / "progress.json",
                {
                    "completed_cases": len(cases),
                    "planned_cases": replicates * 3,
                    "latest_arm": arm,
                    "replicate": replicate,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "arm_completed",
                        "arm": arm,
                        "replicate": replicate,
                        "selected_update": report["selected_update"],
                        "groups": report["selected_groups"],
                    }
                ),
                flush=True,
            )
    selected = {}
    initial_reports = []
    for arm in ("selfplay", "mixed", "imitation_mixed"):
        reports = []
        for case in cases:
            if case["arm"] == arm:
                root = output / case["directory"] / "evaluation"
                reports.append(json.loads((root / "test-selected.json").read_text()))
                if arm == "selfplay":
                    initial_reports.append(
                        json.loads((root / "test-initial.json").read_text())
                    )
        selected[arm] = combine_evaluations(reports)
    baseline = combine_evaluations(initial_reports)
    result = {
        "schema": 1,
        "status": "completed",
        "protocol": protocol,
        "cases": cases,
        "baseline": baseline["summary"],
        "baseline_greedy": compare_evaluations(baseline, baseline, opponent="greedy")[
            "baseline"
        ],
        "versus_initial": {
            arm: compare_evaluations(baseline, report, opponent="greedy")
            for arm, report in selected.items()
        },
        "versus_selfplay": {
            arm: compare_evaluations(
                selected["selfplay"], selected[arm], opponent="greedy"
            )
            for arm in ("mixed", "imitation_mixed")
        },
        "all_opponents": {arm: report["summary"] for arm, report in selected.items()},
        "source_unchanged": sha256(source.read_bytes()).hexdigest() == source_hash,
        "note": protocol["note"],
    }
    if not result["source_unchanged"]:
        raise RuntimeError("source checkpoint changed during comparison")
    _atomic_json(output / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--seconds-per-arm", type=float, default=120)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--teacher-games", type=int, default=64)
    parser.add_argument("--teacher-epochs", type=int, default=3)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--tune-seeds", type=int, default=4)
    parser.add_argument("--test-seeds", type=int, default=16)
    args = parser.parse_args()
    if (
        not 1 <= args.replicates <= 5
        or not 2 <= args.teacher_games <= 1024
        or args.teacher_games % 2
        or not 1 <= args.teacher_epochs <= 20
    ):
        parser.error(
            "use 1-5 replicates, an even 2-1024 teacher games and 1-20 teacher epochs"
        )
    # Exercise the same budget/cap/seed bounds as normal training before writing.
    try:
        RunConfig(
            budget_seconds=args.seconds_per_arm,
            games_per_update=args.games,
            active_games=args.games,
            max_updates=args.max_updates,
            max_actions=args.max_actions,
            evaluation_max_actions=args.max_actions,
            tune_seeds=args.tune_seeds,
            test_seeds=args.test_seeds,
            training_seed_base=TRAINING_SEED_BASE,
            opponent_schedule=MIXED_SCHEDULE,
        )
        result = compare_training(
            args.output,
            args.initial_checkpoint,
            seconds_per_arm=args.seconds_per_arm,
            replicates=args.replicates,
            teacher_games=args.teacher_games,
            teacher_epochs=args.teacher_epochs,
            games_per_update=args.games,
            max_updates=args.max_updates,
            max_actions=args.max_actions,
            tune_seeds=args.tune_seeds,
            test_seeds=args.test_seeds,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()

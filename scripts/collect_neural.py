"""Run an untrained policy/critic and prepare targets; never update weights."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from learning.model import ActorCritic, MODEL_SCHEMA_VERSION
from learning.policy import NeuralPolicy
from rl.collector import collect_rollouts, save_rollouts
from rl.returns import TARGET_SCHEMA_VERSION, compute_targets
from scripts.collect_rollouts import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=(3, 4), default=4)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--discount", choices=("decision", "engine"), default="decision"
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.games,
            args.hidden_size,
            args.threads,
            args.batch_size,
            args.max_actions,
        )
        < 1
    ):
        parser.error("all integer bounds must be positive")
    if not (0 <= args.gamma <= 1 and 0 <= args.gae_lambda <= 1):
        parser.error("gamma and gae-lambda must be finite between zero and one")
    if any(
        path is not None and path.suffix != ".npz"
        for path in (args.output, args.targets)
    ):
        parser.error("output/targets must end in .npz")
    torch.set_num_threads(args.threads)
    model = ActorCritic(args.hidden_size, seed=args.model_seed)
    policy = NeuralPolicy(model)
    start = perf_counter()
    batch = collect_rollouts(
        policy,
        range(args.seed, args.seed + args.games),
        n_players=args.players,
        policy_seed=args.policy_seed,
        max_actions=args.max_actions,
        batch_size=args.batch_size,
        validate=args.validate,
    )
    report = summarize(batch, perf_counter() - start)
    start = perf_counter()
    bootstrap = policy.bootstrap_values(batch)
    target = compute_targets(
        batch,
        bootstrap_values=bootstrap,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        discount=args.discount,
    )
    report.update(
        model_schema=MODEL_SCHEMA_VERSION,
        target_schema=TARGET_SCHEMA_VERSION,
        model_seed=args.model_seed,
        hidden_size=args.hidden_size,
        torch=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()),
        trained=False,
        optimizer_updates=0,
        threads=args.threads,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        discount=args.discount,
        target_seconds=perf_counter() - start,
        mean_value=float(batch.values.mean()),
        mean_advantage=float(target.advantages.mean()),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_rollouts(batch, args.output)
    if args.targets:
        args.targets.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            key: report[key]
            for key in (
                "model_schema",
                "target_schema",
                "model_seed",
                "gamma",
                "gae_lambda",
                "discount",
                "fingerprint",
            )
        }
        np.savez(
            args.targets,
            advantages=target.advantages,
            value_targets=target.value_targets,
            bootstrap_values=bootstrap,
            metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

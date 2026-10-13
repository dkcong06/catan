"""Collect complete reference-policy episodes without training a model."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from catan.state import Phase
from rl.actions import ACTION_SCHEMA_VERSION
from rl.collector import (
    COLLECTOR_SCHEMA_VERSION,
    ReferencePolicy,
    collect_rollouts,
    save_rollouts,
)
from rl.encoding import SCHEMA_VERSION
from rl.env import ENV_SCHEMA_VERSION, REWARD_SCHEME


def summarize(batch, elapsed):
    return {
        "collector_schema": COLLECTOR_SCHEMA_VERSION,
        "environment_schema": ENV_SCHEMA_VERSION,
        "observation_schema": SCHEMA_VERSION,
        "action_schema": ACTION_SCHEMA_VERSION,
        "reward_scheme": REWARD_SCHEME,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "policy_tag": batch.policy_tag,
        "episodes": [asdict(episode) for episode in batch.episodes],
        "games": len(batch.episodes),
        "wins": sum(episode.terminated for episode in batch.episodes),
        "cutoffs": sum(episode.truncated for episode in batch.episodes),
        "actions": len(batch),
        "discards": len(batch.discards.rows),
        "phase_counts": {
            phase.name: int(np.count_nonzero(batch.phases == phase)) for phase in Phase
        },
        "payload_bytes": batch.payload_bytes,
        "elapsed_seconds": elapsed,
        "actions_per_second": len(batch) / elapsed,
        "fingerprint": sha256(
            "".join(episode.fingerprint for episode in batch.episodes).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies",
        default="random",
        help="one name or one comma-separated name per seat",
    )
    parser.add_argument("--players", type=int, choices=(3, 4), default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, help="optional .npz training records")
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="compress the optional archive, outside timed collection",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.games, args.batch_size, args.max_actions) < 1:
        parser.error("games, batch-size and max-actions must be positive")
    try:
        policy = ReferencePolicy(tuple(args.policies.split(",")))
        if len(policy.names) not in (1, args.players):
            raise ValueError("one policy or exactly one policy per seat is required")
        if args.output is not None and args.output.suffix != ".npz":
            raise ValueError("output must end in .npz")
    except ValueError as error:
        parser.error(str(error))
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
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_rollouts(batch, args.output, compressed=args.compressed)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

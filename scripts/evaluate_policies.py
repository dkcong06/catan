"""Evaluate versioned reference policies across seeded board/seat rotations."""

import argparse
import json
from pathlib import Path

from rl.evaluate import evaluate
from rl.policies import POLICIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=POLICIES, default="greedy")
    parser.add_argument("--opponent", choices=POLICIES, default="random")
    parser.add_argument("--players", type=int, choices=(3, 4), default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="distinct boards, each rotated through every seat",
    )
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.seeds, args.max_actions) < 1:
        parser.error("seeds and max-actions must be positive")
    report = evaluate(
        candidate=args.candidate,
        opponent=args.opponent,
        n_players=args.players,
        seeds=range(args.seed, args.seed + args.seeds),
        policy_seed=args.policy_seed,
        max_actions=args.max_actions,
        validate=args.validate,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "games"}, indent=2
        )
    )


if __name__ == "__main__":
    main()

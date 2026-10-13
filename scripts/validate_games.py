"""Validate bounded seeded episodes, report caps, and reproduce rule failures."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from catan.state import Phase
from catan.validation import GameValidationError, validate_game


def _run_case(request):
    seed, players, cap, inplace = request
    try:
        return validate_game(seed, players, max_actions=cap, inplace=inplace)
    except GameValidationError as error:
        return error.context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10_000, help="total episodes")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--players", choices=("3", "4", "both"), default="both")
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.games, args.max_actions, args.workers) < 1:
        parser.error("games, max-actions, and workers must be positive")
    requests = (
        (
            args.seed + index,
            3 + index % 2 if args.players == "both" else int(args.players),
            args.max_actions,
            not args.functional,
        )
        for index in range(args.games)
    )
    report = {
        "schema_version": 1,
        "games_requested": args.games,
        "games_checked": 0,
        "first_seed": args.seed,
        "players": args.players,
        "max_actions": args.max_actions,
        "workers": args.workers,
        "functional": args.functional,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "completed": 0,
        "capped": 0,
        "failures": [],
        "actions": 0,
        "invariant_checks": 0,
        "per_players": {},
        "phase_counts": {phase.name: 0 for phase in Phase},
        "capped_games": [],
    }
    digest = sha256()
    pool = None
    start = perf_counter()
    if args.workers == 1:
        results = map(_run_case, requests)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        # Python 3.14 bounded submission: each worker exclusively owns its states.
        results = pool.map(
            _run_case, requests, chunksize=4, buffersize=2 * args.workers
        )
    try:
        for result in results:
            report["games_checked"] += 1
            if isinstance(result, dict):
                report["failures"].append(result)
                print(f"FAIL: {result}", flush=True)
                break
            digest.update(repr(result).encode())
            stats = report["per_players"].setdefault(
                str(result.players),
                {
                    "games": 0,
                    "completed": 0,
                    "capped": 0,
                    "wins": [0] * result.players,
                },
            )
            stats["games"] += 1
            report["actions"] += result.actions
            report["invariant_checks"] += result.actions + 1
            for phase, count in zip(Phase, result.phase_counts):
                report["phase_counts"][phase.name] += count
            if result.capped:
                report["capped"] += 1
                stats["capped"] += 1
                report["capped_games"].append(asdict(result))
            else:
                report["completed"] += 1
                stats["completed"] += 1
                stats["wins"][result.winner] += 1
            if report["games_checked"] % 100 == 0:
                print(
                    f"{report['games_checked']}/{args.games}: {report['completed']} wins, "
                    f"{report['capped']} caps, {report['actions']} checked actions",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    report["fingerprint"] = digest.hexdigest()
    report["elapsed_seconds"] = perf_counter() - start
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Checked {report['games_checked']} episodes / {report['actions']} actions: "
        f"{report['completed']} completed, {report['capped']} capped, "
        f"{len(report['failures'])} failures in {report['elapsed_seconds']:.2f}s",
        flush=True,
    )
    print(
        "Caps are truncations; no winners are invented. Replay with --games 1 --seed N --players P."
    )
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

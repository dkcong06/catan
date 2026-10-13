"""Measure dataset collection versus the same games without training records.

Warm alternating runs include setup, observations, masks, reference policies,
engine steps and final state/RNG fingerprints. No inference/training/debug checks
or archive compression are timed. Synchronous width does not parallelize Python.
"""

import argparse
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np

from rl.collector import ReferencePolicy, collect_rollouts
from rl.evaluate import play_match
from rl.policies import POLICIES


def episodes(collect, games, cap, width):
    results = []
    actions = payload = discards = 0
    for players in (3, 4):
        names = ("greedy",) + ("random",) * (players - 1)
        if collect:
            batch = collect_rollouts(
                ReferencePolicy(names),
                range(games),
                n_players=players,
                max_actions=cap,
                batch_size=width,
            )
            matches = batch.episodes
            payload += batch.payload_bytes
            discards += len(batch.discards.rows)
        else:
            matches = tuple(
                play_match([POLICIES[name] for name in names], seed, max_actions=cap)
                for seed in range(games)
            )
        for match in matches:
            actions += match.actions
            results.append(
                (
                    players,
                    match.seed,
                    match.actions,
                    match.winner,
                    match.terminated,
                    match.truncated,
                    match.fingerprint,
                )
            )
    return actions, results, payload, discards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8, help="games per player count")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.games, args.repeats, args.batch_size, args.max_actions) < 1:
        parser.error("all bounds must be positive")
    modes = {"no_training_records": False, "collector": True}
    samples = {name: [] for name in modes}
    expected = None
    final = None
    for mode in modes.values():
        result = episodes(mode, args.games, args.max_actions, args.batch_size)
        if expected is None:
            expected = result[:2]
        assert result[:2] == expected
    for repeat in range(args.repeats):
        names = list(modes)
        if repeat % 2:
            names.reverse()
        for name in names:
            start = perf_counter()
            result = episodes(
                modes[name], args.games, args.max_actions, args.batch_size
            )
            samples[name].append(perf_counter() - start)
            assert result[:2] == expected
            if modes[name]:
                final = result
    report = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "games": 2 * args.games,
        "batch_size": args.batch_size,
        "max_actions": args.max_actions,
        "repeats": args.repeats,
        "actions": expected[0],
        "samples_seconds": samples,
        "median_seconds": {name: median(times) for name, times in samples.items()},
        "actions_per_second": {
            name: expected[0] / median(times) for name, times in samples.items()
        },
        "payload_bytes": final[2],
        "discards": final[3],
        "final_results": expected[1],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

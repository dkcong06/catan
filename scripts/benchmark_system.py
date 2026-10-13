"""Paired hot-path and complete-game timings; never a training benchmark."""

import argparse
import json
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter

import numpy as np

from catan.engine import apply_action, apply_action_inplace, is_terminal
from catan.players.random_agent import choose_action
from catan.state import new_game
from catan.validation import _fingerprint
from rl.encoding import FlatEncoder


def _episodes(encoded, functional, cap):
    actions = 0
    summaries = []
    step = apply_action if functional else apply_action_inplace
    for players in (3, 4):
        for seed in range(3):
            state = new_game(seed, players)
            policy = Random(seed)
            encoder = FlatEncoder(state.board) if encoded else None
            for _ in range(cap):
                if encoder is not None:
                    encoder.encode(state)
                state = step(state, choose_action(state, policy))
                actions += 1
                if is_terminal(state):
                    break
            summaries.append((is_terminal(state), _fingerprint(state, policy)))
    return actions, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.repeats, args.max_actions) < 1:
        parser.error("repeats and max-actions must be positive")
    modes = {
        "owned_game": (False, False),
        "copied_game": (False, True),
        "owned_game_with_observations": (True, False),
    }
    samples = {name: [] for name in modes}
    expected = None
    for encoded, functional in modes.values():
        _episodes(encoded, functional, 40)
    for batch in range(args.repeats):
        names = tuple(modes)
        if batch % 2:
            names = names[::-1]
        for name in names:
            start = perf_counter()
            result = _episodes(*modes[name], args.max_actions)
            samples[name].append(perf_counter() - start)
            if expected is None:
                expected = result
            else:
                assert result == expected
    report = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "cases": "seeds 0-2 for both 3/4 seats",
        "max_actions": args.max_actions,
        "repeats": args.repeats,
        "actions": expected[0],
        "wins": sum(win for win, _ in expected[1]),
        "caps": sum(not win for win, _ in expected[1]),
        "samples_seconds": samples,
        "median_seconds": {name: median(v) for name, v in samples.items()},
        "actions_per_second": {
            name: expected[0] / median(v) for name, v in samples.items()
        },
        "final_fingerprints": [fp for _, fp in expected[1]],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        "Includes reset/legal queries/policy and optional encoding; excludes validation/inference/training/rendering."
    )


if __name__ == "__main__":
    main()

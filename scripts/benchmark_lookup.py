"""Measure direct masks against the legal-object oracle and scalar/batch encoding."""

import json
from pathlib import Path
from random import Random
from time import perf_counter
from statistics import median
import numpy as np
from catan.state import new_game, Phase
from catan.engine import apply_action_inplace, is_terminal, legal_actions
from catan.players.random_agent import choose_action
from rl import encoding as enc

from rl.actions import ACTION_DIM, action_index, action_mask


def reference_mask(state):
    mask = np.zeros(ACTION_DIM, np.bool_)
    for action in legal_actions(state):
        mask[action_index(state, action)] = True
    return mask


def batch(cases):
    return enc.encode_batch([s for s, _ in cases], [ctx for _, ctx in cases])


def direct_mask(state):
    return action_mask(state)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Paired masks/batch encoding; excludes gameplay and learning."
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    cases = []
    for seed in range(8):
        s = new_game(seed, 3 + seed % 2)
        ctx = enc.FlatEncoder(s.board)
        policy = Random(seed)
        for step in range(300):
            if step % 17 == 0:
                cases.append((s.copy(), ctx))
            if is_terminal(s):
                break
            apply_action_inplace(s, choose_action(s, policy))
    for state, ctx in cases:
        if state.phase != Phase.DISCARD:
            np.testing.assert_array_equal(reference_mask(state), direct_mask(state))
    for size in (1, 8, 32, 128):
        pool = cases[:size]
        np.testing.assert_array_equal(
            batch(pool), np.stack([ctx.encode(s) for s, ctx in pool])
        )
    samples = {}
    operations = {
        "reference_mask": lambda pool: [
            reference_mask(s) for s, _ in pool if s.phase != Phase.DISCARD
        ],
        "direct_mask": lambda pool: [
            direct_mask(s) for s, _ in pool if s.phase != Phase.DISCARD
        ],
    }
    results = {
        "states": len(cases),
        "numpy": np.__version__,
        "timings_us_per_state": {},
    }
    for size in (1, 8, 32, 128):
        pool = cases[:size]
        ops = {
            "scalar_encode": lambda: np.stack([ctx.encode(s) for s, ctx in pool]),
            "batch_encode": lambda: batch(pool),
        }
        if size == 128:
            ops.update(
                {name: (lambda fn=fn: fn(pool)) for name, fn in operations.items()}
            )
        samples = {name: [] for name in ops}
        iterations = max(100, 4000 // size)
        for fn in ops.values():
            fn()
        for repeat in range(5):
            names = list(ops)
            if repeat % 2:
                names.reverse()
            for name in names:
                start = perf_counter()
                for _ in range(iterations):
                    ops[name]()
                denominator = (
                    size
                    if "mask" not in name
                    else sum(s.phase != Phase.DISCARD for s, _ in pool)
                )
                samples[name].append(
                    (perf_counter() - start) / iterations / denominator * 1e6
                )
        results["timings_us_per_state"][size] = {
            name: median(times) for name, times in samples.items()
        }
    print(json.dumps(results, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

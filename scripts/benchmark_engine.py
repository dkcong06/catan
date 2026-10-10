"""Compare functional and in-place seven/discard transitions, not RL throughput."""

import argparse
import platform
from statistics import median
from time import perf_counter

import numpy as np

from catan.actions import DiscardAction
from catan.engine import (
    apply_action,
    apply_action_inplace,
    resolve_seven,
    resolve_seven_inplace,
)
from catan.state import Phase, new_game


def main() -> None:
    """Use identical prebuilt fixtures and alternate warmed measurement order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--players", type=int, choices=(3, 4), default=4)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")

    roll = new_game(42, args.players)
    roll.hands[:] = (
        (4, 3, 2, 1, 0),
        (1, 2, 0, 3, 1),
        (0, 1, 3, 0, 5),
        (2, 2, 2, 2, 0),
    )[: args.players]
    roll.bank -= roll.hands.sum(axis=0)
    roll.setup_index = len(roll.setup_order)
    roll.phase = Phase.ROLL
    roll.rng.gauss(0, 1)
    roll.validate()
    pending = resolve_seven(roll)
    actions = (
        DiscardAction(0, (4, 0, 0, 1, 0)),
        DiscardAction(2, (0, 0, 0, 0, 4)),
        DiscardAction(3, (0, 0, 2, 2, 0)),
    )[: args.players - 1]

    def functional_round(state):
        state = resolve_seven(state)
        for action in actions:
            state = apply_action(state, action)
        return state

    def inplace_round(state):
        resolve_seven_inplace(state)
        for action in actions:
            apply_action_inplace(state, action)
        return state

    operations = {
        "copy, arrays and full RNG": (roll, lambda s: s.copy()),
        "seven, functional": (roll, resolve_seven),
        "seven, in-place": (roll, resolve_seven_inplace),
        "discard, functional": (pending, lambda s: apply_action(s, actions[0])),
        "discard, in-place": (pending, lambda s: apply_action_inplace(s, actions[0])),
        "round, functional": (roll, functional_round),
        "round, in-place": (roll, inplace_round),
    }
    expected = {}
    for name, (fixture, operation) in operations.items():
        expected[name] = operation(fixture.copy())
        expected[name].validate()
        for _ in range(100):
            operation(fixture.copy())
    assert expected["seven, functional"] == expected["seven, in-place"]
    assert expected["discard, functional"] == expected["discard, in-place"]
    assert expected["round, functional"] == expected["round, in-place"]
    assert expected["round, in-place"].phase == Phase.ROBBER_MOVE

    timings = {name: [] for name in operations}
    for repeat in range(args.repeats):
        names = tuple(operations)
        if repeat % 2:
            names = tuple(reversed(names))
        for name in names:
            fixture, operation = operations[name]
            # Both paths visit distinct, identical states. In-place calls cannot
            # repeatedly consume one already-mutated fixture. Pool allocation,
            # copying/RNG initialization, validation, and cleanup are not timed.
            pool = [fixture.copy() for _ in range(args.iterations)]
            start = perf_counter()
            for state in pool:
                result = operation(state)
            elapsed = perf_counter() - start
            timings[name].append(elapsed / args.iterations)
            assert result == expected[name]
            result.validate()
            del pool, state, result

    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, {args.players} players"
    )
    print(f"{args.repeats} alternating warmed batches x {args.iterations:,} operations")
    for name, samples in timings.items():
        print(f"{name}: {median(samples) * 1e6:.2f} us/operation")
    for transition in ("seven", "discard", "round"):
        functional = median(timings[f"{transition}, functional"])
        inplace = median(timings[f"{transition}, in-place"])
        print(
            f"{transition}: functional/in-place time ratio {functional / inplace:.2f}x"
        )
    print(f"Round = seven + {len(actions)} preselected manual discard actions.")
    print(
        "Excludes fixture pools/resets, validation, board setup, rendering, and policy inference."
    )
    print(
        "Pool memory scales with iterations; these are fixed-fixture microbenchmarks, not game/RL throughput."
    )


if __name__ == "__main__":
    main()

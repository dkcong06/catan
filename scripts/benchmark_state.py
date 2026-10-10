"""Measure state initialization and independent copies without gameplay or ML."""

import argparse
import platform
from statistics import median
from time import perf_counter

import numpy as np

from catan.state import GameState, new_game


def main() -> None:
    """Report medians of alternating batches; no machine-specific pass thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--players", type=int, choices=(3, 4), default=4)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    state = new_game(42, args.players)
    operations = {
        "initialize, supplied board/RNG": lambda: GameState(
            state.board, args.players, state.rng
        ),
        "new_game, seeded board/RNG": lambda: new_game(42, args.players),
        "copy, arrays and independent RNG": state.copy,
    }
    timings: dict[str, list[float]] = {name: [] for name in operations}
    for operation in operations.values():
        for _ in range(100):
            operation()
    for repeat in range(args.repeats):
        names = tuple(operations)
        if repeat % 2:
            names = tuple(reversed(names))
        for name in names:
            start = perf_counter()
            for _ in range(args.iterations):
                operations[name]()
            timings[name].append((perf_counter() - start) / args.iterations)
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, {args.players} players"
    )
    print(
        f"{args.repeats} batches x {args.iterations:,} operations; medians after warmup"
    )
    for name, samples in timings.items():
        print(f"{name}: {median(samples) * 1e6:.2f} us/operation")
    print(f"Mutable array payload: {state.array_bytes} bytes")
    print(
        "Excludes Python/ndarray headers, RNG memory, shared board, imports, and gameplay."
    )


if __name__ == "__main__":
    main()

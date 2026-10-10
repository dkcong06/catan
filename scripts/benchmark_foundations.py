"""Measure Stage 1-3 hot paths; does not estimate gameplay or learning throughput."""

import argparse
import platform
from statistics import median
from time import perf_counter

import numpy as np

from catan.board import _red_tile_sets_by_desert
from catan.state import new_game
from scripts.board_viewer import board_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    state = new_game(42)
    branch = state.copy()
    operations = {
        "red-set cache construction": (
            _red_tile_sets_by_desert,
            max(1, args.iterations // 20),
        ),
        "equal four-player states, including RNG": (
            lambda: state == branch,
            args.iterations,
        ),
        "port access, empty board": (lambda: state.port_access(0), args.iterations),
        "preview JSON, seed 42": (lambda: board_payload(42), args.iterations),
    }
    measurements = {name: [] for name in operations}
    for operation, _ in operations.values():
        for _ in range(30):
            operation()
    for batch in range(args.repeats):
        names = tuple(operations)
        if batch % 2:
            names = tuple(reversed(names))
        for name in names:
            operation, iterations = operations[name]
            start = perf_counter()
            for _ in range(iterations):
                operation()
            measurements[name].append((perf_counter() - start) / iterations)
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, {platform.system()}"
    )
    print(f"{args.repeats} alternating batches after warmup")
    for name, samples in measurements.items():
        print(f"{name}: {median(samples) * 1e6:.2f} us/operation")
    print(f"Seed 42 preview: {len(board_payload(42)):,} bytes")
    print(
        "Cache construction occurs once at board import; previews include new board/RNG."
    )
    print(
        "Equality includes all fields/RNG. Excludes HTTP transport, canvas, imports, gameplay."
    )


if __name__ == "__main__":
    main()

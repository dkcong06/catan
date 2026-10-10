"""Compare direct legal-board sampling with full-token rejection sampling."""

import argparse
from random import Random
from statistics import median
from time import perf_counter

from catan.board import (
    GEOMETRY,
    NUMBER_BAG,
    PORT_BAG,
    RESOURCE_BAG,
    Board,
    Resource,
    make_board,
)


def rejection_board(rng: Random) -> Board:
    """Reference sampler: keep shuffling all tokens until the red rule holds."""
    resources = list(RESOURCE_BAG)
    rng.shuffle(resources)
    desert = resources.index(Resource.DESERT)
    tokens = list(NUMBER_BAG)
    while True:
        rng.shuffle(tokens)
        iterator = iter(tokens)
        numbers = tuple(0 if tile == desert else next(iterator) for tile in range(19))
        if all(
            numbers[neighbor] not in (6, 8)
            for tile, number in enumerate(numbers)
            if number in (6, 8)
            for neighbor in GEOMETRY.tile_neighbors[tile]
        ):
            break
    ports = list(PORT_BAG)
    rng.shuffle(ports)
    return Board(tuple(resources), numbers, desert, tuple(ports))


def main() -> None:
    """Measure paired batches on one CPU; report medians without timing assertions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.boards < 1 or args.repeats < 1:
        parser.error("boards and repeats must be positive")
    samplers = {"direct": make_board, "rejection": rejection_board}
    timings: dict[str, list[float]] = {name: [] for name in samplers}
    for sampler in samplers.values():
        rng = Random(0)
        for _ in range(100):
            sampler(rng)
    for repeat in range(args.repeats):
        names = tuple(samplers) if repeat % 2 == 0 else tuple(reversed(samplers))
        for name in names:
            rng = Random(repeat)
            start = perf_counter()
            for _ in range(args.boards):
                samplers[name](rng)
            timings[name].append((perf_counter() - start) / args.boards)
    direct, rejection = (median(timings[name]) for name in samplers)
    print(f"Paired {args.repeats} batches x {args.boards:,} boards; shared seeded RNG")
    print(f"Direct sampler: {direct * 1e6:.2f} us/board ({1 / direct:,.0f} boards/s)")
    print(f"Rejection sampler: {rejection * 1e6:.2f} us/board")
    print(f"Direct/rejection speedup: {rejection / direct:.2f}x")
    print(
        "Includes resources, legal tokens, and ports; excludes imports, state, and gameplay."
    )


if __name__ == "__main__":
    main()

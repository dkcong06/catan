"""Stress seeded board setup and state branches independently of future game rules."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import fields
from random import Random, getstate
from time import perf_counter

import numpy as np

from catan.board import (
    GEOMETRY,
    NUMBER_BAG,
    PORT_BAG,
    RESOURCE_BAG,
    Resource,
    make_board,
)
from catan.state import new_game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=100_000)
    parser.add_argument("--states", type=int, default=5_000)
    args = parser.parse_args()
    if args.boards < 1 or args.states < 1:
        parser.error("boards and states must be positive")
    original_global_rng = getstate()
    resources, numbers, ports = map(Counter, (RESOURCE_BAG, NUMBER_BAG, PORT_BAG))
    rng = Random(2026)
    deserts = set()
    start = perf_counter()
    for _ in range(args.boards):
        board = make_board(rng)
        assert board.geometry is GEOMETRY
        assert Counter(board.resources) == resources
        assert Counter(token for token in board.numbers if token) == numbers
        assert Counter(board.ports) == ports
        assert board.resources[board.robber_tile] == Resource.DESERT
        assert tuple(t for t, token in enumerate(board.numbers) if token == 0) == (
            board.robber_tile,
        )
        for tile, token in enumerate(board.numbers):
            if token in (6, 8):
                assert all(
                    board.numbers[n] not in (6, 8)
                    for n in GEOMETRY.tile_neighbors[tile]
                )
        deserts.add(board.robber_tile)
    print(
        f"{args.boards:,} boards passed in {perf_counter() - start:.2f}s; {len(deserts)} desert positions",
        flush=True,
    )
    start = perf_counter()
    for seed in range(args.states):
        state = new_game(seed, 3 + seed % 2)
        state.rng.gauss(0, 1)
        state.validate()
        saved_rng = state.rng.getstate()
        first, second = state.copy(), deepcopy(state)
        first.validate()
        second.validate()
        assert first == second == state
        assert first.board is second.board is state.board
        for descriptor in fields(state):
            original = getattr(state, descriptor.name)
            if isinstance(original, np.ndarray):
                assert not np.shares_memory(original, getattr(first, descriptor.name))
                assert not np.shares_memory(original, getattr(second, descriptor.name))
        for _ in range(5):
            assert first.rng.randint(1, 6) == second.rng.randint(1, 6)
            assert first.rng.gauss(0, 1) == second.rng.gauss(0, 1)
        assert first == second
        assert state.rng.getstate() == saved_rng
        for seat in range(state.n_players):
            assert state.port_access(seat) == ()
    assert getstate() == original_global_rng
    print(
        f"{args.states:,} three/four-player states and two branches each passed in {perf_counter() - start:.2f}s"
    )
    print(
        "No shared mutable arrays or global RNG changes; no gameplay transitions are exercised."
    )


if __name__ == "__main__":
    main()

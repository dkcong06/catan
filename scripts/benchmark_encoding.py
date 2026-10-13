"""Measure flat encoding only, using a bounded pool of reached game states."""

import argparse
import json
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter

import numpy as np

from catan.engine import apply_action_inplace, is_terminal
from catan.players.random_agent import choose_action
from catan.state import new_game
from rl.encoding import OBS_DIM, SCHEMA_VERSION, FlatEncoder, encode_flat


def reached_states():
    """At most 64 snapshots per seat count; include every encountered phase."""
    pool = []
    for players in (3, 4):
        state = new_game(0, players)
        seen = set()
        snapshots = []
        policy = Random(0)
        for step in range(10000):
            if len(snapshots) < 63 and (step % 31 == 0 or state.phase not in seen):
                snapshots.append(state.copy())
            seen.add(state.phase)
            if is_terminal(state):
                snapshots.append(state.copy())
                break
            apply_action_inplace(state, choose_action(state, policy))
        else:
            raise RuntimeError("benchmark seed 0 did not reach a win")
        pool.extend(snapshots)
    return tuple(pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.repeats) < 1:
        parser.error("iterations and repeats must be positive")
    states = reached_states()
    boards = {id(state.board): state.board for state in states}
    encoders = {key: FlatEncoder(board) for key, board in boards.items()}
    cases = tuple((state, encoders[id(state.board)]) for state in states)
    out = np.empty(OBS_DIM, dtype=np.float32)
    for state, encoder in cases:
        state.validate()
        expected = encode_flat(state)
        np.testing.assert_array_equal(encoder.encode(state), expected)
        np.testing.assert_array_equal(encoder.encode(state, out=out), expected)
    operations = {
        "prepare_board": lambda state, encoder: FlatEncoder(state.board),
        "one_off_allocating": lambda state, encoder: encode_flat(state),
        "prepared_allocating": lambda state, encoder: encoder.encode(state),
        "prepared_reusing": lambda state, encoder: encoder.encode(state, out=out),
    }
    samples = {name: [] for name in operations}
    for operation in operations.values():
        for state, encoder in cases:
            operation(state, encoder)
    for batch in range(args.repeats):
        names = tuple(operations)
        if batch % 2:
            names = tuple(reversed(names))
        for name in names:
            operation = operations[name]
            start = perf_counter()
            for i in range(args.iterations):
                operation(*cases[i % len(cases)])
            samples[name].append((perf_counter() - start) / args.iterations)
    report = {
        "schema_version": SCHEMA_VERSION,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "states": len(states),
        "seat_counts": sorted({state.n_players for state in states}),
        "phases": sorted({state.phase.name for state in states}),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "observation_floats": OBS_DIM,
        "observation_bytes": out.nbytes,
        "static_bytes_per_board": next(iter(encoders.values()))._base.nbytes,
        "median_us": {name: median(times) * 1e6 for name, times in samples.items()},
        "samples_seconds_per_call": samples,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        "Warm alternating batches; excludes pool creation, inference, training and rendering."
    )
    print("Reused outputs must be consumed or copied before the next encoding call.")


if __name__ == "__main__":
    main()

"""Measure complete headless baseline games; report capped runs separately."""

import argparse
import platform
from random import Random
from statistics import median
from time import perf_counter

import numpy as np

from catan.engine import apply_action, apply_action_inplace, is_terminal, winner
from catan.players.random_agent import choose_action
from catan.state import new_game


def _batch(seed, games, players, max_actions, step):
    steps = 0
    states = []
    for game_seed in range(seed, seed + games):
        state = new_game(game_seed, players)
        policy = Random(game_seed)
        for _ in range(max_actions):
            state = step(state, choose_action(state, policy))
            steps += 1
            if is_terminal(state):
                break
        states.append(state)
    return steps, states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--players", type=int, choices=(3, 4), default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=10000)
    args = parser.parse_args()
    if min(args.games, args.repeats, args.max_actions) < 1:
        parser.error("games, repeats, and max-actions must be positive")
    modes = {"functional": apply_action, "in-place": apply_action_inplace}
    timings = {name: [] for name in modes}
    expected = None
    for step in modes.values():
        _batch(args.seed, 1, args.players, args.max_actions, step)
    for repeat in range(args.repeats):
        names = tuple(modes)
        if repeat % 2:
            names = tuple(reversed(names))
        for name in names:
            start = perf_counter()
            steps, states = _batch(
                args.seed, args.games, args.players, args.max_actions, modes[name]
            )
            elapsed = perf_counter() - start
            timings[name].append(elapsed)
            if expected is None:
                expected = steps, states
            else:
                assert steps == expected[0] and states == expected[1]
            for state in states:
                state.validate()
    completed = sum(is_terminal(state) for state in expected[1])
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, {args.players} players"
    )
    print(
        f"{args.repeats} warmed alternating batches x {args.games} games; {expected[0]} actions/batch"
    )
    print(
        f"Completed: {completed}; capped: {args.games - completed}; cap: {args.max_actions} actions/game"
    )
    print(f"Winners (None means capped): {[winner(state) for state in expected[1]]}")
    for name, samples in timings.items():
        elapsed = median(samples)
        print(f"{name}: {elapsed:.4f} s/batch; {expected[0] / elapsed:.0f} actions/s")
    print(
        f"Functional/in-place time ratio: {median(timings['functional']) / median(timings['in-place']):.2f}x"
    )
    print(
        "Includes reset, legal-choice generation and random-policy selection; excludes rendering, per-step validation and neural inference/training."
    )
    print(
        "Capped games are truncations, not wins. Results are baseline gameplay throughput, not RL throughput."
    )


if __name__ == "__main__":
    main()

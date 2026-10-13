"""Compare the environment with an equivalent direct-engine agent loop.

Both include reset, prepared observation/mask queries, separate per-seat policy
RNGs and final full-state/RNG fingerprints. No validation, inference or training.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter

import numpy as np

from catan.actions import DiscardAction
from catan.engine import apply_action_inplace, is_terminal, winner
from catan.state import Phase, new_game
from catan.validation import _fingerprint
from rl.actions import action_mask, indexed_action
from rl.encoding import FlatEncoder, decision_player
from rl.env import CatanEnv, Decision
from rl.policies import greedy_policy, random_policy


def _query(state, encoder):
    actor = decision_player(state)
    obs = encoder.encode(state)
    if state.phase == Phase.DISCARD:
        return Decision(
            actor,
            state.phase,
            obs,
            None,
            tuple(state.hands[actor].tolist()),
            int(state.discard_pending[actor]),
        )
    return Decision(actor, state.phase, obs, action_mask(state))


def episodes(environment, cap):
    actions = 0
    results = []
    for players in (3, 4):
        for seed in range(3):
            if environment:
                env = CatanEnv(max_actions=cap)
                request = env.reset(seed, n_players=players)
                state = env._state
            else:
                state = new_game(seed, players)
                encoder = FlatEncoder(state.board)
                request = _query(state, encoder)
            streams = [
                Random(f"catan-evaluation-v1:0:{seed}:{p}") for p in range(players)
            ]
            for step in range(1, cap + 1):
                policy = (
                    greedy_policy if request.player == seed % players else random_policy
                )
                choice = policy(request, streams[request.player])
                if environment:
                    outcome = env.step(choice, player=request.player)
                    ended = outcome.terminated or outcome.truncated
                else:
                    action = (
                        DiscardAction(request.player, choice)
                        if state.phase == Phase.DISCARD
                        else indexed_action(state, choice)
                    )
                    apply_action_inplace(state, action)
                    ended = is_terminal(state) or step == cap
                actions += 1
                if ended:
                    break
                request = env.decision() if environment else _query(state, encoder)
            digest = sha256(_fingerprint(state, streams[0]).encode())
            digest.update(repr(tuple(stream.getstate() for stream in streams)).encode())
            results.append((winner(state), step, digest.hexdigest()))
    return actions, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.repeats, args.max_actions) < 1:
        parser.error("repeats and max-actions must be positive")
    modes = {"direct_agent_loop": False, "environment_agent_loop": True}
    expected = None
    samples = {name: [] for name in modes}
    for mode in modes.values():
        result = episodes(mode, args.max_actions)
        if expected is None:
            expected = result
        assert result == expected
    for repeat in range(args.repeats):
        names = tuple(modes)
        if repeat % 2:
            names = names[::-1]
        for name in names:
            start = perf_counter()
            result = episodes(modes[name], args.max_actions)
            samples[name].append(perf_counter() - start)
            assert result == expected
    report = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "repeats": args.repeats,
        "max_actions": args.max_actions,
        "cases": "seeds 0-2, both seat counts, one greedy seat per game",
        "actions": expected[0],
        "wins": sum(winner is not None for winner, _, _ in expected[1]),
        "caps": sum(winner is None for winner, _, _ in expected[1]),
        "samples_seconds": samples,
        "median_seconds": {name: median(times) for name, times in samples.items()},
        "actions_per_second": {
            name: expected[0] / median(times) for name, times in samples.items()
        },
        "final_results": expected[1],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

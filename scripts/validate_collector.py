"""Replay collected reference episodes independently and audit seat links/outcomes.

Full invariants are explicit debug work. Each bounded worker owns its games and
buffers. Elapsed validation is not a performance measurement.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from hashlib import sha256
import json
from math import comb, log
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from catan.actions import DiscardAction
from catan.engine import apply_action_inplace, winner
from catan.state import Phase, new_game
from catan.validation import _fingerprint, assert_invariants
from rl.actions import action_mask, indexed_action
from rl.collector import ReferencePolicy, collect_rollouts
from rl.encoding import FlatEncoder, decision_player
from rl.env import Decision
from rl.policies import POLICIES


def verify_reference_batch(batch, names):
    """Direct engine/policy replay, independent of collector storage/link logic."""
    sparse = {int(row): i for i, row in enumerate(batch.discards.rows)}
    assert len(sparse) == len(batch.discards.rows)
    assert set(sparse) == set(np.flatnonzero(batch.actions == -1))
    checked = observations = 0
    for index, episode in enumerate(batch.episodes):
        state = new_game(episode.seed, episode.players)
        encoder = FlatEncoder(state.board)
        streams = [
            Random(f"catan-evaluation-v1:{episode.policy_seed}:{episode.seed}:{p}")
            for p in range(episode.players)
        ]
        rows = np.flatnonzero(batch.episode_indices == index)
        assert len(rows) == episode.actions
        prior = {}
        assert_invariants(state)
        checked += 1
        for step, row in enumerate(rows):
            actor = decision_player(state)
            obs = encoder.encode(state)
            assert batch.steps[row] == step
            assert batch.players[row] == actor and batch.phases[row] == state.phase
            np.testing.assert_array_equal(batch.observations[row], obs)
            observations += 1
            if state.phase == Phase.DISCARD:
                request = Decision(
                    actor,
                    state.phase,
                    obs,
                    None,
                    tuple(state.hands[actor].tolist()),
                    int(state.discard_pending[actor]),
                )
                d = sparse[int(row)]
                counts = tuple(map(int, batch.discards.counts[d]))
                np.testing.assert_array_equal(
                    batch.discards.hands[d], request.discard_hand
                )
                assert not batch.masks[row].any()
                for r, count in enumerate(counts):
                    assert request.discard_mask(counts[:r])[count]
                probability = (
                    log(
                        np.prod(
                            [comb(h, c) for h, c in zip(request.discard_hand, counts)],
                            dtype=object,
                        )
                        / comb(sum(request.discard_hand), sum(counts))
                    )
                    if names[actor] == "random"
                    else 0
                )
                np.testing.assert_allclose(
                    batch.discards.log_probs[d].sum(), probability, rtol=1e-6, atol=1e-6
                )
                choice = counts
                action = DiscardAction(actor, counts)
            else:
                mask = action_mask(state)
                request = Decision(actor, state.phase, obs, mask)
                np.testing.assert_array_equal(batch.masks[row], mask)
                choice = int(batch.actions[row])
                assert mask[choice]
                probability = (
                    -log(np.count_nonzero(mask)) if names[actor] == "random" else 0
                )
                action = indexed_action(state, choice)
            np.testing.assert_allclose(
                batch.log_probs[row], probability, rtol=1e-6, atol=1e-6
            )
            assert np.isnan(batch.values[row])
            assert choice == POLICIES[names[actor]](request, streams[actor])
            if actor in prior:
                previous = prior[actor]
                assert batch.next_rows[previous] == row
                assert batch.elapsed_actions[previous] == step - batch.steps[previous]
                assert (
                    batch.rewards[previous] == 0
                    and not batch.terminated[previous]
                    and not batch.truncated[previous]
                )
            prior[actor] = int(row)
            apply_action_inplace(state, action)
            assert_invariants(state)
            checked += 1
        won = winner(state)
        assert episode.winner == won and episode.terminated == (won is not None)
        assert episode.truncated != episode.terminated
        for seat in range(episode.players):
            expected = encoder.encode(state, seat)
            np.testing.assert_array_equal(
                batch.final_observations[index, seat], expected
            )
            observations += 1
            row = prior.get(seat, -1)
            assert batch.last_rows[index, seat] == row
            if row >= 0:
                assert batch.next_rows[row] == -1
                assert batch.elapsed_actions[row] == episode.actions - batch.steps[row]
                assert (
                    batch.terminated[row] == episode.terminated
                    and batch.truncated[row] == episode.truncated
                )
                assert batch.rewards[row] == float(seat == won)
        assert (batch.last_rows[index, episode.players :] == -1).all()
        assert not batch.final_observations[index, episode.players :].any()
        assert batch.rewards[rows].sum() == int(episode.terminated)
        digest = sha256(_fingerprint(state, streams[0]).encode())
        digest.update(repr(tuple(stream.getstate() for stream in streams)).encode())
        assert episode.fingerprint == digest.hexdigest()
    assert np.all(batch.elapsed_actions > 0)
    return checked, observations


def _case(request):
    seed, players, mode, cap = request
    names = (
        (mode,) * players
        if mode != "mixed"
        else tuple(
            "greedy" if seat == seed % players else "random" for seat in range(players)
        )
    )
    batch = collect_rollouts(
        ReferencePolicy(names),
        (seed, seed + 100000),
        n_players=players,
        batch_size=2,
        max_actions=cap,
        validate=True,
    )
    checks, observations = verify_reference_batch(batch, names)
    return {
        "seed": seed,
        "players": players,
        "mode": mode,
        "cap": cap,
        "actions": len(batch),
        "invariant_checks": 2 * checks,
        "oracle_observations": observations,
        "discards": len(batch.discards.rows),
        "phase_counts": [
            int(np.count_nonzero(batch.phases == phase)) for phase in Phase
        ],
        "episodes": [asdict(episode) for episode in batch.episodes],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="seeds per player count and policy configuration",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.seeds, args.workers, args.max_actions) < 1:
        parser.error("seeds, workers and max-actions must be positive")
    requests = [
        (seed, players, mode, args.max_actions)
        for seed in range(args.seeds)
        for players in (3, 4)
        for mode in ("random", "greedy", "mixed")
    ]
    # Include stopped games with some seats unacted and with several decisions.
    requests += [
        (seed, players, "random", cap)
        for seed in range(3)
        for players in (3, 4)
        for cap in (1, 2, 17)
    ]
    records = [None] * len(requests)
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(requests))) as pool:
        submitted = 0
        pending = {}
        while pending or submitted < len(requests):
            while len(pending) < 2 * args.workers and submitted < len(requests):
                pending[pool.submit(_case, requests[submitted])] = submitted
                submitted += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    records[index] = future.result()
                except Exception as error:
                    error.add_note(f"collector validation case={requests[index]}")
                    raise
    episodes = [episode for record in records for episode in record["episodes"]]
    digest = sha256()
    for record in records:
        digest.update(
            repr(
                (record["seed"], record["players"], record["mode"], record["cap"])
            ).encode()
        )
        for episode in record["episodes"]:
            digest.update(episode["fingerprint"].encode())
    report = {
        "games": len(episodes),
        "wins": sum(e["terminated"] for e in episodes),
        "cutoffs": sum(e["truncated"] for e in episodes),
        "actions": sum(r["actions"] for r in records),
        "invariant_checks": sum(r["invariant_checks"] for r in records),
        "oracle_observations": sum(r["oracle_observations"] for r in records),
        "discards": sum(r["discards"] for r in records),
        "phase_counts": {
            phase.name: sum(r["phase_counts"][phase.value - 1] for r in records)
            for phase in Phase
        },
        "fingerprint": digest.hexdigest(),
        "elapsed_validation_seconds": perf_counter() - start,
        "cases": records,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"}, indent=2
        )
    )


if __name__ == "__main__":
    main()

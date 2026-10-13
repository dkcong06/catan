"""Audit untrained neural rollouts, direct engine replay, gradients and targets.

No optimizer or parameter updates. Debug validation time is not performance.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch

from catan.actions import DiscardAction
from catan.engine import apply_action_inplace, winner
from catan.state import Phase, new_game
from catan.validation import _fingerprint, assert_invariants
from learning.model import ActorCritic
from learning.policy import NeuralPolicy, action_batch
from rl.actions import action_mask, indexed_action
from rl.collector import collect_rollouts
from rl.encoding import FlatEncoder, decision_player
from rl.env import Decision
from rl.returns import compute_targets


def _verify(batch, policy):
    sparse = {int(row): i for i, row in enumerate(batch.discards.rows)}
    checked = observations = 0
    for index, episode in enumerate(batch.episodes):
        state = new_game(episode.seed, episode.players)
        encoder = FlatEncoder(state.board)
        streams = [
            Random(f"catan-evaluation-v1:{episode.policy_seed}:{episode.seed}:{p}")
            for p in range(episode.players)
        ]
        assert_invariants(state)
        checked += 1
        rows = np.flatnonzero(batch.episode_indices == index)
        for step, row in enumerate(rows):
            actor = decision_player(state)
            assert (
                batch.steps[row] == step
                and batch.players[row] == actor
                and batch.phases[row] == state.phase
            )
            np.testing.assert_array_equal(
                batch.observations[row], encoder.encode(state)
            )
            observations += 1
            if state.phase == Phase.DISCARD:
                d = sparse[int(row)]
                counts = tuple(map(int, batch.discards.counts[d]))
                hand = tuple(state.hands[actor].tolist())
                np.testing.assert_array_equal(batch.discards.hands[d], hand)
                request = Decision(
                    actor,
                    state.phase,
                    batch.observations[row],
                    None,
                    hand,
                    int(state.discard_pending[actor]),
                )
                for resource, count in enumerate(counts):
                    mask = request.discard_mask(counts[:resource])
                    assert mask[count]
                    if np.count_nonzero(mask) > 1:
                        streams[actor].random()
                action = DiscardAction(actor, counts)
            else:
                mask = action_mask(state)
                np.testing.assert_array_equal(batch.masks[row], mask)
                choice = int(batch.actions[row])
                assert mask[choice]
                if np.count_nonzero(mask) > 1:
                    streams[actor].random()
                action = indexed_action(state, choice)
            apply_action_inplace(state, action)
            assert_invariants(state)
            checked += 1
        assert winner(state) == episode.winner
        for seat in range(episode.players):
            np.testing.assert_array_equal(
                batch.final_observations[index, seat], encoder.encode(state, seat)
            )
            observations += 1
        digest = sha256(_fingerprint(state, streams[0]).encode())
        digest.update(repr(tuple(stream.getstate() for stream in streams)).encode())
        assert digest.hexdigest() == episode.fingerprint
    # Different reevaluation batch sizes must preserve behavior logs/values within float32 roundoff.
    for start in range(0, len(batch), 128):
        tensors = action_batch(batch, np.arange(start, min(start + 128, len(batch))))
        evaluated = tensors.evaluate(policy.model)
        torch.testing.assert_close(
            evaluated.log_probs.detach(), tensors.old_log_probs, rtol=2e-5, atol=3e-6
        )
        torch.testing.assert_close(
            evaluated.values.detach(), tensors.old_values, rtol=2e-5, atol=3e-6
        )
        assert torch.isfinite(evaluated.entropies).all()
    bootstrap = policy.bootstrap_values(batch)
    target = compute_targets(batch, bootstrap_values=bootstrap, gae_lambda=1)
    for index, episode in enumerate(batch.episodes):
        for seat in range(episode.players):
            rows = np.flatnonzero(
                (batch.episode_indices == index) & (batch.players == seat)
            )
            expected = (
                float(seat == episode.winner)
                if episode.terminated
                else bootstrap[index, seat]
            )
            np.testing.assert_allclose(
                target.value_targets[rows], expected, rtol=2e-5, atol=3e-6
            )
    # One backward pass checks connectivity; weights must remain byte-identical.
    subset = np.unique(
        np.concatenate((np.arange(min(64, len(batch))), batch.discards.rows[:8]))
    )
    tensors = action_batch(batch, subset)
    evaluated = tensors.evaluate(policy.model)
    policy.model.zero_grad(set_to_none=True)
    loss = (
        -evaluated.log_probs.mean()
        + evaluated.values.square().mean()
        - 0.01 * evaluated.entropies.mean()
    )
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in policy.model.parameters()
    )
    return checked, observations


def _case(request):
    seed, players, cap = request
    torch.set_num_threads(1)
    model = ActorCritic(seed=seed % 7)
    policy = NeuralPolicy(model)
    before = {name: data.clone() for name, data in model.state_dict().items()}
    batch = collect_rollouts(
        policy,
        (seed, seed + 10000),
        n_players=players,
        max_actions=cap,
        batch_size=2,
        validate=True,
    )
    checks, observations = _verify(batch, policy)
    assert all(
        torch.equal(data, before[name]) for name, data in model.state_dict().items()
    )
    return {
        "seed": seed,
        "players": players,
        "cap": cap,
        "actions": len(batch),
        "invariant_checks": 2 * checks,
        "oracle_observations": observations,
        "discards": len(batch.discards.rows),
        "phase_counts": [
            int(np.count_nonzero(batch.phases == phase)) for phase in Phase
        ],
        "episodes": [asdict(e) for e in batch.episodes],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.seeds, args.workers, args.max_actions) < 1:
        parser.error("all bounds must be positive")
    requests = [
        (seed, players, args.max_actions)
        for seed in range(args.seeds)
        for players in (3, 4)
    ]
    requests += [
        (seed, players, cap)
        for seed in range(2)
        for players in (3, 4)
        for cap in (1, 2, 25)
    ]
    records = [None] * len(requests)
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        submitted = 0
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
                    error.add_note(f"learning validation case={requests[index]}")
                    raise
    episodes = [e for record in records for e in record["episodes"]]
    digest = sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
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
        "fingerprint": digest,
        "elapsed_validation_seconds": perf_counter() - start,
        "cases": records,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()

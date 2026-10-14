"""Bounded PPO audit with independent engine replay before/after real updates.

These updates test mechanics and exact checkpoint resumption, not playing strength.
Each worker owns its model, optimizer, simulator and random streams.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from learning.model import ActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from rl.collector import collect_rollouts
from scripts.validate_learning import _verify


def _equal(updater, restored):
    for name, value in updater.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    assert torch.equal(updater.generator.get_state(), restored.generator.get_state())
    assert updater.updates == restored.updates
    assert updater.optimizer_steps == restored.optimizer_steps
    for left, right in zip(
        updater.optimizer.state.values(), restored.optimizer.state.values(), strict=True
    ):
        for name in left:
            assert torch.equal(left[name], right[name])


def _case(request):
    seed, players, cap = request
    torch.set_num_threads(1)
    updater = PPOUpdater(
        ActorCritic(seed=seed % 7), PPOConfig(epochs=2), shuffle_seed=seed
    )
    global_rng = torch.random.get_rng_state().clone()
    episodes, stats = [], []
    checks = observations = actions = discards = 0
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"
        for iteration in range(2):
            policy = updater.policy()
            batch = collect_rollouts(
                policy,
                (seed + 100 * iteration, seed + 10000 + 100 * iteration),
                n_players=players,
                max_actions=cap,
                batch_size=2,
                validate=True,
            )
            checked, observed = _verify(batch, policy)
            checks += 2 * checked  # collection and independent replay invariants
            observations += observed
            actions += len(batch)
            discards += len(batch.discards.rows)
            updater.save(path)
            restored = PPOUpdater.load(path)
            _equal(updater, restored)
            before = {k: v.clone() for k, v in updater.model.state_dict().items()}
            result = updater.update(batch)
            resumed = restored.update(batch)
            assert result == resumed
            _equal(updater, restored)
            assert any(
                not torch.equal(before[k], v)
                for k, v in updater.model.state_dict().items()
            )
            assert torch.equal(global_rng, torch.random.get_rng_state())
            assert all(p.grad is None for p in updater.model.parameters())
            stats.append(asdict(result))
            episodes.extend(asdict(e) for e in batch.episodes)
    return {
        "seed": seed,
        "players": players,
        "cap": cap,
        "actions": actions,
        "invariant_checks": checks,
        "oracle_observations": observations,
        "discards": discards,
        "episodes": episodes,
        "updates": stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=3000)
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
        for cap in (1, 25, 65)
    ]
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(_case, requests))
    episodes = [e for case in cases for e in case["episodes"]]
    updates = [u for case in cases for u in case["updates"]]
    result = {
        "seconds_debug_audit": perf_counter() - start,
        "cases": len(cases),
        "episodes": len(episodes),
        "wins": sum(e["terminated"] for e in episodes),
        "cutoffs": sum(e["truncated"] for e in episodes),
        "actions": sum(c["actions"] for c in cases),
        "invariant_checks": sum(c["invariant_checks"] for c in cases),
        "oracle_observations": sum(c["oracle_observations"] for c in cases),
        "discards": sum(c["discards"] for c in cases),
        "updates": len(updates),
        "optimizer_steps": sum(u["optimizer_steps"] for u in updates),
        "early_stops": sum(u["early_stopped"] for u in updates),
        "exact_resumed_updates": len(updates),
        "fingerprint": sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "details": cases,
        "note": "Bounded optimizer-mechanics tests; debug runtime is not throughput or evidence of learned Catan strength.",
    }
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()

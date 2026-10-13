"""Measure untrained model inference, real collection and target assembly.

No optimizer, learning, debug invariants or archive I/O in timed runs. CPU thread
configuration is explicit here; importing/constructing a model never changes it.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import torch

from learning.model import ActorCritic
from learning.policy import NeuralPolicy
from rl.collector import collect_rollouts
from rl.returns import compute_targets


def _episodes(policy, width, games, cap):
    batches = []
    for players in (3, 4):
        batches.append(
            collect_rollouts(
                policy,
                range(games),
                n_players=players,
                max_actions=cap,
                batch_size=width,
            )
        )
    summary = tuple(
        (
            e.seed,
            e.players,
            e.actions,
            e.winner,
            e.terminated,
            e.truncated,
            e.fingerprint,
        )
        for b in batches
        for e in b.episodes
    )
    return batches, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4, help="games per player count")
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.games,
            args.max_actions,
            args.batch_size,
            args.hidden_size,
            args.threads,
            args.repeats,
        )
        < 1
    ):
        parser.error("all bounds must be positive")
    torch.set_num_threads(args.threads)
    policy = NeuralPolicy(ActorCritic(args.hidden_size, seed=0))
    modes = {"serial": 1, "batched": args.batch_size}
    samples = {name: [] for name in modes}
    expected = None
    final = None
    for width in modes.values():
        batches, summary = _episodes(policy, width, args.games, args.max_actions)
        if expected is None:
            expected = summary
        if summary != expected:
            raise AssertionError(
                "policy/game trajectories differ across batching; compare distributions before timing"
            )
    for repeat in range(args.repeats):
        names = list(modes)
        if repeat % 2:
            names.reverse()
        for name in names:
            start = perf_counter()
            batches, summary = _episodes(
                policy, modes[name], args.games, args.max_actions
            )
            samples[name].append(perf_counter() - start)
            assert summary == expected
            final = batches
    target_samples = []
    for _ in range(args.repeats):
        start = perf_counter()
        targets = [
            compute_targets(batch, bootstrap_values=policy.bootstrap_values(batch))
            for batch in final
        ]
        target_samples.append(perf_counter() - start)
        assert all(np.isfinite(target.advantages).all() for target in targets)
    generator = torch.Generator().manual_seed(10)
    forward = {}
    for width in (1, 8, 32, 128):
        obs = torch.rand(width, 1505, generator=generator)
        times = []
        with torch.inference_mode():
            policy.model(obs)
            for _ in range(args.repeats):
                start = perf_counter()
                for _ in range(300):
                    policy.model(obs)
                times.append((perf_counter() - start) / 300)
        forward[str(width)] = {
            "seconds": median(times),
            "observations_per_second": width / median(times),
        }
    actions = sum(len(batch) for batch in final)
    report = {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "model_seed": 0,
        "hidden_size": args.hidden_size,
        "parameters": sum(p.numel() for p in policy.model.parameters()),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "games": 2 * args.games,
        "max_actions": args.max_actions,
        "actions": actions,
        "samples_seconds": samples,
        "median_seconds": {name: median(times) for name, times in samples.items()},
        "actions_per_second": {
            name: actions / median(times) for name, times in samples.items()
        },
        "targets_plus_bootstrap_median_seconds": median(target_samples),
        "forward": forward,
        "final_results": expected,
        "fingerprint": sha256(repr(expected).encode()).hexdigest(),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

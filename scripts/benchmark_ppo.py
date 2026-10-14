"""Paired CPU PPO update measurements; no sustained training or archive I/O."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import torch

from learning.model import ActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from rl.collector import collect_rollouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.games, args.max_actions, args.repeats, args.threads) < 1:
        parser.error("all bounds must be positive")
    torch.set_num_threads(args.threads)
    source = PPOUpdater(ActorCritic(seed=0))
    start = perf_counter()
    batch = collect_rollouts(
        source.policy(), range(args.games), max_actions=args.max_actions, batch_size=8
    )
    collection_seconds = perf_counter() - start
    modes = [
        (size, foreach) for size in (128, 256, 512, 1024) for foreach in (False, True)
    ]
    samples = {f"minibatch{size}-foreach{foreach}": [] for size, foreach in modes}
    stats = {}
    weights = {}
    for repeat in range(args.repeats + 1):
        order = modes if repeat % 2 == 0 else modes[::-1]
        for size, foreach in order:
            name = f"minibatch{size}-foreach{foreach}"
            config = PPOConfig(
                minibatch_size=size,
                epochs=4,
                target_kl=None,
                adam_mode="foreach" if foreach else "scalar",
            )
            updater = PPOUpdater(ActorCritic(seed=0), config, shuffle_seed=0)
            start = perf_counter()
            result = updater.update(batch)
            elapsed = perf_counter() - start
            if repeat:
                samples[name].append(elapsed)
            stats[name] = asdict(result)
            if repeat == 0:
                weights[name] = {
                    k: v.clone() for k, v in updater.model.state_dict().items()
                }
    # Optimizer variants share row order/objective. Different minibatch sizes do not
    # implement the same parameter trajectory and cannot establish learning quality.
    maximum_difference = {}
    for size in (128, 256, 512, 1024):
        a, b = (
            weights[f"minibatch{size}-foreachFalse"],
            weights[f"minibatch{size}-foreachTrue"],
        )
        delta = max(float((a[k] - b[k]).abs().max()) for k in a)
        maximum_difference[str(size)] = delta
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=2e-5, atol=2e-6)
    # Default safety-enabled update, measured separately from the fixed-epoch sweep.
    defaults = []
    default_stats = None
    for repeat in range(args.repeats):
        updater = PPOUpdater(ActorCritic(seed=0))
        start = perf_counter()
        default_stats = updater.update(batch)
        defaults.append(perf_counter() - start)
    result = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu",
        "threads": args.threads,
        "games": args.games,
        "rows": len(batch),
        "discards": len(batch.discards.rows),
        "wins": sum(e.terminated for e in batch.episodes),
        "cutoffs": sum(e.truncated for e in batch.episodes),
        "collection_seconds_single_sample": collection_seconds,
        "trajectory_fingerprint": sha256(repr(batch.episodes).encode()).hexdigest(),
        "default": {
            "samples_seconds": defaults,
            "median_seconds": median(defaults),
            "stats": asdict(default_stats),
        },
        "sweep": {
            name: {
                "samples_seconds": times,
                "median_seconds": median(times),
                "training_row_visits_per_second": len(batch) * 4 / median(times),
                "stats": stats[name],
            }
            for name, times in samples.items()
        },
        "foreach_parameter_max_abs_difference": maximum_difference,
        "note": "Timing includes preflight/targets/gathers/forward/backward/Adam/diagnostics; model construction, collection, checkpoint I/O excluded. Different batch sizes are performance options, not equivalent learning results.",
    }
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

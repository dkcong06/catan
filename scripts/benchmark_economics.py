"""Isolated optional-feature costs; timings do not measure playing strength."""

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np

from rl.economics import ECONOMIC_OBS_DIM, EconomicEncoder, EconomicEnvFactory
from rl.encoding import FlatEncoder, OBS_DIM
from rl.env import CatanEnv
from rl.policies import POLICIES


def time_calls(callback, count):
    start = perf_counter()
    for _ in range(count):
        callback()
    return (perf_counter() - start) * 1e6 / count


def benchmark(checkpoint, *, repeats=5, iterations=500):
    import torch
    from learning.evaluation import EvaluationPolicy
    from learning.model import EconomicActorCritic
    from learning.policy import NeuralPolicy
    from learning.ppo import PPOUpdater
    from rl.collector import collect_rollouts
    from scripts.benchmark_decisions import digest

    if type(repeats) is not int or not 3 <= repeats <= 20:
        raise ValueError("repeats must be between 3 and 20")
    if type(iterations) is not int or not 10 <= iterations <= 10000:
        raise ValueError("iterations must be between 10 and 10000")
    torch.set_num_threads(1)
    original = PPOUpdater.load(checkpoint).model
    economic = EconomicActorCritic.from_base(original)
    state_env = CatanEnv()
    decision = state_env.reset(670000050, n_players=4)
    rng = Random(1)
    for _ in range(150):
        if decision is None:
            break
        state_env.step(POLICIES["greedy"](decision, rng), player=decision.player)
        decision = state_env.decision()
    state = state_env.snapshot()
    base_encoder = FlatEncoder(state.board)
    encoder = EconomicEncoder(state.board)
    base_out = np.empty(OBS_DIM, dtype=np.float32)
    out = np.empty(ECONOMIC_OBS_DIM, dtype=np.float32)
    callbacks = {
        "base_allocated": lambda: base_encoder.encode(state),
        "base_reused": lambda: base_encoder.encode(state, out=base_out),
        "economic_allocated": lambda: encoder.encode(state),
        "economic_reused": lambda: encoder.encode(state, out=out),
        "features_allocated": lambda: encoder.features.encode(state),
        "prepare_base": lambda: FlatEncoder(state.board),
        "prepare_economic": lambda: EconomicEncoder(state.board),
    }
    for callback in callbacks.values():
        callback()
    samples = {name: [] for name in callbacks}
    for repeat in range(repeats):
        names = list(callbacks)
        Random(repeat).shuffle(names)
        for name in names:
            samples[name].append(time_calls(callbacks[name], iterations))
    collection = {
        name: [] for name in ("legacy", "zero_suffix", "economic", "learner_only")
    }
    for repeat in range(repeats):
        names = list(collection)
        Random(repeat).shuffle(names)
        for name in names:
            actions, seconds = 0, 0.0
            for players in (3, 4):
                model = original if name == "legacy" else economic
                factory = (
                    None
                    if name == "legacy"
                    else EconomicEnvFactory(
                        enabled=name != "zero_suffix",
                        feature_player=0 if name == "learner_only" else None,
                    )
                )
                callback = EvaluationPolicy(NeuralPolicy(model), "greedy", 0)
                start = perf_counter()
                batch = collect_rollouts(
                    callback,
                    (670000051, 670000052),
                    n_players=players,
                    policy_seed=17,
                    env_factory=factory,
                    record_players=(0,),
                )
                elapsed = perf_counter() - start
                actions += sum(episode.actions for episode in batch.episodes)
                seconds += elapsed
                # Compare complete final state and per-seat policy RNG fingerprints.
                if name == names[0]:
                    if players == 3:
                        expected = {}
                    expected[players] = batch.episodes
                elif batch.episodes != expected[players]:
                    raise ValueError("zero-migration gameplay/RNG drift")
            collection[name].append(
                {
                    "actions": actions,
                    "seconds": seconds,
                    "actions_per_second": actions / seconds,
                }
            )
    return {
        "source_checkpoint_sha256": digest(checkpoint),
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
        "cpu_torch_threads": 1,
        "repeats": repeats,
        "iterations": iterations,
        "microseconds": {
            name: {"median": median(values), "samples": values}
            for name, values in samples.items()
        },
        "collection": {
            name: {
                "median_actions_per_second": median(
                    row["actions_per_second"] for row in values
                ),
                "samples": values,
            }
            for name, values in collection.items()
        },
        "collection_roles": "two fixed boards, each 3/4 players, learner seat0 vs greedy; per-seat RNG17",
        "exact_game_and_policy_rng_match": True,
        "additional_observation_bytes_per_row": (ECONOMIC_OBS_DIM - OBS_DIM) * 4,
        "additional_parameters": sum(p.numel() for p in economic.parameters())
        - sum(p.numel() for p in original.parameters()),
        "scope": "CPU encoding and bounded collection timings; no training or strength conclusion",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("benchmark output already exists")
    result = benchmark(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

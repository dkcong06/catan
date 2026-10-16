"""Isolated kernel, collector, update and final-evaluation cost probes."""

import argparse
from dataclasses import asdict
from pathlib import Path
import resource
from statistics import median
from time import perf_counter

import torch

from learning.evaluation import EvaluationPolicy, evaluate_neural
from learning.graph import GraphActorCritic
from learning.model import EconomicActorCritic
from learning.ppo import PPOUpdater
from learning.training import _atomic_json
from rl.collector import collect_rollouts
from rl.economics import EconomicEnvFactory


def kernels():
    model = GraphActorCritic()
    generator = torch.Generator().manual_seed(9)
    results = []
    for batch in (1, 8, 256):
        for name in ("tv", "vt", "ve", "ev"):
            incidence = getattr(model, name)
            neighbors = [torch.nonzero(row).flatten().tolist() for row in incidence]
            width = max(map(len, neighbors))
            indices = torch.tensor([r + [0] * (width - len(r)) for r in neighbors])
            weights = torch.tensor(
                [[1 / len(r)] * len(r) + [0] * (width - len(r)) for r in neighbors]
            )
            x = torch.randn(
                (batch, incidence.shape[1], 12), generator=generator, requires_grad=True
            )

            def dense():
                return incidence @ x

            def gather():
                return (x[:, indices] * weights[None, :, :, None]).sum(2)

            left, right = dense(), gather()
            error = float((left - right).abs().max().detach())
            grad_a = torch.autograd.grad(left.square().sum(), x)[0]
            grad_b = torch.autograd.grad(right.square().sum(), x)[0]
            gradient_error = float((grad_a - grad_b).abs().max())
            times = {"dense": [], "gather": []}
            for repeat in range(9):
                for method in (("dense", dense), ("gather", gather))[repeat % 2 :] + (
                    ("dense", dense),
                    ("gather", gather),
                )[: repeat % 2]:
                    label, function = method
                    started = perf_counter()
                    for _ in range(200):
                        function()
                    times[label].append((perf_counter() - started) / 200)
            results.append(
                dict(
                    batch=batch,
                    relation=name,
                    max_error=error,
                    gradient_error=gradient_error,
                    seconds={k: median(v) for k, v in times.items()},
                )
            )
    return results


def benchmark(economic_checkpoint, output):
    torch.set_num_threads(1)
    result = dict(kernels=kernels(), workloads={})
    for label in ("zero", "economic", "fresh_mlp", "graph"):
        source = PPOUpdater.load(economic_checkpoint)
        if label in ("zero", "economic"):
            updater = source
        else:
            model = (
                EconomicActorCritic(128, seed=1701)
                if label == "fresh_mlp"
                else GraphActorCritic(112, seed=1701)
            )
            updater = PPOUpdater(model, source.config, shuffle_seed=1901)
        model, enabled = updater.model, label != "zero"
        infer = {}
        for batch in (1, 8, 256):
            obs = torch.rand((batch, 2000), generator=torch.Generator().manual_seed(10))
            times = []
            with torch.inference_mode():
                for _ in range(7):
                    started = perf_counter()
                    for _ in range(50):
                        model(obs)
                    times.append((perf_counter() - started) / 50)
            infer[batch] = median(times)
        workloads = []
        for players in (3, 4):
            callback = EvaluationPolicy(updater.policy(), "greedy", 0)
            started = perf_counter()
            batch = collect_rollouts(
                callback,
                range(680300000 + players * 100, 680300008 + players * 100),
                n_players=players,
                batch_size=8,
                max_actions=5000,
                policy_seed=17,
                record_players=(0,),
                policy_tag=updater.policy().tag,
                env_factory=EconomicEnvFactory(enabled, feature_player=0),
            )
            collection_seconds = perf_counter() - started
            torch.set_num_threads(4)
            started = perf_counter()
            update = updater.update(batch)
            update_seconds = perf_counter() - started
            torch.set_num_threads(1)
            workloads.append(
                dict(
                    players=players,
                    rows=len(batch),
                    collection_seconds=collection_seconds,
                    update_seconds=update_seconds,
                    update=asdict(update),
                    engine_actions=sum(e.actions for e in batch.episodes),
                    payload_bytes=sum(
                        a.nbytes for a in (batch.observations, batch.masks)
                    ),
                    episodes=[asdict(e) for e in batch.episodes],
                )
            )
        # Only an isolated cost probe; none of these boards enter the experiment.
        started = perf_counter()
        evaluation = evaluate_neural(
            updater.policy(), [680310000], env_factory=EconomicEnvFactory(enabled)
        )
        result["workloads"][label] = dict(
            parameters=sum(p.numel() for p in model.parameters()),
            inference_seconds=infer,
            cohorts=workloads,
            evaluation_seconds=perf_counter() - started,
            evaluation_roles=len(evaluation["records"]),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        _atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economic-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.economic_checkpoint, args.output)


if __name__ == "__main__":
    main()

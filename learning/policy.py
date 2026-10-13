"""Collector-compatible inference and owned training minibatches, PyTorch optional."""

from bisect import bisect_right
from math import exp, isfinite, log
from typing import NamedTuple

import numpy as np
import torch

from learning.model import ActorCritic, MODEL_SCHEMA_VERSION
from rl.actions import ACTION_DIM
from rl.collector import PolicySample
from rl.encoding import OBS_DIM


def _choose(logits, mask, rng, deterministic):
    legal = np.flatnonzero(mask).tolist()
    if not legal or not np.isfinite(logits).all():
        raise ValueError("decision needs legal choices and finite logits")
    if len(legal) == 1:
        return legal[0], 0.0
    selected = max(legal, key=lambda index: float(logits[index]))
    maximum = float(logits[selected])
    weights = [exp(float(logits[index]) - maximum) for index in legal]
    total = sum(weights)
    if not deterministic:
        cumulative = []
        running = 0.0
        for weight in weights:
            running += weight
            cumulative.append(running)
        position = min(bisect_right(cumulative, rng.random() * total), len(legal) - 1)
        selected = legal[position]
    # Deterministic inference is a point-mass behavior policy, not a softmax draw.
    return selected, 0.0 if deterministic else float(logits[selected]) - maximum - log(
        total
    )


class NeuralPolicy:
    """One trunk call per collector callback, no gradients or parameter updates.

    Stochastic sampling uses only the supplied per-seat Random streams. Forced
    choices draw no randomness. Deterministic mode records point-mass logs zero.
    The caller owns CPU thread settings and model parameters/versioning.
    """

    def __init__(self, model, *, deterministic=False, policy_version="untrained-v1"):
        if not isinstance(model, ActorCritic):
            raise ValueError("model must be ActorCritic")
        if type(deterministic) is not bool or not isinstance(policy_version, str):
            raise ValueError("invalid inference mode or policy_version")
        if model.training:
            model.eval()
        self.model = model
        self.deterministic = deterministic
        self.tag = (
            f"mlp{MODEL_SCHEMA_VERSION}:{policy_version}:h{model.hidden_size}:seed{model.seed}:"
            + ("deterministic" if deterministic else "stochastic")
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __call__(self, decisions, rngs):
        decisions, rngs = tuple(decisions), tuple(rngs)
        if len(decisions) != len(rngs):
            raise ValueError("one policy RNG is required per decision")
        if not decisions:
            return ()
        observations = np.stack([request.observation for request in decisions])
        with torch.inference_mode():
            output = self.model(torch.from_numpy(observations).to(self.device))
            logits = output.logits.cpu().numpy()
            values = output.values.cpu().tolist()
            if not all(isfinite(value) for value in values):
                raise ValueError("critic predictions must be finite")
            samples = [None] * len(decisions)
            discards = []
            for i, request in enumerate(decisions):
                if request.discard_hand is not None:
                    discards.append(i)
                    continue
                if request.mask is None or request.mask.shape != (ACTION_DIM,):
                    raise ValueError("catalog decisions require flat masks")
                action, probability = _choose(
                    logits[i], request.mask, rngs[i], self.deterministic
                )
                samples[i] = PolicySample(action, probability, values[i])
            if discards:
                hidden = output.hidden[discards]
                prefix = np.zeros((len(discards), 5), dtype=np.float32)
                counts = [[] for _ in discards]
                probabilities = [[] for _ in discards]
                for resource in range(5):
                    next_logits = (
                        self.model.discard_logits(
                            hidden, torch.from_numpy(prefix).to(self.device), resource
                        )
                        .cpu()
                        .numpy()
                    )
                    for local, i in enumerate(discards):
                        mask = decisions[i].discard_mask(counts[local])
                        count, probability = _choose(
                            next_logits[local], mask, rngs[i], self.deterministic
                        )
                        counts[local].append(count)
                        probabilities[local].append(probability)
                        prefix[local, resource] = count
                for local, i in enumerate(discards):
                    logs = tuple(probabilities[local])
                    samples[i] = PolicySample(
                        tuple(counts[local]), sum(logs), values[i], logs
                    )
        return tuple(samples)

    def values(self, observations):
        array = np.asarray(observations)
        if array.ndim != 2 or array.shape[1] != OBS_DIM or array.dtype != np.float32:
            raise ValueError("observations must be float32 (N, OBS_DIM)")
        # Own writable storage before making a zero-copy CPU tensor.
        owned = np.array(array, copy=True, order="C")
        with torch.inference_mode():
            predicted = (
                self.model(torch.from_numpy(owned).to(self.device))
                .values.cpu()
                .numpy()
                .copy()
            )
        if not np.isfinite(predicted).all():
            raise ValueError("critic predictions must be finite")
        return predicted

    def bootstrap_values(self, batch):
        """Evaluate only acted cutoff seats; never predict a terminal future."""
        output = np.full(batch.last_rows.shape, np.nan, dtype=np.float32)
        episode_rows, seats = np.nonzero(batch.last_rows >= 0)
        used = np.asarray(
            [batch.episodes[index].truncated for index in episode_rows], dtype=np.bool_
        )
        episode_rows, seats = episode_rows[used], seats[used]
        if len(seats):
            output[episode_rows, seats] = self.values(
                batch.final_observations[episode_rows, seats]
            )
        return output


class ActionBatch(NamedTuple):
    observations: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    discard_rows: torch.Tensor
    discard_hands: torch.Tensor
    discard_counts: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor

    def evaluate(self, model):
        return model.evaluate_actions(*self[:6])


def action_batch(batch, rows, *, device="cpu"):
    """Gather only requested rows; owned writable CPU storage, no pickle/state.

    Repeated source indices are permitted. Sparse rows are remapped to local
    minibatch positions. Old behavior logs/values are detached training data.
    """
    rows = np.asarray(rows)
    if rows.ndim == 1 and rows.size == 0:
        rows = rows.astype(np.int64)
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or np.any(rows < 0)
        or np.any(rows >= len(batch))
    ):
        raise ValueError("rows must be one-dimensional valid integer indices")
    actions = batch.actions[rows].astype(np.int64)
    local = np.flatnonzero(actions == -1)
    positions = np.searchsorted(batch.discards.rows, rows[local])
    if len(local) and (
        np.any(positions >= len(batch.discards.rows))
        or not np.array_equal(batch.discards.rows[positions], rows[local])
    ):
        raise ValueError("missing sparse discard records")
    arrays = (
        batch.observations[rows],
        batch.masks[rows],
        actions,
        local.astype(np.int64),
        batch.discards.hands[positions].astype(np.int64),
        batch.discards.counts[positions].astype(np.int64),
        batch.log_probs[rows],
        batch.values[rows],
    )
    return ActionBatch(*(torch.from_numpy(array).to(device) for array in arrays))

"""Small float32 actor/critic with a shared, prefix-conditioned discard decoder."""

from math import sqrt
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from catan.state import _integer
from rl.actions import ACTION_DIM, ACTION_SCHEMA_VERSION
from rl.encoding import OBS_DIM, SCHEMA_VERSION

MODEL_SCHEMA_VERSION = 1


class NetworkOutput(NamedTuple):
    logits: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor


class ActionEvaluation(NamedTuple):
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor


def _distribution(logits, masks):
    if masks.shape != logits.shape or masks.dtype != torch.bool:
        raise ValueError("masks must be Boolean and match logits")
    if not masks.any(dim=1).all() or not torch.isfinite(logits).all():
        raise ValueError("every distribution needs a legal action and finite logits")
    logs = F.log_softmax(logits.masked_fill(~masks, -torch.inf), dim=1)
    probabilities = logs.exp()
    entropy = -(probabilities * logs.masked_fill(~masks, 0)).sum(dim=1)
    return logs, entropy


class ActorCritic(nn.Module):
    """Two ReLU layers, fused flat-policy/value output, one discard-count head.

    Initialization restores process Torch RNG state. No method changes thread
    settings, draws policy/game randomness, or updates parameters. Values are
    unconstrained reward estimates, not calibrated winning probabilities.
    """

    def __init__(self, hidden_size=128, *, seed=0):
        super().__init__()
        hidden_size = _integer(hidden_size, "hidden_size")
        seed = _integer(seed, "seed")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size, self.seed = hidden_size, seed
        self.schema = (MODEL_SCHEMA_VERSION, SCHEMA_VERSION, ACTION_SCHEMA_VERSION)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.trunk = nn.Sequential(
                nn.Linear(OBS_DIM, hidden_size, dtype=torch.float32, device="cpu"),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size, dtype=torch.float32, device="cpu"),
                nn.ReLU(),
            )
            self.output = nn.Linear(
                hidden_size, ACTION_DIM + 1, dtype=torch.float32, device="cpu"
            )
            self.discard = nn.Linear(
                hidden_size + 10, 20, dtype=torch.float32, device="cpu"
            )
            for layer in (self.trunk[0], self.trunk[2]):
                nn.init.orthogonal_(layer.weight, sqrt(2))
                nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.output.weight, 0.01)
            nn.init.orthogonal_(self.output.weight[-1:], 1.0)
            nn.init.zeros_(self.output.bias)
            nn.init.orthogonal_(self.discard.weight, 0.01)
            nn.init.zeros_(self.discard.bias)
        self.register_buffer(
            "resource_codes", torch.eye(5, dtype=torch.float32, device="cpu")
        )
        self.register_buffer("count_choices", torch.arange(20, device="cpu"))

    def forward(self, observations):
        if (
            observations.ndim != 2
            or observations.shape[1] != OBS_DIM
            or observations.dtype != torch.float32
        ):
            raise ValueError("observations must be float32 of shape (N, OBS_DIM)")
        hidden = self.trunk(observations)
        output = self.output(hidden)
        return NetworkOutput(output[:, :ACTION_DIM], output[:, ACTION_DIM], hidden)

    def discard_logits(self, hidden, prefix, resource):
        code = self.resource_codes[resource].expand(len(hidden), -1)
        return self.discard(torch.cat((hidden, prefix / 19.0, code), dim=1))

    def evaluate_actions(
        self, observations, masks, actions, discard_rows, discard_hands, discard_counts
    ):
        """Differentiable log probabilities for stored choices and prefixes.

        Flat entropy is exact. Discard entropy sums conditional entropies along
        the supplied prefix trajectory; it is not enumerated full joint entropy.
        Indices are local to this minibatch and may include repeated source rows.
        """
        n = len(observations)
        if (
            masks.shape != (n, ACTION_DIM)
            or masks.dtype != torch.bool
            or actions.shape != (n,)
            or actions.dtype != torch.long
        ):
            raise ValueError("invalid mask/action shapes or dtypes")
        if (
            discard_rows.ndim != 1
            or discard_rows.dtype != torch.long
            or discard_hands.shape != (len(discard_rows), 5)
            or discard_counts.shape != discard_hands.shape
        ):
            raise ValueError("invalid sparse discard shapes")
        if discard_hands.dtype != torch.long or discard_counts.dtype != torch.long:
            raise ValueError("discard hands/counts must be int64")
        expected = torch.nonzero(actions == -1).flatten()
        if not torch.equal(discard_rows, expected):
            raise ValueError(
                "discard rows must match actions=-1 in ascending local order"
            )
        regular = torch.nonzero(actions != -1).flatten()
        if ((actions[regular] < 0) | (actions[regular] >= ACTION_DIM)).any():
            raise ValueError("catalog action outside range")
        if len(regular) and not masks[regular, actions[regular]].all():
            raise ValueError("selected catalog action is illegal")
        if len(discard_rows):
            totals = discard_hands.sum(dim=1)
            if (
                (
                    (discard_hands < 0)
                    | (discard_hands > 19)
                    | (discard_counts < 0)
                    | (discard_counts > discard_hands)
                ).any()
                or (totals <= 7).any()
                or not torch.equal(discard_counts.sum(dim=1), totals // 2)
            ):
                raise ValueError("invalid complete physical discard")
        output = self(observations)
        log_probs = output.values.new_zeros(n)
        entropies = output.values.new_zeros(n)
        if len(regular):
            logs, entropy = _distribution(output.logits[regular], masks[regular])
            selected = logs.gather(1, actions[regular, None]).squeeze(1)
            log_probs = log_probs.index_copy(0, regular, selected)
            entropies = entropies.index_copy(0, regular, entropy)
        if len(discard_rows):
            hidden = output.hidden[discard_rows]
            d = len(discard_rows)
            # Recorded prefixes are known: evaluate all five conditional heads in
            # one call. Sampling remains sequential in NeuralPolicy.
            resources = self.count_choices[:5]
            prefix = (
                discard_counts[:, None, :]
                * (resources[:, None] > resources[None, :])[None]
            ).float()
            remaining = discard_counts.sum(dim=1)[:, None] - prefix.sum(dim=2)
            following = discard_hands.sum(dim=1)[:, None] - discard_hands.cumsum(dim=1)
            minimum = (remaining - following).clamp_min(0)
            maximum = torch.minimum(remaining, discard_hands)
            feasible = (self.count_choices >= minimum[:, :, None]) & (
                self.count_choices <= maximum[:, :, None]
            )
            features = torch.cat(
                (
                    hidden[:, None, :].expand(-1, 5, -1),
                    prefix / 19.0,
                    self.resource_codes[None].expand(d, -1, -1),
                ),
                dim=2,
            )
            logs, entropy = _distribution(
                self.discard(features).reshape(d * 5, 20), feasible.reshape(d * 5, 20)
            )
            joint = (
                logs.gather(1, discard_counts.reshape(-1, 1)).reshape(d, 5).sum(dim=1)
            )
            entropy_sum = entropy.reshape(d, 5).sum(dim=1)
            log_probs = log_probs.index_copy(0, discard_rows, joint)
            entropies = entropies.index_copy(0, discard_rows, entropy_sum)
        return ActionEvaluation(log_probs, entropies, output.values)

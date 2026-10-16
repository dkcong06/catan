"""Bounded PPO updates for fresh stochastic Catan rollouts, PyTorch optional.

Collection, model updates and checkpoint I/O are separate. No training loop,
self-play scheduler, replay buffer or change to engine/collector schemas.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
import os
import tempfile
from typing import NamedTuple

import numpy as np
import torch

from catan.state import _integer
from learning.model import (
    ActorCritic,
    EconomicActorCritic,
    KnowledgeActorCritic,
    MODEL_SCHEMA_VERSION,
)
from learning.policy import NeuralPolicy, action_batch
from learning.graph import GraphActorCritic
from rl.actions import ACTION_DIM, ACTION_SCHEMA_VERSION
from rl.collector import COLLECTOR_SCHEMA_VERSION, RolloutBatch
from rl.encoding import SCHEMA_VERSION
from rl.env import ENV_SCHEMA_VERSION, REWARD_SCHEME
from rl.returns import TARGET_SCHEMA_VERSION, compute_targets

PPO_SCHEMA_VERSION = 1
_SCHEMAS = (
    PPO_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ACTION_SCHEMA_VERSION,
    ENV_SCHEMA_VERSION,
    COLLECTOR_SCHEMA_VERSION,
    TARGET_SCHEMA_VERSION,
    REWARD_SCHEME,
)
_ECONOMIC_SCHEMAS = (
    3,
    3,
    4,
    ACTION_SCHEMA_VERSION,
    ENV_SCHEMA_VERSION,
    3,
    TARGET_SCHEMA_VERSION,
    REWARD_SCHEME,
)
_GRAPH_SCHEMAS = (
    4,
    4,
    4,
    ACTION_SCHEMA_VERSION,
    ENV_SCHEMA_VERSION,
    3,
    TARGET_SCHEMA_VERSION,
    REWARD_SCHEME,
)


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    if not isfinite(float(value)) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {name}")
    return float(value)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    clip_ratio: float = 0.2
    value_clip: float | None = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    gamma: float = 1.0
    gae_lambda: float = 0.95
    discount: str = "decision"
    normalize_advantages: bool = True
    adam_mode: str = "fused"
    critic_updates_trunk: bool = True

    def __post_init__(self):
        for name in ("epochs", "minibatch_size"):
            value = _integer(getattr(self, name), name)
            object.__setattr__(self, name, value)
            if value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "max_grad_norm", "clip_ratio"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, positive=True)
            )
        if self.clip_ratio >= 1:
            raise ValueError("clip_ratio must be less than one")
        for name in ("value_clip", "target_kl"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, _number(getattr(self, name), name, positive=True)
                )
        for name in ("value_coefficient", "entropy_coefficient", "gamma", "gae_lambda"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.gamma > 1 or self.gae_lambda > 1:
            raise ValueError("gamma/gae_lambda must be at most one")
        if self.discount not in ("decision", "engine"):
            raise ValueError("discount must be decision or engine")
        if type(self.normalize_advantages) is not bool:
            raise ValueError("normalize_advantages must be Boolean")
        if type(self.critic_updates_trunk) is not bool:
            raise ValueError("critic_updates_trunk must be Boolean")

        if self.adam_mode not in ("scalar", "foreach", "fused"):
            raise ValueError("adam_mode must be scalar, foreach or fused")


class PPOLoss(NamedTuple):
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(evaluation, old_log_probs, old_values, advantages, targets, config):
    """Clipped joint-action objective, detached old data and fixed targets.

    Value loss includes the conventional half factor. Discard entropy is a
    behavior-prefix conditional regularizer, not exact current joint entropy;
    disabled by default. KL is the nonnegative sampled expm1(logratio)-logratio.
    """
    n = evaluation.log_probs.numel()
    columns = (*evaluation, old_log_probs, old_values, advantages, targets)
    if n < 1 or any(x.shape != (n,) or x.dtype != torch.float32 for x in columns):
        raise ValueError("loss columns must be nonempty float32 vectors")
    old_log_probs, old_values, advantages, targets = (
        x.detach() for x in (old_log_probs, old_values, advantages, targets)
    )
    log_ratio = evaluation.log_probs - old_log_probs
    ratio = log_ratio.exp()
    policy = -torch.minimum(
        ratio * advantages,
        ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages,
    ).mean()
    errors = (evaluation.values - targets).square()
    if config.value_clip is not None:
        clipped = old_values + (evaluation.values - old_values).clamp(
            -config.value_clip, config.value_clip
        )
        errors = torch.maximum(errors, (clipped - targets).square())
    value = 0.5 * errors.mean()
    entropy = evaluation.entropies.mean()
    total = (
        policy + config.value_coefficient * value - config.entropy_coefficient * entropy
    )
    approximate_kl = (torch.expm1(log_ratio) - log_ratio).mean().detach().clamp_min(0)
    clip_fraction = ((ratio - 1).abs() > config.clip_ratio).float().mean().detach()
    result = PPOLoss(total, policy, value, entropy, approximate_kl, clip_fraction)
    if not torch.isfinite(torch.stack(result)).all():
        raise ValueError("PPO loss/ratio is nonfinite")
    return result


@dataclass(frozen=True, slots=True)
class UpdateStats:
    rows: int
    update: int
    optimizer_steps: int
    completed_epochs: int
    early_stopped: bool
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    max_unclipped_grad_norm: float
    advantage_mean: float
    advantage_std: float
    policy_tag: str


class PPOUpdater:
    """One fixed-rollout update at a time; caller owns device and thread settings.

    Preflight rejection leaves model, optimizer and shuffle RNG unchanged.
    Runtime numerical failure quarantines this updater: recover from a saved
    checkpoint. No costly per-step weight/optimizer rollback copies are made.
    """

    def __init__(
        self, model, config=None, *, shuffle_seed=0, reward_scheme=REWARD_SCHEME
    ):
        if not isinstance(model, ActorCritic):
            raise ValueError("model must be ActorCritic")
        config = PPOConfig() if config is None else config
        if not isinstance(config, PPOConfig):
            raise ValueError("config must be PPOConfig")
        if any(
            p.dtype != torch.float32 or not torch.isfinite(p).all()
            for p in model.parameters()
        ):
            raise ValueError("model parameters must be finite float32")
        if reward_scheme not in (REWARD_SCHEME, "terminal_win_net_vp_0.1_v1"):
            raise ValueError("unknown reward scheme")
        if reward_scheme != REWARD_SCHEME and model.schema[0] != 2:
            raise ValueError("progress rewards require the versioned experiment model")
        self.model, self.config, self.reward_scheme = model, config, reward_scheme
        self.shuffle_seed = _integer(shuffle_seed, "shuffle_seed")
        self.generator = torch.Generator(device="cpu").manual_seed(self.shuffle_seed)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
            foreach=config.adam_mode == "foreach",
            fused=config.adam_mode == "fused",
        )
        self.updates = self.optimizer_steps = 0
        self._failed = False

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _ready(self):
        if self._failed:
            raise RuntimeError(
                "updater failed; restore a valid checkpoint before continuing"
            )

    def policy(self):
        self._ready()
        return NeuralPolicy(
            self.model, policy_version=f"ppo{self._schemas()[0]}-u{self.updates}"
        )

    def _schemas(self):
        if self.model.schema[0] == 4:
            return _GRAPH_SCHEMAS
        if self.model.schema[0] == 1:
            return _SCHEMAS
        if self.model.schema[0] == 3:
            return _ECONOMIC_SCHEMAS
        return (
            2,
            2,
            3,
            ACTION_SCHEMA_VERSION,
            ENV_SCHEMA_VERSION,
            2,
            TARGET_SCHEMA_VERSION,
            self.reward_scheme,
        )

    def _preflight(self, batch):
        self._ready()
        if not isinstance(batch, RolloutBatch) or len(batch) < 1:
            raise ValueError("a nonempty RolloutBatch is required")
        for name, shape, dtype in (
            ("observations", (len(batch), self.model.observation_dim), np.float32),
            ("masks", (len(batch), ACTION_DIM), np.bool_),
            ("actions", (len(batch),), np.int16),
            ("log_probs", (len(batch),), np.float32),
            ("values", (len(batch),), np.float32),
        ):
            column = getattr(batch, name)
            if column.shape != shape or column.dtype != dtype:
                raise ValueError(f"invalid rollout {name} shape/dtype")
        if (
            batch.observation_schema != self.model.schema[1]
            or batch.reward_scheme != self.reward_scheme
        ):
            raise ValueError("rollout observation/reward profile differs from updater")
        d = len(batch.discards.rows)
        if batch.discards.rows.dtype != np.int32 or not np.array_equal(
            batch.discards.rows, np.flatnonzero(batch.actions == -1)
        ):
            raise ValueError("invalid sparse discard rows")
        for name in ("hands", "counts"):
            column = getattr(batch.discards, name)
            if column.shape != (d, 5) or column.dtype != np.int8:
                raise ValueError("invalid sparse discard hand/count storage")
        logs = batch.discards.log_probs
        if (
            logs.shape != (d, 5)
            or logs.dtype != np.float32
            or not np.isfinite(logs).all()
            or np.any(logs > 1e-6)
            or not np.allclose(
                logs.sum(axis=1),
                batch.log_probs[batch.discards.rows],
                rtol=2e-5,
                atol=3e-6,
            )
        ):
            raise ValueError("invalid conditional discard behavior probabilities")
        policy = self.policy()
        if batch.policy_tag != policy.tag:
            raise ValueError("rollout must come from this version's stochastic policy")
        if (
            not np.isfinite(batch.observations).all()
            or not np.isfinite(batch.log_probs).all()
            or np.any(batch.log_probs > 1e-6)
            or not np.isfinite(batch.values).all()
        ):
            raise ValueError("rollout observations/behavior logs/values must be finite")
        # Check every choice and behavior prediction before any optimizer mutation.
        with torch.no_grad():
            for start in range(0, len(batch), self.config.minibatch_size):
                inputs = action_batch(
                    batch,
                    np.arange(
                        start, min(start + self.config.minibatch_size, len(batch))
                    ),
                    device=self.device,
                )
                evaluated = inputs.evaluate(self.model)
                if not torch.allclose(
                    evaluated.log_probs, inputs.old_log_probs, rtol=2e-5, atol=3e-6
                ):
                    raise ValueError(
                        "behavior probabilities do not match unchanged model"
                    )
                if not torch.allclose(
                    evaluated.values, inputs.old_values, rtol=2e-5, atol=3e-6
                ):
                    # Small collection batches and large GEMMs can round a
                    # near-cancelling critic dot product differently. Verify
                    # only suspect rows with the scalar kernel, using the
                    # identical existing tolerance; never replace stored values.
                    suspect = torch.nonzero(
                        ~torch.isclose(
                            evaluated.values, inputs.old_values, rtol=2e-5, atol=3e-6
                        )
                    ).flatten()
                    scalar = torch.stack(
                        [
                            self.model(inputs.observations[i : i + 1]).values[0]
                            for i in suspect.tolist()
                        ]
                    )
                    if not torch.allclose(
                        scalar, inputs.old_values[suspect], rtol=2e-5, atol=3e-6
                    ):
                        raise ValueError("behavior values do not match unchanged model")
        targets = compute_targets(
            batch,
            bootstrap_values=policy.bootstrap_values(batch),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            discount=self.config.discount,
        )
        advantages = targets.advantages.astype(np.float64)
        mean, std = float(advantages.mean()), float(advantages.std())
        if self.config.normalize_advantages:
            advantages = (advantages - mean) / max(std, 1e-8)
        return advantages.astype(np.float32), targets.value_targets, mean, std

    def _diagnostics(self, batch):
        kl = clipped = 0.0
        with torch.no_grad():
            for start in range(0, len(batch), self.config.minibatch_size):
                rows = np.arange(
                    start, min(start + self.config.minibatch_size, len(batch))
                )
                inputs = action_batch(batch, rows, device=self.device)
                logs = inputs.evaluate(self.model).log_probs - inputs.old_log_probs
                metrics = torch.stack(
                    (
                        (torch.expm1(logs) - logs).sum().clamp_min(0),
                        ((logs.exp() - 1).abs() > self.config.clip_ratio).float().sum(),
                    )
                )
                if not torch.isfinite(metrics).all():
                    raise ValueError("post-update diagnostics are nonfinite")
                a, b = metrics.cpu().tolist()
                kl += a
                clipped += b
        return kl / len(batch), clipped / len(batch)

    def update(self, batch, *, anchor=None):
        if anchor is not None:
            from learning.tactics import ReplayAnchor

            if not isinstance(anchor, ReplayAnchor):
                raise ValueError("anchor must be a bounded ReplayAnchor")
            anchor.prepare(self.model)
        advantages, targets, mean, std = self._preflight(batch)
        accumulated = np.zeros(3, dtype=np.float64)
        seen = steps = completed = 0
        max_norm = kl = clipped = 0.0
        stopped = False
        self.model.train()
        try:
            for _ in range(self.config.epochs):
                order = torch.randperm(len(batch), generator=self.generator).numpy()
                for start in range(0, len(batch), self.config.minibatch_size):
                    rows = order[start : start + self.config.minibatch_size]
                    inputs = action_batch(batch, rows, device=self.device)
                    loss = ppo_loss(
                        inputs.evaluate(
                            self.model,
                            critic_updates_trunk=self.config.critic_updates_trunk,
                        ),
                        inputs.old_log_probs,
                        inputs.old_values,
                        torch.from_numpy(advantages[rows]).to(self.device),
                        torch.from_numpy(targets[rows]).to(self.device),
                        self.config,
                    )
                    if (
                        self.config.target_kl is not None
                        and float(loss.approximate_kl) > self.config.target_kl
                    ):
                        stopped = True
                        break
                    self.optimizer.zero_grad(set_to_none=True)
                    objective = loss.total
                    if anchor is not None:
                        objective = objective + anchor.loss(self.model)
                        if not torch.isfinite(objective):
                            raise FloatingPointError("nonfinite anchored objective")
                    objective.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True,
                        foreach=self.config.adam_mode == "foreach",
                    )
                    self.optimizer.step()
                    self.optimizer_steps += 1
                    steps += 1
                    if any(
                        not torch.isfinite(p).all() for p in self.model.parameters()
                    ):
                        raise ValueError(
                            "optimizer produced nonfinite model parameters"
                        )
                    accumulated += np.asarray(
                        [float(x.detach()) for x in loss[1:4]]
                    ) * len(rows)
                    seen += len(rows)
                    max_norm = max(max_norm, float(norm))
                if not stopped:
                    completed += 1
                # Check all rollout rows, not only the most recent shuffled minibatch.
                kl, clipped = self._diagnostics(batch)
                if stopped or (
                    self.config.target_kl is not None and kl > self.config.target_kl
                ):
                    stopped = True
                    break
            self.updates += 1
            averages = accumulated / seen if seen else accumulated
            return UpdateStats(
                len(batch),
                self.updates,
                steps,
                completed,
                stopped,
                *map(float, averages),
                kl,
                clipped,
                max_norm,
                mean,
                std,
                self.policy().tag,
            )
        except Exception:
            self._failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()

    def save(self, path):
        """Atomic local checkpoint; weights/Adam/local shuffle state, no game archive."""
        self._ready()
        path = Path(path)
        payload = {
            "schemas": self._schemas(),
            "config": asdict(self.config),
            "hidden_size": self.model.hidden_size,
            "model_seed": self.model.seed,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "shuffle_seed": self.shuffle_seed,
            "shuffle_state": self.generator.get_state(),
            "updates": self.updates,
            "optimizer_steps": self.optimizer_steps,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(handle)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, *, device="cpu"):
        """Load into a new updater; strict schemas and weights_only deserialization."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "schemas",
            "config",
            "hidden_size",
            "model_seed",
            "model",
            "optimizer",
            "shuffle_seed",
            "shuffle_state",
            "updates",
            "optimizer_steps",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("incompatible PPO checkpoint/schema")
        schemas = data["schemas"]
        if schemas == _SCHEMAS:
            model_type = ActorCritic
        elif schemas == _ECONOMIC_SCHEMAS:
            model_type = EconomicActorCritic
        elif schemas == _GRAPH_SCHEMAS:
            model_type = GraphActorCritic
        elif schemas in tuple(
            (
                2,
                2,
                3,
                ACTION_SCHEMA_VERSION,
                ENV_SCHEMA_VERSION,
                2,
                TARGET_SCHEMA_VERSION,
                reward,
            )
            for reward in (REWARD_SCHEME, "terminal_win_net_vp_0.1_v1")
        ):
            model_type = KnowledgeActorCritic
        else:
            raise ValueError("incompatible PPO checkpoint/schema")
        config = PPOConfig(**data["config"])
        model = model_type(data["hidden_size"], seed=data["model_seed"])
        expected_state = model.state_dict()
        if set(data["model"]) != set(expected_state):
            raise ValueError("checkpoint model fields do not match architecture")
        for name, tensor in data["model"].items():
            expected_tensor = expected_state[name]
            if (
                tensor.shape != expected_tensor.shape
                or tensor.dtype != expected_tensor.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("invalid checkpoint model tensor")
            if name in getattr(
                model, "fixed_buffers", ("resource_codes", "count_choices")
            ) and not torch.equal(tensor, expected_tensor):
                raise ValueError("invalid fixed model buffer")
        model.load_state_dict(data["model"], strict=True)
        result = cls(
            model.to(device),
            config,
            shuffle_seed=data["shuffle_seed"],
            reward_scheme=schemas[-1],
        )
        for name in ("updates", "optimizer_steps"):
            value = _integer(data[name], name)
            if value < 0:
                raise ValueError("negative checkpoint counter")
            setattr(result, name, value)
        result.generator.set_state(data["shuffle_state"])
        expected = result.optimizer.defaults.copy()
        raw = data["optimizer"]
        if set(raw) != {"state", "param_groups"} or len(raw["param_groups"]) != 1:
            raise ValueError("checkpoint needs one optimizer parameter group")
        group = raw["param_groups"][0]
        for name, value in expected.items():
            if group.get(name) != value:
                raise ValueError("checkpoint optimizer settings do not match config")
        parameters = list(result.model.parameters())
        identifiers = group["params"]
        if (
            len(identifiers) != len(parameters)
            or len(set(identifiers)) != len(identifiers)
            or any(type(key) is not int for key in identifiers)
            or not set(raw["state"]).issubset(identifiers)
        ):
            raise ValueError("invalid Adam parameter mapping")
        # Check raw storage before load_state_dict silently casts moments.
        for identifier, parameter in zip(identifiers, parameters, strict=True):
            if identifier not in raw["state"]:
                continue  # A head without gradients has no initialized Adam state.
            state = raw["state"][identifier]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam checkpoint fields")
            for name, tensor in state.items():
                shape = () if name == "step" else parameter.shape
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != shape
                    or tensor.dtype != torch.float32
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError("invalid Adam checkpoint tensor")
                if name == "step" and (
                    tensor != tensor.floor() or tensor > result.optimizer_steps
                ):
                    raise ValueError("invalid Adam checkpoint step counter")
                if name in ("step", "exp_avg_sq") and (tensor < 0).any():
                    raise ValueError("invalid Adam checkpoint negative state")
        result.optimizer.load_state_dict(raw)
        result.model.eval()
        return result

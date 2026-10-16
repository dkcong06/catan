"""Bounded greedy demonstrations and actor-only imitation; no PPO behavior data."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from catan.state import Phase, _integer
from learning.model import ActorCritic
from learning.policy import action_inputs
from rl.actions import ACTION_DIM, ACTION_SCHEMA_VERSION
from rl.collector import Discards, RolloutBatch
from rl.encoding import OBS_DIM, SCHEMA_VERSION
from rl.env import ENV_SCHEMA_VERSION
from rl.economics import ECONOMIC_OBS_DIM

DEMONSTRATION_SCHEMA_VERSION = 1
_SCHEMAS = (
    DEMONSTRATION_SCHEMA_VERSION,
    ENV_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ACTION_SCHEMA_VERSION,
)


@dataclass(frozen=True, slots=True, eq=False)
class Demonstrations:
    """Read-only supervised labels, deliberately not a PPO RolloutBatch."""

    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    phases: np.ndarray
    discards: Discards
    teacher_tag: str
    observation_schema: int = SCHEMA_VERSION

    def __len__(self):
        return len(self.actions)

    @classmethod
    def from_rollouts(cls, batch):
        if not isinstance(batch, RolloutBatch):
            raise ValueError("teacher collection must be a RolloutBatch")
        # Forced flat decisions have zero actor gradient and need no storage.
        rows = np.flatnonzero((batch.actions == -1) | (batch.masks.sum(axis=1) > 1))
        local = np.flatnonzero(batch.actions[rows] == -1)
        positions = np.searchsorted(batch.discards.rows, rows[local])
        arrays = [
            batch.observations[rows],
            batch.masks[rows],
            batch.actions[rows],
            batch.phases[rows],
            local.astype(np.int32),
            batch.discards.hands[positions],
            batch.discards.counts[positions],
            batch.discards.log_probs[positions],
        ]
        for array in arrays:
            array.flags.writeable = False
        result = cls(
            *arrays[:4],
            Discards(*arrays[4:]),
            batch.policy_tag,
            batch.observation_schema,
        )
        result.validate()
        return result

    def select(self, rows):
        """Owned read-only subset, reindexing complete sparse discards."""
        rows = np.asarray(rows)
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or len(rows) < 1
            or np.any(rows < 0)
            or np.any(rows >= len(self))
            or np.any(np.diff(rows) <= 0)
        ):
            raise ValueError("teacher subset needs nonempty ascending unique rows")
        local = np.flatnonzero(self.actions[rows] == -1)
        positions = np.searchsorted(self.discards.rows, rows[local])
        arrays = [
            self.observations[rows],
            self.masks[rows],
            self.actions[rows],
            self.phases[rows],
            local.astype(np.int32),
            self.discards.hands[positions],
            self.discards.counts[positions],
            self.discards.log_probs[positions],
        ]
        for array in arrays:
            array.flags.writeable = False
        result = type(self)(
            *arrays[:4],
            Discards(*arrays[4:]),
            self.teacher_tag,
            self.observation_schema,
        )
        result.validate()
        return result

    def validate(self):
        n = len(self)
        if n < 1 or not isinstance(self.teacher_tag, str) or not self.teacher_tag:
            raise ValueError("nonempty demonstrations need a teacher tag")
        if self.observation_schema not in (2, 3, 4):
            raise ValueError("unsupported teacher observation schema")
        dimension = {2: OBS_DIM, 3: OBS_DIM + 30, 4: ECONOMIC_OBS_DIM}[
            self.observation_schema
        ]
        for name, shape, dtype in (
            ("observations", (n, dimension), np.float32),
            ("masks", (n, ACTION_DIM), np.bool_),
            ("actions", (n,), np.int16),
            ("phases", (n,), np.int8),
        ):
            array = getattr(self, name)
            if (
                array.shape != shape
                or array.dtype != dtype
                or not array.flags.c_contiguous
            ):
                raise ValueError(f"invalid demonstration {name}")
        if not np.isfinite(self.observations).all():
            raise ValueError("demonstration observations must be finite")
        if not np.isin(
            self.phases, [int(p) for p in Phase if p != Phase.GAME_OVER]
        ).all():
            raise ValueError("invalid demonstration phase")
        discard = self.actions == -1
        if not np.array_equal(discard, self.phases == Phase.DISCARD):
            raise ValueError("discard action/phase mismatch")
        rows = np.flatnonzero(discard)
        if self.discards.rows.dtype != np.int32 or not np.array_equal(
            self.discards.rows, rows
        ):
            raise ValueError("invalid demonstration discard rows")
        for name in ("hands", "counts"):
            array = getattr(self.discards, name)
            if array.shape != (len(rows), 5) or array.dtype != np.int8:
                raise ValueError("invalid demonstration discard storage")
        hands, counts = self.discards.hands, self.discards.counts
        if (
            (hands < 0).any()
            or (hands > 19).any()
            or (counts < 0).any()
            or (counts > hands).any()
            or (hands.sum(axis=1) <= 7).any()
            or not np.array_equal(counts.sum(axis=1), hands.sum(axis=1) // 2)
            or self.masks[rows].any()
        ):
            raise ValueError("invalid complete teacher discard")
        logs = self.discards.log_probs
        if (
            logs.shape != (len(rows), 5)
            or logs.dtype != np.float32
            or not np.isfinite(logs).all()
            or (logs > 1e-6).any()
        ):
            raise ValueError("invalid teacher conditional logs")
        regular = np.flatnonzero(~discard)
        if ((self.actions[regular] < 0) | (self.actions[regular] >= ACTION_DIM)).any():
            raise ValueError("teacher action outside catalog")
        if (
            not self.masks[regular, self.actions[regular]].all()
            or (self.masks[regular].sum(axis=1) <= 1).any()
        ):
            raise ValueError("teacher labels must be legal non-forced choices")

    def save(self, path):
        self.validate()
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("demonstrations require an .npz path")
        schemas = (
            _SCHEMAS
            if self.observation_schema == 2
            else (
                self.observation_schema - 1,
                ENV_SCHEMA_VERSION,
                self.observation_schema,
                ACTION_SCHEMA_VERSION,
            )
        )
        metadata = json.dumps({"schemas": schemas, "teacher_tag": self.teacher_tag})
        # Numeric labels only; exclusive creation preserves existing datasets.
        with path.open("xb") as stream:
            np.savez(
                stream,
                observations=self.observations,
                masks=self.masks,
                actions=self.actions,
                phases=self.phases,
                discard_rows=self.discards.rows,
                discard_hands=self.discards.hands,
                discard_counts=self.discards.counts,
                discard_log_probs=self.discards.log_probs,
                metadata=np.asarray(metadata),
            )

    @classmethod
    def load(cls, path, *, max_rows=40000):
        max_rows = _integer(max_rows, "max_rows")
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        # Reject oversized compressed archives before allocating their arrays.
        with ZipFile(path) as archive:
            if (
                sum(item.file_size for item in archive.infolist())
                > max_rows * (ECONOMIC_OBS_DIM * 4 + ACTION_DIM + 64) + 32768
            ):
                raise ValueError("demonstration archive exceeds memory bound")
        fields = (
            "observations",
            "masks",
            "actions",
            "phases",
            "discard_rows",
            "discard_hands",
            "discard_counts",
            "discard_log_probs",
        )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(fields) | {"metadata"}:
                raise ValueError("invalid demonstration archive fields")
            metadata = json.loads(str(archive["metadata"]))
            schemas = tuple(metadata.get("schemas", ()))
            if schemas not in (
                _SCHEMAS,
                (2, ENV_SCHEMA_VERSION, 3, ACTION_SCHEMA_VERSION),
                (3, ENV_SCHEMA_VERSION, 4, ACTION_SCHEMA_VERSION),
            ):
                raise ValueError("incompatible demonstration schema")
            arrays = [archive[name] for name in fields]
        if len(arrays[2]) > max_rows:
            raise ValueError("too many demonstration rows")
        for array in arrays:
            array.flags.writeable = False
        result = cls(
            *arrays[:4], Discards(*arrays[4:]), metadata["teacher_tag"], schemas[2]
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class ImitationConfig:
    epochs: int = 3
    minibatch_size: int = 256
    learning_rate: float = 0.001
    max_grad_norm: float = 0.5
    balance_phases: bool = True

    def __post_init__(self):
        for name in ("epochs", "minibatch_size"):
            value = _integer(getattr(self, name), name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.learning_rate > 1 or type(self.balance_phases) is not bool:
            raise ValueError("invalid imitation learning rate or phase balancing")


def phase_weights(phases, balanced=True):
    """Mean-one weights; every represented decision phase has equal total mass."""
    phases = np.asarray(phases)
    if phases.ndim != 1 or phases.dtype.kind not in "iu" or len(phases) < 1:
        raise ValueError("nonempty integer phases required")
    if not balanced:
        return np.ones(len(phases), dtype=np.float32)
    _, inverse, counts = np.unique(phases, return_inverse=True, return_counts=True)
    return (len(phases) / (len(counts) * counts[inverse])).astype(np.float32)


class ImitationTrainer:
    """Separate Adam and local shuffle RNG; critic outputs are not training labels.

    fit consumes one bounded shard at a time. Caller owns thread settings, shard
    order and persistence; malformed labels reject before weights/RNG mutation.
    Runtime failures quarantine the trainer rather than copying weights each step.
    """

    def __init__(self, model, config=None, *, shuffle_seed=0):
        config = ImitationConfig() if config is None else config
        if not isinstance(model, ActorCritic) or not isinstance(
            config, ImitationConfig
        ):
            raise ValueError("use ActorCritic and ImitationConfig")
        if any(
            p.dtype != torch.float32 or not torch.isfinite(p).all()
            for p in model.parameters()
        ):
            raise ValueError("model parameters must be finite float32")
        self.model, self.config = model, config
        self.generator = torch.Generator(device="cpu").manual_seed(
            _integer(shuffle_seed, "shuffle_seed")
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, eps=1e-5, fused=True
        )
        self.optimizer_steps = 0
        self.failed = False

    @property
    def device(self):
        return next(self.model.parameters()).device

    def loss(self, data):
        data.validate()
        if data.observation_schema != self.model.schema[1]:
            raise ValueError("teacher/model observation profile mismatch")
        weights = phase_weights(data.phases, self.config.balance_phases)
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(data), self.config.minibatch_size):
                rows = np.arange(
                    start, min(start + self.config.minibatch_size, len(data))
                )
                evaluated = self.model.evaluate_actions(
                    *action_inputs(data, rows, device=self.device),
                    compute_entropy=False,
                )
                if not torch.isfinite(evaluated.log_probs).all():
                    raise ValueError("teacher likelihood must be finite")
                total -= float(
                    (
                        evaluated.log_probs
                        * torch.from_numpy(weights[rows]).to(self.device)
                    ).sum()
                )
        return total / len(data)

    def fit(self, data):
        if self.failed:
            raise RuntimeError("imitation trainer failed; restore a checkpoint")
        if not isinstance(data, Demonstrations):
            raise ValueError("use supervised Demonstrations, not PPO records")
        before = self.loss(data)  # Complete validation before Adam/shuffle mutation.
        weights = phase_weights(data.phases, self.config.balance_phases)
        steps = self.optimizer_steps
        started = perf_counter()
        training = self.model.training
        try:
            self.model.train()
            for _ in range(self.config.epochs):
                order = torch.randperm(len(data), generator=self.generator).numpy()
                for start in range(0, len(data), self.config.minibatch_size):
                    rows = order[start : start + self.config.minibatch_size]
                    self.optimizer.zero_grad(set_to_none=True)
                    evaluation = self.model.evaluate_actions(
                        *action_inputs(data, rows, device=self.device),
                        compute_entropy=False,
                    )
                    loss = -(
                        evaluation.log_probs
                        * torch.from_numpy(weights[rows]).to(self.device)
                    ).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite imitation loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    self.optimizer.step()
                    self.optimizer_steps += 1
            after = self.loss(data)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.model.train(training)
            self.optimizer.zero_grad(set_to_none=True)
        return {
            "rows": len(data),
            "phases": {
                str(int(p)): int(n)
                for p, n in zip(*np.unique(data.phases, return_counts=True))
            },
            "config": asdict(self.config),
            "optimizer_steps": self.optimizer_steps - steps,
            "loss_before": before,
            "loss_after": after,
            "seconds": perf_counter() - started,
        }

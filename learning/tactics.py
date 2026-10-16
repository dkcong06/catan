"""Bounded corrected lessons, set-valued tactical loss and optional PPO replay."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from learning.imitation import Demonstrations, ImitationConfig, phase_weights
from learning.policy import action_inputs
from rl.actions import ACTION_DIM


@dataclass(frozen=True, slots=True)
class TacticalLessons:
    demonstrations: Demonstrations
    wins: np.ndarray

    def __len__(self):
        return len(self.demonstrations)

    def validate(self):
        self.demonstrations.validate()
        data = self.demonstrations
        if self.wins.dtype != np.bool_ or self.wins.shape != (len(self), ACTION_DIM):
            raise ValueError("winning sets must be an N by 254 Boolean array")
        if np.any(self.wins & ~data.masks):
            raise ValueError("winning labels must be legal")
        tactical = self.wins.any(1)
        rows = np.flatnonzero(tactical)
        if len(rows) and (
            np.any(data.actions[rows] < 0)
            or not self.wins[rows, data.actions[rows]].all()
        ):
            raise ValueError("corrected teacher must take a proved winning action")

    @property
    def tactical(self):
        return self.wins.any(1)

    def weights(self):
        tactical = self.tactical
        ordinary = ~tactical
        weights = np.ones(len(self), dtype=np.float32)
        if tactical.any() and ordinary.any():
            weights[tactical] = len(self) / (2 * tactical.sum())
            weights[ordinary] = (
                len(self)
                / (2 * ordinary.sum())
                * phase_weights(self.demonstrations.phases[ordinary])
            )
        elif ordinary.any():
            weights = phase_weights(self.demonstrations.phases)
        return weights

    def save(self, directory):
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.demonstrations.save(directory / "demonstrations.npz")
        with (directory / "winning-sets.npz").open("xb") as stream:
            np.savez(stream, wins=self.wins)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        data = Demonstrations.load(directory / "demonstrations.npz", max_rows=16000)
        # Fixed numeric uncompressed output; reject unexpectedly large payloads.
        path = directory / "winning-sets.npz"
        if path.stat().st_size > 16000 * ACTION_DIM + 4096:
            raise ValueError("winning-set archive exceeds bounds")
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["wins"]:
                raise ValueError("invalid winning-set archive")
            wins = archive["wins"]
        wins.flags.writeable = False
        result = cls(data, wins)
        result.validate()
        return result


def lesson_logs(model, lessons, rows):
    """Log P(any proved winning action), otherwise complete teacher likelihood.

    Split ordinary/tactical rows so each row requires only one trunk evaluation.
    Discards retain the existing five-count joint likelihood.
    """
    data = lessons.demonstrations
    device = next(model.parameters()).device
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or np.any(rows < 0) or np.any(rows >= len(data)):
        raise ValueError("nonempty valid lesson rows required")
    tactical = lessons.wins[rows].any(1)
    result = torch.zeros(len(rows), device=device)
    ordinary = np.flatnonzero(~tactical)
    if len(ordinary):
        values = model.evaluate_actions(
            *action_inputs(data, rows[ordinary], device=device), compute_entropy=False
        ).log_probs
        result = result.index_copy(0, torch.as_tensor(ordinary, device=device), values)
    selected = np.flatnonzero(tactical)
    if len(selected):
        source = rows[selected]
        output = model(torch.from_numpy(data.observations[source]).to(device))
        legal = torch.from_numpy(data.masks[source]).to(device)
        wins = torch.from_numpy(lessons.wins[source]).to(device)
        logs = torch.logsumexp(output.logits.masked_fill(~wins, -torch.inf), 1)
        logs = logs - torch.logsumexp(output.logits.masked_fill(~legal, -torch.inf), 1)
        result = result.index_copy(0, torch.as_tensor(selected, device=device), logs)
    return result


def score_lessons(model, lessons, *, batch_size=256):
    lessons.validate()
    data = lessons.demonstrations
    if data.observation_schema != model.schema[1]:
        raise ValueError("lesson/model observation profile mismatch")
    weights = lessons.weights()
    total = mass = correct = tactical_count = ordinary_correct = ordinary_count = 0
    with torch.no_grad():
        for start in range(0, len(lessons), batch_size):
            rows = np.arange(start, min(start + batch_size, len(lessons)))
            logs = lesson_logs(model, lessons, rows)
            total -= float(
                (logs * torch.from_numpy(weights[rows]).to(logs.device)).sum()
            )
            flat = rows[data.actions[rows] >= 0]
            if not len(flat):
                continue
            logits = model(
                torch.from_numpy(data.observations[flat]).to(logs.device)
            ).logits
            legal = torch.from_numpy(data.masks[flat]).to(logs.device)
            choices = logits.masked_fill(~legal, -torch.inf).argmax(1).cpu().numpy()
            tactical = lessons.wins[flat].any(1)
            picked = lessons.wins[flat, choices]
            correct += int(picked[tactical].sum())
            tactical_count += int(tactical.sum())
            ordinary_correct += int(
                (choices[~tactical] == data.actions[flat[~tactical]]).sum()
            )
            ordinary_count += int((~tactical).sum())
            mass += float(logs[lessons.wins[rows].any(1)].exp().sum())
    return dict(
        rows=len(lessons),
        tactical_rows=tactical_count,
        weighted_loss=total / len(lessons),
        winning_probability=mass / tactical_count if tactical_count else None,
        winning_argmax=correct / tactical_count if tactical_count else None,
        ordinary_flat_agreement=ordinary_correct / ordinary_count
        if ordinary_count
        else None,
    )


class LessonTrainer:
    """Actor-only warm-up with local Adam/shuffle and predeclared pass counts."""

    def __init__(self, model, *, seed=0, learning_rate=1e-4, batch_size=128):
        self.model = model
        self.config = ImitationConfig(
            epochs=1, minibatch_size=batch_size, learning_rate=learning_rate
        )
        self.generator = torch.Generator().manual_seed(seed)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, eps=1e-5, fused=True
        )
        self.steps = 0
        self.failed = False

    def fit(self, lessons):
        if self.failed:
            raise RuntimeError("lesson trainer failed; restore a saved warmup")
        lessons.validate()
        if lessons.demonstrations.observation_schema != self.model.schema[1]:
            raise ValueError("lesson/model observation profile mismatch")
        weights = lessons.weights()
        order = torch.randperm(len(lessons), generator=self.generator).numpy()
        try:
            for start in range(0, len(order), self.config.minibatch_size):
                rows = order[start : start + self.config.minibatch_size]
                logs = lesson_logs(self.model, lessons, rows)
                loss = -(logs * torch.from_numpy(weights[rows]).to(logs.device)).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite lesson loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.steps += 1
        except BaseException:
            self.failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()


class ReplayAnchor:
    """Training-only balanced tactical/ordinary replay; no value or PPO targets.

    The experiment runner persists this separate sampler state alongside each
    PPO checkpoint. Default PPO has no anchor and no replay overhead.
    """

    def __init__(self, lessons, *, coefficient=0.1, batch_size=32, seed=0):
        if (
            not isinstance(lessons, TacticalLessons)
            or type(coefficient) not in (int, float)
            or not np.isfinite(coefficient)
            or not 0 < coefficient <= 1
            or type(batch_size) is not int
            or not 2 <= batch_size <= 256
            or batch_size % 2
        ):
            raise ValueError("invalid bounded replay anchor")
        lessons.validate()
        self.lessons = lessons
        self.coefficient = float(coefficient)
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.groups = tuple(
            np.flatnonzero(mask) for mask in (lessons.tactical, ~lessons.tactical)
        )
        ordinary = self.groups[1]
        self.ordinary_weights = (
            phase_weights(lessons.demonstrations.phases[ordinary])
            if len(ordinary)
            else None
        )

    def prepare(self, model):
        self.lessons.validate()
        if model.schema[1] != self.lessons.demonstrations.observation_schema:
            raise ValueError("anchor/model observation profile mismatch")

    def loss(self, model):
        pieces = []
        for i, group in enumerate(self.groups):
            if not len(group):
                continue
            count = self.batch_size // 2
            positions = torch.randint(
                len(group), (count,), generator=self.generator
            ).numpy()
            logs = lesson_logs(model, self.lessons, group[positions])
            if i == 1:
                logs = logs * torch.from_numpy(self.ordinary_weights[positions]).to(
                    logs.device
                )
            pieces.append(-logs.mean())
        return self.coefficient * torch.stack(pieces).mean()

    def state_dict(self):
        return dict(
            schema=1,
            coefficient=self.coefficient,
            batch_size=self.batch_size,
            generator=self.generator.get_state(),
        )

    def load_state_dict(self, state):
        if (state["schema"], state["coefficient"], state["batch_size"]) != (
            1,
            self.coefficient,
            self.batch_size,
        ):
            raise ValueError("anchor recipe changed")
        self.generator.set_state(state["generator"])

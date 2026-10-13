"""Same-seat generalized advantage estimates and lambda-return value targets.

NumPy only. No optimizer, model access, hidden information or implicit discount.
A caller must supply real cutoff bootstrap estimates; terminal futures are zero.
"""

from math import isfinite
from numbers import Real
from typing import NamedTuple

import numpy as np

TARGET_SCHEMA_VERSION = 1


class Targets(NamedTuple):
    advantages: np.ndarray
    value_targets: np.ndarray


def _unit(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number between zero and one")
    return float(value)


def _values(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "fi":
        raise ValueError(f"{name} must have shape {shape} and a real numeric dtype")
    return result.astype(np.float64, copy=False)


def compute_targets(
    batch,
    *,
    values=None,
    bootstrap_values=None,
    gamma=1.0,
    gae_lambda=0.95,
    discount="decision",
):
    """Unnormalized GAE and value+advantage targets for one owned rollout batch.

    decision discounts once per personal decision. engine raises both gamma and
    lambda to elapsed_actions. Defaults keep the final win objective undiscounted.
    lambda=1 yields bootstrapped Monte Carlo returns; lambda=0 is one-step TD.
    Cutoff bootstrap input has shape (episodes,4); only acted truncated seats need
    finite values. Unused entries may be NaN. No trace crosses a seat/game/closure.
    """
    gamma, gae_lambda = _unit(gamma, "gamma"), _unit(gae_lambda, "gae_lambda")
    if discount not in ("decision", "engine"):
        raise ValueError("discount must be decision or engine")
    n, episodes = len(batch), len(batch.episodes)
    predictions = _values(batch.values if values is None else values, (n,), "values")
    rewards = _values(batch.rewards, (n,), "rewards")
    storage_limit = float(np.finfo(np.float32).max)
    if (
        not np.isfinite(predictions).all()
        or not np.isfinite(rewards).all()
        or np.any(np.abs(predictions) > storage_limit)
        or np.any(np.abs(rewards) > storage_limit)
    ):
        raise ValueError(
            "values and rewards must be finite; missing critic predictions need explicit values"
        )
    links, gaps = batch.next_rows, batch.elapsed_actions
    if (
        links.shape != (n,)
        or links.dtype.kind not in "iu"
        or gaps.shape != (n,)
        or gaps.dtype.kind not in "iu"
    ):
        raise ValueError("invalid successor/duration columns")
    rows = np.arange(n)
    if np.any(links < -1) or np.any(links >= n) or np.any(gaps <= 0):
        raise ValueError("successors must be in range and durations positive")
    internal = links >= 0
    if np.any(links[internal] <= rows[internal]):
        raise ValueError("successor links must point forward")
    if np.any(np.bincount(links[internal], minlength=n) > 1):
        raise ValueError("a successor cannot have multiple previous decisions")
    for column, upper, name in (
        (batch.episode_indices, episodes, "episodes"),
        (batch.players, 4, "players"),
    ):
        if (
            column.shape != (n,)
            or column.dtype.kind not in "iu"
            or np.any(column < 0)
            or np.any(column >= upper)
        ):
            raise ValueError(f"invalid {name} column")
    if np.any(
        batch.episode_indices[internal] != batch.episode_indices[links[internal]]
    ) or np.any(batch.players[internal] != batch.players[links[internal]]):
        raise ValueError("successor crosses episode or player")
    if (
        batch.terminated.shape != (n,)
        or batch.truncated.shape != (n,)
        or batch.terminated.dtype != np.bool_
        or batch.truncated.dtype != np.bool_
    ):
        raise ValueError("invalid closure flag columns")
    closing = ~internal
    if np.any((batch.terminated | batch.truncated) != closing) or np.any(
        batch.terminated & batch.truncated
    ):
        raise ValueError("closure flags must match final per-seat rows")
    present = np.asarray([episode.players for episode in batch.episodes])
    if np.any(batch.players >= present[batch.episode_indices]):
        raise ValueError("player is absent in its episode")
    closing_seats = batch.episode_indices[closing] * 4 + batch.players[closing]
    if np.any(np.bincount(closing_seats, minlength=episodes * 4) > 1):
        raise ValueError("a seat has multiple closing records")
    next_values = np.zeros(n, dtype=np.float64)
    next_values[internal] = predictions[links[internal]]
    cuts = batch.truncated
    if np.any(cuts):
        if bootstrap_values is None:
            raise ValueError("truncated seats require explicit bootstrap_values")
        bootstrap = _values(bootstrap_values, (episodes, 4), "bootstrap_values")
        used = bootstrap[batch.episode_indices[cuts], batch.players[cuts]]
        if not np.isfinite(used).all() or np.any(np.abs(used) > storage_limit):
            raise ValueError("acted truncated seats need finite bootstrap values")
        next_values[cuts] = used
    elif bootstrap_values is not None:
        _values(bootstrap_values, (episodes, 4), "bootstrap_values")
    powers = gaps if discount == "engine" else np.ones(n, dtype=np.int8)
    factors = np.power(gamma, powers)
    traces = factors * np.power(gae_lambda, powers)
    residuals = rewards + factors * next_values - predictions
    advantages = _scan_advantages(residuals, traces, links)
    value_targets = predictions + advantages
    limit = float(np.finfo(np.float32).max)
    if (
        not np.isfinite(advantages).all()
        or not np.isfinite(value_targets).all()
        or np.any(np.abs(advantages) > limit)
        or np.any(np.abs(value_targets) > limit)
    ):
        raise ValueError("targets are outside finite float32 storage")
    output = Targets(advantages.astype(np.float32), value_targets.astype(np.float32))
    for array in output:
        array.flags.writeable = False
    return output


def _scan_advantages(residuals, traces, links):
    # Many independent trajectories favor bounded vector waves. Small/few long
    # trajectories favor Python floats over repeated NumPy scalar dispatch.
    n = len(links)
    if n >= 4096 and np.count_nonzero(links == -1) >= 64:
        previous = np.full(n, -1, dtype=np.int32)
        indices = np.flatnonzero(links >= 0)
        previous[links[indices]] = indices
        frontier = np.flatnonzero(links == -1)
        result = np.empty(n, dtype=np.float64)
        result[frontier] = residuals[frontier]
        while len(frontier):
            frontier = previous[frontier]
            frontier = frontier[frontier >= 0]
            result[frontier] = (
                residuals[frontier] + traces[frontier] * result[links[frontier]]
            )
        return result
    deltas, factors, successors = residuals.tolist(), traces.tolist(), links.tolist()
    advantages = [0.0] * n
    for row in range(n - 1, -1, -1):
        link = successors[row]
        advantages[row] = deltas[row] + (
            factors[row] * advantages[link] if link >= 0 else 0
        )
    return np.asarray(advantages, dtype=np.float64)

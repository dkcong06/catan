"""Independent analytical and per-trajectory target oracles."""

from types import SimpleNamespace

import numpy as np
import pytest

from rl.collector import ReferencePolicy, collect_rollouts
from rl.returns import compute_targets


class Fixture(SimpleNamespace):
    def __len__(self):
        return len(self.values)


def fixture(*, cuts=False):
    return Fixture(
        values=np.array([0.2, 0.4, 0.3, 0.1], dtype=np.float32),
        rewards=np.array([0, 0, 1, 0], dtype=np.float32),
        next_rows=np.array([2, 3, -1, -1], dtype=np.int32),
        elapsed_actions=np.array([3, 2, 1, 1], dtype=np.int32),
        episode_indices=np.zeros(4, dtype=np.int32),
        players=np.array([0, 1, 0, 1], dtype=np.int8),
        terminated=np.array([False, False, not cuts, not cuts]),
        truncated=np.array([False, False, cuts, cuts]),
        episodes=(SimpleNamespace(players=3, actions=4),),
    )


@pytest.mark.parametrize("gamma", (0, 0.7, 1))
@pytest.mark.parametrize("lam", (0, 0.5, 1))
@pytest.mark.parametrize("discount", ("decision", "engine"))
def test_analytical_interleaved_players_and_trace_units(gamma, lam, discount):
    batch = fixture()
    expected = np.zeros(4)
    for trajectory in ((0, 2), (1, 3)):
        following = 0
        for row in reversed(trajectory):
            power = int(batch.elapsed_actions[row]) if discount == "engine" else 1
            d = gamma**power
            residual = (
                batch.rewards[row]
                + d
                * (
                    batch.values[batch.next_rows[row]]
                    if batch.next_rows[row] >= 0
                    else 0
                )
                - batch.values[row]
            )
            following = residual + d * lam**power * following
            expected[row] = following
    actual = compute_targets(batch, gamma=gamma, gae_lambda=lam, discount=discount)
    np.testing.assert_allclose(actual.advantages, expected, atol=1e-7)
    np.testing.assert_allclose(actual.value_targets, expected + batch.values, atol=1e-7)


def test_full_returns_equal_actual_outcomes_and_no_cross_player_credit():
    result = compute_targets(fixture(), gae_lambda=1)
    np.testing.assert_allclose(result.value_targets, [1, 0, 1, 0], atol=1e-7)
    np.testing.assert_allclose(result.advantages, [0.8, -0.4, 0.7, -0.1], atol=1e-7)


def test_cutoff_requires_real_values_and_never_invents_zero_future():
    batch = fixture(cuts=True)
    batch.rewards.fill(0)
    with pytest.raises(ValueError, match="bootstrap"):
        compute_targets(batch)
    bootstrap = np.array([[0.8, 0.6, np.nan, np.nan]], dtype=np.float32)
    result = compute_targets(batch, bootstrap_values=bootstrap, gae_lambda=1)
    np.testing.assert_allclose(result.value_targets, [0.8, 0.6, 0.8, 0.6], atol=1e-7)
    for row in (0, 1):
        bad = bootstrap.copy()
        bad[0, row] = np.nan
        with pytest.raises(ValueError, match="finite"):
            compute_targets(batch, bootstrap_values=bad)
    # True terminal futures are zero even when an unused estimate is NaN/large.
    normal = compute_targets(fixture(), bootstrap_values=bootstrap, gae_lambda=1)
    np.testing.assert_allclose(normal.value_targets, [1, 0, 1, 0], atol=1e-7)


@pytest.mark.parametrize("discount", ("decision", "engine"))
def test_collector_trajectories_match_recursive_oracle_and_source_isolation(discount):
    batch = collect_rollouts(
        ReferencePolicy(), (0, 1), max_actions=120, batch_size=2, n_players=3
    )
    rng = np.random.default_rng(15)
    values = rng.uniform(-0.5, 1.5, len(batch)).astype(np.float32)
    bootstrap = rng.uniform(-0.5, 1.5, (2, 4)).astype(np.float32)
    before = values.copy(), bootstrap.copy(), batch.next_rows.copy()
    expected = np.empty(len(batch))
    gamma, lam = 0.9, 0.8
    for episode in range(2):
        for player in range(3):
            rows = np.flatnonzero(
                (batch.episode_indices == episode) & (batch.players == player)
            )
            following = 0
            for position in range(len(rows) - 1, -1, -1):
                row = rows[position]
                power = int(batch.elapsed_actions[row]) if discount == "engine" else 1
                next_value = (
                    values[rows[position + 1]]
                    if position + 1 < len(rows)
                    else bootstrap[episode, player]
                )
                delta = (
                    float(batch.rewards[row])
                    + gamma**power * float(next_value)
                    - float(values[row])
                )
                following = delta + gamma**power * lam**power * following
                expected[row] = following
    actual = compute_targets(
        batch,
        values=values,
        bootstrap_values=bootstrap,
        gamma=gamma,
        gae_lambda=lam,
        discount=discount,
    )
    np.testing.assert_allclose(actual.advantages, expected, atol=2e-7)
    np.testing.assert_array_equal(values, before[0])
    np.testing.assert_array_equal(bootstrap, before[1])
    np.testing.assert_array_equal(batch.next_rows, before[2])
    assert all(
        a.dtype == np.float32 and a.flags.c_contiguous and not a.flags.writeable
        for a in actual
    )
    with pytest.raises(ValueError, match="missing"):
        compute_targets(batch, bootstrap_values=bootstrap)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"gamma": True},
        {"gamma": -1},
        {"gamma": 1.1},
        {"gamma": np.nan},
        {"gae_lambda": np.inf},
        {"discount": "turn"},
        {"values": [0]},
        {"values": [np.nan] * 4},
        {"values": [True] * 4},
        {"bootstrap_values": np.zeros((1, 3))},
    ),
)
def test_invalid_target_configuration(kwargs):
    with pytest.raises(ValueError):
        compute_targets(fixture(), **kwargs)


@pytest.mark.parametrize(
    "column,bad",
    (
        ("next_rows", [1, 3, -1, -1]),
        ("next_rows", [0, 3, -1, -1]),
        ("next_rows", [9, 3, -1, -1]),
        ("next_rows", [2, 3, -2, -1]),
        ("next_rows", [2.0, 3.0, -1.0, -1.0]),
        ("elapsed_actions", [0, 2, 1, 1]),
        ("players", [0, 1, 2, 1]),
        ("episode_indices", [0, 0, 1, 0]),
        ("terminated", [False, False, False, True]),
        ("truncated", [False, False, True, False]),
    ),
)
def test_invalid_links_or_closure_are_rejected(column, bad):
    batch = fixture()
    setattr(batch, column, np.asarray(bad))
    with pytest.raises(ValueError):
        compute_targets(batch)


def test_empty_targets_are_supported():
    batch = fixture()
    for name in (
        "values",
        "rewards",
        "next_rows",
        "elapsed_actions",
        "episode_indices",
        "players",
        "terminated",
        "truncated",
    ):
        setattr(batch, name, getattr(batch, name)[:0])
    assert compute_targets(batch).advantages.shape == (0,)


def test_duplicate_successors_or_absent_seats_are_rejected():
    batch = fixture()
    batch.players.fill(0)
    batch.next_rows[1] = 2
    with pytest.raises(ValueError, match="multiple"):
        compute_targets(batch)
    batch = fixture()
    batch.players[[1, 3]] = 3
    with pytest.raises(ValueError, match="absent"):
        compute_targets(batch)
    with pytest.raises(ValueError, match="finite"):
        compute_targets(fixture(), values=np.full(4, 1e300))


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("discount", ("decision", "engine"))
def test_large_vector_wave_path_matches_independent_player_trajectories(
    players, discount
):
    games, length = 32, 64
    n = games * length * players
    rows = np.arange(n)
    closing = rows % (length * players) >= (length - 1) * players
    links = (rows + players).astype(np.int32)
    links[closing] = -1
    rng = np.random.default_rng(51)
    batch = Fixture(
        values=rng.normal(0, 0.3, n).astype(np.float32),
        rewards=(closing & (rows % players == 0)).astype(np.float32),
        next_rows=links,
        elapsed_actions=rng.integers(1, 5, n, dtype=np.int32),
        episode_indices=(rows // (length * players)).astype(np.int32),
        players=(rows % players).astype(np.int8),
        terminated=closing,
        truncated=np.zeros(n, dtype=bool),
        episodes=tuple(SimpleNamespace(players=players) for _ in range(games)),
    )
    gamma, lam = 0.97, 0.83
    expected = np.empty(n)
    for game in range(games):
        for seat in range(players):
            trace = 0.0
            indices = np.arange(
                game * length * players + seat, (game + 1) * length * players, players
            )
            for position in range(length - 1, -1, -1):
                row = indices[position]
                power = int(batch.elapsed_actions[row]) if discount == "engine" else 1
                next_value = (
                    float(batch.values[indices[position + 1]])
                    if position + 1 < length
                    else 0.0
                )
                delta = (
                    float(batch.rewards[row])
                    + gamma**power * next_value
                    - float(batch.values[row])
                )
                trace = delta + gamma**power * lam**power * trace
                expected[row] = trace
    actual = compute_targets(batch, gamma=gamma, gae_lambda=lam, discount=discount)
    np.testing.assert_allclose(actual.advantages, expected, rtol=2e-6, atol=2e-7)

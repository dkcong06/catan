"""Independent public-feature oracles, observation isolation and engine replay."""

from random import Random

import numpy as np
import pytest

from catan.board import GEOMETRY, Resource
from catan.engine import _produce, _trade_ratios, resolve_seven_inplace
from catan.state import Phase, new_game
from rl.collector import ReferencePolicy, collect_rollouts
from rl.economics import (
    COSTS,
    ECONOMIC_DIM,
    ECONOMIC_OBS_DIM,
    ROLLS,
    EconomicEncoder,
    EconomicEnvFactory,
    EconomicFeatures,
)
from rl.encoding import OBS_DIM, FlatEncoder


def oracle(state, observer):
    production = np.zeros((4, 5, 10), dtype=np.float32)
    sites = np.zeros((54, 5), dtype=np.float32)
    for tile, vertices in enumerate(GEOMETRY.tile_vertices):
        resource, number = state.board.resources[tile], state.board.numbers[tile]
        if resource == Resource.DESERT or tile == state.robber_tile:
            continue
        for vertex in vertices:
            sites[vertex, resource] += (6 - abs(7 - number)) / 36
            owner = int(state.vertex_owner[vertex])
            if owner != -1:
                production[
                    (owner - observer) % state.n_players, resource, ROLLS.index(number)
                ] += int(state.vertex_building[vertex]) / 19
    costs = ((1, 0, 0, 1, 0), (1, 1, 1, 1, 0), (0, 0, 2, 0, 3), (0, 1, 1, 0, 1))
    gaps = [
        max(0, wanted - int(state.hands[observer, r])) / 3
        for cost in costs
        for r, wanted in enumerate(cost)
    ]
    ratios = [4] * 5
    ports = []
    for v, owner in enumerate(state.vertex_owner):
        slot = GEOMETRY.vertex_ports[v]
        if owner == observer and slot != -1:
            ports.append(int(state.board.ports[slot]))
    for r in range(5):
        ratios[r] = 2 if r in ports else (3 if 5 in ports else 4)
    return np.concatenate(
        (production.ravel(), gaps, sites.ravel(), np.asarray(ratios) / 4)
    ).astype(np.float32)


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("seed", (670000001, 670000002, 670000003))
def test_all_rolls_cities_robbers_and_perspectives_match_engine(seed, players):
    state = new_game(seed, players)
    rng = Random(seed)
    for vertex in rng.sample(range(54), 12):
        state.vertex_owner[vertex] = rng.randrange(players)
        state.vertex_building[vertex] = rng.choice((1, 2))
    features = EconomicFeatures(state.board)
    before = state.copy()
    for robber in range(19):
        state.robber_tile = robber
        for observer in range(players):
            actual = features.encode(state, observer)
            np.testing.assert_allclose(actual, oracle(state, observer), atol=3e-8)
            for roll_index, roll in enumerate(ROLLS):
                paid = state.copy()
                paid.hands.fill(0)
                paid.bank.fill(1000)  # Oracle fixture removes finite-bank rationing.
                _produce(paid, roll)
                for relative in range(players):
                    seat = (observer + relative) % players
                    np.testing.assert_allclose(
                        actual[:200].reshape(4, 5, 10)[relative, :, roll_index] * 19,
                        paid.hands[seat],
                        atol=1e-5,
                    )
    state.robber_tile = before.robber_tile
    assert state == before


def test_costs_ports_private_information_and_rng_are_separate():
    state = new_game(670000004, 4)
    features = EconomicFeatures(state.board)
    for i, (a, _) in enumerate(GEOMETRY.port_vertices):
        state.vertex_owner[a] = 2
        state.vertex_building[a] = 1
        state.current_player = 2
        actual = features.encode(state, 2)
        np.testing.assert_array_equal(actual[-5:] * 4, _trade_ratios(state))
        np.testing.assert_allclose(actual, oracle(state, 2), atol=3e-8)
        if i == 0:
            assert (actual[-5:] > 0).all()
    state.hands[2] = (1, 0, 2, 1, 2)
    gaps = features.encode(state, 2)[200:220].reshape(4, 5) * 3
    np.testing.assert_allclose(gaps, np.maximum(COSTS - state.hands[2], 0), atol=1e-6)
    expected = features.encode(state, 2)
    state.hands[[0, 1, 3]] = ((4, 1, 1, 0, 0), (0, 6, 0, 0, 0), (0, 0, 0, 2, 4))
    state.development[[0, 1, 3]] = 0
    state.rng.random()
    np.testing.assert_array_equal(features.encode(state, 2), expected)
    state.bank.fill(0)
    np.testing.assert_array_equal(features.encode(state, 2), expected)
    assert isinstance(features._tile_columns, tuple)
    for lookup in (features._site_yields, COSTS):
        with pytest.raises(ValueError):
            lookup.setflags(write=True)


def test_manual_discard_terminal_buffers_and_rejections_are_isolated():
    state = new_game(670000005, 3)
    state.current_player = 1
    state.hands[0] = (4, 4, 2, 0, 0)
    state.phase = Phase.ROLL
    resolve_seven_inplace(state)
    encoder = EconomicEncoder(state.board)
    saved = state.copy()
    out = np.full(ECONOMIC_OBS_DIM, np.nan, dtype=np.float32)
    assert encoder.encode(state, out=out) is out
    np.testing.assert_array_equal(out[:OBS_DIM], FlatEncoder(state.board).encode(state))
    np.testing.assert_array_equal(out[OBS_DIM:], oracle(state, 0))
    assert state == saved
    for bad in (np.empty(ECONOMIC_OBS_DIM), out[::2], out.reshape(1, -1)):
        with pytest.raises(ValueError):
            encoder.encode(state, out=bad)
    with pytest.raises(ValueError, match="different board"):
        encoder.encode(new_game(670000006, 3), out=out)
    with pytest.raises(ValueError, match="outside"):
        encoder.encode(state, 3)
    backing = np.empty(ECONOMIC_OBS_DIM * 2, dtype=np.int16)
    state.hands = backing[:15].reshape(3, 5)
    with pytest.raises(ValueError, match="share memory"):
        encoder.encode(state, out=backing.view(np.float32))
    state.hands = saved.hands.copy()
    state.phase = Phase.GAME_OVER
    np.testing.assert_allclose(
        encoder.encode(state)[OBS_DIM:], oracle(state, 1), atol=3e-8
    )
    assert EconomicFeatures(state.board).encode(state).shape == (ECONOMIC_DIM,)


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("enabled", (False, True))
def test_complete_games_keep_original_actions_rng_and_observation_prefix(
    players, enabled
):
    args = dict(n_players=players, policy_seed=19, validate=True)
    base = collect_rollouts(ReferencePolicy("greedy"), (670000007,), **args)
    new = collect_rollouts(
        ReferencePolicy("greedy"),
        (670000007,),
        env_factory=EconomicEnvFactory(enabled),
        **args,
    )
    assert base.episodes == new.episodes
    assert new.observation_schema == 4
    np.testing.assert_array_equal(base.observations, new.observations[:, :OBS_DIM])
    np.testing.assert_array_equal(base.actions, new.actions)
    np.testing.assert_array_equal(base.discards.counts, new.discards.counts)
    if not enabled:
        assert not new.observations[:, OBS_DIM:].any()


def test_factory_rejects_ambiguous_flags():
    with pytest.raises(ValueError):
        EconomicEnvFactory(enabled=1)
    for invalid in (-1, 4, True):
        with pytest.raises(ValueError):
            EconomicEnvFactory(feature_player=invalid)
    env = EconomicEnvFactory(feature_player=3)(max_actions=100)
    env.reset(670000008, n_players=4)
    before = env.snapshot()
    with pytest.raises(ValueError, match="outside"):
        env.reset(670000009, n_players=3)
    assert env.snapshot() == before


def test_explicit_feature_seat_gating_and_terminal_observers():
    state = new_game(670000010, 4)
    full = EconomicEncoder(state.board)
    scoped = EconomicEncoder(state.board, feature_player=2)
    for phase in (Phase.SETUP_SETTLEMENT, Phase.GAME_OVER):
        state.phase = phase
        for observer in range(4):
            expected = full.encode(state, observer)
            actual = scoped.encode(state, observer)
            np.testing.assert_array_equal(actual[:OBS_DIM], expected[:OBS_DIM])
            if observer == 2:
                np.testing.assert_array_equal(actual, expected)
            else:
                assert not actual[OBS_DIM:].any()

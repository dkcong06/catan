from random import Random

import numpy as np
import pytest

from catan.actions import DiscardAction, KnightAction, RoadBuildingAction
from catan.cards import DevelopmentCard as Card
from catan.engine import apply_action, apply_action_inplace, resolve_seven_inplace
from catan.players.random_agent import choose_action
from catan.state import Phase, _ARRAY_FIELDS, new_game
from catan.validation import assert_invariants
from rl.actions import action_index, action_mask, discard_count_mask
from rl.encoding import FlatEncoder, OBS_DIM, FEATURE_SLICES
from rl.env import CatanEnv
from test_development import build, fund, grant, main_state, own_roads, path


def fixture_env(state, cap=5000):
    env = CatanEnv(max_actions=cap)
    env._state = state.copy()
    env._encoder = FlatEncoder(env._state.board)
    env._zeros = (0.0,) * state.n_players
    return env


@pytest.mark.parametrize(
    "operation", ("observe", "snapshot", "decision", "step", "terminated", "truncated")
)
def test_reset_required(operation):
    env = CatanEnv()
    with pytest.raises(ValueError, match="reset"):
        if operation == "step":
            env.step(0, player=0)
        elif operation in ("terminated", "truncated"):
            getattr(env, operation)
        else:
            getattr(env, operation)()


@pytest.mark.parametrize("value", (0, -1, True, 2.5, "10"))
def test_invalid_cap(value):
    with pytest.raises((TypeError, ValueError)):
        CatanEnv(max_actions=value)


@pytest.mark.parametrize("players", (3, 4))
def test_seeded_reset_observation_outputs_and_snapshot_are_independent(players):
    env = CatanEnv()
    request = env.reset(42, n_players=players)
    before = env.snapshot()
    expected = FlatEncoder(before.board).encode(before)
    assert request.player == 0 and request.phase == Phase.SETUP_SETTLEMENT
    np.testing.assert_array_equal(request.observation, expected)
    np.testing.assert_array_equal(request.mask, action_mask(before))
    assert request.observation.shape == (OBS_DIM,)
    request.observation.fill(99)
    request.mask.fill(False)
    before.hands.fill(12)
    before.rng.random()
    assert env.snapshot() == new_game(42, players)
    np.testing.assert_array_equal(env.observe(), expected)
    output = np.empty(OBS_DIM, dtype=np.float32)
    assert env.observe(out=output) is output
    np.testing.assert_array_equal(output, expected)
    env.step(int(np.flatnonzero(env.decision().mask)[0]), player=0)
    old = env.snapshot()
    for kwargs in ({"seed": True}, {"n_players": 2}):
        with pytest.raises((TypeError, ValueError)):
            env.reset(**kwargs)
        assert env.snapshot() == old and env.actions == 1
    env.reset(42, n_players=players)
    assert env.actions == 0 and env.snapshot() == new_game(42, players)


@pytest.mark.parametrize(
    "action,player",
    ((72, 1), (True, 0), (-1, 0), (219, 0), (254, 0), (72, True), ((0, 0, 0, 0, 0), 0)),
)
def test_rejection_preserves_every_field_rng_identity_and_action_count(action, player):
    env = CatanEnv()
    env.reset(4)
    original = env.snapshot()
    identities = tuple(id(getattr(env._state, name)) for name in _ARRAY_FIELDS) + (
        id(env._state.rng),
    )
    with pytest.raises((ValueError, TypeError)):
        env.step(action, player=player)
    assert env.actions == 0 and env.snapshot() == original
    assert identities == tuple(
        id(getattr(env._state, name)) for name in _ARRAY_FIELDS
    ) + (id(env._state.rng),)


def test_manual_discard_actor_masks_and_obligations_are_snapshot_choices():
    state = main_state()
    state.phase, state.last_roll = Phase.ROLL, None
    fund(state, 2, (4, 3, 2, 0, 0))
    fund(state, 3, (3, 3, 2, 0, 0))
    resolve_seven_inplace(state)
    env = fixture_env(state)
    request = env.decision()
    assert request.player == 2 and env.snapshot().current_player == 0
    assert request.mask is None and request.discard_required == 4
    assert request.discard_hand == (4, 3, 2, 0, 0)
    for prefix in ((), (2,), (2, 1), (2, 1, 1), (2, 1, 1, 0)):
        np.testing.assert_array_equal(
            request.discard_mask(prefix), discard_count_mask(state, prefix)
        )
    for counts, player in (
        ((5, 0, 0, 0, 0), 2),
        ((2, 1, 0, 0, 0), 2),
        ((2, 1, 1, 0, 0), 0),
        (72, 2),
    ):
        before = env.snapshot()
        with pytest.raises((ValueError, TypeError)):
            env.step(counts, player=player)
        assert env.snapshot() == before and env.actions == 0
    first = env.step((2, 1, 1, 0, 0), player=2)
    assert first.player == 2 and first.next_player == 3 and first.rewards == (0.0,) * 4
    assert env.actions == 1 and env.snapshot().current_player == 0
    np.testing.assert_array_equal(
        request.discard_mask((2,)), discard_count_mask(state, (2,))
    )
    second = env.step((2, 1, 1, 0, 0), player=3)
    assert second.next_player == 0 and env.decision().phase == Phase.ROBBER_MOVE
    assert env.actions == 2
    with pytest.raises(ValueError, match="not a discard"):
        env.decision().discard_mask()


@pytest.mark.parametrize(
    "card,action",
    ((Card.KNIGHT, KnightAction()), (Card.ROAD_BUILDING, RoadBuildingAction())),
)
def test_pre_roll_multichoice_card_does_not_skip_roll(card, action):
    state = main_state()
    state.phase, state.last_roll = Phase.ROLL, None
    build(state, 0)
    grant(state, card)
    env = fixture_env(state)
    env.step(action_index(state, action), player=0)
    while env._state.phase != Phase.ROLL:
        request = env.decision()
        env.step(int(np.flatnonzero(request.mask)[0]), player=request.player)
    assert env.snapshot().last_roll is None
    assert env.decision().mask[220]
    assert_invariants(env.snapshot())


def test_truncation_preserves_nonterminal_state_perspectives_and_hidden_vp():
    state = main_state()
    grant(state, Card.VICTORY_POINT, player=1)
    env = fixture_env(state, cap=1)
    result = env.step(219, player=0)
    assert result.truncated and not result.terminated and result.winner is None
    assert result.next_player == 1 and result.rewards == (0.0,) * 4
    assert env.decision() is None and env.snapshot().phase == Phase.ROLL
    for player in range(4):
        np.testing.assert_array_equal(
            env.observe(player), FlatEncoder(state.board).encode(env._state, player)
        )
    player_rows = env.observe(0)[FEATURE_SLICES["players"]].reshape(4, 9)
    assert player_rows[1, 1] == 0  # Opponent VP remains private at external cap.
    before = env.snapshot()
    with pytest.raises(ValueError, match="closed"):
        env.step(220, player=1)
    assert env.snapshot() == before and env.actions == 1


def test_actual_win_at_cap_takes_precedence_and_rewards_winner_not_last_actor():
    state = main_state()
    vertices = []
    for v in range(54):
        if not set(state.board.geometry.vertex_neighbors[v]) & set(vertices):
            vertices.append(v)
        if len(vertices) == 4:
            break
    for v in vertices:
        build(state, v, 1)
    grant(state, Card.VICTORY_POINT, 4, player=1)
    own_roads(state, path(state, 5), 1)
    assert_invariants(state)
    env = fixture_env(state, cap=1)
    result = env.step(219, player=0)
    assert result.player == 0 and result.winner == 1 and result.next_player is None
    assert result.terminated and not result.truncated
    assert result.rewards == (0.0, 1.0, 0.0, 0.0)
    assert env.decision() is None and env.terminated and not env.truncated
    assert env.snapshot().phase == Phase.GAME_OVER
    with pytest.raises(ValueError, match="closed"):
        env.step(220, player=1)
    assert env.actions == 1


@pytest.mark.parametrize("players", (3, 4))
def test_complete_engine_replay_every_phase_fields_rng_and_owned_identities(players):
    covered = set()
    for seed in range(6):
        direct = new_game(seed, players)
        env = CatanEnv(max_actions=5000)
        request = env.reset(seed, n_players=players)
        rng = Random(seed)
        encoder = FlatEncoder(direct.board)
        ids = tuple(id(getattr(env._state, name)) for name in _ARRAY_FIELDS) + (
            id(env._state.rng),
        )
        for step in range(5000):
            covered.add(request.phase)
            np.testing.assert_array_equal(request.observation, encoder.encode(direct))
            action = choose_action(direct, rng)
            if isinstance(action, DiscardAction):
                choice = action.resources
            else:
                choice = action_index(direct, action)
                assert request.mask[choice]
            functional = apply_action(direct, action)
            result = env.step(choice, player=request.player)
            apply_action_inplace(direct, action)
            assert direct == functional == env._state
            assert ids == tuple(
                id(getattr(env._state, name)) for name in _ARRAY_FIELDS
            ) + (id(env._state.rng),)
            assert env.actions == step + 1
            assert_invariants(env._state)
            if result.terminated or result.truncated:
                assert result.terminated and result.rewards[result.winner] == 1
                assert sum(result.rewards) == 1
                break
            assert result.rewards == (0.0,) * players
            request = env.decision()
        covered.add(env._state.phase)
    assert covered == set(Phase)

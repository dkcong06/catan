from dataclasses import fields, replace
from random import Random, getstate

import numpy as np
import pytest

from catan.actions import DiscardAction
from catan.engine import apply_action_inplace, is_terminal, resolve_seven_inplace
from catan.players.random_agent import choose_action, play_random_game
from catan.state import GameState, Phase, _ARRAY_FIELDS, new_game
from rl.encoding import (
    FEATURE_SHAPES,
    FEATURE_SLICES,
    OBS_DIM,
    FlatEncoder,
    decision_player,
    encode_flat,
)


def group(observation, name):
    return observation[FEATURE_SLICES[name]].reshape(FEATURE_SHAPES[name])


def oracle(state, player=None):
    # Scalar implementation, independent of encoder tables, preparation and gathers.
    n = state.n_players
    actor = None if state.phase == Phase.GAME_OVER else state.current_player
    if state.phase == Phase.DISCARD:
        actor = next(
            (state.current_player + i) % n
            for i in range(n)
            if state.discard_pending[(state.current_player + i) % n] > 0
        )
    if player is None:
        player = state.current_player if actor is None else actor
    flat = []
    for tile, resource in enumerate(state.board.resources):
        number = state.board.numbers[tile]
        flat.extend(float(resource == r) for r in range(6))
        flat.extend(
            (
                0 if number == 0 else (6 - abs(7 - number)) / 36,
                float(number > 7),
                float(tile == state.robber_tile),
            )
        )
    for vertex, owner in enumerate(state.vertex_owner):
        relative = 4 if owner == -1 else (int(owner) - player) % n
        flat.extend(float(relative == i) for i in range(5))
        flat.extend(float(state.vertex_building[vertex] == b) for b in range(3))
        port = state.board.geometry.vertex_ports[vertex]
        kind = 6 if port == -1 else int(state.board.ports[port])
        flat.extend(float(kind == k) for k in range(7))
        flat.append(float(state.setup_vertex == vertex))
    for owner in state.edge_owner:
        relative = 4 if owner == -1 else (int(owner) - player) % n
        flat.extend(float(relative == i) for i in range(5))
    flat.extend(int(count) / 19 for count in state.hands[player])
    order = [(player + i) % n for i in range(n)]
    flat.extend(sum(map(int, state.hands[p])) / 95 for p in order[1:])
    flat.extend([0] * (4 - n))
    for i in range(4):
        if i >= n:
            flat.extend([0] * 9)
        else:
            p = order[i]
            flat.extend(
                (
                    1,
                    (
                        int(state.vp[p])
                        if p == player or state.phase == Phase.GAME_OVER
                        else int(state.vp[p]) - int(state.development[p, 1])
                    )
                    / 10,
                    int(state.pieces_left[p, 0]) / 5,
                    int(state.pieces_left[p, 1]) / 4,
                    int(state.pieces_left[p, 2]) / 15,
                    int(state.discard_pending[p]) / 47,
                    sum(map(int, state.development[p])) / 25,
                    int(state.knights_played[p]) / 14,
                    int(state.road_lengths[p]) / 15,
                )
            )
    flat.extend(int(count) / 19 for count in state.bank)
    flat.extend(float(state.phase == p) for p in Phase)
    flat.extend(float(state.last_roll == total) for total in (None, *range(2, 13)))
    flat.extend(float((state.current_player - player) % n == i) for i in range(4))
    flat.extend(
        float(actor is not None and (actor - player) % n == i) for i in range(4)
    )
    flat.append(state.setup_index / (2 * n))
    flat.extend(
        int(count) / supply
        for count, supply in zip(state.development[player], (14, 5, 2, 2, 2))
    )
    flat.extend(
        int(count) / supply
        for count, supply in zip(state.development_new[player], (14, 5, 2, 2, 2))
    )
    flat.append(sum(map(int, state.development_deck)) / 25)
    flat.extend(
        int(count) / supply
        for count, supply in zip(state.development_played, (14, 5, 2, 2, 2))
    )
    for name in ("longest_road_owner", "largest_army_owner"):
        owner = getattr(state, name)
        relative = 4 if owner == -1 else (owner - player) % n
        flat.extend(float(relative == i) for i in range(5))
    flat.extend((float(state.development_used), state.free_roads_remaining / 2))
    flat.extend(
        float(state.resume_phase == phase) for phase in (None, Phase.ROLL, Phase.MAIN)
    )
    return np.asarray(flat, dtype=np.float32)


def rotate_seats(state, offset):
    other = state.copy()
    for name in (
        "hands",
        "vp",
        "pieces_left",
        "discard_pending",
        "development",
        "development_new",
        "knights_played",
        "road_lengths",
    ):
        getattr(other, name)[:] = np.roll(getattr(state, name), offset, axis=0)
    for name in ("vertex_owner", "edge_owner"):
        source = getattr(state, name)
        getattr(other, name)[:] = np.where(
            source == -1, -1, (source + offset) % state.n_players
        )
    for name in ("longest_road_owner", "largest_army_owner"):
        owner = getattr(state, name)
        setattr(other, name, -1 if owner == -1 else (owner + offset) % state.n_players)
    other.current_player = (state.current_player + offset) % state.n_players
    return other


@pytest.mark.parametrize("players", (3, 4))
def test_all_decision_phases_and_full_seeded_game_match_scalar_oracle(players):
    global_rng = getstate()
    state = new_game(0, players)
    encoder = FlatEncoder(state.board)
    policy = Random(0)
    out = np.full(OBS_DIM, np.nan, dtype=np.float32)
    coverage = set()
    for _ in range(10000):
        coverage.add(state.phase)
        before = state.copy()
        identities = {f.name: id(getattr(state, f.name)) for f in fields(GameState)}
        actual = encoder.encode(state, out=out)
        assert actual is out and actual.shape == (OBS_DIM,)
        assert actual.dtype == np.float32 and actual.flags.c_contiguous
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, oracle(state), rtol=1e-6, atol=1e-7)
        assert state == before
        assert identities == {
            f.name: id(getattr(state, f.name)) for f in fields(GameState)
        }
        if is_terminal(state):
            break
        apply_action_inplace(state, choose_action(state, policy))
    assert is_terminal(state)
    assert coverage == set(Phase)
    assert getstate() == global_rng


@pytest.mark.parametrize("players", (3, 4))
def test_fixed_schema_and_rotated_seats_across_varied_reachable_states(players):
    checks = 0
    for seed in range(30):
        state = new_game(seed, players)
        encoder = FlatEncoder(state.board)
        policy = Random(seed)
        for step in range(240):
            if step % 40 == 0:
                for observer in range(players):
                    actual = encoder.encode(state, observer)
                    np.testing.assert_allclose(
                        actual, oracle(state, observer), rtol=1e-6, atol=1e-7
                    )
                    assert actual.shape == (OBS_DIM,)
                    for offset in range(players):
                        rotated = rotate_seats(state, offset)
                        np.testing.assert_array_equal(
                            actual,
                            encoder.encode(rotated, (observer + offset) % players),
                        )
                        if state.phase != Phase.GAME_OVER:
                            np.testing.assert_array_equal(
                                encoder.encode(state), encoder.encode(rotated)
                            )
                    checks += 1
            if is_terminal(state):
                break
            apply_action_inplace(state, choose_action(state, policy))
    assert checks >= 500


def test_three_player_padding_distinguishes_absent_from_empty_real_players():
    actual = encode_flat(new_game(0, 3))
    players = group(actual, "players")
    assert players[:3, 0].tolist() == [1, 1, 1]
    assert not players[3].any()
    assert not group(actual, "opponent_hand_sizes").any()
    assert not group(actual, "vertices")[:, 3].any()
    assert not group(actual, "edges")[:, 3].any()
    assert group(actual, "vertices")[:, 4].tolist() == [1] * 54
    assert group(actual, "edges")[:, 4].tolist() == [1] * 72


def test_opponent_resource_composition_and_rng_are_hidden():
    state = new_game(4)
    state.hands[1] = [5, 1, 0, 0, 0]
    state.hands[2] = [1, 5, 0, 0, 0]
    state.bank -= state.hands.sum(axis=0).astype(np.int16)
    other = state.copy()
    other.hands[1] = state.hands[2]
    other.hands[2] = state.hands[1]
    other.rng.seed(100)
    other.rng.gauss(0, 1)
    encoder = FlatEncoder(state.board)
    np.testing.assert_array_equal(encoder.encode(state, 0), encoder.encode(other, 0))
    # The same swap becomes visible to either owner, whose own cards are public to them.
    assert not np.array_equal(encoder.encode(state, 1), encoder.encode(other, 1))
    state.validate()
    other.validate()


def test_discards_orient_to_the_chooser_and_preserve_turn_owner():
    state = new_game(42, 3)
    policy = Random(42)
    while state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply_action_inplace(state, choose_action(state, policy))
    state.current_player = 1
    state.hands[:] = [[3, 3, 3, 0, 0], [0, 1, 1, 1, 1], [2, 3, 3, 0, 0]]
    state.bank[:] = 19 - state.hands.sum(axis=0)
    resolve_seven_inplace(state)
    encoder = FlatEncoder(state.board)
    out = np.empty(OBS_DIM, dtype=np.float32)
    for player in (2, 0):
        assert decision_player(state) == player and state.current_player == 1
        actual = encoder.encode(state, out=out)
        np.testing.assert_allclose(actual, oracle(state), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(group(actual, "my_hand"), state.hands[player] / 19)
        assert group(actual, "acting_player").tolist() == [1, 0, 0, 0]
        assert group(actual, "turn_owner")[(1 - player) % 3] == 1
        apply_action_inplace(state, DiscardAction(player, (1, 1, 2, 0, 0)))
    assert state.phase == Phase.ROBBER_MOVE and decision_player(state) == 1
    np.testing.assert_allclose(
        encoder.encode(state, out=out), oracle(state), rtol=1e-6, atol=1e-7
    )


def test_terminal_default_and_explicit_value_perspectives():
    state = play_random_game(0)
    encoder = FlatEncoder(state.board)
    assert decision_player(state) is None
    for player in range(4):
        actual = encoder.encode(state, player)
        assert not group(actual, "acting_player").any()
        assert group(actual, "turn_owner")[(state.current_player - player) % 4] == 1
        np.testing.assert_allclose(actual, oracle(state, player), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(
        encoder.encode(state), encoder.encode(state, state.current_player)
    )


def test_probability_and_high_number_flag_preserve_token_identity():
    state = new_game(2)
    changed = replace(
        state.board,
        numbers=tuple(8 if n == 6 else 6 if n == 8 else n for n in state.board.numbers),
    )
    other = GameState(changed, rng=Random(2))
    first = group(encode_flat(state), "tiles")
    second = group(encode_flat(other), "tiles")
    np.testing.assert_array_equal(first[:, 6], second[:, 6])
    assert not np.array_equal(first[:, 7], second[:, 7])
    for tile, number in enumerate(state.board.numbers):
        if number:
            pips = round(float(first[tile, 6]) * 36)
            assert (13 - pips if first[tile, 7] else pips + 1) == number
        else:
            assert first[tile, 6] == first[tile, 7] == 0


def test_port_features_show_unowned_locations_and_setup_anchor_is_cleared():
    state = new_game(0)
    encoder = FlatEncoder(state.board)
    policy = Random(0)
    out = np.empty(OBS_DIM, dtype=np.float32)
    ports = group(encoder.encode(state), "vertices")[:, 8:15].copy()
    for vertex, slot in enumerate(state.board.geometry.vertex_ports):
        kind = 6 if slot == -1 else int(state.board.ports[slot])
        assert ports[vertex, kind] == 1 and ports[vertex].sum() == 1
    apply_action_inplace(state, choose_action(state, policy))
    actual = encoder.encode(state, out=out)
    assert group(actual, "vertices")[:, 15].sum() == 1
    assert group(actual, "vertices")[state.setup_vertex, 15] == 1
    apply_action_inplace(state, choose_action(state, policy))
    actual = encoder.encode(state, out=out)
    assert not group(actual, "vertices")[:, 15].any()
    np.testing.assert_array_equal(ports, group(actual, "vertices")[:, 8:15])


def test_normalization_keeps_exact_sensible_maxima_without_clipping():
    state = new_game(0)
    state.hands[0] = 19
    state.bank[:] = 0
    actual = encode_flat(state)
    np.testing.assert_array_equal(group(actual, "my_hand"), np.ones(5))
    assert not group(actual, "bank").any()
    assert group(actual, "players")[0, 2:5].tolist() == [1, 1, 1]
    state.validate()


def test_prepared_features_are_immutable_and_outputs_are_independent():
    state = new_game(0)
    encoder = FlatEncoder(state.board)
    before = encoder._base.copy()
    with pytest.raises(ValueError):
        encoder._base.setflags(write=True)
    first, second = encoder.encode(state), encoder.encode(state)
    assert not np.shares_memory(first, second) and not np.shares_memory(
        first, encoder._base
    )
    first[:] = 99
    np.testing.assert_array_equal(second, encode_flat(state))
    np.testing.assert_array_equal(encoder._base, before)
    with pytest.raises(TypeError):
        FEATURE_SLICES["tiles"] = slice(1)
    with pytest.raises(TypeError):
        FEATURE_SHAPES["tiles"] = (1,)


@pytest.mark.parametrize("name", _ARRAY_FIELDS)
def test_caller_output_must_not_alias_any_state_array(name):
    state = new_game(0)
    encoder = FlatEncoder(state.board)
    out = np.zeros(OBS_DIM, dtype=np.float32)
    original = getattr(state, name)
    alias = out.view(original.dtype)[: original.size].reshape(original.shape)
    alias[:] = original
    setattr(state, name, alias)
    before = state.copy()
    saved = out.copy()
    with pytest.raises(ValueError, match="share memory"):
        encoder.encode(state, out=out)
    assert state == before
    np.testing.assert_array_equal(out, saved)


@pytest.mark.parametrize(
    "kind", ("list", "shape", "dtype", "rank", "strided", "readonly")
)
def test_invalid_output_is_rejected_before_writing(kind):
    state = new_game(0)
    if kind == "list":
        out = [0] * OBS_DIM
    elif kind == "shape":
        out = np.zeros(OBS_DIM + 1, dtype=np.float32)
    elif kind == "dtype":
        out = np.zeros(OBS_DIM, dtype=np.float64)
    elif kind == "rank":
        out = np.zeros((1, OBS_DIM), dtype=np.float32)
    elif kind == "strided":
        out = np.zeros(OBS_DIM * 2, dtype=np.float32)[::2]
    else:
        out = np.zeros(OBS_DIM, dtype=np.float32)
        out.flags.writeable = False
    before = state.copy()
    saved = np.array(out)
    with pytest.raises(ValueError, match="out"):
        FlatEncoder(state.board).encode(state, out=out)
    assert state == before
    np.testing.assert_array_equal(out, saved)


@pytest.mark.parametrize("player", (-1, 3, True, np.bool_(False), 1.5, "0"))
def test_invalid_perspectives_are_rejected(player):
    state = new_game(0, 3)
    out = np.full(OBS_DIM, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="player"):
        FlatEncoder(state.board).encode(state, player, out=out)
    assert np.isnan(out).all()


def test_new_board_requires_preparation_and_no_validation_enters_rollout_path(
    monkeypatch,
):
    state = new_game(0)
    encoder = FlatEncoder(state.board)
    out = np.full(OBS_DIM, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="different board"):
        encoder.encode(new_game(0), out=out)
    assert np.isnan(out).all()
    with pytest.raises(TypeError, match="Board"):
        FlatEncoder(None)

    def fail(_):
        raise AssertionError("full validation entered observation path")

    monkeypatch.setattr(GameState, "validate", fail)
    np.testing.assert_array_equal(encoder.encode(state), encode_flat(state))
    assert encoder.encode(state, np.int64(0)).shape == (OBS_DIM,)

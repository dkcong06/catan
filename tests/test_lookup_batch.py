from random import Random

import numpy as np
import pytest

from catan.actions import DiscardAction
from catan.engine import (
    apply_action_inplace,
    is_terminal,
    legal_actions,
    resolve_seven_inplace,
)
from catan.players.random_agent import choose_action
from catan.state import Phase, _ARRAY_FIELDS, new_game
from rl.actions import (
    ACTION_DIM,
    action_index,
    action_mask,
    discard_count_mask,
    indexed_action,
)
from rl.encoding import OBS_DIM, FlatEncoder, encode_batch


def pool():
    result = []
    for seed in range(12):
        state = new_game(seed, 3 + seed % 2)
        encoder = FlatEncoder(state.board)
        policy = Random(seed)
        for step in range(1200):
            if step < 16 or step % 17 == 0 or state.phase == Phase.ROAD_BUILDING:
                result.append((state.copy(), encoder))
            if is_terminal(state):
                break
            apply_action_inplace(state, choose_action(state, policy))
    return result


def test_direct_masks_match_all_legal_actions_and_roundtrip_every_reached_choice():
    phases = set()
    for state, _ in pool():
        if state.phase == Phase.DISCARD:
            continue
        phases.add(state.phase)
        before = state.copy()
        expected = np.zeros(ACTION_DIM, dtype=np.bool_)
        for action in legal_actions(state):
            index = action_index(state, action)
            assert indexed_action(state, index) == action
            expected[index] = True
        mask = np.full(ACTION_DIM, True, dtype=np.bool_)
        assert action_mask(state, out=mask) is mask
        np.testing.assert_array_equal(mask, expected)
        assert state == before
    assert phases >= set(Phase) - {Phase.DISCARD, Phase.GAME_OVER}


@pytest.mark.parametrize("size", (0, 1, 7, 8, 32, 128, 259))
def test_batch_exact_scalar_results_mixed_boards_seats_and_observers(size):
    cases = pool()[:size]
    states = [s for s, _ in cases]
    encoders = [e for _, e in cases]
    before = [s.copy() for s in states]
    for players in (None, [i % s.n_players for i, s in enumerate(states)]):
        out = np.full((size, OBS_DIM), np.nan, dtype=np.float32)
        actual = encode_batch(states, encoders, players, out=out)
        assert actual is out
        expected = (
            np.stack(
                [
                    e.encode(s, None if players is None else players[i])
                    for i, (s, e) in enumerate(cases)
                ]
            )
            if size
            else np.empty((0, OBS_DIM), np.float32)
        )
        np.testing.assert_array_equal(actual, expected)
        assert states == before
    if states:
        old = states[0].copy()
        actual[0, 0] = 123
        assert states[0] == old


def test_batch_bad_context_or_observer_rejects_before_writing():
    states = [new_game(1), new_game(2)]
    encoders = [FlatEncoder(s.board) for s in states]
    out = np.full((2, OBS_DIM), 17, dtype=np.float32)
    for contexts, players in (
        (encoders, [0, 4]),
        (encoders[::-1], None),
        (encoders[:1], None),
        (encoders, [0]),
    ):
        with pytest.raises(ValueError):
            encode_batch(states, contexts, players, out=out)
        assert np.all(out == 17)
    for bad in (out.astype(np.float64), out[:, ::-1], out[:1]):
        with pytest.raises(ValueError):
            encode_batch(states, encoders, out=bad)
    out.flags.writeable = False
    with pytest.raises(ValueError):
        encode_batch(states, encoders, out=out)


def test_outputs_aliasing_state_are_rejected_even_with_valid_shape():
    state = new_game(0)
    old = state.vertex_owner
    backing = np.zeros(OBS_DIM * 2, dtype=np.float32)
    state.vertex_owner = backing.view(np.int8)[:54]
    try:
        with pytest.raises(ValueError, match="share memory"):
            encode_batch(
                [state, state],
                [FlatEncoder(state.board)] * 2,
                out=backing.reshape(2, OBS_DIM),
            )
        with pytest.raises(ValueError, match="share memory"):
            action_mask(state, out=backing.view(np.bool_)[:ACTION_DIM])
    finally:
        state.vertex_owner = old


def test_discard_count_masks_enumerate_exact_small_hand_choices_without_rng():
    state = new_game(1, 3)
    state.phase = Phase.ROLL
    state.hands[0] = [3, 2, 2, 1, 1]
    state.bank -= state.hands[0]
    resolve_seven_inplace(state)
    before = state.copy()

    def enumerate_counts(prefix=()):
        if len(prefix) == 5:
            return [DiscardAction(0, prefix)]
        return [
            action
            for count in np.flatnonzero(discard_count_mask(state, prefix))
            for action in enumerate_counts(prefix + (int(count),))
        ]

    assert enumerate_counts() == list(legal_actions(state))
    assert state == before
    for prefix in ((4,), (-1,), (3, 2), (True,), (0,) * 5):
        with pytest.raises(ValueError):
            discard_count_mask(state, prefix)
    with pytest.raises(ValueError, match="discard_count_mask"):
        action_mask(state)


def test_maximum_hand_discard_masks_remain_twenty_choices_each():
    state = new_game(1)
    state.phase = Phase.ROLL
    state.hands[0] = 19
    state.bank[:] = 0
    resolve_seven_inplace(state)
    prefix = ()
    for _ in range(5):
        mask = discard_count_mask(state, prefix)
        assert mask.shape == (20,) and mask.any()
        prefix += (int(np.flatnonzero(mask)[-1]),)
    assert sum(prefix) == 47
    apply_action_inplace(state, DiscardAction(0, prefix))
    state.validate()


@pytest.mark.parametrize("index", (-1, 254, True, 1.2, "2"))
def test_bad_action_indices_reject(index):
    with pytest.raises(ValueError):
        indexed_action(new_game(0), index)


def test_private_cards_deck_composition_and_freshness_are_hidden_from_opponents():
    state = new_game(4)
    state.development[1, 0] = 1
    state.development_deck[0] -= 1
    other = state.copy()
    other.development[1, 0] = 0
    other.development[1, 1] = 1
    other.development_deck[0] += 1
    other.development_deck[1] -= 1
    other.vp[1] += 1
    encoder = FlatEncoder(state.board)
    np.testing.assert_array_equal(encoder.encode(state, 0), encoder.encode(other, 0))
    assert not np.array_equal(encoder.encode(state, 1), encoder.encode(other, 1))
    state.validate()
    other.validate()
    # Deck is revealed only by size, not the remaining multiset.
    assert OBS_DIM == 1505


def test_no_ml_dependency_or_state_field_is_added_by_lookup_and_encoding():
    state = new_game(1)
    before = state.copy()
    for name in _ARRAY_FIELDS:
        assert not np.shares_memory(getattr(state, name), getattr(before, name))
    encode_batch([state], [FlatEncoder(state.board)])
    action_mask(state)
    assert state == before

from copy import deepcopy
from dataclasses import FrozenInstanceError
from random import Random, getstate

import numpy as np
import pytest

from catan.actions import DiscardAction
from catan.engine import apply_action, resolve_seven
from catan.state import GameState, Phase, new_game


def roll_state(hands, roller=0) -> GameState:
    state = new_game(42, len(hands))
    state.hands[:] = hands
    state.bank[:] -= state.hands.sum(axis=0)
    state.phase = Phase.ROLL
    state.setup_index = len(state.setup_order)
    state.current_player = roller
    state.validate()
    return state


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("total", range(20))
def test_seven_threshold_and_floor_with_no_automatic_card_selection(players, total):
    source = roll_state([[total, 0, 0, 0, 0]] + [[0] * 5] * (players - 1))
    before = source.copy()
    state = resolve_seven(source)
    assert source == before
    assert state.last_roll == 7
    assert state.discard_pending.tolist() == [total // 2 if total > 7 else 0] + [0] * (
        players - 1
    )
    assert state.phase == (Phase.DISCARD if total > 7 else Phase.ROBBER_MOVE)
    assert state.discard_player == (0 if total > 7 else None)
    np.testing.assert_array_equal(state.hands, source.hands)
    np.testing.assert_array_equal(state.bank, source.bank)
    assert state.rng.getstate() == source.rng.getstate()
    state.validate()


@pytest.mark.parametrize("players", (3, 4))
def test_each_pending_player_chooses_exact_resources_then_roller_moves_robber(players):
    hands = [[4, 3, 2, 1, 0], [1, 2, 0, 3, 1], [0, 1, 3, 0, 5], [2, 2, 2, 2, 0]]
    state = resolve_seven(roll_state(hands[:players], roller=2))
    assert state.discard_player == 2
    choices = {2: (0, 0, 0, 0, 4), 3: (0, 0, 2, 2, 0), 0: (4, 0, 0, 1, 0)}
    order = [2, 3, 0] if players == 4 else [2, 0]
    initial_bank = state.bank.copy()
    for player in order:
        assert state.discard_player == player
        before = state.copy()
        following = apply_action(state, DiscardAction(player, choices[player]))
        assert state == before
        np.testing.assert_array_equal(
            following.hands[player], state.hands[player] - choices[player]
        )
        np.testing.assert_array_equal(following.bank, state.bank + choices[player])
        assert following.current_player == 2
        assert following.robber_tile == state.robber_tile
        assert following.rng.getstate() == state.rng.getstate()
        assert following.board is state.board
        following.validate()
        state = following
    assert state.phase == Phase.ROBBER_MOVE
    assert state.discard_player is None
    assert not state.discard_pending.any()
    np.testing.assert_array_equal(state.hands[1], hands[1])  # Seven cards are kept.
    np.testing.assert_array_equal(
        state.bank, initial_bank + np.array([choices[p] for p in order]).sum(axis=0)
    )


@pytest.mark.parametrize("players, roller", [(n, p) for n in (3, 4) for p in range(n)])
def test_pending_order_skips_finished_and_exempt_seats(players, roller):
    hands = [[0] * 5 for _ in range(players)]
    hands[0][0] = hands[-1][1] = 8
    state = resolve_seven(roll_state(hands, roller))
    order = [(roller + offset) % players for offset in range(players)]
    order = [p for p in order if p in (0, players - 1)]
    for player in order:
        assert state.discard_player == player
        counts = (4, 0, 0, 0, 0) if player == 0 else (0, 4, 0, 0, 0)
        state = apply_action(state, DiscardAction(player, counts))
    assert state.current_player == roller and state.phase == Phase.ROBBER_MOVE


def test_large_hand_discards_once_even_when_more_than_seven_remain():
    state = resolve_seven(roll_state([[19] * 5, [0] * 5, [0] * 5]))
    assert state.discard_pending[0] == 47
    state = apply_action(state, DiscardAction(0, (19, 19, 9, 0, 0)))
    assert state.hands[0].sum() == 48
    assert state.phase == Phase.ROBBER_MOVE
    state.validate()
    with pytest.raises(ValueError, match="DISCARD phase"):
        apply_action(state, DiscardAction(0, (0, 0, 0, 19, 5)))


def test_actions_normalize_and_freeze_input_without_retaining_mutable_counts():
    counts = [np.int64(1), 2, 3, 4, 5]
    action = DiscardAction(np.int64(2), counts)
    counts[0] = 9
    assert action.resources == (1, 2, 3, 4, 5)
    assert action.player == 2
    assert hash(action) == hash(DiscardAction(2, (1, 2, 3, 4, 5)))
    assert action == DiscardAction(2, np.array([1, 2, 3, 4, 5]))
    with pytest.raises(FrozenInstanceError):
        action.player = 0


@pytest.mark.parametrize(
    "player, counts",
    [(True, (0,) * 5), (-1, (0,) * 5), (1.0, (0,) * 5), ("0", (0,) * 5)]
    + [
        (0, c)
        for c in (
            None,
            "12345",
            (),
            (0,) * 6,
            (1.0, 0, 0, 0, 0),
            (True, 0, 0, 0, 0),
            (-1, 0, 0, 0, 0),
            (20, 0, 0, 0, 0),
            (10**100, 0, 0, 0, 0),
            (np.bool_(False), 0, 0, 0, 0),
            {0: 1, 1: 0, 2: 0, 3: 0, 4: 0},
            {0, 1, 2, 3, 4},
            iter([0] * 5),
            np.array(0),
        )
    ],
)
def test_invalid_action_values_are_rejected(player, counts):
    with pytest.raises(ValueError):
        DiscardAction(player, counts)


@pytest.mark.parametrize(
    "action, message",
    [
        (DiscardAction(1, (4, 0, 0, 0, 0)), "next pending"),
        (DiscardAction(99, (4, 0, 0, 0, 0)), "next pending"),
        (DiscardAction(0, (3, 0, 0, 0, 0)), "exactly"),
        (DiscardAction(0, (5, 0, 0, 0, 0)), "exactly"),
        (DiscardAction(0, (0, 0, 0, 0, 4)), "more of a resource"),
    ],
)
def test_rejected_choices_do_not_mutate_state_or_rng(action, message):
    state = resolve_seven(roll_state([[8, 0, 0, 0, 0], [0] * 5, [0] * 5]))
    before = deepcopy(state)
    with pytest.raises(ValueError, match=message):
        apply_action(state, action)
    assert state == before


@pytest.mark.parametrize("phase", [p for p in Phase if p != Phase.ROLL])
def test_seven_cannot_interrupt_another_phase(phase):
    state = new_game(0)
    state.phase = phase
    before = state.copy()
    with pytest.raises(ValueError, match="ROLL phase"):
        resolve_seven(state)
    assert state == before


def test_non_discard_actions_and_stale_rolls_are_rejected():
    state = roll_state([[8, 0, 0, 0, 0], [0] * 5, [0] * 5])
    with pytest.raises(TypeError, match="DiscardAction"):
        apply_action(state, object())
    state.discard_pending[0] = 4
    with pytest.raises(ValueError, match="existing pending"):
        resolve_seven(state)


@pytest.mark.parametrize("phase", [p for p in Phase if p != Phase.DISCARD])
def test_discards_cannot_interrupt_other_phases(phase):
    state = new_game(0)
    state.phase = phase
    before = state.copy()
    with pytest.raises(ValueError, match="DISCARD phase"):
        apply_action(state, DiscardAction(0, (0, 0, 0, 0, 0)))
    assert state == before


@pytest.mark.parametrize(
    "failure",
    (
        "negative",
        "outside",
        "empty",
        "roll",
        "count",
        "small",
        "shape",
        "dtype",
        "readonly",
    ),
)
def test_pending_discard_state_invariants(failure):
    state = resolve_seven(roll_state([[8, 0, 0, 0, 0], [0] * 5, [0] * 5]))
    if failure == "negative":
        state.discard_pending[0] = -1
    elif failure == "outside":
        state.phase = Phase.MAIN
    elif failure == "empty":
        state.discard_pending[:] = 0
    elif failure == "roll":
        state.last_roll = 8
    elif failure == "count":
        state.discard_pending[0] = 3
    elif failure == "small":
        state.hands[0, 0] -= 2
        state.bank[0] += 2
    elif failure == "shape":
        state.discard_pending = np.zeros(4, dtype=np.int16)
    elif failure == "dtype":
        state.discard_pending = state.discard_pending.astype(np.int8)
    elif failure == "readonly":
        state.discard_pending.flags.writeable = False
    with pytest.raises(ValueError, match="discard|DISCARD"):
        state.validate()


def test_varied_rounds_conserve_cards_and_replay_copies_without_rng_draws():
    global_rng = getstate()
    fixture_rng = Random(2026)
    for seed in range(500):
        players = 3 + seed % 2
        hands = [[0] * 5 for _ in range(players)]
        for resource in range(5):
            for _ in range(fixture_rng.randrange(20)):
                hands[fixture_rng.randrange(players)][resource] += 1
        source = roll_state(hands, fixture_rng.randrange(players))
        source.rng.gauss(0, 1)
        rng_state = source.rng.getstate()
        first = resolve_seven(source)
        second = deepcopy(first)
        expected_totals = [t // 2 + t % 2 if t > 7 else t for t in map(sum, hands)]
        while first.phase == Phase.DISCARD:
            player = first.discard_player
            cards = [r for r, n in enumerate(first.hands[player]) for _ in range(n)]
            selected = fixture_rng.sample(cards, int(first.discard_pending[player]))
            counts = tuple(selected.count(r) for r in range(5))
            action = DiscardAction(player, counts)
            first, second = apply_action(first, action), apply_action(second, action)
            assert first == second
            first.validate()
        assert first.hands.sum(axis=1).tolist() == expected_totals
        assert first.rng.getstate() == source.rng.getstate() == rng_state
        assert source.hands.tolist() == hands
    assert getstate() == global_rng

from dataclasses import fields
from random import Random, getstate

import numpy as np
import pytest

from catan.actions import DiscardAction
from catan.engine import (
    apply_action,
    apply_action_inplace,
    resolve_seven,
    resolve_seven_inplace,
)
from catan.state import GameState, Phase, new_game


def roll_state(hands, roller=0):
    state = new_game(42, len(hands))
    state.hands[:] = hands
    state.bank -= state.hands.sum(axis=0)
    state.phase = Phase.ROLL
    state.setup_index = len(state.setup_order)
    state.current_player = roller
    state.validate()
    return state


@pytest.mark.parametrize("players,roller", [(n, p) for n in (3, 4) for p in range(n)])
def test_varied_rounds_match_functional_rules_and_retain_all_mutable_objects(
    players, roller
):
    fixture_rng = Random(1234 + players * 10 + roller)
    global_rng = getstate()
    for _ in range(100):
        hands = [[0] * 5 for _ in range(players)]
        for resource in range(5):
            for _ in range(fixture_rng.randrange(20)):
                hands[fixture_rng.randrange(players)][resource] += 1
        source = roll_state(hands, roller)
        source.rng.gauss(0, 1)  # Preserve the cached Gaussian as well as MT state.
        original = source.copy()
        worker = source.copy()
        arrays = {
            f.name: getattr(worker, f.name)
            for f in fields(GameState)
            if isinstance(getattr(worker, f.name), np.ndarray)
        }
        worker_rng = worker.rng
        rng_state = worker_rng.getstate()
        functional = resolve_seven(source)
        assert resolve_seven_inplace(worker) is worker
        assert functional == worker
        assert source == original
        assert worker.discard_pending.tolist() == [
            sum(hand) // 2 if sum(hand) > 7 else 0 for hand in hands
        ]
        while worker.phase == Phase.DISCARD:
            player = worker.discard_player
            cards = [r for r, n in enumerate(worker.hands[player]) for _ in range(n)]
            selected = fixture_rng.sample(cards, int(worker.discard_pending[player]))
            action = DiscardAction(player, tuple(selected.count(r) for r in range(5)))
            prior = functional
            before = prior.copy()
            functional = apply_action(prior, action)
            assert apply_action_inplace(worker, action) is worker
            assert prior == before
            assert functional == worker
            worker.validate()
        assert worker.phase == Phase.ROBBER_MOVE
        assert worker.hands.sum(axis=1).tolist() == [
            (sum(hand) + 1) // 2 if sum(hand) > 7 else sum(hand) for hand in hands
        ]
        assert worker.current_player == roller
        assert worker.board is source.board
        assert worker.rng is worker_rng
        assert worker.rng.getstate() == rng_state
        assert source == original
        for name, array in arrays.items():
            assert getattr(worker, name) is array
            assert not np.shares_memory(array, getattr(source, name))
        worker.validate()
        assert [
            worker.rng.gauss(0, 1),
            worker.rng.random(),
            worker.rng.randrange(6),
        ] == [
            functional.rng.gauss(0, 1),
            functional.rng.random(),
            functional.rng.randrange(6),
        ]
    assert getstate() == global_rng


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("total", (0, 7, 8, 9, 19, 95))
def test_thresholds_and_large_hands_without_a_copy_or_repeated_discard(
    monkeypatch, players, total
):
    hand = [min(19, max(0, total - 19 * r)) for r in range(5)]
    state = roll_state([hand] + [[0] * 5] * (players - 1))
    rng_state = state.rng.getstate()

    def forbid_copy(_):
        pytest.fail("exclusive-owner transitions must not clone state")

    monkeypatch.setattr(GameState, "copy", forbid_copy)
    assert resolve_seven_inplace(state) is state
    if total > 7:
        owed = total // 2
        counts = []
        for held in hand:
            chosen = min(held, owed)
            counts.append(chosen)
            owed -= chosen
        assert apply_action_inplace(state, DiscardAction(0, counts)) is state
        with pytest.raises(ValueError, match="DISCARD phase"):
            apply_action_inplace(state, DiscardAction(0, (0,) * 5))
    assert state.phase == Phase.ROBBER_MOVE
    assert state.hands[0].sum() == ((total + 1) // 2 if total > 7 else total)
    assert state.rng.getstate() == rng_state
    state.validate()


@pytest.mark.parametrize(
    "action,error,message",
    [
        (object(), TypeError, "DiscardAction"),
        (DiscardAction(1, (4, 0, 0, 0, 0)), ValueError, "next pending"),
        (DiscardAction(99, (4, 0, 0, 0, 0)), ValueError, "next pending"),
        (DiscardAction(0, (3, 0, 0, 0, 0)), ValueError, "exactly"),
        (DiscardAction(0, (5, 0, 0, 0, 0)), ValueError, "exactly"),
        (DiscardAction(0, (0, 0, 0, 0, 4)), ValueError, "more of a resource"),
    ],
)
def test_rejected_inplace_actions_preserve_values_and_mutable_identities(
    action, error, message
):
    state = roll_state([[8, 0, 0, 0, 0], [0] * 5, [0] * 5])
    resolve_seven_inplace(state)
    before = state.copy()
    identities = {f.name: id(getattr(state, f.name)) for f in fields(GameState)}
    with pytest.raises(error, match=message):
        apply_action_inplace(state, action)
    assert state == before
    assert identities == {f.name: id(getattr(state, f.name)) for f in fields(GameState)}


@pytest.mark.parametrize("phase", [p for p in Phase if p != Phase.ROLL])
def test_inplace_seven_rejects_other_phases_without_mutation(phase):
    state = new_game(0)
    state.phase = phase
    before = state.copy()
    with pytest.raises(ValueError, match="ROLL phase"):
        resolve_seven_inplace(state)
    assert state == before


def test_inplace_seven_rejects_existing_pending_counts_without_mutation():
    state = roll_state([[8, 0, 0, 0, 0], [0] * 5, [0] * 5])
    state.discard_pending[0] = 4
    before = state.copy()
    with pytest.raises(ValueError, match="existing pending"):
        resolve_seven_inplace(state)
    assert state == before


@pytest.mark.parametrize("phase", [p for p in Phase if p != Phase.DISCARD])
def test_inplace_discard_rejects_other_phases_without_mutation(phase):
    state = new_game(0)
    state.phase = phase
    before = state.copy()
    with pytest.raises(ValueError, match="DISCARD phase"):
        apply_action_inplace(state, DiscardAction(0, (0,) * 5))
    assert state == before

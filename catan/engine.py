"""Seven/discard rules with functional and exclusive-owner in-place transitions.

Other roll totals, setup/building, robber moves/steals, and complete turns remain
future increments. Callers supply valid GameState objects; validate() is an
explicit debug gate rather than repeated in every hot transition.
"""

import numpy as np

from catan.actions import DiscardAction
from catan.state import GameState, Phase


def _validate_seven(state: GameState) -> None:
    if state.phase != Phase.ROLL:
        raise ValueError("a seven can only be resolved in ROLL phase")
    if any(state.discard_pending):
        raise ValueError("cannot resolve a seven with existing pending discards")


def _resolve_seven(state: GameState) -> GameState:
    # Reuse the pending array as reduction output, avoiding a totals array and
    # np.where temporaries. Valid hands contain at most 95 cards per seat.
    np.sum(state.hands, axis=1, out=state.discard_pending)
    pending = False
    for player, total in enumerate(state.discard_pending):
        count = int(total) // 2 if total > 7 else 0
        state.discard_pending[player] = count
        pending = pending or count > 0
    state.last_roll = 7
    state.phase = Phase.DISCARD if pending else Phase.ROBBER_MOVE
    return state


def resolve_seven(state: GameState) -> GameState:
    """Resolve an already-known seven on an independent successor state.

    Every player with more than seven resource cards owes half, rounded down.
    Snapshot these counts once so a large remaining hand is not discarded twice.
    Neither the source state nor its RNG is advanced.
    """
    _validate_seven(state)
    return _resolve_seven(state.copy())


def resolve_seven_inplace(state: GameState) -> GameState:
    """Resolve an already-known seven on a valid, exclusively owned state.

    Return the same object, retaining its arrays and RNG without cloning them.
    Rule rejection occurs before mutation. Callers must not share mutable state
    with another worker, saved snapshot, or concurrent reader.
    """
    _validate_seven(state)
    return _resolve_seven(state)


def _validate_discard(state: GameState, action: DiscardAction) -> np.ndarray:
    if not isinstance(action, DiscardAction):
        raise TypeError("this engine increment accepts DiscardAction only")
    if state.phase != Phase.DISCARD:
        raise ValueError("discard actions require DISCARD phase")
    if action.player != state.discard_player:
        raise ValueError("this player is not the next pending discard player")
    if sum(action.resources) != int(state.discard_pending[action.player]):
        raise ValueError("choose exactly the required number of cards")
    if any(
        chosen > held
        for chosen, held in zip(action.resources, state.hands[action.player])
    ):
        raise ValueError("cannot discard more of a resource than the player holds")
    # Convert once for both resource updates rather than coercing the tuple twice.
    # Allocate before mutation; no mutable scratch buffer is shared between calls.
    return np.asarray(action.resources, dtype=np.int16)


def _apply_discard(state: GameState, player: int, counts: np.ndarray) -> GameState:
    state.hands[player] -= counts
    state.bank += counts
    state.discard_pending[player] = 0
    if not any(state.discard_pending):
        state.phase = Phase.ROBBER_MOVE
    return state


def apply_action(state: GameState, action: DiscardAction) -> GameState:
    """Validate a manual discard and return its independent successor state.

    Pending players choose clockwise starting with the roller. Reject invalid
    choices before allocating a branch. After the last choice, the roller moves
    the robber; this increment stops at ROBBER_MOVE.
    """
    counts = _validate_discard(state, action)
    return _apply_discard(state.copy(), action.player, counts)


def apply_action_inplace(state: GameState, action: DiscardAction) -> GameState:
    """Apply a manual discard to a valid state exclusively owned by the caller.

    Return the same object and retain all array/RNG identities. All rule checks
    and count conversion finish before mutation, so rejected choices preserve
    the state. Only DiscardAction is implemented; this is not a full RL step.
    """
    counts = _validate_discard(state, action)
    return _apply_discard(state, action.player, counts)

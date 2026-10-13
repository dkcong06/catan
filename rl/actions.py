"""Fixed action indices and masks, separate from engine state and RNG.

Discards use five conditional count masks rather than enumerating hand vectors.
"""

import numpy as np

from catan import actions as a, engine as e
from catan.state import Phase, _integer

ACTION_SCHEMA_VERSION = 1
ACTION_DIM = 254
_CATALOG = (
    e._ROADS
    + e._SETTLEMENTS
    + e._CITIES
    + e._ROBBER_MOVES
    + tuple(action for group in e._TRADES for action in group)
    + (e._END_TURN, e._ROLL)
    + e._STEALS  # Relative victim slots; converted at the API boundary.
    + (e._BUY_DEVELOPMENT, e._KNIGHT, e._ROAD_BUILDING)
    + e._MONOPOLIES
    + e._PLENTIES
)
_INDEX = {action: i for i, action in enumerate(_CATALOG)}


def action_index(state, action):
    """Encode a typed action; membership does not imply legality."""
    if isinstance(action, a.StealAction):
        if action.victim >= state.n_players:
            raise ValueError("victim is outside the game")
        return 221 + (action.victim - state.current_player) % state.n_players
    try:
        return _INDEX[action]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "action is outside the flat catalog; discards use count masks"
        ) from error


def indexed_action(state, index):
    """Decode an index; callers must apply the legality mask before choosing."""
    index = _integer(index, "index")
    if not 0 <= index < ACTION_DIM:
        raise ValueError("index is outside the action catalog")
    if 221 <= index < 225:
        relative = index - 221
        if relative >= state.n_players:
            raise ValueError("victim slot is absent in this game")
        return e._STEALS[(state.current_player + relative) % state.n_players]
    return _CATALOG[index]


def action_mask(state, *, out=None):
    """Boolean legality using the engine's predicates, without object enumeration.

    GAME_OVER returns all false. DISCARD requires discard_count_mask instead.
    Valid state is assumed; no invariant checking, copies or RNG draws occur.
    """
    if state.phase == Phase.DISCARD:
        raise ValueError("DISCARD uses discard_count_mask")
    if out is None:
        out = np.zeros(ACTION_DIM, dtype=np.bool_)
    else:
        if (
            not isinstance(out, np.ndarray)
            or out.shape != (ACTION_DIM,)
            or out.dtype != np.bool_
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            raise ValueError(
                "out must be writable contiguous bool of shape (ACTION_DIM,)"
            )
        from catan.state import _ARRAY_FIELDS

        if any(
            np.may_share_memory(out, getattr(state, name)) for name in _ARRAY_FIELDS
        ):
            raise ValueError("out must not share memory with state arrays")
        out.fill(False)
    p, phase = state.current_player, state.phase
    if phase == Phase.SETUP_SETTLEMENT:
        if state.pieces_left[p, 0]:
            for vertex in range(54):
                out[72 + vertex] = e._settlement_spot(state, vertex, True)
    elif phase == Phase.SETUP_ROAD:
        if state.pieces_left[p, 2]:
            for edge in state.board.geometry.vertex_edges[state.setup_vertex]:
                out[edge] = e._road_spot(state, edge, True)
    elif phase == Phase.ROLL:
        out[220] = True
    elif phase == Phase.ROBBER_MOVE:
        out[180:199] = True
        out[180 + state.robber_tile] = False
    elif phase == Phase.ROBBER_STEAL:
        for victim in e._victims(state):
            out[221 + (victim - p) % state.n_players] = True
    elif phase == Phase.MAIN:
        if state.pieces_left[p, 0] and e._affordable(state, e._SETTLEMENT_COST):
            vertices = {
                v
                for edge in np.flatnonzero(state.edge_owner == p)
                for v in state.board.geometry.edge_vertices[edge]
            }
            for vertex in vertices:
                out[72 + vertex] = e._settlement_spot(state, vertex, False)
        if state.pieces_left[p, 1] and e._affordable(state, e._CITY_COST):
            out[126:180] = (state.vertex_owner == p) & (state.vertex_building == 1)
        ratios = e._trade_ratios(state)
        for give in range(5):
            if state.hands[p, give] >= ratios[give]:
                for receive in range(5):
                    if receive != give and state.bank[receive]:
                        out[199 + give * 4 + receive - (receive > give)] = True
        out[219] = True
        out[225] = any(state.development_deck) and e._affordable(
            state, e._DEVELOPMENT_COST
        )
    elif phase not in (Phase.ROAD_BUILDING, Phase.GAME_OVER):
        raise ValueError("unrecognized phase")
    if phase == Phase.ROAD_BUILDING or (
        phase == Phase.MAIN
        and state.pieces_left[p, 2]
        and e._affordable(state, e._ROAD_COST)
    ):
        for action in e._road_choices(state):
            out[action.edge] = True
    if phase in (Phase.ROLL, Phase.MAIN):
        for action in e._development_choices(state):
            out[_INDEX[action]] = True
    return out


def discard_count_mask(state, prefix=()):
    """Twenty choices for the next resource count given zero to four earlier counts.

    Every accepted prefix admits a complete physical discard. Order is resource
    enum order. Choose one true count, append it, and request the next mask.
    """
    if state.phase != Phase.DISCARD:
        raise ValueError("count masks require DISCARD")
    player = state.discard_player
    return _discard_count_mask(
        state.hands[player].tolist(), int(state.discard_pending[player]), prefix
    )


def _discard_count_mask(hand, required, prefix):
    # Shared by live state queries and independent environment decisions.
    prefix = tuple(_integer(count, "count") for count in prefix)
    if len(prefix) > 4:
        raise ValueError("prefix must contain zero to four counts")
    if any(count < 0 or count > hand[r] for r, count in enumerate(prefix)):
        raise ValueError("prefix exceeds the hand")
    remaining = required - sum(prefix)
    resource = len(prefix)
    minimum = max(0, remaining - sum(hand[resource + 1 :]))
    maximum = min(hand[resource], remaining)
    if minimum > maximum:
        raise ValueError("prefix cannot complete the required discard")
    mask = np.zeros(20, dtype=np.bool_)
    mask[minimum : maximum + 1] = True
    return mask

"""Fixed float32 observations for the expanded game; no learning framework required.

Only the observer's hand is exposed by resource. Prepared encoders share immutable
board features and own no rollout state. Output buffers are fully overwritten.
"""

from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from catan.board import Board, GEOMETRY
from catan.state import GameState, Phase, _ARRAY_FIELDS, _integer


SCHEMA_VERSION = 2
_GROUPS = (
    ("tiles", (len(GEOMETRY.tile_coordinates), 9)),
    ("vertices", (len(GEOMETRY.vertex_coordinates), 16)),
    ("edges", (len(GEOMETRY.edge_vertices), 5)),
    ("my_hand", (5,)),
    ("opponent_hand_sizes", (3,)),
    ("players", (4, 9)),
    ("bank", (5,)),
    ("phase", (len(Phase),)),
    ("last_roll", (12,)),
    ("turn_owner", (4,)),
    ("acting_player", (4,)),
    ("setup_progress", (1,)),
    ("my_development", (5,)),
    ("my_development_new", (5,)),
    ("development_deck_size", (1,)),
    ("development_played", (5,)),
    ("longest_road_owner", (5,)),
    ("largest_army_owner", (5,)),
    ("development_used", (1,)),
    ("free_roads_remaining", (1,)),
    ("resume_phase", (3,)),
)
_slices = {}
_offset = 0
for _name, _shape in _GROUPS:
    _size = prod(_shape)
    _slices[_name] = slice(_offset, _offset + _size)
    _offset += _size
FEATURE_SLICES = MappingProxyType(_slices)
FEATURE_SHAPES = MappingProxyType(dict(_GROUPS))
OBS_DIM = _offset
del _slices, _offset, _name, _shape, _size


def _readonly(values, dtype=np.float32):
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


_RESOURCE_TYPES = _readonly(np.eye(6, dtype=np.float32))
_BUILDING_TYPES = _readonly(np.eye(3, dtype=np.float32))
_PORT_TYPES = _readonly(np.eye(7, dtype=np.float32))
_PROBABILITIES = _readonly(
    np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=np.float32) / 36
)
# Python arithmetic avoids tiny ufunc/reduction costs; retain float32 constants.
_RESOURCE_SCALE, _HAND_SCALE, _VP_SCALE, _DISCARD_SCALE = (
    float(np.float32(1 / maximum)) for maximum in (19, 95, 10, 47)
)
_PIECE_SCALE = tuple(float(np.float32(1 / maximum)) for maximum in (5, 4, 15))
_SEAT_ORDERS = {}
_OWNER_TYPES = {}
for _players in (3, 4):
    for _me in range(_players):
        _SEAT_ORDERS[_players, _me] = tuple(
            (_me + i) % _players for i in range(_players)
        )
        _owners = np.zeros((5, 5), dtype=np.float32)
        for _owner in range(_players):
            _owners[_owner, (_owner - _me) % _players] = 1
        _owners[-1, -1] = 1
        _OWNER_TYPES[_players, _me] = _readonly(_owners)
_SEAT_ORDERS = MappingProxyType(_SEAT_ORDERS)
_OWNER_TYPES = MappingProxyType(_OWNER_TYPES)
del _players, _me, _owner, _owners


def decision_player(state: GameState) -> int | None:
    """Return the manual discarder, normal turn owner, or None at GAME_OVER."""
    if state.phase == Phase.GAME_OVER:
        return None
    return (
        state.discard_player if state.phase == Phase.DISCARD else state.current_player
    )


@dataclass(frozen=True, slots=True, eq=False)
class FlatEncoder:
    """Prepare once per immutable board, then encode many valid states.

    Functional successors/copies share the board and can reuse this encoder.
    A new board/reset requires a new encoder; there is no global board cache.
    """

    board: Board
    _base: NDArray[np.float32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.board, Board):
            raise TypeError("board must be a Board")
        base = np.zeros(OBS_DIM, dtype=np.float32)
        tiles = base[FEATURE_SLICES["tiles"]].reshape(FEATURE_SHAPES["tiles"])
        tiles[:, :6] = _RESOURCE_TYPES[list(self.board.resources)]
        tiles[:, 6] = _PROBABILITIES[list(self.board.numbers)]
        # Probability alone conflates 6 with 8, etc. One bit retains the number
        # identity and simultaneous-roll correlations without eleven extra flags.
        tiles[:, 7] = np.asarray(self.board.numbers) > 7
        vertices = base[FEATURE_SLICES["vertices"]].reshape(FEATURE_SHAPES["vertices"])
        ports = [
            6 if p == -1 else int(self.board.ports[p])
            for p in self.board.geometry.vertex_ports
        ]
        vertices[:, 8:15] = _PORT_TYPES[ports]
        object.__setattr__(self, "_base", _readonly(base))

    def encode(
        self,
        state: GameState,
        player: int | None = None,
        *,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Encode without mutating state/RNG or running full state validation.

        Default perspective is the decision maker (winner at GAME_OVER).
        Explicit perspectives are supported for terminal/value observations.
        out must be writable contiguous float32 of shape (OBS_DIM,), disjoint
        from state arrays. Reused output must be consumed/copied before reuse.
        """
        if state.board is not self.board:
            raise ValueError(
                "encoder belongs to a different board; prepare after reset"
            )
        actor = decision_player(state)
        if player is None:
            player = state.current_player if actor is None else actor
        else:
            player = _integer(player, "player")
        n = state.n_players
        if not 0 <= player < n:
            raise ValueError("player is outside the game")
        if out is None:
            out = np.empty(OBS_DIM, dtype=np.float32)
        else:
            if (
                not isinstance(out, np.ndarray)
                or out.shape != (OBS_DIM,)
                or out.dtype != np.float32
            ):
                raise ValueError("out must have shape (OBS_DIM,) and dtype float32")
            if not out.flags.c_contiguous or not out.flags.writeable:
                raise ValueError("out must be contiguous and writable")
            if any(
                np.may_share_memory(out, getattr(state, name)) for name in _ARRAY_FIELDS
            ):
                raise ValueError("out must not share memory with state arrays")
        np.copyto(out, self._base)
        tiles = out[FEATURE_SLICES["tiles"]].reshape(FEATURE_SHAPES["tiles"])
        tiles[state.robber_tile, 8] = 1
        vertices = out[FEATURE_SLICES["vertices"]].reshape(FEATURE_SHAPES["vertices"])
        owners = _OWNER_TYPES[n, player]
        vertices[:, :5] = owners[state.vertex_owner]
        vertices[:, 5:8] = _BUILDING_TYPES[state.vertex_building]
        if state.setup_vertex is not None:
            vertices[state.setup_vertex, 15] = 1
        edges = out[FEATURE_SLICES["edges"]].reshape(FEATURE_SHAPES["edges"])
        edges[:] = owners[state.edge_owner]
        _write_globals(state, player, actor, out)
        return out


_DECK_SCALE, _KNIGHT_SCALE = float(np.float32(1 / 25)), float(np.float32(1 / 14))
_CARD_SCALES = tuple(float(np.float32(1 / n)) for n in (14, 5, 2, 2, 2))


_FLAGS = {
    size: tuple(tuple(float(i == j) for i in range(size)) for j in range(size))
    for size in (3, 4, 5, 9, 12)
}


def _write_globals(state, player, actor, out):
    n = state.n_players
    hands = state.hands.tolist()
    order = _SEAT_ORDERS[n, player]
    values = [count * _RESOURCE_SCALE for count in hands[player]]
    values.extend(sum(hands[p]) * _HAND_SCALE for p in order[1:])
    values.extend([0] * (4 - n))
    vp = state.vp.tolist()
    cards = state.development.tolist()
    knights = state.knights_played.tolist()
    roads = state.road_lengths.tolist()
    pieces = state.pieces_left.tolist()
    pending = state.discard_pending.tolist()
    for p in order:
        public_vp = (
            vp[p]
            if p == player or state.phase == Phase.GAME_OVER
            else vp[p] - cards[p][1]
        )
        values.extend(
            (
                1,
                public_vp * _VP_SCALE,
                pieces[p][0] * _PIECE_SCALE[0],
                pieces[p][1] * _PIECE_SCALE[1],
                pieces[p][2] * _PIECE_SCALE[2],
                pending[p] * _DISCARD_SCALE,
                sum(cards[p]) * _DECK_SCALE,
                knights[p] * _KNIGHT_SCALE,
                roads[p] * _PIECE_SCALE[2],
            )
        )
    values.extend([0] * (9 * (4 - n)))
    values.extend(count * _RESOURCE_SCALE for count in state.bank.tolist())
    values.extend(_FLAGS[9][state.phase.value - 1])
    values.extend(_FLAGS[12][0 if state.last_roll is None else state.last_roll - 1])
    values.extend(_FLAGS[4][(state.current_player - player) % n])
    values.extend((0, 0, 0, 0) if actor is None else _FLAGS[4][(actor - player) % n])
    values.append(state.setup_index / len(state.setup_order))
    values.extend(count * scale for count, scale in zip(cards[player], _CARD_SCALES))
    values.extend(
        count * scale
        for count, scale in zip(state.development_new[player].tolist(), _CARD_SCALES)
    )
    values.append(sum(map(int, state.development_deck)) / 25)
    values.extend(
        count * scale
        for count, scale in zip(state.development_played.tolist(), _CARD_SCALES)
    )
    for name in ("longest_road_owner", "largest_army_owner"):
        owner = getattr(state, name)
        values.extend(_FLAGS[5][4 if owner == -1 else (owner - player) % n])
    values.extend((state.development_used, state.free_roads_remaining / 2))
    values.extend(
        _FLAGS[3][
            0
            if state.resume_phase is None
            else 1
            if state.resume_phase == Phase.ROLL
            else 2
        ]
    )
    out[FEATURE_SLICES["my_hand"].start :] = values


def encode_flat(
    state: GameState,
    player: int | None = None,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """One-off convenience; reuse FlatEncoder(state.board) in rollout loops."""
    return FlatEncoder(state.board).encode(state, player, out=out)


def encode_batch(states, encoders, players=None, *, out=None):
    """Encode mixed boards/seats in bounded chunks; exact scalar output per row.

    Supply one prepared encoder per row. Small batches use the scalar path.
    All contexts/observers/buffer aliases are checked before writing output.
    """
    states, encoders = tuple(states), tuple(encoders)
    if len(encoders) != len(states):
        raise ValueError("one encoder is required per state")
    if players is None:
        players = (None,) * len(states)
    else:
        players = tuple(players)
        if len(players) != len(states):
            raise ValueError("one player is required per state")
    observers = []
    for state, encoder, player in zip(states, encoders, players):
        if not isinstance(encoder, FlatEncoder) or state.board is not encoder.board:
            raise ValueError("encoder belongs to a different board")
        actor = decision_player(state)
        me = (
            (state.current_player if actor is None else actor)
            if player is None
            else _integer(player, "player")
        )
        if not 0 <= me < state.n_players:
            raise ValueError("player is outside the game")
        observers.append(me)
    size = len(states)
    if out is None:
        out = np.empty((size, OBS_DIM), dtype=np.float32)
    else:
        if (
            not isinstance(out, np.ndarray)
            or out.shape != (size, OBS_DIM)
            or out.dtype != np.float32
        ):
            raise ValueError("out must have shape (batch, OBS_DIM) and dtype float32")
        if not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("out must be contiguous and writable")
        if any(
            np.may_share_memory(out, getattr(state, name))
            for state in states
            for name in _ARRAY_FIELDS
        ):
            raise ValueError("out must not share memory with state arrays")
    if size < 8:
        for i, (state, encoder, me) in enumerate(zip(states, encoders, observers)):
            out[i] = encoder.encode(state, me)
        return out
    for start in range(0, size, 128):
        stop = min(start + 128, size)
        chunk, contexts, perspective = (
            states[start:stop],
            encoders[start:stop],
            observers[start:stop],
        )
        target = out[start:stop]
        target[:] = np.stack([encoder._base for encoder in contexts])
        rows = np.arange(len(chunk))
        owners = np.stack(
            [_OWNER_TYPES[state.n_players, me] for state, me in zip(chunk, perspective)]
        )
        tiles = target[:, FEATURE_SLICES["tiles"]].reshape(-1, 19, 9)
        tiles[rows, [state.robber_tile for state in chunk], 8] = 1
        vertices = target[:, FEATURE_SLICES["vertices"]].reshape(-1, 54, 16)
        vertices[:, :, :5] = owners[
            rows[:, None], np.stack([state.vertex_owner for state in chunk])
        ]
        vertices[:, :, 5:8] = _BUILDING_TYPES[
            np.stack([state.vertex_building for state in chunk])
        ]
        edges = target[:, FEATURE_SLICES["edges"]].reshape(-1, 72, 5)
        edges[:] = owners[
            rows[:, None], np.stack([state.edge_owner for state in chunk])
        ]
        for i, (state, me) in enumerate(zip(chunk, perspective)):
            if state.setup_vertex is not None:
                vertices[i, state.setup_vertex, 15] = 1
            _write_globals(state, me, decision_player(state), target[i])
    return out

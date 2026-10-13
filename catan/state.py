"""Compact mutable game state, independent of rules and learning code."""

from dataclasses import dataclass, field
from enum import IntEnum
from operator import index
from random import Random

import numpy as np
from numpy.typing import NDArray

from catan.board import Board, PortType, make_board
from catan.cards import DEVELOPMENT_SUPPLY, DevelopmentCard


class Phase(IntEnum):
    """Explicit decisions for setup and the game's turn sequence."""

    SETUP_SETTLEMENT = 1
    SETUP_ROAD = 2
    ROLL = 3
    ROBBER_MOVE = 4
    ROBBER_STEAL = 5
    MAIN = 6
    GAME_OVER = 7
    DISCARD = 8  # Append to preserve the existing phase values.
    ROAD_BUILDING = 9


_PIECE_SUPPLY = (5, 4, 15)  # settlement, city, road
_SETUP_ORDERS = {3: (0, 1, 2, 2, 1, 0), 4: (0, 1, 2, 3, 3, 2, 1, 0)}
_ARRAY_FIELDS = (
    "hands",
    "vp",
    "pieces_left",
    "vertex_owner",
    "vertex_building",
    "edge_owner",
    "bank",
    "discard_pending",
    "development",
    "development_new",
    "development_deck",
    "development_played",
    "knights_played",
    "road_lengths",
)
_SCALAR_FIELDS = (
    "n_players",
    "robber_tile",
    "current_player",
    "phase",
    "last_roll",
    "setup_index",
    "setup_vertex",
    "longest_road_owner",
    "largest_army_owner",
    "development_used",
    "free_roads_remaining",
    "resume_phase",
)


def _integer(value: object, name: str) -> int:
    """Accept integer indices, including NumPy integers, without coercing floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _player_count(value: int) -> int:
    count = _integer(value, "n_players")
    if count not in _SETUP_ORDERS:
        raise ValueError("n_players must be 3 or 4")
    return count


def _arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare small integer arrays without separate NumPy reductions.

    C-order bytes retain shape/dtype/value semantics even for strided inputs.
    Fall back to NumPy for noninteger arrays (not valid game storage), preserving
    numeric equality for signed zero, NaN, and object values in debug comparisons.
    """
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    if first.dtype.kind in "iu":
        return first.tobytes() == second.tobytes()
    return bool(np.array_equal(first, second))


@dataclass(slots=True, eq=False)
class GameState:
    """Own mutable arrays and an RNG; share the immutable starting board.

    Prefer ``new_game(seed)`` so board setup and subsequent random draws use one
    stream. Passing an RNG to this constructor transfers its use to the state;
    copies clone it rather than advancing or sharing the original.
    """

    board: Board = field(repr=False)
    n_players: int = 4
    rng: Random = field(default_factory=Random, repr=False)
    hands: NDArray[np.int16] = field(init=False, repr=False)
    vp: NDArray[np.int16] = field(init=False, repr=False)
    pieces_left: NDArray[np.int8] = field(init=False, repr=False)
    vertex_owner: NDArray[np.int8] = field(init=False, repr=False)
    vertex_building: NDArray[np.int8] = field(init=False, repr=False)
    edge_owner: NDArray[np.int8] = field(init=False, repr=False)
    bank: NDArray[np.int16] = field(init=False, repr=False)
    discard_pending: NDArray[np.int16] = field(init=False, repr=False)
    development: NDArray[np.int8] = field(init=False, repr=False)
    development_new: NDArray[np.int8] = field(init=False, repr=False)
    development_deck: NDArray[np.int8] = field(init=False, repr=False)
    development_played: NDArray[np.int8] = field(init=False, repr=False)
    knights_played: NDArray[np.int8] = field(init=False, repr=False)
    road_lengths: NDArray[np.int8] = field(init=False, repr=False)
    longest_road_owner: int = field(init=False, default=-1)
    largest_army_owner: int = field(init=False, default=-1)
    development_used: bool = field(init=False, default=False)
    free_roads_remaining: int = field(init=False, default=0)
    resume_phase: Phase | None = field(init=False, default=None)
    robber_tile: int = field(init=False)
    current_player: int = field(init=False, default=0)
    phase: Phase = field(init=False, default=Phase.SETUP_SETTLEMENT)
    last_roll: int | None = field(init=False, default=None)
    setup_index: int = field(init=False, default=0)
    setup_vertex: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.n_players = _player_count(self.n_players)
        if type(self.rng) is not Random:
            raise TypeError(
                "rng must be a standard random.Random for reproducible copies"
            )
        self.hands = np.zeros((self.n_players, 5), dtype=np.int16)
        self.vp = np.zeros(self.n_players, dtype=np.int16)
        self.pieces_left = np.empty((self.n_players, 3), dtype=np.int8)
        self.pieces_left[:] = _PIECE_SUPPLY
        self.vertex_owner = np.full(
            len(self.board.geometry.vertex_coordinates), -1, dtype=np.int8
        )
        self.vertex_building = np.zeros(self.vertex_owner.shape, dtype=np.int8)
        self.edge_owner = np.full(
            len(self.board.geometry.edge_vertices), -1, dtype=np.int8
        )
        self.bank = np.full(5, 19, dtype=np.int16)
        self.discard_pending = np.zeros(self.n_players, dtype=np.int16)
        self.development = np.zeros((self.n_players, 5), dtype=np.int8)
        self.development_new = np.zeros_like(self.development)
        # Conditional weighted draws equal a shuffled hidden deck's distribution,
        # with only five counters and no setup RNG draw or redundant deck cursor.
        self.development_deck = np.array(DEVELOPMENT_SUPPLY, dtype=np.int8)
        self.development_played = np.zeros(5, dtype=np.int8)
        self.knights_played = np.zeros(self.n_players, dtype=np.int8)
        self.road_lengths = np.zeros(self.n_players, dtype=np.int8)
        self.robber_tile = self.board.robber_tile

    @property
    def setup_order(self) -> tuple[int, ...]:
        """Shared snake order, including the last seat's consecutive placements."""
        return _SETUP_ORDERS[self.n_players]

    @property
    def array_bytes(self) -> int:
        """Mutable array payload bytes, excluding Python objects, RNG, and board."""
        return sum(getattr(self, name).nbytes for name in _ARRAY_FIELDS)

    @property
    def discard_player(self) -> int | None:
        """Next pending seat, clockwise from the player whose turn it remains.

        Serializing these independent choices makes the engine deterministic.
        ``current_player`` stays the roller, who moves the robber after all choices.
        """
        if self.phase == Phase.DISCARD:
            for offset in range(self.n_players):
                player = (self.current_player + offset) % self.n_players
                if self.discard_pending[player] > 0:
                    return player
        return None

    def copy(self) -> "GameState":
        """Clone every mutable field without rebuilding the board or initializing arrays."""
        clone = object.__new__(GameState)
        clone.board = self.board
        clone.n_players = self.n_players
        clone.robber_tile = self.robber_tile
        clone.current_player = self.current_player
        clone.phase = self.phase
        clone.last_roll = self.last_roll
        clone.setup_index = self.setup_index
        clone.setup_vertex = self.setup_vertex
        clone.longest_road_owner = self.longest_road_owner
        clone.largest_army_owner = self.largest_army_owner
        clone.development_used = self.development_used
        clone.free_roads_remaining = self.free_roads_remaining
        clone.resume_phase = self.resume_phase
        clone.hands = self.hands.copy()
        clone.vp = self.vp.copy()
        clone.pieces_left = self.pieces_left.copy()
        clone.vertex_owner = self.vertex_owner.copy()
        clone.vertex_building = self.vertex_building.copy()
        clone.edge_owner = self.edge_owner.copy()
        clone.bank = self.bank.copy()
        clone.discard_pending = self.discard_pending.copy()
        clone.development = self.development.copy()
        clone.development_new = self.development_new.copy()
        clone.development_deck = self.development_deck.copy()
        clone.development_played = self.development_played.copy()
        clone.knights_played = self.knights_played.copy()
        clone.road_lengths = self.road_lengths.copy()
        # Restore the complete RNG state before any draw; no unused reseeding.
        clone.rng = Random.__new__(Random)
        clone.rng.setstate(self.rng.getstate())
        return clone

    def __copy__(self) -> "GameState":
        """Python's copy.copy also isolates all mutable state."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, object]) -> "GameState":
        """Python's deepcopy shares the immutable board and preserves mutable aliases."""
        clone = self.copy()
        memo[id(self)] = clone
        for name in (*_ARRAY_FIELDS, "rng"):
            source = getattr(self, name)
            copied = memo.setdefault(id(source), getattr(clone, name))
            setattr(clone, name, copied)
        memo.setdefault(id(self.board), self.board)
        memo.setdefault(id(self.board.geometry), self.board.geometry)
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return NotImplemented
        return (
            self.board == other.board
            and all(
                getattr(self, name) == getattr(other, name) for name in _SCALAR_FIELDS
            )
            and all(
                _arrays_equal(getattr(self, name), getattr(other, name))
                for name in _ARRAY_FIELDS
            )
            and self.rng.getstate() == other.rng.getstate()
        )

    def port_access(self, player: int) -> tuple[PortType, ...]:
        """Return distinct port types granted by this player's settlements/cities.

        Query nine ports instead of scanning all vertices. Roads do not grant access.
        No ownership or trade-ratio cache needs to be synchronized with future moves.
        """
        player = _integer(player, "player")
        if not 0 <= player < self.n_players:
            raise ValueError("player is outside the game")
        return tuple(
            sorted(
                {
                    port
                    for port, (first, second) in zip(
                        self.board.ports, self.board.geometry.port_vertices
                    )
                    if (
                        self.vertex_owner[first] == player
                        and self.vertex_building[first] > 0
                    )
                    or (
                        self.vertex_owner[second] == player
                        and self.vertex_building[second] > 0
                    )
                }
            )
        )

    def validate(self) -> None:
        """Raise ValueError on broken minimal-game storage/bookkeeping invariants.

        This debug check is explicit, not part of the hot copy path. It checks
        counts and the distance rule; action legality/connectivity belongs to Phase 4.
        """
        n_players = _player_count(self.n_players)
        specifications = (
            ("hands", (n_players, 5), np.int16),
            ("vp", (n_players,), np.int16),
            ("pieces_left", (n_players, 3), np.int8),
            ("vertex_owner", (54,), np.int8),
            ("vertex_building", (54,), np.int8),
            ("edge_owner", (72,), np.int8),
            ("bank", (5,), np.int16),
            ("discard_pending", (n_players,), np.int16),
            ("development", (n_players, 5), np.int8),
            ("development_new", (n_players, 5), np.int8),
            ("development_deck", (5,), np.int8),
            ("development_played", (5,), np.int8),
            ("knights_played", (n_players,), np.int8),
            ("road_lengths", (n_players,), np.int8),
        )
        for name, shape, dtype in specifications:
            array = getattr(self, name)
            if (
                not isinstance(array, np.ndarray)
                or array.shape != shape
                or array.dtype != dtype
            ):
                raise ValueError(f"{name} has an invalid shape or dtype")
            if not array.flags.c_contiguous or not array.flags.writeable:
                raise ValueError(f"{name} must be contiguous and writable")
        # These arrays contain at most 72 values. Convert once per debug check
        # instead of allocating masks/reductions repeatedly for each player.
        hands = self.hands.tolist()
        bank = self.bank.tolist()
        if any(count < 0 for hand in hands for count in hand) or any(
            count < 0 for count in bank
        ):
            raise ValueError("resource counts must be nonnegative")
        if any(
            sum(hand[r] for hand in hands) + count != 19 for r, count in enumerate(bank)
        ):
            raise ValueError("resource supply is not conserved")
        development = self.development.tolist()
        fresh = self.development_new.tolist()
        deck = self.development_deck.tolist()
        played = self.development_played.tolist()
        if any(c < 0 for row in development + fresh for c in row) or any(
            c < 0 for c in deck + played
        ):
            raise ValueError("development card counts must be nonnegative")
        if any(
            new > held
            for row, held_row in zip(fresh, development)
            for new, held in zip(row, held_row)
        ):
            raise ValueError("new development cards exceed held cards")
        if any(
            sum(row[k] for row in development) + deck[k] + played[k] != supply
            for k, supply in enumerate(DEVELOPMENT_SUPPLY)
        ):
            raise ValueError("development card supply is not conserved")
        knights = self.knights_played.tolist()
        lengths = self.road_lengths.tolist()
        if (
            any(not 0 <= count <= 14 for count in knights)
            or sum(knights) != played[DevelopmentCard.KNIGHT]
        ):
            raise ValueError("played knight counts disagree")
        if played[DevelopmentCard.VICTORY_POINT]:
            raise ValueError("victory point cards remain held")
        if any(not 0 <= count <= 15 for count in lengths):
            raise ValueError("road lengths must be from 0 to 15")
        for name, values, minimum in (
            ("longest_road_owner", lengths, 5),
            ("largest_army_owner", knights, 3),
        ):
            owner = _integer(getattr(self, name), name)
            if not -1 <= owner < n_players:
                raise ValueError(f"{name} is outside the game")
            if owner != -1 and (
                values[owner] < minimum or values[owner] != max(values)
            ):
                raise ValueError(f"{name} does not meet its award requirements")
        if type(self.development_used) is not bool:
            raise ValueError("development_used must be bool")
        if any(any(row) for p, row in enumerate(fresh) if p != self.current_player):
            raise ValueError("only the current turn retains newly bought cards")
        remaining = _integer(self.free_roads_remaining, "free_roads_remaining")
        if self.phase == Phase.ROAD_BUILDING:
            if remaining not in (1, 2) or not self.development_used:
                raise ValueError("ROAD_BUILDING requires free roads and a played card")
        elif remaining != 0:
            raise ValueError("free roads require ROAD_BUILDING phase")
        if self.resume_phase is not None:
            if self.resume_phase not in (Phase.ROLL, Phase.MAIN) or not isinstance(
                self.resume_phase, Phase
            ):
                raise ValueError("resume_phase must be ROLL or MAIN")
            if self.phase not in (
                Phase.ROAD_BUILDING,
                Phase.ROBBER_MOVE,
                Phase.ROBBER_STEAL,
                Phase.DISCARD,
            ):
                raise ValueError("resume_phase requires a card/robber/discard decision")
        if self.phase == Phase.ROAD_BUILDING and self.resume_phase is None:
            raise ValueError("free roads require a resume phase")
        pieces = self.pieces_left.tolist()
        if any(
            not 0 <= count <= supply
            for row in pieces
            for count, supply in zip(row, _PIECE_SUPPLY)
        ):
            raise ValueError("remaining piece counts are outside their supply")
        vertex_owner = self.vertex_owner.tolist()
        edge_owner = self.edge_owner.tolist()
        for name, owners in (
            ("vertex_owner", vertex_owner),
            ("edge_owner", edge_owner),
        ):
            if any(not -1 <= owner < n_players for owner in owners):
                raise ValueError(f"{name} contains an invalid owner")
        buildings = self.vertex_building.tolist()
        if any(not 0 <= building <= 2 for building in buildings):
            raise ValueError("invalid building type")
        if any(
            (owner == -1) != (building == 0)
            for owner, building in zip(vertex_owner, buildings)
        ):
            raise ValueError("building types disagree with vertex ownership")
        placed = [[0, 0, 0] for _ in range(n_players)]
        for owner, building in zip(vertex_owner, buildings):
            if owner != -1:
                placed[owner][building - 1] += 1
        for owner in edge_owner:
            if owner != -1:
                placed[owner][2] += 1
        for player, (remaining, used) in enumerate(zip(pieces, placed)):
            if any(
                left + count != supply
                for left, count, supply in zip(remaining, used, _PIECE_SUPPLY)
            ):
                raise ValueError("placed and remaining pieces disagree")
            expected_vp = (
                used[0]
                + 2 * used[1]
                + development[player][DevelopmentCard.VICTORY_POINT]
                + 2 * (self.longest_road_owner == player)
                + 2 * (self.largest_army_owner == player)
            )
            if self.vp[player] != expected_vp:
                raise ValueError(
                    "victory points disagree with buildings, cards and awards"
                )
        if any(
            buildings[first] and buildings[second]
            for first, second in self.board.geometry.edge_vertices
        ):
            raise ValueError("adjacent buildings violate the distance rule")
        if not 0 <= _integer(self.robber_tile, "robber_tile") < 19:
            raise ValueError("robber_tile is outside the board")
        if not 0 <= _integer(self.current_player, "current_player") < n_players:
            raise ValueError("current_player is outside the game")
        if not isinstance(self.phase, Phase):
            raise ValueError("phase must be a Phase")
        pending = self.discard_pending.tolist()
        if any(count < 0 for count in pending):
            raise ValueError("discard_pending must be nonnegative")
        if self.phase == Phase.DISCARD:
            if self.last_roll != 7 or not any(pending):
                raise ValueError("DISCARD requires a seven and pending discards")
            for hand, required in zip(hands, pending):
                if required > 0:
                    total = sum(hand)
                    if total <= 7 or required != total // 2:
                        raise ValueError("pending discard counts disagree with hands")
        elif any(pending):
            raise ValueError("pending discards require DISCARD phase")
        if (
            self.last_roll is not None
            and not 2 <= _integer(self.last_roll, "last_roll") <= 12
        ):
            raise ValueError("last_roll must be None or a two-dice total")
        if not 0 <= _integer(self.setup_index, "setup_index") <= len(self.setup_order):
            raise ValueError("setup_index is outside the snake order")
        if self.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            if (
                self.setup_index == len(self.setup_order)
                or self.current_player != self.setup_order[self.setup_index]
            ):
                raise ValueError("setup player disagrees with the snake order")
        if self.phase == Phase.SETUP_ROAD:
            if (
                self.setup_vertex is None
                or not 0 <= _integer(self.setup_vertex, "setup_vertex") < 54
            ):
                raise ValueError("setup road requires a valid setup_vertex")
            if (
                self.vertex_owner[self.setup_vertex] != self.current_player
                or self.vertex_building[self.setup_vertex] != 1
            ):
                raise ValueError(
                    "setup road must attach to the current player's settlement"
                )
        elif self.setup_vertex is not None:
            raise ValueError("setup_vertex is only retained during SETUP_ROAD")
        if type(self.rng) is not Random:
            raise ValueError(
                "rng must be a standard random.Random for reproducible copies"
            )


def new_game(seed: int | None = None, n_players: int = 4) -> GameState:
    """Initialize an empty game with one RNG for board setup and future events.

    Supply an integer seed for repeatable games. None chooses system entropy.
    """
    n_players = _player_count(n_players)
    if seed is not None:
        seed = _integer(seed, "seed")
    rng = Random(seed)
    return GameState(make_board(rng), n_players=n_players, rng=rng)

"""Optional public economic summaries; no learning framework or state caches.

Production is gross entitlement with the current robber, before finite-bank
rationing. Site yields describe one settlement, not legality or strategic value.
The original 1505-float observation remains an unchanged prefix.
"""

from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from catan.board import Board, GEOMETRY, PortType, Resource
from catan.state import _ARRAY_FIELDS, _integer, _player_count
from rl.encoding import FlatEncoder, OBS_DIM, _readonly, decision_player
from rl.env import CatanEnv, REWARD_SCHEME

ECONOMIC_SCHEMA_VERSION = 4
ECONOMIC_DIM = 495
ECONOMIC_OBS_DIM = OBS_DIM + ECONOMIC_DIM
ROLLS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
PURCHASES = ("road", "settlement", "city", "development")
COSTS = _readonly(
    ((1, 0, 0, 1, 0), (1, 1, 1, 1, 0), (0, 0, 2, 0, 3), (0, 1, 1, 0, 1)),
    np.int16,
)
ECONOMIC_SLICES = MappingProxyType(
    {
        "production": slice(0, 200),  # relative player, resource, producing roll
        "shortfalls": slice(200, 220),  # purchase, resource
        "sites": slice(220, 490),  # vertex, resource; expected cards per roll
        "trade_ratios": slice(490, 495),  # observer's current 2/3/4-to-1 ratios
    }
)
_TILE_VERTICES = _readonly(GEOMETRY.tile_vertices, np.intp)


def _observer(state, player):
    if player is None:
        actor = decision_player(state)
        return state.current_player if actor is None else actor
    player = _integer(player, "player")
    if not 0 <= player < state.n_players:
        raise ValueError("player is outside the game")
    return player


def _feature_player(player):
    if player is None:
        return None
    player = _integer(player, "feature_player")
    if not 0 <= player < 4:
        raise ValueError("feature_player must be a seat from 0 to 3")
    return player


def _output(state, out, width):
    if out is None:
        return np.empty(width, dtype=np.float32)
    if (
        not isinstance(out, np.ndarray)
        or out.shape != (width,)
        or out.dtype != np.float32
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise ValueError(f"out must be writable contiguous float32 ({width},)")
    if any(np.may_share_memory(out, getattr(state, name)) for name in _ARRAY_FIELDS):
        raise ValueError("out must not share memory with state arrays")
    return out


@dataclass(frozen=True, slots=True, eq=False)
class EconomicFeatures:
    """Small immutable board lookups, rebuilt only when the board changes."""

    board: Board
    _tile_columns: tuple[int, ...] = field(init=False, repr=False)
    _site_yields: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.board, Board):
            raise TypeError("board must be a Board")
        columns = []
        sites = np.zeros((54, 5), dtype=np.float32)
        for tile, (resource, number) in enumerate(
            zip(self.board.resources, self.board.numbers, strict=True)
        ):
            if resource == Resource.DESERT:
                columns.append(-1)
                continue
            columns.append(resource * 10 + ROLLS.index(number))
            sites[_TILE_VERTICES[tile], resource] += (6 - abs(7 - number)) / 36
        object.__setattr__(self, "_tile_columns", tuple(columns))
        object.__setattr__(self, "_site_yields", _readonly(sites))

    def encode(self, state, player=None, *, out=None):
        if state.board is not self.board:
            raise ValueError("encoder belongs to a different board")
        player = _observer(state, player)
        out = _output(state, out, ECONOMIC_DIM)
        self._write(state, player, out)
        return out

    def _write(self, state, player, out):
        # Sparse occupied vertices beat tiny matrix kernels in measured gameplay
        # states. Convert once; reuse Python owners for the nine port endpoints.
        owners = state.vertex_owner.tolist()
        buildings = state.vertex_building.tolist()
        amounts = [[0] * 50 for _ in range(4)]
        for vertex, owner in enumerate(owners):
            if owner == -1:
                continue
            row = amounts[(owner - player) % state.n_players]
            for tile in GEOMETRY.vertex_tiles[vertex]:
                column = self._tile_columns[tile]
                if column != -1 and tile != state.robber_tile:
                    row[column] += buildings[vertex]
        production = out[:200].reshape(4, 50)
        production[:] = amounts
        production *= np.float32(1 / 19)
        shortfalls = out[200:220].reshape(4, 5)
        np.maximum(COSTS - state.hands[player], 0, out=shortfalls)
        shortfalls *= np.float32(1 / 3)
        sites = out[220:490].reshape(54, 5)
        np.copyto(sites, self._site_yields)
        robber = state.robber_tile
        resource = self.board.resources[robber]
        if resource != Resource.DESERT:
            sites[_TILE_VERTICES[robber], resource] -= (
                6 - abs(7 - self.board.numbers[robber])
            ) / 36
        ratios = out[490:495]
        ratios.fill(4)
        generic = False
        for port, (a, b) in zip(
            self.board.ports, self.board.geometry.port_vertices, strict=True
        ):
            if owners[a] != player and owners[b] != player:
                continue
            if port == PortType.GENERIC:
                generic = True
            else:
                ratios[port] = 2
        if generic:
            ratios[ratios == 4] = 3
        ratios *= np.float32(1 / 4)


class EconomicEncoder:
    """Original observation plus economic suffix; supports caller-owned buffers."""

    __slots__ = ("base", "features", "enabled", "feature_player")

    def __init__(self, board, *, enabled=True, feature_player=None):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        self.base = FlatEncoder(board)
        self.features = EconomicFeatures(board)
        self.enabled = enabled
        self.feature_player = _feature_player(feature_player)

    def encode(self, state, player=None, *, out=None):
        player = _observer(state, player)
        if self.feature_player is not None and self.feature_player >= state.n_players:
            raise ValueError("feature_player is outside the game")
        out = _output(state, out, ECONOMIC_OBS_DIM)
        self.base.encode(state, player, out=out[:OBS_DIM])
        if self.enabled and (
            self.feature_player is None or player == self.feature_player
        ):
            self.features._write(state, player, out[OBS_DIM:])
        else:
            out[OBS_DIM:].fill(0)
        return out


class EconomicEnv(CatanEnv):
    __slots__ = ("enabled", "feature_player")

    def __init__(self, *, enabled=True, feature_player=None, max_actions=5000):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        super().__init__(max_actions=max_actions)
        self.enabled = enabled
        self.feature_player = _feature_player(feature_player)

    def reset(self, seed=None, *, n_players=4):
        if self.feature_player is not None and self.feature_player >= _player_count(
            n_players
        ):
            raise ValueError("feature_player is outside the game")
        super().reset(seed, n_players=n_players)
        self._encoder = EconomicEncoder(
            self._state.board, enabled=self.enabled, feature_player=self.feature_player
        )
        return self.decision()


@dataclass(frozen=True, slots=True)
class EconomicEnvFactory:
    enabled: bool = True
    feature_player: int | None = None
    observation_dim = ECONOMIC_OBS_DIM
    observation_schema = ECONOMIC_SCHEMA_VERSION
    reward_scheme = REWARD_SCHEME

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a bool")
        object.__setattr__(self, "feature_player", _feature_player(self.feature_player))

    def __call__(self, *, max_actions):
        return EconomicEnv(
            enabled=self.enabled,
            feature_player=self.feature_player,
            max_actions=max_actions,
        )

    def for_observer(self, player):
        """Reference-opponent evaluation needs summaries only for its candidate."""
        return replace(self, feature_player=player)

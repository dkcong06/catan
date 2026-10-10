"""Immutable player choices; resource columns are wood, sheep, wheat, brick, ore."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from catan.state import _integer


@dataclass(frozen=True, slots=True)
class DiscardAction:
    """Choose exactly which resource cards one player returns to the bank.

    Normalize count sequences to an immutable tuple. The engine checks the seat,
    held resources, and required total against the current state.
    """

    player: int
    resources: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        player = _integer(self.player, "player")
        if player < 0:
            raise ValueError("player must be nonnegative")
        try:
            valid = (
                isinstance(self.resources, (Sequence, np.ndarray))
                and not isinstance(self.resources, (str, bytes))
                and len(self.resources) == 5
            )
        except TypeError as error:
            raise ValueError("resources must contain five integer counts") from error
        if not valid:
            raise ValueError("resources must contain five integer counts")
        counts = tuple(_integer(value, "resource count") for value in self.resources)
        if any(value < 0 or value > 19 for value in counts):
            raise ValueError("resource counts must be from 0 to 19")
        object.__setattr__(self, "player", player)
        object.__setattr__(self, "resources", counts)


def _bounded(value: int, name: str, limit: int) -> int:
    value = _integer(value, name)
    if not 0 <= value < limit:
        raise ValueError(f"{name} must be from 0 to {limit - 1}")
    return value


@dataclass(frozen=True, slots=True)
class SettlementAction:
    """Place a settlement during setup or the main phase."""

    vertex: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertex", _bounded(self.vertex, "vertex", 54))


@dataclass(frozen=True, slots=True)
class RoadAction:
    edge: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge", _bounded(self.edge, "edge", 72))


@dataclass(frozen=True, slots=True)
class CityAction:
    vertex: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertex", _bounded(self.vertex, "vertex", 54))


@dataclass(frozen=True, slots=True)
class RollAction:
    """Draw two dice from the game RNG; callers cannot select the total."""


@dataclass(frozen=True, slots=True)
class RobberMoveAction:
    tile: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile", _bounded(self.tile, "tile", 19))


@dataclass(frozen=True, slots=True)
class StealAction:
    victim: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "victim", _bounded(self.victim, "victim", 4))


@dataclass(frozen=True, slots=True)
class BankTradeAction:
    """Receive one card using the best available bank/port ratio."""

    give: int
    receive: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "give", _bounded(self.give, "give", 5))
        object.__setattr__(self, "receive", _bounded(self.receive, "receive", 5))
        if self.give == self.receive:
            raise ValueError("give and receive must be different resources")


@dataclass(frozen=True, slots=True)
class EndTurnAction:
    pass


Action = (
    SettlementAction
    | RoadAction
    | CityAction
    | RollAction
    | DiscardAction
    | RobberMoveAction
    | StealAction
    | BankTradeAction
    | EndTurnAction
)

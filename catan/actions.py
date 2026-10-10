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

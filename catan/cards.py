"""Base-game development card identities and finite supply; stdlib only."""

from enum import IntEnum


class DevelopmentCard(IntEnum):
    KNIGHT = 0
    VICTORY_POINT = 1
    ROAD_BUILDING = 2
    YEAR_OF_PLENTY = 3
    MONOPOLY = 4


DEVELOPMENT_SUPPLY = (14, 5, 2, 2, 2)

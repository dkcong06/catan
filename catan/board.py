from dataclasses import dataclass
from enum import IntEnum
from math import sqrt
from random import Random


CubeCoordinate = tuple[int, int, int]
VertexCoordinate = tuple[int, int]
IndexList = tuple[int, ...]
EdgeVertices = tuple[int, int]
Point2D = tuple[float, float]


class Resource(IntEnum):
    """Stable resource indices; the five producing resources precede the desert."""

    WOOD = 0
    SHEEP = 1
    WHEAT = 2
    BRICK = 3
    ORE = 4
    DESERT = 5


RESOURCE_BAG = (
    (Resource.WOOD,) * 4
    + (Resource.SHEEP,) * 4
    + (Resource.WHEAT,) * 4
    + (Resource.BRICK,) * 3
    + (Resource.ORE,) * 3
    + (Resource.DESERT,)
)
NUMBER_BAG = (2, 12) + (3, 4, 5, 6, 8, 9, 10, 11) * 2

_CUBE_DIRECTIONS: tuple[CubeCoordinate, ...] = (
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    (-1, 1, 0),
    (-1, 0, 1),
    (0, -1, 1),
)
_CORNER_OFFSETS: tuple[VertexCoordinate, ...] = (
    (1, -1),
    (1, 1),
    (0, 2),
    (-1, 1),
    (-1, -1),
    (0, -2),
)


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    """Static tables and unit-hex 2D positions; edge positions are midpoints."""

    tile_coordinates: tuple[CubeCoordinate, ...]
    tile_neighbors: tuple[IndexList, ...]
    vertex_coordinates: tuple[VertexCoordinate, ...]
    tile_vertices: tuple[IndexList, ...]
    vertex_tiles: tuple[IndexList, ...]
    vertex_neighbors: tuple[IndexList, ...]
    edge_vertices: tuple[EdgeVertices, ...]
    vertex_edges: tuple[IndexList, ...]
    tile_positions: tuple[Point2D, ...]
    vertex_positions: tuple[Point2D, ...]
    edge_positions: tuple[Point2D, ...]


def generate_tiles() -> list[CubeCoordinate]:
    """Return the 19 base-game tile positions as radius-two cube coordinates."""
    tiles = []

    for x in range(-2, 3):
        for y in range(max(-2, -x - 2), min(2, -x + 2) + 1):
            tiles.append((x, y, -x - y))

    return tiles


def generate_geometry() -> BoardGeometry:
    """Generate the immutable lookup tables for the base-game board."""
    tile_coordinates = tuple(generate_tiles())
    tile_ids = {
        coordinate: tile_id for tile_id, coordinate in enumerate(tile_coordinates)
    }
    tile_neighbors = tuple(
        tuple(
            sorted(
                tile_ids[neighbor]
                for dx, dy, dz in _CUBE_DIRECTIONS
                if (neighbor := (x + dx, y + dy, z + dz)) in tile_ids
            )
        )
        for x, y, z in tile_coordinates
    )

    # These scaled 2D coordinates make shared hex corners exact integer values.
    tile_corners = []
    vertex_coordinate_set = set()
    for x, _, z in tile_coordinates:
        center_x, center_y = 2 * x + z, 3 * z
        corners = tuple(
            (center_x + offset_x, center_y + offset_y)
            for offset_x, offset_y in _CORNER_OFFSETS
        )
        tile_corners.append(corners)
        vertex_coordinate_set.update(corners)

    vertex_coordinates = tuple(sorted(vertex_coordinate_set))
    vertex_ids = {
        coordinate: vertex_id for vertex_id, coordinate in enumerate(vertex_coordinates)
    }
    tile_vertices = tuple(
        tuple(vertex_ids[corner] for corner in corners) for corners in tile_corners
    )

    vertex_tiles = [[] for _ in vertex_coordinates]
    edge_vertex_set = set()
    for tile_id, vertices in enumerate(tile_vertices):
        for corner_id, vertex_id in enumerate(vertices):
            vertex_tiles[vertex_id].append(tile_id)
            next_vertex_id = vertices[(corner_id + 1) % len(vertices)]
            edge_vertex_set.add(tuple(sorted((vertex_id, next_vertex_id))))

    edge_vertices = tuple(sorted(edge_vertex_set))
    vertex_edges = [[] for _ in vertex_coordinates]
    vertex_neighbors = [[] for _ in vertex_coordinates]
    for edge_id, (first_vertex, second_vertex) in enumerate(edge_vertices):
        vertex_edges[first_vertex].append(edge_id)
        vertex_edges[second_vertex].append(edge_id)
        vertex_neighbors[first_vertex].append(second_vertex)
        vertex_neighbors[second_vertex].append(first_vertex)

    horizontal_scale = sqrt(3) / 2
    vertex_positions = tuple(
        (horizontal_scale * x, y / 2) for x, y in vertex_coordinates
    )

    return BoardGeometry(
        tile_coordinates=tile_coordinates,
        tile_neighbors=tile_neighbors,
        vertex_coordinates=vertex_coordinates,
        tile_vertices=tile_vertices,
        vertex_tiles=tuple(tuple(tiles) for tiles in vertex_tiles),
        vertex_neighbors=tuple(
            tuple(sorted(neighbors)) for neighbors in vertex_neighbors
        ),
        edge_vertices=edge_vertices,
        vertex_edges=tuple(tuple(edges) for edges in vertex_edges),
        tile_positions=tuple(
            (horizontal_scale * (2 * x + z), 1.5 * z) for x, _, z in tile_coordinates
        ),
        vertex_positions=vertex_positions,
        edge_positions=tuple(
            (
                (vertex_positions[first][0] + vertex_positions[second][0]) / 2,
                (vertex_positions[first][1] + vertex_positions[second][1]) / 2,
            )
            for first, second in edge_vertices
        ),
    )


GEOMETRY = generate_geometry()


@dataclass(frozen=True, slots=True)
class Board:
    """Immutable starting layout; token 0 means no token on the desert.

    ``robber_tile`` is the initial location. Its live position belongs in GameState.
    """

    resources: tuple[Resource, ...]
    numbers: tuple[int, ...]
    robber_tile: int
    geometry: BoardGeometry = GEOMETRY


def make_board(rng: Random) -> Board:
    """Shuffle a base-game layout using the caller's seeded RNG.

    Adjacent 6/8 tokens and ports are deferred, as permitted by the build guide.
    """
    resources = list(RESOURCE_BAG)
    tokens = list(NUMBER_BAG)
    rng.shuffle(resources)
    rng.shuffle(tokens)
    token_iterator = iter(tokens)
    numbers = tuple(
        0 if resource == Resource.DESERT else next(token_iterator)
        for resource in resources
    )
    return Board(tuple(resources), numbers, resources.index(Resource.DESERT))

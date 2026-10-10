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


class PortType(IntEnum):
    """Specific ports share resource indices; generic ports trade any resource."""

    WOOD = 0
    SHEEP = 1
    WHEAT = 2
    BRICK = 3
    ORE = 4
    GENERIC = 5

    @property
    def ratio(self) -> int:
        """Number of matching resource cards offered for one received card."""
        return 3 if self == PortType.GENERIC else 2

    @property
    def resource(self) -> Resource | None:
        """Required offered resource, or None for a generic port."""
        return None if self == PortType.GENERIC else Resource(self.value)


RESOURCE_BAG = (
    (Resource.WOOD,) * 4
    + (Resource.SHEEP,) * 4
    + (Resource.WHEAT,) * 4
    + (Resource.BRICK,) * 3
    + (Resource.ORE,) * 3
    + (Resource.DESERT,)
)
NUMBER_BAG = (2, 12) + (3, 4, 5, 6, 8, 9, 10, 11) * 2
PORT_BAG = (PortType.GENERIC,) * 4 + tuple(PortType(i) for i in range(5))
_RED_NUMBERS = (6, 6, 8, 8)
_OTHER_NUMBERS = tuple(number for number in NUMBER_BAG if number not in (6, 8))
# Nine slots on the 30-edge coast, with cyclic edge gaps 3, 3, 4 repeated.
_PORT_COAST_OFFSETS = (0, 3, 6, 10, 13, 16, 20, 23, 26)

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
    coast_edges: IndexList
    port_edges: IndexList
    port_vertices: tuple[EdgeVertices, ...]
    vertex_ports: IndexList
    port_positions: tuple[Point2D, ...]

    def __copy__(self) -> "BoardGeometry":
        """All geometry fields are immutable, so copies can reuse this object."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "BoardGeometry":
        return self


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
    tile_positions = tuple(
        (horizontal_scale * (2 * x + z), 1.5 * z) for x, _, z in tile_coordinates
    )
    edge_positions = tuple(
        (
            (vertex_positions[first][0] + vertex_positions[second][0]) / 2,
            (vertex_positions[first][1] + vertex_positions[second][1]) / 2,
        )
        for first, second in edge_vertices
    )

    # Walk the boundary in a stable order; no angles or floating-point sorting.
    coast_neighbors: dict[int, list[tuple[int, int]]] = {}
    coast_tiles = {}
    for edge_id, (first, second) in enumerate(edge_vertices):
        shared_tiles = set(vertex_tiles[first]).intersection(vertex_tiles[second])
        if len(shared_tiles) == 1:
            coast_tiles[edge_id] = next(iter(shared_tiles))
            coast_neighbors.setdefault(first, []).append((second, edge_id))
            coast_neighbors.setdefault(second, []).append((first, edge_id))
    start = min(coast_neighbors)
    previous, current = -1, start
    coast_edges = []
    while True:
        following, edge_id = min(
            pair for pair in coast_neighbors[current] if pair[0] != previous
        )
        coast_edges.append(edge_id)
        previous, current = current, following
        if current == start:
            break

    port_edges = tuple(coast_edges[offset] for offset in _PORT_COAST_OFFSETS)
    port_vertices = tuple(edge_vertices[edge] for edge in port_edges)
    vertex_ports = [-1] * len(vertex_coordinates)
    port_positions = []
    for port_id, edge_id in enumerate(port_edges):
        for vertex in port_vertices[port_id]:
            vertex_ports[vertex] = port_id
        midpoint = edge_positions[edge_id]
        center = tile_positions[coast_tiles[edge_id]]
        port_positions.append(
            tuple(mid + 0.85 * (mid - origin) for mid, origin in zip(midpoint, center))
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
        tile_positions=tile_positions,
        vertex_positions=vertex_positions,
        edge_positions=edge_positions,
        coast_edges=tuple(coast_edges),
        port_edges=port_edges,
        port_vertices=port_vertices,
        vertex_ports=tuple(vertex_ports),
        port_positions=tuple(port_positions),
    )


GEOMETRY = generate_geometry()


def _red_tile_sets_by_desert() -> tuple[tuple[IndexList, ...], ...]:
    """Cache every independent four-tile set, grouped by excluded desert tile."""
    count = len(GEOMETRY.tile_coordinates)
    neighbors = tuple(frozenset(row) for row in GEOMETRY.tile_neighbors)
    # Prune illegal prefixes rather than constructing every four-tile candidate.
    # Ascending loops preserve the original lexicographic order and seeded layouts.
    red_sets = tuple(
        (a, b, c, d)
        for a in range(count - 3)
        for b in range(a + 1, count - 2)
        if b not in neighbors[a]
        for c in range(b + 1, count - 1)
        if c not in neighbors[a] and c not in neighbors[b]
        for d in range(c + 1, count)
        if d not in neighbors[a] and d not in neighbors[b] and d not in neighbors[c]
    )
    return tuple(
        tuple(tiles for tiles in red_sets if desert not in tiles)
        for desert in range(count)
    )


_RED_TILE_SETS_BY_DESERT = _red_tile_sets_by_desert()


@dataclass(frozen=True, slots=True)
class Board:
    """Immutable starting layout; token 0 means no token on the desert.

    ``robber_tile`` is the initial location. Its live position belongs in GameState.
    """

    resources: tuple[Resource, ...]
    numbers: tuple[int, ...]
    robber_tile: int
    ports: tuple[PortType, ...]
    geometry: BoardGeometry = GEOMETRY

    def __copy__(self) -> "Board":
        """Reuse the immutable layout, including when nested in copied states."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "Board":
        return self


def make_board(rng: Random) -> Board:
    """Sample a reproducible layout with no adjacent 6/8s and nine coastal ports.

    Legal red-token sets are sampled uniformly without rejection or board repairs.
    All randomness advances the caller's RNG; no global RNG is used.
    """
    resources = list(RESOURCE_BAG)
    rng.shuffle(resources)
    desert = resources.index(Resource.DESERT)
    red_tiles = rng.choice(_RED_TILE_SETS_BY_DESERT[desert])
    red_numbers = list(_RED_NUMBERS)
    rng.shuffle(red_numbers)
    tokens = list(_OTHER_NUMBERS)
    rng.shuffle(tokens)
    numbers = [0] * len(resources)
    for tile, number in zip(red_tiles, red_numbers):
        numbers[tile] = number
    token_iterator = iter(tokens)
    for tile in range(len(resources)):
        if tile != desert and not numbers[tile]:
            numbers[tile] = next(token_iterator)
    ports = list(PORT_BAG)
    rng.shuffle(ports)
    return Board(tuple(resources), tuple(numbers), desert, tuple(ports))

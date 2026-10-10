from collections import Counter
from dataclasses import FrozenInstanceError
from math import dist
from random import Random
from itertools import combinations
from hashlib import sha256
import json

import pytest

from catan.board import (
    GEOMETRY,
    NUMBER_BAG,
    PORT_BAG,
    RESOURCE_BAG,
    PortType,
    Resource,
    _RED_TILE_SETS_BY_DESERT,
    generate_geometry,
    make_board,
)


def test_geometry_has_standard_base_game_counts() -> None:
    assert len(GEOMETRY.tile_coordinates) == 19
    assert len(GEOMETRY.vertex_coordinates) == 54
    assert len(GEOMETRY.edge_vertices) == 72


def test_lookup_tables_are_reciprocal() -> None:
    for tile_id, vertices in enumerate(GEOMETRY.tile_vertices):
        assert len(vertices) == 6
        for vertex_id in vertices:
            assert tile_id in GEOMETRY.vertex_tiles[vertex_id]

    for edge_id, (first_vertex, second_vertex) in enumerate(GEOMETRY.edge_vertices):
        assert edge_id in GEOMETRY.vertex_edges[first_vertex]
        assert edge_id in GEOMETRY.vertex_edges[second_vertex]
        assert second_vertex in GEOMETRY.vertex_neighbors[first_vertex]
        assert first_vertex in GEOMETRY.vertex_neighbors[second_vertex]


def test_vertex_and_tile_adjacency_is_valid() -> None:
    for vertex_id, adjacent_tiles in enumerate(GEOMETRY.vertex_tiles):
        assert 1 <= len(adjacent_tiles) <= 3
        for neighbor in GEOMETRY.vertex_neighbors[vertex_id]:
            assert vertex_id in GEOMETRY.vertex_neighbors[neighbor]

    for tile_id, neighbors in enumerate(GEOMETRY.tile_neighbors):
        for neighbor in neighbors:
            assert tile_id in GEOMETRY.tile_neighbors[neighbor]


def test_geometry_generation_is_deterministic() -> None:
    assert generate_geometry() == GEOMETRY


def test_geometry_ids_and_positions_remain_compatible() -> None:
    # Fingerprint the eleven original tables before optimization, including ID order.
    names = (
        "tile_coordinates",
        "tile_neighbors",
        "vertex_coordinates",
        "tile_vertices",
        "vertex_tiles",
        "vertex_neighbors",
        "edge_vertices",
        "vertex_edges",
        "tile_positions",
        "vertex_positions",
        "edge_positions",
    )
    encoded = json.dumps(
        [getattr(GEOMETRY, name) for name in names], separators=(",", ":")
    )
    assert (
        sha256(encoded.encode()).hexdigest()
        == "f4a5575d6c1a2815c5a97da745969e6fe515078affa4cc13b2a3e572f39895b9"
    )


def test_geometry_is_one_connected_board_without_holes() -> None:
    reached = {0}
    pending = [0]
    while pending:
        for neighbor in GEOMETRY.vertex_neighbors[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    assert reached == set(range(54))
    assert Counter(map(len, GEOMETRY.vertex_neighbors)) == {2: 18, 3: 36}
    assert (
        len(GEOMETRY.vertex_coordinates)
        - len(GEOMETRY.edge_vertices)
        + len(GEOMETRY.tile_coordinates)
        == 1
    )  # Euler characteristic of the bounded land faces.


def test_2d_positions_form_regular_hexagons_and_unit_edges() -> None:
    assert len(GEOMETRY.tile_positions) == 19
    assert len(GEOMETRY.vertex_positions) == 54
    assert len(GEOMETRY.edge_positions) == 72
    for tile_id, vertices in enumerate(GEOMETRY.tile_vertices):
        center = GEOMETRY.tile_positions[tile_id]
        for vertex in vertices:
            assert dist(center, GEOMETRY.vertex_positions[vertex]) == pytest.approx(1)
    for edge_id, (first, second) in enumerate(GEOMETRY.edge_vertices):
        start, end = GEOMETRY.vertex_positions[first], GEOMETRY.vertex_positions[second]
        assert dist(start, end) == pytest.approx(1)
        midpoint = GEOMETRY.edge_positions[edge_id]
        assert dist(start, midpoint) == pytest.approx(0.5)
        assert dist(end, midpoint) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(100))
def test_board_setup_has_base_game_composition(seed: int) -> None:
    board = make_board(Random(seed))
    assert len(board.resources) == len(board.numbers) == 19
    assert Counter(board.resources) == {
        Resource.WOOD: 4,
        Resource.SHEEP: 4,
        Resource.WHEAT: 4,
        Resource.BRICK: 3,
        Resource.ORE: 3,
        Resource.DESERT: 1,
    }
    assert Counter(number for number in board.numbers if number) == {
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        12: 1,
    }
    assert board.resources[board.robber_tile] == Resource.DESERT
    assert [i for i, number in enumerate(board.numbers) if number == 0] == [
        board.robber_tile
    ]
    assert board.geometry is GEOMETRY
    assert Counter(board.ports) == Counter(PORT_BAG)
    for tile, number in enumerate(board.numbers):
        if number in (6, 8):
            assert all(
                board.numbers[neighbor] not in (6, 8)
                for neighbor in GEOMETRY.tile_neighbors[tile]
            )


def test_seed_replays_board_and_preserves_rng_continuation() -> None:
    first_rng, second_rng = Random(42), Random(42)
    assert make_board(first_rng) == make_board(second_rng)
    assert first_rng.getstate() == second_rng.getstate()
    # A shared RNG advances across successive games rather than resetting each one.
    assert make_board(first_rng) != make_board(Random(42))
    assert make_board(Random(1)) != make_board(Random(2))


def test_optimization_preserves_a_thousand_board_rng_stream() -> None:
    rng = Random(42)
    layouts = []
    for _ in range(1000):
        board = make_board(rng)
        layouts.append((board.resources, board.numbers, board.robber_tile, board.ports))
    encoded = json.dumps((layouts, rng.getstate()), separators=(",", ":"))
    assert (
        sha256(encoded.encode()).hexdigest()
        == "a009aa8b3601b6ed917be01c3b06b5f9d1473d245bb242101082be84569dc505"
    )


def test_board_layout_is_immutable() -> None:
    board = make_board(Random(0))
    with pytest.raises(FrozenInstanceError):
        board.robber_tile = 0
    with pytest.raises(TypeError):
        board.numbers[0] = 7
    with pytest.raises(TypeError):
        board.ports[0] = PortType.GENERIC


def test_ports_cover_nine_separated_coastal_edges() -> None:
    geometry = GEOMETRY
    assert len(geometry.coast_edges) == len(set(geometry.coast_edges)) == 30
    expected_coast = {
        edge
        for edge, (first, second) in enumerate(geometry.edge_vertices)
        if len(set(geometry.vertex_tiles[first]) & set(geometry.vertex_tiles[second]))
        == 1
    }
    assert set(geometry.coast_edges) == expected_coast
    for index, edge in enumerate(geometry.coast_edges):
        following = geometry.coast_edges[(index + 1) % 30]
        assert (
            len(
                set(geometry.edge_vertices[edge])
                & set(geometry.edge_vertices[following])
            )
            == 1
        )
    assert len(geometry.port_edges) == len(set(geometry.port_edges)) == 9
    assert len(geometry.port_positions) == len(geometry.port_vertices) == 9
    port_vertices = {vertex for pair in geometry.port_vertices for vertex in pair}
    assert len(port_vertices) == 18
    assert len(geometry.vertex_ports) == 54
    for port_id, edge in enumerate(geometry.port_edges):
        assert edge in expected_coast
        assert geometry.port_vertices[port_id] == geometry.edge_vertices[edge]
        first, second = geometry.edge_vertices[edge]
        (adjacent_tile,) = set(geometry.vertex_tiles[first]) & set(
            geometry.vertex_tiles[second]
        )
        position = geometry.port_positions[port_id]
        # Labels are outside the land, beyond their own edge midpoint.
        assert dist(position, geometry.tile_positions[adjacent_tile]) > 1
        for vertex in (first, second):
            assert geometry.vertex_ports[vertex] == port_id
    for vertex, port_id in enumerate(geometry.vertex_ports):
        assert (port_id == -1) == (vertex not in port_vertices)


def test_port_types_and_trade_metadata() -> None:
    assert Counter(PORT_BAG) == {
        PortType.GENERIC: 4,
        **{PortType(i): 1 for i in range(5)},
    }
    for port in PortType:
        assert port.ratio == (3 if port == PortType.GENERIC else 2)
        assert port.resource == (None if port == PortType.GENERIC else Resource(port))


def test_red_token_cache_contains_every_legal_set_for_every_desert() -> None:
    # Independently derive adjacency from shared sides, rather than the sampler table.
    touching = {
        (first, second)
        for first, second in combinations(range(19), 2)
        if len(set(GEOMETRY.tile_vertices[first]) & set(GEOMETRY.tile_vertices[second]))
        == 2
    }
    for desert, cached in enumerate(_RED_TILE_SETS_BY_DESERT):
        expected = {
            tiles
            for tiles in combinations((t for t in range(19) if t != desert), 4)
            if all(pair not in touching for pair in combinations(tiles, 2))
        }
        assert expected
        assert len(cached) == len(set(cached))
        assert set(cached) == expected
        assert cached == tuple(sorted(expected))


def test_ten_thousand_seeded_boards_obey_setup_rules() -> None:
    deserts = set()
    port_layouts = set()
    for seed in range(10_000):
        board = make_board(Random(seed))
        deserts.add(board.robber_tile)
        port_layouts.add(board.ports)
        assert Counter(board.resources) == Counter(RESOURCE_BAG)
        assert Counter(token for token in board.numbers if token) == Counter(NUMBER_BAG)
        assert Counter(board.ports) == Counter(PORT_BAG)
        assert board.resources[board.robber_tile] == Resource.DESERT
        assert [t for t, number in enumerate(board.numbers) if number == 0] == [
            board.robber_tile
        ]
        for tile, number in enumerate(board.numbers):
            if number in (6, 8):
                assert all(
                    board.numbers[n] not in (6, 8)
                    for n in GEOMETRY.tile_neighbors[tile]
                )
    assert deserts == set(range(19))
    assert len(port_layouts) > 1

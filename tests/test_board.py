from collections import Counter
from dataclasses import FrozenInstanceError
from math import dist
from random import Random

import pytest

from catan.board import GEOMETRY, Resource, generate_geometry, make_board


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


def test_seed_replays_board_and_preserves_rng_continuation() -> None:
    first_rng, second_rng = Random(42), Random(42)
    assert make_board(first_rng) == make_board(second_rng)
    assert first_rng.getstate() == second_rng.getstate()
    # A shared RNG advances across successive games rather than resetting each one.
    assert make_board(first_rng) != make_board(Random(42))
    assert make_board(Random(1)) != make_board(Random(2))


def test_board_layout_is_immutable() -> None:
    board = make_board(Random(0))
    with pytest.raises(FrozenInstanceError):
        board.robber_tile = 0
    with pytest.raises(TypeError):
        board.numbers[0] = 7

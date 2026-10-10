import sys
from copy import copy, deepcopy
from dataclasses import fields
from random import Random, SystemRandom, getstate
from subprocess import run

import numpy as np
import pytest

from catan.board import GEOMETRY, PortType, make_board
from catan.state import GameState, Phase, new_game


@pytest.mark.parametrize("players", (3, 4))
def test_new_game_initializes_every_quantity(players: int) -> None:
    state = new_game(42, players)
    expected = {
        "hands": (np.int16, np.zeros((players, 5), dtype=np.int16)),
        "vp": (np.int16, np.zeros(players, dtype=np.int16)),
        "pieces_left": (np.int8, np.array([[5, 4, 15]] * players, dtype=np.int8)),
        "vertex_owner": (np.int8, np.full(54, -1, dtype=np.int8)),
        "vertex_building": (np.int8, np.zeros(54, dtype=np.int8)),
        "edge_owner": (np.int8, np.full(72, -1, dtype=np.int8)),
        "bank": (np.int16, np.full(5, 19, dtype=np.int16)),
        "discard_pending": (np.int16, np.zeros(players, dtype=np.int16)),
    }
    for name, (dtype, values) in expected.items():
        actual = getattr(state, name)
        assert actual.dtype == dtype
        assert actual.shape == values.shape
        assert actual.flags.c_contiguous and actual.flags.writeable
        np.testing.assert_array_equal(actual, values)
    assert state.n_players == players
    assert state.current_player == state.setup_index == 0
    assert state.setup_vertex is state.last_roll is None
    assert state.phase == Phase.SETUP_SETTLEMENT
    assert state.robber_tile == state.board.robber_tile
    assert state.board.geometry is GEOMETRY
    assert state.array_bytes == sum(values.nbytes for _, values in expected.values())
    assert not hasattr(state, "__dict__")
    state.validate()


@pytest.mark.parametrize("players", (3, 4))
def test_seeded_initialization_preserves_the_board_rng_stream(players: int) -> None:
    rng = Random(42)
    expected_board = make_board(rng)
    state = new_game(42, players)
    assert state.board == expected_board
    assert state.rng.getstate() == rng.getstate()
    assert [state.rng.randint(1, 6) for _ in range(100)] == [
        rng.randint(1, 6) for _ in range(100)
    ]
    assert new_game(42, players) == new_game(42, players)
    assert new_game(42, players) != new_game(43, players)
    assert new_game(42, 3) != new_game(42, 4)


def test_constructor_uses_supplied_rng_without_drawing_or_cloning() -> None:
    rng = Random(0)
    board = make_board(rng)
    before = rng.getstate()
    state = GameState(board, rng=rng)
    assert state.rng is rng
    assert state.board is board
    assert rng.getstate() == before
    with pytest.raises(TypeError, match="rng"):
        GameState(board, rng=None)
    with pytest.raises(TypeError, match="rng"):
        GameState(board, rng=SystemRandom())


@pytest.mark.parametrize("players", (0, 1, 2, 5, -1, 4.0, "4", True, None))
def test_invalid_player_counts_are_rejected(players) -> None:
    with pytest.raises(ValueError, match="n_players"):
        new_game(0, players)
    with pytest.raises(ValueError, match="n_players"):
        GameState(make_board(Random(0)), players)


@pytest.mark.parametrize("seed", (True, 1.5, "42", np.bool_(False)))
def test_noninteger_seeds_are_rejected(seed) -> None:
    with pytest.raises(ValueError, match="seed"):
        new_game(seed)


def test_integer_like_values_and_entropy_seed_are_supported() -> None:
    assert new_game(np.int64(42), np.int64(3)) == new_game(42, 3)
    new_game(None).validate()


@pytest.mark.parametrize("clone_state", (GameState.copy, copy, deepcopy))
@pytest.mark.parametrize("players", (3, 4))
def test_copies_preserve_all_fields_and_isolate_every_mutable_array(
    clone_state, players
) -> None:
    original = new_game(42, players)
    original.rng.gauss(0, 1)  # Include the RNG's cached Gaussian in branch copies.
    cloned = clone_state(original)
    assert cloned is not original and cloned == original
    assert cloned.board is original.board
    assert cloned.board.geometry is original.board.geometry
    assert cloned.rng is not original.rng
    for descriptor in fields(original):
        before, after = (
            getattr(original, descriptor.name),
            getattr(cloned, descriptor.name),
        )
        if isinstance(before, np.ndarray):
            assert not np.shares_memory(before, after)
            after.flat[0] = before.flat[0] + 1
            assert before.flat[0] != after.flat[0]
        elif descriptor.name != "rng":
            assert before == after
    cloned.current_player = 1
    cloned.robber_tile = (original.robber_tile + 1) % 19
    assert original.current_player == 0
    assert original.robber_tile == original.board.robber_tile
    original.validate()


def test_copied_rngs_replay_mixed_draws_without_advancing_the_parent() -> None:
    parent = new_game(7)
    parent.rng.gauss(0, 1)
    initial_rng = parent.rng.getstate()
    first, second = parent.copy(), deepcopy(parent)
    for _ in range(50):
        assert first.rng.randint(1, 6) == second.rng.randint(1, 6)
        assert first.rng.randrange(95) == second.rng.randrange(95)
        assert first.rng.gauss(0, 1) == second.rng.gauss(0, 1)
    assert first == second
    assert parent.rng.getstate() == initial_rng
    first.rng.random()
    assert first != second


@pytest.mark.parametrize("state_first", (False, True))
def test_deepcopy_preserves_aliases_inside_composite_payloads(
    state_first: bool,
) -> None:
    state = new_game(0)
    references = [
        state.board,
        state.board.geometry,
        state.hands,
        state.rng,
        state.discard_pending,
    ]
    source = [state, *references] if state_first else [*references, state]
    cloned = deepcopy(source)
    branch = cloned[0] if state_first else cloned[-1]
    reference_copies = cloned[1:] if state_first else cloned[:-1]
    assert branch == state
    assert reference_copies[0] is branch.board is state.board
    assert reference_copies[1] is branch.board.geometry is state.board.geometry
    assert reference_copies[2] is branch.hands
    assert reference_copies[3] is branch.rng
    assert reference_copies[4] is branch.discard_pending
    assert not np.shares_memory(branch.discard_pending, state.discard_pending)
    assert not np.shares_memory(branch.hands, state.hands)
    assert branch.rng is not state.rng


@pytest.mark.parametrize(
    "name",
    (
        "hands",
        "vp",
        "pieces_left",
        "vertex_owner",
        "vertex_building",
        "edge_owner",
        "bank",
        "discard_pending",
    ),
)
def test_array_changes_affect_state_equality(name: str) -> None:
    first = new_game(42)
    second = first.copy()
    getattr(second, name).flat[0] += 1
    assert first != second


@pytest.mark.parametrize(
    "name, value",
    (
        ("current_player", 1),
        ("robber_tile", -1),
        ("phase", Phase.ROLL),
        ("last_roll", 8),
        ("setup_index", 1),
        ("setup_vertex", 0),
    ),
)
def test_scalar_changes_affect_equality_and_survive_copy(name: str, value) -> None:
    first = new_game(42)
    second = first.copy()
    setattr(second, name, value)
    assert first != second
    assert second == second.copy()
    assert getattr(second.copy(), name) == value


def test_equality_checks_board_dtype_and_unrelated_objects() -> None:
    first = new_game(42)
    second = first.copy()
    second.board = make_board(Random(43))
    assert first != second
    second = first.copy()
    second.hands = second.hands.astype(np.int32)
    assert first != second
    assert first != object()
    with pytest.raises(TypeError):
        hash(first)


def test_equality_retains_shape_strided_and_noninteger_semantics() -> None:
    first = new_game(42)
    second = first.copy()
    second.hands = second.hands.reshape(-1)
    assert first != second  # Same bytes do not imply the same shaped values.
    second = first.copy()
    backing = np.zeros((4, 10), dtype=np.int16)
    second.hands = backing[:, ::2]
    assert not second.hands.flags.c_contiguous
    assert first == second
    second.hands[3, 4] = 1
    assert first != second
    first.hands = first.hands.astype(float)
    second.hands = first.hands.copy()
    second.hands[0, 0] = -0.0
    assert first == second
    first.hands[0, 0] = second.hands[0, 0] = np.nan
    assert first != second  # NumPy's default NaN semantics must be preserved.


@pytest.mark.parametrize("players", (3, 4))
def test_varied_populated_states_validate_and_isolate_branches(players: int) -> None:
    for seed in range(100):
        state = new_game(seed, players)
        fixture_rng = Random(seed)
        for resource in range(5):
            for _ in range(19):
                seat = fixture_rng.randrange(players + 1)
                if seat < players:
                    state.hands[seat, resource] += 1
                    state.bank[resource] -= 1
        available = set(range(54))
        for seat in range(players):
            for building in (1, 2):
                vertex = fixture_rng.choice(sorted(available))
                add_building(state, seat, vertex, building)
                available.difference_update(
                    (vertex, *GEOMETRY.vertex_neighbors[vertex])
                )
        edges = fixture_rng.sample(range(72), players * 5)
        for seat in range(players):
            state.edge_owner[edges[seat * 5 : (seat + 1) * 5]] = seat
            state.pieces_left[seat, 2] -= 5
        state.phase = Phase.MAIN
        state.setup_index = len(state.setup_order)
        state.current_player = seed % players
        state.last_roll = fixture_rng.randint(2, 12)
        state.robber_tile = fixture_rng.randrange(19)
        state.rng.gauss(0, 1)
        state.validate()
        branch = deepcopy(state)
        branch.validate()
        assert branch == state
        for seat in range(players):
            assert branch.port_access(seat) == state.port_access(seat)
        saved = state.copy()
        # Transfer a held card back to the bank only on the branch.
        seat, resource = np.argwhere(branch.hands > 0)[0]
        branch.hands[seat, resource] -= 1
        branch.bank[resource] += 1
        branch.rng.random()
        branch.validate()
        assert state == saved and branch != state


@pytest.mark.parametrize(
    "players, order", ((3, (0, 1, 2, 2, 1, 0)), (4, (0, 1, 2, 3, 3, 2, 1, 0)))
)
def test_setup_snake_order(players: int, order: tuple[int, ...]) -> None:
    state = new_game(0, players)
    assert state.setup_order == order
    assert state.setup_order is state.copy().setup_order


def add_building(state: GameState, player: int, vertex: int, building: int = 1) -> None:
    state.vertex_owner[vertex] = player
    state.vertex_building[vertex] = building
    state.pieces_left[player, building - 1] -= 1
    state.vp[player] += building


@pytest.mark.parametrize("port_id", range(9))
@pytest.mark.parametrize("endpoint", (0, 1))
@pytest.mark.parametrize("building", (1, 2))
def test_either_port_endpoint_grants_access_only_to_its_owner(
    port_id: int, endpoint: int, building: int
) -> None:
    state = new_game(42)
    assert state.port_access(0) == ()
    vertex = state.board.geometry.port_vertices[port_id][endpoint]
    add_building(state, 0, vertex, building)
    assert state.port_access(0) == (state.board.ports[port_id],)
    assert state.port_access(1) == ()
    state.validate()


def test_roads_and_nonport_vertices_do_not_grant_port_access() -> None:
    state = new_game(0)
    edge = GEOMETRY.port_edges[0]
    state.edge_owner[edge] = 0
    state.pieces_left[0, 2] -= 1
    assert state.port_access(0) == ()
    vertex = GEOMETRY.vertex_ports.index(-1)
    add_building(state, 0, vertex)
    assert state.port_access(0) == ()
    state.validate()


def test_duplicate_ports_return_one_type_and_access_tracks_copies() -> None:
    state = new_game(42)
    generic = [
        i for i, port in enumerate(state.board.ports) if port == PortType.GENERIC
    ]
    for port_id in generic[:2]:
        add_building(state, 0, GEOMETRY.port_vertices[port_id][0])
    assert state.port_access(0) == (PortType.GENERIC,)
    cloned = state.copy()
    vertex = GEOMETRY.port_vertices[state.board.ports.index(PortType.WOOD)][0]
    add_building(cloned, 0, vertex)
    assert cloned.port_access(0) == (PortType.WOOD, PortType.GENERIC)
    assert state.port_access(0) == (PortType.GENERIC,)
    cloned.validate()
    state.validate()


@pytest.mark.parametrize("player", (-1, 4, 1.0, True))
def test_invalid_port_query_player_is_rejected(player) -> None:
    with pytest.raises(ValueError, match="player"):
        new_game(0).port_access(player)


@pytest.mark.parametrize(
    "name, value, message",
    (
        ("robber_tile", 19, "robber_tile"),
        ("current_player", -1, "current_player"),
        ("phase", 0, "phase"),
        ("last_roll", 13, "last_roll"),
        ("setup_index", 9, "setup_index"),
        ("setup_vertex", 0, "setup_vertex"),
        ("rng", None, "rng"),
    ),
)
def test_scalar_invariant_failures_are_reported(name: str, value, message: str) -> None:
    state = new_game(0)
    setattr(state, name, value)
    with pytest.raises(ValueError, match=message):
        state.validate()


def test_validation_accepts_conserved_resources_buildings_and_setup_road() -> None:
    state = new_game(0)
    state.hands[0] = [1, 2, 3, 4, 5]
    state.bank -= state.hands[0]
    vertex = 0
    add_building(state, 0, vertex)
    state.phase = Phase.SETUP_ROAD
    state.setup_vertex = vertex
    state.validate()
    state.robber_tile = (state.robber_tile + 1) % 19
    state.phase = Phase.MAIN
    state.setup_index = len(state.setup_order)
    state.setup_vertex = None
    state.last_roll = 7
    state.validate()


@pytest.mark.parametrize(
    "failure, message",
    (
        ("resources", "conserved"),
        ("negative", "nonnegative"),
        ("pieces", "pieces disagree"),
        ("owner", "invalid owner"),
        ("building", "invalid building"),
        ("ownership", "ownership"),
        ("vp", "victory points"),
        ("distance", "distance rule"),
        ("setup_player", "snake order"),
        ("setup_road", "setup_vertex"),
        ("setup_owner", "current player's settlement"),
        ("dtype", "dtype"),
        ("shape", "shape"),
        ("readonly", "writable"),
    ),
)
def test_validation_rejects_corrupted_state(failure: str, message: str) -> None:
    state = new_game(0)
    if failure == "resources":
        state.bank[0] -= 1
    elif failure == "negative":
        state.hands[0, 0] = -1
    elif failure == "pieces":
        state.pieces_left[0, 0] -= 1
    elif failure == "owner":
        state.edge_owner[0] = 4
    elif failure == "building":
        state.vertex_building[0] = 3
    elif failure == "ownership":
        state.vertex_owner[0] = 0
    elif failure == "vp":
        state.vp[0] = 1
    elif failure == "distance":
        for vertex in GEOMETRY.edge_vertices[0]:
            add_building(state, 0, vertex)
    elif failure == "setup_player":
        state.current_player = 1
    elif failure == "setup_road":
        state.phase = Phase.SETUP_ROAD
    elif failure == "setup_owner":
        state.phase = Phase.SETUP_ROAD
        state.setup_vertex = 0
    elif failure == "dtype":
        state.hands = state.hands.astype(np.int32)
    elif failure == "shape":
        state.bank = np.zeros(4, dtype=np.int16)
    elif failure == "readonly":
        state.hands.flags.writeable = False
    with pytest.raises(ValueError, match=message):
        state.validate()


def test_seeded_states_and_copies_preserve_invariants_and_global_rng() -> None:
    global_rng = getstate()
    for players in (3, 4):
        for seed in range(500):
            state = new_game(seed, players)
            state.validate()
            cloned = state.copy()
            cloned.validate()
            assert state == cloned
            for tile, number in enumerate(state.board.numbers):
                if number in (6, 8):
                    assert all(
                        state.board.numbers[n] not in (6, 8)
                        for n in GEOMETRY.tile_neighbors[tile]
                    )
    assert getstate() == global_rng


def test_board_and_viewer_imports_remain_free_of_numpy_and_torch() -> None:
    result = run(
        [
            sys.executable,
            "-c",
            "import catan.board, scripts.board_viewer, sys; assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

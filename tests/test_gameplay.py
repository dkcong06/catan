from dataclasses import fields
from itertools import islice, product
from random import Random, getstate

import numpy as np
import pytest

from catan.actions import (
    BankTradeAction,
    CityAction,
    DiscardAction,
    EndTurnAction,
    RobberMoveAction,
    RoadAction,
    RollAction,
    SettlementAction,
    StealAction,
)
from catan.board import Resource
from catan.engine import (
    apply_action,
    apply_action_inplace,
    is_terminal,
    legal_actions,
    winner,
)
from catan.players.random_agent import choose_action, play_random_game
from catan.state import GameState, Phase, new_game


def empty_main(players=4):
    state = new_game(42, players)
    state.setup_index = len(state.setup_order)
    state.phase = Phase.MAIN
    return state


def building(state, vertex, player, code=1):
    state.vertex_owner[vertex] = player
    state.vertex_building[vertex] = code
    state.pieces_left[player, code - 1] -= 1
    state.vp[player] += code


def road(state, edge, player):
    state.edge_owner[edge] = player
    state.pieces_left[player, 2] -= 1


def fund(state, player, counts):
    state.hands[player] += counts
    state.bank -= counts


def rejected(state, action):
    before = state.copy()
    identities = {f.name: id(getattr(state, f.name)) for f in fields(GameState)}
    for step in (apply_action, apply_action_inplace):
        with pytest.raises((ValueError, TypeError)):
            step(state, action)
        assert state == before
        assert identities == {
            f.name: id(getattr(state, f.name)) for f in fields(GameState)
        }


def roll_total(state, total):
    for seed in range(1000):
        rng = Random(seed)
        if rng.randint(1, 6) + rng.randint(1, 6) == total:
            state.rng.seed(seed)
            state.phase = Phase.ROLL
            expected_rng = rng.getstate()
            following = apply_action(state, RollAction())
            assert following.last_roll == total
            assert following.rng.getstate() == expected_rng
            return following
    pytest.fail("fixture could not select dice seed")


@pytest.mark.parametrize("players", (3, 4))
def test_seeded_setup_snake_order_costs_resources_and_no_rng_draws(players):
    for seed in range(50):
        state = new_game(seed, players)
        rng_state = state.rng.getstate()
        policy = Random(seed)
        grants = np.zeros_like(state.hands)
        for index, player in enumerate(state.setup_order):
            assert state.phase == Phase.SETUP_SETTLEMENT
            assert state.current_player == player
            assert state.setup_index == index
            action = policy.choice(tuple(legal_actions(state)))
            vertex = action.vertex
            if index >= players:
                for tile in state.board.geometry.vertex_tiles[vertex]:
                    resource = state.board.resources[tile]
                    if resource != Resource.DESERT:
                        grants[player, resource] += 1
            apply_action_inplace(state, action)
            assert state.phase == Phase.SETUP_ROAD
            assert state.setup_vertex == vertex
            choices = tuple(legal_actions(state))
            assert choices
            assert all(
                vertex in state.board.geometry.edge_vertices[a.edge] for a in choices
            )
            apply_action_inplace(state, policy.choice(choices))
            state.validate()
        assert state.phase == Phase.ROLL
        assert state.current_player == 0
        assert state.setup_vertex is None
        np.testing.assert_array_equal(state.hands, grants)
        np.testing.assert_array_equal(state.bank, 19 - grants.sum(axis=0))
        np.testing.assert_array_equal(state.vp, np.full(players, 2))
        np.testing.assert_array_equal(
            state.pieces_left, np.tile((3, 4, 13), (players, 1))
        )
        assert state.rng.getstate() == rng_state


def test_setup_rejects_distance_rule_occupied_spots_and_unattached_roads():
    state = new_game(42)
    apply_action_inplace(state, SettlementAction(0))
    unattached = next(
        e for e, vs in enumerate(state.board.geometry.edge_vertices) if 0 not in vs
    )
    rejected(state, RoadAction(unattached))
    rejected(state, SettlementAction(1))
    apply_action_inplace(state, next(legal_actions(state)))
    rejected(state, SettlementAction(0))
    for neighbor in state.board.geometry.vertex_neighbors[0]:
        rejected(state, SettlementAction(neighbor))
    state.validate()


@pytest.mark.parametrize(
    "cls,name,limit",
    [
        (SettlementAction, "vertex", 54),
        (CityAction, "vertex", 54),
        (RoadAction, "edge", 72),
        (RobberMoveAction, "tile", 19),
        (StealAction, "victim", 4),
    ],
)
@pytest.mark.parametrize("value", (-1, True, 1.5, "0", np.bool_(False)))
def test_action_indices_are_not_silently_coerced(cls, name, limit, value):
    with pytest.raises(ValueError, match=name):
        cls(value)
    with pytest.raises(ValueError, match=name):
        cls(limit)
    assert getattr(cls(np.int64(0)), name) == 0


@pytest.mark.parametrize("give,receive", [(0, 0), (-1, 1), (1, 5), (True, 2), (1, 1.5)])
def test_malformed_bank_trades_are_rejected(give, receive):
    with pytest.raises(ValueError):
        BankTradeAction(give, receive)


def test_building_costs_connectivity_upgrades_and_reclaimed_pieces():
    state = empty_main()
    building(state, 0, 0)
    fund(state, 0, (3, 1, 3, 3, 3))
    geometry = state.board.geometry
    first_edge = geometry.vertex_edges[0][0]
    other = next(v for v in geometry.edge_vertices[first_edge] if v != 0)
    second_edge = next(e for e in geometry.vertex_edges[other] if e != first_edge)
    destination = next(v for v in geometry.edge_vertices[second_edge] if v != other)
    disconnected = next(
        e for e in range(72) if not set(geometry.edge_vertices[e]) & {0}
    )
    rejected(state, RoadAction(disconnected))
    rejected(state, SettlementAction(destination))
    apply_action_inplace(state, RoadAction(first_edge))
    rejected(state, SettlementAction(other))  # Adjacent to own settlement.
    apply_action_inplace(state, RoadAction(second_edge))
    apply_action_inplace(state, SettlementAction(destination))
    assert state.hands[0].tolist() == [0, 0, 2, 0, 3]
    assert state.vp[0] == 2
    assert state.pieces_left[0].tolist() == [3, 4, 13]
    apply_action_inplace(state, CityAction(0))
    assert state.hands[0].tolist() == [0] * 5
    assert state.vp[0] == 3
    assert state.vertex_building[0] == 2
    assert state.pieces_left[0].tolist() == [4, 3, 13]
    rejected(state, CityAction(0))
    rejected(state, RoadAction(first_edge))
    rejected(state, SettlementAction(0))
    state.validate()


def test_opponent_building_blocks_extension_but_own_building_allows_it():
    state = empty_main()
    junction = next(
        v
        for v, edges in enumerate(state.board.geometry.vertex_edges)
        if len(edges) == 3
    )
    incoming, outgoing, _ = state.board.geometry.vertex_edges[junction]
    road(state, incoming, 0)
    building(state, junction, 1)
    fund(state, 0, (1, 0, 0, 1, 0))
    state.validate()
    assert RoadAction(outgoing) not in tuple(legal_actions(state))
    rejected(state, RoadAction(outgoing))
    state.vertex_owner[junction] = 0
    state.pieces_left[1, 0] += 1
    state.vp[1] -= 1
    state.pieces_left[0, 0] -= 1
    state.vp[0] += 1
    assert RoadAction(outgoing) in tuple(legal_actions(state))
    apply_action_inplace(state, RoadAction(outgoing))
    state.validate()


def test_piece_and_resource_limits_remove_building_choices():
    state = empty_main()
    building(state, 0, 0)
    rejected(state, CityAction(0))
    rejected(state, RoadAction(state.board.geometry.vertex_edges[0][0]))
    assert tuple(legal_actions(state)) == (EndTurnAction(),)
    fund(state, 0, (1, 1, 2, 1, 3))
    state.pieces_left[0, 1] = 0  # Deliberately synthetic storage to test the gate.
    rejected(state, CityAction(0))
    assert CityAction(0) not in tuple(legal_actions(state))
    state.pieces_left[0, 1] = 4
    rejected(state, CityAction(2))  # Empty vertex, despite sufficient resources.


def production_expected(state, total):
    owed = np.zeros_like(state.hands)
    # Independent traversal: each building visits its incident tiles.
    for vertex, owner in enumerate(state.vertex_owner):
        if owner != -1:
            for tile in state.board.geometry.vertex_tiles[vertex]:
                if state.board.numbers[tile] == total and tile != state.robber_tile:
                    owed[owner, state.board.resources[tile]] += state.vertex_building[
                        vertex
                    ]
    return owed


def test_seeded_dice_distribute_all_matching_tiles_and_city_multipliers():
    state = empty_main()
    tile = next(t for t, r in enumerate(state.board.resources) if r != Resource.DESERT)
    vertices = state.board.geometry.tile_vertices[tile]
    building(state, vertices[0], 0)
    building(state, vertices[2], 1, 2)
    total = state.board.numbers[tile]
    owed = production_expected(state, total)
    before = state.copy()
    following = roll_total(state, total)
    assert following.phase == Phase.MAIN
    np.testing.assert_array_equal(following.hands, owed)
    np.testing.assert_array_equal(following.bank, 19 - owed.sum(axis=0))
    assert state.hands.tolist() == before.hands.tolist()
    following.validate()


@pytest.mark.parametrize("claimants", (1, 2))
@pytest.mark.parametrize("available", (0, 1))
def test_bank_shortages_partial_for_one_recipient_none_for_multiple(
    claimants, available
):
    state = empty_main()
    tile = next(t for t, r in enumerate(state.board.resources) if r != Resource.DESERT)
    vertices = state.board.geometry.tile_vertices[tile]
    building(state, vertices[0], 0, 2)
    if claimants == 2:
        building(state, vertices[2], 1)
    resource = state.board.resources[tile]
    counts = [0] * 5
    counts[resource] = 19 - available
    fund(state, 2, counts)
    state.validate()
    following = roll_total(state, state.board.numbers[tile])
    assert following.hands[0, resource] == (available if claimants == 1 else 0)
    assert following.hands[1, resource] == 0
    assert following.bank[resource] == (0 if claimants == 1 else available)
    following.validate()


def test_robber_blocks_production_without_blocking_other_matching_tiles():
    state = empty_main()
    tile = next(t for t, r in enumerate(state.board.resources) if r != Resource.DESERT)
    vertices = state.board.geometry.tile_vertices[tile]
    building(state, vertices[0], 0, 2)
    state.robber_tile = tile
    total = state.board.numbers[tile]
    expected = production_expected(state, total)
    following = roll_total(state, total)
    np.testing.assert_array_equal(following.hands, expected)
    following.validate()


def test_rolled_seven_defers_robber_until_explicit_discards_finish():
    state = empty_main()
    fund(state, 0, (4, 3, 2, 0, 0))
    following = roll_total(state, 7)
    assert following.phase == Phase.DISCARD
    assert following.discard_player == 0
    assert following.discard_pending.tolist() == [4, 0, 0, 0]
    old_robber = following.robber_tile
    rejected(following, RobberMoveAction((old_robber + 1) % 19))
    rng_state = following.rng.getstate()
    apply_action_inplace(following, DiscardAction(0, (2, 1, 1, 0, 0)))
    assert following.phase == Phase.ROBBER_MOVE
    assert following.robber_tile == old_robber
    assert following.rng.getstate() == rng_state
    following.validate()


def test_robber_victims_are_distinct_nonempty_opponents_and_steal_is_weighted():
    state = empty_main()
    state.phase = Phase.ROBBER_MOVE
    state.last_roll = 7
    tile = next(t for t in range(19) if t != state.robber_tile)
    vertices = state.board.geometry.tile_vertices[tile]
    building(state, vertices[0], 0)
    building(state, vertices[2], 1, 2)
    building(state, vertices[4], 2)
    fund(state, 1, (2, 1, 0, 0, 4))
    rng_state = state.rng.getstate()
    rejected(state, RobberMoveAction(state.robber_tile))
    apply_action_inplace(state, RobberMoveAction(tile))
    assert state.phase == Phase.ROBBER_STEAL
    assert tuple(legal_actions(state)) == (StealAction(1),)
    assert state.rng.getstate() == rng_state
    rejected(state, StealAction(0))
    rejected(state, StealAction(2))
    replay = state.copy()
    draw = replay.rng.randrange(7)
    resource = [0, 0, 1, 4, 4, 4, 4][draw]
    before = state.copy()
    apply_action_inplace(state, StealAction(1))
    assert state.phase == Phase.MAIN
    assert state.hands[0, resource] == 1
    assert state.hands[1, resource] == before.hands[1, resource] - 1
    np.testing.assert_array_equal(state.bank, before.bank)
    assert state.rng.getstate() == replay.rng.getstate()
    state.validate()


def test_empty_victim_tile_skips_steal_without_consuming_rng():
    state = empty_main()
    state.phase = Phase.ROBBER_MOVE
    state.last_roll = 7
    tile = (state.robber_tile + 1) % 19
    vertex = state.board.geometry.tile_vertices[tile][0]
    building(state, vertex, 1)
    rng_state = state.rng.getstate()
    apply_action_inplace(state, RobberMoveAction(tile))
    assert state.phase == Phase.MAIN
    assert state.rng.getstate() == rng_state
    state.validate()


@pytest.mark.parametrize(
    "port_kind,ratio", [(None, 4), ("generic", 3), ("specific", 2)]
)
def test_bank_and_port_trade_ratios_and_supply(port_kind, ratio):
    state = empty_main()
    resource = Resource.WOOD
    if port_kind is not None:
        port = next(
            i
            for i, p in enumerate(state.board.ports)
            if (
                p.resource is None if port_kind == "generic" else p.resource == resource
            )
        )
        building(state, state.board.geometry.port_vertices[port][0], 0)
    fund(state, 0, (ratio, 0, 0, 0, 0))
    state.validate()
    action = BankTradeAction(Resource.WOOD, Resource.ORE)
    assert action in tuple(legal_actions(state))
    before = state.copy()
    apply_action_inplace(state, action)
    assert state.hands[0].tolist() == [0, 0, 0, 0, 1]
    assert state.bank[0] == before.bank[0] + ratio
    assert state.bank[4] == before.bank[4] - 1
    rejected(state, action)
    state.validate()


def test_specific_port_does_not_discount_unrelated_resources_and_empty_bank_rejects():
    state = empty_main()
    port = next(
        i for i, p in enumerate(state.board.ports) if p.resource == Resource.ORE
    )
    building(state, state.board.geometry.port_vertices[port][0], 0)
    fund(state, 0, (3, 0, 0, 0, 0))
    rejected(state, BankTradeAction(0, 1))
    fund(state, 1, (0, 19, 0, 0, 0))
    fund(state, 0, (1, 0, 0, 0, 0))
    rejected(state, BankTradeAction(0, 1))
    assert BankTradeAction(0, 1) not in tuple(legal_actions(state))
    state.validate()


def test_ten_points_stop_immediately_and_end_turn_rotates_without_changing_resources():
    state = empty_main()
    occupied = []
    for vertex in range(54):
        if all(
            v not in state.board.geometry.vertex_neighbors[vertex] for v in occupied
        ):
            occupied.append(vertex)
        if len(occupied) == 6:
            break
    for index, vertex in enumerate(occupied):
        building(state, vertex, 0, 2 if index < 3 else 1)
    assert state.vp[0] == 9
    fund(state, 0, (0, 0, 2, 0, 3))
    state.validate()
    apply_action_inplace(state, CityAction(occupied[3]))
    assert is_terminal(state)
    assert winner(state) == 0
    assert state.vp[0] == 10
    assert tuple(legal_actions(state)) == ()
    rejected(state, EndTurnAction())
    rejected(state, RollAction())
    state.validate()
    state = empty_main(3)
    state.current_player = 2
    state.last_roll = 8
    before = state.copy()
    apply_action_inplace(state, EndTurnAction())
    assert state.current_player == 0 and state.phase == Phase.ROLL
    assert state.last_roll is None
    assert winner(state) is None
    np.testing.assert_array_equal(state.hands, before.hands)
    np.testing.assert_array_equal(state.bank, before.bank)
    assert state.rng.getstate() == before.rng.getstate()


def test_discard_choice_stream_is_lazy_complete_unique_and_matches_direct_choices():
    state = empty_main(3)
    state.phase = Phase.ROLL
    fund(state, 0, (4, 3, 1, 0, 0))
    from catan.engine import resolve_seven_inplace

    resolve_seven_inplace(state)
    choices = tuple(legal_actions(state))
    expected = {
        DiscardAction(0, c)
        for c in product(range(5), range(4), range(2), range(1), range(1))
        if sum(c) == 4
    }
    assert set(choices) == expected
    assert len(choices) == len(expected)
    for action in choices:
        apply_action(state, action).validate()
    large = empty_main(3)
    large.phase = Phase.ROLL
    fund(large, 0, (19,) * 5)
    resolve_seven_inplace(large)
    stream = legal_actions(large)
    assert iter(stream) is stream
    first = tuple(islice(stream, 2))
    assert len(first) == 2
    for action in first:
        apply_action(large, action).validate()
    before = large.copy()
    choose_action(large, Random(0))
    assert large == before


@pytest.mark.parametrize("players", (3, 4))
def test_full_games_replay_both_apis_with_identical_states_and_rng(players):
    global_rng = getstate()
    for seed in range(3):
        worker = new_game(seed, players)
        functional = worker.copy()
        rng = Random(seed)
        arrays = {
            f.name: getattr(worker, f.name)
            for f in fields(GameState)
            if isinstance(getattr(worker, f.name), np.ndarray)
        }
        worker_rng = worker.rng
        for step in range(10000):
            action = choose_action(worker, rng)
            before = functional.copy()
            successor = apply_action(functional, action)
            assert functional == before
            assert apply_action_inplace(worker, action) is worker
            assert worker == successor
            worker.validate()
            functional = successor
            if is_terminal(worker):
                break
        assert is_terminal(worker), f"capped game {players}/{seed}"
        assert winner(worker) == winner(functional)
        assert worker.rng is worker_rng
        for name, array in arrays.items():
            assert getattr(worker, name) is array
    assert getstate() == global_rng


@pytest.mark.parametrize("players", (3, 4))
def test_bounded_random_games_and_every_yielded_action_is_accepted(players):
    for seed in range(3):
        result = play_random_game(seed, players, validate=True, max_actions=10000)
        assert is_terminal(result)
        assert result.vp[winner(result)] >= 10
    state = new_game(0, players)
    policy = Random(0)
    for step in range(1500):
        if step < players * 4 or step % 50 == 0:
            before = state.copy()
            choices = (
                islice(legal_actions(state), 20)
                if state.phase == Phase.DISCARD
                else legal_actions(state)
            )
            found = False
            for action in choices:
                found = True
                apply_action(state, action).validate()
                assert state == before
            assert found
        apply_action_inplace(state, choose_action(state, policy))
        if is_terminal(state):
            break
    assert is_terminal(state)
    with pytest.raises(RuntimeError, match="did not finish"):
        play_random_game(0, players, max_actions=1)
    with pytest.raises(ValueError, match="positive"):
        play_random_game(0, players, max_actions=0)


def test_minimal_game_stalemate_is_capped_without_inventing_a_winner():
    state = new_game(3, 4)
    policy = Random(3)
    for _ in range(10000):
        apply_action_inplace(state, choose_action(state, policy))
    state.validate()
    assert not is_terminal(state)
    assert winner(state) is None
    assert max(state.vp) < 10
    assert not state.pieces_left[:, 1:].any()  # All cities and roads used.
    for player in range(4):
        probe = state.copy()
        probe.phase = Phase.MAIN
        probe.current_player = player
        probe.discard_pending[:] = 0
        probe.hands[:] = 0
        probe.hands[player] = 10
        probe.bank[:] = 9
        probe.validate()
        assert not any(
            isinstance(a, (SettlementAction, RoadAction, CityAction))
            for a in legal_actions(probe)
        )
    with pytest.raises(RuntimeError, match="did not finish"):
        play_random_game(3, 4, max_actions=10000)


@pytest.mark.parametrize(
    "phase,action",
    [
        (phase, action)
        for action, allowed in (
            (SettlementAction(0), (Phase.SETUP_SETTLEMENT, Phase.MAIN)),
            (RoadAction(0), (Phase.SETUP_ROAD, Phase.MAIN)),
            (CityAction(0), (Phase.MAIN,)),
            (RollAction(), (Phase.ROLL,)),
            (RobberMoveAction(0), (Phase.ROBBER_MOVE,)),
            (StealAction(1), (Phase.ROBBER_STEAL,)),
            (BankTradeAction(0, 1), (Phase.MAIN,)),
            (EndTurnAction(), (Phase.MAIN,)),
        )
        for phase in Phase
        if phase not in allowed
    ],
)
def test_other_phase_actions_reject_before_mutation_or_random_draws(phase, action):
    state = new_game(42)
    state.phase = phase
    state.rng.gauss(0, 1)
    rejected(state, action)


def test_robber_lists_multiple_victims_once_each_even_with_two_buildings():
    state = empty_main()
    state.phase = Phase.ROBBER_MOVE
    state.last_roll = 7
    tile = next(t for t in range(19) if t != state.robber_tile)
    vertices = state.board.geometry.tile_vertices[tile]
    building(state, vertices[0], 1)
    building(state, vertices[2], 1)
    building(state, vertices[4], 2)
    fund(state, 1, (1, 0, 0, 0, 0))
    fund(state, 2, (0, 1, 0, 0, 0))
    apply_action_inplace(state, RobberMoveAction(tile))
    assert tuple(legal_actions(state)) == (StealAction(1), StealAction(2))
    state.validate()

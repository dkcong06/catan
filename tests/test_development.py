from dataclasses import fields
from random import Random

import numpy as np
import pytest

from catan.actions import (
    BuyDevelopmentAction,
    EndTurnAction,
    KnightAction,
    MonopolyAction,
    RoadAction,
    RoadBuildingAction,
    RobberMoveAction,
    RollAction,
    StealAction,
    YearOfPlentyAction,
)
from catan.awards import longest_road_length, update_army, update_roads
from catan.cards import DevelopmentCard as Card
from catan.engine import apply_action, apply_action_inplace, legal_actions, is_terminal
from catan.state import GameState, Phase, new_game
from catan.validation import assert_invariants
from catan.players.random_agent import choose_action


def main_state():
    state = new_game(3)
    state.phase, state.last_roll = Phase.MAIN, 8
    state.setup_index = len(state.setup_order)
    return state


def grant(state, card, count=1, player=0, fresh=False):
    state.development_deck[card] -= count
    state.development[player, card] += count
    if fresh:
        state.development_new[player, card] += count
    if card == Card.VICTORY_POINT:
        state.vp[player] += count


def fund(state, player, values):
    state.hands[player] += values
    state.bank -= values


def path(state, length, forbidden=()):
    geometry = state.board.geometry

    def search(vertex, used, visited):
        if len(used) == length:
            return used
        for edge in geometry.vertex_edges[vertex]:
            ends = geometry.edge_vertices[edge]
            other = next(v for v in ends if v != vertex)
            if edge in forbidden or other in visited:
                continue
            result = search(other, used + [edge], visited | {other})
            if result is not None:
                return result

    for vertex in range(54):
        result = search(vertex, [], {vertex})
        if result is not None:
            return result
    raise AssertionError("no requested simple path")


def own_roads(state, edges, player=0):
    state.edge_owner[list(edges)] = player
    state.pieces_left[player, 2] -= len(edges)
    update_roads(state, (player,))


def build(state, vertex, player=0):
    state.vertex_owner[vertex] = player
    state.vertex_building[vertex] = 1
    state.pieces_left[player, 0] -= 1
    state.vp[player] += 1
    update_roads(state, range(state.n_players))


def assert_paths_equal(state, action):
    before = state.copy()
    functional = apply_action(state, action)
    assert state == before
    owned = state.copy()
    ids = {
        f.name: id(getattr(owned, f.name))
        for f in fields(GameState)
        if isinstance(getattr(owned, f.name), np.ndarray) or f.name == "rng"
    }
    assert apply_action_inplace(owned, action) is owned
    assert owned == functional
    assert ids == {name: id(getattr(owned, name)) for name in ids}
    owned.validate()
    return owned


def test_all_25_purchases_exhaust_deck_and_preserve_distribution():
    state = main_state()
    before_rng = state.rng.getstate()
    for _ in range(25):
        fund(state, 0, (0, 1, 1, 0, 1))
        state = assert_paths_equal(state, BuyDevelopmentAction())
    assert state.development[0].tolist() == [14, 5, 2, 2, 2]
    assert np.array_equal(state.development_new[0], state.development[0])
    assert state.vp[0] == 5 and not state.development_deck.any()
    assert state.rng.getstate() != before_rng
    rejected = state.copy()
    with pytest.raises(ValueError, match="buy"):
        apply_action_inplace(state, BuyDevelopmentAction())
    assert state == rejected


@pytest.mark.parametrize(
    "action,card",
    [
        (KnightAction(), Card.KNIGHT),
        (RoadBuildingAction(), Card.ROAD_BUILDING),
        (MonopolyAction(0), Card.MONOPOLY),
        (YearOfPlentyAction((2, 0, 0, 0, 0)), Card.YEAR_OF_PLENTY),
    ],
)
def test_new_cards_and_one_card_limit_reject_without_mutation(action, card):
    state = main_state()
    build(state, 0)
    grant(state, card, fresh=True)
    for used in (False, True):
        state.development_used = used
        before = state.copy()
        for step in (apply_action, apply_action_inplace):
            with pytest.raises(ValueError):
                step(state, action)
            assert state == before
    state.development_new[:] = 0
    state.development_used = False
    played = assert_paths_equal(state, action)
    assert played.development_used
    assert played.development_played[card] == 1
    if played.phase == Phase.MAIN:
        grant(played, Card.KNIGHT)
        before = played.copy()
        with pytest.raises(ValueError):
            apply_action_inplace(played, KnightAction())
        assert played == before


def test_new_and_old_card_of_same_type_can_play_only_the_old_one():
    state = main_state()
    grant(state, Card.MONOPOLY, 2)
    state.development_new[0, Card.MONOPOLY] = 1
    state = assert_paths_equal(state, MonopolyAction(3))
    assert (
        state.development[0, Card.MONOPOLY]
        == state.development_new[0, Card.MONOPOLY]
        == 1
    )
    assert_paths_equal(state, EndTurnAction())


@pytest.mark.parametrize("phase", (Phase.ROLL, Phase.MAIN))
def test_knight_never_discards_and_resumes_its_original_phase(phase):
    state = main_state()
    state.phase = phase
    state.last_roll = None if phase == Phase.ROLL else 8
    grant(state, Card.KNIGHT)
    fund(state, 0, (9, 0, 0, 0, 0))
    tile = next(t for t in range(19) if t != state.robber_tile)
    build(state, state.board.geometry.tile_vertices[tile][0], 1)
    fund(state, 1, (0, 1, 0, 0, 0))
    state = assert_paths_equal(state, KnightAction())
    assert not state.discard_pending.any() and state.hands[0, 0] == 9
    assert state.resume_phase == phase and state.knights_played[0] == 1
    state = assert_paths_equal(state, RobberMoveAction(tile))
    assert state.phase == Phase.ROBBER_STEAL
    state = assert_paths_equal(state, StealAction(1))
    assert state.phase == phase and state.resume_phase is None
    assert state.hands[0, 1] == 1


def test_pre_roll_knight_without_victim_returns_to_roll():
    state = main_state()
    state.phase = Phase.ROLL
    state.last_roll = None
    grant(state, Card.KNIGHT)
    state = apply_action(state, KnightAction())
    tile = (state.robber_tile + 1) % 19
    state = assert_paths_equal(state, RobberMoveAction(tile))
    assert state.phase == Phase.ROLL
    state = assert_paths_equal(state, RollAction())
    assert state.last_roll is not None


def test_monopoly_transfers_all_opponents_and_no_bank_or_randomness():
    state = main_state()
    grant(state, Card.MONOPOLY)
    fund(state, 0, (0, 0, 0, 1, 0))
    fund(state, 1, (0, 0, 0, 4, 0))
    fund(state, 2, (0, 0, 0, 5, 0))
    rng, bank = state.rng.getstate(), state.bank.copy()
    result = assert_paths_equal(state, MonopolyAction(3))
    assert result.hands[:, 3].tolist() == [10, 0, 0, 0]
    assert result.rng.getstate() == rng and np.array_equal(bank, result.bank)


@pytest.mark.parametrize(
    "bank,chosen",
    [
        ([19] * 5, (2, 0, 0, 0, 0)),
        ([19] * 5, (0, 1, 0, 0, 1)),
        ([0, 0, 1, 0, 0], (0, 0, 1, 0, 0)),
        ([0] * 5, (0, 0, 0, 0, 0)),
    ],
)
def test_plenty_respects_shortage_and_exact_available_count(bank, chosen):
    state = main_state()
    grant(state, Card.YEAR_OF_PLENTY)
    state.bank[:] = bank
    state.hands[1] = 19 - np.array(bank)
    result = assert_paths_equal(state, YearOfPlentyAction(chosen))
    assert result.hands[0].tolist() == list(chosen)
    assert result.bank.tolist() == [a - b for a, b in zip(bank, chosen)]


def test_plenty_invalid_choice_preserves_card_bank_and_rng():
    state = main_state()
    grant(state, Card.YEAR_OF_PLENTY)
    for counts in ((1, 0, 0, 0, 0), (0, 0, 0, 0, 0)):
        before = state.copy()
        with pytest.raises(ValueError):
            apply_action_inplace(state, YearOfPlentyAction(counts))
        assert state == before


@pytest.mark.parametrize("phase", (Phase.ROLL, Phase.MAIN))
def test_road_building_places_two_sequential_free_roads_and_resumes(phase):
    state = main_state()
    state.phase = phase
    state.last_roll = None if phase == Phase.ROLL else 8
    build(state, 0)
    grant(state, Card.ROAD_BUILDING)
    bank, hands = state.bank.copy(), state.hands.copy()
    state = assert_paths_equal(state, RoadBuildingAction())
    assert state.phase == Phase.ROAD_BUILDING and state.free_roads_remaining == 2
    for remaining in (1, 0):
        road = next(a for a in legal_actions(state) if isinstance(a, RoadAction))
        state = assert_paths_equal(state, road)
        assert state.free_roads_remaining == remaining
    assert state.phase == phase and state.resume_phase is None
    assert np.array_equal(bank, state.bank) and np.array_equal(hands, state.hands)


def test_road_building_with_one_piece_finishes_after_one():
    state = main_state()
    edges = path(state, 14)
    own_roads(state, edges)
    grant(state, Card.ROAD_BUILDING)
    state = assert_paths_equal(state, RoadBuildingAction())
    assert state.free_roads_remaining == 1
    road = next(legal_actions(state))
    state = assert_paths_equal(state, road)
    assert state.phase == Phase.MAIN and state.pieces_left[0, 2] == 0


def test_road_building_without_any_connection_cannot_start():
    state = main_state()
    grant(state, Card.ROAD_BUILDING)
    before = state.copy()
    with pytest.raises(ValueError):
        apply_action_inplace(state, RoadBuildingAction())
    assert state == before


def test_largest_army_threshold_incumbent_tie_and_transfer():
    state = main_state()
    for player, count in ((0, 2), (0, 3), (1, 3), (1, 4)):
        delta = count - int(state.knights_played[player])
        state.knights_played[player] = count
        state.development_deck[0] -= delta
        state.development_played[0] += delta
        update_army(state)
        assert state.largest_army_owner == (
            -1 if count == 2 else 0 if count == 3 else 1
        )
        state.validate()
    assert state.vp.tolist() == [0, 2, 0, 0]


def trail_oracle(state, player):
    geometry = state.board.geometry
    owned = {e for e, p in enumerate(state.edge_owner) if p == player}

    def walk(v, used):
        if used and state.vertex_owner[v] not in (-1, player):
            return len(used)
        best = len(used)
        for edge in geometry.vertex_edges[v]:
            if edge in owned - used:
                other = next(x for x in geometry.edge_vertices[edge] if x != v)
                best = max(best, walk(other, used | {edge}))
        return best

    return max(walk(v, set()) for v in range(54))


def test_road_length_matches_independent_exhaustive_oracle_with_loops_and_blocks():
    random = Random(84)
    for _ in range(120):
        state = main_state()
        edges = random.sample(range(72), random.randrange(1, 16))
        state.edge_owner[edges] = 0
        for vertex in random.sample(range(54), 8):
            state.vertex_owner[vertex] = random.choice((0, 1, 2))
        assert longest_road_length(state, 0) == trail_oracle(state, 0)
    state = main_state()
    ring = set(state.board.geometry.tile_vertices[0])
    edges = [
        e
        for e, ends in enumerate(state.board.geometry.edge_vertices)
        if set(ends) <= ring
    ]
    own_roads(state, edges)
    assert longest_road_length(state, 0) == 6
    v = next(iter(ring))
    state.vertex_owner[v] = 1
    assert (
        longest_road_length(state, 0) == 6
    )  # Whole loop with the blocked vertex as both endpoints.
    other = next(x for x in ring if x != v)
    state.vertex_owner[other] = 2
    assert longest_road_length(state, 0) == trail_oracle(state, 0) < 6


def test_longest_road_threshold_ties_transfer_and_revocation():
    state = main_state()
    first = path(state, 6)
    second = path(state, 6, first)
    own_roads(state, first[:4])
    assert state.longest_road_owner == -1
    own_roads(state, first[4:5])
    assert state.longest_road_owner == 0
    own_roads(state, second[:5], 1)
    assert state.longest_road_owner == 0
    own_roads(state, second[5:], 1)
    assert state.longest_road_owner == 1
    own_roads(state, first[5:])
    assert state.longest_road_owner == 1
    state.edge_owner[second[-1]] = -1
    state.pieces_left[1, 2] += 1
    update_roads(state, (1,))
    assert state.longest_road_owner == 0
    state.edge_owner[first[-2:]] = -1
    state.pieces_left[0, 2] += 2
    update_roads(state, (0,))
    assert state.longest_road_owner == 1
    state.edge_owner[second[-2]] = -1
    state.pieces_left[1, 2] += 1
    update_roads(state, (1,))
    assert state.longest_road_owner == -1
    state.validate()


def test_vp_card_can_win_immediately_when_fresh_and_pre_roll_award_can_win():
    state = main_state()
    # Five settlements plus four held VP, last VP is the only card in the deck.
    vertices = []
    for v in range(54):
        if not set(state.board.geometry.vertex_neighbors[v]) & set(vertices):
            vertices.append(v)
        if len(vertices) == 5:
            break
    for v in vertices:
        build(state, v)
    grant(state, Card.VICTORY_POINT, 4)
    for card in (Card.KNIGHT, Card.ROAD_BUILDING, Card.YEAR_OF_PLENTY, Card.MONOPOLY):
        grant(state, card, int(state.development_deck[card]), player=1)
    fund(state, 0, (0, 1, 1, 0, 1))
    result = assert_paths_equal(state, BuyDevelopmentAction())
    assert is_terminal(result) and result.development_new[0, 1] == 1
    state = main_state()
    state.phase = Phase.ROLL
    state.last_roll = None
    for v in vertices[:3]:
        build(state, v)
    grant(state, Card.VICTORY_POINT, 5)
    state.knights_played[0] = 2
    state.development_played[0] = 2
    state.development_deck[0] -= 2
    grant(state, Card.KNIGHT)
    result = assert_paths_equal(state, KnightAction())
    assert is_terminal(result) and result.vp[0] == 10 and result.resume_phase is None


def test_seeded_full_games_validate_every_new_field_and_phase():
    coverage = set()
    played = set()
    for players in (3, 4):
        for seed in range(5):
            state = new_game(seed, players)
            policy = Random(seed)
            for _ in range(10000):
                assert_invariants(state)
                coverage.add(state.phase)
                played.update(np.flatnonzero(state.development_played))
                if is_terminal(state):
                    break
                apply_action_inplace(state, choose_action(state, policy))
            assert is_terminal(state)
    assert coverage == set(Phase)
    assert played == {0, 2, 3, 4}


def test_road_building_finishes_if_the_first_road_exhausts_legal_spaces():
    state = main_state()
    build(state, 0)
    grant(state, Card.ROAD_BUILDING)
    geometry = state.board.geometry
    edge = geometry.vertex_edges[0][0]
    other = next(v for v in geometry.edge_vertices[edge] if v != 0)
    occupied = set(geometry.vertex_edges[0] + geometry.vertex_edges[other]) - {edge}
    own_roads(state, occupied, 1)
    state = assert_paths_equal(state, RoadBuildingAction())
    assert tuple(legal_actions(state)) == (RoadAction(edge),)
    state = assert_paths_equal(state, RoadAction(edge))
    assert state.phase == Phase.MAIN and state.free_roads_remaining == 0


def test_settlement_splits_opponent_road_and_revokes_its_award():
    from catan.actions import SettlementAction

    state = main_state()
    edges = path(state, 6)
    own_roads(state, edges)
    geometry = state.board.geometry
    junction, split = next(
        (v, i + 1)
        for i in range(1, 4)
        for v in set(geometry.edge_vertices[edges[i]])
        & set(geometry.edge_vertices[edges[i + 1]])
        if len(geometry.vertex_edges[v]) == 3
    )
    connection = next(e for e in geometry.vertex_edges[junction] if e not in edges)
    own_roads(state, (connection,), 1)
    state.current_player = 1
    fund(state, 1, (1, 1, 1, 1, 0))
    state = assert_paths_equal(state, SettlementAction(junction))
    assert (
        state.road_lengths[0] == max(split, 6 - split)
        and state.longest_road_owner == -1
    )
    assert state.vp.tolist() == [0, 1, 0, 0]


def test_off_turn_ten_points_wait_for_that_players_turn():
    state = main_state()
    vertices = []
    for v in range(54):
        if not set(state.board.geometry.vertex_neighbors[v]) & set(vertices):
            vertices.append(v)
        if len(vertices) == 4:
            break
    for v in vertices:
        build(state, v, 1)
    grant(state, Card.VICTORY_POINT, 4, player=1)
    edges = path(state, 5)
    own_roads(state, edges, 1)
    assert state.vp[1] == 10 and state.current_player == 0
    assert_invariants(state)  # A player can receive an award off turn.
    state = assert_paths_equal(state, EndTurnAction())
    assert is_terminal(state) and state.current_player == 1 and state.last_roll is None


def test_road_award_reverts_to_unheld_when_former_holder_is_below_tied_leaders():
    from catan.awards import _update_award

    state = main_state()
    # Isolate history/tie bookkeeping from independently tested trail geometry.
    state.road_lengths[:] = [6, 5, 5, 0]
    state.longest_road_owner = 0
    state.vp[0] = 2
    state.road_lengths[0] = 4
    _update_award(state, "longest_road_owner", state.road_lengths.tolist(), 5)
    assert state.longest_road_owner == -1 and state.vp.tolist() == [0, 0, 0, 0]

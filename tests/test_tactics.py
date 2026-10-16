"""Observation-only tactical proofs versus independent direct engine branches."""

from random import Random

import numpy as np
import pytest

from catan.actions import KnightAction
from catan.cards import DevelopmentCard as Card
from catan.engine import apply_action, winner
from catan.state import Phase
from catan.validation import assert_invariants
from rl.actions import indexed_action
from rl.env import CatanEnv
from rl.policies import POLICY_VERSIONS, greedy_policy, tactical_policy
from rl.tactics import winning_actions
from test_development import build, fund, grant, main_state, own_roads, path
from test_env import fixture_env


def score_state():
    state = main_state()
    edges = path(state, 5)
    root = next(
        v
        for v in state.board.geometry.edge_vertices[edges[0]]
        if v not in state.board.geometry.edge_vertices[edges[1]]
    )
    build(state, root)
    for vertex in range(54):
        if state.vp[0] == 3:
            break
        if state.vertex_owner[vertex] == -1 and all(
            state.vertex_owner[v] == -1
            for v in state.board.geometry.vertex_neighbors[vertex]
        ):
            build(state, vertex)
    grant(state, Card.VICTORY_POINT, 5)
    assert state.vp[0] == 8
    return state, edges


def oracle(state, request):
    result = np.zeros(254, dtype=np.bool_)
    if request.mask is None or state.phase not in (
        Phase.MAIN,
        Phase.ROLL,
        Phase.ROAD_BUILDING,
    ):
        return result

    def wins_after(candidate, action):
        successor = apply_action(candidate, indexed_action(candidate, action))
        if winner(successor) == request.player:
            return True
        if successor.phase == Phase.ROAD_BUILDING:
            following = fixture_env(successor).decision()
            return any(
                wins_after(successor, int(i))
                for i in np.flatnonzero(following.mask[:72])
            )
        return False

    for choice in np.flatnonzero(request.mask):
        if choice < 180 or choice in (226, 227):
            result[choice] = wins_after(state, int(choice))
    return result


@pytest.mark.parametrize("phase", (Phase.MAIN, Phase.ROLL, Phase.ROAD_BUILDING))
def test_knight_and_sequential_free_road_wins_match_engine(phase):
    state, edges = score_state()
    own_roads(state, edges[:3])
    grant(state, Card.ROAD_BUILDING)
    if phase == Phase.ROAD_BUILDING:
        state = apply_action(state, indexed_action(state, 227))
    elif phase == Phase.ROLL:
        state.phase, state.last_roll = phase, None
    assert_invariants(state)
    request = fixture_env(state).decision()
    before = state.copy()
    expected = oracle(state, request)
    np.testing.assert_array_equal(winning_actions(request), expected)
    assert expected.any() and state == before
    selected = tactical_policy(request, Random(9))
    assert expected[selected]


def test_taking_longest_road_precedes_legacy_build_preferences():
    state, edges = score_state()
    own_roads(state, edges[:4])
    fund(state, 0, (1, 1, 1, 1, 0))
    assert_invariants(state)
    request = fixture_env(state).decision()
    wins = winning_actions(request)
    np.testing.assert_array_equal(wins, oracle(state, request))
    assert wins[edges[4]]
    assert wins[tactical_policy(request, Random(0))]
    assert POLICY_VERSIONS["greedy"] == 1 and POLICY_VERSIONS["tactical"] == 2


@pytest.mark.parametrize("phase", (Phase.MAIN, Phase.ROLL))
def test_largest_army_knight_wins_without_inspecting_robbery(phase):
    state, _ = score_state()
    state.knights_played[0] = 2
    state.development_played[Card.KNIGHT] = 2
    state.development_deck[Card.KNIGHT] -= 2
    grant(state, Card.KNIGHT)
    state.phase = phase
    state.last_roll = None if phase == Phase.ROLL else 8
    assert_invariants(state)
    request = fixture_env(state).decision()
    assert winning_actions(request)[226]
    assert winner(apply_action(state, KnightAction())) == 0
    state.development_new[0, Card.KNIGHT] = 1
    assert not winning_actions(fixture_env(state).decision())[226]


def test_tactics_cannot_see_hidden_hands_cards_or_rng():
    state, edges = score_state()
    own_roads(state, edges[:4])
    fund(state, 0, (1, 0, 0, 1, 0))
    other = state.copy()
    fund(other, 1, (2, 1, 0, 0, 0))
    other.hands[1] = (0, 0, 1, 1, 1)
    other.bank[:] = 19 - other.hands.sum(0)
    other.rng.seed(999)
    first, second = fixture_env(state).decision(), fixture_env(other).decision()
    # Opponent totals changed here, but tactical conclusions cannot depend on composition.
    np.testing.assert_array_equal(winning_actions(first), winning_actions(second))
    for request in (first, second):
        rng = Random(17)
        before = rng.getstate()
        tactical_policy(request, rng)
        assert rng.getstate() == before


@pytest.mark.parametrize("players", (3, 4))
def test_fresh_games_have_legal_tactics_and_exact_engine_win_sets(players):
    checked = 0
    for seed in range(2):
        env = CatanEnv()
        request = env.reset(3_500_000_100 + seed, n_players=players)
        while request is not None:
            wins = winning_actions(request)
            if request.phase in (Phase.MAIN, Phase.ROLL, Phase.ROAD_BUILDING):
                state = env.snapshot()
                if state.vp[request.player] >= 7:
                    np.testing.assert_array_equal(wins, oracle(state, request))
                    checked += 1
            choice = tactical_policy(request, Random(0))
            if request.mask is not None:
                assert request.mask[choice]
                if wins.any():
                    assert wins[choice]
            else:
                assert sum(choice) == request.discard_required
            env.step(choice, player=request.player)
            request = env.decision()
        assert_invariants(env.snapshot())
    assert checked > 0


def test_early_game_falls_back_to_exact_legacy_policy():
    env = CatanEnv()
    request = env.reset(7)
    for _ in range(30):
        assert not winning_actions(request).any()
        assert tactical_policy(request, Random(42)) == greedy_policy(
            request, Random(42)
        )
        env.step(greedy_policy(request, Random(0)), player=request.player)
        request = env.decision()


def test_settlement_can_win_from_seven_by_transferring_longest_road():
    state = main_state()
    geometry = state.board.geometry
    opponent = path(state, 6)
    endpoint = next(
        v
        for v in geometry.edge_vertices[opponent[0]]
        if v not in geometry.edge_vertices[opponent[1]]
    )
    build(state, endpoint, 0)
    junction = next(
        v
        for i in range(1, 4)
        for v in set(geometry.edge_vertices[opponent[i]])
        & set(geometry.edge_vertices[opponent[i + 1]])
        if len(geometry.vertex_edges[v]) == 3
    )
    branch = next(e for e in geometry.vertex_edges[junction] if e not in opponent)

    def extend(vertex, used, visited):
        if len(used) == 5:
            if vertex in geometry.vertex_neighbors[endpoint] or vertex == endpoint:
                return None
            return used, vertex
        for edge in geometry.vertex_edges[vertex]:
            other = next(v for v in geometry.edge_vertices[edge] if v != vertex)
            if edge not in opponent and edge not in used and other not in visited:
                found = extend(other, used + [edge], visited | {other})
                if found:
                    return found
        return None

    first = next(v for v in geometry.edge_vertices[branch] if v != junction)
    own, root = extend(first, [branch], {junction, first})
    build(state, root, 1)
    for vertex in range(54):
        if (
            state.vertex_owner[vertex] == -1
            and vertex not in geometry.vertex_neighbors[junction]
            and vertex != junction
            and all(
                state.vertex_owner[v] == -1 for v in geometry.vertex_neighbors[vertex]
            )
        ):
            build(state, vertex, 1)
            break
    grant(state, Card.VICTORY_POINT, 5, player=1)
    own_roads(state, opponent, 0)
    own_roads(state, own, 1)
    state.current_player = 1
    fund(state, 1, (1, 1, 1, 1, 0))
    assert state.vp[1] == 7 and state.longest_road_owner == 0
    assert_invariants(state)
    request = fixture_env(state).decision()
    np.testing.assert_array_equal(winning_actions(request), oracle(state, request))
    assert winning_actions(request)[72 + junction]

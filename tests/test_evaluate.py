from random import Random
from subprocess import run
import sys

import numpy as np
import pytest

from catan.actions import DiscardAction
from catan.cards import DevelopmentCard as Card
from catan.engine import apply_action_inplace, resolve_seven_inplace
from catan.players.random_agent import choose_action
from catan.state import Phase, new_game
from catan.validation import assert_invariants
from rl.actions import action_index, indexed_action
from rl.encoding import FlatEncoder, decision_player
from rl.env import CatanEnv
from rl.evaluate import evaluate, play_match
from rl.policies import greedy_policy, random_policy
from test_development import fund, grant, main_state
from test_env import fixture_env


@pytest.mark.parametrize("policy", (random_policy, greedy_policy))
def test_policies_choose_feasible_explicit_maximum_hand_discards(policy):
    state = main_state()
    state.phase, state.last_roll = Phase.ROLL, None
    fund(state, 0, (19,) * 5)
    resolve_seven_inplace(state)
    env = fixture_env(state)
    request = env.decision()
    original = env.snapshot()
    for seed in range(30):
        choice = policy(request, Random(seed))
        assert len(choice) == 5 and sum(choice) == 47
        for r, count in enumerate(choice):
            assert request.discard_mask(choice[:r])[count]
        assert env.snapshot() == original
    result = env.step(choice, player=0)
    assert result.next_player == 0 and env.snapshot().phase == Phase.ROBBER_MOVE
    assert_invariants(env.snapshot())


def test_opponent_hidden_composition_freshness_private_vp_and_deck_do_not_change_policies():
    state = main_state()
    fund(state, 0, (3, 2, 1, 3, 2))
    fund(state, 1, (2, 0, 0, 0, 0))
    fund(state, 2, (0, 2, 0, 0, 0))
    grant(state, Card.KNIGHT, player=1)
    grant(state, Card.VICTORY_POINT, player=2)
    other = state.copy()
    other.hands[[1, 2]] = other.hands[[2, 1]]
    other.development[[1, 2]] = other.development[[2, 1]]
    other.vp[1] += 1
    other.vp[2] -= 1
    third = state.copy()
    third.development[1, Card.KNIGHT] -= 1
    third.development[1, Card.ROAD_BUILDING] += 1
    third.development_deck[Card.KNIGHT] += 1
    third.development_deck[Card.ROAD_BUILDING] -= 1
    for candidate in (state, other, third):
        assert_invariants(candidate)
    requests = [
        fixture_env(candidate).decision() for candidate in (state, other, third)
    ]
    for request in requests[1:]:
        np.testing.assert_array_equal(request.observation, requests[0].observation)
        np.testing.assert_array_equal(request.mask, requests[0].mask)
    for policy in (random_policy, greedy_policy):
        outcomes = []
        for request in requests:
            rng = Random(19)
            outcomes.append((policy(request, rng), rng.getstate()))
        assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("players", (3, 4))
def test_policy_legality_all_reached_phases_and_rng_state_isolation(players):
    covered = set()
    for seed in range(5):
        state = new_game(seed, players)
        env = CatanEnv(max_actions=5000)
        request = env.reset(seed, n_players=players)
        rng = Random(seed)
        for step in range(5000):
            covered.add(request.phase)
            before = state.copy()
            for policy in (random_policy, greedy_policy):
                choice = policy(request, Random(step))
                if request.phase == Phase.DISCARD:
                    for r, count in enumerate(choice):
                        assert request.discard_mask(choice[:r])[count]
                else:
                    assert request.mask[choice]
            assert state == before
            action = choose_action(state, rng)
            choice = (
                action.resources
                if isinstance(action, DiscardAction)
                else action_index(state, action)
            )
            result = env.step(choice, player=request.player)
            apply_action_inplace(state, action)
            assert env._state == state
            if result.terminated:
                break
            request = env.decision()
    assert covered == set(Phase) - {Phase.GAME_OVER}


@pytest.mark.parametrize("players", (3, 4))
def test_complete_match_reproduces_direct_engine_and_exact_repeat(players):
    seed = 12
    direct = new_game(seed, players)
    encoder = FlatEncoder(direct.board)

    def shadow(request, rng):
        assert request.player == decision_player(direct)
        np.testing.assert_array_equal(request.observation, encoder.encode(direct))
        choice = random_policy(request, rng)
        action = (
            DiscardAction(request.player, choice)
            if request.phase == Phase.DISCARD
            else indexed_action(direct, choice)
        )
        apply_action_inplace(direct, action)
        return choice

    result = play_match([shadow] * players, seed, policy_seed=7, validate=True)
    repeat = play_match([random_policy] * players, seed, policy_seed=7, validate=True)
    assert result == repeat
    assert result.terminated and not result.truncated
    assert result.scores == tuple(direct.vp)
    assert sum(result.rewards) == 1 and result.rewards[result.winner] == 1
    assert sum(result.phase_counts) == result.actions + 1
    assert_invariants(direct)


@pytest.mark.parametrize("players", (3, 4))
def test_rotated_reports_reproduce_and_same_policy_control_has_known_win_accounting(
    players,
):
    first = evaluate(
        candidate="random", opponent="random", seeds=(3, 4), n_players=players
    )
    second = evaluate(
        candidate="random", opponent="random", seeds=(3, 4), n_players=players
    )
    for report in (first, second):
        report.pop("elapsed_seconds")
        report.pop("actions_per_second")
    assert first == second
    summary = first["summary"]
    assert summary["games"] == 2 * players and summary["completed"] == 2 * players
    assert summary["candidate_wins"] == 2 and summary["win_rate_all"] == 1 / players
    assert summary["capped"] == 0 and len(first["by_seat"]) == players
    assert first["matches_played"] == 2
    assert first["executed_actions"] * players == first["actions"]
    assert all(item["games"] == 2 for item in first["by_seat"].values())
    assert [(game["seed"], game["candidate_seat"]) for game in first["games"]] == [
        (seed, seat) for seed in (3, 4) for seat in range(players)
    ]
    assert (
        first["reward_scheme"] == "terminal_win_v1" and first["observation_schema"] == 2
    )


def test_capped_reports_do_not_invent_wins_losses_rewards_or_completed_denominator():
    report = evaluate(seeds=(0,), max_actions=1, n_players=3)
    summary = report["summary"]
    assert summary["capped"] == 3 and summary["completed"] == 0
    assert summary["candidate_wins"] == summary["candidate_losses"] == 0
    assert summary["win_rate_completed"] is None
    assert report["actions"] == 3
    for game in report["games"]:
        assert game["winner"] is None and game["truncated"] and not game["terminated"]
        assert game["rewards"] == (0.0,) * 3 and game["completed_turns"] == 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"seeds": ()},
        {"seeds": (0, 0)},
        {"seeds": (True,)},
        {"max_actions": 0},
        {"candidate": "missing"},
        {"n_players": 2},
        {"validate": 1},
        {"policy_seed": True},
    ),
)
def test_invalid_evaluation_configuration_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        evaluate(**kwargs)


def test_policy_failure_retains_replay_context_and_fails_instead_of_reporting_success():
    with pytest.raises(ValueError) as failure:
        play_match([lambda request, rng: 219] * 3, 24)
    note = " ".join(failure.value.__notes__)
    assert (
        "seed=24" in note
        and "action=1" in note
        and "actor=0" in note
        and "SETUP_SETTLEMENT" in note
    )


def test_cli_reports_caps_and_rejects_invalid_bounds():
    result = run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_policies",
            "--seeds",
            "1",
            "--players",
            "3",
            "--max-actions",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"capped": 3' in result.stdout and '"candidate_wins": 0' in result.stdout
    invalid = run(
        [sys.executable, "-m", "scripts.evaluate_policies", "--seeds", "0"],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode != 0 and "must be positive" in invalid.stderr


def test_different_policy_rotations_execute_every_match_and_outputs_do_not_alias():
    report = evaluate(seeds=(8,), n_players=3, max_actions=100)
    assert report["matches_played"] == 3
    assert report["executed_actions"] == report["actions"]
    original = report["games"][1]["candidate_seat"]
    report["games"][0]["candidate_seat"] = -1
    assert report["games"][1]["candidate_seat"] == original


@pytest.mark.parametrize("available", (0, 1, 2))
def test_greedy_plenty_selects_exact_available_bank_counts(available):
    state = main_state()
    grant(state, Card.YEAR_OF_PLENTY)
    # Conserve resource cards while leaving zero, one or two available bank cards.
    state.hands[1] = state.bank
    state.bank.fill(0)
    state.bank[4] = available
    state.hands[1, 4] -= available
    env = fixture_env(state)
    request = env.decision()
    choice = greedy_policy(request, Random(0))
    assert request.mask[choice]
    assert 233 <= choice < 254
    action = indexed_action(state, choice)
    assert sum(action.resources) == min(2, available)
    env.step(choice, player=0)
    assert_invariants(env.snapshot())


def test_policy_calls_preserve_request_observations_masks_and_policy_rng_for_greedy():
    env = CatanEnv()
    request = env.reset(5)
    obs = request.observation.copy()
    mask = request.mask.copy()
    rng = Random(14)
    original_rng = rng.getstate()
    first = greedy_policy(request, rng)
    second = greedy_policy(request, rng)
    assert first == second and rng.getstate() == original_rng
    np.testing.assert_array_equal(request.observation, obs)
    np.testing.assert_array_equal(request.mask, mask)


@pytest.mark.parametrize(
    "fail_on,action,phase", ((1, 0, "INITIALIZE"), (2, 1, "SETUP_SETTLEMENT"))
)
def test_debug_failure_coordinates_include_initialization_and_actual_accepted_step(
    monkeypatch, fail_on, action, phase
):
    import rl.evaluate as evaluation

    calls = 0

    def fail(state):
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise ValueError("injected debug failure")

    monkeypatch.setattr(evaluation, "assert_invariants", fail)
    with pytest.raises(ValueError, match="injected") as caught:
        play_match([random_policy] * 3, 42, validate=True)
    note = " ".join(caught.value.__notes__)
    assert (
        "seed=42" in note and f"action={action}," in note and f"phase={phase}" in note
    )

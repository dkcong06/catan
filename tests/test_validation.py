from dataclasses import fields
import json
from random import Random, getstate
import subprocess
import sys

import numpy as np
import pytest

from catan.awards import update_roads
from catan.actions import RollAction, SettlementAction
from catan.engine import apply_action, apply_action_inplace, is_terminal
from catan.players.random_agent import choose_action
from catan.state import GameState, Phase, new_game
from catan.validation import GameValidationError, assert_invariants, validate_game


def main_state():
    state = new_game(0)
    state.setup_index = len(state.setup_order)
    state.phase = Phase.MAIN
    state.last_roll = 8
    return state


def add_building(state, vertex, player=0, code=1):
    state.vertex_owner[vertex] = player
    state.vertex_building[vertex] = code
    state.pieces_left[player, code - 1] -= 1
    state.vp[player] += code
    update_roads(state, range(state.n_players))


def add_road(state, edge, player=0):
    state.edge_owner[edge] = player
    state.pieces_left[player, 2] -= 1
    update_roads(state, (player,))


@pytest.mark.parametrize("players", (3, 4))
def test_validation_replays_complete_games_on_both_paths_without_global_rng_use(
    players,
):
    before = getstate()
    coverage = set()
    for seed in range(3):
        owned = validate_game(seed, players)
        copied = validate_game(seed, players, inplace=False)
        assert owned == copied
        assert not owned.capped
        assert owned.scores[owned.winner] >= 10
        assert sum(owned.phase_counts) == owned.actions + 1
        coverage.update(p for p, count in zip(Phase, owned.phase_counts) if count)
    assert coverage == set(Phase)
    assert getstate() == before


def test_short_caps_replay_exactly_without_inventing_wins():
    result = validate_game(0, max_actions=1)
    assert result.capped and result.winner is None
    assert result.actions == 1 and sum(result.phase_counts) == 2
    assert result == validate_game(0, max_actions=1, inplace=False)
    assert result.fingerprint != validate_game(1, max_actions=1).fingerprint


@pytest.mark.parametrize("cap", (0, -1, True, 1.5, "100"))
def test_validator_rejects_invalid_caps(cap):
    with pytest.raises(ValueError, match="max_actions"):
        validate_game(0, max_actions=cap)


def test_invariants_preserve_all_array_and_rng_identities_and_values():
    state = new_game(7)
    policy = Random(7)
    for _ in range(400):
        state = apply_action_inplace(state, choose_action(state, policy))
    before = state.copy()
    identities = {f.name: id(getattr(state, f.name)) for f in fields(GameState)}
    assert_invariants(state)
    assert state == before
    assert identities == {f.name: id(getattr(state, f.name)) for f in fields(GameState)}


def test_unrooted_components_are_rejected_even_when_piece_counts_balance():
    state = main_state()
    add_building(state, 0)
    geometry = state.board.geometry
    add_road(state, geometry.vertex_edges[0][0])
    far_edge = next(
        e
        for e, ends in enumerate(geometry.edge_vertices)
        if not set(ends) & {0, *geometry.vertex_neighbors[0]}
    )
    add_road(state, far_edge)
    state.validate()
    with pytest.raises(ValueError, match="unrooted"):
        assert_invariants(state)


def test_closed_road_cycle_without_a_building_is_rejected():
    state = main_state()
    ring = set(state.board.geometry.tile_vertices[0])
    for edge, ends in enumerate(state.board.geometry.edge_vertices):
        if set(ends) <= ring:
            add_road(state, edge)
    state.validate()
    with pytest.raises(ValueError, match="unrooted"):
        assert_invariants(state)


def test_two_separate_setup_roots_are_valid():
    state = main_state()
    geometry = state.board.geometry
    second = next(
        v for v in range(54) if v != 0 and v not in geometry.vertex_neighbors[0]
    )
    for vertex in (0, second):
        add_building(state, vertex)
        edge = next(
            e for e in geometry.vertex_edges[vertex] if state.edge_owner[e] == -1
        )
        add_road(state, edge)
    assert_invariants(state)


def test_later_opponent_settlement_does_not_invalidate_existing_roads():
    state = main_state()
    geometry = state.board.geometry
    first = geometry.vertex_edges[0][0]
    middle = next(v for v in geometry.edge_vertices[first] if v != 0)
    second = next(e for e in geometry.vertex_edges[middle] if e != first)
    junction = next(v for v in geometry.edge_vertices[second] if v != middle)
    third = next(e for e in geometry.vertex_edges[junction] if e != second)
    add_building(state, 0)
    for edge in (first, second, third):
        add_road(state, edge)
    assert_invariants(state)
    add_building(state, junction, 1)
    assert_invariants(state)


@pytest.mark.parametrize(
    "phase,last_roll,setup_index,message",
    [
        (Phase.SETUP_SETTLEMENT, 8, 0, "setup cannot"),
        (Phase.ROLL, None, 0, "completed setup"),
        (Phase.ROLL, 8, 8, "cleared last_roll"),
        (Phase.ROBBER_MOVE, 8, 8, "seven"),
        (Phase.ROBBER_STEAL, None, 8, "seven"),
        (Phase.MAIN, None, 8, "dice roll"),
        (Phase.GAME_OVER, 8, 8, "winning current player"),
    ],
)
def test_temporal_phase_errors_are_rejected(phase, last_roll, setup_index, message):
    state = new_game(0)
    state.phase = phase
    state.last_roll = last_roll
    state.setup_index = setup_index
    state.validate()
    with pytest.raises(ValueError, match=message):
        assert_invariants(state)


def test_winning_buildings_require_terminal_phase_and_winning_current_player():
    state = main_state()
    vertices = []
    for vertex in range(54):
        if not set(state.board.geometry.vertex_neighbors[vertex]) & set(vertices):
            vertices.append(vertex)
        if len(vertices) == 6:
            break
    for index, vertex in enumerate(vertices):
        add_building(state, vertex, code=2 if index < 4 else 1)
    state.validate()
    with pytest.raises(ValueError, match="requires GAME_OVER"):
        assert_invariants(state)
    state.phase = Phase.GAME_OVER
    assert_invariants(state)
    state.current_player = 1
    with pytest.raises(ValueError, match="winning current player"):
        assert_invariants(state)


def test_validation_errors_retain_the_first_failing_step_and_action(monkeypatch):
    import catan.validation as validation

    original = validation.assert_invariants
    calls = 0

    def corrupt(state):
        nonlocal calls
        calls += 1
        if calls == 2:
            state.bank[0] -= 1
        original(state)

    monkeypatch.setattr(validation, "assert_invariants", corrupt)
    with pytest.raises(GameValidationError) as raised:
        validate_game(12, 3)
    context = raised.value.context
    assert context["seed"] == 12 and context["players"] == 3
    assert context["step"] == 1 and context["phase"] == "SETUP_SETTLEMENT"
    assert context["action"].startswith("SettlementAction(")
    assert "conserved" in context["error"]


def test_normal_engine_steps_do_not_call_debug_validation(monkeypatch):
    def fail(_):
        raise AssertionError("debug check entered hot path")

    monkeypatch.setattr(GameState, "validate", fail)
    state = new_game(0)
    apply_action(state, SettlementAction(0))
    apply_action_inplace(state, SettlementAction(0))


def run_cli(path, *args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.validate_games", "--report", str(path), *args],
        capture_output=True,
        text=True,
        timeout=45,
    )


def test_parallel_serial_and_functional_reports_replay_identically(tmp_path):
    reports = []
    for name, args in (
        ("serial", ()),
        ("parallel", ("--workers", "2")),
        ("copied", ("--functional",)),
    ):
        path = tmp_path / f"{name}.json"
        result = run_cli(path, "--games", "6", "--max-actions", "20", *args)
        assert result.returncode == 0, result.stderr
        assert "0 completed, 6 capped, 0 failures" in result.stdout
        assert "--games 1 --seed N --players P" in result.stdout
        report = json.loads(path.read_text())
        assert report["invariant_checks"] == 126
        assert report["games_checked"] == report["games_requested"] == 6
        assert report["per_players"]["3"]["games"] == 3
        assert report["per_players"]["4"]["games"] == 3
        assert sum(report["phase_counts"].values()) == 126
        for key in ("elapsed_seconds", "workers", "functional"):
            report.pop(key)
        reports.append(report)
    assert reports[0] == reports[1] == reports[2]


@pytest.mark.parametrize("option", ("--games", "--max-actions", "--workers"))
def test_cli_rejects_nonpositive_workload(tmp_path, option):
    result = run_cli(tmp_path / "unused.json", option, "0")
    assert result.returncode == 2 and "must be positive" in result.stderr


@pytest.mark.parametrize("players", (3, 4))
def test_all_dice_totals_against_independent_production_oracle(players):
    # Use states reached through real setup/building/trade actions, not detached roads.
    state = new_game(14, players)
    policy = Random(14)
    checks = 0
    for step in range(800):
        if is_terminal(state):
            break
        apply_action_inplace(state, choose_action(state, policy))
        if step not in (350, 700):
            continue
        for total in range(2, 13):
            if total == 7:
                continue
            probe = state.copy()
            probe.phase = Phase.ROLL
            probe.resume_phase = None
            probe.free_roads_remaining = 0
            probe.discard_pending[:] = 0
            probe.last_roll = None
            # Exercise independent resource shortages alongside normal supply.
            for resource in range(5):
                available = resource % 3
                moved = max(0, int(probe.bank[resource]) - available)
                probe.hands[0, resource] += moved
                probe.bank[resource] -= moved
            owed = [[0] * 5 for _ in range(players)]
            for vertex, owner in enumerate(probe.vertex_owner.tolist()):
                if owner == -1:
                    continue
                for tile in probe.board.geometry.vertex_tiles[vertex]:
                    if tile != probe.robber_tile and probe.board.numbers[tile] == total:
                        owed[owner][probe.board.resources[tile]] += int(
                            probe.vertex_building[vertex]
                        )
            expected = probe.hands.copy()
            for resource in range(5):
                recipients = [p for p in range(players) if owed[p][resource]]
                demand = sum(owed[p][resource] for p in recipients)
                supply = int(probe.bank[resource])
                if demand <= supply:
                    for player in recipients:
                        expected[player, resource] += owed[player][resource]
                elif len(recipients) == 1:
                    expected[recipients[0], resource] += supply
            for seed in range(1000):
                dice = Random(seed)
                if dice.randint(1, 6) + dice.randint(1, 6) == total:
                    probe.rng.seed(seed)
                    break
            result = apply_action(probe, RollAction())
            assert result.last_roll == total
            np.testing.assert_array_equal(result.hands, expected)
            np.testing.assert_array_equal(result.bank, 19 - expected.sum(axis=0))
            assert_invariants(result)
            checks += 1
    assert checks >= 10


def test_roll_eight_pays_both_tiles_and_double_city_production():
    state = main_state()
    tiles = [tile for tile, number in enumerate(state.board.numbers) if number == 8]
    assert len(tiles) == 2
    first, second = (state.board.geometry.tile_vertices[t][0] for t in tiles)
    add_building(state, first, 0, code=2)
    add_building(state, second, 1)
    state.phase = Phase.ROLL
    state.last_roll = None
    for seed in range(1000):
        dice = Random(seed)
        if dice.randint(1, 6) + dice.randint(1, 6) == 8:
            state.rng.seed(seed)
            break
    result = apply_action(state, RollAction())
    assert result.hands[0, state.board.resources[tiles[0]]] == 2
    assert result.hands[1, state.board.resources[tiles[1]]] == 1
    assert int(result.hands.sum()) == 3
    assert_invariants(result)


def test_cli_writes_first_rule_failure_and_stops_immediately(
    monkeypatch, tmp_path, capsys
):
    from scripts import validate_games as cli

    calls = []

    def fail(seed, players, **kwargs):
        calls.append((seed, players))
        raise GameValidationError(
            seed,
            players,
            1,
            "SETUP_SETTLEMENT",
            SettlementAction(0),
            ValueError("broken conservation"),
        )

    path = tmp_path / "failure.json"
    monkeypatch.setattr(cli, "validate_game", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_games",
            "--games",
            "3",
            "--seed",
            "5",
            "--players",
            "4",
            "--report",
            str(path),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    assert calls == [(5, 4)]
    report = json.loads(path.read_text())
    assert report["games_checked"] == 1 and report["games_requested"] == 3
    assert report["completed"] == report["capped"] == 0
    assert len(report["failures"]) == 1
    assert report["failures"][0]["seed"] == 5
    assert report["failures"][0]["step"] == 1
    assert "broken conservation" in report["failures"][0]["error"]
    assert "1 failures" in capsys.readouterr().out


def test_cli_counts_real_terminal_wins_separately_from_caps(tmp_path):
    path = tmp_path / "win.json"
    result = run_cli(path, "--games", "1", "--seed", "0", "--players", "4")
    assert result.returncode == 0, result.stderr
    report = json.loads(path.read_text())
    assert report["completed"] == 1 and report["capped"] == 0
    assert report["failures"] == report["capped_games"] == []
    assert report["phase_counts"]["GAME_OVER"] == 1
    assert sum(report["per_players"]["4"]["wins"]) == 1
    assert report["invariant_checks"] == report["actions"] + 1


def test_initialization_failures_also_retain_seed_and_zero_step(monkeypatch):
    import catan.validation as validation

    def fail(*args):
        raise ValueError("broken board initialization")

    monkeypatch.setattr(validation, "new_game", fail)
    with pytest.raises(GameValidationError) as raised:
        validate_game(31, 3)
    assert raised.value.context == {
        "seed": 31,
        "players": 3,
        "step": 0,
        "phase": "INITIALIZE",
        "action": "None",
        "error": "broken board initialization",
    }
    assert isinstance(raised.value.__cause__, ValueError)

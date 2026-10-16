"""Frozen decision diagnostics: exact replay, privacy, controls and accounting."""

# ruff: noqa: E402
from collections import Counter
from io import StringIO
import json
from random import Random
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan.board import GEOMETRY, Resource
from catan.state import Phase, new_game
from learning.model import ActorCritic
from learning.ppo import PPOUpdater
from rl.env import Decision
from rl.evaluate import play_match
from rl.policies import POLICIES
from scripts.diagnose_policy import (
    inspect_match,
    override_choice,
    paired_change,
    production_pips,
    summarize,
)


@pytest.fixture(scope="module", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def policy():
    return PPOUpdater(ActorCritic(16, seed=2)).policy()


def test_pips_matches_independent_dice_enumeration():
    board = new_game(3).board
    frequencies = Counter(a + b for a in range(1, 7) for b in range(1, 7))
    for vertex, tiles in enumerate(GEOMETRY.vertex_tiles):
        expected = [0] * 5
        for tile in tiles:
            resource = board.resources[tile]
            if resource != Resource.DESERT:
                expected[resource] += frequencies[board.numbers[tile]]
        assert production_pips(board, vertex) == expected


@pytest.mark.parametrize(
    "phase,variant,expected",
    (
        (Phase.MAIN, "baseline", 219),
        (Phase.MAIN, "deterministic", 219),
        (Phase.MAIN, "greedy_main", 79),
        (Phase.MAIN, "build_first", 79),
        (Phase.MAIN, "greedy_setup", 219),
        (Phase.SETUP_SETTLEMENT, "greedy_setup", 79),
        (Phase.SETUP_ROAD, "greedy_setup", 79),
    ),
)
def test_phase_controls(phase, variant, expected):
    mask = np.zeros(254, dtype=bool)
    mask[[79, 219]] = True
    request = Decision(0, phase, np.zeros(1505, dtype=np.float32), mask)
    assert override_choice(request, 219, 79, variant) == expected


def test_no_build_and_discards_keep_complete_proposal():
    mask = np.zeros(254, dtype=bool)
    mask[[0, 219]] = True
    d = Decision(0, Phase.MAIN, np.zeros(1505, dtype=np.float32), mask)
    assert override_choice(d, 219, 0, "build_first") == 219
    d = Decision(0, Phase.DISCARD, d.observation, None, (4, 4, 0, 0, 0), 4)
    for variant in (
        "baseline",
        "deterministic",
        "greedy_setup",
        "build_first",
        "greedy_main",
    ):
        assert override_choice(d, (4, 0, 0, 0, 0), (2, 2, 0, 0, 0), variant) == (
            4,
            0,
            0,
            0,
            0,
        )


@pytest.mark.parametrize(
    "players,variant",
    (
        (3, "baseline"),
        (4, "baseline"),
        (3, "greedy_setup"),
        (4, "build_first"),
        (3, "greedy_main"),
    ),
)
def test_instrumentation_matches_plain_game_rng(players, variant):
    p = policy()
    weights = {k: v.clone() for k, v in p.model.state_dict().items()}
    global_rng = torch.get_rng_state().clone()
    trace = StringIO()
    result = inspect_match(
        p, 9, players, 1, variant=variant, max_actions=120, trace=trace, validate=True
    )

    def candidate(request, rng):
        proposed = p((request,), (rng,))[0].action
        teacher = POLICIES["greedy"](request, Random(0))
        if variant == "greedy_setup" and request.phase in (
            Phase.SETUP_SETTLEMENT,
            Phase.SETUP_ROAD,
        ):
            return teacher
        if variant == "greedy_main" and request.phase == Phase.MAIN:
            return teacher
        if (
            variant == "build_first"
            and request.phase == Phase.MAIN
            and request.mask[72:180].any()
        ):
            return teacher
        return proposed

    callbacks = [POLICIES["greedy"]] * players
    callbacks[1] = candidate
    plain = play_match(callbacks, 9, policy_seed=17, max_actions=120, validate=True)
    assert result["fingerprint"] == plain.fingerprint
    assert result["scores"] == list(plain.scores)
    assert result["winner"] == plain.winner and result["actions"] == plain.actions
    assert sum(result["completed_turns"]) == plain.completed_turns
    for name, value in p.model.state_dict().items():
        assert torch.equal(value, weights[name])
    assert torch.equal(torch.get_rng_state(), global_rng)
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    actor_turns = [0] * players
    first_builds = [{} for _ in range(players)]
    for row in rows:
        actor = row["actor"]
        if row["phase"] == "MAIN" and row["action_type"] in (
            "SettlementAction",
            "CityAction",
        ):
            first_builds[actor].setdefault(row["action_type"], actor_turns[actor] + 1)
        if row["action_type"] == "EndTurnAction":
            actor_turns[actor] += 1
    assert result["first_build"] == first_builds
    assert len(rows) == plain.actions
    assert [r["step"] for r in rows] == list(range(1, plain.actions + 1))
    assert all(
        "hands" not in r and "observation" not in r and "development_deck" not in r
        for r in rows
    )
    assert all(len(r["candidate_hand_before"]) == 5 for r in rows)
    for r in rows:
        if r["actor"] != 1:
            assert (
                r["neural_softmax_probabilities"] is None and r["neural_value"] is None
            )


def test_trace_is_optional_and_does_not_change_choices():
    p = policy()
    plain = inspect_match(p, 12, 3, 0, max_actions=100)
    traced = inspect_match(p, 12, 3, 0, max_actions=100, trace=StringIO())
    assert plain == traced
    assert sum(sum(c.values()) for c in plain["action_counts"]) == plain["actions"]
    summary = summarize([plain])
    assert summary["cutoffs"] == 1 and summary["wins"] == 0 and summary["losses"] == 0


def test_invalid_controller_rejected_without_rng_or_model_changes():
    p = policy()
    before = torch.get_rng_state().clone()
    for kwargs in ({"variant": "unknown"}, {"opponent": "unknown"}, {"seat": 3}):
        options = {"seed": 0, "players": 3, "seat": 0}
        options.update(kwargs)
        with pytest.raises(ValueError):
            inspect_match(p, **options)
    assert torch.equal(before, torch.get_rng_state())


def test_cluster_pairs_and_duplicate_rejection():
    baseline = [
        dict(seed=s, players=n, candidate_seat=p, winner=0)
        for s in (1, 2, 3)
        for n in (3, 4)
        for p in range(n)
    ]
    candidate = [dict(g, winner=g["candidate_seat"]) for g in baseline]
    result = paired_change(baseline, candidate)
    assert result["paired_games"] == 21 and result["seed_clusters"] == 3
    assert result["win_rate_change"] == pytest.approx(5 / 7)
    assert result["seed_cluster_bootstrap_95_percent_interval"] == pytest.approx(
        [5 / 7, 5 / 7]
    )
    with pytest.raises(ValueError):
        paired_change(baseline + baseline[:1], candidate)
    with pytest.raises(ValueError):
        paired_change(baseline, candidate[:-1])


def test_cli_writes_bounded_json_and_refuses_existing_output(tmp_path):
    learner = PPOUpdater(ActorCritic(8, seed=0))
    checkpoint = tmp_path / "model.pt"
    learner.save(checkpoint)
    folder = tmp_path / "report"
    command = [
        sys.executable,
        "-m",
        "scripts.diagnose_policy",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(folder),
        "--seeds",
        "1",
        "--max-actions",
        "20",
        "--variants",
        "baseline",
        "greedy_setup",
        "--trace-seed",
        "3000003000",
        "--trace-players",
        "3",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads((folder / "summary.json").read_text())
    assert report["weights_unchanged"] and report["policy_version"] == 0
    assert set(report["summaries"]) == {"baseline", "greedy_setup"}
    for variant, summary in report["summaries"].items():
        assert (
            summary["games"] == 7 and summary["cutoffs"] == 7 and summary["wins"] == 0
        )
        assert len(json.loads((folder / f"{variant}.json").read_text())) == 7
        assert len((folder / f"trace-{variant}.jsonl").read_text().splitlines()) == 20
    again = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert again.returncode != 0 and "FileExistsError" in again.stderr


@pytest.mark.parametrize(
    "variant,phase,proposed,teacher,expected",
    (
        ("greedy_trades", Phase.MAIN, 200, 219, 219),
        ("greedy_trades", Phase.MAIN, 219, 200, 200),
        ("greedy_trades", Phase.MAIN, 219, 79, 219),
        ("greedy_cards", Phase.MAIN, 225, 219, 219),
        ("greedy_cards", Phase.ROLL, 220, 226, 226),
        ("greedy_cards", Phase.MAIN, 219, 200, 219),
        ("greedy_setup_main", Phase.SETUP_SETTLEMENT, 79, 80, 80),
        ("greedy_setup_main", Phase.MAIN, 219, 79, 79),
    ),
)
def test_targeted_controls(variant, phase, proposed, teacher, expected):
    mask = np.zeros(254, dtype=bool)
    mask[[proposed, teacher]] = True
    request = Decision(0, phase, np.zeros(1505, dtype=np.float32), mask)
    assert override_choice(request, proposed, teacher, variant) == expected


def test_discard_control_is_complete_and_does_not_affect_other_phases():
    request = Decision(
        0, Phase.DISCARD, np.zeros(1505, dtype=np.float32), None, (4, 4, 0, 0, 0), 4
    )
    assert override_choice(
        request, (4, 0, 0, 0, 0), (2, 2, 0, 0, 0), "greedy_discards"
    ) == (2, 2, 0, 0, 0)
    request = Decision(0, Phase.MAIN, request.observation, np.ones(254, dtype=bool))
    assert override_choice(request, 219, 225, "greedy_discards") == 219

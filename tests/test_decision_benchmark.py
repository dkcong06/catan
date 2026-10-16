"""Frozen development locations, bounded selection, labels and hash validation."""
# ruff: noqa: E402

from random import Random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan.engine import apply_action_inplace, winner
from learning.model import ActorCritic, EconomicActorCritic
from learning.ppo import PPOUpdater
from rl.actions import indexed_action
from rl.collector import ReferencePolicy
from scripts.benchmark_decisions import (
    CATEGORIES,
    CasePool,
    build_suite,
    describe,
    inspect_game,
    load_suite,
    score_model,
)


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_pool_is_bounded_order_independent_and_does_not_consume_randomness():
    left, right = CasePool(3), CasePool(3)
    rng = Random(91)
    state = rng.getstate()
    cases = [{"role": [670000030, 3, 0, "greedy"], "step": i} for i in range(20)]
    for case in cases:
        left.offer("placement", case)
    for case in reversed(cases):
        right.offer("placement", case)
    assert left.cases() == right.cases()
    assert len(left.cases()) == 3 and rng.getstate() == state
    with pytest.raises(ValueError):
        CasePool(0)


def test_frozen_suite_reproduces_and_zero_migration_scores_match(tmp_path):
    source = PPOUpdater(ActorCritic(16, seed=9))
    source.save(tmp_path / "source.pt")
    for name in ("a", "b"):
        build_suite(
            tmp_path / "source.pt",
            tmp_path / name,
            boards=1,
            seed=670000031,
            include_historical=False,
            max_actions=40,
            per_category=8,
        )
    a, cases_a, arrays_a = load_suite(tmp_path / "a")
    b, cases_b, arrays_b = load_suite(tmp_path / "b")
    assert cases_a == cases_b
    assert a == b
    for name in arrays_a:
        np.testing.assert_array_equal(arrays_a[name], arrays_b[name])
    before = {k: v.clone() for k, v in source.model.state_dict().items()}
    migrated = EconomicActorCritic.from_base(source.model)
    old = score_model(source.model, tmp_path / "a")
    new = score_model(migrated, tmp_path / "a")
    assert old["groups"] == new["groups"]
    for key in before:
        assert torch.equal(before[key], source.model.state_dict()[key])
    with pytest.raises(FileExistsError):
        build_suite(tmp_path / "source.pt", tmp_path / "a", boards=1)
    (tmp_path / "a" / "cases.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        load_suite(tmp_path / "a")


def test_labels_and_probe_replay_exact_game_engine_and_policy_rng():
    pool = CasePool(10)
    role = (670000032, 3, 1, "greedy")
    a, captured = inspect_game(
        ReferencePolicy("greedy"), role, pool, validate=True, probe=2
    )
    b, _ = inspect_game(ReferencePolicy("greedy"), role, validate=True)
    assert a == b and not a["cutoff"]
    assert captured is not None
    categories = {category for category, _ in pool.cases()}
    assert {"placement", "affordable_build", "trade"} <= categories
    assert categories <= set(CATEGORIES)
    winning_role = (role[0], role[1], a["winner"], role[3])
    inspect_game(ReferencePolicy("greedy"), winning_role, pool, validate=True)
    assert any(category == "win" for category, _ in pool.cases())
    for category, case in pool.cases():
        if category != "win":
            continue
        _, probe = inspect_game(
            ReferencePolicy("greedy"), tuple(case["role"]), probe=case["step"]
        )
        state, decision, _, _, wins = probe
        saved = state.copy()
        assert not wins[180:226].any() and not wins[227:].any()
        for choice in np.flatnonzero(wins):
            fork = state.copy()
            apply_action_inplace(fork, indexed_action(fork, int(choice)))
            assert winner(fork) == decision.player
        assert state == saved


def test_discard_gap_lower_bound_matches_exhaustive_legal_counts():
    from itertools import product
    from types import SimpleNamespace
    from catan.state import Phase

    hand = np.array((2, 3, 2, 1, 0))
    decision = SimpleNamespace(player=0, phase=Phase.DISCARD)
    state = SimpleNamespace(hands=hand[None])
    choice = (1, 1, 1, 1, 0)
    detail = describe(state, decision, choice, np.zeros(495))
    from scripts.benchmark_decisions import gaps

    possible = [
        gaps(hand - counts)
        for counts in product(*(range(v + 1) for v in hand))
        if sum(counts) == 4
    ]
    assert detail["best_individual_gaps_after"] == np.min(possible, axis=0).tolist()

"""Experiment model compatibility, pause/resume and four-arm mechanics."""

# ruff: noqa: E402
from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning.evaluation import EvaluationPaused, evaluate_neural
from learning.hypotheses import CaseSpec, ExperimentConfig, HypothesisRun, digest
from learning.imitation import Demonstrations
from learning.model import ActorCritic, KnowledgeActorCritic
from learning.policy import NeuralPolicy
from learning.ppo import PPOConfig, PPOUpdater
from rl.collector import ReferencePolicy, collect_rollouts
from rl.hypotheses import HypothesisEnvFactory, PROGRESS_REWARD
from scripts.train_hypotheses import run_experiment


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def prepared(path, hidden=16):
    base = ActorCritic(hidden)
    updater = PPOUpdater(
        KnowledgeActorCritic.from_base(base), PPOConfig(epochs=1, gae_lambda=1)
    )
    updater.save(path)
    return updater


@pytest.mark.parametrize("width", (16, 128))
def test_zero_projection_matches_original_outputs_rng_and_choices(width):
    base = ActorCritic(width, seed=19)
    rng = torch.get_rng_state().clone()
    model = KnowledgeActorCritic.from_base(base)
    assert torch.equal(rng, torch.get_rng_state())
    batch = collect_rollouts(NeuralPolicy(base), (420000030,), max_actions=100)
    old = torch.from_numpy(batch.observations.copy())
    new = torch.cat((old, torch.ones((len(old), 30))), dim=1)
    with torch.no_grad():
        for left, right in zip(base(old), model(new), strict=True):
            assert torch.equal(left, right)
    migrated = collect_rollouts(
        NeuralPolicy(model),
        (420000030,),
        max_actions=100,
        env_factory=HypothesisEnvFactory(knowledge=True),
    )
    assert migrated.episodes == batch.episodes
    for field in ("actions", "log_probs"):
        assert np.array_equal(getattr(migrated, field), getattr(batch, field))
    assert np.array_equal(migrated.discards.counts, batch.discards.counts)
    assert np.array_equal(migrated.discards.log_probs, batch.discards.log_probs)


@pytest.mark.parametrize("shaped", (False, True))
def test_update_checkpoint_and_profile_rejection_preserve_legacy(tmp_path, shaped):
    reward = PROGRESS_REWARD if shaped else "terminal_win_v1"
    updater = PPOUpdater(
        KnowledgeActorCritic(16),
        PPOConfig(epochs=1, gae_lambda=1),
        reward_scheme=reward,
    )
    batch = collect_rollouts(
        updater.policy(),
        (420000031,),
        max_actions=100,
        env_factory=HypothesisEnvFactory(point_progress=shaped, knowledge=True),
    )
    rng = updater.generator.get_state().clone()
    with pytest.raises(ValueError, match="profile"):
        updater.update(replace(batch, observation_schema=2))
    assert updater.updates == 0 and not updater.optimizer.state
    assert torch.equal(rng, updater.generator.get_state())
    updater.update(batch)
    path = tmp_path / "new.pt"
    updater.save(path)
    restored = PPOUpdater.load(path)
    assert restored.reward_scheme == reward and restored.model.schema == (2, 3, 1)
    assert restored.updates == 1
    for key, tensor in updater.model.state_dict().items():
        assert torch.equal(tensor, restored.model.state_dict()[key])
    legacy = PPOUpdater(ActorCritic(16))
    legacy.save(tmp_path / "old.pt")
    assert PPOUpdater.load(tmp_path / "old.pt").model.schema == (1, 2, 1)


def test_demonstration_roundtrip_and_sparse_subset(tmp_path):
    batch = collect_rollouts(
        ReferencePolicy("greedy"),
        (420000032,),
        max_actions=120,
        env_factory=HypothesisEnvFactory(),
    )
    data = Demonstrations.from_rollouts(batch)
    selected = data.select(np.arange(0, len(data), 2))
    selected.save(tmp_path / "teacher.npz")
    restored = Demonstrations.load(tmp_path / "teacher.npz")
    assert restored.observation_schema == 3
    assert np.array_equal(restored.observations, selected.observations)
    assert np.array_equal(restored.discards.counts, selected.discards.counts)


def test_pause_resume_keeps_remaining_time_and_exact_next_update(tmp_path):
    initial = tmp_path / "initial.pt"
    prepared(initial)
    config = ExperimentConfig(
        training_seconds=120,
        replicates=1,
        test_seeds=1,
        teacher_segments=2,
        teacher_validation_segments=2,
        games_per_update=2,
        max_actions=40,
        max_updates=3,
    )
    spec = CaseSpec("K", 0, 0, digest(initial))
    left = HypothesisRun(
        tmp_path / "a", config, spec, initial, initial_checkpoint=initial
    )
    right = HypothesisRun(
        tmp_path / "b", config, spec, initial, initial_checkpoint=initial
    )
    try:
        left.step()
        right.step()
        remaining = right._remaining()
        right.save(status="paused")
        saved = json.loads((tmp_path / "b/state.json").read_text())["hypotheses"][
            "remaining_training_seconds"
        ]
        right.close()
        right = HypothesisRun(tmp_path / "b", config, spec, initial, resume=True)
        assert right._remaining() <= saved <= remaining + 0.1
        first, second = left.step(), right.step()
        assert first["episodes"] == second["episodes"]
        for key, tensor in left.updater.model.state_dict().items():
            assert torch.equal(tensor, right.updater.model.state_dict()[key])
        right.close()
        with pytest.raises(ValueError, match="configuration"):
            HypothesisRun(
                tmp_path / "b",
                replace(config, training_seconds=121),
                spec,
                initial,
                resume=True,
            )
        state_path = tmp_path / "b/state.json"
        tampered = json.loads(state_path.read_text())
        tampered["config"]["policy_seed"] += 1
        state_path.write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match="configuration"):
            HypothesisRun(tmp_path / "b", config, spec, initial, resume=True)
    finally:
        left.close()
        right.close()


def test_evaluation_cursor_resumes_exactly_and_rejects_changed_model(tmp_path):
    updater = prepared(tmp_path / "initial.pt")
    factory = HypothesisEnvFactory(knowledge=True)
    path = tmp_path / "partial.json"
    calls = 0

    def pause():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(EvaluationPaused):
        evaluate_neural(
            updater.policy(),
            (420000033,),
            max_actions=30,
            batch_size=1,
            env_factory=factory,
            progress_path=path,
            progress_identity="original",
            should_pause=pause,
        )
    cursor = json.loads(path.read_text())
    assert cursor["completed_chunks"] == 2 and len(cursor["records"]) == 2
    with pytest.raises(ValueError, match="changed"):
        evaluate_neural(
            updater.policy(),
            (420000033,),
            max_actions=30,
            batch_size=1,
            env_factory=factory,
            progress_path=path,
            progress_identity="changed",
        )
    resumed = evaluate_neural(
        updater.policy(),
        (420000033,),
        max_actions=30,
        batch_size=1,
        env_factory=factory,
        progress_path=path,
        progress_identity="original",
    )
    direct = evaluate_neural(
        updater.policy(),
        (420000033,),
        max_actions=30,
        batch_size=1,
        env_factory=factory,
    )
    assert resumed == direct


def test_four_arm_smoke_preserves_source_and_uses_final_checkpoint(tmp_path):
    source = tmp_path / "source.pt"
    PPOUpdater(ActorCritic(16)).save(source)
    original_hash = digest(source)
    config = ExperimentConfig(
        training_seconds=60,
        replicates=1,
        test_seeds=1,
        teacher_segments=2,
        teacher_validation_segments=2,
        games_per_update=2,
        max_actions=30,
        max_updates=1,
    )
    root = tmp_path / "experiment"
    result = run_experiment(root, source, config=config)
    assert result["status"] == "completed" and digest(source) == original_hash
    assert len(result["cases"]) == 4
    assert set(result["comparisons"]) == {"P", "R", "K"}
    for case in result["cases"]:
        assert case["update"] == 1 and case["checkpoint"] == "u00000001.pt"
        assert case["totals"]["episodes"] == case["totals"]["cutoffs"] == 2
        assert case["config"]["opponent_schedule"] == [
            "greedy",
            "frozen_original",
            "greedy",
            "random",
        ]
        assert case["budget"]["clock"] == "charged_active_v1"
    assert (
        next(c for c in result["cases"] if c["arm"] == "R")["reward_scheme"]
        == PROGRESS_REWARD
    )
    assert run_experiment(root, source, resume=True) == result


def test_requested_pause_exits_without_spending_training_allowance(tmp_path):
    initial = tmp_path / "initial.pt"
    prepared(initial)
    config = ExperimentConfig(
        training_seconds=120,
        replicates=1,
        test_seeds=1,
        teacher_segments=2,
        teacher_validation_segments=2,
        games_per_update=2,
        max_actions=30,
        max_updates=1,
    )
    spec = CaseSpec("B", 0, 0, digest(initial))
    run = HypothesisRun(
        tmp_path / "case", config, spec, initial, initial_checkpoint=initial
    )
    try:
        remaining = run._remaining()
        assert run.train(lambda: True)["status"] == "paused"
        saved = json.loads((tmp_path / "case/state.json").read_text())
        assert saved["update"] == 0
        assert 0 <= remaining - saved["hypotheses"]["remaining_training_seconds"] < 2
    finally:
        run.close()

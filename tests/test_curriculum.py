"""Teacher-label, mixed-opponent and learner-only PPO correctness gates."""

# ruff: noqa: E402
from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan.state import Phase
from learning.evaluation import EvaluationPolicy, compare_evaluations, evaluate_neural
from learning.imitation import (
    Demonstrations,
    ImitationConfig,
    ImitationTrainer,
    phase_weights,
)
from learning.model import ActorCritic
from learning.policy import action_inputs
from learning.ppo import PPOConfig, PPOUpdater
from learning.training import RunConfig, TrainingRun
from rl.actions import _discard_count_mask
from rl.collector import ReferencePolicy, collect_rollouts
from test_training import same_models, small_config


@pytest.fixture(scope="module", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def demonstrations():
    batch = collect_rollouts(ReferencePolicy("greedy"), (8, 9, 10, 11), max_actions=400)
    return Demonstrations.from_rollouts(batch)


def test_numeric_demonstration_roundtrip_forced_pruning_and_no_ppo(
    tmp_path, demonstrations
):
    data = demonstrations
    assert (data.masks[data.actions >= 0].sum(axis=1) > 1).all()
    assert len(data.discards.rows) > 0
    path = tmp_path / "teacher.npz"
    data.save(path)
    restored = Demonstrations.load(path)
    for name in ("observations", "masks", "actions", "phases"):
        np.testing.assert_array_equal(getattr(data, name), getattr(restored, name))
        assert not getattr(restored, name).flags.writeable
    for name in ("rows", "hands", "counts", "log_probs"):
        np.testing.assert_array_equal(
            getattr(data.discards, name), getattr(restored.discards, name)
        )
    assert restored.teacher_tag == data.teacher_tag
    with pytest.raises(FileExistsError):
        data.save(path)
    with pytest.raises(ValueError, match="bound|rows"):
        Demonstrations.load(path, max_rows=1)
    with pytest.raises(ValueError, match="RolloutBatch"):
        PPOUpdater(ActorCritic(16)).update(data)


def test_phase_balancing_equal_mass_and_uniform_control():
    phases = np.array([1, 1, 1, 2, 3, 3], dtype=np.int8)
    weighted = phase_weights(phases)
    assert weighted.mean() == pytest.approx(1)
    for phase in np.unique(phases):
        assert weighted[phases == phase].sum() == pytest.approx(2)
    np.testing.assert_array_equal(phase_weights(phases, False), np.ones(6))
    with pytest.raises(ValueError):
        phase_weights(np.array([]))


def test_likelihood_only_path_preserves_joint_logs_values_and_exact_gradients(
    demonstrations,
):
    rows = np.arange(min(256, len(demonstrations)))
    inputs = action_inputs(demonstrations, rows)
    model = ActorCritic(16, seed=2)
    normal = model.evaluate_actions(*inputs)
    (-normal.log_probs.mean()).backward()
    gradients = {
        n: None if p.grad is None else p.grad.clone()
        for n, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    fast = model.evaluate_actions(*inputs, compute_entropy=False)
    assert torch.equal(normal.log_probs, fast.log_probs)
    assert torch.equal(normal.values, fast.values) and not fast.entropies.any()
    (-fast.log_probs.mean()).backward()
    for name, parameter in model.named_parameters():
        before = gradients[name]
        assert (before is None) == (parameter.grad is None)
        if before is not None:
            assert torch.equal(before, parameter.grad)


def test_joint_discard_teacher_likelihood_matches_independent_prefix_oracle(
    demonstrations,
):
    data = demonstrations
    model = ActorCritic(16, seed=9)
    rows = data.discards.rows[:3]
    inputs = action_inputs(data, rows)
    actual = model.evaluate_actions(*inputs).log_probs
    hidden = model(inputs[0]).hidden
    expected = []
    for i, (hand, counts) in enumerate(
        zip(inputs[4].numpy(), inputs[5].numpy(), strict=True)
    ):
        prefix = torch.zeros((1, 5))
        total = torch.zeros(())
        for resource in range(5):
            logits = model.discard_logits(hidden[i : i + 1], prefix, resource)[0]
            feasible = _discard_count_mask(
                hand, int(counts.sum()), tuple(map(int, counts[:resource]))
            )
            logs = torch.log_softmax(
                logits.masked_fill(~torch.from_numpy(feasible), -torch.inf), dim=0
            )
            total = total + logs[counts[resource]]
            prefix[0, resource] = int(counts[resource])
        expected.append(total)
    torch.testing.assert_close(actual, torch.stack(expected))


def test_imitation_learns_labels_with_local_rng_and_preserves_input(demonstrations):
    model = ActorCritic(16, seed=4)
    trainer = ImitationTrainer(
        model, ImitationConfig(epochs=8, learning_rate=0.01, minibatch_size=64)
    )
    before_rng = torch.random.get_rng_state().clone()
    before_obs = demonstrations.observations.copy()
    result = trainer.fit(demonstrations)
    assert result["loss_after"] < result["loss_before"]
    assert result["optimizer_steps"] > 0
    assert torch.equal(before_rng, torch.random.get_rng_state())
    np.testing.assert_array_equal(before_obs, demonstrations.observations)
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize(
    "field,value",
    (
        ("epochs", 0),
        ("epochs", True),
        ("minibatch_size", 0),
        ("learning_rate", float("nan")),
        ("learning_rate", 2),
        ("max_grad_norm", 0),
        ("balance_phases", 1),
    ),
)
def test_invalid_imitation_config(field, value):
    with pytest.raises(ValueError):
        ImitationConfig(**{field: value})


@pytest.mark.parametrize("damage", ("observation", "action", "discard", "phase"))
def test_bad_teacher_data_rejects_before_model_adam_or_shuffle_mutation(
    demonstrations, damage
):
    data = demonstrations
    if damage == "observation":
        observations = data.observations.copy()
        observations[0, 0] = np.nan
        data = replace(data, observations=observations)
    elif damage == "action":
        actions = data.actions.copy()
        actions[0] = 999
        data = replace(data, actions=actions)
    elif damage == "phase":
        phases = data.phases.copy()
        phases[0] = Phase.GAME_OVER
        data = replace(data, phases=phases)
    else:
        counts = data.discards.counts.copy()
        counts[0, 0] = 20
        data = replace(data, discards=replace(data.discards, counts=counts))
    trainer = ImitationTrainer(ActorCritic(16))
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    shuffle = trainer.generator.get_state().clone()
    with pytest.raises(ValueError):
        trainer.fit(data)
    assert trainer.optimizer_steps == 0 and not trainer.optimizer.state
    assert torch.equal(shuffle, trainer.generator.get_state())
    for k, v in weights.items():
        assert torch.equal(v, trainer.model.state_dict()[k])


@pytest.mark.parametrize("opponent", ("greedy", "random", "neural"))
@pytest.mark.parametrize("players", (3, 4))
def test_mixed_learner_records_are_fresh_ppo_and_opponent_views_are_absent(
    opponent, players
):
    updater = PPOUpdater(ActorCritic(16, seed=3), PPOConfig(epochs=1))
    learner = updater.policy()
    reference = (
        PPOUpdater(ActorCritic(16, seed=8)).policy()
        if opponent == "neural"
        else opponent
    )
    routed = EvaluationPolicy(learner, reference, 1)
    full = collect_rollouts(routed, (12, 13), n_players=players, max_actions=120)
    selected = collect_rollouts(
        routed,
        (12, 13),
        n_players=players,
        max_actions=120,
        record_players=(1,),
        policy_tag=learner.tag,
    )
    assert full.episodes == selected.episodes
    assert np.all(selected.players == 1) and np.isfinite(selected.values).all()
    assert np.all(selected.last_rows[:, [0, 2, 3]] == -1)
    assert not selected.final_observations[:, [0, 2, 3]].any()
    np.testing.assert_array_equal(
        full.observations[full.players == 1], selected.observations
    )
    with pytest.raises(ValueError):
        updater.update(full)
    with pytest.raises(ValueError):
        updater.update(replace(full, policy_tag=learner.tag))
    result = updater.update(selected)
    assert result.rows == len(selected) and updater.updates == 1


def test_mixed_resume_keeps_seed_schedule_history_and_exact_update(tmp_path):
    config = replace(
        small_config(),
        max_updates=4,
        opponent_schedule=("greedy", "previous"),
        tuning_opponent="greedy",
    )
    a = TrainingRun(tmp_path / "a", config=config, ppo_config=PPOConfig(epochs=1))
    b = TrainingRun(tmp_path / "b", config=config, ppo_config=PPOConfig(epochs=1))
    try:
        a.initialize_baselines()
        b.initialize_baselines()
        for _ in range(2):
            a.step()
            b.step()
        deadline = b.deadline
        b.close()
        b = TrainingRun(tmp_path / "b", resume=True)
        assert b.deadline == deadline
        for _ in range(2):
            left, right = a.step(), b.step()
            assert left["opponent_tag"] == right["opponent_tag"]
            assert left["learner_seat"] == right["learner_seat"]
            assert left["episodes"] == right["episodes"]
            same_models(a, b)
        assert a.totals["learner_rows"] < a.totals["actions"]
        assert len(a._opponents) <= 2
        assert b.updater.config == a.updater.config
    finally:
        a.close()
        b.close()


def test_warm_start_copies_weights_but_resets_optimizer_and_preserves_source(tmp_path):
    source = PPOUpdater(ActorCritic(16, seed=0), PPOConfig(epochs=1))
    batch = collect_rollouts(source.policy(), (17,), max_actions=30)
    source.update(batch)
    path = tmp_path / "source.pt"
    source.save(path)
    original = path.read_bytes()
    run = TrainingRun(tmp_path / "run", config=small_config(), initial_checkpoint=path)
    try:
        assert run.updater.updates == 0 and not run.updater.optimizer.state
        assert run.initial_source["source_update"] == 1
        for k, v in source.model.state_dict().items():
            assert torch.equal(v, run.updater.model.state_dict()[k])
        assert path.read_bytes() == original
    finally:
        run.close()


def test_schema_one_selfplay_run_remains_readable(tmp_path):
    root = tmp_path / "legacy"
    run = TrainingRun(root, config=small_config())
    run.close()
    path = root / "state.json"
    state = json.loads(path.read_text())
    state["schema"] = 1
    state.pop("initial_source")
    state["config"].pop("opponent_schedule")
    state["config"].pop("tuning_opponent")
    state["totals"].pop("learner_rows")
    state["totals"].pop("learner_wins")
    path.write_text(json.dumps(state))
    restored = TrainingRun(root, resume=True)
    try:
        assert restored.config.opponent_schedule == ("self",)
        assert restored.initial_source is None
        assert restored.totals["learner_rows"] == 0
    finally:
        restored.close()


@pytest.mark.parametrize(
    "field,value",
    (
        ("opponent_schedule", ()),
        ("opponent_schedule", ("unknown",)),
        ("tuning_opponent", "random"),
    ),
)
def test_invalid_curriculum_config(field, value):
    with pytest.raises(ValueError):
        RunConfig(**{field: value})


def test_greedy_comparison_counts_only_matching_reference_roles():
    policy = PPOUpdater(ActorCritic(16)).policy()
    report = evaluate_neural(policy, (21, 22), max_actions=25)
    greedy = compare_evaluations(report, report, opponent="greedy", draws=100)
    assert greedy["paired_games"] == 14 and greedy["baseline"]["cutoffs"] == 14
    assert greedy["win_rate_all_change"] == 0
    with pytest.raises(ValueError):
        compare_evaluations(report, report, opponent="unknown")


def test_complete_three_arm_comparison_keeps_sources_and_seeds_separate(tmp_path):
    from scripts.compare_training import compare_training

    path = tmp_path / "source.pt"
    PPOUpdater(ActorCritic(16)).save(path)
    original = path.read_bytes()
    output = tmp_path / "comparison"
    result = compare_training(
        output,
        path,
        seconds_per_arm=120,
        replicates=1,
        teacher_games=2,
        teacher_epochs=1,
        games_per_update=2,
        max_updates=1,
        max_actions=30,
        tune_seeds=1,
        test_seeds=1,
    )
    assert result["status"] == "completed" and result["source_unchanged"]
    assert path.read_bytes() == original
    assert len(result["cases"]) == 3
    for case in result["cases"]:
        report = case["result"]
        assert report["update"] == 1
        assert report["totals"]["episodes"] == report["totals"]["cutoffs"] == 2
        assert report["totals"]["wins"] == 0
        assert report["config"]["tuning_opponent"] == "greedy"
        assert report["initial_source"]["optimizer_reset"]
        assert report["config"]["training_seed_base"] < 1000000000
    warm = result["cases"][-1]
    assert warm["charged_setup_seconds"] > warm["warm_start"]["seconds"]
    assert warm["result"]["config"]["budget_seconds"] < 120
    assert result["versus_selfplay"]["imitation_mixed"]["paired_games"] == 7
    assert (output / "comparison.json").exists()

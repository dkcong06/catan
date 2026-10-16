"""Direct-gradient isolation, balanced coverage, milestone curves and exact recovery."""
# ruff: noqa: E402

from collections import Counter
from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from learning import evaluation
from learning.model import ActorCritic, KnowledgeActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from learning.training import RunConfig, TrainingRun, balanced_assignment
from scripts.compare_critic import ComparisonConfig, CriticRun, run_comparison


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def action_fixture(model):
    obs = torch.rand(
        (5, model.observation_dim), generator=torch.Generator().manual_seed(92)
    )
    mask = torch.zeros((5, 254), dtype=torch.bool)
    mask[:3, [1, 5, 225]] = True
    return (
        obs,
        mask,
        torch.tensor([1, 5, 225, -1, -1]),
        torch.tensor([3, 4]),
        torch.tensor([[4, 3, 2, 2, 0], [0, 5, 4, 1, 0]]),
        torch.tensor([[1, 1, 1, 2, 0], [0, 2, 2, 1, 0]]),
    )


@pytest.mark.parametrize("model_class", (ActorCritic, KnowledgeActorCritic))
def test_shield_preserves_forward_actor_gradients_and_trains_value_head_only(
    model_class,
):
    model = model_class(16, seed=7)
    inputs = action_fixture(model)
    ordinary = model.evaluate_actions(*inputs)
    shielded = model.evaluate_actions(*inputs, critic_updates_trunk=False)
    assert all(torch.equal(a, b) for a, b in zip(ordinary, shielded, strict=True))
    params = tuple(model.parameters())
    actor_a = torch.autograd.grad(ordinary.log_probs.sum(), params, allow_unused=True)
    actor_b = torch.autograd.grad(shielded.log_probs.sum(), params, allow_unused=True)
    for a, b in zip(actor_a, actor_b, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b)
    ordinary = model.evaluate_actions(*inputs)
    shielded = model.evaluate_actions(*inputs, critic_updates_trunk=False)
    grad_a = dict(
        zip(
            dict(model.named_parameters()),
            torch.autograd.grad(ordinary.values.sum(), params, allow_unused=True),
            strict=True,
        )
    )
    grad_b = dict(
        zip(
            dict(model.named_parameters()),
            torch.autograd.grad(shielded.values.sum(), params, allow_unused=True),
            strict=True,
        )
    )
    assert grad_a["trunk.0.weight"].norm() > 0
    assert all(
        grad_b[name] is None
        for name in grad_b
        if name.startswith(("trunk.", "knowledge."))
    )
    assert torch.allclose(grad_a["output.weight"], grad_b["output.weight"], atol=2e-6)
    assert torch.equal(grad_a["output.bias"], grad_b["output.bias"])
    assert grad_b["output.weight"][-1].norm() > 0
    assert not grad_b["output.weight"][:-1].any()
    with pytest.raises(ValueError, match="Boolean"):
        model.evaluate_actions(*inputs, critic_updates_trunk=1)


def test_balanced_cartesian_coverage_with_duplicate_opponent_weights_and_cycle_rotation():
    schedule = ("greedy", "frozen_original", "greedy", "random")
    for cycle in range(5):
        records = [balanced_assignment(cycle * 28 + i, schedule) for i in range(28)]
        counted = Counter(records)
        assert {p for p, _, o in records if o == "greedy"} == {3, 4}
        for players in (3, 4):
            for seat in range(players):
                assert counted[players, seat, "greedy"] == 2
                assert counted[players, seat, "random"] == 1
                assert counted[players, seat, "frozen_original"] == 1
        assert sum(p == 3 for p, _, _ in records) == 12
        assert sum(p == 4 for p, _, _ in records) == 16
    with pytest.raises(ValueError):
        balanced_assignment(-1, schedule)


@pytest.mark.parametrize(
    "changes",
    (
        {"balanced_schedule": 1},
        {"milestones": (2, 1)},
        {"milestones": (1, 1)},
        {"milestones": (0,)},
        {"milestones": (100001,)},
        {"milestones": (True,)},
    ),
)
def test_invalid_scheduling_controls(changes):
    with pytest.raises(ValueError):
        RunConfig(**changes)


@pytest.mark.parametrize("value", (0, 1, None, "false"))
def test_critic_flag_requires_bool(value):
    with pytest.raises(ValueError, match="Boolean"):
        PPOConfig(critic_updates_trunk=value)


def small_run(**kwargs):
    return RunConfig(
        budget_seconds=300,
        max_updates=5,
        games_per_update=1,
        active_games=1,
        max_actions=16,
        evaluation_max_actions=16,
        tune_seeds=1,
        test_seeds=1,
        evaluate_every=1000,
        checkpoint_every=1,
        hidden_size=16,
        collection_threads=1,
        update_threads=1,
        evaluation_threads=1,
        opponent_schedule=("random",),
        balanced_schedule=True,
        milestones=(1, 2, 4),
        training_seed_base=540000000,
        evaluation_seed_offset=8000000,
        **kwargs,
    )


def assert_same_updater(a, b):
    assert a.updates == b.updates and a.optimizer_steps == b.optimizer_steps
    assert a.config == b.config
    assert torch.equal(a.generator.get_state(), b.generator.get_state())
    for name, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[name])
    for left, right in zip(
        a.optimizer.state.values(), b.optimizer.state.values(), strict=True
    ):
        for name, value in left.items():
            assert torch.equal(value, right[name])


def test_milestone_retention_curves_and_shielded_resume_match_uninterrupted(tmp_path):
    config = small_run()
    ppo = PPOConfig(epochs=1, critic_updates_trunk=False)
    left = TrainingRun(tmp_path / "left", config=config, ppo_config=ppo)
    right = TrainingRun(tmp_path / "right", config=config, ppo_config=ppo)
    try:
        for _ in range(3):
            assert left.step()["episodes"] == right.step()["episodes"]
        right.close()
        right = TrainingRun(tmp_path / "right", resume=True)
        for _ in range(2):
            assert left.step()["episodes"] == right.step()["episodes"]
        assert_same_updater(left.updater, right.updater)
        for run in (left, right):
            for u in (1, 2, 4):
                curve = json.loads(
                    (run.root / "evaluation" / f"milestone-u{u:08d}.json").read_text()
                )
                assert curve["update"] == u and curve["checkpoint_sha256"]
                assert (run.root / "checkpoints" / curve["checkpoint"]).exists()
                assert curve["evaluation"]["seeds"] == [1008000000]
            assert (run.root / "checkpoints/u00000001.pt").exists()
    finally:
        left.close()
        right.close()


def test_legacy_checkpoint_and_run_config_default_to_historical_behavior(tmp_path):
    run = TrainingRun(
        tmp_path / "old",
        config=replace(small_run(), balanced_schedule=False, milestones=()),
    )
    run.step()
    run.close()
    path = tmp_path / "old/state.json"
    state = json.loads(path.read_text())
    state["config"].pop("balanced_schedule")
    state["config"].pop("milestones")
    state["ppo_config"].pop("critic_updates_trunk")
    checkpoint = tmp_path / "old/checkpoints" / state["checkpoint"]
    payload = torch.load(checkpoint, weights_only=True)
    payload["config"].pop("critic_updates_trunk")
    torch.save(payload, checkpoint)
    from hashlib import sha256

    state["checkpoint_sha256"] = sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(json.dumps(state))
    resumed = TrainingRun(tmp_path / "old", resume=True)
    try:
        assert resumed.updater.config.critic_updates_trunk
        assert not resumed.config.balanced_schedule and resumed.config.milestones == ()
        resumed.step()
    finally:
        resumed.close()


def test_evaluation_filter_drops_unused_rows_with_identical_full_records(monkeypatch):
    policy = PPOUpdater(ActorCritic(16, seed=9)).policy()
    original = evaluation.collect_rollouts
    sizes = []
    force_all = False

    def collect(*args, **kwargs):
        assert len(kwargs["record_players"]) == 1
        if force_all:
            kwargs["record_players"] = None
        result = original(*args, **kwargs)
        sizes.append((len(result), result.payload_bytes))
        return result

    monkeypatch.setattr(evaluation, "collect_rollouts", collect)
    filtered = evaluation.evaluate_neural(policy, (810000001,), max_actions=32)
    cut = len(sizes)
    force_all = True
    all_seats = evaluation.evaluate_neural(policy, (810000001,), max_actions=32)
    assert filtered == all_seats
    assert sum(s[0] for s in sizes[:cut]) < sum(s[0] for s in sizes[cut:]) / 2
    assert sum(s[1] for s in sizes[:cut]) < sum(s[1] for s in sizes[cut:]) / 2


def test_fixed_cohort_runner_pause_resume_cache_and_tamper_guards(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.pt"
    PPOUpdater(ActorCritic(16)).save(source)
    config = ComparisonConfig(
        updates=2,
        replicates=1,
        development_seeds=1,
        test_seeds=1,
        games_per_update=1,
        max_actions=16,
        seed_offset=8100000,
        training_seed_base=550000000,
    )
    output = tmp_path / "paired"
    ordinary = CriticRun.step

    def step(self):
        result = ordinary(self)
        if self.root.name == "coupled-r0" and result["update"] == 1:
            (output / "pause.request").touch()
        return result

    monkeypatch.setattr(CriticRun, "step", step)
    result = run_comparison(output, source, config=config)
    assert result["status"] == "paused"
    assert json.loads((output / "coupled-r0/state.json").read_text())["update"] == 1
    progress = json.loads((output / "coupled-r0/progress.json").read_text())
    assert (
        progress["clock"] == "fixed_cohorts_v1" and progress["remaining_cohorts"] == 1
    )
    assert "remaining_seconds" not in progress and "deadline_utc" not in progress
    (output / "pause.request").unlink()
    monkeypatch.setattr(CriticRun, "step", ordinary)
    result = run_comparison(output, source, resume=True)
    assert result["status"] == "completed" and result["evaluations"] == 3
    assert run_comparison(output, source, resume=True) == result
    with pytest.raises(ValueError, match="configuration"):
        run_comparison(output, source, config=replace(config, updates=3), resume=True)
    full = tmp_path / "full"
    run_comparison(full, source, config=config)
    for arm in ("coupled", "head_only"):
        checkpoint = f"{arm}-r0/checkpoints/u00000002.pt"
        assert_same_updater(
            PPOUpdater.load(output / checkpoint), PPOUpdater.load(full / checkpoint)
        )
        for u in (0, 1, 2):
            assert (output / f"{arm}-r0/evaluation/milestone-u{u:08d}.json").exists()
    bad = output / next(iter(result["checkpoints"]))
    bad.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint"):
        run_comparison(output, source, resume=True)

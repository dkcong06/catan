"""Optional PPO analytic objectives, isolation, bounded updates and resumption."""
# ruff: noqa: E402

from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning.model import ActorCritic, ActionEvaluation
from learning.policy import NeuralPolicy, action_batch
from learning.ppo import PPOConfig, PPOUpdater, ppo_loss
from rl.collector import collect_rollouts, ReferencePolicy


@pytest.fixture(scope="module", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def snapshot(updater):
    return {
        "model": {k: v.clone() for k, v in updater.model.state_dict().items()},
        "optimizer": updater.optimizer.state_dict(),
        "shuffle": updater.generator.get_state().clone(),
        "counts": (updater.updates, updater.optimizer_steps),
    }


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "ratio,advantage,expected,gradient",
    (
        (1.5, 2.0, -2.4, 0.0),
        (0.5, -2.0, 1.6, 0.0),
        (0.5, 2.0, -1.0, -1.0),
        (1.5, -2.0, 3.0, 3.0),
        (1.0, 2.0, -2.0, -2.0),
        (1.0, -2.0, 2.0, 2.0),
    ),
)
def test_signed_clipping_matches_independent_formula(
    ratio, advantage, expected, gradient
):
    logs = torch.tensor([np.log(ratio)], dtype=torch.float32, requires_grad=True)
    value = torch.zeros(1, requires_grad=True)
    old = torch.zeros(1, requires_grad=True)
    advantages = torch.tensor([advantage], requires_grad=True)
    result = ppo_loss(
        ActionEvaluation(logs, value * 0, value),
        old,
        old,
        advantages,
        old,
        PPOConfig(value_coefficient=0),
    )
    assert float(result.policy.detach()) == pytest.approx(expected)
    assert float(result.approximate_kl) == pytest.approx(
        ratio - 1 - np.log(ratio), abs=1e-7
    )
    result.total.backward()
    assert logs.grad.item() == pytest.approx(gradient)
    assert old.grad is None and advantages.grad is None


@pytest.mark.parametrize(
    "value,target,clip,expected",
    (
        (0.5, 1.0, 0.2, 0.32),
        (0.5, 1.0, None, 0.125),
        (-0.5, 1.0, 0.2, 1.125),
        (0.1, 1.0, 0.2, 0.405),
    ),
)
def test_value_clipping_half_mse(value, target, clip, expected):
    vector = torch.tensor([value], requires_grad=True)
    zero = torch.zeros(1)
    result = ppo_loss(
        ActionEvaluation(zero, zero, vector),
        zero,
        zero,
        zero,
        torch.tensor([target]),
        PPOConfig(value_clip=clip),
    )
    assert result.value.item() == pytest.approx(expected)
    result.total.backward()
    assert torch.isfinite(vector.grad).all()


@pytest.mark.parametrize(
    "field,value",
    (
        ("epochs", 0),
        ("epochs", True),
        ("epochs", 1.5),
        ("minibatch_size", -1),
        ("minibatch_size", True),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("clip_ratio", 1),
        ("clip_ratio", -0.2),
        ("value_clip", 0),
        ("value_coefficient", -1),
        ("entropy_coefficient", -1),
        ("max_grad_norm", 0),
        ("target_kl", 0),
        ("gamma", 1.01),
        ("gae_lambda", float("inf")),
        ("discount", "other"),
        ("normalize_advantages", 1),
        ("adam_mode", "other"),
    ),
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        PPOConfig(**{field: value})


def test_nonfinite_objective_rejected():
    zero = torch.zeros(1)
    with pytest.raises(ValueError, match="nonfinite"):
        ppo_loss(
            ActionEvaluation(torch.tensor([1000.0]), zero, zero),
            zero,
            zero,
            torch.ones(1),
            zero,
            PPOConfig(),
        )
    with pytest.raises(ValueError, match="vectors"):
        ppo_loss(
            ActionEvaluation(zero[:0], zero[:0], zero[:0]),
            zero[:0],
            zero[:0],
            zero[:0],
            zero[:0],
            PPOConfig(),
        )


@pytest.mark.parametrize("players", (3, 4))
def test_real_update_changes_weights_preserves_data_global_rng_and_version(players):
    updater = PPOUpdater(
        ActorCritic(32, seed=12), PPOConfig(epochs=2, minibatch_size=64)
    )
    rng = torch.random.get_rng_state().clone()
    batch = collect_rollouts(
        updater.policy(),
        (4, 5),
        n_players=players,
        max_actions=500,
        batch_size=2,
        validate=True,
    )
    arrays = {
        name: getattr(batch, name).copy()
        for name in (
            "observations",
            "masks",
            "log_probs",
            "values",
            "actions",
            "rewards",
        )
    }
    before = snapshot(updater)
    assert len(batch.discards.rows)
    stats = updater.update(batch)
    assert stats.optimizer_steps > 0 and stats.rows == len(batch) and stats.update == 1
    assert not all(
        torch.equal(value, updater.model.state_dict()[key])
        for key, value in before["model"].items()
    )
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(parameter.grad is None for parameter in updater.model.parameters())
    assert all(np.isfinite(v) for v in asdict(stats).values() if isinstance(v, float))
    assert (
        stats.policy_tag == updater.policy().tag
        and stats.policy_tag != batch.policy_tag
    )
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, getattr(batch, name))
    changed = snapshot(updater)
    with pytest.raises(ValueError, match="version"):
        updater.update(batch)
    same(changed, snapshot(updater))
    follow = collect_rollouts(
        updater.policy(), (6,), n_players=players, max_actions=50, validate=True
    )
    assert np.isfinite(follow.values).all()


@pytest.mark.parametrize(
    "kind",
    (
        "reference",
        "deterministic",
        "logs",
        "values",
        "observations",
        "mutated_model",
        "links",
    ),
)
def test_preflight_rejections_are_atomic(kind):
    updater = PPOUpdater(ActorCritic(16), PPOConfig(epochs=1, minibatch_size=32))
    policy = updater.policy()
    if kind == "reference":
        policy = ReferencePolicy()
    if kind == "deterministic":
        policy = NeuralPolicy(
            updater.model, deterministic=True, policy_version="ppo1-u0"
        )
    batch = collect_rollouts(policy, (0,), max_actions=50)
    if kind in ("logs", "values", "observations"):
        name = {
            "logs": "log_probs",
            "values": "values",
            "observations": "observations",
        }[kind]
        array = getattr(batch, name).copy()
        array.flat[0] += 0.5 if kind != "observations" else float("nan")
        batch = replace(batch, **{name: array})
    elif kind == "mutated_model":
        with torch.no_grad():
            updater.model.output.bias[-1].add_(1)
    elif kind == "links":
        links = batch.next_rows.copy()
        links[0] = 0
        batch = replace(batch, next_rows=links)
    before = snapshot(updater)
    with pytest.raises(ValueError):
        updater.update(batch)
    same(before, snapshot(updater))
    assert not updater._failed


@pytest.mark.parametrize("cap,normalize", ((1, True), (65, True), (65, False)))
def test_singleton_tail_constant_advantages_and_global_normalization(cap, normalize):
    updater = PPOUpdater(
        ActorCritic(16),
        PPOConfig(
            epochs=1, minibatch_size=32, normalize_advantages=normalize, target_kl=None
        ),
    )
    batch = collect_rollouts(updater.policy(), (0,), max_actions=cap)
    adv, targets, mean, std = updater._preflight(batch)
    assert np.isfinite(adv).all()
    if normalize and len(adv) > 1 and std > 1e-8:
        assert adv.mean() == pytest.approx(0, abs=1e-7)
        assert adv.std() == pytest.approx(1, abs=1e-6)
    if cap == 1 and normalize:
        assert adv[0] == 0
    stats = updater.update(batch)
    assert stats.optimizer_steps == (cap + 31) // 32
    assert np.isfinite(targets).all() and np.isfinite(mean)


def test_tiny_kl_stops_before_later_steps_and_final_kl_is_full_rollout():
    updater = PPOUpdater(
        ActorCritic(32),
        PPOConfig(learning_rate=0.02, epochs=8, minibatch_size=50, target_kl=1e-8),
    )
    batch = collect_rollouts(updater.policy(), (2,), max_actions=200)
    stats = updater.update(batch)
    assert stats.early_stopped and 0 < stats.optimizer_steps < 32
    inputs = action_batch(batch, np.arange(len(batch)))
    with torch.no_grad():
        logs = inputs.evaluate(updater.model).log_probs - inputs.old_log_probs
    independent = (torch.expm1(logs) - logs).mean().item()
    assert stats.approximate_kl == pytest.approx(independent, abs=2e-7)


def test_gradient_failure_quarantines_without_first_step(tmp_path):
    updater = PPOUpdater(ActorCritic(16), PPOConfig(epochs=1))
    batch = collect_rollouts(updater.policy(), (1,), max_actions=40)
    before = snapshot(updater)
    parameter = next(updater.model.parameters())
    handle = parameter.register_hook(lambda grad: grad * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        updater.update(batch)
    handle.remove()
    assert updater.optimizer_steps == 0
    same(before["model"], snapshot(updater)["model"])
    for call in (
        updater.policy,
        lambda: updater.update(batch),
        lambda: updater.save(tmp_path / "bad.pt"),
    ):
        with pytest.raises(RuntimeError, match="restore"):
            call()


def test_safe_checkpoint_exact_resumption_and_atomic_replacement(tmp_path):
    config = PPOConfig(epochs=2, minibatch_size=32, target_kl=None)
    updater = PPOUpdater(ActorCritic(16, seed=10), config, shuffle_seed=99)
    path = tmp_path / "nested" / "checkpoint.pt"
    updater.save(path)
    empty = PPOUpdater.load(path)
    same(snapshot(updater), snapshot(empty))
    batch = collect_rollouts(updater.policy(), (0,), max_actions=80)
    updater.update(batch)
    updater.save(path)
    restored = PPOUpdater.load(path)
    same(snapshot(updater), snapshot(restored))
    left = collect_rollouts(updater.policy(), (1,), max_actions=100)
    right = collect_rollouts(restored.policy(), (1,), max_actions=100)
    assert left.episodes == right.episodes
    np.testing.assert_array_equal(left.log_probs, right.log_probs)
    assert updater.update(left) == restored.update(right)
    same(snapshot(updater), snapshot(restored))
    assert list(path.parent.iterdir()) == [path]
    data = torch.load(path, weights_only=True)
    data["schemas"] = (999,)
    torch.save(data, path)
    with pytest.raises(ValueError, match="schema"):
        PPOUpdater.load(path)


@pytest.mark.parametrize(
    "damage",
    (
        "nan_model",
        "nan_moment",
        "negative_moment",
        "counter",
        "learning_rate",
        "moment_shape",
    ),
)
def test_corrupt_checkpoint_rejected(tmp_path, damage):
    updater = PPOUpdater(ActorCritic(16), PPOConfig(epochs=1))
    updater.update(collect_rollouts(updater.policy(), (0,), max_actions=20))
    path = tmp_path / "checkpoint.pt"
    updater.save(path)
    data = torch.load(path, weights_only=True)
    state = next(iter(data["optimizer"]["state"].values()))
    if damage == "nan_model":
        data["model"]["output.bias"][0] = float("nan")
    elif damage == "nan_moment":
        state["exp_avg"].fill_(float("nan"))
    elif damage == "negative_moment":
        state["exp_avg_sq"].fill_(-1)
    elif damage == "counter":
        data["updates"] = -1
    elif damage == "learning_rate":
        data["optimizer"]["param_groups"][0]["lr"] *= 2
    else:
        state["exp_avg"] = torch.zeros(1)
    torch.save(data, path)
    with pytest.raises(ValueError):
        PPOUpdater.load(path)


def test_toy_actor_and_critic_move_in_expected_direction():
    # Independent two-action fixture proves optimizer direction, not Catan strength.
    model = ActorCritic(16, seed=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    obs = torch.zeros(64, 1505)
    mask = torch.zeros(64, 254, dtype=torch.bool)
    mask[:, :2] = True
    actions = torch.arange(64) % 2
    empty = torch.empty(0, dtype=torch.long)
    hands = torch.empty(0, 5, dtype=torch.long)
    old = model.evaluate_actions(obs, mask, actions, empty, hands, hands)
    advantages = torch.where(actions == 0, 1.0, -1.0)
    targets = torch.full((64,), 0.75)
    config = PPOConfig(value_clip=None)
    initial_error = (old.values.detach() - targets).square().mean()
    for _ in range(8):
        evaluated = model.evaluate_actions(obs, mask, actions, empty, hands, hands)
        loss = ppo_loss(
            evaluated, old.log_probs, old.values, advantages, targets, config
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()
    final = model.evaluate_actions(obs, mask, actions, empty, hands, hands)
    assert final.log_probs[0].exp() > old.log_probs[0].exp()
    assert (final.values.detach() - targets).square().mean() < initial_error


def test_fused_discard_reevaluation_matches_independent_sequential_head_gradients():
    from learning.model import _distribution
    from torch.nn import functional as F

    model = ActorCritic(32, seed=7)
    obs = torch.zeros(3, 1505)
    hands = torch.tensor([[4, 3, 2, 0, 0], [19, 19, 19, 19, 19], [8, 0, 0, 0, 0]])
    counts = torch.tensor([[2, 1, 1, 0, 0], [19, 19, 9, 0, 0], [4, 0, 0, 0, 0]])
    fused = model.evaluate_actions(
        obs,
        torch.zeros(3, 254, dtype=torch.bool),
        torch.full((3,), -1),
        torch.arange(3),
        hands,
        counts,
    )
    hidden = model(obs).hidden
    joint = torch.zeros(3)
    entropy = torch.zeros(3)
    for r in range(5):
        prefix = F.pad(counts[:, :r], (0, 5 - r)).float()
        remaining = counts.sum(1) - counts[:, :r].sum(1)
        lo = (remaining - hands[:, r + 1 :].sum(1)).clamp_min(0)
        hi = torch.minimum(remaining, hands[:, r])
        mask = (model.count_choices >= lo[:, None]) & (
            model.count_choices <= hi[:, None]
        )
        logs, ent = _distribution(model.discard_logits(hidden, prefix, r), mask)
        joint = joint + logs.gather(1, counts[:, r, None]).squeeze(1)
        entropy = entropy + ent
    torch.testing.assert_close(fused.log_probs, joint, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(fused.entropies, entropy, rtol=2e-6, atol=2e-6)
    parameters = (model.trunk[0].weight, model.discard.weight, model.discard.bias)
    fused_grad = torch.autograd.grad(
        (fused.log_probs + 0.01 * fused.entropies).sum(), parameters
    )
    sequential_grad = torch.autograd.grad((joint + 0.01 * entropy).sum(), parameters)
    for a, b in zip(fused_grad, sequential_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_entropy_coefficient_and_joint_discard_ratio_are_explicit():
    # A complete discard's log ratio sums all five conditional differences.
    differences = torch.tensor([0.03, -0.02, 0.05, 0.01, -0.01])
    joint = differences.sum()[None].requires_grad_()
    zero = torch.zeros(1)
    entropy = torch.tensor([2.0], requires_grad=True)
    config = PPOConfig(value_coefficient=0, entropy_coefficient=0.1)
    result = ppo_loss(
        ActionEvaluation(joint, entropy, zero), zero, zero, torch.ones(1), zero, config
    )
    assert result.policy.item() == pytest.approx(
        -float(differences.exp().prod()), rel=1e-6
    )
    assert result.total.item() == pytest.approx(result.policy.item() - 0.2)
    result.total.backward()
    assert entropy.grad.item() == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "damage",
    ("action_dtype", "mask_dtype", "discard_dtype", "discard_logs", "illegal_action"),
)
def test_malformed_storage_and_choices_reject_before_update(damage):
    from dataclasses import replace

    updater = PPOUpdater(ActorCritic(16, seed=9), PPOConfig(epochs=1))
    batch = collect_rollouts(updater.policy(), (4,), max_actions=500)
    assert len(batch.discards.rows)
    if damage == "action_dtype":
        batch = replace(batch, actions=batch.actions.astype(np.float64))
    elif damage == "mask_dtype":
        batch = replace(batch, masks=batch.masks.astype(np.int8))
    elif damage == "discard_dtype":
        batch = replace(
            batch,
            discards=replace(
                batch.discards, counts=batch.discards.counts.astype(np.float32)
            ),
        )
    elif damage == "discard_logs":
        logs = batch.discards.log_probs.copy()
        logs[0, 0] -= 0.1
        batch = replace(batch, discards=replace(batch.discards, log_probs=logs))
    else:
        actions = batch.actions.copy()
        actions[0] = 253
        assert not batch.masks[0, 253]
        batch = replace(batch, actions=actions)
    before = snapshot(updater)
    with pytest.raises(ValueError):
        updater.update(batch)
    same(before, snapshot(updater))


def test_checkpoint_failed_write_keeps_previous_file_and_cleans_temporary(
    tmp_path, monkeypatch
):
    updater = PPOUpdater(ActorCritic(16))
    path = tmp_path / "checkpoint.pt"
    updater.save(path)
    previous = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="write failure"):
        updater.save(path)
    assert path.read_bytes() == previous and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "damage",
    ("buffer", "model_dtype", "moment_dtype", "fractional_step", "future_step"),
)
def test_checkpoint_tensor_metadata_and_fixed_buffers_are_checked(tmp_path, damage):
    updater = PPOUpdater(ActorCritic(16), PPOConfig(epochs=1))
    updater.update(collect_rollouts(updater.policy(), (0,), max_actions=20))
    path = tmp_path / "checkpoint.pt"
    updater.save(path)
    data = torch.load(path, weights_only=True)
    state = next(iter(data["optimizer"]["state"].values()))
    if damage == "buffer":
        data["model"]["count_choices"][0] = 1
    elif damage == "model_dtype":
        data["model"]["output.weight"] = data["model"]["output.weight"].double()
    elif damage == "moment_dtype":
        state["exp_avg"] = state["exp_avg"].double()
    elif damage == "fractional_step":
        state["step"].fill_(0.5)
    else:
        state["step"].fill_(10000)
    torch.save(data, path)
    with pytest.raises(ValueError):
        PPOUpdater.load(path)


@pytest.mark.parametrize("mode", ("scalar", "foreach", "fused"))
def test_each_optimizer_mode_roundtrips_and_resumes(tmp_path, mode):
    config = PPOConfig(epochs=1, minibatch_size=16, adam_mode=mode)
    updater = PPOUpdater(ActorCritic(16), config)
    updater.update(collect_rollouts(updater.policy(), (0,), max_actions=30))
    path = tmp_path / "checkpoint.pt"
    updater.save(path)
    other = PPOUpdater.load(path)
    assert other.config == config
    batch = collect_rollouts(updater.policy(), (1,), max_actions=30)
    assert updater.update(batch) == other.update(batch)
    same(snapshot(updater), snapshot(other))


def test_benchmark_cli_exercises_all_optimizer_options(tmp_path):
    import json
    import sys
    from subprocess import run

    path = tmp_path / "report.json"
    result = run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_ppo",
            "--games",
            "1",
            "--max-actions",
            "25",
            "--repeats",
            "1",
            "--report",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(path.read_text())
    assert report["rows"] == 25 and len(report["sweep"]) == 8
    assert report["default"]["stats"]["optimizer_steps"] > 0
    assert all(
        delta == 0 for delta in report["foreach_parameter_max_abs_difference"].values()
    )
    rejected = run(
        [sys.executable, "-m", "scripts.benchmark_ppo", "--repeats", "0"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0 and "error:" in rejected.stderr

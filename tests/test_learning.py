"""Optional Torch model, masking, replay, ownership and backward checks."""
# ruff: noqa: E402
# The optional dependency gate must run before importing Torch model modules.

from math import log
from random import Random
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning.model import ActorCritic, _distribution
from learning.policy import NeuralPolicy, action_batch
from rl.collector import collect_rollouts
from rl.encoding import OBS_DIM
from rl.returns import compute_targets
from test_development import fund, main_state
from test_env import fixture_env
from catan.engine import resolve_seven_inplace
from catan.state import Phase


@pytest.fixture(scope="module", autouse=True)
def bounded_torch_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("width", (32, 128, 256))
def test_model_shapes_initialization_preserves_rng_and_state_dict_roundtrip(width):
    before = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    model = ActorCritic(width, seed=42)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    observations = torch.zeros(7, OBS_DIM)
    output = model(observations)
    assert output.logits.shape == (7, 254) and output.values.shape == (7,)
    assert output.hidden.shape == (7, width)
    other = ActorCritic(width, seed=42)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, other.state_dict()[name])
    with TemporaryDirectory() as directory:
        path = Path(directory) / "weights.pt"
        torch.save(model.state_dict(), path)
        loaded = ActorCritic(width, seed=13)
        loaded.load_state_dict(torch.load(path, weights_only=True))
        assert torch.equal(model(observations).logits, loaded(observations).logits)


def test_exact_masked_probabilities_entropy_and_zero_illegal_gradients():
    logits = torch.tensor([[2.0, 99.0, 2.0], [3.0, -2.0, 1.0]], requires_grad=True)
    masks = torch.tensor([[True, False, True], [False, True, False]])
    logs, entropy = _distribution(logits, masks)
    torch.testing.assert_close(logs.exp(), torch.tensor([[0.5, 0, 0.5], [0, 1, 0]]))
    torch.testing.assert_close(entropy, torch.tensor([log(2), 0]))
    loss = -logs[0, 0] - logs[1, 1] + 0.1 * entropy.sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0 and not logits.grad[1].any()
    with pytest.raises(ValueError):
        _distribution(logits, torch.zeros_like(masks))
    with pytest.raises(ValueError):
        _distribution(logits, masks.float())


@pytest.mark.parametrize("players", (3, 4))
def test_neural_collection_metadata_reevaluation_bootstraps_and_backward(players):
    model = ActorCritic(seed=9)
    policy = NeuralPolicy(model)
    weights = {name: data.clone() for name, data in model.state_dict().items()}
    global_rng = torch.random.get_rng_state().clone()
    batch = collect_rollouts(
        policy, (4, 5), n_players=players, max_actions=250, batch_size=2, validate=True
    )
    assert np.isfinite(batch.values).all() and np.isfinite(batch.log_probs).all()
    assert len(batch.discards.rows) > 0
    tensors = action_batch(batch, np.arange(len(batch)))
    evaluated = tensors.evaluate(model)
    torch.testing.assert_close(
        evaluated.log_probs.detach(), tensors.old_log_probs, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        evaluated.values.detach(), tensors.old_values, rtol=2e-5, atol=2e-6
    )
    assert (
        torch.isfinite(evaluated.entropies).all() and (evaluated.entropies >= 0).all()
    )
    bootstrap = policy.bootstrap_values(batch)
    target = compute_targets(batch, bootstrap_values=bootstrap)
    loss = (
        -evaluated.log_probs.mean()
        + (evaluated.values - torch.from_numpy(target.value_targets.copy()))
        .square()
        .mean()
        - 0.01 * evaluated.entropies.mean()
    )
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert torch.equal(global_rng, torch.random.get_rng_state())
    for name, data in model.state_dict().items():
        assert torch.equal(
            weights[name], data
        )  # Backward verifies gradients; no optimizer update.
    for source, tensor in (
        (batch.observations, tensors.observations),
        (batch.masks, tensors.masks),
    ):
        assert not np.shares_memory(source, tensor.numpy())
    tensors.observations.fill_(-99)
    assert not np.any(batch.observations == -99)


def test_seeded_repeat_and_deterministic_behavior_probabilities():
    policy = NeuralPolicy(ActorCritic(seed=3))
    first = collect_rollouts(policy, (20,), max_actions=100)
    second = collect_rollouts(policy, (20,), max_actions=100)
    assert first.episodes == second.episodes
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.log_probs, second.log_probs)
    deterministic = NeuralPolicy(policy.model, deterministic=True)
    batch = collect_rollouts(deterministic, (21,), max_actions=40)
    assert not batch.log_probs.any() and np.isfinite(batch.values).all()
    assert "deterministic" in batch.policy_tag


def test_conditional_discard_probability_normalizes_and_prefix_affects_logits():
    model = ActorCritic(seed=2)
    # A small hand permits enumeration of every complete count vector.
    hand = torch.tensor([[4, 3, 2, 0, 0]])
    combinations = [
        (a, b, 4 - a - b, 0, 0)
        for a in range(5)
        for b in range(4)
        if 0 <= 4 - a - b <= 2
    ]
    obs = torch.zeros(len(combinations), OBS_DIM)
    rows = torch.arange(len(combinations))
    evaluated = model.evaluate_actions(
        obs,
        torch.zeros(len(rows), 254, dtype=torch.bool),
        torch.full((len(rows),), -1),
        rows,
        hand.expand(len(rows), -1),
        torch.tensor(combinations),
    )
    torch.testing.assert_close(
        evaluated.log_probs.exp().sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6
    )
    hidden = model(obs[:1]).hidden
    before = model.discard_logits(hidden, torch.zeros(1, 5), 1)
    prefix = torch.tensor([[3.0, 0, 0, 0, 0]])
    after = model.discard_logits(hidden, prefix, 1)
    assert not torch.equal(before, after)


def test_maximum_discard_is_complete_and_model_roundtrip_is_consistent():
    state = main_state()
    fund(state, 0, (19,) * 5)
    state.phase, state.last_roll = Phase.ROLL, None
    resolve_seven_inplace(state)
    request = fixture_env(state).decision()
    model = ActorCritic(seed=8)
    policy = NeuralPolicy(model)
    for seed in range(10):
        sample = policy((request,), (Random(seed),))[0]
        assert sum(sample.action) == 47
        for r, count in enumerate(sample.action):
            assert request.discard_mask(sample.action[:r])[count]
        evaluation = model.evaluate_actions(
            torch.from_numpy(request.observation.copy())[None],
            torch.zeros(1, 254, dtype=torch.bool),
            torch.tensor([-1]),
            torch.tensor([0]),
            torch.tensor([request.discard_hand]),
            torch.tensor([sample.action]),
        )
        assert float(evaluation.log_probs.detach()[0]) == pytest.approx(
            sample.log_prob, abs=3e-6
        )


def test_empty_minibatches_and_repeated_discard_rows_are_remapped():
    policy = NeuralPolicy(ActorCritic(seed=4))
    batch = collect_rollouts(policy, (2,), max_actions=400)
    row = int(batch.discards.rows[0])
    tensors = action_batch(batch, [row, 0, row])
    assert tensors.discard_rows.tolist() == [0, 2]
    torch.testing.assert_close(tensors.discard_counts[0], tensors.discard_counts[1])
    assert action_batch(batch, []).evaluate(policy.model).log_probs.shape == (0,)
    assert policy((), ()) == ()
    for rows in ([-1], [len(batch)], [True], [1.5], [[0]]):
        with pytest.raises(ValueError):
            action_batch(batch, rows)


@pytest.mark.parametrize("width", (0, -1, True, 1.5))
def test_invalid_model_width(width):
    with pytest.raises(ValueError):
        ActorCritic(width)


def test_invalid_network_inputs_and_sparse_choices():
    model = ActorCritic()
    for obs in (
        torch.zeros(1505),
        torch.zeros(2, 1504),
        torch.zeros(2, 1505, dtype=torch.float64),
    ):
        with pytest.raises(ValueError):
            model(obs)
    obs = torch.zeros(1, OBS_DIM)
    masks = torch.ones(1, 254, dtype=torch.bool)
    for args in (
        (
            torch.tensor([254]),
            torch.tensor([], dtype=torch.long),
            torch.empty(0, 5, dtype=torch.long),
            torch.empty(0, 5, dtype=torch.long),
        ),
        (
            torch.tensor([-1]),
            torch.tensor([0]),
            torch.tensor([[4, 3, 2, 0, 0]]),
            torch.tensor([[0, 0, 0, 0, 0]]),
        ),
    ):
        with pytest.raises(ValueError):
            model.evaluate_actions(obs, masks, *args)


def test_hidden_opponent_hand_composition_cannot_change_model_choices():
    state = main_state()
    fund(state, 0, (3, 2, 1, 3, 2))
    fund(state, 1, (2, 0, 0, 0, 0))
    fund(state, 2, (0, 2, 0, 0, 0))
    other = state.copy()
    other.hands[[1, 2]] = other.hands[[2, 1]]
    requests = (fixture_env(state).decision(), fixture_env(other).decision())
    np.testing.assert_array_equal(requests[0].observation, requests[1].observation)
    policy = NeuralPolicy(ActorCritic(seed=7))
    a, b = Random(19), Random(19)
    left, right = policy((requests[0],), (a,))[0], policy((requests[1],), (b,))[0]
    assert left == right and a.getstate() == b.getstate()


def test_forced_and_deterministic_choices_do_not_draw_policy_rng():
    from learning.policy import _choose

    rng = Random(3)
    before = rng.getstate()
    assert _choose(
        np.array([1.0, 5.0, -1.0]), np.array([False, True, False]), rng, False
    ) == (1, 0.0)
    assert rng.getstate() == before
    assert _choose(
        np.array([1.0, 5.0, -1.0]), np.array([True, True, True]), rng, True
    ) == (1, 0.0)
    assert rng.getstate() == before


def test_no_critic_call_for_true_terminal_bootstraps():
    from types import SimpleNamespace

    policy = NeuralPolicy(ActorCritic())
    batch = SimpleNamespace(
        last_rows=np.array([[1, 2, -1, -1]]),
        episodes=(SimpleNamespace(truncated=False),),
    )

    def fail(observations):
        raise AssertionError("terminal critic call")

    policy.values = fail
    assert np.isnan(policy.bootstrap_values(batch)).all()


def test_neural_cli_creates_records_and_targets_without_optimizer(tmp_path):
    import json
    import sys
    from subprocess import run

    record_path = tmp_path / "records.npz"
    target_path = tmp_path / "targets.npz"
    result = run(
        [
            sys.executable,
            "-m",
            "scripts.collect_neural",
            "--games",
            "1",
            "--max-actions",
            "25",
            "--output",
            str(record_path),
            "--targets",
            str(target_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["optimizer_updates"] == 0 and report["trained"] is False
    assert report["actions"] == 25 and report["cutoffs"] == 1
    with (
        np.load(record_path, allow_pickle=False) as records,
        np.load(target_path, allow_pickle=False) as targets,
    ):
        assert np.isfinite(records["values"]).all()
        assert np.isfinite(targets["advantages"]).all() and targets[
            "advantages"
        ].shape == (25,)
        assert json.loads(str(targets["metadata"]))["target_schema"] == 1
    rejected = run(
        [sys.executable, "-m", "scripts.collect_neural", "--gamma", "nan"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0 and "error:" in rejected.stderr

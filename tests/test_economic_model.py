"""Versioned feature migration, complete decisions and exact optimizer resumption."""
# ruff: noqa: E402

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning.model import ActorCritic, EconomicActorCritic
from learning.evaluation import EvaluationPolicy
from learning.policy import NeuralPolicy, action_batch
from learning.ppo import PPOConfig, PPOUpdater
from rl.collector import collect_rollouts, save_rollouts
from rl.economics import EconomicEnvFactory
from rl.encoding import OBS_DIM


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("width", (16, 128))
def test_zero_migration_preserves_real_decisions_discard_joint_logs_and_rng(
    players, width
):
    base = ActorCritic(width, seed=31)
    rng = torch.get_rng_state().clone()
    new = EconomicActorCritic.from_base(base)
    assert torch.equal(rng, torch.get_rng_state())
    args = dict(n_players=players, policy_seed=17, max_actions=400, validate=True)
    old_batch = collect_rollouts(NeuralPolicy(base), (670000020,), **args)
    batch = collect_rollouts(
        NeuralPolicy(new), (670000020,), env_factory=EconomicEnvFactory(), **args
    )
    assert old_batch.episodes == batch.episodes
    for name in ("actions", "log_probs", "values"):
        np.testing.assert_array_equal(getattr(batch, name), getattr(old_batch, name))
    np.testing.assert_array_equal(batch.discards.counts, old_batch.discards.counts)
    np.testing.assert_array_equal(
        batch.discards.log_probs, old_batch.discards.log_probs
    )
    tensors = action_batch(batch, np.arange(len(batch)))
    old_tensors = tensors._replace(
        observations=tensors.observations[:, :OBS_DIM].contiguous()
    )
    with torch.no_grad():
        left = old_tensors.evaluate(base)
        right = tensors.evaluate(new)
        for a, b in zip(left, right, strict=True):
            assert torch.equal(a, b)
    assert sum(p.numel() for p in new.parameters()) == (
        sum(p.numel() for p in base.parameters()) + 495 * width
    )


def test_profile_rejection_and_checkpoint_exact_next_update(tmp_path):
    updater = PPOUpdater(
        EconomicActorCritic(16),
        PPOConfig(epochs=1, minibatch_size=64, gae_lambda=1),
        shuffle_seed=23,
    )
    batch = collect_rollouts(
        updater.policy(),
        (670000021,),
        max_actions=200,
        env_factory=EconomicEnvFactory(),
    )
    rng = updater.generator.get_state().clone()
    for bad in (
        replace(batch, observation_schema=3),
        replace(batch, reward_scheme="terminal_win_net_vp_0.1_v1"),
    ):
        with pytest.raises(ValueError, match="profile"):
            updater.update(bad)
        assert updater.updates == 0 and not updater.optimizer.state
        assert torch.equal(rng, updater.generator.get_state())
    updater.update(batch)
    assert updater.model.economics.weight.detach().count_nonzero() > 0
    updater.save(tmp_path / "economic.pt")
    restored = PPOUpdater.load(tmp_path / "economic.pt")
    assert restored.model.schema == (3, 4, 1)
    same(updater.model.state_dict(), restored.model.state_dict())
    same(updater.optimizer.state_dict(), restored.optimizer.state_dict())
    next_batch = collect_rollouts(
        updater.policy(),
        (670000022,),
        max_actions=150,
        env_factory=EconomicEnvFactory(),
    )
    a, b = updater.update(next_batch), restored.update(next_batch)
    assert a == b
    same(updater.model.state_dict(), restored.model.state_dict())
    same(updater.optimizer.state_dict(), restored.optimizer.state_dict())
    assert torch.equal(updater.generator.get_state(), restored.generator.get_state())
    save_rollouts(batch, tmp_path / "economic.npz")
    with np.load(tmp_path / "economic.npz", allow_pickle=False) as archive:
        import json

        assert json.loads(str(archive["metadata"]))["collector_schema"] == 3
    with pytest.raises(ValueError, match="original"):
        EconomicActorCritic.from_base(restored.model)


@pytest.mark.parametrize("players", (3, 4))
def test_learner_feature_scope_preserves_nonzero_policy_against_reference(players):
    model = EconomicActorCritic(16, seed=4)
    with torch.no_grad():
        model.economics.weight.fill_(
            0.01
        )  # Exercise a policy that actually uses the suffix.
    for seat in range(players):
        policy = EvaluationPolicy(NeuralPolicy(model), "greedy", seat)
        args = dict(
            n_players=players,
            max_actions=250,
            policy_seed=17,
            record_players=(seat,),
            validate=True,
        )
        full = collect_rollouts(
            policy, (670000023,), env_factory=EconomicEnvFactory(), **args
        )
        scoped = collect_rollouts(
            policy,
            (670000023,),
            env_factory=EconomicEnvFactory(feature_player=seat),
            **args,
        )
        assert full.episodes == scoped.episodes
        for name in (
            "observations",
            "actions",
            "log_probs",
            "values",
            "final_observations",
        ):
            np.testing.assert_array_equal(getattr(full, name), getattr(scoped, name))
        np.testing.assert_array_equal(full.discards.counts, scoped.discards.counts)

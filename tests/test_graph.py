"""Independent graph equations, relabeling, complete decisions and exact resume."""
# ruff: noqa: E402

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from catan.board import GEOMETRY
from learning.graph import GraphActorCritic
from learning.imitation import Demonstrations, ImitationConfig, ImitationTrainer
from learning.model import EconomicActorCritic
from learning.policy import action_batch
from learning.ppo import PPOConfig, PPOUpdater
from rl.collector import ReferencePolicy, collect_rollouts
from rl.economics import EconomicEnvFactory


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_constructor_ignores_caller_default_device_and_dtype():
    device, dtype = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float64)
        model = GraphActorCritic(16, seed=8)
        assert all(
            p.device.type == "cpu" and p.dtype == torch.float32
            for p in model.parameters()
        )
        assert all(b.device.type == "cpu" for b in model.buffers())
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)


def oracle(model, observations):
    n = len(observations)
    tiles = model.tile_input(observations[:, :171].reshape(n, 19, 9)).relu()
    vertices = model.vertex_input(observations[:, 171:1035].reshape(n, 54, 16)).relu()
    edges = model.edge_input(observations[:, 1035:1395].reshape(n, 72, 5)).relu()

    def means(values, neighbors):
        return torch.stack(
            [values[:, list(ids)].sum(1) / len(ids) for ids in neighbors], dim=1
        )

    for layer in model.rounds:
        old_t, old_v, old_e = tiles, vertices, edges
        tiles = F.linear(
            torch.cat((old_t, means(old_v, GEOMETRY.tile_vertices)), 2),
            layer.tiles.weight,
            layer.tiles.bias,
        ).relu()
        vertices = F.linear(
            torch.cat(
                (
                    old_v,
                    means(old_t, GEOMETRY.vertex_tiles),
                    means(old_e, GEOMETRY.vertex_edges),
                ),
                2,
            ),
            layer.vertices.weight,
            layer.vertices.bias,
        ).relu()
        edges = F.linear(
            torch.cat((old_e, means(old_v, GEOMETRY.edge_vertices)), 2),
            layer.edges.weight,
            layer.edges.bias,
        ).relu()
    return tiles, vertices, edges


def test_graph_and_gradients_match_independent_neighbor_reductions():
    model = GraphActorCritic(16, seed=9)
    x = torch.rand((3, 2000), generator=torch.Generator().manual_seed(1))
    expected = oracle(model, x)
    actual = model.board_features(x)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=3e-7, rtol=2e-6)
    parameters = tuple(model.parameters())
    a = torch.autograd.grad(
        sum(t.square().sum() for t in actual), parameters, allow_unused=True
    )
    b = torch.autograd.grad(
        sum(t.square().sum() for t in expected), parameters, allow_unused=True
    )
    for left, right in zip(a, b, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=1e-5, rtol=2e-5)


def test_board_encoder_is_equivariant_under_joint_node_and_relation_relabeling():
    model = GraphActorCritic(16, seed=8)
    other = deepcopy(model)
    rng = torch.Generator().manual_seed(15)
    t, v, e = (torch.randperm(n, generator=rng) for n in (19, 54, 72))
    x = torch.rand((2, 2000), generator=rng)
    y = torch.cat(
        (
            x[:, :171].reshape(2, 19, 9)[:, t].flatten(1),
            x[:, 171:1035].reshape(2, 54, 16)[:, v].flatten(1),
            x[:, 1035:1395].reshape(2, 72, 5)[:, e].flatten(1),
            x[:, 1395:],
        ),
        1,
    )
    other.tv, other.vt = model.tv[t][:, v], model.vt[v][:, t]
    other.ve, other.ev = model.ve[v][:, e], model.ev[e][:, v]
    for a, b, perm in zip(model.board_features(x), other.board_features(y), (t, v, e)):
        torch.testing.assert_close(a[:, perm], b, atol=3e-7, rtol=3e-6)
    # The retained flat output head is intentionally not claimed equivariant.


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


def test_scoped_rng_capacity_schema_and_fixed_graph_buffers(tmp_path):
    rng = torch.get_rng_state().clone()
    graph = GraphActorCritic()
    assert torch.equal(rng, torch.get_rng_state())
    assert (
        abs(
            sum(p.numel() for p in graph.parameters())
            / sum(p.numel() for p in EconomicActorCritic().parameters())
            - 1
        )
        < 0.01
    )
    for name in ("tv", "vt", "ve", "ev"):
        torch.testing.assert_close(
            getattr(graph, name).sum(1), torch.ones(getattr(graph, name).shape[0])
        )
    updater = PPOUpdater(graph)
    updater.save(tmp_path / "graph.pt")
    restored = PPOUpdater.load(tmp_path / "graph.pt")
    assert restored.model.schema == (4, 4, 1)
    same(graph.state_dict(), restored.model.state_dict())
    payload = torch.load(tmp_path / "graph.pt", weights_only=True)
    payload["model"]["tv"][0, 0] += 0.01
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="fixed"):
        PPOUpdater.load(tmp_path / "bad.pt")
    for bad in (torch.zeros((1, 1505)), torch.zeros((1, 2000), dtype=torch.float64)):
        with pytest.raises(ValueError):
            graph(bad)


@pytest.mark.parametrize("players", (3, 4))
def test_graph_complete_decisions_ppo_and_exact_next_update_resume(players, tmp_path):
    updater = PPOUpdater(
        GraphActorCritic(16, seed=5),
        PPOConfig(epochs=1, minibatch_size=64, gae_lambda=1),
        shuffle_seed=3,
    )
    batch = collect_rollouts(
        updater.policy(),
        (680000001,),
        n_players=players,
        max_actions=400,
        policy_seed=17,
        env_factory=EconomicEnvFactory(),
        validate=True,
    )
    inputs = action_batch(batch, np.arange(len(batch)))
    with torch.no_grad():
        predictions = inputs.evaluate(updater.model)
    np.testing.assert_allclose(
        predictions.log_probs, batch.log_probs, atol=4e-6, rtol=0
    )
    before = updater.generator.get_state().clone()
    with pytest.raises(ValueError, match="profile"):
        updater.update(replace(batch, observation_schema=3))
    assert not updater.optimizer.state and torch.equal(
        before, updater.generator.get_state()
    )
    updater.update(batch)
    updater.save(tmp_path / "updated.pt")
    restored = PPOUpdater.load(tmp_path / "updated.pt")
    next_batch = collect_rollouts(
        updater.policy(),
        (680000002,),
        max_actions=100,
        n_players=players,
        env_factory=EconomicEnvFactory(),
    )
    assert updater.update(next_batch) == restored.update(next_batch)
    same(updater.model.state_dict(), restored.model.state_dict())
    same(updater.optimizer.state_dict(), restored.optimizer.state_dict())
    assert torch.equal(updater.generator.get_state(), restored.generator.get_state())


def test_common_economic_teacher_roundtrip_and_both_models_consume_same_labels(
    tmp_path,
):
    batch = collect_rollouts(
        ReferencePolicy("greedy"),
        (680000003,),
        max_actions=120,
        env_factory=EconomicEnvFactory(),
    )
    teacher = Demonstrations.from_rollouts(batch)
    teacher.save(tmp_path / "common.npz")
    common = Demonstrations.load(tmp_path / "common.npz")
    np.testing.assert_array_equal(common.observations, teacher.observations)
    assert common.observation_schema == 4
    for model in (EconomicActorCritic(16), GraphActorCritic(16)):
        trainer = ImitationTrainer(model, ImitationConfig(epochs=1), shuffle_seed=1)
        stats = trainer.fit(common)
        assert stats["optimizer_steps"] > 0 and np.isfinite(trainer.loss(common))

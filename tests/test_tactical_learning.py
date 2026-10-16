"""Set-valued tactical supervision, immutable data and exact replay continuation."""
# ruff: noqa: E402

from random import Random
from dataclasses import replace
from types import MethodType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning.model import ActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from learning.tactics import (
    LessonTrainer,
    ReplayAnchor,
    TacticalLessons,
    lesson_logs,
    score_lessons,
)
from rl.collector import collect_rollouts
from scripts.compare_tactical import ComparisonConfig, generate_lessons
from scripts.game_viewer import GameSessions, PolicyRegistry
from test_ppo import same, snapshot


@pytest.fixture(scope="module", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def lessons(tmp_path_factory):
    path = tmp_path_factory.mktemp("lessons") / "data"
    return generate_lessons(path, seed=3_550_000_001, games=4)


def test_set_likelihood_matches_scalar_oracle_and_credits_every_win(lessons):
    model = ActorCritic(16, seed=31)
    rows = np.flatnonzero(lessons.tactical)
    assert len(rows) > 0
    data = lessons.demonstrations
    actual = lesson_logs(model, lessons, rows)
    logits = model(torch.from_numpy(data.observations[rows])).logits.detach().numpy()
    for i, row in enumerate(rows):
        legal = logits[i, data.masks[row]].astype(np.float64)
        wins = logits[i, lessons.wins[row]].astype(np.float64)
        maximum = float(legal.max())
        expected = np.log(np.exp(wins - maximum).sum() / np.exp(legal - maximum).sum())
        assert float(actual[i].detach()) == pytest.approx(expected, abs=2e-6)
    all_wins = lessons.wins.copy()
    all_wins[rows] = data.masks[rows]
    broadened = TacticalLessons(data, all_wins)
    logs = lesson_logs(model, broadened, rows)
    assert torch.allclose(logs, torch.zeros_like(logs), atol=1e-6)
    logs.sum().backward()
    assert all(
        p.grad is None or torch.max(torch.abs(p.grad)) < 1e-5
        for p in model.parameters()
    )


def test_bucket_weights_and_numeric_roundtrip_and_rejections(lessons, tmp_path):
    weights = lessons.weights()
    assert weights.sum() == pytest.approx(len(lessons))
    assert weights[lessons.tactical].sum() == pytest.approx(len(lessons) / 2)
    lessons.save(tmp_path / "copy")
    restored = TacticalLessons.load(tmp_path / "copy")
    np.testing.assert_array_equal(lessons.wins, restored.wins)
    assert not restored.wins.flags.writeable
    with pytest.raises(FileExistsError):
        lessons.save(tmp_path / "copy")
    for bad in (
        np.zeros((len(lessons), 253), dtype=np.bool_),
        lessons.wins.astype(np.int8),
        np.ones_like(lessons.wins),
    ):
        with pytest.raises(ValueError):
            TacticalLessons(lessons.demonstrations, bad).validate()


def test_actor_warmup_learns_without_value_head_labels(lessons):
    model = ActorCritic(32, seed=21)
    trainer = LessonTrainer(model, learning_rate=1e-3)
    weight, bias = (
        model.output.weight[-1].detach().clone(),
        model.output.bias[-1].detach().clone(),
    )
    before = score_lessons(model, lessons)
    for _ in range(4):
        trainer.fit(lessons)
    after = score_lessons(model, lessons)
    assert after["weighted_loss"] < before["weighted_loss"]
    assert torch.equal(weight, model.output.weight[-1])
    assert torch.equal(bias, model.output.bias[-1])
    assert trainer.steps > 0 and all(p.grad is None for p in model.parameters())


def test_default_ppo_path_is_exact_and_bad_anchor_rejects_before_mutation(lessons):
    left = PPOUpdater(ActorCritic(16, seed=33), PPOConfig(epochs=1), shuffle_seed=7)
    right = PPOUpdater(ActorCritic(16, seed=33), PPOConfig(epochs=1), shuffle_seed=7)
    batch = collect_rollouts(left.policy(), (65,), max_actions=40)
    before = snapshot(left)
    with pytest.raises(ValueError):
        left.update(batch, anchor=object())
    same(before, snapshot(left))
    left.update(batch)
    right.update(batch, anchor=None)
    same(snapshot(left), snapshot(right))


def test_anchor_sampling_and_ppo_checkpoint_resume_are_exact(lessons, tmp_path):
    updater = PPOUpdater(
        ActorCritic(16, seed=29), PPOConfig(epochs=1, target_kl=None), shuffle_seed=8
    )
    anchor = ReplayAnchor(lessons, seed=99)
    batch = collect_rollouts(updater.policy(), (33,), max_actions=48)
    updater.update(batch, anchor=anchor)
    updater.save(tmp_path / "ppo.pt")
    torch.save(anchor.state_dict(), tmp_path / "anchor.pt")
    restored = PPOUpdater.load(tmp_path / "ppo.pt")
    other = ReplayAnchor(lessons, seed=0)
    other.load_state_dict(torch.load(tmp_path / "anchor.pt", weights_only=True))
    batch = collect_rollouts(updater.policy(), (34,), max_actions=48)
    updater.update(batch, anchor=anchor)
    restored.update(batch, anchor=other)
    same(snapshot(updater), snapshot(restored))
    same(anchor.state_dict(), other.state_dict())


def test_named_neural_viewer_model_is_readonly_and_uses_actual_stream(tmp_path):
    updater = PPOUpdater(ActorCritic(16, seed=9))
    path = tmp_path / "bot.pt"
    updater.save(path)
    games = GameSessions(PolicyRegistry({"neural": path}))
    status, created = games.handle(
        "/api/game/new",
        dict(seed=57, players=3, bots=["neural", "tactical", "greedy"], max_actions=20),
    )
    assert status == 200
    model, _ = games.registry.loaded["neural"]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    env = games.sessions[created["session"]]["env"]
    decision = env.decision()
    stream = Random("catan-evaluation-v1:17:57:0")
    expected = updater.policy()((decision,), (stream,))[0]
    status, response = games.handle(
        "/api/game/step", dict(session=created["session"], revision=0)
    )
    assert status == 200 and response["frame"]["event"]["action"] == expected.action
    assert len(response["frame"]["event"]["top"]) == 5
    assert games.sessions[created["session"]]["rngs"][0].getstate() == stream.getstate()
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in before.items()
    )


def test_comparison_allowances_and_split_configuration_reject_invalid_values():
    for config in (
        dict(cohorts=0),
        dict(replicates=4),
        dict(seed=4_200_000_000),
        dict(anchor_coefficient=float("nan")),
        dict(max_actions=True),
    ):
        with pytest.raises(ValueError):
            ComparisonConfig(**config)


def test_value_preflight_uses_same_tolerance_for_scalar_rounding_and_rejects_stale():
    model = ActorCritic(16, seed=36)
    with torch.no_grad():
        model.output.weight[-1].zero_()
        model.output.bias[-1].zero_()
    original = model.forward

    def rounded(self, observations):
        output = original(observations)
        return output._replace(
            values=output.values + (4e-6 if len(observations) >= 4 else 0)
        )

    model.forward = MethodType(rounded, model)
    updater = PPOUpdater(model, PPOConfig(epochs=1))
    batch = collect_rollouts(updater.policy(), (83,), max_actions=40, batch_size=1)
    before = snapshot(updater)
    updater._preflight(batch)
    same(before, snapshot(updater))
    stale = replace(batch, values=batch.values + np.float32(0.01))
    with pytest.raises(ValueError, match="behavior values"):
        updater._preflight(stale)
    same(before, snapshot(updater))

"""Fairness, profile isolation and transactional comparison integration."""
# ruff: noqa: E402

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning.architecture import (
    ArchitectureConfig,
    ArchitectureRun,
    OriginalPolicy,
    teachers,
    warmup,
)
from learning.hypotheses import digest, read
from learning.model import ActorCritic, EconomicActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from learning.training import balanced_assignment
from rl.collector import collect_rollouts
from rl.economics import EconomicEnvFactory
from scripts.compare_architecture import run_comparison


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sources(tmp_path):
    original = ActorCritic(16, seed=8)
    config = PPOConfig(epochs=1, gae_lambda=1)
    a, b = tmp_path / "original.pt", tmp_path / "economic.pt"
    PPOUpdater(original, config).save(a)
    PPOUpdater(EconomicActorCritic.from_base(original), config).save(b)
    return a, b, config


def small():
    return ArchitectureConfig(
        feature_updates=2,
        graph_updates=1,
        replicates=1,
        development_seeds=1,
        test_seeds=1,
        teacher_games=2,
        validation_games=2,
        warmup_epochs=1,
        games_per_update=1,
        max_actions=16,
        target_seconds=60,
        seed_shift=40000000,
    )


def same_updater(a, b):
    assert a.updates == b.updates and a.optimizer_steps == b.optimizer_steps
    assert torch.equal(a.generator.get_state(), b.generator.get_state())
    assert all(
        torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items()
    )
    for astate, bstate in zip(
        a.optimizer.state.values(), b.optimizer.state.values(), strict=True
    ):
        assert all(torch.equal(v, bstate[k]) for k, v in astate.items())


def test_original_prefix_adapter_preserves_complete_collection():
    base = PPOUpdater(ActorCritic(16, seed=8)).policy()
    adapted = OriginalPolicy(base.model, policy_version="prefix")
    kwargs = dict(
        n_players=4, max_actions=600, batch_size=2, policy_seed=17, validate=True
    )
    left = collect_rollouts(base, [680200001, 680200002], **kwargs)
    right = collect_rollouts(
        adapted, [680200001, 680200002], env_factory=EconomicEnvFactory(True), **kwargs
    )
    assert left.episodes == right.episodes
    assert np.array_equal(left.actions, right.actions)
    assert np.array_equal(left.observations, right.observations[:, :1505])
    assert np.array_equal(left.discards.counts, right.discards.counts)
    assert np.array_equal(left.log_probs, right.log_probs)


def test_rotating_seat_scopes_and_exact_run_resume(tmp_path):
    original, economic, ppo = sources(tmp_path)
    config = small().run_config("features", 0, PPOUpdater.load(economic).model)
    config = replace(config, update_threads=1)
    left = ArchitectureRun(
        tmp_path / "left",
        enabled=True,
        original_checkpoint=original,
        config=config,
        ppo_config=ppo,
        initial_checkpoint=economic,
    )
    right = ArchitectureRun(
        tmp_path / "right",
        enabled=True,
        original_checkpoint=original,
        config=config,
        ppo_config=ppo,
        initial_checkpoint=economic,
    )
    try:
        for i in range(28):
            players, seat, name = balanced_assignment(i, config.opponent_schedule)
            _, recorded, opponent, _ = left._collection_policy(i, players)
            assert recorded == (seat,) and opponent == name
            assert left.env_factory == EconomicEnvFactory(True, feature_player=seat)
        assert left.step()["episodes"] == right.step()["episodes"]
        right.close()
        with pytest.raises(ValueError, match="profile"):
            ArchitectureRun(
                tmp_path / "right",
                enabled=False,
                original_checkpoint=original,
                config=config,
                ppo_config=ppo,
                resume=True,
            )
        right = ArchitectureRun(
            tmp_path / "right",
            enabled=True,
            original_checkpoint=original,
            config=config,
            ppo_config=ppo,
            resume=True,
        )
        assert left.step()["episodes"] == right.step()["episodes"]
        same_updater(left.updater, right.updater)
    finally:
        left.close()
        right.close()


def test_teacher_and_warmup_exact_shard_resume_and_identity(tmp_path):
    _, economic, _ = sources(tmp_path)
    manifest = teachers(tmp_path / "teacher", small())
    assert teachers(tmp_path / "teacher", small()) == manifest
    assert {s["identity"]["role"] for s in manifest["shards"]} == {
        "training",
        "validation",
    }
    left, right = tmp_path / "left.pt", tmp_path / "right.pt"
    warmup(economic, left, tmp_path / "teacher", manifest, epochs=2, shuffle_seed=11)
    calls = 0

    def pause():
        nonlocal calls
        calls += 1
        return calls == 2

    assert (
        warmup(
            economic,
            right,
            tmp_path / "teacher",
            manifest,
            epochs=2,
            shuffle_seed=11,
            should_pause=pause,
        )
        is None
    )
    warmup(economic, right, tmp_path / "teacher", manifest, epochs=2, shuffle_seed=11)
    same_updater(PPOUpdater.load(left), PPOUpdater.load(right))
    assert (
        read(left.with_suffix(".json"))["optimizer_steps"]
        == read(right.with_suffix(".json"))["optimizer_steps"]
    )
    with pytest.raises(ValueError, match="recipe"):
        warmup(
            economic, right, tmp_path / "teacher", manifest, epochs=3, shuffle_seed=11
        )
    receipt = tmp_path / "teacher" / manifest["shards"][0]["file"]
    receipt.write_bytes(receipt.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        teachers(tmp_path / "teacher", small())


def test_tiny_two_stage_comparison_cached_resume_and_tamper(tmp_path):
    original, economic, _ = sources(tmp_path)
    output = tmp_path / "comparison"
    result = run_comparison(output, original, economic, config=small())
    assert result["status"] == "completed"
    assert read(output / "state.json")["trained_cases"] == 4
    assert read(output / "state.json")["evaluated_models"] == 6
    for stage in ("features", "graph"):
        reports = [read(p) for p in (output / stage / "evaluation").glob("*-r0.json")]
        assert len(reports) == 3
        assert all(len(r["records"]) == 14 for r in reports)
        assert all("candidate_vp" in row for r in reports for row in r["records"])
    assert run_comparison(output, original, economic, resume=True) == result
    with pytest.raises(ValueError, match="configuration"):
        run_comparison(
            output,
            original,
            economic,
            config=replace(small(), graph_updates=2),
            resume=True,
        )
    checkpoint = output / next(iter(result["checkpoints"]))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        run_comparison(output, original, economic, resume=True)
    assert digest(original) == result["protocol"]["original_sha256"]


def test_outer_pause_resumes_only_committed_cohorts(tmp_path, monkeypatch):
    original, economic, _ = sources(tmp_path)
    output = tmp_path / "comparison"
    progress = ArchitectureRun.progress

    def paused(run, record=None, *, status="running"):
        result = progress(run, record, status=status)
        if result["committed_update"] == 1 and status == "running":
            (output / "pause.request").touch()
        return result

    monkeypatch.setattr(ArchitectureRun, "progress", paused)
    result = run_comparison(output, original, economic, config=small())
    assert result["status"] == "paused"
    state = read(output / "features/zero-r0/state.json")
    assert state["update"] == 1
    # The flag remains effective even with a resume invocation.
    assert run_comparison(output, original, economic, resume=True)["status"] == "paused"
    monkeypatch.setattr(ArchitectureRun, "progress", progress)
    (output / "pause.request").unlink()
    assert (
        run_comparison(output, original, economic, resume=True)["status"] == "completed"
    )
    rows = (output / "features/zero-r0/metrics.jsonl").read_text().splitlines()
    assert len(rows) == small().feature_updates


@pytest.mark.parametrize(
    "changes",
    (
        {"feature_updates": 0},
        {"graph_updates": True},
        {"teacher_games": 3},
        {"replicates": 4},
    ),
)
def test_invalid_comparison_bounds(changes):
    with pytest.raises(ValueError):
        replace(small(), **changes)

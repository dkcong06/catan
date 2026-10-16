"""Optional training experiment seed, evaluation, persistence and budget gates."""

# ruff: noqa: E402
from dataclasses import replace
import json
from time import perf_counter

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning.evaluation import EvaluationPolicy, compare_evaluations, evaluate_neural
from learning.model import ActorCritic
from learning.policy import NeuralPolicy
from learning.ppo import PPOConfig, PPOUpdater
from learning.training import RunConfig, TrainingRun, TUNE_SEED_BASE
from rl.evaluate import play_match
from rl.policies import POLICIES


@pytest.fixture(scope="module", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def small_config(**kwargs):
    return RunConfig(
        budget_seconds=300,
        max_updates=2,
        games_per_update=2,
        active_games=2,
        max_actions=65,
        evaluation_max_actions=25,
        evaluation_batch_size=2,
        tune_seeds=1,
        test_seeds=2,
        evaluate_every=1,
        checkpoint_every=1,
        hidden_size=16,
        collection_threads=1,
        update_threads=1,
        evaluation_threads=1,
        **kwargs,
    )


def same_models(a, b):
    assert a.updater.updates == b.updater.updates
    assert torch.equal(a.updater.generator.get_state(), b.updater.generator.get_state())
    for name, value in a.updater.model.state_dict().items():
        assert torch.equal(value, b.updater.model.state_dict()[name])
    for left, right in zip(
        a.updater.optimizer.state.values(),
        b.updater.optimizer.state.values(),
        strict=True,
    ):
        for name in left:
            assert torch.equal(left[name], right[name])


@pytest.mark.parametrize(
    "field,value",
    (
        ("budget_seconds", 0),
        ("budget_seconds", float("nan")),
        ("budget_seconds", float("inf")),
        ("budget_seconds", True),
        ("games_per_update", 0),
        ("games_per_update", True),
        ("max_updates", 0),
        ("training_seed_base", TUNE_SEED_BASE),
        ("max_actions", 1000000),
        ("evaluation_max_actions", 1000000),
        ("test_seeds", 10001),
        ("update_threads", 0),
    ),
)
def test_invalid_run_bounds(field, value):
    with pytest.raises(ValueError):
        RunConfig(**{field: value})


def test_real_batched_evaluation_matches_independent_serial_engine_and_rotations():
    policy = NeuralPolicy(ActorCritic(16, seed=5), policy_version="test")
    weights = {k: v.clone() for k, v in policy.model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    seeds = (4, 5)
    batched = evaluate_neural(
        policy, seeds, max_actions=120, batch_size=2, validate=True
    )
    serial = evaluate_neural(
        policy, seeds, max_actions=120, batch_size=1, validate=True
    )
    assert (
        batched["records"] == serial["records"]
        and batched["fingerprint"] == serial["fingerprint"]
    )
    assert len(batched["records"]) == 28

    def neural(request, stream):
        return policy((request,), (stream,))[0].action

    for row in batched["records"]:
        policies = [POLICIES[row["opponent"]]] * row["players"]
        policies[row["candidate_seat"]] = neural
        match = play_match(
            policies, row["seed"], policy_seed=17, max_actions=120, validate=True
        )
        assert match.fingerprint == row["fingerprint"]
        assert (match.actions, match.winner, match.terminated, match.truncated) == (
            row["actions"],
            row["winner"],
            row["terminated"],
            row["truncated"],
        )
    for k, v in weights.items():
        assert torch.equal(v, policy.model.state_dict()[k])
    assert torch.equal(rng, torch.random.get_rng_state())


def test_reference_evaluation_rows_are_never_accepted_for_ppo():
    from rl.collector import collect_rollouts

    updater = PPOUpdater(ActorCritic(16))
    mixed = EvaluationPolicy(updater.policy(), "random", 0)
    batch = collect_rollouts(mixed, (0,), max_actions=50)
    assert np.isnan(batch.values[batch.players != 0]).all()
    with pytest.raises(ValueError, match="version"):
        updater.update(batch)
    with pytest.raises(ValueError):
        EvaluationPolicy(updater.policy(), "unknown", 0)


def test_seed_cluster_interval_pairing_json_roundtrip_and_caps():
    policy = NeuralPolicy(ActorCritic(16))
    baseline = evaluate_neural(policy, (2, 3), max_actions=25, batch_size=2)
    restored = json.loads(json.dumps(baseline))
    same = compare_evaluations(restored, baseline, draws=100)
    assert same["seed_clusters"] == 2 and same["paired_games"] == 28
    assert same["win_rate_all_change"] == 0
    assert same["seed_cluster_bootstrap_95_percent_interval"] == [0, 0]
    assert same["baseline"]["cutoffs"] == 28 and same["baseline"]["losses"] == 0
    changed = json.loads(json.dumps(baseline))
    for record in changed["records"]:
        record["winner"] = record["candidate_seat"]
    interval = compare_evaluations(baseline, changed, draws=100)
    assert interval["win_rate_all_change"] == 1 and interval[
        "seed_cluster_bootstrap_95_percent_interval"
    ] == [1, 1]
    changed["records"].append(changed["records"][0])
    with pytest.raises(ValueError, match="uniquely"):
        compare_evaluations(baseline, changed)
    mismatch = {**baseline, "policy_seed": 18}
    with pytest.raises(ValueError):
        compare_evaluations(baseline, mismatch)


def test_two_real_updates_seed_schedule_final_reports_and_bounded_checkpoints(tmp_path):
    root = tmp_path / "run"
    run = TrainingRun(
        root, config=small_config(), ppo_config=PPOConfig(epochs=1, gae_lambda=1)
    )
    try:
        rng = torch.random.get_rng_state().clone()
        report = run.run()
        assert report["status"] == "completed" and report["update"] == 2
        assert report["totals"]["episodes"] == 4 and report["totals"]["actions"] == 260
        assert report["totals"]["cutoffs"] == 4 and report["totals"]["wins"] == 0
        assert (
            report["selected_update"] == 0
            and report["latest_checkpoint"] == "u00000002.pt"
        )
        assert (root / "evaluation/test-selected.json").exists()
        metrics = [
            json.loads(row) for row in (root / "metrics.jsonl").read_text().splitlines()
        ]
        assert [r["players"] for r in metrics] == [3, 4]
        seeds = [e["seed"] for r in metrics for e in r["episodes"]]
        assert seeds == list(range(1000000, 1000004))
        assert all(seed < TUNE_SEED_BASE for seed in seeds)
        assert len(list((root / "checkpoints").glob("*.pt"))) <= 4
        assert torch.equal(rng, torch.random.get_rng_state())
        assert run.run() == report
    finally:
        run.close()
    resumed = TrainingRun(root, resume=True)
    try:
        assert resumed.run() == report
    finally:
        resumed.close()


def test_committed_resume_replays_same_update_and_trims_orphan_metrics(tmp_path):
    config = small_config()
    a = TrainingRun(tmp_path / "a", config=config, ppo_config=PPOConfig(epochs=1))
    b = TrainingRun(tmp_path / "b", config=config, ppo_config=PPOConfig(epochs=1))
    try:
        a.initialize_baselines()
        b.initialize_baselines()
        a.step()
        b.step()
        b.close()
        with (tmp_path / "b/metrics.jsonl").open("a") as out:
            out.write(json.dumps({"update": 999}) + "\n" + '{"partial"')
        b = TrainingRun(tmp_path / "b", resume=True)
        same_models(a, b)
        left, right = a.step(), b.step()
        assert left["episodes"] == right["episodes"]
        same_models(a, b)
        assert a.totals == b.totals
        assert len((tmp_path / "b/metrics.jsonl").read_text().splitlines()) == 2
        assert a.deadline != b.deadline  # Each new experiment has its own deadline.
    finally:
        a.close()
        b.close()


def test_writer_lock_and_existing_directory_are_protected(tmp_path):
    run = TrainingRun(tmp_path / "run", config=small_config())
    try:
        with pytest.raises(RuntimeError, match="another process"):
            TrainingRun(tmp_path / "run", resume=True)
        with pytest.raises(FileExistsError):
            TrainingRun(tmp_path / "run", config=small_config())
    finally:
        run.close()


def test_tampered_manifest_and_checkpoint_digest_reject_resume(tmp_path):
    run = TrainingRun(tmp_path / "run", config=small_config())
    run.close()
    path = tmp_path / "run/state.json"
    original = json.loads(path.read_text())
    changed = {**original, "schema": 999}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="schema"):
        TrainingRun(tmp_path / "run", resume=True)
    changed = {**original, "checkpoint_sha256": "wrong"}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest"):
        TrainingRun(tmp_path / "run", resume=True)


def test_pause_preserves_committed_deadline_and_latest_model(tmp_path):
    root = tmp_path / "run"
    run = TrainingRun(root, config=small_config())
    try:
        run.initialize_baselines()
        run.step()
        run.stopping = True
        paused = run.run()
        assert (
            paused["status"] == "paused" and not (root / "final-report.json").exists()
        )
        deadline = run.deadline
    finally:
        run.close()
    restored = TrainingRun(root, resume=True)
    try:
        assert restored.deadline == deadline and restored.updater.updates == 1
        assert restored.config == small_config()
    finally:
        restored.close()


def test_deadline_stops_new_cycles_but_retains_baseline_and_final_evaluation(tmp_path):
    run = TrainingRun(tmp_path / "run", config=small_config())
    try:
        run.initialize_baselines()
        run._stop_at = perf_counter() - 1
        report = run.run()
        assert report["update"] == 0 and report["status"] == "completed"
        assert report["selected_comparison"]["win_rate_all_change"] == 0
    finally:
        run.close()


def test_cli_runs_bounded_update_and_rejects_invalid_or_resume_overrides(tmp_path):
    import sys
    from subprocess import run

    directory = tmp_path / "cli"
    result = run(
        [
            sys.executable,
            "-m",
            "scripts.train_ppo",
            "--output",
            str(directory),
            "--minutes",
            "2",
            "--max-updates",
            "1",
            "--games",
            "1",
            "--max-actions",
            "25",
            "--evaluation-max-actions",
            "25",
            "--tune-seeds",
            "1",
            "--test-seeds",
            "1",
            "--evaluate-every",
            "1",
            "--checkpoint-every",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((directory / "final-report.json").read_text())
    assert report["update"] == 1 and report["totals"]["episodes"] == 1
    resumed = run(
        [
            sys.executable,
            "-m",
            "scripts.train_ppo",
            "--output",
            str(directory),
            "--resume",
        ],
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    for extra in (["--minutes", "0"], ["--resume", "--minutes", "1"]):
        bad = run(
            [
                sys.executable,
                "-m",
                "scripts.train_ppo",
                "--output",
                str(directory),
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        assert bad.returncode != 0 and "error:" in bad.stderr


def test_checkpoint_retention_and_restore_config_validation(tmp_path):
    config = replace(small_config(), max_updates=5)
    run = TrainingRun(tmp_path / "run", config=config, ppo_config=PPOConfig(epochs=1))
    try:
        run.initialize_baselines()
        for _ in range(5):
            run.step()
        names = {p.name for p in (tmp_path / "run/checkpoints").glob("*.pt")}
        assert names == {"u00000000.pt", "u00000004.pt", "u00000005.pt"}
    finally:
        run.close()
    path = tmp_path / "run/state.json"
    data = json.loads(path.read_text())
    data["ppo_config"]["learning_rate"] = 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="configuration"):
        TrainingRun(tmp_path / "run", resume=True)


def test_baseline_written_before_interruption_recovers_selection_rank(tmp_path):
    root = tmp_path / "run"
    run = TrainingRun(root, config=small_config())
    try:
        run.initialize_baselines()
        state = json.loads((root / "state.json").read_text())
        state["best_tuning_wins"] = -1
        (root / "state.json").write_text(json.dumps(state))
    finally:
        run.close()
    resumed = TrainingRun(root, resume=True)
    try:
        resumed.initialize_baselines()
        assert resumed.best_wins == 0
        assert json.loads((root / "state.json").read_text())["best_tuning_wins"] == 0
    finally:
        resumed.close()


@pytest.mark.parametrize("offset", (-1, 100000001))
def test_invalid_evaluation_offset(offset):
    with pytest.raises(ValueError):
        RunConfig(evaluation_seed_offset=offset)


def test_reserved_evaluation_offset_avoids_pilot_boards_and_survives_resume(tmp_path):
    root = tmp_path / "run"
    run = TrainingRun(root, config=small_config(evaluation_seed_offset=1000))
    try:
        run.initialize_baselines()
        tune = json.loads((root / "evaluation/tuning-initial.json").read_text())
        test = json.loads((root / "evaluation/test-initial.json").read_text())
        assert tune["seeds"] == [TUNE_SEED_BASE + 1000]
        assert test["seeds"] == [2000001000, 2000001001]
    finally:
        run.close()
    resumed = TrainingRun(root, resume=True)
    try:
        assert resumed.config.evaluation_seed_offset == 1000
    finally:
        resumed.close()

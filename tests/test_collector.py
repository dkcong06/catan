from dataclasses import fields
from itertools import product
import json
from math import comb, exp, log
from random import Random

import numpy as np
import pytest

from catan.cards import DevelopmentCard as Card
from catan.engine import resolve_seven_inplace
from catan.state import Phase
from rl.collector import (
    COLLECTOR_SCHEMA_VERSION,
    PolicySample,
    ReferencePolicy,
    RolloutBatch,
    _discard_logs,
    collect_rollouts,
    save_rollouts,
)
from rl.encoding import FEATURE_SLICES, OBS_DIM
from rl.env import CatanEnv
from rl.evaluate import play_match
from rl.policies import POLICIES
from test_development import build, fund, grant, main_state, own_roads, path
from test_env import fixture_env


def arrays(batch):
    return {
        field.name: getattr(batch, field.name)
        for field in fields(RolloutBatch)
        if isinstance(getattr(batch, field.name), np.ndarray)
    }


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("name", ("greedy", "random"))
def test_recorded_seats_preserve_full_trajectories_links_and_private_boundaries(
    players, name
):
    policy = ReferencePolicy(name)
    full = collect_rollouts(policy, (27, 28), n_players=players, max_actions=250)
    selected = collect_rollouts(
        policy, (27, 28), n_players=players, max_actions=250, record_players=(1,)
    )
    assert full.episodes == selected.episodes
    chosen = np.flatnonzero(full.players == 1)
    remap = {int(row): local for local, row in enumerate(chosen)}
    for field in (
        "observations",
        "masks",
        "actions",
        "log_probs",
        "values",
        "rewards",
        "elapsed_actions",
        "episode_indices",
        "phases",
        "steps",
        "terminated",
        "truncated",
    ):
        np.testing.assert_array_equal(
            getattr(full, field)[chosen], getattr(selected, field)
        )
    np.testing.assert_array_equal(
        selected.next_rows, [remap.get(int(full.next_rows[row]), -1) for row in chosen]
    )
    assert not selected.final_observations[:, [0, 2, 3]].any()
    np.testing.assert_array_equal(
        selected.final_observations[:, 1], full.final_observations[:, 1]
    )
    assert (selected.last_rows[:, [0, 2, 3]] == -1).all()
    assert selected.payload_bytes < full.payload_bytes


@pytest.mark.parametrize("seats", ((), (0, 0), (-1,), (4,), (True,)))
def test_invalid_recorded_seats_reject_before_any_policy_call(seats):
    def never(*args):
        raise AssertionError("policy called for invalid recording configuration")

    with pytest.raises(ValueError):
        collect_rollouts(never, record_players=seats)


def assert_same_records(left, right):
    assert left.episodes == right.episodes
    for name in arrays(left):
        if name in ("final_observations",):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    for episode in range(len(left.episodes)):
        a = np.flatnonzero(left.episode_indices == episode)
        b = np.flatnonzero(right.episode_indices == episode)
        assert len(a) == len(b)
        for name in arrays(left):
            if name in ("final_observations", "last_rows", "next_rows"):
                continue
            np.testing.assert_array_equal(
                getattr(left, name)[a], getattr(right, name)[b]
            )
        # Global links can differ with scheduling, but point to identical steps.
        for x, y in zip(a, b, strict=True):
            lx, ly = left.next_rows[x], right.next_rows[y]
            assert (lx < 0) == (ly < 0)
            if lx >= 0:
                assert left.steps[lx] == right.steps[ly]
        da = np.flatnonzero(np.isin(left.discards.rows, a))
        db = np.flatnonzero(np.isin(right.discards.rows, b))
        for name in ("hands", "counts", "log_probs"):
            np.testing.assert_array_equal(
                getattr(left.discards, name)[da], getattr(right.discards, name)[db]
            )


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("name", ("random", "greedy"))
def test_complete_batches_match_independent_evaluation_and_same_seat_oracle(
    players, name
):
    batch = collect_rollouts(
        ReferencePolicy(name), (0, 1, 2), n_players=players, batch_size=2, validate=True
    )
    all_phases = set()
    for index, episode in enumerate(batch.episodes):
        records = []

        def shadow(request, rng):
            choice = POLICIES[name](request, rng)
            records.append(
                (
                    request.player,
                    request.phase,
                    request.observation.copy(),
                    None if request.mask is None else request.mask.copy(),
                    request.discard_hand,
                    choice,
                )
            )
            return choice

        direct = play_match([shadow] * players, episode.seed, validate=True)
        assert (
            episode.actions,
            episode.winner,
            episode.fingerprint,
            episode.terminated,
            episode.truncated,
        ) == (
            direct.actions,
            direct.winner,
            direct.fingerprint,
            direct.terminated,
            direct.truncated,
        )
        rows = np.flatnonzero(batch.episode_indices == index)
        assert len(rows) == len(records) == episode.actions
        sparse = {int(row): i for i, row in enumerate(batch.discards.rows)}
        for step, (row, record) in enumerate(zip(rows, records, strict=True)):
            player, phase, obs, mask, hand, action = record
            all_phases.add(phase)
            assert batch.players[row] == player and batch.phases[row] == phase
            assert batch.steps[row] == step
            np.testing.assert_array_equal(batch.observations[row], obs)
            if hand is None:
                assert batch.actions[row] == action
                np.testing.assert_array_equal(batch.masks[row], mask)
                assert batch.log_probs[row] == pytest.approx(
                    0 if name == "greedy" else -log(np.count_nonzero(mask))
                )
            else:
                assert batch.actions[row] == -1 and not batch.masks[row].any()
                d = sparse[row]
                np.testing.assert_array_equal(batch.discards.hands[d], hand)
                np.testing.assert_array_equal(batch.discards.counts[d], action)
                expected = (
                    0
                    if name == "greedy"
                    else log(
                        np.prod(
                            [comb(h, c) for h, c in zip(hand, action)], dtype=object
                        )
                        / comb(sum(hand), sum(action))
                    )
                )
                assert batch.log_probs[row] == pytest.approx(expected, abs=1e-6)
            successors = [r for r in rows[step + 1 :] if batch.players[r] == player]
            if successors:
                next_row = successors[0]
                assert batch.next_rows[row] == next_row
                assert batch.elapsed_actions[row] == batch.steps[next_row] - step
                assert not batch.terminated[row] and not batch.truncated[row]
                assert batch.rewards[row] == 0
            else:
                assert batch.next_rows[row] == -1
                assert batch.last_rows[index, player] == row
                assert batch.elapsed_actions[row] == episode.actions - step
                assert batch.terminated[row] == episode.terminated
                assert batch.truncated[row] == episode.truncated
                assert batch.rewards[row] == float(player == episode.winner)
        assert batch.rewards[rows].sum() == int(episode.terminated)
        assert np.isnan(batch.values[rows]).all()
        assert (batch.last_rows[index, players:] == -1).all()
        assert not batch.final_observations[index, players:].any()
    assert {
        Phase.SETUP_SETTLEMENT,
        Phase.SETUP_ROAD,
        Phase.ROLL,
        Phase.MAIN,
    } <= all_phases
    masks = batch.discards.masks()
    for i, counts in enumerate(batch.discards.counts):
        for r, count in enumerate(counts):
            assert masks[i, r, count]
    assert masks.shape == (len(batch.discards.rows), 5, 20)


@pytest.mark.parametrize("players", (3, 4))
def test_scheduling_refill_is_reproducible_across_batch_sizes(players):
    policy = ReferencePolicy(("greedy",) + ("random",) * (players - 1))
    serial = collect_rollouts(
        policy, range(6), n_players=players, batch_size=1, policy_seed=-3
    )
    for size in (2, 4, 10):
        batched = collect_rollouts(
            policy, range(6), n_players=players, batch_size=size, policy_seed=-3
        )
        assert_same_records(serial, batched)


@pytest.mark.parametrize("cap", (1, 2, 8, 19))
@pytest.mark.parametrize("players", (3, 4))
def test_cutoffs_close_only_acted_seats_without_fake_rewards(players, cap):
    batch = collect_rollouts(
        ReferencePolicy(), (4, 5), n_players=players, max_actions=cap
    )
    assert len(batch) == cap * 2 and not batch.rewards.any()
    for e, episode in enumerate(batch.episodes):
        assert episode.truncated and not episode.terminated and episode.winner is None
        rows = np.flatnonzero(batch.episode_indices == e)
        assert batch.truncated[rows].sum() == len(set(batch.players[rows]))
        for seat in range(players):
            env = CatanEnv(max_actions=cap)
            request = env.reset(episode.seed, n_players=players)
            streams = [
                Random(f"catan-evaluation-v1:0:{episode.seed}:{p}")
                for p in range(players)
            ]
            while request is not None:
                env.step(
                    POLICIES["random"](request, streams[request.player]),
                    player=request.player,
                )
                request = env.decision()
            np.testing.assert_array_equal(
                batch.final_observations[e, seat], env.observe(seat)
            )
    closing = np.flatnonzero(batch.next_rows == -1)
    successors = batch.successor_observations(closing)
    np.testing.assert_array_equal(
        successors,
        batch.final_observations[
            batch.episode_indices[closing], batch.players[closing]
        ],
    )


def inject(monkeypatch, state, cap):
    class InjectedEnv(CatanEnv):
        def reset(self, seed=None, *, n_players=4):
            replacement = fixture_env(state, cap)
            self._state, self._encoder, self._zeros = (
                replacement._state,
                replacement._encoder,
                replacement._zeros,
            )
            self._actions = 0
            return self.decision()

    monkeypatch.setattr("rl.collector.CatanEnv", InjectedEnv)


def test_manual_discard_actor_roller_hidden_info_and_value_metadata(monkeypatch):
    state = main_state()
    state.phase, state.last_roll = Phase.ROLL, None
    fund(state, 2, (4, 3, 2, 0, 0))
    fund(state, 3, (3, 3, 2, 0, 0))
    grant(state, Card.VICTORY_POINT, player=1)
    resolve_seven_inplace(state)
    inject(monkeypatch, state, 3)
    reference = ReferencePolicy()

    def with_value(decisions, streams):
        return tuple(
            sample._replace(value=0.25) for sample in reference(decisions, streams)
        )

    batch = collect_rollouts(
        with_value, (7,), max_actions=3, policy_tag="test-v1", validate=True
    )
    assert batch.players.tolist() == [2, 3, 0]
    assert batch.actions[:2].tolist() == [-1, -1]
    assert batch.discards.rows.tolist() == [0, 1]
    assert batch.values.tolist() == [0.25] * 3
    assert batch.elapsed_actions.tolist() == [3, 2, 1]
    assert batch.policy_tag == "test-v1"
    for row, seat in enumerate((2, 3, 0)):
        np.testing.assert_array_equal(
            batch.observations[row][FEATURE_SLICES["turn_owner"]],
            np.eye(4)[(-seat) % 4],
        )
        assert (
            batch.final_observations[0, seat][FEATURE_SLICES["players"]].reshape(4, 9)[
                (1 - seat) % 4, 1
            ]
            == 0
        )


def test_actual_last_action_win_and_off_turn_winner_are_not_submitting_actor(
    monkeypatch,
):
    state = main_state()
    vertices = []
    for v in range(54):
        if not set(state.board.geometry.vertex_neighbors[v]) & set(vertices):
            vertices.append(v)
        if len(vertices) == 4:
            break
    for v in vertices:
        build(state, v, 1)
    grant(state, Card.VICTORY_POINT, 4, player=1)
    own_roads(state, path(state, 5), 1)
    inject(monkeypatch, state, 1)
    batch = collect_rollouts(
        lambda decisions, streams: (PolicySample(219, 0),),
        (2,),
        max_actions=1,
        validate=True,
    )
    assert batch.episodes[0].winner == 1
    assert batch.episodes[0].terminated and not batch.episodes[0].truncated
    assert batch.players.tolist() == [0] and batch.rewards.tolist() == [0]
    assert batch.last_rows[0, 1] == -1  # No invented action for an unacted seat.
    assert batch.terminated.tolist() == [True] and not batch.truncated.any()


def test_random_discard_joint_probability_normalizes_and_adapter_preserves_rng():
    state = main_state()
    fund(state, 0, (4, 3, 2, 0, 0))
    state.phase, state.last_roll = Phase.ROLL, None
    resolve_seven_inplace(state)
    request = fixture_env(state).decision()
    probabilities = []
    for counts in product(range(5), range(4), range(3), (0,), (0,)):
        if sum(counts) == 4:
            logs = _discard_logs(request.discard_hand, counts)
            exact = (
                comb(4, counts[0])
                * comb(3, counts[1])
                * comb(2, counts[2])
                / comb(9, 4)
            )
            assert exp(sum(logs)) == pytest.approx(exact)
            probabilities.append(exact)
    assert sum(probabilities) == pytest.approx(1)
    for name in ("random", "greedy"):
        for seed in range(40):
            a, b = Random(seed), Random(seed)
            expected = POLICIES[name](request, a)
            sample = ReferencePolicy(name)((request,), (b,))[0]
            assert sample.action == expected and a.getstate() == b.getstate()
            assert sample.log_prob == pytest.approx(sum(sample.discard_log_probs))
            assert len(sample.discard_log_probs) == 5


def test_maximum_hand_sparse_discard_storage(monkeypatch):
    state = main_state()
    fund(state, 0, (19,) * 5)
    state.phase, state.last_roll = Phase.ROLL, None
    resolve_seven_inplace(state)
    inject(monkeypatch, state, 1)
    batch = collect_rollouts(ReferencePolicy(), (1,), max_actions=1, validate=True)
    assert batch.discards.hands.tolist() == [[19] * 5]
    assert batch.discards.counts.sum() == 47
    assert np.isfinite(batch.discards.log_probs).all()
    assert batch.log_probs[0] == pytest.approx(batch.discards.log_probs[0].sum())
    assert batch.discards.masks().shape == (1, 5, 20)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"seeds": (True,)},
        {"n_players": 2},
        {"policy_seed": True},
        {"max_actions": 0},
        {"max_actions": True},
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"validate": 1},
        {"policy_tag": 7},
        {"max_actions": 2**31},
        {"n_players": True},
    ),
)
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        collect_rollouts(ReferencePolicy(), **kwargs)


def test_invalid_policy_and_seat_configuration():
    for value in (None, 7):
        with pytest.raises(ValueError):
            collect_rollouts(value)
    for names in ((), ("random", "random"), "unknown"):
        with pytest.raises(ValueError):
            ReferencePolicy(names)
    with pytest.raises(ValueError, match="match"):
        collect_rollouts(ReferencePolicy(("random",) * 3))


@pytest.mark.parametrize(
    "sample",
    (
        72,
        PolicySample(True, 0),
        PolicySample(0, 0),
        PolicySample(72, 0.1),
        PolicySample(72, float("nan")),
        PolicySample(72, 0, float("inf")),
        PolicySample(72, 0, True),
        PolicySample(72, False),
        PolicySample(72, 0, 1e50),
        PolicySample(72, 0, None, (0,) * 5),
    ),
)
def test_all_samples_validated_before_any_game_steps(monkeypatch, sample):
    calls = []
    original = CatanEnv.step

    def spy(self, action, *, player):
        calls.append(action)
        return original(self, action, player=player)

    monkeypatch.setattr(CatanEnv, "step", spy)

    def policy(decisions, streams):
        first = int(np.flatnonzero(decisions[0].mask)[0])
        return PolicySample(first, 0), sample

    with pytest.raises((ValueError, TypeError)) as caught:
        collect_rollouts(policy, (1, 2), max_actions=1)
    assert calls == []
    assert any(
        "seed=2" in note and "action=1" in note for note in caught.value.__notes__
    )


@pytest.mark.parametrize(
    "sample",
    (
        PolicySample((4, 4, 0, 0, 0), 0, None, (0,) * 5),
        PolicySample((0,) * 5, 0, None, (0,) * 5),
        PolicySample((2, 1, 1, 0, 0), 0),
        PolicySample((2, 1, 1, 0, 0), -1, None, (0,) * 5),
        PolicySample((2, 1, 1, 0, 0), 0, None, (0,) * 4),
        PolicySample((2, 1, 1, 0, 0), 0, None, (0, 0, float("nan"), 0, 0)),
        PolicySample((2, 1, 1, 0, False), 0, None, (0,) * 5),
    ),
)
def test_invalid_discard_metadata(monkeypatch, sample):
    state = main_state()
    fund(state, 0, (4, 3, 2, 0, 0))
    state.phase, state.last_roll = Phase.ROLL, None
    resolve_seven_inplace(state)
    inject(monkeypatch, state, 1)
    with pytest.raises((ValueError, TypeError)):
        collect_rollouts(lambda requests, streams: (sample,), (2,), max_actions=1)


def test_callback_count_failure_and_accidental_mutation_have_batch_context():
    for policy in (lambda ds, rs: (), lambda ds, rs: ds[0].observation.fill(1)):
        with pytest.raises((ValueError, TypeError)) as caught:
            collect_rollouts(policy, (9, 10), max_actions=1)
        assert any("collector policy batch" in note for note in caught.value.__notes__)


def test_owned_outputs_stay_stable_are_contiguous_and_have_exact_payload():
    saved = []
    reference = ReferencePolicy()

    def retain(decisions, streams):
        saved.extend(decisions)
        return reference(decisions, streams)

    batch = collect_rollouts(retain, (0, 1), max_actions=100)
    copies = {name: value.copy() for name, value in arrays(batch).items()}
    for value in arrays(batch).values():
        assert not value.flags.writeable and value.flags.c_contiguous
    for request in saved:
        assert not np.shares_memory(batch.observations, request.observation)
    another = collect_rollouts(ReferencePolicy(), (0, 1), max_actions=100)
    for name, value in arrays(batch).items():
        np.testing.assert_array_equal(value, copies[name])
        assert not np.shares_memory(value, getattr(another, name))
    payload = sum(value.nbytes for value in arrays(batch).values()) + sum(
        getattr(batch.discards, name).nbytes
        for name in ("rows", "hands", "counts", "log_probs")
    )
    assert batch.payload_bytes == payload
    selected = np.array([0, 1, 30, 99, 199], dtype=np.int64)
    successors = batch.successor_observations(selected)
    for i, row in enumerate(selected):
        link = batch.next_rows[row]
        expected = (
            batch.observations[link]
            if link >= 0
            else batch.final_observations[
                batch.episode_indices[row], batch.players[row]
            ]
        )
        np.testing.assert_array_equal(successors[i], expected)
    assert successors.flags.c_contiguous and successors.flags.writeable
    successors.fill(-100)
    np.testing.assert_array_equal(batch.observations, copies["observations"])
    for rows in ((-1,), (len(batch),), (True,), (0.1,), [[0]], "0"):
        with pytest.raises(ValueError):
            batch.successor_observations(rows)


@pytest.mark.parametrize("compressed", (False, True))
def test_numeric_archive_roundtrip_without_pickle(tmp_path, compressed):
    batch = collect_rollouts(ReferencePolicy(), (3, 4), max_actions=30)
    path = tmp_path / "rollouts.npz"
    save_rollouts(batch, path, compressed=compressed)
    with np.load(path, allow_pickle=False) as saved:
        for name, value in arrays(batch).items():
            np.testing.assert_array_equal(saved[name], value)
        for name in ("rows", "hands", "counts", "log_probs"):
            np.testing.assert_array_equal(
                saved["discard_" + name], getattr(batch.discards, name)
            )
        metadata = json.loads(str(saved["metadata"]))
        assert metadata["collector_schema"] == COLLECTOR_SCHEMA_VERSION
        assert metadata["policy_tag"] == batch.policy_tag
        assert metadata["episodes"][0]["fingerprint"] == batch.episodes[0].fingerprint
    with pytest.raises(ValueError):
        save_rollouts(batch, tmp_path / "wrong.zip")
    with pytest.raises(ValueError):
        save_rollouts(batch, path, compressed=1)


def test_cli_collects_numeric_records_and_rejects_bad_bounds(tmp_path):
    import sys
    from subprocess import run

    destination = tmp_path / "rollout.npz"
    result = run(
        [
            sys.executable,
            "-m",
            "scripts.collect_rollouts",
            "--games",
            "2",
            "--players",
            "3",
            "--max-actions",
            "12",
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["games"] == report["cutoffs"] == 2 and report["actions"] == 24
    assert report["wins"] == 0
    with np.load(destination, allow_pickle=False) as data:
        assert data["observations"].shape == (24, OBS_DIM)
    for options in (
        ("--games", "0"),
        ("--batch-size", "0"),
        ("--policies", "random,greedy"),
        ("--output", "wrong.zip"),
    ):
        rejected = run(
            [sys.executable, "-m", "scripts.collect_rollouts", *options],
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0 and "error:" in rejected.stderr


@pytest.mark.parametrize("post_action", (False, True))
def test_debug_failures_preserve_actual_replay_coordinates(monkeypatch, post_action):
    calls = 0

    def fail(state):
        nonlocal calls
        calls += 1
        if calls == (2 if post_action else 1):
            raise AssertionError("check failed")

    monkeypatch.setattr("rl.collector.assert_invariants", fail)
    with pytest.raises(AssertionError, match="check failed") as caught:
        collect_rollouts(ReferencePolicy(), (17,), validate=True)
    assert any(
        "seed=17" in note and f"action={int(post_action)}" in note
        for note in caught.value.__notes__
    )


def test_empty_minibatch_and_empty_discard_shapes():
    batch = collect_rollouts(ReferencePolicy(), max_actions=1)
    assert batch.successor_observations([]).shape == (0, OBS_DIM)
    assert batch.discards.masks().shape == (0, 5, 20)
    assert batch.discards.counts.shape == (0, 5)

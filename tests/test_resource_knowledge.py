"""Resource-history privacy, engine replay and same-seat progress accounting."""

from types import SimpleNamespace

import numpy as np
import pytest

from catan.actions import RollAction
from rl.collector import ReferencePolicy, collect_rollouts
from rl.encoding import FEATURE_SLICES, OBS_DIM
from rl.env import CatanEnv
from rl.hypotheses import (
    EXPERIMENT_OBS_DIM,
    HypothesisEnvFactory,
    PROGRESS_REWARD,
    ResourceBounds,
    ResourceEvent,
    progress,
    public_flows,
)


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("knowledge", (False, True))
def test_public_tracker_replays_full_greedy_games_and_every_invariant(
    players, knowledge
):
    original = collect_rollouts(
        ReferencePolicy("greedy"),
        (420000001, 420000002),
        n_players=players,
        policy_seed=17,
        validate=True,
    )
    extended = collect_rollouts(
        ReferencePolicy("greedy"),
        (420000001, 420000002),
        n_players=players,
        policy_seed=17,
        validate=True,
        env_factory=HypothesisEnvFactory(knowledge=knowledge),
    )
    assert original.episodes == extended.episodes
    assert extended.observation_schema == 3
    assert extended.observations.shape[1] == EXPERIMENT_OBS_DIM
    assert np.array_equal(original.observations, extended.observations[:, :OBS_DIM])
    assert np.array_equal(original.actions, extended.actions)
    if not knowledge:
        assert not extended.observations[:, OBS_DIM:].any()


def initial_event():
    flows = ((0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (2, 0, 0, 0, 1))
    return ResourceEvent(flows, (0, 1, 3), (16, 19, 19, 19, 18), (0,) * 5)


def test_third_party_hidden_steal_keeps_both_possible_resource_outcomes():
    tracker = ResourceBounds(3, 0)
    tracker.update(initial_event())
    tracker.update(
        ResourceEvent(
            ((0,) * 5,) * 3, (0, 2, 2), (16, 19, 19, 19, 18), (0,) * 5, steal=(1, 2)
        )
    )
    result = tracker.features().reshape(3, 5, 2)[:2] * 19
    # Both wood and ore theft produce exactly this observer's public history.
    for actual in (
        np.array(((2, 0, 0, 0, 0), (1, 0, 0, 0, 1))),
        np.array(((1, 0, 0, 0, 1), (2, 0, 0, 0, 0))),
    ):
        assert ((result[:, :, 0] <= actual) & (actual <= result[:, :, 1])).all()
    before = tracker.bounds.copy()
    assert np.array_equal(tracker.features(), tracker.features())
    assert np.array_equal(tracker.bounds, before)


def test_participant_can_use_own_private_steal_type():
    tracker = ResourceBounds(3, 1)
    event = initial_event()
    tracker.update(
        ResourceEvent(event.flows, event.totals, event.bank, (1, 0, 0, 0, 0))
    )
    tracker.update(
        ResourceEvent(
            ((0,) * 5,) * 3,
            (0, 2, 2),
            event.bank,
            (1, 0, 0, 0, 1),
            steal=(1, 2),
            known_resource=4,
        )
    )
    bounds = tracker.features().reshape(3, 5, 2)
    # Relative first opponent is player2; their exact remaining hand is deducible.
    assert np.allclose(bounds[0, :, 0] * 19, (2, 0, 0, 0, 0))
    assert np.array_equal(bounds[0, :, 0], bounds[0, :, 1])


@pytest.mark.parametrize("same_owner,expected", ((False, 0), (True, 1)))
def test_production_scarcity_uses_public_inputs_only(same_owner, expected):
    board = SimpleNamespace(
        numbers=(6, 6),
        resources=(0, 0),
        geometry=SimpleNamespace(tile_vertices=((0,), (1,))),
    )
    state = SimpleNamespace(
        n_players=3,
        board=board,
        robber_tile=-1,
        last_roll=6,
        vertex_owner=np.array((0, 0 if same_owner else 1)),
        vertex_building=np.array((2, 1)),
    )
    flows = public_flows(
        state, RollAction(), 0, (1, 19, 19, 19, 19), (0, 0, 0), (0, 0, 0)
    )
    assert flows[0][0] == expected and sum(map(sum, flows)) == expected


class RewardProbe(CatanEnv):
    reward_scheme = PROGRESS_REWARD

    def step(self, action, *, player):
        result = super().step(action, player=player)
        return result._replace(
            rewards=(0.01, 0, 0) if 12 < self.actions <= 20 else (0, 0, 0)
        )


def test_rewards_on_opponent_actions_accumulate_on_correct_previous_learner_row():
    batch = collect_rollouts(
        ReferencePolicy("greedy"),
        (19,),
        n_players=3,
        max_actions=23,
        record_players=(0,),
        env_factory=RewardProbe,
    )
    expected = np.zeros(len(batch), dtype=np.float32)
    for step in range(13, 21):
        row = np.flatnonzero(batch.steps <= step - 1)[-1]
        expected[row] += 0.01
    assert np.allclose(batch.rewards, expected, atol=1e-8)
    assert batch.rewards.sum() == pytest.approx(0.08)
    assert np.all(batch.players == 0)


@pytest.mark.parametrize("players", (3, 4))
@pytest.mark.parametrize("cap", (65, 5000))
def test_vp_reward_telescope_matches_actual_final_own_score(players, cap):
    batch = collect_rollouts(
        ReferencePolicy("greedy"),
        (420000004,),
        n_players=players,
        max_actions=cap,
        env_factory=HypothesisEnvFactory(point_progress=True),
        validate=True,
    )
    final = batch.final_observations[0, :players, :OBS_DIM]
    start = FEATURE_SLICES["players"].start
    scores = np.rint(final[:, start + 1] * 10).astype(int)
    win = [float(batch.episodes[0].winner == p) for p in range(players)]
    expected = np.asarray(win) + 0.1 * np.asarray(progress(scores))
    for p in range(players):
        assert batch.rewards[batch.players == p].sum() == pytest.approx(
            expected[p], abs=3e-7
        )
    assert batch.reward_scheme == PROGRESS_REWARD


def test_progress_excludes_setup_caps_ten_and_cancels_award_cycles():
    assert progress((0, 1, 2, 3, 9, 10, 12)) == (0, 0, 0, 0.125, 0.875, 1, 1)
    values = [progress((vp,))[0] for vp in (3, 5, 3)]
    assert sum(0.1 * (b - a) for a, b in zip(values, values[1:])) == 0


def test_knowledge_validation_and_scheduling_do_not_change_observations_or_rng():
    factory = HypothesisEnvFactory(knowledge=True)
    a = collect_rollouts(
        ReferencePolicy("greedy"),
        (420000008, 420000009),
        max_actions=200,
        batch_size=1,
        env_factory=factory,
        validate=False,
    )
    b = collect_rollouts(
        ReferencePolicy("greedy"),
        (420000008, 420000009),
        max_actions=200,
        batch_size=8,
        env_factory=factory,
        validate=True,
    )
    assert a.episodes == b.episodes
    for episode in range(2):
        left, right = a.episode_indices == episode, b.episode_indices == episode
        assert np.array_equal(a.observations[left], b.observations[right])
        assert np.array_equal(a.actions[left], b.actions[right])

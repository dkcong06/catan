"""Seeded observation-only evaluation with terminal rewards and seat rotations.

Game/RNG summaries are reporting data, never policy inputs. Elapsed time includes
reset, policy/encoding/masks, steps, optional checks and final fingerprints.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import platform
from random import Random
from time import perf_counter

import numpy as np

from catan.state import Phase, _integer, _player_count
from catan.validation import _fingerprint, assert_invariants
from rl.actions import ACTION_SCHEMA_VERSION
from rl.encoding import SCHEMA_VERSION
from rl.env import CatanEnv, ENV_SCHEMA_VERSION, REWARD_SCHEME
from rl.policies import POLICIES, POLICY_VERSIONS

EVALUATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MatchResult:
    seed: int
    policy_seed: int
    players: int
    actions: int
    completed_turns: int
    winner: int | None
    terminated: bool
    truncated: bool
    rewards: tuple[float, ...]
    scores: tuple[int, ...]
    phase_counts: tuple[int, ...]
    fingerprint: str


def play_match(policies, seed, *, policy_seed=0, max_actions=5000, validate=False):
    """Call one policy per seat, each with only a Decision and its own RNG.

    Policies return a flat index or a complete five-count discard tuple. Reward
    is one for the actual winning seat, zero otherwise; cuts are not losses.
    """
    policies = tuple(policies)
    players = _player_count(len(policies))
    if any(not callable(policy) for policy in policies):
        raise ValueError("each seat needs a callable policy")
    seed, policy_seed = _integer(seed, "seed"), _integer(policy_seed, "policy_seed")
    if type(validate) is not bool:
        raise ValueError("validate must be a bool")
    env = CatanEnv(max_actions=max_actions)
    request = None
    try:
        request = env.reset(seed, n_players=players)
        if validate:
            assert_invariants(env._state)
    except Exception as error:
        error.add_note(
            f"evaluation seed={seed}, policy_seed={policy_seed}, players={players}, "
            f"action=0, actor={getattr(request, 'player', None)}, phase=INITIALIZE"
        )
        raise
    streams = [
        Random(f"catan-evaluation-v1:{policy_seed}:{seed}:{p}") for p in range(players)
    ]
    counts = [0] * len(Phase)
    counts[request.phase.value - 1] += 1
    turns = 0
    while True:
        action_number = env.actions + 1
        try:
            choice = policies[request.player](request, streams[request.player])
            result = env.step(choice, player=request.player)
            if request.phase != Phase.DISCARD and int(choice) == 219:
                turns += 1
            if validate:
                assert_invariants(env._state)
        except Exception as error:
            error.add_note(
                f"evaluation seed={seed}, policy_seed={policy_seed}, players={players}, action={action_number}, actor={request.player}, phase={request.phase.name}"
            )
            raise
        counts[env._state.phase.value - 1] += 1
        if result.terminated or result.truncated:
            break
        request = env.decision()
    digest = sha256(_fingerprint(env._state, streams[0]).encode())
    digest.update(repr(tuple(stream.getstate() for stream in streams)).encode())
    return MatchResult(
        seed,
        policy_seed,
        players,
        env.actions,
        turns,
        result.winner,
        result.terminated,
        result.truncated,
        result.rewards,
        tuple(map(int, env._state.vp)),
        tuple(counts),
        digest.hexdigest(),
    )


def evaluate(
    *,
    candidate="greedy",
    opponent="random",
    seeds=range(10),
    n_players=4,
    policy_seed=0,
    max_actions=5000,
    validate=False,
):
    """One candidate seat against opponents, rotating all seats per distinct seed.

    Same-policy controls reuse the same game per rotation; observations are not
    independent samples. Different policies can change subsequent random draws.
    """
    if candidate not in POLICIES or opponent not in POLICIES:
        raise ValueError("unknown policy name")
    players = _player_count(n_players)
    seeds = tuple(_integer(seed, "seed") for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    policy_seed = _integer(policy_seed, "policy_seed")
    CatanEnv(max_actions=max_actions)  # Validate bounds before starting work.
    if type(validate) is not bool:
        raise ValueError("validate must be a bool")
    records = []
    matches_played = executed_actions = 0
    start = perf_counter()
    for seed in seeds:
        control = None
        for seat in range(players):
            policies = [POLICIES[opponent]] * players
            policies[seat] = POLICIES[candidate]
            if control is None:
                match = play_match(
                    policies,
                    seed,
                    policy_seed=policy_seed,
                    max_actions=max_actions,
                    validate=validate,
                )
                matches_played += 1
                executed_actions += match.actions
                # Built-in policies are stateless. Identical-policy rotations
                # differ only in the candidate label, not the played trajectory.
                if candidate == opponent:
                    control = match
            else:
                match = control
            record = asdict(match)
            record["candidate_seat"] = seat
            records.append(record)
    elapsed = perf_counter() - start

    def summarize(games):
        completed = sum(game["terminated"] for game in games)
        wins = sum(game["winner"] == game["candidate_seat"] for game in games)
        return {
            "games": len(games),
            "completed": completed,
            "capped": len(games) - completed,
            "candidate_wins": wins,
            "candidate_losses": completed - wins,
            "win_rate_all": wins / len(games),
            "win_rate_completed": wins / completed if completed else None,
            "mean_actions": sum(game["actions"] for game in games) / len(games),
            "mean_completed_turns": sum(game["completed_turns"] for game in games)
            / len(games),
            "mean_candidate_score_all": sum(
                game["scores"][game["candidate_seat"]] for game in games
            )
            / len(games),
        }

    actions = sum(game["actions"] for game in records)
    ordered = sha256()
    for game in records:
        ordered.update(
            repr((game["seed"], game["candidate_seat"], game["fingerprint"])).encode()
        )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "environment_schema": ENV_SCHEMA_VERSION,
        "observation_schema": SCHEMA_VERSION,
        "action_schema": ACTION_SCHEMA_VERSION,
        "reward_scheme": REWARD_SCHEME,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "candidate": candidate,
        "candidate_version": POLICY_VERSIONS[candidate],
        "opponent": opponent,
        "opponent_version": POLICY_VERSIONS[opponent],
        "seeds": seeds,
        "players": players,
        "policy_seed": policy_seed,
        "max_actions": int(max_actions),
        "validation_enabled": validate,
        "summary": summarize(records),
        "by_seat": {
            str(seat): summarize(
                [game for game in records if game["candidate_seat"] == seat]
            )
            for seat in range(players)
        },
        "actions": actions,  # Sum over reported role assignments.
        "matches_played": matches_played,
        "executed_actions": executed_actions,
        "elapsed_seconds": elapsed,
        "actions_per_second": executed_actions / elapsed,
        "fingerprint": ordered.hexdigest(),
        "games": records,
    }

"""Bounded batched neural-versus-reference evaluation, no optimizer or targets."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from catan.state import _integer
from learning.policy import NeuralPolicy
from rl.collector import ReferencePolicy, collect_rollouts
from rl.encoding import FEATURE_SLICES
from rl.policies import POLICIES

EVALUATION_SCHEMA_VERSION = 1


class EvaluationPaused(RuntimeError):
    """Completed chunks are committed; no in-flight policy/game state is needed."""


class EvaluationPolicy:
    """Route one neural seat and fixed reference/neural opponents in batches.

    The combined tag deliberately cannot enter PPO. Training must explicitly
    record only the learner seat, with its own version tag and verified behavior.
    """

    def __init__(self, policy, opponent, seat):
        if not isinstance(policy, NeuralPolicy):
            raise ValueError("use a neural policy and a known reference opponent")
        self.policy = policy
        if isinstance(opponent, NeuralPolicy):
            self.reference = opponent
        elif isinstance(opponent, str) and opponent in POLICIES:
            self.reference = ReferencePolicy(opponent)
        else:
            raise ValueError("use a known reference or a fixed neural opponent")
        self.seat = _integer(seat, "seat")
        if not 0 <= self.seat < 4:
            raise ValueError("candidate seat outside range")
        self.tag = f"evaluation:{policy.tag}:vs:{self.reference.tag}:seat{seat}"

    def __call__(self, decisions, rngs):
        if len(decisions) != len(rngs):
            raise ValueError("one RNG per evaluation decision")
        selected = [i for i, d in enumerate(decisions) if d.player == self.seat]
        others = [i for i, d in enumerate(decisions) if d.player != self.seat]
        output = [None] * len(decisions)
        for indices, callback in ((selected, self.policy), (others, self.reference)):
            samples = callback(
                tuple(decisions[i] for i in indices), tuple(rngs[i] for i in indices)
            )
            for i, sample in zip(indices, samples, strict=True):
                output[i] = sample
        return tuple(output)


def _summary(records):
    games = len(records)
    completed = sum(r["terminated"] for r in records)
    wins = sum(r["winner"] == r["candidate_seat"] for r in records)
    result = {
        "games": games,
        "completed": completed,
        "cutoffs": games - completed,
        "wins": wins,
        "losses": completed - wins,
        "win_rate_all": wins / games if games else None,
        "win_rate_completed": wins / completed if completed else None,
        "mean_actions": sum(r["actions"] for r in records) / games if games else None,
    }
    if records and all("candidate_vp" in r for r in records):
        result["mean_candidate_vp"] = sum(r["candidate_vp"] for r in records) / games
    return result


def evaluate_neural(
    policy,
    seeds,
    *,
    max_actions=5000,
    batch_size=8,
    policy_seed=17,
    validate=False,
    env_factory=None,
    progress_path=None,
    progress_identity=None,
    should_pause=None,
):
    """Rotate both player counts/all seats against random and greedy opponents.

    Temporary collector columns are released after each bounded seed chunk. They
    never enter PPO: reference rows deliberately have no invented critic values.
    Thread settings and model immutability throughout evaluation belong to caller.
    """
    if not isinstance(policy, NeuralPolicy):
        raise ValueError("policy must be NeuralPolicy")
    seeds = tuple(_integer(s, "seed") for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be distinct and nonempty")
    batch_size = _integer(batch_size, "batch_size")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = []
    completed = 0
    identity = {
        "checkpoint": progress_identity,
        "seeds": list(seeds),
        "max_actions": max_actions,
        "batch_size": batch_size,
        "policy_seed": policy_seed,
        "policy_tag": policy.tag,
        "deterministic": policy.deterministic,
        "environment": repr(env_factory),
    }
    if progress_path is not None:
        if not isinstance(progress_identity, str) or not progress_identity:
            raise ValueError("resumable evaluation requires checkpoint identity")
        progress_path = Path(progress_path)
        if progress_path.exists():
            saved = json.loads(progress_path.read_text())
            if saved.get("schema") != 1 or saved["identity"] != identity:
                raise ValueError("evaluation checkpoint/protocol changed")
            completed, records = saved["completed_chunks"], saved["records"]
            if type(completed) is not int or completed < 0:
                raise ValueError("invalid evaluation cursor")
            scheduled = [
                (s, players, opponent, seat)
                for players in (3, 4)
                for opponent in ("random", "greedy")
                for seat in range(players)
                for start in range(0, len(seeds), batch_size)
                for s in seeds[start : start + batch_size]
            ]
            chunks_per_role = (len(seeds) + batch_size - 1) // batch_size
            if completed > 14 * chunks_per_role:
                raise ValueError("invalid evaluation cursor")
            full_roles, tail_chunks = divmod(completed, chunks_per_role)
            record_count = full_roles * len(seeds) + min(
                tail_chunks * batch_size, len(seeds)
            )
            actual = [
                (r["seed"], r["players"], r["opponent"], r["candidate_seat"])
                for r in records
            ]
            if actual != scheduled[:record_count]:
                raise ValueError("evaluation partial roles/cursor are inconsistent")
    chunk = 0
    expected_keys = []
    for players in (3, 4):
        for opponent in ("random", "greedy"):
            for seat in range(players):
                callback = EvaluationPolicy(policy, opponent, seat)
                for start in range(0, len(seeds), batch_size):
                    chunk_seeds = seeds[start : start + batch_size]
                    keys = [(s, players, opponent, seat) for s in chunk_seeds]
                    expected_keys.extend(keys)
                    chunk += 1
                    if chunk <= completed:
                        continue
                    if should_pause is not None and should_pause():
                        raise EvaluationPaused("evaluation paused before next chunk")
                    batch = collect_rollouts(
                        callback,
                        chunk_seeds,
                        n_players=players,
                        max_actions=max_actions,
                        batch_size=batch_size,
                        policy_seed=policy_seed,
                        validate=validate,
                        env_factory=(
                            env_factory.for_observer(seat)
                            if env_factory is not None
                            and hasattr(env_factory, "for_observer")
                            else env_factory
                        ),
                        record_players=(seat,),
                    )
                    for episode_index, episode in enumerate(batch.episodes):
                        record = asdict(episode)
                        record.update(opponent=opponent, candidate_seat=seat)
                        if env_factory is not None:
                            # Own final score is diagnostic reporting, never a policy
                            # input or a substitute for an actual terminal win.
                            score_column = FEATURE_SLICES["players"].start + 1
                            record["candidate_vp"] = int(
                                round(
                                    float(
                                        batch.final_observations[
                                            episode_index, seat, score_column
                                        ]
                                    )
                                    * 10
                                )
                            )
                        records.append(record)
                    del batch
                    if progress_path is not None:
                        from learning.training import _atomic_json

                        _atomic_json(
                            progress_path,
                            {
                                "schema": 1,
                                "identity": identity,
                                "completed_chunks": chunk,
                                "records": records,
                            },
                        )
    actual_keys = [
        (r["seed"], r["players"], r["opponent"], r["candidate_seat"]) for r in records
    ]
    if completed > chunk or actual_keys != expected_keys:
        raise ValueError("evaluation partial roles/cursor are inconsistent")
    groups = {
        f"{players}p-vs-{opponent}": _summary(
            [
                r
                for r in records
                if r["players"] == players and r["opponent"] == opponent
            ]
        )
        for players in (3, 4)
        for opponent in ("random", "greedy")
    }
    return {
        "schema": EVALUATION_SCHEMA_VERSION,
        "policy_tag": policy.tag,
        "deterministic": policy.deterministic,
        "seeds": seeds,
        "policy_seed": policy_seed,
        "max_actions": max_actions,
        "batch_size": batch_size,
        "summary": _summary(records),
        "groups": groups,
        "actions": sum(r["actions"] for r in records),
        "records": records,
        "fingerprint": sha256(repr(records).encode()).hexdigest(),
    }


def compare_evaluations(
    baseline,
    candidate,
    *,
    bootstrap_seed=71,
    draws=10000,
    opponent=None,
    confidence=0.95,
):
    """Paired win-rate change; bootstrap whole board seeds, never individual seats.

    A cap is reported separately and contributes no real win to the all-game
    rate. Shared seeds control initial boards, not subsequent policy/chance draws.
    """
    if tuple(baseline["seeds"]) != tuple(candidate["seeds"]) or any(
        baseline[name] != candidate[name]
        for name in ("policy_seed", "max_actions", "deterministic")
    ):
        raise ValueError("comparisons require the same evaluation protocol/seeds")
    draws = _integer(draws, "draws")
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if opponent not in (None, "random", "greedy"):
        raise ValueError("unknown comparison opponent")

    def key(record):
        return (
            record["seed"],
            record["players"],
            record["opponent"],
            record["candidate_seat"],
        )

    left = {key(r): r for r in baseline["records"]}
    right = {key(r): r for r in candidate["records"]}
    if (
        len(left) != len(baseline["records"])
        or left.keys() != right.keys()
        or len(right) != len(candidate["records"])
    ):
        raise ValueError("comparison roles must match uniquely")
    if opponent is not None:
        left = {k: r for k, r in left.items() if k[2] == opponent}
        right = {k: r for k, r in right.items() if k[2] == opponent}
    changes = []
    for seed in baseline["seeds"]:
        pairs = [k for k in left if k[0] == seed]
        if not pairs:
            raise ValueError("every comparison seed needs matched roles")
        changes.append(
            np.mean(
                [
                    float(right[k]["winner"] == k[3]) - float(left[k]["winner"] == k[3])
                    for k in pairs
                ]
            )
        )
    deltas = np.asarray(changes, dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    sampled = deltas[rng.integers(len(deltas), size=(draws, len(deltas)))].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(
        sampled, (0.025, 0.975) if confidence == 0.95 else (tail, 1 - tail)
    )
    result = {
        "win_rate_all_change": float(deltas.mean()),
        "seed_clusters": len(deltas),
        "paired_games": len(left),
        "bootstrap_draws": draws,
        "opponent": opponent,
        (
            "seed_cluster_bootstrap_95_percent_interval"
            if confidence == 0.95
            else "seed_cluster_bootstrap_interval"
        ): [float(low), float(high)],
        "baseline": baseline["summary"]
        if opponent is None
        else _summary(tuple(left.values())),
        "candidate": candidate["summary"]
        if opponent is None
        else _summary(tuple(right.values())),
        "note": "Interval resamples board seeds across all roles; conditional on this seed set/protocol, not a universal strength guarantee.",
    }
    if confidence != 0.95:
        result["confidence"] = confidence
    return result

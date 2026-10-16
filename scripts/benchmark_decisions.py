"""Bounded, frozen decision locations for development diagnostics, never PPO data.

Only engine-proved winning deterministic actions have correctness labels.
Production and material-gap scores are descriptive, not optimal-play targets.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import numpy as np
import torch

from catan.engine import apply_action_inplace, winner
from catan.state import Phase
from catan.validation import _fingerprint, assert_invariants
from learning.model import ActorCritic, EconomicActorCritic
from learning.graph import GraphActorCritic
from learning.policy import NeuralPolicy
from learning.ppo import PPOUpdater
from rl.actions import ACTION_DIM, indexed_action
from rl.economics import COSTS, ECONOMIC_DIM, EconomicFeatures
from rl.encoding import OBS_DIM
from rl.env import CatanEnv, Decision
from rl.policies import POLICIES

SUITE_SCHEMA = 1
CATEGORIES = (
    "placement",
    "affordable_build",
    "near_purchase",
    "trade",
    "discard",
    "win",
)
HISTORICAL_ROLE = (3000003013, 4, 2, "greedy")
POLICY_SEED = 17


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def gaps(hand):
    """One gap per purchase; ignores legality, pieces and future strategy."""
    return np.maximum(COSTS - np.asarray(hand), 0).sum(axis=1).astype(int).tolist()


def winning_actions(state, decision):
    """Prove deterministic road/build/knight wins on isolated engine copies.

    Never inspect a future roll, steal or development-card draw to label a move.
    This is a deliberately incomplete set of possible winning actions.
    """
    result = np.zeros(ACTION_DIM, dtype=np.bool_)
    if state.phase not in (Phase.MAIN, Phase.ROLL) or state.vp[decision.player] < 8:
        return result
    choices = list(np.flatnonzero(decision.mask[:180]))
    if decision.mask[226]:
        choices.append(226)
    for choice in choices:
        fork = state.copy()
        apply_action_inplace(fork, indexed_action(fork, int(choice)))
        result[choice] = winner(fork) == decision.player
    return result


def describe(state, decision, choice, features):
    hand = state.hands[decision.player].astype(int)
    detail = {"gaps_before": gaps(hand)}
    if decision.phase == Phase.SETUP_SETTLEMENT:
        sites = features[220:490].reshape(54, 5).sum(axis=1)
        vertices = np.flatnonzero(decision.mask[72:126])
        best = float(sites[vertices].max())
        detail["production_ratio"] = (
            float(sites[int(choice) - 72]) / best if best else 1.0
        )
    if decision.phase == Phase.MAIN and 199 <= choice < 219:
        action = indexed_action(state, choice)
        hand[action.give] -= round(float(features[490 + action.give]) * 4)
        hand[action.receive] += 1
        detail["gaps_after"] = gaps(hand)
    if decision.phase == Phase.DISCARD:
        after = hand - choice
        keep = int(after.sum())
        # Exact per-purchase lower bounds: keep as many matching cards as possible.
        best = np.maximum(
            COSTS.sum(axis=1) - np.minimum(np.minimum(COSTS, hand).sum(axis=1), keep), 0
        )
        detail["gaps_after"] = gaps(after)
        detail["best_individual_gaps_after"] = best.astype(int).tolist()
    return detail


class CasePool:
    """Stable hash-priority cap, independent of every policy/game RNG."""

    def __init__(self, per_category=128):
        if type(per_category) is not int or not 1 <= per_category <= 1024:
            raise ValueError("per_category must be between 1 and 1024")
        self.limit = per_category
        self.groups = {name: [] for name in CATEGORIES}

    def offer(self, category, case):
        if category not in self.groups:
            raise ValueError("unknown decision category")
        key = json.dumps([category, case["role"], case["step"]], separators=(",", ":"))
        rank = sha256(key.encode()).hexdigest()
        group = self.groups[category]
        group.append((rank, case))
        group.sort(key=lambda item: item[0])
        del group[self.limit :]

    def cases(self):
        return [
            (category, case)
            for category in CATEGORIES
            for _, case in self.groups[category]
        ]


def inspect_game(
    policy,
    role,
    pool=None,
    *,
    historical=False,
    validate=False,
    enabled_profile=False,
    max_actions=5000,
    probe=None,
):
    """Offline snapshots/labels; only Decisions reach the acting policies."""
    seed, players, seat, opponent = role
    env = CatanEnv(max_actions=max_actions)
    decision = env.reset(seed, n_players=players)
    features = EconomicFeatures(env.snapshot().board)
    streams = [
        Random(f"catan-evaluation-v1:{POLICY_SEED}:{seed}:{p}") for p in range(players)
    ]
    seen = Counter()
    captured = None
    while decision is not None:
        actor = decision.player
        if validate:
            assert_invariants(env.snapshot())
        if actor == seat:
            state = env.snapshot()
            extra = features.encode(state, actor)
            policy_decision = decision
            if enabled_profile:
                from dataclasses import replace

                policy_decision = replace(
                    decision, observation=np.concatenate((decision.observation, extra))
                )
            sample = policy((policy_decision,), (streams[actor],))[0]
            choice = sample.action
            won = winning_actions(state, decision)
            categories = []
            if decision.phase == Phase.SETUP_SETTLEMENT:
                categories.append("placement")
            elif decision.phase == Phase.MAIN:
                if decision.mask[72:180].any():
                    categories.append("affordable_build")
                if 1 in gaps(state.hands[actor])[1:]:
                    categories.append("near_purchase")
                if 199 <= choice < 219:
                    categories.append("trade")
            elif decision.phase == Phase.DISCARD:
                categories.append("discard")
            if won.any():
                categories.append("win")
            if historical:
                categories = ["win"] if won.any() else []
            if probe == env.actions:
                captured = (state, decision, choice, extra, won)
            if pool is not None:
                for category in categories:
                    # Bound contribution from a single trajectory before global cap.
                    if seen[category] >= 2:
                        continue
                    seen[category] += 1
                    pool.offer(
                        category,
                        {
                            "role": list(role),
                            "step": env.actions,
                            "historical": historical,
                            "phase": int(decision.phase),
                            "observation": decision.observation.copy(),
                            "economics": extra,
                            "mask": (
                                decision.mask.copy()
                                if decision.mask is not None
                                else np.zeros(ACTION_DIM, dtype=np.bool_)
                            ),
                            "action": -1 if decision.mask is None else int(choice),
                            "hand": list(map(int, state.hands[actor])),
                            "counts": list(map(int, choice))
                            if decision.mask is None
                            else [0] * 5,
                            "wins": won,
                            "detail": describe(state, decision, choice, extra),
                        },
                    )
        else:
            choice = POLICIES[opponent](decision, streams[actor])
        env.step(choice, player=actor)
        decision = env.decision()
    state = env.snapshot()
    if validate:
        assert_invariants(state)
    result = {
        "role": list(role),
        "historical": historical,
        "actions": env.actions,
        "winner": winner(state),
        "cutoff": env.truncated,
        "candidate_vp": int(state.vp[seat]),
        "fingerprint": _fingerprint(state, Random(0)),
        "policy_rng_sha256": sha256(
            repr([r.getstate() for r in streams]).encode()
        ).hexdigest(),
    }
    return result, captured


def build_suite(
    checkpoint,
    output,
    *,
    boards=16,
    seed=3300000000,
    per_category=128,
    include_historical=True,
    max_actions=5000,
):
    if type(boards) is not int or not 1 <= boards <= 64:
        raise ValueError("boards must be between 1 and 64")
    if type(seed) is not int or seed < 0 or seed + boards >= 2**32:
        raise ValueError("invalid board seed range")
    updater = PPOUpdater.load(checkpoint)
    if type(updater.model) is not ActorCritic:
        raise ValueError("freeze decision locations with the original model1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pool = CasePool(per_category)
    episodes = []
    policy = NeuralPolicy(updater.model)
    for board_seed in range(seed, seed + boards):
        for players in (3, 4):
            for opponent in ("random", "greedy"):
                for seat in range(players):
                    result, _ = inspect_game(
                        policy,
                        (board_seed, players, seat, opponent),
                        pool,
                        max_actions=max_actions,
                    )
                    episodes.append(result)
    if include_historical:
        # Kept separately: this board was already used in an earlier diagnosis.
        result, _ = inspect_game(
            policy, HISTORICAL_ROLE, pool, historical=True, max_actions=max_actions
        )
        episodes.append(result)
    selected = pool.cases()
    if not selected:
        raise ValueError("no decision cases generated")
    fields = ("observation", "economics", "mask", "action", "hand", "counts", "wins")
    arrays = {name: np.stack([case[name] for _, case in selected]) for name in fields}
    np.savez_compressed(output / "decisions.npz", **arrays)
    cases = [
        {"category": category, **{k: v for k, v in case.items() if k not in fields}}
        for category, case in selected
    ]
    write_json(output / "cases.json", cases)
    write_json(output / "episodes.json", episodes)
    manifest = {
        "schema": SUITE_SCHEMA,
        "source_checkpoint_sha256": digest(checkpoint),
        "source_model_schema": list(updater.model.schema),
        "policy_seed": POLICY_SEED,
        "max_actions": max_actions,
        "boards": boards,
        "board_seed_start": seed,
        "per_category": per_category,
        "rows": len(cases),
        "purpose": "development diagnostics only; not final testing or training data",
        "label_scope": "engine-proved deterministic road/build/knight wins only; other scores descriptive",
        "files": {
            name: digest(output / name)
            for name in ("decisions.npz", "cases.json", "episodes.json")
        },
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "baseline-scores.json", score_model(updater.model, output))
    return manifest


def load_suite(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema"] != SUITE_SCHEMA or set(manifest["files"]) != {
        "decisions.npz",
        "cases.json",
        "episodes.json",
    }:
        raise ValueError("incompatible decision suite")
    for name, expected in manifest["files"].items():
        if digest(path / name) != expected:
            raise ValueError(f"changed decision suite file: {name}")
    cases = json.loads((path / "cases.json").read_text())
    with np.load(path / "decisions.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    n = len(cases)
    for name, shape in (
        ("observation", (n, OBS_DIM)),
        ("economics", (n, ECONOMIC_DIM)),
        ("mask", (n, ACTION_DIM)),
        ("action", (n,)),
        ("hand", (n, 5)),
        ("counts", (n, 5)),
        ("wins", (n, ACTION_DIM)),
    ):
        if arrays[name].shape != shape:
            raise ValueError(f"invalid decision column: {name}")
    if n != manifest["rows"]:
        raise ValueError("invalid decision count")
    return manifest, cases, arrays


def score_model(model, path):
    manifest, cases, arrays = load_suite(path)
    if type(model) not in (ActorCritic, EconomicActorCritic, GraphActorCritic):
        raise ValueError("benchmark supports original, economic or graph models")
    observation = arrays["observation"]
    if model.schema[1] == 4:
        observation = np.concatenate((observation, arrays["economics"]), axis=1)
    mask = torch.from_numpy(arrays["mask"])
    actions = arrays["action"].astype(np.int64)
    discards = np.flatnonzero(actions == -1)
    with torch.inference_mode():
        output = model(torch.from_numpy(observation))
        regular = actions != -1
        probabilities = (
            output.logits[regular].masked_fill(~mask[regular], -torch.inf).softmax(1)
        )
        winning_mass = (probabilities * torch.from_numpy(arrays["wins"][regular])).sum(
            1
        )
        evaluation = model.evaluate_actions(
            torch.from_numpy(observation),
            mask,
            torch.from_numpy(actions),
            torch.from_numpy(discards),
            torch.from_numpy(arrays["hand"][discards]),
            torch.from_numpy(arrays["counts"][discards]),
        )
    regular_rows = np.flatnonzero(regular).tolist()
    probability_rows = dict(zip(regular_rows, probabilities.numpy(), strict=True))
    groups = {}
    win_lookup = dict(zip(regular_rows, winning_mass.tolist(), strict=True))
    for category in CATEGORIES:
        rows = [i for i, case in enumerate(cases) if case["category"] == category]
        groups[category] = {"rows": len(rows)}
        if rows:
            groups[category]["mean_recorded_action_log_probability"] = float(
                evaluation.log_probs[rows].mean()
            )
        if category == "win" and rows:
            groups[category].update(
                mean_verified_win_probability=float(
                    np.mean([win_lookup[i] for i in rows])
                ),
                source_actions_winning=sum(
                    bool(arrays["wins"][i, actions[i]]) for i in rows
                ),
            )
        if category == "placement" and rows:
            groups[category]["source_mean_production_ratio"] = float(
                np.mean([cases[i]["detail"]["production_ratio"] for i in rows])
            )
            ratios = []
            for i in rows:
                yields = arrays["economics"][i, 220:490].reshape(54, 5).sum(axis=1)
                legal = arrays["mask"][i, 72:126]
                best = float(yields[legal].max())
                ratios.append(
                    float(probability_rows[i][72:126] @ yields) / best if best else 1.0
                )
            groups[category]["mean_expected_production_ratio"] = float(np.mean(ratios))
        if category in ("affordable_build", "near_purchase", "trade") and rows:
            groups[category].update(
                mean_build_probability=float(
                    np.mean([probability_rows[i][72:180].sum() for i in rows])
                ),
                mean_trade_probability=float(
                    np.mean([probability_rows[i][199:219].sum() for i in rows])
                ),
                mean_end_turn_probability=float(
                    np.mean([probability_rows[i][219] for i in rows])
                ),
            )
        if category == "discard" and rows:
            requests = [
                Decision(
                    0,
                    Phase.DISCARD,
                    observation[i],
                    None,
                    tuple(map(int, arrays["hand"][i])),
                    int(arrays["hand"][i].sum()) // 2,
                )
                for i in rows
            ]
            samples = NeuralPolicy(model, deterministic=True)(
                requests, [Random(0) for _ in rows]
            )
            excess = [
                np.asarray(gaps(arrays["hand"][i] - sample.action))
                - cases[i]["detail"]["best_individual_gaps_after"]
                for i, sample in zip(rows, samples, strict=True)
            ]
            groups[category]["deterministic_discard_mean_gap_excess_by_purchase"] = (
                np.mean(excess, axis=0).tolist()
            )
    return {
        "suite_manifest_sha256": digest(Path(path) / "manifest.json"),
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "model_schema": list(model.schema),
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--boards", type=int, default=16)
    build.add_argument("--seed", type=int, default=3300000000)
    score = sub.add_parser("score")
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--suite", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "build":
        result = build_suite(
            args.checkpoint, args.output, boards=args.boards, seed=args.seed
        )
    else:
        if args.output.exists():
            raise ValueError("score output already exists")
        model = PPOUpdater.load(args.checkpoint).model
        result = score_model(model, args.suite)
        result["evaluated_checkpoint_sha256"] = digest(args.checkpoint)
        write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

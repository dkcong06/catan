"""Inspect frozen policies and phase controls; no training or hot-path hooks."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import numpy as np
import torch

from catan.actions import DiscardAction
from catan.board import GEOMETRY, Resource
from catan.engine import _trade_ratios
from catan.state import Phase, _integer
from catan.validation import _fingerprint, assert_invariants
from learning.policy import NeuralPolicy
from learning.ppo import PPOUpdater
from rl.actions import indexed_action
from rl.env import CatanEnv
from rl.policies import POLICIES, _COSTS, _deficit

VARIANTS = (
    "baseline",
    "deterministic",
    "greedy_setup",
    "build_first",
    "greedy_main",
    "greedy_trades",
    "greedy_discards",
    "greedy_cards",
    "greedy_setup_main",
)


def production_pips(board, vertex):
    """Expected production numerator out of 36; ignores robber and scarcity."""
    output = [0] * 5
    for tile in GEOMETRY.vertex_tiles[vertex]:
        resource, number = board.resources[tile], board.numbers[tile]
        if resource != Resource.DESERT:
            output[resource] += 6 - abs(7 - number)
    return output


def purchase_deficit(state, player, hand):
    """Material distance to settlement/city/card, not reachability or optimality."""
    costs = [
        cost
        for i, cost in enumerate(_COSTS)
        if (state.pieces_left[player, i] if i < 2 else any(state.development_deck))
    ]
    return min((_deficit(hand, cost) for cost in costs), default=0)


def override_choice(request, proposed, greedy, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown diagnostic variant")
    if request.phase == Phase.DISCARD:
        return greedy if variant == "greedy_discards" else proposed
    if variant in ("greedy_setup", "greedy_setup_main") and request.phase in (
        Phase.SETUP_SETTLEMENT,
        Phase.SETUP_ROAD,
    ):
        return greedy
    if variant in ("greedy_main", "greedy_setup_main") and request.phase == Phase.MAIN:
        return greedy
    if (
        variant == "build_first"
        and request.phase == Phase.MAIN
        and request.mask[72:180].any()
    ):
        return greedy
    if (
        variant == "greedy_trades"
        and request.phase == Phase.MAIN
        and (199 <= proposed < 219 or 199 <= greedy < 219)
    ):
        return greedy
    if (
        variant == "greedy_cards"
        and request.phase in (Phase.MAIN, Phase.ROLL)
        and (proposed >= 225 or greedy >= 225)
    ):
        return greedy
    return proposed


def _action(state, player, choice):
    return (
        DiscardAction(player, choice)
        if state.phase == Phase.DISCARD
        else indexed_action(state, choice)
    )


def _probabilities(policy, request, chosen, greedy):
    if request.mask is None:
        return None
    with torch.inference_mode():
        output = policy.model(
            torch.tensor(request.observation[None, :], device=policy.device)
        )
        logits = output.logits[0].masked_fill(
            ~torch.tensor(request.mask, device=policy.device), -torch.inf
        )
        probabilities = logits.softmax(0).cpu().tolist()
    return dict(
        chosen=probabilities[chosen],
        greedy=probabilities[greedy],
        settlement=sum(probabilities[72:126]),
        city=sum(probabilities[126:180]),
        road=sum(probabilities[:72]),
        trade=sum(probabilities[199:219]),
        end_turn=probabilities[219],
        buy_development=probabilities[225],
    )


def inspect_match(
    policy,
    seed,
    players,
    seat,
    *,
    opponent="greedy",
    variant="baseline",
    max_actions=5000,
    trace=None,
    validate=False,
):
    """Snapshot-only diagnostics never provide extra state to either policy.

    Phase controls consume the neural proposal first, then replace selected
    choices. They are evaluation-only hybrid controllers, not trained policies.
    Changed actions can change all subsequent chance/policy trajectories.
    """
    if variant not in VARIANTS or opponent not in POLICIES:
        raise ValueError("unknown diagnostic variant or opponent")
    seat = _integer(seat, "seat")
    if not 0 <= seat < players:
        raise ValueError("candidate seat outside game")
    env = CatanEnv(max_actions=max_actions)
    request = env.reset(seed, n_players=players)
    streams = [Random(f"catan-evaluation-v1:17:{seed}:{p}") for p in range(players)]
    turns = [0] * players
    actions = [Counter() for _ in range(players)]
    setups = [[] for _ in range(players)]
    income = np.zeros((players, 5), dtype=np.int64)
    first_build = [{} for _ in range(players)]
    metrics = Counter()
    examples = {name: [] for name in ("setup", "road_over_build", "trade", "discard")}
    while request is not None:
        state = env.snapshot()  # Deliberate diagnostic work, never normal training.
        if validate:
            assert_invariants(state)
        actor = request.player
        greedy = POLICIES["greedy"](request, Random(0))
        sample = policy((request,), (streams[actor],))[0] if actor == seat else None
        proposed = (
            sample.action
            if sample is not None
            else POLICIES[opponent](request, streams[actor])
        )
        choice = (
            override_choice(request, proposed, greedy, variant)
            if actor == seat
            else proposed
        )
        action = _action(state, actor, choice)
        kind = type(action).__name__
        actions[actor][kind] += 1
        interesting = None
        detail = {}
        if actor == seat:
            metrics["overridden_choices"] += int(choice != proposed)
        if request.phase == Phase.SETUP_SETTLEMENT:
            vertex = action.vertex
            pips = production_pips(state.board, vertex)
            legal = np.flatnonzero(request.mask[72:126])
            best = max(sum(production_pips(state.board, int(v))) for v in legal)
            detail = dict(
                vertex=vertex,
                tiles=[
                    dict(
                        tile=t,
                        resource=state.board.resources[t].name,
                        number=state.board.numbers[t],
                    )
                    for t in GEOMETRY.vertex_tiles[vertex]
                ],
                production_pips=sum(pips),
                resource_pips=pips,
                best_legal_pips=best,
                potential_ratio=sum(pips) / best if best else None,
            )
            setups[actor].append(detail)
            interesting = "setup"
        if actor == seat and request.phase == Phase.MAIN:
            builds = bool(request.mask[72:180].any())
            cities = bool(request.mask[126:180].any())
            metrics["main_decisions"] += 1
            metrics["build_available"] += int(builds)
            metrics["build_selected_when_available"] += int(
                builds and 72 <= choice < 180
            )
            metrics["city_available"] += int(cities)
            metrics["city_selected"] += int(126 <= choice < 180)
            metrics["end_with_build_available"] += int(builds and choice == 219)
            metrics["road_over_build"] += int(builds and kind == "RoadAction")
            metrics["trade_over_build"] += int(builds and kind == "BankTradeAction")
            metrics["greedy_agreement_main"] += int(choice == greedy)
            metrics["winning_point_available"] += int(builds and state.vp[actor] == 9)
            metrics["winning_point_declined"] += int(
                builds and state.vp[actor] == 9 and not 72 <= choice < 180
            )
            detail.update(
                legal_settlements=np.flatnonzero(request.mask[72:126]).tolist(),
                legal_cities=np.flatnonzero(request.mask[126:180]).tolist(),
            )
            if builds and kind == "RoadAction":
                interesting = "road_over_build"
        if request.phase == Phase.MAIN and kind in ("SettlementAction", "CityAction"):
            first_build[actor].setdefault(kind, turns[actor] + 1)
        if kind == "BankTradeAction":
            hand = state.hands[actor].tolist()
            ratio = _trade_ratios(state)[action.give]
            after_hand = hand.copy()
            after_hand[action.give] -= ratio
            after_hand[action.receive] += 1
            before_deficit = purchase_deficit(state, actor, hand)
            after_deficit = purchase_deficit(state, actor, after_hand)
            detail.update(
                give=Resource(action.give).name,
                receive=Resource(action.receive).name,
                ratio=ratio,
                deficit_before=before_deficit,
                deficit_after=after_deficit,
            )
            if actor == seat:
                metrics["trades"] += 1
                metrics["trade_reduces_purchase_deficit"] += int(
                    after_deficit < before_deficit
                )
                metrics["trade_increases_purchase_deficit"] += int(
                    after_deficit > before_deficit
                )
                metrics["trade_equal_purchase_deficit"] += int(
                    after_deficit == before_deficit
                )
                interesting = "trade"
        if actor == seat and kind == "DiscardAction":
            hand = state.hands[actor].tolist()
            remaining = [h - c for h, c in zip(hand, choice, strict=True)]
            greedy_remaining = [h - c for h, c in zip(hand, greedy, strict=True)]
            detail.update(
                counts=choice,
                greedy_counts=greedy,
                deficit_after=purchase_deficit(state, actor, remaining),
                greedy_deficit_after=purchase_deficit(state, actor, greedy_remaining),
            )
            metrics["discards"] += 1
            metrics["discard_worse_material_deficit"] += int(
                detail["deficit_after"] > detail["greedy_deficit_after"]
            )
            metrics["discard_better_material_deficit"] += int(
                detail["deficit_after"] < detail["greedy_deficit_after"]
            )
            interesting = "discard"
        if kind == "EndTurnAction":
            turns[actor] += 1
        retain_example = (
            actor == seat and interesting and len(examples[interesting]) < 12
        )
        row = None
        if trace is not None or retain_example:
            row = dict(
                step=env.actions + 1,
                actor=actor,
                phase=request.phase.name,
                candidate_turn=turns[seat] + (kind != "EndTurnAction" or actor != seat),
                candidate_hand_before=state.hands[seat].tolist(),
                candidate_score_before=int(state.vp[seat]),
                choice=asdict(action),
                action_type=kind,
                neural_proposal=proposed if actor == seat else None,
                greedy_proposal=greedy if actor == seat else None,
                neural_value=sample.value if sample is not None else None,
                neural_softmax_probabilities=_probabilities(
                    policy, request, choice, greedy
                )
                if actor == seat
                else None,
                detail=detail,
            )
        transition = env.step(choice, player=actor)
        after = env.snapshot()
        if kind == "RollAction":
            income += np.maximum(after.hands.astype(np.int64) - state.hands, 0)
        if row is not None:
            row.update(
                candidate_hand_after=after.hands[seat].tolist(),
                candidate_score_after=int(after.vp[seat]),
                last_roll=after.last_roll,
            )
            if retain_example:
                examples[interesting].append(row)
            if trace is not None:
                trace.write(json.dumps(row, allow_nan=False) + "\n")
        request = env.decision()
    if validate:
        assert_invariants(after)
    digest = sha256(_fingerprint(after, streams[0]).encode())
    digest.update(repr(tuple(stream.getstate() for stream in streams)).encode())
    setup_pips = [
        [sum(s["resource_pips"][r] for s in placements) for r in range(5)]
        for placements in setups
    ]
    return dict(
        seed=seed,
        players=players,
        candidate_seat=seat,
        variant=variant,
        opponent=opponent,
        actions=env.actions,
        winner=transition.winner,
        terminated=transition.terminated,
        truncated=transition.truncated,
        scores=after.vp.tolist(),
        completed_turns=turns,
        action_counts=[dict(c) for c in actions],
        first_build=first_build,
        setup=setups,
        setup_resource_pips=setup_pips,
        production_cards=income.tolist(),
        candidate_metrics=dict(metrics),
        examples=examples,
        fingerprint=digest.hexdigest(),
    )


def summarize(games):
    counts, candidate_actions = Counter(), Counter()
    placements, productions, setup_totals, missing_ore, missing_wheat = [], [], [], 0, 0
    for g in games:
        seat = g["candidate_seat"]
        counts.update(g["candidate_metrics"])
        candidate_actions.update(g["action_counts"][seat])
        placements.extend(g["setup"][seat])
        productions.append(sum(g["production_cards"][seat]))
        setup_totals.append(sum(g["setup_resource_pips"][seat]))
        missing_ore += int(g["setup_resource_pips"][seat][4] == 0)
        missing_wheat += int(g["setup_resource_pips"][seat][2] == 0)
    complete = sum(g["terminated"] for g in games)
    wins = sum(g["winner"] == g["candidate_seat"] for g in games)
    ratios = [
        p["potential_ratio"] for p in placements if p["potential_ratio"] is not None
    ]
    return dict(
        games=len(games),
        completed=complete,
        cutoffs=len(games) - complete,
        wins=wins,
        losses=complete - wins,
        win_rate_all=wins / len(games),
        mean_candidate_score=float(
            np.mean([g["scores"][g["candidate_seat"]] for g in games])
        ),
        mean_setup_pips=float(np.mean(setup_totals)),
        mean_setup_legal_potential_ratio=float(np.mean(ratios)) if ratios else None,
        setups_without_ore=missing_ore,
        setups_without_wheat=missing_wheat,
        mean_production_cards=float(np.mean(productions)),
        candidate_actions=dict(candidate_actions),
        candidate_metrics=dict(counts),
        actions=sum(g["actions"] for g in games),
    )


def paired_change(baseline, candidate):
    def key(g):
        return (g["seed"], g["players"], g["candidate_seat"])

    left, right = {key(g): g for g in baseline}, {key(g): g for g in candidate}
    if (
        len(left) != len(baseline)
        or len(right) != len(candidate)
        or left.keys() != right.keys()
    ):
        raise ValueError("phase comparisons need uniquely matching roles")
    seeds = sorted({g["seed"] for g in baseline})
    deltas = np.array(
        [
            np.mean(
                [
                    float(right[k]["winner"] == k[2]) - float(left[k]["winner"] == k[2])
                    for k in left
                    if k[0] == seed
                ]
            )
            for seed in seeds
        ]
    )
    draws = deltas[
        np.random.default_rng(91).integers(len(seeds), size=(10000, len(seeds)))
    ].mean(axis=1)
    return dict(
        seed_clusters=len(seeds),
        paired_games=len(left),
        win_rate_change=float(deltas.mean()),
        seed_cluster_bootstrap_95_percent_interval=np.quantile(
            draws, (0.025, 0.975)
        ).tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=3000003000)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--opponent", choices=POLICIES, default="greedy")
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:5])
    )
    parser.add_argument("--trace-seed", type=int)
    parser.add_argument("--trace-players", type=int, choices=(3, 4), default=4)
    parser.add_argument("--trace-seat", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 100 or not 1 <= args.max_actions <= 5000:
        parser.error("use 1-100 seeds and a 1-5000 action cap")
    if len(set(args.variants)) != len(args.variants) or args.variants[0] != "baseline":
        parser.error("include baseline first and distinct variants")
    if args.trace_seed is not None and (
        args.trace_seed not in range(args.seed_start, args.seed_start + args.seeds)
        or not 0 <= args.trace_seat < args.trace_players
    ):
        parser.error("trace seed/seat must belong to the declared game roles")
    torch.set_num_threads(1)
    checkpoint_hash = sha256(args.checkpoint.read_bytes()).hexdigest()
    learner = PPOUpdater.load(args.checkpoint)
    weights = {
        name: tensor.clone() for name, tensor in learner.model.state_dict().items()
    }
    args.output.mkdir(parents=True, exist_ok=False)
    all_games, summaries = {}, {}
    for variant in args.variants:
        policy = NeuralPolicy(
            learner.model,
            deterministic=variant == "deterministic",
            policy_version=f"ppo1-u{learner.updates}",
        )
        games = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for players in (3, 4):
                for seat in range(players):
                    is_trace = (seed, players, seat) == (
                        args.trace_seed,
                        args.trace_players,
                        args.trace_seat,
                    )
                    path = args.output / f"trace-{variant}.jsonl"
                    stream = path.open("w", encoding="utf-8") if is_trace else None
                    try:
                        games.append(
                            inspect_match(
                                policy,
                                seed,
                                players,
                                seat,
                                opponent=args.opponent,
                                variant=variant,
                                max_actions=args.max_actions,
                                trace=stream,
                                validate=args.validate,
                            )
                        )
                    finally:
                        if stream is not None:
                            stream.close()
        all_games[variant] = games
        summaries[variant] = summarize(games)
        (args.output / f"{variant}.json").write_text(
            json.dumps(games, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"variant": variant, "summary": summaries[variant]}, allow_nan=False
            ),
            flush=True,
        )
    if any(
        not torch.equal(weights[name], value)
        for name, value in learner.model.state_dict().items()
    ):
        raise RuntimeError("diagnostics changed model weights")
    if checkpoint_hash != sha256(args.checkpoint.read_bytes()).hexdigest():
        raise RuntimeError("checkpoint changed during diagnostics")
    comparisons = {
        v: paired_change(all_games["baseline"], games)
        for v, games in all_games.items()
        if v != "baseline"
    }
    result = dict(
        schema=1,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_hash,
        policy_version=learner.updates,
        weights_unchanged=True,
        policy_seed=17,
        seed_start=args.seed_start,
        seeds=args.seeds,
        max_actions=args.max_actions,
        opponent=args.opponent,
        variants=args.variants,
        summaries=summaries,
        paired_comparisons=comparisons,
        note="Frozen-policy diagnostic controls, not newly trained agents. Pips ignore robber/scarcity; material deficit is a heuristic. Same seeds fix initial boards, not subsequent chance streams. No test-set checkpoint selection.",
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

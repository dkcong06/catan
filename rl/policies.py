"""Versioned observation-only reference policies, not trained agents.

Random is uniform over legal flat indices and samples physical discard cards.
Greedy uses fixed public placement/build/card/trade priorities; no strength claim.
"""

from types import MappingProxyType

import numpy as np

from catan.board import GEOMETRY
from catan.state import Phase
from rl.actions import _CATALOG
from rl.encoding import FEATURE_SHAPES, FEATURE_SLICES

POLICY_VERSIONS = MappingProxyType({"random": 1, "greedy": 1})
_ROAD_FRONTIERS = tuple(
    tuple(
        sorted(
            {
                v
                for endpoint in ends
                for v in (endpoint,) + GEOMETRY.vertex_neighbors[endpoint]
            }
        )
    )
    for ends in GEOMETRY.edge_vertices
)
_COSTS = ((1, 1, 1, 1, 0), (0, 0, 2, 0, 3), (0, 1, 1, 0, 1))


def random_policy(decision, rng):
    if decision.discard_hand is not None:
        cards = [
            r for r, count in enumerate(decision.discard_hand) for _ in range(count)
        ]
        selected = rng.sample(cards, decision.discard_required)
        return tuple(selected.count(r) for r in range(5))
    choices = np.flatnonzero(decision.mask).tolist()
    if not choices:
        raise ValueError("decision has no legal actions")
    return rng.choice(choices)


def _deficit(hand, cost):
    return sum(max(0, needed - held) for needed, held in zip(cost, hand))


def greedy_policy(decision, rng):
    """Deterministic public heuristic; rng is accepted for the shared protocol."""
    if decision.discard_hand is not None:
        hand = list(decision.discard_hand)
        counts = [0] * 5
        for _ in range(decision.discard_required):
            resource = max(range(5), key=lambda r: (hand[r], -r))
            counts[resource] += 1
            hand[resource] -= 1
        return tuple(counts)
    legal = np.flatnonzero(decision.mask).tolist()
    if not legal:
        raise ValueError("decision has no legal actions")
    if len(legal) == 1:
        return legal[0]
    obs = decision.observation
    tiles = obs[FEATURE_SLICES["tiles"]].reshape(FEATURE_SHAPES["tiles"])
    vertices = obs[FEATURE_SLICES["vertices"]].reshape(FEATURE_SHAPES["vertices"])
    players = obs[FEATURE_SLICES["players"]].reshape(FEATURE_SHAPES["players"])
    probabilities = tiles[:, 6].tolist()
    potential_cache = {}
    empty = None

    def potential(vertex):
        if vertex not in potential_cache:
            potential_cache[vertex] = sum(
                probabilities[tile] for tile in GEOMETRY.vertex_tiles[vertex]
            )
        return potential_cache[vertex]

    def placement(indices, offset):
        return max(indices, key=lambda index: (potential(index - offset), -index))

    def road_score(index):
        nonlocal empty
        if empty is None:
            empty = vertices[:, 5].tolist()
        return max(
            (
                potential(v)
                for v in _ROAD_FRONTIERS[index]
                if empty[v] and all(empty[n] for n in GEOMETRY.vertex_neighbors[v])
            ),
            default=0,
        )

    if decision.phase == Phase.SETUP_SETTLEMENT:
        return placement(legal, 72)
    if decision.phase in (Phase.SETUP_ROAD, Phase.ROAD_BUILDING):
        return max(legal, key=lambda index: (road_score(index), -index))
    if decision.phase == Phase.ROBBER_MOVE:
        buildings = vertices[:, 6:8].tolist()
        own = vertices[:, 0].tolist()

        def robbery(index):
            tile = index - 180
            return probabilities[tile] * sum(
                (buildings[v][0] + 2 * buildings[v][1]) * (-2 if own[v] else 1)
                for v in GEOMETRY.tile_vertices[tile]
            )

        return max(legal, key=lambda index: (robbery(index), -index))
    if decision.phase == Phase.ROBBER_STEAL:
        totals = obs[FEATURE_SLICES["opponent_hand_sizes"]]
        return max(
            legal,
            key=lambda index: (
                float(players[index - 221, 1]) + float(totals[index - 222]),
                -index,
            ),
        )
    if decision.phase == Phase.MAIN:
        for low, high in ((72, 126), (126, 180)):
            candidates = [index for index in legal if low <= index < high]
            if candidates:
                return placement(candidates, low)
    for index in (227, 226):
        if index in legal:
            return index
    hand = [round(float(value) * 19) for value in obs[FEATURE_SLICES["my_hand"]]]
    targets = [
        cost
        for i, cost in enumerate(_COSTS)
        if (
            players[0, 2 + i] > 0
            if i < 2
            else obs[FEATURE_SLICES["development_deck_size"]][0] > 0
        )
    ]
    target = (
        min(targets, key=lambda cost: _deficit(hand, cost)) if targets else (0,) * 5
    )

    plenties = [index for index in legal if index >= 233]
    if plenties:
        return min(
            plenties,
            key=lambda index: (
                _deficit(
                    [
                        held + chosen
                        for held, chosen in zip(hand, _CATALOG[index].resources)
                    ],
                    target,
                ),
                index,
            ),
        )
    monopolies = [index for index in legal if 228 <= index < 233]
    if monopolies:
        return max(
            monopolies,
            key=lambda index: (max(0, target[index - 228] - hand[index - 228]), -index),
        )
    if 225 in legal:
        return 225
    trades = [index for index in legal if 199 <= index < 219]
    if trades:
        ratios = [4] * 5
        for vertex in np.flatnonzero(vertices[:, 0]):
            ports = vertices[vertex, 8:14].tolist()
            if ports[5]:
                ratios = [min(3, n) for n in ratios]
            for resource in range(5):
                if ports[resource]:
                    ratios[resource] = 2

        def after_trade(index):
            action = _CATALOG[index]
            next_hand = hand.copy()
            next_hand[action.give] -= ratios[action.give]
            next_hand[action.receive] += 1
            return _deficit(next_hand, target)

        trade = min(trades, key=lambda index: (after_trade(index), index))
        if after_trade(trade) < _deficit(hand, target):
            return trade
    roads = [index for index in legal if index < 72]
    if roads:
        return max(roads, key=lambda index: (road_score(index), -index))
    return 219 if 219 in legal else 220 if 220 in legal else legal[0]


POLICIES = MappingProxyType({"random": random_policy, "greedy": greedy_policy})

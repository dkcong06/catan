"""Deterministic win proofs from an observation, never a private GameState.

Only public ownership/awards and the observer's score are reconstructed. No
opponent cards, deck, RNG, dice or hypothetical draws exist in this view.
"""

from types import SimpleNamespace

import numpy as np

from catan.awards import update_army, update_roads
from catan.board import GEOMETRY
from catan.engine import _road_spot
from catan.state import Phase
from rl.actions import ACTION_DIM
from rl.encoding import FEATURE_SHAPES, FEATURE_SLICES


def _view(observation):
    def group(name):
        return observation[FEATURE_SLICES[name]].reshape(FEATURE_SHAPES[name])

    players = group("players")
    n = int(players[:, 0].sum())
    vertices, edges = group("vertices"), group("edges")
    owners = vertices[:, :5].argmax(1).astype(np.int8)
    owners[owners == 4] = -1
    road_owners = edges.argmax(1).astype(np.int8)
    road_owners[road_owners == 4] = -1
    return SimpleNamespace(
        board=SimpleNamespace(geometry=GEOMETRY),
        n_players=n,
        current_player=0,
        vertex_owner=owners,
        edge_owner=road_owners,
        vp=np.rint(players[:n, 1] * 10).astype(np.int16),
        road_lengths=np.rint(players[:n, 8] * 15).astype(np.int8),
        knights_played=np.rint(players[:n, 7] * 14).astype(np.int8),
        longest_road_owner=int(group("longest_road_owner").argmax()),
        largest_army_owner=int(group("largest_army_owner").argmax()),
    )


def _award_owner(view):
    for field in ("longest_road_owner", "largest_army_owner"):
        if getattr(view, field) == 4:
            setattr(view, field, -1)


def _road_wins(view, edge, remaining):
    """Explore at most two free roads; restore every local working array."""
    old_lengths, old_vp = view.road_lengths.copy(), view.vp.copy()
    old_holder = view.longest_road_owner
    view.edge_owner[edge] = 0
    try:
        update_roads(view, (0,))
        if view.vp[0] >= 10:
            return True
        if remaining > 1:
            return any(
                _road_wins(view, second, 1)
                for second in range(72)
                if _road_spot(view, second, False)
            )
        return False
    finally:
        view.edge_owner[edge] = -1
        view.road_lengths[:] = old_lengths
        view.vp[:] = old_vp
        view.longest_road_owner = old_holder


def winning_actions(decision):
    """All proved wins among legal builds/knights and bounded free-road plans.

    A marked Road Building card/first free road may require the next explicitly
    chosen free road. No trade chains, uncertain card purchases or chance actions
    are called wins. This is a conservative tactical subset, not an optimal solver.
    """
    result = np.zeros(ACTION_DIM, dtype=np.bool_)
    if decision.mask is None or decision.phase not in (
        Phase.MAIN,
        Phase.ROLL,
        Phase.ROAD_BUILDING,
    ):
        return result
    score = round(float(decision.observation[FEATURE_SLICES["players"]][1]) * 10)
    # A settlement can add one point and transfer Longest Road (+2).
    if score < 7:
        return result
    view = _view(decision.observation)
    _award_owner(view)
    legal = np.flatnonzero(decision.mask)
    if score >= 10:
        # Ending an already won turn needs no chance outcome.
        if decision.mask[219]:
            result[219] = True
        return result
    free_count = (
        round(
            float(decision.observation[FEATURE_SLICES["free_roads_remaining"]][0]) * 2
        )
        if decision.phase == Phase.ROAD_BUILDING
        else 1
    )
    for choice in legal:
        choice = int(choice)
        if choice < 72:
            result[choice] = score >= 8 and _road_wins(view, choice, free_count)
        elif 72 <= choice < 126:
            vertex = choice - 72
            view.vertex_owner[vertex] = 0
            vp, lengths, holder = (
                view.vp.copy(),
                view.road_lengths.copy(),
                view.longest_road_owner,
            )
            view.vp[0] += 1
            try:
                blocked = {
                    int(view.edge_owner[e]) for e in GEOMETRY.vertex_edges[vertex]
                }
                update_roads(view, blocked - {-1, 0})
                result[choice] = view.vp[0] >= 10
            finally:
                view.vertex_owner[vertex] = -1
                view.vp[:], view.road_lengths[:] = vp, lengths
                view.longest_road_owner = holder
        elif choice < 180:
            result[choice] = score + 1 >= 10
        elif choice == 226:
            vp, holder = view.vp.copy(), view.largest_army_owner
            view.knights_played[0] += 1
            update_army(view)
            result[choice] = view.vp[0] >= 10
            view.knights_played[0] -= 1
            view.vp[:] = vp
            view.largest_army_owner = holder
        elif choice == 227:
            if score < 8:
                continue
            players = decision.observation[FEATURE_SLICES["players"]]
            count = min(2, round(float(players[4]) * 15))
            result[choice] = any(
                _road_wins(view, edge, count)
                for edge in range(72)
                if _road_spot(view, edge, False)
            )
    return result

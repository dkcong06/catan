"""Exact edge trails and history-sensitive awards; recompute on graph changes."""


def longest_road_length(state, player: int) -> int:
    """Longest trail without repeated edges, stopped by opponent buildings.

    A blocked vertex may be an endpoint. Loops may revisit vertices, and own
    buildings do not block traversal. At most fifteen roads belong to one seat.
    """
    from catan.state import _integer

    player = _integer(player, "player")
    if not 0 <= player < state.n_players:
        raise ValueError("player is outside the game")
    owned = [e for e, owner in enumerate(state.edge_owner.tolist()) if owner == player]
    if not owned:
        return 0
    adjacency = [[] for _ in state.vertex_owner]
    for bit, edge in enumerate(owned):
        first, second = state.board.geometry.edge_vertices[edge]
        adjacency[first].append((1 << bit, second))
        adjacency[second].append((1 << bit, first))
    blocked = [owner not in (-1, player) for owner in state.vertex_owner.tolist()]
    memo = {}

    def walk(vertex, used):
        if used and blocked[vertex]:
            return 0
        key = vertex, used
        if key not in memo:
            memo[key] = max(
                (
                    1 + walk(other, used | bit)
                    for bit, other in adjacency[vertex]
                    if not used & bit
                ),
                default=0,
            )
        return memo[key]

    best = 0
    starts = sorted(
        (v for v, edges in enumerate(adjacency) if edges),
        key=lambda v: len(adjacency[v]),
    )
    for vertex in starts:
        best = max(best, walk(vertex, 0))
        if best == len(owned):
            break
    return best


def _update_award(state, field, counts, minimum):
    old = getattr(state, field)
    maximum = max(counts)
    new = -1
    if maximum >= minimum:
        if old != -1 and counts[old] == maximum:
            new = old  # Incumbent retains a tie.
        else:
            leaders = [p for p, count in enumerate(counts) if count == maximum]
            if len(leaders) == 1:
                new = leaders[0]
    if new != old:
        if old != -1:
            state.vp[old] -= 2
        if new != -1:
            state.vp[new] += 2
        setattr(state, field, new)


def update_roads(state, players):
    for player in players:
        state.road_lengths[player] = longest_road_length(state, player)
    _update_award(state, "longest_road_owner", state.road_lengths.tolist(), 5)


def update_army(state):
    _update_award(state, "largest_army_owner", state.knights_played.tolist(), 3)

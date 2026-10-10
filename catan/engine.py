"""Minimal Catan rules with independent and exclusive-owner transitions.

Callers supply valid GameState objects. Legal choices are lazy; full invariant
validation is an explicit debug gate, outside the training hot path.
"""

from collections.abc import Iterator

import numpy as np

from catan.actions import (
    Action,
    BankTradeAction,
    CityAction,
    DiscardAction,
    EndTurnAction,
    RobberMoveAction,
    RoadAction,
    RollAction,
    SettlementAction,
    StealAction,
)
from catan.board import Resource
from catan.state import GameState, Phase


def _validate_seven(state: GameState) -> None:
    if state.phase != Phase.ROLL:
        raise ValueError("a seven can only be resolved in ROLL phase")
    if any(state.discard_pending):
        raise ValueError("cannot resolve a seven with existing pending discards")


def _resolve_seven(state: GameState) -> GameState:
    # Reuse the pending array as reduction output, avoiding a totals array and
    # np.where temporaries. Valid hands contain at most 95 cards per seat.
    np.sum(state.hands, axis=1, out=state.discard_pending)
    pending = False
    for player, total in enumerate(state.discard_pending):
        count = int(total) // 2 if total > 7 else 0
        state.discard_pending[player] = count
        pending = pending or count > 0
    state.last_roll = 7
    state.phase = Phase.DISCARD if pending else Phase.ROBBER_MOVE
    return state


def resolve_seven(state: GameState) -> GameState:
    """Resolve an already-known seven on an independent successor state.

    Every player with more than seven resource cards owes half, rounded down.
    Snapshot these counts once so a large remaining hand is not discarded twice.
    Neither the source state nor its RNG is advanced.
    """
    _validate_seven(state)
    return _resolve_seven(state.copy())


def resolve_seven_inplace(state: GameState) -> GameState:
    """Resolve an already-known seven on a valid, exclusively owned state.

    Return the same object, retaining its arrays and RNG without cloning them.
    Rule rejection occurs before mutation. Callers must not share mutable state
    with another worker, saved snapshot, or concurrent reader.
    """
    _validate_seven(state)
    return _resolve_seven(state)


def _validate_discard(state: GameState, action: DiscardAction) -> np.ndarray:
    if not isinstance(action, DiscardAction):
        raise TypeError("this engine increment accepts DiscardAction only")
    if state.phase != Phase.DISCARD:
        raise ValueError("discard actions require DISCARD phase")
    if action.player != state.discard_player:
        raise ValueError("this player is not the next pending discard player")
    if sum(action.resources) != int(state.discard_pending[action.player]):
        raise ValueError("choose exactly the required number of cards")
    if any(
        chosen > held
        for chosen, held in zip(action.resources, state.hands[action.player])
    ):
        raise ValueError("cannot discard more of a resource than the player holds")
    # Convert once for both resource updates rather than coercing the tuple twice.
    # Allocate before mutation; no mutable scratch buffer is shared between calls.
    return np.asarray(action.resources, dtype=np.int16)


def _apply_discard(state: GameState, player: int, counts: np.ndarray) -> GameState:
    state.hands[player] -= counts
    state.bank += counts
    state.discard_pending[player] = 0
    if not any(state.discard_pending):
        state.phase = Phase.ROBBER_MOVE
    return state


def apply_action(state: GameState, action: Action) -> GameState:
    """Validate before copying, then update an independent successor and RNG."""
    payload = _validate_action(state, action)
    return _mutate(state.copy(), action, payload)


def apply_action_inplace(state: GameState, action: Action) -> GameState:
    """Validate and update valid state exclusively owned by this caller.

    Return the same object, retaining every array/RNG identity. Rule rejection
    precedes mutation and random draws. Saved aliases observe accepted updates;
    use apply_action when independent branches or snapshots are needed.
    """
    payload = _validate_action(state, action)
    return _mutate(state, action, payload)


# Immutable action objects are shared across legal-choice queries and states.
_SETTLEMENTS = tuple(SettlementAction(v) for v in range(54))
_ROADS = tuple(RoadAction(e) for e in range(72))
_CITIES = tuple(CityAction(v) for v in range(54))
_ROBBER_MOVES = tuple(RobberMoveAction(t) for t in range(19))
_STEALS = tuple(StealAction(p) for p in range(4))
_TRADES = tuple(BankTradeAction(g, r) for g in range(5) for r in range(5) if g != r)
_ROLL = RollAction()
_END_TURN = EndTurnAction()
_ROAD_COST = ((0, 1), (3, 1))
_SETTLEMENT_COST = ((0, 1), (1, 1), (2, 1), (3, 1))
_CITY_COST = ((2, 2), (4, 3))


def _affordable(state: GameState, cost: tuple) -> bool:
    hand = state.hands[state.current_player]
    return all(hand[resource] >= count for resource, count in cost)


def _starting_resources(state: GameState, vertex: int) -> tuple[int, ...]:
    if state.setup_index < state.n_players:
        return (0, 0, 0, 0, 0)
    counts = [0] * 5
    for tile in state.board.geometry.vertex_tiles[vertex]:
        resource = state.board.resources[tile]
        if resource != Resource.DESERT:
            counts[resource] += 1
    return tuple(counts)


def _settlement_spot(state: GameState, vertex: int, setup: bool) -> bool:
    geometry = state.board.geometry
    if state.vertex_owner[vertex] != -1 or any(
        state.vertex_owner[v] != -1 for v in geometry.vertex_neighbors[vertex]
    ):
        return False
    if setup:
        return all(
            held >= count
            for held, count in zip(state.bank, _starting_resources(state, vertex))
        )
    return any(
        state.edge_owner[e] == state.current_player
        for e in geometry.vertex_edges[vertex]
    )


def _road_spot(state: GameState, edge: int, setup: bool) -> bool:
    if state.edge_owner[edge] != -1:
        return False
    geometry = state.board.geometry
    vertices = geometry.edge_vertices[edge]
    if setup:
        return state.setup_vertex in vertices
    player = state.current_player
    for vertex in vertices:
        owner = state.vertex_owner[vertex]
        if owner == player or (
            owner == -1
            and any(
                state.edge_owner[e] == player for e in geometry.vertex_edges[vertex]
            )
        ):
            return True
    return False  # An opponent's building interrupts road connectivity.


def _trade_ratios(state: GameState) -> list[int]:
    ratios = [4] * 5
    for port in state.port_access(state.current_player):
        if port.resource is None:
            ratios = [min(ratio, 3) for ratio in ratios]
        else:
            ratios[port.resource] = 2
    return ratios


def _victims(state: GameState) -> tuple[int, ...]:
    owners = {
        int(state.vertex_owner[v])
        for v in state.board.geometry.tile_vertices[state.robber_tile]
        if state.vertex_owner[v] not in (-1, state.current_player)
    }
    return tuple(p for p in sorted(owners) if any(state.hands[p]))


def _validate_action(state: GameState, action: Action):
    if isinstance(action, DiscardAction):
        return _validate_discard(state, action)
    player = state.current_player
    if isinstance(action, SettlementAction):
        setup = state.phase == Phase.SETUP_SETTLEMENT
        if (
            state.phase not in (Phase.SETUP_SETTLEMENT, Phase.MAIN)
            or state.pieces_left[player, 0] == 0
            or not _settlement_spot(state, action.vertex, setup)
            or (not setup and not _affordable(state, _SETTLEMENT_COST))
        ):
            raise ValueError("illegal settlement placement")
    elif isinstance(action, RoadAction):
        setup = state.phase == Phase.SETUP_ROAD
        if (
            state.phase not in (Phase.SETUP_ROAD, Phase.MAIN)
            or state.pieces_left[player, 2] == 0
            or not _road_spot(state, action.edge, setup)
            or (not setup and not _affordable(state, _ROAD_COST))
        ):
            raise ValueError("illegal road placement")
    elif isinstance(action, CityAction):
        if (
            state.phase != Phase.MAIN
            or state.pieces_left[player, 1] == 0
            or state.vertex_owner[action.vertex] != player
            or state.vertex_building[action.vertex] != 1
            or not _affordable(state, _CITY_COST)
        ):
            raise ValueError("illegal city upgrade")
    elif isinstance(action, RollAction):
        _validate_seven(state)
    elif isinstance(action, RobberMoveAction):
        if state.phase != Phase.ROBBER_MOVE or action.tile == state.robber_tile:
            raise ValueError("robber must move to a different tile in ROBBER_MOVE")
    elif isinstance(action, StealAction):
        if state.phase != Phase.ROBBER_STEAL or action.victim not in _victims(state):
            raise ValueError("victim must have cards and a building on the robber tile")
    elif isinstance(action, BankTradeAction):
        if state.phase != Phase.MAIN:
            raise ValueError("bank trades require MAIN phase")
        ratio = _trade_ratios(state)[action.give]
        if state.hands[player, action.give] < ratio or state.bank[action.receive] == 0:
            raise ValueError("bank trade lacks offered cards or bank supply")
        return ratio
    elif isinstance(action, EndTurnAction):
        if state.phase != Phase.MAIN:
            raise ValueError("ending a turn requires MAIN phase")
    else:
        raise TypeError(
            "unsupported action type; expected a Catan action such as DiscardAction"
        )
    return None


def _pay(state: GameState, cost: tuple) -> None:
    for resource, count in cost:
        state.hands[state.current_player, resource] -= count
        state.bank[resource] += count


def _produce(state: GameState, total: int) -> None:
    owed = np.zeros_like(state.hands)
    for tile, number in enumerate(state.board.numbers):
        if number != total or tile == state.robber_tile:
            continue
        resource = state.board.resources[tile]
        for vertex in state.board.geometry.tile_vertices[tile]:
            owner = state.vertex_owner[vertex]
            if owner != -1:
                owed[owner, resource] += state.vertex_building[vertex]
    for resource in range(5):
        payments = owed[:, resource]
        required = int(payments.sum())
        if required == 0:
            continue
        available = int(state.bank[resource])
        if required <= available:
            state.hands[:, resource] += payments
            state.bank[resource] -= required
        else:
            recipients = [p for p in range(state.n_players) if payments[p] > 0]
            if len(recipients) == 1:
                state.hands[recipients[0], resource] += available
                state.bank[resource] = 0
            # Multiple claimants receive none of this scarce resource.


def _mutate(state: GameState, action: Action, payload) -> GameState:
    player = state.current_player
    if isinstance(action, DiscardAction):
        _apply_discard(state, player=action.player, counts=payload)
    elif isinstance(action, SettlementAction):
        setup = state.phase == Phase.SETUP_SETTLEMENT
        if setup:
            grants = _starting_resources(state, action.vertex)
            for resource, count in enumerate(grants):
                state.hands[player, resource] += count
                state.bank[resource] -= count
            state.setup_vertex = action.vertex
            state.phase = Phase.SETUP_ROAD
        else:
            _pay(state, _SETTLEMENT_COST)
        state.vertex_owner[action.vertex] = player
        state.vertex_building[action.vertex] = 1
        state.pieces_left[player, 0] -= 1
        state.vp[player] += 1
    elif isinstance(action, RoadAction):
        if state.phase == Phase.SETUP_ROAD:
            state.setup_index += 1
            state.setup_vertex = None
            if state.setup_index == len(state.setup_order):
                state.current_player = 0
                state.phase = Phase.ROLL
            else:
                state.current_player = state.setup_order[state.setup_index]
                state.phase = Phase.SETUP_SETTLEMENT
        else:
            _pay(state, _ROAD_COST)
        state.edge_owner[action.edge] = player
        state.pieces_left[player, 2] -= 1
    elif isinstance(action, CityAction):
        _pay(state, _CITY_COST)
        state.vertex_building[action.vertex] = 2
        state.pieces_left[player, 1] -= 1
        state.pieces_left[player, 0] += 1
        state.vp[player] += 1
    elif isinstance(action, RollAction):
        total = state.rng.randint(1, 6) + state.rng.randint(1, 6)
        state.last_roll = total
        if total == 7:
            _resolve_seven(state)
        else:
            _produce(state, total)
            state.phase = Phase.MAIN
    elif isinstance(action, RobberMoveAction):
        state.robber_tile = action.tile
        state.phase = Phase.ROBBER_STEAL if _victims(state) else Phase.MAIN
    elif isinstance(action, StealAction):
        draw = state.rng.randrange(int(state.hands[action.victim].sum()))
        for resource, count in enumerate(state.hands[action.victim]):
            if draw < count:
                state.hands[action.victim, resource] -= 1
                state.hands[player, resource] += 1
                break
            draw -= int(count)
        state.phase = Phase.MAIN
    elif isinstance(action, BankTradeAction):
        state.hands[player, action.give] -= payload
        state.bank[action.give] += payload
        state.hands[player, action.receive] += 1
        state.bank[action.receive] -= 1
    elif isinstance(action, EndTurnAction):
        if state.vp[player] < 10:
            state.current_player = (player + 1) % state.n_players
            state.last_roll = None
            state.phase = Phase.ROLL
    if state.phase == Phase.MAIN and state.vp[state.current_player] >= 10:
        state.phase = Phase.GAME_OVER
    return state


def _discard_choices(state: GameState) -> Iterator[DiscardAction]:
    player = state.discard_player
    held = tuple(int(count) for count in state.hands[player])
    required = int(state.discard_pending[player])
    suffix = [sum(held[r:]) for r in range(6)]

    def choices(resource, remaining, prefix):
        if resource == 4:
            if 0 <= remaining <= held[4]:
                yield DiscardAction(player, prefix + (remaining,))
            return
        minimum = max(0, remaining - suffix[resource + 1])
        maximum = min(held[resource], remaining)
        for count in range(minimum, maximum + 1):
            yield from choices(resource + 1, remaining - count, prefix + (count,))

    yield from choices(0, required, ())


def legal_actions(state: GameState) -> Iterator[Action]:
    """Yield legal choices in stable order, without copying state or using RNG.

    Consume choices before mutating the state. Discard choices are generated only
    on demand; submit manual counts directly rather than materializing that space.
    Training masks/encoding are a separate later layer.
    """
    phase = state.phase
    player = state.current_player
    if phase == Phase.SETUP_SETTLEMENT:
        if state.pieces_left[player, 0] > 0:
            for action in _SETTLEMENTS:
                if _settlement_spot(state, action.vertex, True):
                    yield action
    elif phase == Phase.SETUP_ROAD:
        if state.pieces_left[player, 2] > 0:
            for edge in state.board.geometry.vertex_edges[state.setup_vertex]:
                if _road_spot(state, edge, True):
                    yield _ROADS[edge]
    elif phase == Phase.ROLL:
        yield _ROLL
    elif phase == Phase.DISCARD:
        yield from _discard_choices(state)
    elif phase == Phase.ROBBER_MOVE:
        for action in _ROBBER_MOVES:
            if action.tile != state.robber_tile:
                yield action
    elif phase == Phase.ROBBER_STEAL:
        for victim in _victims(state):
            yield _STEALS[victim]
    elif phase == Phase.MAIN:
        if state.pieces_left[player, 2] > 0 and _affordable(state, _ROAD_COST):
            for action in _ROADS:
                if _road_spot(state, action.edge, False):
                    yield action
        if state.pieces_left[player, 0] > 0 and _affordable(state, _SETTLEMENT_COST):
            for action in _SETTLEMENTS:
                if _settlement_spot(state, action.vertex, False):
                    yield action
        if state.pieces_left[player, 1] > 0 and _affordable(state, _CITY_COST):
            for action in _CITIES:
                if (
                    state.vertex_owner[action.vertex] == player
                    and state.vertex_building[action.vertex] == 1
                ):
                    yield action
        ratios = _trade_ratios(state)
        for action in _TRADES:
            if (
                state.hands[player, action.give] >= ratios[action.give]
                and state.bank[action.receive] > 0
            ):
                yield action
        yield _END_TURN
    elif phase != Phase.GAME_OVER:
        raise ValueError("unrecognized phase")


def is_terminal(state: GameState) -> bool:
    return state.phase == Phase.GAME_OVER


def winner(state: GameState) -> int | None:
    """Return the winning seat at GAME_OVER; no duplicate winner field is stored."""
    return state.current_player if is_terminal(state) else None

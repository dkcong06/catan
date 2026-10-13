"""Opt-in engine invariants and deterministic bounded random-game validation.

No checks are installed in engine hot paths; callers explicitly request this gate.
"""

from dataclasses import dataclass
from hashlib import sha256
from random import Random

from catan.engine import apply_action, apply_action_inplace, is_terminal, winner
from catan.players.random_agent import choose_action
from catan.state import (
    GameState,
    Phase,
    _integer,
    _player_count,
    new_game,
    _ARRAY_FIELDS,
    _SCALAR_FIELDS,
)
from catan.awards import longest_road_length


def assert_invariants(state: GameState) -> None:
    """Check storage, bookkeeping, phase consistency, and rooted road components.

    Road traversal ignores opponent buildings: a later settlement can interrupt
    extension without making existing roads illegal. Action validation enforces
    that interruption at construction time. Both independent setup roots count.
    """
    state.validate()
    setup = state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)
    if setup:
        if state.last_roll is not None:
            raise ValueError("setup cannot have a dice roll")
    elif state.setup_index != len(state.setup_order):
        raise ValueError("normal play requires completed setup")
    if state.phase == Phase.ROLL and state.last_roll is not None:
        raise ValueError("ROLL requires a cleared last_roll")
    if state.phase == Phase.DISCARD and state.last_roll != 7:
        raise ValueError("discards require a seven")
    if state.phase in (Phase.ROBBER_MOVE, Phase.ROBBER_STEAL):
        if state.resume_phase is None and state.last_roll != 7:
            raise ValueError("robber decisions require a seven or knight")
    if state.resume_phase == Phase.ROLL and state.last_roll is not None:
        raise ValueError("pre-roll card effects require a cleared last_roll")
    if (
        state.phase == Phase.MAIN or state.resume_phase == Phase.MAIN
    ) and state.last_roll is None:
        raise ValueError("main play requires a dice roll")
    scores = state.vp.tolist()
    if state.phase == Phase.GAME_OVER:
        if scores[state.current_player] < 10:
            raise ValueError("GAME_OVER requires a winning current player")
    elif scores[state.current_player] >= 10:
        raise ValueError("winning VP requires GAME_OVER on the player's turn")
    for player in range(state.n_players):
        if state.road_lengths[player] != longest_road_length(state, player):
            raise ValueError("cached road length disagrees with the road graph")
    for field, counts, minimum in (
        ("longest_road_owner", state.road_lengths.tolist(), 5),
        ("largest_army_owner", state.knights_played.tolist(), 3),
    ):
        maximum = max(counts)
        leaders = [p for p, count in enumerate(counts) if count == maximum]
        if (
            maximum >= minimum
            and len(leaders) == 1
            and getattr(state, field) != leaders[0]
        ):
            raise ValueError("unique award leader must hold the award")

    geometry = state.board.geometry
    roots = [[] for _ in range(state.n_players)]
    roads = [[] for _ in range(state.n_players)]
    for vertex, owner in enumerate(state.vertex_owner.tolist()):
        if owner != -1:
            roots[owner].append(vertex)
    edge_owner = state.edge_owner.tolist()
    for edge, owner in enumerate(edge_owner):
        if owner != -1:
            roads[owner].append(edge)
    for player, stack in enumerate(roots):
        reached = set(stack)
        while stack:
            vertex = stack.pop()
            for edge in geometry.vertex_edges[vertex]:
                if edge_owner[edge] != player:
                    continue
                first, second = geometry.edge_vertices[edge]
                neighbor = second if first == vertex else first
                if neighbor not in reached:
                    reached.add(neighbor)
                    stack.append(neighbor)
        if any(
            geometry.edge_vertices[edge][0] not in reached for edge in roads[player]
        ):
            raise ValueError(f"player {player} has an unrooted road component")


@dataclass(frozen=True, slots=True)
class ValidatedGame:
    seed: int
    players: int
    actions: int
    winner: int | None
    scores: tuple[int, ...]
    phase_counts: tuple[int, ...]
    fingerprint: str

    @property
    def capped(self) -> bool:
        return self.winner is None


class GameValidationError(RuntimeError):
    """Replay coordinates for a rule/check failure; no full trajectory is retained."""

    def __init__(self, seed, players, step, phase, action, error):
        self.context = {
            "seed": seed,
            "players": players,
            "step": step,
            "phase": phase,
            "action": repr(action),
            "error": str(error),
        }
        super().__init__(f"game validation failed: {self.context}")


def _fingerprint(state: GameState, policy: Random) -> str:
    digest = sha256()
    digest.update(
        repr(
            (
                state.board.resources,
                state.board.numbers,
                state.board.ports,
                tuple(getattr(state, name) for name in _SCALAR_FIELDS),
                state.rng.getstate(),
                policy.getstate(),
            )
        ).encode()
    )
    for name in _ARRAY_FIELDS:
        digest.update(getattr(state, name).tobytes())
    return digest.hexdigest()


def validate_game(
    seed: int,
    n_players: int = 4,
    *,
    max_actions: int = 10_000,
    inplace: bool = True,
) -> ValidatedGame:
    """Check every state of one seeded episode, reporting a real win or a cap.

    Policy and game RNGs are independent. A cap is not an engine error or win.
    Checks remain enabled for setup, every decision, and the final state.
    """
    seed = _integer(seed, "seed")
    max_actions = _integer(max_actions, "max_actions")
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    n_players = _player_count(n_players)
    index = 0
    action = None
    phase = "INITIALIZE"
    try:
        state = new_game(seed, n_players)
        policy = Random(seed)
        step = apply_action_inplace if inplace else apply_action
        counts = [0] * len(Phase)
        phase = state.phase.name
        assert_invariants(state)
        counts[state.phase.value - 1] += 1
        for index in range(1, max_actions + 1):
            phase = state.phase.name
            action = choose_action(state, policy)
            state = step(state, action)
            assert_invariants(state)
            counts[state.phase.value - 1] += 1
            if is_terminal(state):
                break
        return ValidatedGame(
            seed,
            state.n_players,
            index,
            winner(state),
            tuple(map(int, state.vp)),
            tuple(counts),
            _fingerprint(state, policy),
        )
    except Exception as error:
        raise GameValidationError(
            seed, n_players, index, phase, action, error
        ) from error

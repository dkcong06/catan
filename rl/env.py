"""Decision-by-decision agent interface, separate from game rules and training.

Each environment exclusively owns its mutable state. Policies receive independent
observations/masks. A complete manual discard is one accepted engine action.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from catan.actions import DiscardAction
from catan.engine import apply_action_inplace, is_terminal, winner
from catan.state import Phase, _integer, new_game
from rl.actions import _discard_count_mask, action_mask, indexed_action
from rl.encoding import FlatEncoder, decision_player

ENV_SCHEMA_VERSION = 1
REWARD_SCHEME = "terminal_win_v1"


@dataclass(frozen=True, slots=True, eq=False)
class Decision:
    player: int
    phase: Phase
    observation: np.ndarray
    mask: np.ndarray | None
    discard_hand: tuple[int, ...] | None = None
    discard_required: int = 0

    def discard_mask(self, prefix=()):
        """Conditional choices from this snapshot, without state/RNG access."""
        if self.discard_hand is None:
            raise ValueError("this decision is not a discard")
        return _discard_count_mask(self.discard_hand, self.discard_required, prefix)


class Transition(NamedTuple):
    player: int
    next_player: int | None
    rewards: tuple[float, ...]
    terminated: bool
    truncated: bool
    winner: int | None
    actions: int


class CatanEnv:
    """One atomic action per step; no implicit opponent moves or auto-choices.

    reset returns the first Decision. step accepts a catalog index (or all five
    discard counts), returning lightweight status; request decision() when ready
    for the next policy call. player is mandatory to reject wrong-seat choices.
    External truncation preserves the nonterminal state and its hidden-info rules.
    observe(player) remains available for every seat after a cutoff/real win.
    """

    __slots__ = ("_state", "_encoder", "_actions", "_max_actions", "_zeros")

    def __init__(self, *, max_actions=5000):
        max_actions = _integer(max_actions, "max_actions")
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self._max_actions = max_actions
        self._state = self._encoder = None
        self._actions = 0
        self._zeros = ()

    def _ready(self):
        if self._state is None:
            raise ValueError("reset the environment first")
        return self._state

    @property
    def actions(self):
        return self._actions

    @property
    def terminated(self):
        return is_terminal(self._ready())

    @property
    def truncated(self):
        state = self._ready()
        return self._actions >= self._max_actions and not is_terminal(state)

    def reset(self, seed=None, *, n_players=4):
        # A rejected reset preserves the previous episode.
        state = new_game(seed, n_players)
        encoder = FlatEncoder(state.board)
        self._state, self._encoder = state, encoder
        self._actions = 0
        self._zeros = (0.0,) * state.n_players
        return self.decision()

    def observe(self, player=None, *, out=None):
        """Independent output by default; optional buffers follow FlatEncoder."""
        state = self._ready()
        return self._encoder.encode(state, player, out=out)

    def snapshot(self):
        """Explicit full copy for debugging/replays, outside the policy interface."""
        return self._ready().copy()

    def decision(self):
        state = self._ready()
        if is_terminal(state) or self._actions >= self._max_actions:
            return None
        player = decision_player(state)
        observation = self._encoder.encode(state)
        if state.phase == Phase.DISCARD:
            return Decision(
                player,
                state.phase,
                observation,
                None,
                tuple(state.hands[player].tolist()),
                int(state.discard_pending[player]),
            )
        return Decision(player, state.phase, observation, action_mask(state))

    def step(self, action, *, player):
        state = self._ready()
        if is_terminal(state) or self._actions >= self._max_actions:
            raise ValueError("episode is closed; reset before stepping")
        actor = decision_player(state)
        if _integer(player, "player") != actor:
            raise ValueError("player is not the current decision maker")
        choice = (
            DiscardAction(actor, action)
            if state.phase == Phase.DISCARD
            else indexed_action(state, action)
        )
        apply_action_inplace(state, choice)
        self._actions += 1
        won = winner(state)
        terminated = won is not None
        truncated = not terminated and self._actions >= self._max_actions
        rewards = (
            tuple(float(p == won) for p in range(state.n_players))
            if terminated
            else self._zeros
        )
        return Transition(
            actor,
            decision_player(state),
            rewards,
            terminated,
            truncated,
            won,
            self._actions,
        )

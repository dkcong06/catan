"""Small random baseline for engine smoke tests; no learning dependencies."""

from random import Random

from catan.actions import Action, DiscardAction
from catan.engine import apply_action, apply_action_inplace, is_terminal, legal_actions
from catan.state import GameState, Phase, new_game


def choose_action(state: GameState, rng: Random) -> Action:
    """Choose with a policy RNG, leaving the game RNG untouched.

    Discards sample physical cards directly, avoiding enumeration of the count
    space. Only this explicit agent selects cards; the engine never does so.
    """
    if state.phase == Phase.DISCARD:
        player = state.discard_player
        cards = [r for r, count in enumerate(state.hands[player]) for _ in range(count)]
        selected = rng.sample(cards, int(state.discard_pending[player]))
        return DiscardAction(player, tuple(selected.count(r) for r in range(5)))
    choices = tuple(legal_actions(state))
    if not choices:
        raise ValueError("no legal action; the game may already be over")
    return rng.choice(choices)


def play_random_game(
    seed: int,
    n_players: int = 4,
    *,
    inplace: bool = True,
    max_actions: int = 100_000,
    validate: bool = False,
) -> GameState:
    """Play a bounded baseline game, returning real GAME_OVER or raising at cap."""
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    state = new_game(seed, n_players)
    policy_rng = Random(seed)
    step = apply_action_inplace if inplace else apply_action
    for _ in range(max_actions):
        state = step(state, choose_action(state, policy_rng))
        if validate:
            state.validate()
        if is_terminal(state):
            return state
    raise RuntimeError(f"game {seed} did not finish within {max_actions} actions")

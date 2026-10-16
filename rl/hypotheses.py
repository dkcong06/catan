"""Optional NumPy experiment views: public resource bounds and own VP progress.

No engine fields or RNG changes. Hidden hand composition is read only for the
observer's own private view, or by validate_knowledge in explicit diagnostic runs.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from catan.actions import (
    DiscardAction,
    EndTurnAction,
    KnightAction,
    MonopolyAction,
    RoadAction,
    RoadBuildingAction,
    RobberMoveAction,
    RollAction,
    StealAction,
)
from catan.engine import _produce
from catan.state import Phase, _integer
from rl.actions import indexed_action
from rl.encoding import OBS_DIM, decision_player
from rl.env import CatanEnv, REWARD_SCHEME

KNOWLEDGE_DIM = 30
EXPERIMENT_OBS_DIM = OBS_DIM + KNOWLEDGE_DIM
EXPERIMENT_OBS_SCHEMA = 3
PROGRESS_REWARD = "terminal_win_net_vp_0.1_v1"


def progress(vp):
    return tuple(min(1.0, max(0.0, (int(score) - 2) / 8)) for score in vp)


@dataclass(frozen=True, slots=True)
class ResourceEvent:
    """One observer's observable event; no other player's exact hand."""

    flows: tuple
    totals: tuple
    bank: tuple
    own_hand: tuple
    steal: tuple | None = None
    known_resource: int | None = None


class ResourceBounds:
    """Conservative intervals for one observer, with bounded streaming state."""

    __slots__ = ("observer", "bounds", "totals", "bank", "exact")

    def __init__(self, players, observer):
        if not 0 <= observer < players:
            raise ValueError("observer must be a seat in this game")
        self.observer = observer
        self.bounds = np.zeros((players, 5, 2), dtype=np.int16)
        self.totals = (0,) * players
        self.bank = (19,) * 5
        self.exact = True

    def update(self, event):
        b = self.bounds
        for player, flow in enumerate(event.flows):
            if any(flow):
                b[player] += np.asarray(flow, dtype=np.int16)[:, None]
                np.maximum(b[player, :, 0], 0, out=b[player, :, 0])
        if event.steal is not None:
            thief, victim = event.steal
            resource = event.known_resource
            if resource is None:
                self.exact = False
                np.maximum(b[victim, :, 0] - 1, 0, out=b[victim, :, 0])
                np.minimum(b[thief, :, 1] + 1, 19, out=b[thief, :, 1])
            else:
                b[victim, resource] -= 1
                b[victim, resource, 0] = max(0, b[victim, resource, 0])
                b[thief, resource] += 1
        b[self.observer] = np.asarray(event.own_hand, dtype=np.int16)[:, None]
        self.totals, self.bank = event.totals, event.bank

    def features(self):
        # Tighten only when this observer acts, rather than for every opponent move.
        b = self.bounds.copy()
        lo, hi = b[:, :, 0], b[:, :, 1]
        totals = np.asarray(self.totals, dtype=np.int16)[:, None]
        aggregate = (19 - np.asarray(self.bank, dtype=np.int16))[None, :]
        if not self.exact:
            for _ in range(2):
                np.maximum(lo, totals - (hi.sum(axis=1)[:, None] - hi), out=lo)
                np.minimum(hi, totals - (lo.sum(axis=1)[:, None] - lo), out=hi)
                np.maximum(lo, aggregate - (hi.sum(axis=0)[None, :] - hi), out=lo)
                np.minimum(hi, aggregate - (lo.sum(axis=0)[None, :] - lo), out=hi)
        if (lo < 0).any() or (lo > hi).any() or (hi > 19).any():
            raise RuntimeError("inconsistent public resource history")
        players = len(b)
        order = [(self.observer + i) % players for i in range(1, players)]
        result = np.zeros((3, 5, 2), dtype=np.float32)
        result[: players - 1] = b[order] / np.float32(19)
        return result.reshape(-1)


def public_flows(state, choice, actor, old_bank, old_totals, totals):
    """Use only public board, bank, hand sizes and announced choice."""
    players = state.n_players
    flows = [[0] * 5 for _ in range(players)]
    if isinstance(choice, StealAction):
        return tuple(map(tuple, flows))
    if isinstance(choice, RollAction):
        if state.last_roll != 7:
            public = SimpleNamespace(
                n_players=players,
                board=state.board,
                robber_tile=state.robber_tile,
                vertex_owner=state.vertex_owner,
                vertex_building=state.vertex_building,
                bank=np.asarray(old_bank, dtype=np.int16).copy(),
                hands=np.zeros((players, 5), dtype=np.int16),
            )
            # Share the engine's production/scarcity rule on a public-only fixture.
            _produce(public, state.last_roll)
            flows = public.hands.tolist()
    elif isinstance(choice, MonopolyAction):
        for player in range(players):
            flows[player][choice.resource] = totals[player] - old_totals[player]
    else:
        flows[actor] = [
            int(before) - int(after) for before, after in zip(old_bank, state.bank)
        ]
    return tuple(map(tuple, flows))


class HypothesisEnv(CatanEnv):
    """Experimental wrapper, preserving CatanEnv's default terminal-win contract."""

    __slots__ = ("knowledge", "point_progress", "observer", "_tracker")
    observation_dim = EXPERIMENT_OBS_DIM
    observation_schema = EXPERIMENT_OBS_SCHEMA

    def __init__(
        self, *, max_actions=5000, knowledge=False, point_progress=False, observer=0
    ):
        super().__init__(max_actions=max_actions)
        self.knowledge, self.point_progress, self.observer = (
            knowledge,
            point_progress,
            observer,
        )
        self._tracker = None

    @property
    def reward_scheme(self):
        return PROGRESS_REWARD if self.point_progress else REWARD_SCHEME

    def reset(self, seed=None, *, n_players=4):
        self._tracker = None
        super().reset(seed, n_players=n_players)
        if self.knowledge:
            self._tracker = ResourceBounds(n_players, self.observer)
        return self.decision()

    def _extend(self, base, player):
        result = np.zeros(EXPERIMENT_OBS_DIM, dtype=np.float32)
        result[:OBS_DIM] = base
        if self._tracker is not None and player == self.observer:
            result[OBS_DIM:] = self._tracker.features()
        return result

    def decision(self):
        request = super().decision()
        return (
            None
            if request is None
            else replace(
                request, observation=self._extend(request.observation, request.player)
            )
        )

    def observe(self, player=None, *, out=None):
        if out is not None:
            raise ValueError("experimental observations require owned storage")
        state = self._ready()
        chosen = player if player is not None else decision_player(state)
        chosen = state.current_player if chosen is None else chosen
        return self._extend(super().observe(player), chosen)

    def step(self, action, *, player):
        state = self._ready()
        before_points = tuple(state.vp.tolist()) if self.point_progress else None
        track = False
        if self._tracker is not None:
            choice = (
                DiscardAction(player, action)
                if state.phase == Phase.DISCARD
                else indexed_action(state, action)
            )
            track = not isinstance(
                choice,
                (EndTurnAction, KnightAction, RoadBuildingAction, RobberMoveAction),
            )
            if isinstance(choice, RoadAction) and state.phase in (
                Phase.SETUP_ROAD,
                Phase.ROAD_BUILDING,
            ):
                track = False
            old_bank = self._tracker.bank
            old_totals = self._tracker.totals
            own_before = (
                self._tracker.bounds[self.observer, :, 0]
                if isinstance(choice, StealAction)
                else None
            )
        result = super().step(action, player=player)
        if track:
            bank = tuple(state.bank.tolist())
            track = bank != old_bank or isinstance(
                choice, (StealAction, MonopolyAction)
            )
        if track:
            totals = tuple(state.hands.sum(axis=1).tolist())
            own_after = tuple(state.hands[self.observer].tolist())
            steal = None
            resource = None
            if isinstance(choice, StealAction) and totals[player] > old_totals[player]:
                steal = (player, choice.victim)
                if self.observer in steal:
                    resource = next(
                        r for r in range(5) if own_before[r] != own_after[r]
                    )
            self._tracker.update(
                ResourceEvent(
                    public_flows(state, choice, player, old_bank, old_totals, totals),
                    totals,
                    bank,
                    own_after,
                    steal,
                    resource,
                )
            )
        after_points = tuple(state.vp.tolist()) if before_points is not None else None
        if before_points is not None and before_points != after_points:
            before_progress, after = progress(before_points), progress(after_points)
            result = result._replace(
                rewards=tuple(
                    reward + 0.1 * (new - old)
                    for reward, new, old in zip(result.rewards, after, before_progress)
                )
            )
        return result

    def validate_knowledge(self):
        """Privileged test oracle, never called during normal collection."""
        if self._tracker is not None:
            bounds = self._tracker.bounds
            actual = self._state.hands
            if not ((bounds[:, :, 0] <= actual) & (actual <= bounds[:, :, 1])).all():
                raise AssertionError("true hand excluded from public resource bounds")


@dataclass(frozen=True, slots=True)
class HypothesisEnvFactory:
    knowledge: bool = False
    point_progress: bool = False
    observer: int = 0
    observation_dim = EXPERIMENT_OBS_DIM
    observation_schema = EXPERIMENT_OBS_SCHEMA

    def __post_init__(self):
        if type(self.knowledge) is not bool or type(self.point_progress) is not bool:
            raise ValueError("experiment flags must be Boolean")
        if not 0 <= _integer(self.observer, "observer") < 4:
            raise ValueError("observer outside supported seats")

    @property
    def reward_scheme(self):
        return PROGRESS_REWARD if self.point_progress else REWARD_SCHEME

    def for_observer(self, seat):
        return replace(self, observer=seat)

    def __call__(self, *, max_actions):
        return HypothesisEnv(
            max_actions=max_actions,
            knowledge=self.knowledge,
            point_progress=self.point_progress,
            observer=self.observer,
        )

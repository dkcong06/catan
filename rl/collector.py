"""Complete-episode, per-seat training records; NumPy only, no trainer.

Policy callbacks receive only independent Decisions and per-seat RNGs. Rows are
interleaved across games; next_rows always links the same game and acting seat.
The last row for each seat receives its outcome and closes at that seat's final
observation. Opponent decisions are never mistaken for that seat's next state.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import comb, isclose, isfinite, log
from numbers import Real
from random import Random
from typing import NamedTuple

import numpy as np

from catan.state import _integer, _player_count
from catan.validation import _fingerprint, assert_invariants
from rl.actions import ACTION_DIM, ACTION_SCHEMA_VERSION, _discard_count_mask
from rl.encoding import OBS_DIM, SCHEMA_VERSION
from rl.economics import ECONOMIC_OBS_DIM, ECONOMIC_SCHEMA_VERSION
from rl.env import CatanEnv, ENV_SCHEMA_VERSION, REWARD_SCHEME
from rl.policies import POLICIES, POLICY_VERSIONS

COLLECTOR_SCHEMA_VERSION = 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)


class PolicySample(NamedTuple):
    """Probability of the actual submitted action, not an invented target.

    Discards require five conditional log probabilities whose sum is log_prob.
    value=None means no critic was used; stored values are NaN in that case.
    """

    action: object
    log_prob: float
    value: float | None = None
    discard_log_probs: tuple[float, ...] | None = None


def _discard_logs(hand, counts):
    remaining, total = sum(counts), sum(hand)
    logs = []
    for held, chosen in zip(hand, counts):
        logs.append(
            log(comb(held, chosen))
            + log(comb(total - held, remaining - chosen))
            - log(comb(total, remaining))
        )
        remaining -= chosen
        total -= held
    return tuple(logs)


class ReferencePolicy:
    """Batched adapter preserving the existing policies' exact choices/RNG.

    A name applies to every seat; a three/four-name sequence assigns each seat.
    Random discard probabilities describe physical-card sampling, not uniform
    resource-count vectors. No extra random draws are used for probabilities.
    """

    __slots__ = ("names", "tag")

    def __init__(self, names="random"):
        names = (names,) if isinstance(names, str) else tuple(names)
        if len(names) not in (1, 3, 4) or any(name not in POLICIES for name in names):
            raise ValueError("use one policy name or three/four known seat policies")
        self.names = names
        self.tag = "reference:" + ",".join(
            f"{name}-v{POLICY_VERSIONS[name]}" for name in names
        )

    def __call__(self, decisions, rngs):
        samples = []
        for decision, rng in zip(decisions, rngs, strict=True):
            name = (
                self.names[0] if len(self.names) == 1 else self.names[decision.player]
            )
            action = POLICIES[name](decision, rng)
            if decision.discard_hand is not None:
                logs = (
                    _discard_logs(decision.discard_hand, action)
                    if name == "random"
                    else (0.0,) * 5
                )
                samples.append(PolicySample(action, sum(logs), None, logs))
            else:
                probability = (
                    -log(np.count_nonzero(decision.mask)) if name == "random" else 0.0
                )
                samples.append(PolicySample(action, probability))
        return tuple(samples)


@dataclass(frozen=True, slots=True)
class Episode:
    seed: int
    policy_seed: int
    players: int
    actions: int
    winner: int | None
    terminated: bool
    truncated: bool
    fingerprint: str


@dataclass(frozen=True, slots=True, eq=False)
class Discards:
    rows: np.ndarray
    hands: np.ndarray
    counts: np.ndarray
    log_probs: np.ndarray

    def masks(self):
        """Reconstruct conditional masks only when needed for a training batch."""
        result = np.empty((len(self.rows), 5, 20), dtype=np.bool_)
        for i, (hand, counts) in enumerate(zip(self.hands, self.counts, strict=True)):
            chosen = tuple(map(int, counts))
            for resource in range(5):
                result[i, resource] = _discard_count_mask(
                    hand, sum(chosen), chosen[:resource]
                )
        return result


@dataclass(frozen=True, slots=True, eq=False)
class RolloutBatch:
    """Owned, read-only contiguous columns, valid after later collections.

    next_rows=-1 denotes the seat's final observation, not another player.
    elapsed_actions is a positive engine-action gap; no discount is imposed.
    Missing value estimates are NaN. Only final per-seat rows carry close flags
    and terminal rewards. Absent/unacted seats have last_rows=-1.
    """

    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    next_rows: np.ndarray
    elapsed_actions: np.ndarray
    episode_indices: np.ndarray
    players: np.ndarray
    phases: np.ndarray
    steps: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    discards: Discards
    episodes: tuple[Episode, ...]
    final_observations: np.ndarray
    last_rows: np.ndarray
    policy_tag: str
    observation_schema: int = SCHEMA_VERSION
    reward_scheme: str = REWARD_SCHEME

    def __len__(self):
        return len(self.actions)

    def successor_observations(self, rows):
        """Gather only requested successors; do not duplicate every observation.

        Internal successors reuse recorded observations. Closing rows use the
        same seat's final view; true terminals must not be value-bootstrapped.
        """
        rows = np.asarray(rows)
        if rows.ndim == 1 and rows.size == 0:
            return np.empty((0, self.observations.shape[1]), dtype=np.float32)
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or np.any(rows >= len(self))
            or np.any(rows < 0)
        ):
            raise ValueError("rows must be a one-dimensional array of valid integers")
        links = self.next_rows[rows]
        out = np.empty((len(rows), self.observations.shape[1]), dtype=np.float32)
        internal = links >= 0
        out[internal] = self.observations[links[internal]]
        closing = rows[~internal]
        out[~internal] = self.final_observations[
            self.episode_indices[closing], self.players[closing]
        ]
        return out

    @property
    def payload_bytes(self):
        arrays = (getattr(self, name) for name, _, _ in _ROW_SPEC)
        return sum(array.nbytes for array in arrays) + sum(
            array.nbytes
            for array in (
                self.discards.rows,
                self.discards.hands,
                self.discards.counts,
                self.discards.log_probs,
                self.final_observations,
                self.last_rows,
            )
        )


_ROW_SPEC = (
    ("observations", np.float32, (OBS_DIM,)),
    ("masks", np.bool_, (ACTION_DIM,)),
    ("actions", np.int16, ()),
    ("log_probs", np.float32, ()),
    ("values", np.float32, ()),
    ("rewards", np.float32, ()),
    ("next_rows", np.int32, ()),
    ("elapsed_actions", np.int32, ()),
    ("episode_indices", np.int32, ()),
    ("players", np.int8, ()),
    ("phases", np.int8, ()),
    ("steps", np.int32, ()),
    ("terminated", np.bool_, ()),
    ("truncated", np.bool_, ()),
)


class _Rows:
    """Private geometric storage; views never escape before finalization."""

    def __init__(self, limit, observation_dim=OBS_DIM):
        self.limit = limit
        self.spec = tuple(
            (name, dtype, (observation_dim,) if name == "observations" else shape)
            for name, dtype, shape in _ROW_SPEC
        )
        self.capacity = min(256, limit)
        self.size = 0
        self.arrays = {
            name: np.empty((self.capacity,) + shape, dtype=dtype)
            for name, dtype, shape in self.spec
        }
        self.discards = []

    def append(self, request, sample, episode, step):
        if self.size == self.capacity:
            self.capacity = min(self.limit, self.capacity * 2)
            for name, _, shape in self.spec:
                self.arrays[name].resize((self.capacity,) + shape, refcheck=False)
        row = self.size
        self.size += 1
        a = self.arrays
        a["observations"][row] = request.observation
        a["masks"][row] = False if request.mask is None else request.mask
        discard = request.discard_hand is not None
        a["actions"][row] = -1 if discard else sample.action
        a["log_probs"][row] = sample.log_prob
        a["values"][row] = np.nan if sample.value is None else sample.value
        a["rewards"][row] = 0
        a["next_rows"][row] = -1
        a["elapsed_actions"][row] = 0
        a["episode_indices"][row] = episode
        a["players"][row] = request.player
        a["phases"][row] = request.phase
        a["steps"][row] = step
        a["terminated"][row] = a["truncated"][row] = False
        if discard:
            self.discards.append(
                (row, request.discard_hand, sample.action, sample.discard_log_probs)
            )
        return row

    def finish(self):
        for name, _, shape in self.spec:
            self.arrays[name].resize((self.size,) + shape, refcheck=False)
            self.arrays[name].flags.writeable = False
        records = self.discards
        columns = (
            np.asarray([r[0] for r in records], dtype=np.int32),
            np.asarray([r[1] for r in records], dtype=np.int8).reshape(-1, 5),
            np.asarray([r[2] for r in records], dtype=np.int8).reshape(-1, 5),
            np.asarray([r[3] for r in records], dtype=np.float32).reshape(-1, 5),
        )
        for column in columns:
            column.flags.writeable = False
        return self.arrays, Discards(*columns)


def _number(value, name, *, probability=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result) or (probability and result > 0):
        raise ValueError(
            f"{name} must be finite" + (" and nonpositive" if probability else "")
        )
    if abs(result) > _FLOAT32_MAX:
        raise ValueError(f"{name} is outside float32 storage")
    return result


def _sample(request, sample):
    if not isinstance(sample, PolicySample):
        raise ValueError("policy must return one PolicySample per decision")
    probability = _number(sample.log_prob, "log_prob", probability=True)
    value = None if sample.value is None else _number(sample.value, "value")
    if request.discard_hand is None:
        action = _integer(sample.action, "action")
        if not 0 <= action < ACTION_DIM or not request.mask[action]:
            raise ValueError("policy selected an illegal catalog action")
        if sample.discard_log_probs is not None:
            raise ValueError("catalog actions cannot have discard log probabilities")
        return PolicySample(action, probability, value)
    action = tuple(_integer(count, "discard count") for count in sample.action)
    if (
        len(action) != 5
        or sum(action) != request.discard_required
        or any(
            count < 0 or count > held
            for count, held in zip(action, request.discard_hand)
        )
    ):
        raise ValueError("policy selected an illegal complete discard")
    if sample.discard_log_probs is None or len(sample.discard_log_probs) != 5:
        raise ValueError("discards require five conditional log probabilities")
    logs = tuple(
        _number(p, "discard log_prob", probability=True)
        for p in sample.discard_log_probs
    )
    if not isclose(sum(logs), probability, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("conditional discard log probabilities must sum to log_prob")
    return PolicySample(action, probability, value, logs)


def collect_rollouts(
    policy,
    seeds=(0,),
    *,
    n_players=4,
    policy_seed=0,
    max_actions=5000,
    batch_size=8,
    policy_tag=None,
    validate=False,
    record_players=None,
    env_factory=None,
):
    """Collect finite complete episodes with at most batch_size live games.

    policy(tuple[Decision], tuple[Random]) returns equally many PolicySamples.
    No game state/RNG is passed to it. Games refill vacated slots in seed order.
    The caller bounds data memory by seed count and the per-game accepted-action
    cap; at most len(seeds)*max_actions rows are retained. Repeated calls own
    disjoint output storage. This is synchronous batching, not process parallelism.
    record_players optionally retains only those seats' rows/final views; every
    seat still acts with its own unchanged RNG and the game continues normally.
    """
    if not callable(policy):
        raise ValueError("policy must be callable")
    players = _player_count(n_players)
    seeds = tuple(_integer(seed, "seed") for seed in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and distinct")
    policy_seed = _integer(policy_seed, "policy_seed")
    batch_size = _integer(batch_size, "batch_size")
    max_actions = _integer(max_actions, "max_actions")
    if min(batch_size, max_actions) < 1:
        raise ValueError("batch_size and max_actions must be positive")
    if len(seeds) * max_actions > np.iinfo(np.int32).max:
        raise ValueError("row bound exceeds int32 indexing")
    if type(validate) is not bool:
        raise ValueError("validate must be a bool")
    policy_tag = (
        getattr(policy, "tag", "unspecified") if policy_tag is None else policy_tag
    )
    if not isinstance(policy_tag, str):
        raise ValueError("policy_tag must be a string")
    if isinstance(policy, ReferencePolicy) and len(policy.names) not in (1, players):
        raise ValueError("reference seat policies must match n_players")
    recorded = (
        tuple(range(players))
        if record_players is None
        else tuple(_integer(seat, "record_player") for seat in record_players)
    )
    if (
        not recorded
        or len(set(recorded)) != len(recorded)
        or any(not 0 <= seat < players for seat in recorded)
    ):
        raise ValueError("record_players must contain distinct seats in this game")
    keep = tuple(seat in recorded for seat in range(players))
    factory = CatanEnv if env_factory is None else env_factory
    if not callable(factory):
        raise ValueError("env_factory must be callable")
    observation_dim = _integer(
        getattr(factory, "observation_dim", OBS_DIM), "observation_dim"
    )
    observation_schema = getattr(factory, "observation_schema", SCHEMA_VERSION)
    reward_scheme = getattr(factory, "reward_scheme", REWARD_SCHEME)
    if (observation_dim, observation_schema) not in (
        (OBS_DIM, 2),
        (OBS_DIM + 30, 3),
        (ECONOMIC_OBS_DIM, ECONOMIC_SCHEMA_VERSION),
    ):
        raise ValueError("unsupported observation profile")
    if reward_scheme not in (REWARD_SCHEME, "terminal_win_net_vp_0.1_v1"):
        raise ValueError("unsupported reward profile")
    shaped = reward_scheme != REWARD_SCHEME
    storage = _Rows(len(seeds) * max_actions, observation_dim)
    last = np.full((len(seeds), 4), -1, dtype=np.int32)
    final = np.zeros((len(seeds), 4, observation_dim), dtype=np.float32)
    episodes = [None] * len(seeds)
    active = []
    launched = 0

    def launch(index):
        env = factory(max_actions=max_actions)
        request = None
        try:
            request = env.reset(seeds[index], n_players=players)
            if validate:
                assert_invariants(env._state)
        except Exception as error:
            error.add_note(
                f"collector episode={index}, seed={seeds[index]}, policy_seed={policy_seed}, action=0, actor={getattr(request, 'player', None)}, phase=INITIALIZE"
            )
            raise
        streams = tuple(
            Random(f"catan-evaluation-v1:{policy_seed}:{seeds[index]}:{p}")
            for p in range(players)
        )
        return index, env, streams, request

    while active or launched < len(seeds):
        while len(active) < batch_size and launched < len(seeds):
            active.append(launch(launched))
            launched += 1
        decisions = tuple(item[3] for item in active)
        rngs = tuple(item[2][item[3].player] for item in active)
        for request in decisions:
            request.observation.flags.writeable = False
            if request.mask is not None:
                request.mask.flags.writeable = False
        try:
            outputs = tuple(policy(decisions, rngs))
            if len(outputs) != len(active):
                raise ValueError("policy must return exactly one sample per decision")
        except Exception as error:
            error.add_note(
                "collector policy batch: "
                + repr(
                    tuple(
                        (
                            i,
                            seeds[i],
                            env.actions + 1,
                            request.player,
                            request.phase.name,
                        )
                        for i, env, _, request in active
                    )
                )
            )
            raise
        samples = []
        for (index, env, _, request), output in zip(active, outputs, strict=True):
            try:
                samples.append(_sample(request, output))
            except Exception as error:
                error.add_note(
                    f"collector episode={index}, seed={seeds[index]}, policy_seed={policy_seed}, action={env.actions + 1}, actor={request.player}, phase={request.phase.name}"
                )
                raise
        remaining = []
        for (index, env, streams, request), sample in zip(active, samples, strict=True):
            step = env.actions
            try:
                outcome = env.step(sample.action, player=request.player)
                if validate:
                    assert_invariants(env._state)
                    if hasattr(env, "validate_knowledge"):
                        env.validate_knowledge()
            except Exception as error:
                error.add_note(
                    f"collector episode={index}, seed={seeds[index]}, policy_seed={policy_seed}, action={step + 1}, actor={request.player}, phase={request.phase.name}"
                )
                raise
            if keep[request.player]:
                row = storage.append(request, sample, index, step)
                previous = int(last[index, request.player])
                if previous >= 0:
                    storage.arrays["next_rows"][previous] = row
                    storage.arrays["elapsed_actions"][previous] = step - int(
                        storage.arrays["steps"][previous]
                    )
                last[index, request.player] = row
            if shaped and any(outcome.rewards):
                for seat in recorded:
                    closing = int(last[index, seat])
                    if closing >= 0:
                        storage.arrays["rewards"][closing] += outcome.rewards[seat]
            if not outcome.terminated and not outcome.truncated:
                remaining.append((index, env, streams, env.decision()))
                continue
            for seat in recorded:
                final[index, seat] = env.observe(seat)
                closing = int(last[index, seat])
                if closing >= 0:
                    if not shaped:
                        storage.arrays["rewards"][closing] = outcome.rewards[seat]
                    storage.arrays["elapsed_actions"][closing] = env.actions - int(
                        storage.arrays["steps"][closing]
                    )
                    storage.arrays["terminated"][closing] = outcome.terminated
                    storage.arrays["truncated"][closing] = outcome.truncated
            digest = sha256(_fingerprint(env._state, streams[0]).encode())
            digest.update(repr(tuple(stream.getstate() for stream in streams)).encode())
            episodes[index] = Episode(
                seeds[index],
                policy_seed,
                players,
                env.actions,
                outcome.winner,
                outcome.terminated,
                outcome.truncated,
                digest.hexdigest(),
            )
        active = remaining
    arrays, discards = storage.finish()
    final.flags.writeable = last.flags.writeable = False
    return RolloutBatch(
        **arrays,
        discards=discards,
        episodes=tuple(episodes),
        final_observations=final,
        last_rows=last,
        policy_tag=policy_tag,
        observation_schema=observation_schema,
        reward_scheme=reward_scheme,
    )


def save_rollouts(batch, path, *, compressed=False):
    """Numeric NPZ plus JSON metadata, readable with allow_pickle=False.

    Persistence is optional and outside the collection loop. No game/RNG objects
    or arbitrary Python objects are serialized. The path must end in .npz.
    """
    from pathlib import Path

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("rollout archive path must end in .npz")
    if type(compressed) is not bool:
        raise ValueError("compressed must be a bool")
    metadata = {
        "collector_schema": 1
        if batch.observation_schema == 2 and batch.reward_scheme == REWARD_SCHEME
        else (3 if batch.observation_schema == 4 else 2),
        "environment_schema": ENV_SCHEMA_VERSION,
        "observation_schema": batch.observation_schema,
        "action_schema": ACTION_SCHEMA_VERSION,
        "reward_scheme": batch.reward_scheme,
        "policy_tag": batch.policy_tag,
        "episodes": [asdict(episode) for episode in batch.episodes],
    }
    arrays = {name: getattr(batch, name) for name, _, _ in _ROW_SPEC}
    arrays.update(
        discard_rows=batch.discards.rows,
        discard_hands=batch.discards.hands,
        discard_counts=batch.discards.counts,
        discard_log_probs=batch.discards.log_probs,
        final_observations=batch.final_observations,
        last_rows=batch.last_rows,
        metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
    )
    writer = np.savez_compressed if compressed else np.savez
    with path.open("wb") as file:
        writer(file, **arrays)

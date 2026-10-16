"""Profile-aware fixed-cohort runs for isolated feature and board comparisons."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import torch

from learning.cohorts import FixedCohortRun
from learning.evaluation import EvaluationPolicy
from learning.hypotheses import digest, read
from learning.imitation import Demonstrations, ImitationConfig, ImitationTrainer
from learning.policy import NeuralPolicy
from learning.ppo import PPOUpdater
from learning.training import RunConfig, _atomic_json, balanced_assignment
from rl.economics import EconomicEnvFactory
from rl.encoding import OBS_DIM
from rl.collector import ReferencePolicy, collect_rollouts


class OriginalPolicy(NeuralPolicy):
    """An explicit observation-prefix adapter for the frozen model1 opponent."""

    def __call__(self, decisions, rngs):
        return super().__call__(
            tuple(replace(d, observation=d.observation[:OBS_DIM]) for d in decisions),
            rngs,
        )


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    feature_updates: int = 84
    graph_updates: int = 84
    replicates: int = 3
    development_seeds: int = 4
    test_seeds: int = 64
    teacher_games: int = 128
    validation_games: int = 16
    warmup_epochs: int = 2
    games_per_update: int = 8
    max_actions: int = 5000
    target_seconds: int = 10800
    seed_shift: int = 3000000

    def __post_init__(self):
        for name, low, high in (
            ("feature_updates", 1, 1024),
            ("graph_updates", 1, 1024),
            ("replicates", 1, 3),
            ("development_seeds", 1, 16),
            ("test_seeds", 1, 128),
            ("teacher_games", 2, 512),
            ("validation_games", 2, 64),
            ("warmup_epochs", 1, 4),
            ("games_per_update", 1, 16),
            ("max_actions", 16, 5000),
            ("target_seconds", 60, 21600),
            ("seed_shift", 0, 50000000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} outside comparison bounds")
        if self.teacher_games % 2 or self.validation_games % 2:
            raise ValueError("teacher/validation games must be even")

    def run_config(self, stage, replicate, model):
        if stage not in ("features", "graph") or not 0 <= replicate < self.replicates:
            raise ValueError("unknown stage/replicate")
        updates = self.feature_updates if stage == "features" else self.graph_updates
        milestones = tuple(
            sorted(
                {
                    u
                    for u in (1, 2, 4, 8, 16, 28, 56, 112, 224, 448, 896, updates)
                    if u <= updates
                }
            )
        )
        return RunConfig(
            budget_seconds=86400,
            max_updates=updates,
            games_per_update=self.games_per_update,
            active_games=self.games_per_update,
            max_actions=self.max_actions,
            evaluation_max_actions=self.max_actions,
            tune_seeds=self.development_seeds,
            test_seeds=self.test_seeds,
            evaluate_every=100000,
            checkpoint_every=1,
            hidden_size=model.hidden_size,
            model_seed=model.seed,
            shuffle_seed=1901 + replicate,
            training_seed_base=(600000000 if stage == "features" else 700000000)
            + replicate * 1000000
            + self.seed_shift,
            evaluation_seed_offset=(11000000 if stage == "features" else 12000000)
            + replicate * 1000
            + self.seed_shift,
            opponent_schedule=("greedy", "frozen_original", "greedy", "random"),
            balanced_schedule=True,
            tuning_opponent="greedy",
            milestones=milestones,
        )


class ArchitectureRun(FixedCohortRun):
    """Persist feature/opponent identities; derive scoped seats from committed u."""

    def __init__(
        self,
        root,
        *,
        enabled,
        original_checkpoint,
        config,
        ppo_config,
        initial_checkpoint=None,
        resume=False,
        on_progress=None,
    ):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        self.profile = {
            "schema": 1,
            "observation": 4,
            "economic_enabled": enabled,
            "collection_feature_scope": "current learner only",
            "reference_evaluation_scope": "candidate only",
            "original_opponent_sha256": digest(original_checkpoint),
        }
        self.on_progress = on_progress
        if (
            resume
            and read(Path(root) / "state.json").get("architecture_profile")
            != self.profile
        ):
            raise ValueError("stored environment/opponent profile changed")
        original = PPOUpdater.load(original_checkpoint)
        if original.model.schema != (1, 2, 1):
            raise ValueError("fixed opponent requires original model1")
        self.original = OriginalPolicy(original.model, policy_version="frozen-original")
        super().__init__(
            root,
            config=config,
            ppo_config=ppo_config,
            initial_checkpoint=initial_checkpoint,
            resume=resume,
            env_factory=EconomicEnvFactory(enabled),
        )
        if self.updater.model.schema[0] not in (3, 4):
            self.close()
            raise ValueError("use the economic or graph model")

    def _state(self, status):
        return {**super()._state(status), "architecture_profile": self.profile}

    def _collection_policy(self, update, players):
        expected_players, seat, name = balanced_assignment(
            update, self.config.opponent_schedule
        )
        if players != expected_players:
            raise ValueError("balanced player assignment changed")
        self.env_factory = EconomicEnvFactory(
            self.profile["economic_enabled"], feature_player=seat
        )
        callback = EvaluationPolicy(
            self.updater.policy(),
            self.original if name == "frozen_original" else name,
            seat,
        )
        return callback, (seat,), name, callback.reference.tag

    def _evaluate(self, updater, seeds, path):
        # Restore reference evaluation's rotating candidate scope, not the last
        # training seat. evaluate_neural applies for_observer to every role.
        self.env_factory = EconomicEnvFactory(self.profile["economic_enabled"])
        result = super()._evaluate(updater, seeds, path)
        result["architecture_profile"] = self.profile
        _atomic_json(path, result)
        return result

    def progress(self, record=None, *, status="running"):
        result = super().progress(record, status=status)
        if self.on_progress is not None:
            self.on_progress(result)
        return result


def teachers(root, config, should_pause=lambda: False):
    """Shared observation-only labels; each bounded shard commits independently."""
    root = Path(root)
    root.mkdir(exist_ok=True)
    shards = []
    for role, count, base in (
        ("training", config.teacher_games, 850000000),
        ("validation", config.validation_games, 850100000),
    ):
        for players in (3, 4):
            for start in range(0, count // 2, 8):
                if should_pause():
                    return None
                seeds = list(
                    range(
                        base + config.seed_shift + (players - 3) * 10000 + start,
                        base
                        + config.seed_shift
                        + (players - 3) * 10000
                        + min(start + 8, count // 2),
                    )
                )
                name = f"{role}-{players}p-{start:04d}"
                path, receipt = root / f"{name}.npz", root / f"{name}.json"
                identity = dict(
                    role=role,
                    players=players,
                    seeds=seeds,
                    max_actions=config.max_actions,
                    policy_seed=17,
                    observation=4,
                    teacher=ReferencePolicy("greedy").tag,
                )
                if receipt.exists():
                    item = read(receipt)
                    if item["identity"] != identity or digest(path) != item["sha256"]:
                        raise ValueError("teacher identity/hash changed")
                else:
                    started = perf_counter()
                    batch = collect_rollouts(
                        ReferencePolicy("greedy"),
                        seeds,
                        n_players=players,
                        max_actions=config.max_actions,
                        batch_size=8,
                        policy_seed=17,
                        env_factory=EconomicEnvFactory(True),
                    )
                    data = Demonstrations.from_rollouts(batch)
                    pending = root / f"{name}.pending.npz"
                    pending.unlink(missing_ok=True)  # Only our uncommitted shard.
                    data.save(pending)
                    pending.replace(path)
                    item = dict(
                        file=path.name,
                        identity=identity,
                        sha256=digest(path),
                        rows=len(data),
                        seconds=perf_counter() - started,
                        episodes=[asdict(e) for e in batch.episodes],
                    )
                    _atomic_json(receipt, item)
                    del data, batch
                shards.append(item)
    result = dict(schema=1, shards=shards)
    _atomic_json(root / "manifest.json", result)
    return result


def warmup(
    cold,
    destination,
    teacher_root,
    manifest,
    *,
    epochs,
    shuffle_seed,
    should_pause=lambda: False,
):
    """Exact shard-boundary Adam/shuffle resume; PPO starts with fresh Adam."""
    cold, destination, teacher_root = map(Path, (cold, destination, teacher_root))
    config = ImitationConfig(epochs=1, learning_rate=0.0001)
    identity = dict(
        schema=1,
        cold_sha256=digest(cold),
        epochs=epochs,
        shuffle_seed=shuffle_seed,
        config=asdict(config),
        teacher=manifest,
    )
    receipt, state_path = (
        destination.with_suffix(".json"),
        destination.with_suffix(".warm.pt"),
    )
    if receipt.exists():
        result = read(receipt)
        if result["identity"] != identity or result["sha256"] != digest(destination):
            raise ValueError("warm-up checkpoint/recipe changed")
        return result
    source = PPOUpdater.load(cold)
    trainer = ImitationTrainer(source.model, config, shuffle_seed=shuffle_seed)
    sequence = []
    training = [s for s in manifest["shards"] if s["identity"]["role"] == "training"]
    for epoch in range(epochs):
        sequence.extend(
            (epoch, s) for s in (training if epoch % 2 == 0 else training[::-1])
        )

    def data(shard):
        path = teacher_root / shard["file"]
        if digest(path) != shard["sha256"]:
            raise ValueError("teacher shard changed")
        return Demonstrations.load(path)

    def validation():
        total = rows = 0
        for shard in manifest["shards"]:
            if shard["identity"]["role"] == "validation":
                labels = data(shard)
                total += trainer.loss(labels) * len(labels)
                rows += len(labels)
        return total / rows

    records, seconds = [], 0.0
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or not 0 <= len(saved["records"]) <= len(
            sequence
        ):
            raise ValueError("warm-up resume identity/cursor changed")
        trainer.model.load_state_dict(saved["model"], strict=True)
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.generator.set_state(saved["generator"])
        trainer.optimizer_steps = saved["optimizer_steps"]
        records, seconds, before = saved["records"], saved["seconds"], saved["before"]
    else:
        before = validation()
    for epoch, shard in sequence[len(records) :]:
        if should_pause():
            return None
        started = perf_counter()
        fitted = trainer.fit(data(shard))
        seconds += perf_counter() - started
        records.append(dict(epoch=epoch, file=shard["file"], **fitted))
        pending = state_path.with_suffix(".tmp")
        torch.save(
            dict(
                identity=identity,
                model=trainer.model.state_dict(),
                optimizer=trainer.optimizer.state_dict(),
                generator=trainer.generator.get_state(),
                optimizer_steps=trainer.optimizer_steps,
                records=records,
                seconds=seconds,
                before=before,
            ),
            pending,
        )
        pending.replace(state_path)
    # No validation-dependent selection or tuning: the last fixed pass is used.
    after = validation()
    PPOUpdater(trainer.model, source.config, shuffle_seed=shuffle_seed).save(
        destination
    )
    result = dict(
        identity=identity,
        sha256=digest(destination),
        optimizer_steps=trainer.optimizer_steps,
        validation_loss_before=before,
        validation_loss_after=after,
        seconds=seconds,
        records=records,
        note="Actor-only labels; fresh PPO Adam and unchanged win reward.",
    )
    _atomic_json(receipt, result)
    return result

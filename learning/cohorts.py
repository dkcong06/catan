"""Finite, transactional cohort runs shared by controlled comparisons."""

from pathlib import Path
import json

from learning.hypotheses import digest, read
from learning.ppo import PPOConfig
from learning.training import RunConfig, TrainingRun, _atomic_json, TUNE_SEED_BASE


class FixedCohortRun(TrainingRun):
    """Finite cohort allowance; cooperative pause never consumes update allowance."""

    def __init__(
        self,
        root,
        *,
        config,
        ppo_config,
        initial_checkpoint=None,
        resume=False,
        env_factory=None,
    ):
        if resume:
            state = read(Path(root) / "state.json")
            if (
                state.get("clock") != "fixed_cohorts_v1"
                or RunConfig(**state["config"]) != config
                or PPOConfig(**state["ppo_config"]) != ppo_config
            ):
                raise ValueError("stored comparison configuration changed")
        super().__init__(
            root,
            config=config,
            ppo_config=ppo_config,
            resume=resume,
            initial_checkpoint=initial_checkpoint,
            env_factory=env_factory,
        )

    def _state(self, status):
        result = super()._state(status)
        result["clock"] = "fixed_cohorts_v1"
        return result

    def _progress_allowance(self):
        return {
            "clock": "fixed_cohorts_v1",
            "remaining_cohorts": max(0, self.config.max_updates - self.updater.updates),
        }

    def train(self, should_pause=lambda: False):
        if should_pause():
            self.save(status="paused")
            return self.progress(status="paused")
        zero = self.root / "evaluation/milestone-u00000000.json"
        if not zero.exists():
            if self.updater.updates != 0:
                raise ValueError("initial development curve missing")
            report = self._evaluate(
                self.updater,
                range(
                    TUNE_SEED_BASE + self.config.evaluation_seed_offset,
                    TUNE_SEED_BASE
                    + self.config.evaluation_seed_offset
                    + self.config.tune_seeds,
                ),
                self.root / "evaluation/tuning-initial.json",
            )
            _atomic_json(
                zero,
                {
                    "update": 0,
                    "checkpoint": self._checkpoint_name(0),
                    "checkpoint_sha256": digest(
                        self.root / "checkpoints" / self._checkpoint_name(0)
                    ),
                    "evaluation": report,
                },
            )
            self.best_wins = self._selection_wins(report)
            self.save()
        while self.updater.updates < self.config.max_updates and not should_pause():
            record = self.step()
            print(
                json.dumps(
                    {
                        "event": "update",
                        "case": self.root.name,
                        "update": record["update"],
                        "rows": record["rows"],
                    }
                ),
                flush=True,
            )
        status = (
            "trained" if self.updater.updates == self.config.max_updates else "paused"
        )
        for update in (0, *self.config.milestones):
            if update > self.updater.updates:
                continue
            curve = read(self.root / "evaluation" / f"milestone-u{update:08d}.json")
            checkpoint = self.root / "checkpoints" / self._checkpoint_name(update)
            if (
                curve["update"] != update
                or curve["checkpoint"] != checkpoint.name
                or curve["checkpoint_sha256"] != digest(checkpoint)
            ):
                raise ValueError("milestone checkpoint identity changed")
        self.save(status=status)
        return self.progress(status=status)

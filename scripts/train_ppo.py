"""Run or resume a bounded CPU Catan PPO experiment inside the learning container."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys

from learning.ppo import PPOConfig
from learning.training import RunConfig, TrainingRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--max-updates", type=int, default=100000)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument("--evaluation-max-actions", type=int, default=5000)
    parser.add_argument("--tune-seeds", type=int, default=8)
    parser.add_argument("--test-seeds", type=int, default=32)
    parser.add_argument("--evaluate-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--evaluation-seed-offset", type=int, default=0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=("self", "greedy", "random", "previous"),
        default=("self",),
    )
    parser.add_argument("--tuning-opponent", choices=("all", "greedy"), default="all")
    args = parser.parse_args()
    if args.resume and any(
        name.split("=")[0] not in ("--output", "--resume")
        for name in sys.argv[1:]
        if name.startswith("--")
    ):
        parser.error(
            "resume uses the stored configuration/deadline; pass only --output and --resume"
        )
    try:
        config = RunConfig(
            budget_seconds=args.minutes * 60,
            max_updates=args.max_updates,
            games_per_update=args.games,
            active_games=args.games,
            max_actions=args.max_actions,
            evaluation_max_actions=args.evaluation_max_actions,
            tune_seeds=args.tune_seeds,
            test_seeds=args.test_seeds,
            evaluate_every=args.evaluate_every,
            checkpoint_every=args.checkpoint_every,
            evaluation_seed_offset=args.evaluation_seed_offset,
            opponent_schedule=tuple(args.opponents),
            tuning_opponent=args.tuning_opponent,
        )
        ppo = PPOConfig(gae_lambda=args.gae_lambda)
    except ValueError as error:
        parser.error(str(error))
    run = TrainingRun(
        args.output,
        config=config,
        ppo_config=ppo,
        resume=args.resume,
        initial_checkpoint=args.initial_checkpoint,
    )

    def stop(signum, frame):
        run.stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        print(
            json.dumps(
                {
                    "event": "started",
                    "config": asdict(run.config),
                    "ppo": asdict(run.updater.config),
                    "deadline": run.deadline,
                }
            ),
            flush=True,
        )
        result = run.run()
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    except Exception as error:
        run.progress(status="failed")
        error.add_note(
            f"training directory={args.output}; resume from committed state.json after fixing the cause"
        )
        raise
    finally:
        run.close()


if __name__ == "__main__":
    main()

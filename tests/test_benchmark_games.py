import subprocess
import sys

import pytest


@pytest.mark.parametrize("players", (3, 4))
def test_game_benchmark_reports_completions_and_caps_without_conflating_them(players):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_games",
            "--games",
            "2",
            "--repeats",
            "1",
            "--players",
            str(players),
            "--max-actions",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Completed: 0; capped: 2" in result.stdout
    assert "Winners (None means capped): [None, None]" in result.stdout
    assert "actions/s" in result.stdout


def test_game_benchmark_rejects_nonpositive_batches():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.benchmark_games", "--games", "0"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "must be positive" in result.stderr

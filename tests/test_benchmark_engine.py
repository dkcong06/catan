import subprocess
import sys

import pytest


@pytest.mark.parametrize("players", (3, 4))
def test_transition_benchmark_executes_valid_rounds_for_both_player_counts(players):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_engine",
            "--iterations",
            "2",
            "--repeats",
            "1",
            "--players",
            str(players),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert f"{players} players" in result.stdout
    assert f"Round = seven + {players - 1} preselected" in result.stdout
    assert "round: functional/in-place time ratio" in result.stdout
    assert "not game/RL throughput" in result.stdout


@pytest.mark.parametrize("option,value", [("--iterations", "0"), ("--repeats", "-1")])
def test_transition_benchmark_rejects_nonpositive_measurement_counts(option, value):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.benchmark_engine", option, value],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "iterations and repeats must be positive" in result.stderr

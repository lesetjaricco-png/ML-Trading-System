"""Smoke tests for the Docker-based workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docker_build_and_run_smoke() -> None:
    """Verify the Docker image can build and the default entry point starts."""
    if os.environ.get("SKIP_DOCKER_SMOKE", "0") == "1":
        return

    docker_cmd = os.environ.get("DOCKER_CMD", "docker")
    compose_cmd = [docker_cmd, "compose", "build", "backtest"]
    build_result = subprocess.run(
        compose_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert build_result.returncode == 0, (
        "Docker build failed\n"
        f"STDOUT:\n{build_result.stdout}\n"
        f"STDERR:\n{build_result.stderr}"
    )

    run_result = subprocess.run(
        [docker_cmd, "compose", "run", "--rm", "backtest", "python", "-c", "print('smoke ok')"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert run_result.returncode == 0, (
        "Docker smoke run failed\n"
        f"STDOUT:\n{run_result.stdout}\n"
        f"STDERR:\n{run_result.stderr}"
    )

"""Utility helpers: config loading, logging setup, plotting."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import yaml


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and return the YAML configuration file."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_dir: str = "logs",
) -> None:
    """Configure root logger with console (and optional file) handlers."""
    os.makedirs(log_dir, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(os.path.join(log_dir, log_file)))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=handlers,
    )

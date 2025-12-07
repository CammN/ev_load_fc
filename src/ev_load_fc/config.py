from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


# Automatically detect the top-level project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path:Path) -> Dict[str, Any]:
    """
    Load a YAML file into a dictionary.
    Returns an empty dict if the file does not exist.
    """
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_config() -> Dict[str, Any]:
    """Load the global project config from configs/config.yaml. """
    return _load_yaml(PROJECT_ROOT / "configs" / "config.yaml")


# Load once at import time
CFG = load_config()


def resolve_path(relative_path:str) -> Path:
    """Convert a path defined in config.yaml into an absolute path relative to the project root."""
    return PROJECT_ROOT / relative_path
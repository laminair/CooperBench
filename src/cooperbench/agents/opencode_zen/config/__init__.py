"""Configuration files and utilities for the OpenCode Zen adapter."""

from pathlib import Path

builtin_config_dir = Path(__file__).parent


def get_config_path(name: str) -> Path:
    path = builtin_config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"OpenCode Zen config not found: {path}")
    return path

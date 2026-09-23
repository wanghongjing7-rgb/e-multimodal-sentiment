"""Load explicit YAML paths with recursive task overrides."""
from pathlib import Path
from typing import Any
import yaml


def load_config(base_path: str | Path, *override_paths: str | Path | None) -> dict[str, Any]:
    """Load shared config then ordered task/local overrides from explicit paths."""
    def read(path: str | Path) -> dict[str, Any]:
        with Path(path).open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError("Config must be a mapping.")
        return value

    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        return result

    result = read(base_path)
    for path in override_paths:
        if path is not None:
            result = merge(result, read(path))
    return result

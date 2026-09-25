"""Load settings from config/config.yaml.

Every module takes a ``Config`` so tests can point paths at a temporary folder.
Set the ``CTS_CONFIG`` environment variable to use another YAML file.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


class Config(dict):
    """Dict with attribute access for nested sections (``cfg.labels.tp_atr``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config(_wrap(copy.deepcopy(dict(self), memo)))


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Read the YAML config, optionally deep-merging ``overrides`` on top."""
    path = Path(path or os.environ.get("CTS_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _merge(data, overrides)
    return _wrap(data)


def resolve_path(cfg: Config, key: str) -> Path:
    """Absolute path for ``cfg.paths[key]`` (relative paths are taken from the project root)."""
    p = Path(cfg.paths[key])
    return p if p.is_absolute() else ROOT / p


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

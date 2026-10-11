"""Portable paths and validated environment settings."""
from __future__ import annotations

import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".opus"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name}: 1 또는 0을 지정하세요.")


def env_number(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}: 0보다 큰 유한한 숫자를 지정하세요.")
    return value


def project_path(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


DATA_DIR = project_path("MSS_DATA_DIR", "data/raw")
MODEL_DIR = project_path("MSS_MODEL_DIR", "checkpoints/pretrained")
OUTPUT_DIR = project_path("MSS_OUTPUT_DIR", "results")
CACHE_DIR = project_path("MSS_CACHE_DIR", ".cache")

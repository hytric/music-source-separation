"""Scientific tensor, configuration, path, and cache-key contracts."""
from __future__ import annotations
import hashlib, json, unicodedata
from pathlib import Path
from typing import Any
import numpy as np

SOURCES = ("vocals", "drums", "bass", "other")
SAMPLE_RATE, SAMPLES, CONFIG_SCHEMA = 44_100, 1_323_000, 1
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": CONFIG_SCHEMA, "sample_rate": SAMPLE_RATE, "channels": 2, "samples": SAMPLES,
    "sources": list(SOURCES), "split_salt": "banelp-v1", "dev_tracks": 10, "evaluation_tracks": 40,
    "crop_seconds": 30, "stft": {"nperseg": 4096, "hop": 1024, "window": "hann-periodic", "boundary": "zeros", "padded": True},
    "metrics": {"center": False, "pinv_rcond": 1e-10, "energy_epsilon": 1e-12, "db_cap": 120.0},
    "separator": {"name": "htdemucs", "device": "cuda", "fp32": True, "batch_size": 1, "shifts": 0, "split": True, "segment": 7.0, "overlap": 0.25},
    "selection": {"sar_policy_status": "confirmed", "sar_floor_db": -0.5, "minimum_sdr_gain_db": 0.2, "baseline_train_data_policy": "allow-unknown-with-disclosure"}}

def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def cache_key(*, code_revision: str, lock_hash: str, source_manifest_hash: str, rir_manifest_hash: str,
              model_hashes: list[str], inference: dict[str, Any], synthesis: dict[str, Any] | None = None) -> str:
    obj = {"code_revision": code_revision, "lock_hash": lock_hash, "source_manifest_hash": source_manifest_hash,
           "rir_manifest_hash": rir_manifest_hash, "model_hashes": sorted(model_hashes), "inference": inference,
           "synthesis": synthesis or {}}
    return hashlib.sha256(canonical_json(obj)).hexdigest()

def validate_runtime_path(path: str | Path, runtime_dir: str | Path) -> Path:
    root, target = Path(runtime_dir).expanduser().resolve(), Path(path).expanduser().resolve()
    if target == root or root not in target.parents: raise ValueError(f"path must be beneath runtime_dir: {target}")
    return target

def normalize_track_name(name: str) -> str: return unicodedata.normalize("NFC", str(name))

def deterministic_split(names: list[str], salt: str = "banelp-v1") -> dict[str, str]:
    ordered = sorted((normalize_track_name(n) for n in names), key=lambda n: (hashlib.sha256(salt.encode()+b"\0"+n.encode()).hexdigest(), n))
    if len(ordered) != 50 or len(set(ordered)) != 50: raise ValueError("expected exactly 50 unique MUSDB18-HQ test track names")
    return {name: ("dev" if i < 10 else "evaluation") for i, name in enumerate(ordered)}

def validate_tensors(x: np.ndarray, estimates: np.ndarray, *, samples: int | None = None) -> None:
    if x.ndim != 2 or x.shape[0] != 2: raise ValueError(f"mixture must have shape (2,N), got {x.shape}")
    if estimates.ndim != 3 or estimates.shape[:2] != (4, 2): raise ValueError(f"estimates must have shape (4,2,N), got {estimates.shape}")
    if estimates.shape[-1] != x.shape[-1] or (samples is not None and x.shape[-1] != samples): raise ValueError("mixture/stem lengths differ from contract")
    if not np.isfinite(x).all() or not np.isfinite(estimates).all(): raise ValueError("tensors must be finite")

def validate_practical_config(config: dict[str, Any]) -> None:
    forbidden = {"targets", "ground_truth", "oracle", "rir", "rt60"}.intersection(config)
    if forbidden: raise ValueError(f"practical config cannot contain oracle fields: {sorted(forbidden)}")
    if config.get("schema_version") != CONFIG_SCHEMA: raise ValueError("unsupported config schema")
    if tuple(config.get("sources", ())) != SOURCES: raise ValueError("invalid source order")
    if config.get("sample_rate") != SAMPLE_RATE or config.get("channels") != 2: raise ValueError("config must be 44.1 kHz stereo")
    samples = config.get("samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0: raise ValueError("config samples must be a positive integer")
    if "selected" in config.get("selection", {}):
        raise ValueError("config.selection.selected is not allowed; the seal derives the candidate from results/develop.json")

def config_copy() -> dict[str, Any]: return json.loads(json.dumps(DEFAULT_CONFIG))

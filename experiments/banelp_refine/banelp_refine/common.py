"""Immutable inputs, atomic artifacts, and explicit experiment bindings."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

SOURCES = ("vocals", "drums", "bass", "other")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def emit(event, **fields):
    print(json.dumps({"event": event, "at": time.time(), **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def source_hash(names=None):
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) if names is None else [root / name for name in names]
    return digest({p.name: sha256(p) for p in paths})


def environment():
    return {"python": sys.version.split()[0], **{name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scipy", "soundfile", "demucs")}}


def load_config(path):
    c = read_json(path)
    if c.get("schema_version") != 1 or c.get("sources") != list(SOURCES):
        raise ValueError("unsupported configuration or source order")
    a = c["audio"]
    if (a["sample_rate"], a["channels"], a["record_seconds"]) != (44100, 2, 30):
        raise ValueError("this experiment requires 44.1kHz stereo 30s records")
    if sum(c["data"]["split_counts"].values()) != 100:
        raise ValueError("the song split must partition the 100 official train songs")
    if c["training"]["batch_size"] != 1 or c["training"]["precision"] != "fp32":
        raise ValueError("the verified implementation uses batch 1 and FP32")
    if c["model"]["widths"] != [24, 48, 96]:
        raise ValueError("changing the architecture requires a new experiment config/version")
    if not 0 < a["crop_seconds"] <= a["record_seconds"] - 2 * a["edge_seconds"]:
        raise ValueError("training crop does not fit the retained interior")
    if c["loss"]["variant"] not in ("base", "extended"):
        raise ValueError("loss variant must be base or extended")
    if c["training"]["max_steps"] <= 0 or c["training"]["validate_every"] <= 0:
        raise ValueError("training/validation intervals must be positive")
    return c


def baseline_import(repo):
    repo = Path(repo).resolve()
    if not (repo / "experiments/banelp/separators.py").is_file():
        raise ValueError("--baseline-repo must contain experiments/banelp")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def file_entry(path, root):
    path = Path(path)
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_entry(root, entry):
    root = Path(root).resolve()
    path = (root / entry["path"]).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"missing or unsafe artifact: {entry['path']}")
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"artifact content changed: {entry['path']}")
    return path


@contextlib.contextmanager
def exclusive(root, name):
    """Kernel lock releases even when a preparation process crashes."""
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f".{name}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another {name} operation is running") from exc
        yield


def runtime_root(path, baseline_runtime=None):
    root = Path(path).expanduser().resolve()
    if baseline_runtime:
        old = Path(baseline_runtime).resolve()
        if root == old or old in root.parents or root in old.parents:
            raise ValueError("refinement runtime must be separate from the sealed baseline runtime")
    root.mkdir(parents=True, exist_ok=True)
    return root

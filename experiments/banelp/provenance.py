"""Code/lock/manifest bindings shared by develop, seal and evaluate."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from .contracts import canonical_json, sha256_file

PACKAGE = Path(__file__).resolve().parent
LOCK_FILE = PACKAGE / "requirements.lock"

def code_files() -> list[Path]:
    """Code and lock files; the run config is bound separately through `config_hash` and the seal policy."""
    return sorted(PACKAGE.glob("*.py")) + [LOCK_FILE]

def code_hash() -> str:
    h = hashlib.sha256()
    for path in code_files():
        h.update(path.name.encode() + b"\0" + bytes.fromhex(sha256_file(path)))
    return h.hexdigest()

def lock_hash() -> str: return sha256_file(LOCK_FILE)

def experiment_config_hash(config_path: str | Path) -> str:
    """Config hash without the `selection` policy block, which the seal binds separately."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return hashlib.sha256(canonical_json({k: v for k, v in config.items() if k != "selection"})).hexdigest()

def manifest_paths(runtime: Path) -> dict[str, Path]:
    return {"dataset": runtime / "manifests/dataset.json", "rirs": runtime / "manifests/rirs.json",
            "models": runtime / "manifests/models.json"}

def current_binding(runtime: Path, config_path: str | Path, *, require=("dataset", "rirs", "models")) -> dict:
    """Hashes of the inputs that a result depends on; missing required manifests fail closed."""
    paths = manifest_paths(runtime)
    for name in require:
        if not paths[name].is_file(): raise FileNotFoundError(f"required {name} manifest missing: {paths[name]}")
    def maybe(name): return sha256_file(paths[name]) if paths[name].is_file() else None
    return {"code_hash": code_hash(), "lock_hash": lock_hash(), "config_hash": experiment_config_hash(config_path),
            "manifest_hash": maybe("dataset"), "rir_manifest_hash": maybe("rirs"), "models_manifest_hash": maybe("models")}

BOUND_KEYS = ("code_hash", "lock_hash", "config_hash", "manifest_hash", "rir_manifest_hash", "models_manifest_hash")

def verify_binding(recorded: dict, current: dict, *, what: str, keys=BOUND_KEYS) -> None:
    """One comparison used by seal (against develop.json) and evaluate (against the seal)."""
    for key in keys:
        if recorded.get(key) is None or recorded.get(key) != current.get(key):
            raise ValueError(f"{what}: {key} changed or missing (recorded {recorded.get(key)}, current {current.get(key)})")

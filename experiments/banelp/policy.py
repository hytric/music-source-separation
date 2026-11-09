"""Pre-registered selection policy lock and per-dependency checkpoint eligibility.

`lock-policy` writes the lock before `develop`; `develop`, `seal` and `evaluate` recompute its content from the current
config, registry and registered models and refuse any difference."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from .contracts import canonical_json
from .data import atomic_json
from .selection import BOOTSTRAP, CI_LEVEL, SEED, STATISTIC_WORDS
from .separators import record_eligibility

LOCK_REL = "manifests/selection-policy.json"
LOCK_SCHEMA = 1
RANKING = ["-sample_median_delta_db", "additional_inference_calls", "config_hash"]

BASELINE_DISCLOSURE_KO = ("기준 분리기 {model_id}({signature})는 clean winner가 아니다: 곡별 학습 데이터 확인이 없다 — 논문의 split 설명은 MUSDB18 test를 "
                          "학습에 쓰지 않았다고 하지만, 추가 800곡은 곡별로 감사되지 않았다.")
BASELINE_DISCLOSURE_EN = ("Baseline separator {model_id} ({signature}) is not a clean winner: training data was not verified per song — the paper's "
                          "split describes the MUSDB18 test set as unused, but the 800 extra songs were not audited per song.")

def lock_sha256(content: dict) -> str: return hashlib.sha256(canonical_json(content)).hexdigest()

def _signature(model: dict) -> str:
    members = model.get("members") or []
    return Path(members[0]["path"]).name.split("-")[0] if model.get("family", "demucs") == "demucs" and len(members) == 1 else model["model_id"]

def check_baseline_decision(model: dict, selection: dict) -> dict:
    """The explicit user decision bound to this exact model (id + sha256, and signature when given)."""
    matches = [d for d in selection.get("baseline_decisions", []) if d.get("model_id") == model["model_id"] and d.get("sha256") == model["sha256"]]
    if len(matches) != 1: raise ValueError(f"baseline {model['model_id']} (sha256 {model['sha256']}) has unverified training data and no matching user decision record in selection.baseline_decisions")
    decision = matches[0]
    if decision.get("decided_by") != "user" or not decision.get("decision") or not decision.get("date"): raise ValueError("baseline decision record needs decided_by=user, decision and date")
    if decision.get("demucs_signature") not in (None, _signature(model)): raise ValueError(f"baseline decision names signature {decision['demucs_signature']}, registered model is {_signature(model)}")
    return decision

def dependency_entry(model: dict, *, role: str, selection: dict) -> dict:
    """Eligibility of ONE checkpoint the pipeline depends on; baseline may open only through a bound decision record."""
    eligibility = record_eligibility(model)
    base = {"model_id": model["model_id"], "sha256": model["sha256"], "role": role, "train_data_status": eligibility["train_data_status"],
            "clean_winner_eligible": eligibility["clean_winner_eligible"], "candidate_eligible": eligibility["candidate_eligible"]}
    if model.get("clean_winner_eligible") is not None and bool(model["clean_winner_eligible"]) != eligibility["clean_winner_eligible"]:
        raise ValueError(f"model {model['model_id']}: stored clean_winner_eligible disagrees with its training-data status/evidence")
    if eligibility["train_data_status"] == "verified-included": raise ValueError(f"model {model['model_id']} was trained on evaluation tracks (verified-included); refused")
    if eligibility["candidate_eligible"]:
        return {**base, "route": "verified-excluded" if eligibility["clean_winner_eligible"] else "documented-train-only",
                "label_ko": eligibility["label_ko"], "label_en": eligibility["label_en"], "decision": None}
    if role != "baseline": raise ValueError(f"candidate model {model['model_id']} is not eligible (status {eligibility['train_data_status']}); only verified-excluded or documented-train-only models may be used")
    if selection.get("baseline_train_data_policy") != "allow-unknown-with-disclosure":
        raise ValueError("baseline without verified training-data exclusion requires selection.baseline_train_data_policy=allow-unknown-with-disclosure")
    decision = check_baseline_decision(model, selection)
    signature = _signature(model)
    return {**base, "route": "user-decision", "label_ko": BASELINE_DISCLOSURE_KO.format(model_id=model["model_id"], signature=signature),
            "label_en": BASELINE_DISCLOSURE_EN.format(model_id=model["model_id"], signature=signature), "decision": decision}

def pipeline_dependencies(models_by_id: dict, baseline_id: str, spec_models: list[str], selection: dict) -> list[dict]:
    """Baseline plus the checkpoints one candidate uses, each checked on its own; unrelated registered models are ignored."""
    ids = [baseline_id] + [m for m in spec_models if m != baseline_id]
    out = []
    for model_id in ids:
        if model_id not in models_by_id: raise ValueError(f"model {model_id} is not registered in manifests/models.json")
        out.append(dependency_entry(models_by_id[model_id], role="baseline" if model_id == baseline_id else "candidate-model", selection=selection))
    return sorted(out, key=lambda e: e["model_id"])

def build_lock_content(config: dict, registry: dict, models_by_id: dict, baseline_id: str) -> dict:
    """Everything a person fixes before seeing dev scores. Raises if any depended-on checkpoint is not eligible."""
    selection = config.get("selection", {})
    if selection.get("sar_policy_status") != "confirmed": raise ValueError("selection policy lock blocked until the SAR policy status is confirmed")
    for key in ("sar_floor_db", "minimum_sdr_gain_db"):
        if isinstance(selection.get(key), bool) or not isinstance(selection.get(key), (int, float)): raise ValueError(f"selection.{key} must be numeric")
    candidates, depended = [], {baseline_id}
    for name, spec in registry.items():
        if name == "identity": continue
        models = sorted(set(spec["models"]))
        candidates.append({"name": name, "params": spec["params"], "config_hash": hashlib.sha256(canonical_json({"name": name, "params": spec["params"]})).hexdigest(),
                           "additional_inference_calls": spec["additional_inference_calls"], "requires": spec["requires"], "models": models})
        depended.update(models)
    for mid in depended:
        if mid not in models_by_id: raise ValueError(f"model {mid} is not registered in manifests/models.json")
    entries = {mid: dependency_entry(models_by_id[mid], role="baseline" if mid == baseline_id else "candidate-model", selection=selection) for mid in depended}
    baseline = entries[baseline_id]
    return {"schema_version": LOCK_SCHEMA,
            "adoption": {"minimum_sdr_gain_db": float(selection["minimum_sdr_gain_db"]), "sar_floor_db": float(selection["sar_floor_db"]),
                         "sar_policy_status": "confirmed", "rule": "sample-median gain >= minimum AND sample-median CI lower bound > 0 AND every stem's median dSI-SAR >= floor; none -> identity",
                         "ranking": RANKING},
            "statistics": {**STATISTIC_WORDS, "bootstrap": BOOTSTRAP, "seed": SEED, "ci_level": CI_LEVEL},
            "baseline": {"model_id": baseline_id, "sha256": baseline["sha256"], "eligibility": baseline},
            "candidates": sorted(candidates, key=lambda c: c["name"]),
            "models": [entries[k] for k in sorted(entries)],
            "selection_policy": selection}

def lock_path(runtime: Path) -> Path: return Path(runtime) / LOCK_REL

def read_lock(runtime: Path) -> dict:
    path = lock_path(runtime)
    if not path.is_file(): raise FileNotFoundError(f"selection policy lock missing: run lock-policy before develop ({path})")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("sha256") != lock_sha256(lock.get("content", {})): raise ValueError("selection policy lock hash mismatch (file edited)")
    return lock

def write_lock(runtime: Path, artifact: Path, content: dict) -> dict:
    """Create the lock; an existing lock with different content is never superseded."""
    path = lock_path(runtime); lock = {"schema_version": LOCK_SCHEMA, "content": content, "sha256": lock_sha256(content)}
    if path.is_file():
        existing = read_lock(runtime)
        if existing["content"] != content: raise ValueError(f"refusing to overwrite the selection policy lock {path} with different content (no silent supersede)")
        return existing
    atomic_json(path, lock); atomic_json(Path(artifact) / "evidence/manifests/selection-policy.json", lock)
    return lock

def verify_against(lock: dict, content: dict, *, what: str) -> None:
    if lock["content"] != content:
        changed = sorted(k for k in set(content) | set(lock["content"]) if content.get(k) != lock["content"].get(k))
        raise ValueError(f"{what}: current policy/candidate set/models differ from the pre-registered lock in {changed}")

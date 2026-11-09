"""Reproduction entry point.

Host-run order: prepare -> rirs -> build-wet -> register-model -> smoke -> lock-policy -> develop -> seal -> evaluate.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from . import contracts, policy as policy_lock, provenance, rir
from .contracts import cache_key, sha256_file, validate_practical_config, validate_runtime_path, validate_tensors
from .metrics import project_metrics, mixture_residual
from .data import ROOMS, atomic_json, build_wet_cache, load_wet, read_cache_entry, validate_acquired_dataset, wet_cache_key, write_cache_entry
from .candidates import candidate_registry, donor_reseparation, identity, t1, t2, t5, consistency_iteration, norbert_available, run_candidate
from .separators import (HTDemucsCUDA, MSSTSeparator, check_registration_allowed, declaration_record, load_registered_demucs,
                         model_record, official_demucs_reference, record_eligibility, state_dict_sha256)
from .selection import _median, claim_complete, claim_start, holm_adjust, paired_summary, seal_selection, select_candidate, validate_seal
from .diagnostics import oracle_ratio_mask

SEPARATOR_FACTORY = HTDemucsCUDA  # replaced by a CPU stub in tests; production is CUDA-only
MSST_FACTORY = MSSTSeparator
SEQUENCE = "prepare -> rirs -> build-wet -> register-model -> smoke -> lock-policy -> develop -> seal -> evaluate"
HELP = {
    "validate": "check the config and run the synthetic CPU self-test",
    "prepare": f"validate an extracted MUSDB18-HQ test tree and write manifests/dataset.json (step 1 of: {SEQUENCE})",
    "rirs": "acquire RIRs (gpuRIR fixed layout with --gpurir; else OpenAIR, public substitute, seeded synthetic) and write manifests/rirs.json",
    "build-wet": "write cache/wet/<split>/<track>/R1..R4 with complete.json from the dataset and RIR manifests",
    "register-model": "hash checkpoint files and record the model in manifests/models.json",
    "smoke": "CUDA baseline + donor forward (and the official-path parity check) on the first dev track, R4, first 5 or 30 s; --model smokes one registered model",
    "lock-policy": "write the pre-registration (criteria, statistics, exact candidate set, baseline eligibility) to manifests/selection-policy.json before develop",
    "develop": "run every locked practical candidate on the 10 dev tracks and write results/develop.json",
    "seal": "select a candidate from results/develop.json with the pre-registered rule and seal it",
    "evaluate": "run the sealed candidate once on the 40 evaluation tracks",
}

def _parser():
    p = argparse.ArgumentParser(prog="python -m experiments.banelp.cli", description=f"Host-run order: {SEQUENCE}")
    sub = p.add_subparsers(dest="command", required=True)
    for name, text in HELP.items():
        q = sub.add_parser(name, help=text, description=text)
        q.add_argument("--runtime-dir", required=True); q.add_argument("--artifact-dir", required=True); q.add_argument("--config", required=True)
        if name == "validate": q.add_argument("--synthetic", action="store_true")
        if name == "prepare": q.add_argument("--track-root", required=True, help="extracted MUSDB18-HQ test directory beneath --runtime-dir")
        if name == "rirs":
            q.add_argument("--sources", help="verified RIR source list (schema: experiments/banelp/rir_sources.example.json)")
            q.add_argument("--synthetic-fallback", action="store_true", help="use a seeded synthetic room when every listed source fails")
            q.add_argument("--offline-synthetic", action="store_true", help="skip downloads; all rooms synthetic")
            q.add_argument("--gpurir", metavar="GEOMETRY_JSON", help="generate all rooms with gpuRIR from this fixed-layout JSON (experiments/banelp/rir_geometry.json)")
            q.add_argument("--gpurir-build-record", metavar="JSON", help="toolchain build record written by bootstrap_gpurir.sh (required with --gpurir)")
        if name == "build-wet":
            q.add_argument("--split", choices=("all", "dev", "evaluation"), default="all"); q.add_argument("--limit", type=int)
        if name == "register-model":
            q.add_argument("--model-id", required=True); q.add_argument("--checkpoint", action="append", required=True, help="repeat for bag members")
            q.add_argument("--native-source-order", help="comma separated, e.g. drums,bass,other,vocals"); q.add_argument("--samplerate", type=int)
            q.add_argument("--train-data-status", required=True, choices=("verified-excluded", "verified-included", "unknown", "documented-train-only"))
            q.add_argument("--evidence", action="append", default=[]); q.add_argument("--url"); q.add_argument("--revision")
            q.add_argument("--introspect", action="store_true", help="demucs only: load the registered files through the resolved member set and take sources/samplerate/bag size from them")
            q.add_argument("--family", choices=("demucs", "msst"), default="demucs"); q.add_argument("--model-type", help="MSST model_type, e.g. bs_roformer or scnet")
            q.add_argument("--msst-config", help="MSST config yaml (hashed and recorded)"); q.add_argument("--msst-config-url"); q.add_argument("--msst-commit", help="pinned commit of the MSST checkout under $R/src/MSST")
            q.add_argument("--msst-batch-size", type=int, help="MSST only: recorded override of config.inference.batch_size (batching does not change eval-mode results); default is the config's value")
            q.add_argument("--declaration-url"); q.add_argument("--declaration-quote"); q.add_argument("--release", help="release tag of the distributed checkpoint")
        if name == "smoke":
            q.add_argument("--device", default="cuda"); q.add_argument("--require-cuda", action="store_true")
            q.add_argument("--split", default="dev", choices=("dev", "evaluation")); q.add_argument("--track-index", type=int, default=0)
            q.add_argument("--seconds", type=int, default=5); q.add_argument("--model", help="smoke this registered model id instead of the baseline (5 s and 30 s)")
            q.add_argument("--msst-batch-size", type=int, help="with --model <msst id>: batch size for this smoke only (recorded in the smoke file); the registration's value is unchanged")
        if name == "evaluate": q.add_argument("--seal", required=True)
    return p

def _load(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8")); validate_practical_config(config)
    runtime = Path(args.runtime_dir).resolve(); artifact = Path(args.artifact_dir).resolve()
    runtime.mkdir(parents=True, exist_ok=True); artifact.mkdir(parents=True, exist_ok=True)
    return config, runtime, artifact

def _now(): return datetime.now(timezone.utc).isoformat()
def _read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def _finite(value): return None if value is None or not np.isfinite(value) else float(value)

def _synthetic():
    rng = np.random.default_rng(20261108); n = 32768
    stems = rng.normal(0, .02, (4, 2, n)).astype(np.float32); x = stems.sum(axis=0, dtype=np.float64).astype(np.float32)
    validate_tensors(x, stems)
    target = stems[0]; exact = project_metrics(target, target, stems)
    outputs = {"identity": identity(x, stems), "T1": t1(x, stems), "T2": t2(x, stems), "T5": t5(x, stems), "C2-negative-control": consistency_iteration(x, stems, 0)}
    for name, out in outputs.items():
        validate_tensors(x, out)
        if not np.isfinite(out).all(): raise ValueError(f"{name} produced nonfinite output")
    if not np.allclose(outputs["identity"], stems): raise AssertionError("identity changed output")
    if exact["target_gain"] is None or abs(exact["target_gain"] - 1) > 1e-8: raise AssertionError("projection identity gain failed")
    return {"synthetic_samples": n, "outputs": list(outputs), "identity_projection": exact, "mixture_residual": mixture_residual(x, stems)}

def _smoke_input(mixture, seconds, sample_rate, samples=contracts.SAMPLES):
    """First `seconds` of the fixed manifest crop, so 5 s and 30 s smokes share one lineage."""
    if mixture.shape != (2, samples): raise ValueError(f"wet cache mixture must have shape (2,{samples}), got {mixture.shape}")
    return np.ascontiguousarray(mixture[:, :int(seconds * sample_rate)])

def _models(runtime):
    return _read_json(provenance.manifest_paths(runtime)["models"])

def _models_by_id(runtime):
    return {m["model_id"]: m for m in _models(runtime).get("models", [])}

def _baseline_model(runtime, config):
    name = config["separator"]["name"]
    matches = [m for m in _models(runtime).get("models", []) if m["model_id"] == name]
    if len(matches) != 1: raise ValueError(f"baseline model {name} must be registered exactly once in manifests/models.json")
    return matches[0]

class _LazySeparator:
    """Create the registered separator once, on first use."""
    def __init__(self, config, record, runtime=None, batch_size=None): self.config, self.record, self.runtime, self.batch_size, self.instance = config, record, runtime, batch_size, None
    def __call__(self):
        if self.instance is None:
            if self.record.get("family", "demucs") == "msst":
                kwargs = {} if self.batch_size is None else {"batch_size": self.batch_size}
                self.instance = MSST_FACTORY(registered=self.record, runtime_dir=self.runtime, **kwargs)
            else:
                sep = self.config["separator"]
                self.instance = SEPARATOR_FACTORY(model_name=self.record["model_id"], segment=sep.get("segment_seconds", 7.0), overlap=sep["overlap"], registered=self.record)
        return self.instance

def _context(config, runtime, separator_getter=None):
    """Registry from the config's model candidates, registered models by id, and the baseline record."""
    baseline = _baseline_model(runtime, config)
    registry = candidate_registry(separator_getter, model_candidates=config.get("selection", {}).get("model_candidates", []), baseline_id=baseline["model_id"])
    return registry, _models_by_id(runtime), baseline

def _lock_content(config, runtime):
    registry, models, baseline = _context(config, runtime)
    return policy_lock.build_lock_content(config, registry, models, baseline["model_id"])

def _verified_lock(config, runtime, *, what):
    """The lock file, hash-checked, equal to what the current config/registry/models produce."""
    lock = policy_lock.read_lock(runtime)
    policy_lock.verify_against(lock, _lock_content(config, runtime), what=what)
    return lock

def _inference_config(config, model):
    if model.get("family", "demucs") == "msst":
        return {"family": "msst", "model_type": model["model_type"], "config_sha256": model["config_file"]["sha256"], "msst_commit": model["msst_commit"],
                "adapter": "official-sequence-v2", "fp32": True, "normalization": "official-by-config"}
    return config["separator"]

def _keys(binding, config, model):
    samples = config["samples"]
    wet = wet_cache_key(code_hash=binding["code_hash"], lock_hash=binding["lock_hash"], manifest_hash=binding["manifest_hash"],
                        rir_manifest_hash=binding["rir_manifest_hash"], samples=samples)
    est = None if model is None else cache_key(code_revision=binding["code_hash"], lock_hash=binding["lock_hash"], source_manifest_hash=binding["manifest_hash"],
                                               rir_manifest_hash=binding["rir_manifest_hash"], model_hashes=[model["sha256"]],
                                               inference=_inference_config(config, model), synthesis={"wet_cache_key": wet})
    return wet, est

def _estimates(runtime, *, split, track, room, x, wet_record, key, model, separator):
    """Estimates of one registered model from a verified cache entry, regenerated (practical input only) when missing or stale."""
    root = validate_runtime_path(runtime / "cache/separation" / model["model_id"] / split / track / room, runtime)
    identity_fields = {"split": split, "track": track, "room": room, "mixture_sha256": wet_record["files"]["mixture"]["sha256"]}
    try:
        arrays, _ = read_cache_entry(root, ("estimates",), cache_key=key, identity=identity_fields)
        estimates = arrays["estimates"].astype(np.float32)
    except (FileNotFoundError, ValueError):
        estimates = np.asarray(separator()(x), dtype=np.float32)
        validate_tensors(x, estimates, samples=x.shape[-1])
        write_cache_entry(root, {"estimates": estimates}, {"schema_version": 1, "cache_key": key, "model_sha256": model["sha256"], **identity_fields})
    validate_tensors(x, estimates, samples=x.shape[-1])
    return estimates

def _db(metric): return np.nan if metric is None or metric["value_db"] is None else metric["value_db"]

METRICS = ("si_sdr", "si_sir", "si_sar")

def _metric_kwargs(config):
    metrics = config.get("metrics", {})
    return {"rcond": metrics.get("pinv_rcond", 1e-10), "energy_epsilon": metrics.get("energy_epsilon", 1e-12), "db_cap": metrics.get("db_cap", 120.0)}

def _new_scores(shape):
    return {"values": {m: np.full(shape, np.nan) for m in METRICS}, "gain": np.full(shape, np.nan),
            "states": {m: np.full(shape, "", dtype=object) for m in METRICS}, "state": np.full(shape, "", dtype=object)}

def _stem_metrics(estimates, targets, metric_kwargs=None):
    """Per stem in SOURCES order: dB values (NaN marks silent/undefined/infinite), target gain, and the reason for every NaN."""
    kwargs = metric_kwargs or {}
    out = {"values": {m: np.full(4, np.nan) for m in METRICS}, "gain": np.full(4, np.nan), "states": {m: [""] * 4 for m in METRICS}, "state": [""] * 4}
    for i in range(4):
        r = project_metrics(targets[i], estimates[i], targets, **kwargs)
        out["state"][i] = r["state"]
        if r["target_gain"] is not None: out["gain"][i] = r["target_gain"]
        for m in METRICS:
            if r[m] is None: out["states"][m][i] = r["state"]
            else: out["values"][m][i] = _db(r[m]); out["states"][m][i] = r[m]["state"]
    return out

def _store(scores, ti, cols, current):
    for m in METRICS:
        scores["values"][m][ti, cols] = current["values"][m]; scores["states"][m][ti, cols] = current["states"][m]
    scores["gain"][ti, cols] = current["gain"]; scores["state"][ti, cols] = current["state"]

def _observations(scores, tracks):
    """Every (track, room, stem) observation with the three metrics, gain and states, for rebuilding tables without rescoring."""
    rows = []
    for ti, track in enumerate(tracks):
        for ri, room in enumerate(ROOMS):
            for si, stem in enumerate(contracts.SOURCES):
                c = ri * 4 + si
                rows.append({"track": track, "room": room, "stem": stem, **{m: _finite(scores["values"][m][ti, c]) for m in METRICS},
                             "target_gain": _finite(scores["gain"][ti, c]), "state": str(scores["state"][ti, c]),
                             "metric_states": {m: str(scores["states"][m][ti, c]) for m in METRICS}})
    return rows

def _sample_mean(per_sample):
    """(tracks, rooms, 4) -> (tracks, rooms) mean over the defined stems; NaN where no stem is defined."""
    count = np.isfinite(per_sample).sum(axis=2)
    total = np.where(np.isfinite(per_sample), per_sample, 0.0).sum(axis=2)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)

def _stem_medians(values):
    out = {}
    for i, stem in enumerate(contracts.SOURCES):
        cell = values[:, :, i].reshape(-1); cell = cell[np.isfinite(cell)]
        out[stem] = _finite(np.median(cell)) if cell.size else None
    return out

def _improvement(delta):
    defined = np.isfinite(delta); n, k = int(defined.sum()), int((delta[defined] > 0).sum())
    return {"improved": k, "defined": n, "ratio": None if n == 0 else k / n}

def _reasons(states, defined):
    counts = {}
    for value, ok in zip(np.asarray(states).reshape(-1), np.asarray(defined).reshape(-1)):
        if not ok: counts[str(value) or "unknown"] = counts.get(str(value) or "unknown", 0) + 1
    return counts

def _baseline_summary(sdr, sir, sar):
    """Baseline per-stem sample medians (track x room) of the three metrics; enough for the chapter-3 baseline table."""
    out = {}
    for name, values in (("si_sdr", sdr), ("si_sir", sir), ("si_sar", sar)):
        cube = values.reshape(len(values), len(ROOMS), 4)
        out[name] = {"stem_median": _stem_medians(cube), "sample_median": _finite(_median(_sample_mean(cube).reshape(-1)))}
    return out

def _candidate_row(name, spec, sdr, sar, base_sdr, base_sar, *, sir=None, base_sir=None, states=None, base_states=None):
    """Fixed (tracks, room*stem) slots. An observation is dropped for every candidate when the baseline
    metric is undefined there; a candidate undefined where the baseline is defined is invalid.

    Gating numbers are sample-based: per (track, room) sample the mean dSI-SDR over the defined stems, then the median over
    all samples (`summary.sample_median_delta_db`); per-stem dSI-SAR medians are taken over the same track x room samples."""
    mask, sar_mask = np.isfinite(base_sdr), np.isfinite(base_sar)
    row = {"name": name, "params": spec["params"], "additional_inference_calls": spec["additional_inference_calls"],
           "config_hash": hashlib.sha256(contracts.canonical_json({"name": name, "params": spec["params"]})).hexdigest(),
           "observations": int(mask.sum())}
    if (np.isnan(sdr) & mask).any() or (np.isnan(sar) & sar_mask).any():
        return {**row, "status": "invalid-observations", "undefined_where_baseline_defined": int(((np.isnan(sdr) & mask) | (np.isnan(sar) & sar_mask)).sum())}
    if (~mask).all(axis=1).any(): return {**row, "status": "track-without-observations"}
    tracks = len(sdr)
    delta_sdr = np.where(mask, sdr - base_sdr, np.nan).reshape(tracks, len(ROOMS), 4)
    delta_sar = np.where(sar_mask, sar - base_sar, np.nan).reshape(tracks, len(ROOMS), 4)
    delta_sir = None if sir is None else np.where(np.isfinite(base_sir) & np.isfinite(sir), sir - base_sir, np.nan).reshape(tracks, len(ROOMS), 4)
    samples = _sample_mean(delta_sdr)
    if not np.isfinite(samples).any(axis=1).all(): return {**row, "status": "track-without-observations"}
    summary = paired_summary(samples)
    deltas = {"si_sdr": delta_sdr, "si_sar": delta_sar, **({} if delta_sir is None else {"si_sir": delta_sir})}
    detail = {"overall_sample_median": {m: _finite(_median(_sample_mean(d).reshape(-1))) for m, d in deltas.items()},
              "stem_median": {m: _stem_medians(d) for m, d in deltas.items()},
              "improvement_ratio": {m: _improvement(d) for m, d in deltas.items()}}
    if states is not None and base_states is not None:
        cand_values, base_values = {"si_sdr": sdr, "si_sar": sar, "si_sir": sir}, {"si_sdr": base_sdr, "si_sar": base_sar, "si_sir": base_sir}
        detail["undefined_by_reason"] = {m: {"candidate": _reasons(states[m], np.isfinite(cand_values[m])), "baseline": _reasons(base_states[m], np.isfinite(base_values[m]))}
                                         for m in METRICS if cand_values[m] is not None and base_values[m] is not None}
    return {**row, "status": "ok", "summary": summary, "wilcoxon_p": summary["wilcoxon_p"],
            "sample_median_delta_db": summary["sample_median_delta_db"], "delta_detail": detail,
            "stem_median_delta_sdr": detail["stem_median"]["si_sdr"], "stem_median_delta_sar": detail["stem_median"]["si_sar"],
            "stem_median_delta_sir": detail["stem_median"].get("si_sir")}

def _require_policy(config):
    policy = config.get("selection", {})
    if policy.get("sar_policy_status") != "confirmed": raise ValueError("selection seal and evaluation blocked until owner confirms SAR policy")
    if not isinstance(policy.get("sar_floor_db"), (int, float)): raise ValueError("confirmed SAR policy must set a numeric sar_floor_db")
    return policy

def _baseline_disclosure(model, policy):
    """What the seal and the results must say about the baseline's training data (eligibility recomputed from status + evidence)."""
    entry = policy_lock.dependency_entry(model, role="baseline", selection=policy)
    base = {"baseline_model_id": model["model_id"], "baseline_model_sha256": model["sha256"], "baseline_train_data_status": entry["train_data_status"]}
    if entry["route"] == "verified-excluded":
        return {**base, "clean_winner_eligible": True, "clean_winner_claim": "eligible", "route": entry["route"], "disclosure_ko": None, "disclosure_en": None, "decision": None}
    return {**base, "clean_winner_eligible": False, "clean_winner_claim": "not-eligible", "route": entry["route"],
            "disclosure_ko": entry["label_ko"], "disclosure_en": entry["label_en"], "decision": entry["decision"]}

def _dependencies(config, runtime, selected):
    """Eligibility of the checkpoints the sealed pipeline uses: the baseline and the selected candidate's models only."""
    registry, models, baseline = _context(config, runtime)
    if selected not in registry: raise ValueError(f"unsupported sealed candidate: {selected}")
    return policy_lock.pipeline_dependencies(models, baseline["model_id"], registry[selected]["models"], config.get("selection", {}))

def _lock_policy(args, config, runtime, artifact):
    _require_policy(config)
    content = _lock_content(config, runtime)
    lock = policy_lock.write_lock(runtime, artifact, content)
    return {"policy_lock": str(policy_lock.lock_path(runtime)), "sha256": lock["sha256"], "candidates": [c["name"] for c in content["candidates"]],
            "models": [{"model_id": m["model_id"], "route": m["route"]} for m in content["models"]], "next": "develop"}

def _develop(args, config, runtime, artifact):
    policy = _require_policy(config)
    lock = _verified_lock(config, runtime, what="develop refused")
    binding = provenance.current_binding(runtime, args.config)
    manifest = _read_json(provenance.manifest_paths(runtime)["dataset"])
    tracks = [t for t in manifest["tracks"] if t["split"] == "dev"]
    if len(tracks) != config["dev_tracks"]: raise ValueError(f"development requires exactly {config['dev_tracks']} dev tracks")
    model = _baseline_model(runtime, config); wet_key, est_key = _keys(binding, config, model)
    disclosure = _baseline_disclosure(model, policy)
    separator = _LazySeparator(config, model, runtime); registry, models, _ = _context(config, runtime, separator)
    status = {}
    for name, spec in registry.items():
        if spec["requires"] == "norbert" and not norbert_available(): status[name] = "unavailable-dependency:norbert"
    if any(s["requires"] == "separator" for s in registry.values()):
        try: separator()
        except Exception as exc:
            for name, spec in registry.items():
                if spec["requires"] == "separator": status[name] = f"unavailable:{type(exc).__name__}:{exc}"
    other_ids = sorted({m for s in registry.values() if s["requires"] == "model-estimates" for m in s["models"]} - {model["model_id"]})
    others = {mid: _LazySeparator(config, models[mid], runtime) for mid in other_ids}
    for mid, lazy in others.items():
        try: lazy()
        except Exception as exc:
            for name, spec in registry.items():
                if spec["requires"] == "model-estimates" and mid in spec["models"]: status[name] = f"unavailable:{mid}:{type(exc).__name__}:{exc}"
    practical = [n for n in registry if n != "identity"]
    active = [n for n in practical if n not in status]
    needed = [mid for mid in other_ids if any(mid in registry[n]["models"] for n in active)]
    slots = len(ROOMS) * 4; shape = (len(tracks), slots); metric_kwargs = _metric_kwargs(config)
    base = _new_scores(shape); scores = {n: _new_scores(shape) for n in active}; oracle = _new_scores(shape)
    for ti, track in enumerate(tracks):
        for ri, room in enumerate(ROOMS):
            x, targets, wet_record = load_wet(runtime, split="dev", track=track["name"], room=room, cache_key=wet_key, samples=config["samples"])
            estimates = _estimates(runtime, split="dev", track=track["name"], room=room, x=x, wet_record=wet_record, key=est_key, model=model, separator=separator)
            by_model = {model["model_id"]: estimates}
            for mid in needed:
                by_model[mid] = _estimates(runtime, split="dev", track=track["name"], room=room, x=x, wet_record=wet_record,
                                           key=_keys(binding, config, models[mid])[1], model=models[mid], separator=others[mid])
            cols = slice(ri * 4, ri * 4 + 4); _store(base, ti, cols, _stem_metrics(estimates, targets, metric_kwargs))
            for name in active:
                out = run_candidate(registry[name], x, estimates, by_model); validate_tensors(x, out)
                if not np.isfinite(out).all(): raise ValueError(f"{name} produced nonfinite output")
                _store(scores[name], ti, cols, _stem_metrics(out, targets, metric_kwargs))
            # Oracle results are diagnostics only and never enter candidate selection.
            _store(oracle, ti, cols, _stem_metrics(oracle_ratio_mask(x, estimates, targets, 2), targets, metric_kwargs))
    rows = []
    for name in practical:
        spec = registry[name]
        if name in status:
            rows.append({"name": name, "status": status[name], "params": spec["params"], "additional_inference_calls": spec["additional_inference_calls"]}); continue
        c = scores[name]
        rows.append(_candidate_row(name, spec, c["values"]["si_sdr"], c["values"]["si_sar"], base["values"]["si_sdr"], base["values"]["si_sar"],
                                   sir=c["values"]["si_sir"], base_sir=base["values"]["si_sir"], states=c["states"], base_states=base["states"]))
    ok = {r["name"]: r["wilcoxon_p"] for r in rows if r["status"] == "ok"}
    adjusted = holm_adjust(ok) if ok else {}
    for r in rows:
        if r["name"] in adjusted: r["holm_adjusted_p"] = adjusted[r["name"]]
    names = [t["name"] for t in tracks]
    oracle_rows = [{"track": o["track"], "room": o["room"], "stem": o["stem"], "si_sdr": o["si_sdr"], "si_sir": o["si_sir"], "si_sar": o["si_sar"],
                    "target_gain": o["target_gain"], "metric_states": o["metric_states"]} for o in _observations(oracle, names)]
    result = {"schema_version": 3, "split": "dev", "tracks": names, "binding": binding, "policy_lock_sha256": lock["sha256"], "baseline_disclosure": disclosure,
              "baseline_model_sha256": model["sha256"], "samples": config["samples"], "policy_at_develop": config.get("selection", {}),
              "metrics_config": config.get("metrics", {}),
              "observation_policy": "fixed room x stem slots per track; slots with undefined baseline metric dropped for all candidates",
              "baseline": {"observations": _observations(base, names), "summary": _baseline_summary(base["values"]["si_sdr"], base["values"]["si_sir"], base["values"]["si_sar"])},
              "candidate_observations": {n: _observations(scores[n], names) for n in active},
              "candidates": rows, "holm_adjusted_p": adjusted, "oracle_diagnostics": oracle_rows,
              "adapter_status": status, "selection_status": "not-sealed", "metric_contract": "four-source wet-target projection"}
    atomic_json(runtime / "results/develop.json", result); atomic_json(artifact / "evidence/develop.json", result)
    return {"develop": str(runtime / "results/develop.json"), "sha256": sha256_file(runtime / "results/develop.json"),
            "ok_candidates": sorted(ok), "unavailable": status, "policy_lock_sha256": lock["sha256"]}

def _seal(args, config, runtime, artifact):
    policy = _require_policy(config)
    binding = provenance.current_binding(runtime, args.config)
    develop_path = runtime / "results/develop.json"
    if not develop_path.is_file(): raise FileNotFoundError(f"run develop first: {develop_path}")
    develop = _read_json(develop_path)
    provenance.verify_binding(develop.get("binding", {}), binding, what="develop results are stale")
    lock = _verified_lock(config, runtime, what="seal refused")
    if develop.get("policy_lock_sha256") != lock["sha256"]: raise ValueError("seal refused: develop.json was produced under a different policy lock")
    model = _baseline_model(runtime, config)
    if develop.get("baseline_model_sha256") != model["sha256"]: raise ValueError("develop results used a different baseline model")
    disclosure = _baseline_disclosure(model, policy)
    rows = [r for r in develop["candidates"] if r["status"] == "ok"]
    selection = select_candidate(rows, sar_policy_status=policy["sar_policy_status"], sar_floor_db=float(policy["sar_floor_db"]),
                                 minimum_sdr_gain_db=float(policy.get("minimum_sdr_gain_db", .2)))
    selection["excluded"] = [{"name": r["name"], "status": r["status"]} for r in develop["candidates"] if r["status"] != "ok"]
    dependencies = _dependencies(config, runtime, selection["selected"])
    seal = seal_selection(selection, policy=policy, policy_lock=lock, develop_sha256=sha256_file(develop_path),
                          binding={k: v for k, v in binding.items() if k != "models_manifest_hash"},
                          model_hashes=dependencies, baseline_model_sha256=model["sha256"], baseline_disclosure=disclosure)
    atomic_json(artifact / "selection-seal.json", seal)
    return {"seal": str(artifact / "selection-seal.json"), "sha256": seal["sha256"], "selected": selection["selected"], "baseline_disclosure": disclosure,
            "policy_lock_sha256": lock["sha256"]}

def _evaluate(args, config, runtime, artifact):
    seal = _read_json(args.seal); validate_seal(seal, split="evaluation")
    policy = _require_policy(config)
    if policy != seal.get("policy"): raise ValueError("selection policy changed since the seal")
    binding = provenance.current_binding(runtime, args.config)
    provenance.verify_binding(seal, binding, what="sealed input changed")
    lock = _verified_lock(config, runtime, what="evaluate refused")
    if lock["sha256"] != seal.get("policy_lock_sha256") or lock["content"] != seal.get("policy_lock"): raise ValueError("evaluate refused: policy lock differs from the one sealed")
    model = _baseline_model(runtime, config)
    if model["sha256"] != seal.get("baseline_model_sha256"): raise ValueError("sealed baseline model changed")
    if _baseline_disclosure(model, policy) != seal.get("baseline_disclosure"): raise ValueError("sealed baseline disclosure changed")
    selected = seal["selection"]["selected"]
    if _dependencies(config, runtime, selected) != seal.get("model_hashes"): raise ValueError("sealed dependency models changed or are no longer eligible")
    separator = _LazySeparator(config, model, runtime); registry, models, _ = _context(config, runtime, separator)
    spec = registry[selected]
    if spec["requires"] == "norbert" and not norbert_available(): raise RuntimeError("sealed candidate needs norbert, which is not installed")
    extra = [m for m in spec["models"] if m != model["model_id"]]
    others = {mid: _LazySeparator(config, models[mid], runtime) for mid in extra}
    manifest = _read_json(provenance.manifest_paths(runtime)["dataset"])
    tracks = [t for t in manifest["tracks"] if t["split"] == "evaluation"]
    if len(tracks) != config["evaluation_tracks"]: raise ValueError(f"evaluation requires exactly {config['evaluation_tracks']} tracks, found {len(tracks)}")
    wet_key, est_key = _keys(binding, config, model); samples = config["samples"]; metric_kwargs = _metric_kwargs(config)
    def by_model_for(x, wet_record, track, room, estimates):
        by_model = {model["model_id"]: estimates}
        for mid in extra:
            by_model[mid] = _estimates(runtime, split="evaluation", track=track, room=room, x=x, wet_record=wet_record, key=_keys(binding, config, models[mid])[1], model=models[mid], separator=others[mid])
        return by_model
    # Preflight before the claim: every wet entry verified, every baseline estimate present, the candidate runnable.
    inputs = []
    for track in tracks:
        for room in ROOMS:
            x, _, wet_record = load_wet(runtime, split="evaluation", track=track["name"], room=room, cache_key=wet_key, samples=samples)
            estimates = _estimates(runtime, split="evaluation", track=track["name"], room=room, x=x, wet_record=wet_record, key=est_key, model=model, separator=separator)
            by_model_for(x, wet_record, track["name"], room, estimates)
            inputs.append((track["name"], room))
    probe = (np.random.default_rng(20261108).normal(0, .01, (2, samples))).astype(np.float32)
    probe_est = np.stack([probe / 4] * 4).astype(np.float32)
    probe_out = run_candidate(spec, probe, probe_est, {mid: probe_est for mid in [model["model_id"], *extra]})
    validate_tensors(probe, probe_out)
    if not np.isfinite(probe_out).all(): raise ValueError("sealed candidate produced nonfinite preflight output")
    claims = runtime / "claims"; claims.mkdir(parents=True, exist_ok=True)
    results_path = runtime / "results/evaluation.json"; claim_path = claims / "evaluation-40.claim"
    with open(claims / "evaluation-40.lock", "w") as lock_file:
        try: fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError("another evaluation run holds the holdout lock") from exc
        claim = claim_start(claim_path, seal_sha256=seal["sha256"], results_path=results_path, now=_now())
        if claim["action"] == "finalize":
            result = _read_json(results_path)
        else:
            records = []
            for track, room in inputs:
                x, targets, wet_record = load_wet(runtime, split="evaluation", track=track, room=room, cache_key=wet_key, samples=samples)
                estimates = _estimates(runtime, split="evaluation", track=track, room=room, x=x, wet_record=wet_record, key=est_key, model=model, separator=separator)
                out = run_candidate(spec, x, estimates, by_model_for(x, wet_record, track, room, estimates)); validate_tensors(x, out)
                before = {n: project_metrics(targets[i], estimates[i], targets, **metric_kwargs) for i, n in enumerate(contracts.SOURCES)}
                after = {n: project_metrics(targets[i], out[i], targets, **metric_kwargs) for i, n in enumerate(contracts.SOURCES)}
                records.append({"track": track, "room": room, "baseline": before, "selected": after, "mixture_residual": mixture_residual(x, out)})
            result = {"split": "evaluation", "selected": selected, "seal_sha256": seal["sha256"], "policy_lock_sha256": lock["sha256"], "baseline_disclosure": seal["baseline_disclosure"],
                      "model_hashes": seal["model_hashes"], "records": records, "tracks": len(tracks), "claim": str(claim_path), "claim_attempt": claim["attempt"]}
            atomic_json(results_path, result); atomic_json(artifact / "evidence/evaluation.json", result)
        claim_complete(claim_path, seal_sha256=seal["sha256"], results_sha256=sha256_file(results_path), now=_now())
    return result

def _smoke_model(args, config, runtime, artifact, *, binding, x, track, wet_key):
    """One registered model on the 5 s or 30 s input: peak VRAM, RTF, finiteness; a model that cannot run in 6 GB is recorded, not dropped."""
    import torch
    models = _models_by_id(runtime)
    if args.model not in models: raise ValueError(f"model {args.model} is not registered")
    record = models[args.model]; torch.cuda.reset_peak_memory_stats()
    result = {"model_id": args.model, "model_sha256": record["sha256"], "family": record.get("family", "demucs"), "split": args.split, "track": track, "condition": "R4",
              "seconds": args.seconds, "device_name": torch.cuda.get_device_name(), "dtype": "float32", "eligibility": record_eligibility(record),
              "binding": binding, "wet_cache_key": wet_key, "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
    try:
        start = time.perf_counter(); separator = _LazySeparator(config, record, runtime, batch_size=args.msst_batch_size)(); load_seconds = time.perf_counter() - start
        if hasattr(separator, "attestation"): result["msst_attestation"] = separator.attestation
        torch.cuda.synchronize(); t0 = time.perf_counter(); estimates = separator(x); torch.cuda.synchronize(); cold = time.perf_counter() - t0
        t0 = time.perf_counter(); separator(x); torch.cuda.synchronize(); warm = time.perf_counter() - t0
        result.update(status="ok", load_seconds=load_seconds, cold_forward_seconds=cold, warm_forward_seconds=warm, rtf_cold=cold / args.seconds, rtf_warm=warm / args.seconds,
                      shape=list(estimates.shape), finite=bool(np.isfinite(estimates).all()), output_sha256=hashlib.sha256(estimates.tobytes()).hexdigest())
        if not result["finite"]: raise ValueError("smoke produced nonfinite outputs")
    except Exception as exc:
        if not (isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()): raise
        result.update(status="excluded-cannot-run-in-6GB", error=f"{type(exc).__name__}: {exc}"[:2000])
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(artifact / f"reviews/smoke-{args.model}-{args.seconds}s.json", result)
    return result

def _smoke(args, config, runtime, artifact):
    if not args.require_cuda or args.device != "cuda": raise ValueError("smoke requires --device cuda --require-cuda")
    if args.seconds not in (5, 30): raise ValueError("smoke duration must be 5 or 30 seconds")
    binding = provenance.current_binding(runtime, args.config)
    manifest_path = provenance.manifest_paths(runtime)["dataset"]; manifest = _read_json(manifest_path)
    tracks = [t for t in manifest["tracks"] if t["split"] == args.split]
    if not 0 <= args.track_index < len(tracks): raise IndexError("track-index outside split")
    track = tracks[args.track_index]["name"]; model = _baseline_model(runtime, config); wet_key, _ = _keys(binding, config, model)
    x, _, _ = load_wet(runtime, split=args.split, track=track, room="R4", cache_key=wet_key, samples=config["samples"])
    x = _smoke_input(x, args.seconds, config["sample_rate"], config["samples"])
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; smoke failed without CPU fallback")
    if args.model: return _smoke_model(args, config, runtime, artifact, binding=binding, x=x, track=track, wet_key=wet_key)
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    separator = _LazySeparator(config, model, runtime)()
    load_seconds = time.perf_counter() - start
    torch.cuda.synchronize(); cold_start = time.perf_counter(); estimates = separator(x); torch.cuda.synchronize(); cold_seconds = time.perf_counter() - cold_start
    warm_start = time.perf_counter(); separator(x); torch.cuda.synchronize(); warm_seconds = time.perf_counter() - warm_start
    donor_start = time.perf_counter(); candidate = donor_reseparation(x, estimates, separator, donors=("other",), strength=.5); torch.cuda.synchronize()
    donor_seconds = time.perf_counter() - donor_start
    if not np.isfinite(estimates).all() or not np.isfinite(candidate).all(): raise ValueError("smoke produced nonfinite outputs")
    # Official-path parity: verbatim demucs separate.py:168-177 versus the wrapper on the same input and the same loaded model.
    with torch.inference_mode():
        audio = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to("cuda")
        official = official_demucs_reference(separator.model, audio, device="cuda", overlap=separator.overlap, segment=separator.segment)
    official = official[separator.source_order].detach().cpu().numpy().astype(np.float32)
    parity = {"reference": "demucs 4.0.1 separate.py:168-177 (verbatim copy: separators.official_demucs_reference)", "max_abs_diff": float(np.max(np.abs(official - estimates))),
              "normalization": separator.last_normalization, "std_zero_policy": "centre only, scale 1 (separators.STD_ZERO_POLICY)"}
    result = {"track": track, "split": args.split, "condition": "R4", "seconds": args.seconds, "device": str(torch.cuda.current_device()),
              "device_name": torch.cuda.get_device_name(), "input_device": "cuda:0", "output_device": "cuda:0", "dtype": "float32",
              "shape": list(estimates.shape), "load_seconds": load_seconds, "cold_forward_seconds": cold_seconds, "warm_forward_seconds": warm_seconds,
              "donor_forward_seconds": donor_seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "output_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
              "official_path_parity": parity,
              "model_state_sha256": state_dict_sha256(separator.model.state_dict()), "registered_model_sha256": model["sha256"],
              "dataset_manifest_sha256": binding["manifest_hash"], "binding": binding, "wet_cache_key": wet_key,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "code_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()}
    atomic_json(artifact / f"reviews/smoke-attestation-{args.seconds}s.json", result); atomic_json(artifact / "reviews/smoke-attestation.json", result)
    return result

def _register_model(args, config, runtime, artifact):
    native, samplerate, checkpoints = args.native_source_order, args.samplerate, [Path(p).resolve() for p in args.checkpoint]
    declaration = declaration_record(args.declaration_url, args.declaration_quote, args.release)
    config_file = Path(args.msst_config).resolve() if args.family == "msst" and args.msst_config else None
    if args.msst_batch_size is not None and (args.family != "msst" or args.msst_batch_size < 1): raise ValueError("--msst-batch-size is for MSST models and must be >= 1")
    check_registration_allowed(model_id=args.model_id, filenames=[p.name for p in checkpoints], status=args.train_data_status, release=args.release, family=args.family,
                               checkpoint_sha256={p.name: sha256_file(p) for p in checkpoints}, config_name=config_file.name if config_file else None,
                               config_sha256=sha256_file(config_file) if config_file and config_file.is_file() else None)
    extra = {"family": args.family, "declaration": declaration}
    if args.family == "msst":
        if args.introspect: raise ValueError("--introspect is for demucs checkpoints; give --native-source-order and --samplerate for MSST")
        if not (args.model_type and args.msst_config and args.msst_commit): raise ValueError("MSST registration needs --model-type, --msst-config and --msst-commit")
        config_path = Path(args.msst_config).resolve()
        if not config_path.is_file(): raise FileNotFoundError(config_path)
        extra.update(model_type=args.model_type, msst_commit=args.msst_commit,
                     config_file={"path": str(config_path), "sha256": sha256_file(config_path), "url": args.msst_config_url},
                     inference_overrides={"batch_size": args.msst_batch_size} if args.msst_batch_size is not None else None)
    elif args.introspect:
        provisional = {"model_id": args.model_id, "members": [{"path": str(p), "sha256": sha256_file(p)} for p in checkpoints]}
        loaded = load_registered_demucs(provisional, args.model_id)  # hash + resolved-member-set check, then load from exactly those files
        found = list(loaded.sources)
        if native and native.split(",") != found: raise ValueError(f"declared source order {native} differs from loaded {found}")
        if samplerate and samplerate != loaded.samplerate: raise ValueError("declared samplerate differs from loaded model")
        native, samplerate = ",".join(found), loaded.samplerate
    if not native or not samplerate: raise ValueError("give --native-source-order and --samplerate, or use --introspect")
    record = model_record(model_id=args.model_id, checkpoints=checkpoints, native_source_order=native.split(","), samplerate=samplerate,
                          train_data_status=args.train_data_status, training_data_evidence=args.evidence, url=args.url, revision=args.revision, **extra)
    record["introspected"] = bool(args.introspect)
    path = provenance.manifest_paths(runtime)["models"]
    models = _read_json(path) if path.is_file() else {"schema_version": 1, "models": []}
    models["models"] = sorted([m for m in models["models"] if m["model_id"] != args.model_id] + [record], key=lambda m: m["model_id"])
    atomic_json(path, models); atomic_json(artifact / "evidence/manifests/models.json", models)
    return {"models_manifest": str(path), "sha256": sha256_file(path), "model": record}

def main(argv=None):
    args = _parser().parse_args(argv); config, runtime, artifact = _load(args)
    if args.command == "validate":
        result = {"config_sha256": hashlib.sha256(contracts.canonical_json(config)).hexdigest(), "synthetic": _synthetic() if args.synthetic else None, "status": "PASS"}
    elif args.command == "prepare":
        path = provenance.manifest_paths(runtime)["dataset"]; manifest = validate_acquired_dataset(args.track_root, runtime, path)
        atomic_json(artifact / "evidence/manifests/dataset.json", manifest)
        result = {"tracks": len(manifest["tracks"]), "dev": sum(t["split"] == "dev" for t in manifest["tracks"]),
                  "evaluation": sum(t["split"] == "evaluation" for t in manifest["tracks"]), "dataset_manifest": str(path), "next": "rirs"}
    elif args.command == "rirs":
        if args.gpurir:
            if args.sources or args.synthetic_fallback or args.offline_synthetic: raise ValueError("--gpurir cannot be combined with --sources, --synthetic-fallback or --offline-synthetic")
            if not args.gpurir_build_record: raise ValueError("--gpurir needs --gpurir-build-record (run bootstrap_gpurir.sh first)")
            manifest = rir.acquire_rirs(runtime, sources=None, synthetic_fallback=False, manifest_path=provenance.manifest_paths(runtime)["rirs"],
                                        gpurir_geometry=rir.load_geometry(args.gpurir), gpurir_build_record=_read_json(args.gpurir_build_record))
        else:
            if args.gpurir_build_record: raise ValueError("--gpurir-build-record is only used with --gpurir")
            sources = None if args.offline_synthetic else rir.load_sources(args.sources)
            if sources is None and not (args.offline_synthetic or args.synthetic_fallback): raise ValueError("give --sources, --gpurir, or --offline-synthetic / --synthetic-fallback")
            manifest = rir.acquire_rirs(runtime, sources=sources, synthetic_fallback=args.synthetic_fallback or args.offline_synthetic,
                                        manifest_path=provenance.manifest_paths(runtime)["rirs"])
        atomic_json(artifact / "evidence/manifests/rirs.json", manifest)
        result = {"rir_manifest": str(provenance.manifest_paths(runtime)["rirs"]), "rooms": {r: v["kind"] for r, v in manifest["rooms"].items()},
                  "targets": {r: v["target_rt60_seconds"] for r, v in manifest["rooms"].items()},
                  "measured_t30": {r: {p: e["measured_t30_seconds"] for p, e in v["positions"].items()} for r, v in manifest["rooms"].items()},
                  "measured": {r: {p: {k: e.get(f"measured_{k}_seconds") for k in ("edt", "t20", "t30")} for p, e in v["positions"].items()} for r, v in manifest["rooms"].items()},
                  "next": "build-wet"}
    elif args.command == "build-wet":
        binding = provenance.current_binding(runtime, args.config, require=("dataset", "rirs"))
        paths = provenance.manifest_paths(runtime); rir_manifest = _read_json(paths["rirs"])
        wet_key, _ = _keys(binding, config, None)
        splits = ("dev", "evaluation") if args.split == "all" else (args.split,)
        summary = build_wet_cache(runtime, dataset=_read_json(paths["dataset"]), rirs=rir.load_rirs(runtime, rir_manifest), rir_manifest=rir_manifest,
                                  cache_key=wet_key, samples=config["samples"], splits=splits, limit=args.limit)
        result = {**summary, "wet_cache_key": wet_key, "next": "register-model"}
    elif args.command == "register-model": result = _register_model(args, config, runtime, artifact)
    elif args.command == "smoke": result = _smoke(args, config, runtime, artifact)
    elif args.command == "lock-policy": result = _lock_policy(args, config, runtime, artifact)
    elif args.command == "develop": result = _develop(args, config, runtime, artifact)
    elif args.command == "seal": result = _seal(args, config, runtime, artifact)
    else: result = _evaluate(args, config, runtime, artifact)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)); return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); raise SystemExit(2)

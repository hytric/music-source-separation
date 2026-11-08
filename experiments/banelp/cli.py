"""Reproduction entry point.

Host-run order: prepare -> rirs -> build-wet -> register-model -> smoke -> develop -> seal -> evaluate.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from . import contracts, provenance, rir
from .contracts import cache_key, sha256_file, validate_practical_config, validate_runtime_path, validate_tensors
from .metrics import project_metrics, mixture_residual
from .data import ROOMS, atomic_json, build_wet_cache, load_wet, read_cache_entry, validate_acquired_dataset, wet_cache_key, write_cache_entry
from .candidates import candidate_registry, donor_reseparation, identity, t1, t2, t5, consistency_iteration, norbert_available
from .separators import HTDemucsCUDA, model_record, state_dict_sha256
from .selection import claim_complete, claim_start, holm_adjust, paired_summary, seal_selection, select_candidate, validate_seal
from .diagnostics import oracle_ratio_mask

SEPARATOR_FACTORY = HTDemucsCUDA  # replaced by a CPU stub in tests; production is CUDA-only
SEQUENCE = "prepare -> rirs -> build-wet -> register-model -> smoke -> develop -> seal -> evaluate"
HELP = {
    "validate": "check the config and run the synthetic CPU self-test",
    "prepare": f"validate an extracted MUSDB18-HQ test tree and write manifests/dataset.json (step 1 of: {SEQUENCE})",
    "rirs": "acquire RIRs (OpenAIR, then public substitute, then seeded synthetic) and write manifests/rirs.json",
    "build-wet": "write cache/wet/<split>/<track>/R1..R4 with complete.json from the dataset and RIR manifests",
    "register-model": "hash checkpoint files and record the model in manifests/models.json",
    "smoke": "CUDA baseline + donor forward on the first dev track, R4, first 5 or 30 s",
    "develop": "run every practical candidate on the 10 dev tracks and write results/develop.json",
    "seal": "select a candidate from results/develop.json with the preregistered rule and seal it",
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
        if name == "build-wet":
            q.add_argument("--split", choices=("all", "dev", "evaluation"), default="all"); q.add_argument("--limit", type=int)
        if name == "register-model":
            q.add_argument("--model-id", required=True); q.add_argument("--checkpoint", action="append", required=True, help="repeat for bag members")
            q.add_argument("--native-source-order", help="comma separated, e.g. drums,bass,other,vocals"); q.add_argument("--samplerate", type=int)
            q.add_argument("--train-data-status", required=True, choices=("verified-excluded", "verified-included", "unknown"))
            q.add_argument("--evidence", action="append", default=[]); q.add_argument("--url"); q.add_argument("--revision")
            q.add_argument("--introspect", action="store_true", help="load the model with demucs and take sources/samplerate/bag size from it")
        if name == "smoke":
            q.add_argument("--device", default="cuda"); q.add_argument("--require-cuda", action="store_true")
            q.add_argument("--split", default="dev", choices=("dev", "evaluation")); q.add_argument("--track-index", type=int, default=0)
            q.add_argument("--seconds", type=int, default=5)
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

def _baseline_model(runtime, config):
    name = config["separator"]["name"]
    matches = [m for m in _models(runtime).get("models", []) if m["model_id"] == name]
    if len(matches) != 1: raise ValueError(f"baseline model {name} must be registered exactly once in manifests/models.json")
    return matches[0]

def _eligible_hashes(runtime):
    return sorted(m["sha256"] for m in _models(runtime).get("models", []) if m.get("clean_winner_eligible"))

class _LazySeparator:
    """Create the registered separator once, on first use."""
    def __init__(self, config, record): self.config, self.record, self.instance = config, record, None
    def __call__(self):
        if self.instance is None:
            sep = self.config["separator"]
            self.instance = SEPARATOR_FACTORY(model_name=sep["name"], segment=sep.get("segment_seconds", 7.0), overlap=sep["overlap"], registered=self.record)
        return self.instance

def _keys(binding, config, model):
    samples = config["samples"]
    wet = wet_cache_key(code_hash=binding["code_hash"], lock_hash=binding["lock_hash"], manifest_hash=binding["manifest_hash"],
                        rir_manifest_hash=binding["rir_manifest_hash"], samples=samples)
    est = None if model is None else cache_key(code_revision=binding["code_hash"], lock_hash=binding["lock_hash"], source_manifest_hash=binding["manifest_hash"],
                                               rir_manifest_hash=binding["rir_manifest_hash"], model_hashes=[model["sha256"]],
                                               inference=config["separator"], synthesis={"wet_cache_key": wet})
    return wet, est

def _estimates(runtime, *, split, track, room, x, wet_record, key, model, separator):
    """Baseline estimates from a verified cache entry, regenerated (practical input only) when missing or stale."""
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

def _stem_metrics(estimates, targets):
    """(si_sdr, si_sar) per stem in SOURCES order; NaN marks silent/undefined/infinite states."""
    out = np.full((4, 2), np.nan)
    for i in range(4):
        m = project_metrics(targets[i], estimates[i], targets)
        out[i] = (_db(m["si_sdr"]), _db(m["si_sar"]))
    return out

def _candidate_row(name, spec, sdr, sar, base_sdr, base_sar):
    """Fixed (tracks, room*stem) slots. An observation is dropped for every candidate when the baseline
    metric is undefined there; a candidate undefined where the baseline is defined is invalid."""
    mask, sar_mask = np.isfinite(base_sdr), np.isfinite(base_sar)
    row = {"name": name, "params": spec["params"], "additional_inference_calls": spec["additional_inference_calls"],
           "config_hash": hashlib.sha256(contracts.canonical_json({"name": name, "params": spec["params"]})).hexdigest(),
           "observations": int(mask.sum())}
    if (np.isnan(sdr) & mask).any() or (np.isnan(sar) & sar_mask).any():
        return {**row, "status": "invalid-observations", "undefined_where_baseline_defined": int(((np.isnan(sdr) & mask) | (np.isnan(sar) & sar_mask)).sum())}
    delta = np.where(mask, sdr - base_sdr, np.nan)
    if (~mask).all(axis=1).any(): return {**row, "status": "track-without-observations"}
    per_track = np.nanmean(delta, axis=1)
    summary = paired_summary(per_track[:, None])
    per_sample = delta.reshape(len(delta), len(ROOMS), 4)
    sample_means = np.nanmean(per_sample, axis=2) if np.isfinite(per_sample).any(axis=2).all() else None
    sar_delta = np.where(sar_mask, sar - base_sar, np.nan).reshape(len(delta), len(ROOMS), 4)
    sdr_stem = per_sample
    def stem_median(values, i):
        cell = values[:, :, i]; tracks = np.isfinite(cell).any(axis=1)
        return _finite(np.median(np.nanmean(cell[tracks], axis=1))) if tracks.any() else None
    return {**row, "status": "ok", "summary": summary, "wilcoxon_p": summary["wilcoxon_p"],
            "sample_median_delta_db": None if sample_means is None else _finite(np.median(sample_means)),
            "stem_median_delta_sdr": {s: stem_median(sdr_stem, i) for i, s in enumerate(contracts.SOURCES)},
            "stem_median_delta_sar": {s: stem_median(sar_delta, i) for i, s in enumerate(contracts.SOURCES)}}

def _develop(args, config, runtime, artifact):
    binding = provenance.current_binding(runtime, args.config)
    manifest = _read_json(provenance.manifest_paths(runtime)["dataset"])
    tracks = [t for t in manifest["tracks"] if t["split"] == "dev"]
    if len(tracks) != config["dev_tracks"]: raise ValueError(f"development requires exactly {config['dev_tracks']} dev tracks")
    model = _baseline_model(runtime, config); wet_key, est_key = _keys(binding, config, model)
    separator = _LazySeparator(config, model); registry = candidate_registry(separator)
    status = {}
    for name, spec in registry.items():
        if spec["requires"] == "norbert" and not norbert_available(): status[name] = "unavailable-dependency:norbert"
    if any(s["requires"] == "separator" for s in registry.values()):
        try: separator()
        except Exception as exc:
            for name, spec in registry.items():
                if spec["requires"] == "separator": status[name] = f"unavailable:{type(exc).__name__}:{exc}"
    practical = [n for n in registry if n != "identity"]
    active = [n for n in practical if n not in status]
    slots = len(ROOMS) * 4; shape = (len(tracks), slots)
    base_sdr, base_sar = np.full(shape, np.nan), np.full(shape, np.nan)
    sdr = {n: np.full(shape, np.nan) for n in active}; sar = {n: np.full(shape, np.nan) for n in active}
    oracle, baselines = [], []
    for ti, track in enumerate(tracks):
        for ri, room in enumerate(ROOMS):
            x, targets, wet_record = load_wet(runtime, split="dev", track=track["name"], room=room, cache_key=wet_key, samples=config["samples"])
            estimates = _estimates(runtime, split="dev", track=track["name"], room=room, x=x, wet_record=wet_record, key=est_key, model=model, separator=separator)
            cols = slice(ri * 4, ri * 4 + 4); base = _stem_metrics(estimates, targets)
            base_sdr[ti, cols], base_sar[ti, cols] = base[:, 0], base[:, 1]
            for name in active:
                out = registry[name]["fn"](x, estimates); validate_tensors(x, out)
                if not np.isfinite(out).all(): raise ValueError(f"{name} produced nonfinite output")
                current = _stem_metrics(out, targets); sdr[name][ti, cols], sar[name][ti, cols] = current[:, 0], current[:, 1]
            # Oracle results are diagnostics only and never enter candidate selection.
            oracle_db = _stem_metrics(oracle_ratio_mask(x, estimates, targets, 2), targets)[:, 0]
            oracle.append({"track": track["name"], "room": room, "si_sdr": {s: _finite(v) for s, v in zip(contracts.SOURCES, oracle_db)}})
        baselines.append({"track": track["name"], "mean_si_sdr": _finite(np.nanmean(base_sdr[ti])) if np.isfinite(base_sdr[ti]).any() else None})
    rows = []
    for name in practical:
        spec = registry[name]
        if name in status:
            rows.append({"name": name, "status": status[name], "params": spec["params"], "additional_inference_calls": spec["additional_inference_calls"]}); continue
        rows.append(_candidate_row(name, spec, sdr[name], sar[name], base_sdr, base_sar))
    ok = {r["name"]: r["wilcoxon_p"] for r in rows if r["status"] == "ok"}
    adjusted = holm_adjust(ok) if ok else {}
    for r in rows:
        if r["name"] in adjusted: r["holm_adjusted_p"] = adjusted[r["name"]]
    result = {"schema_version": 2, "split": "dev", "tracks": [t["name"] for t in tracks], "binding": binding,
              "baseline_model_sha256": model["sha256"], "samples": config["samples"], "policy_at_develop": config.get("selection", {}),
              "observation_policy": "fixed room x stem slots per track; slots with undefined baseline metric dropped for all candidates",
              "baseline": baselines, "candidates": rows, "holm_adjusted_p": adjusted, "oracle_diagnostics": oracle,
              "adapter_status": status, "selection_status": "not-sealed", "metric_contract": "four-source wet-target projection"}
    atomic_json(runtime / "results/develop.json", result); atomic_json(artifact / "evidence/develop.json", result)
    return {"develop": str(runtime / "results/develop.json"), "sha256": sha256_file(runtime / "results/develop.json"),
            "ok_candidates": sorted(ok), "unavailable": status}

def _require_policy(config):
    policy = config.get("selection", {})
    if policy.get("sar_policy_status") != "confirmed": raise ValueError("selection seal and evaluation blocked until owner confirms SAR policy")
    if not isinstance(policy.get("sar_floor_db"), (int, float)): raise ValueError("confirmed SAR policy must set a numeric sar_floor_db")
    return policy

def _seal(args, config, runtime, artifact):
    policy = _require_policy(config)
    binding = provenance.current_binding(runtime, args.config)
    develop_path = runtime / "results/develop.json"
    if not develop_path.is_file(): raise FileNotFoundError(f"run develop first: {develop_path}")
    develop = _read_json(develop_path)
    provenance.verify_binding(develop.get("binding", {}), binding, what="develop results are stale")
    model = _baseline_model(runtime, config)
    if develop.get("baseline_model_sha256") != model["sha256"]: raise ValueError("develop results used a different baseline model")
    eligible = _eligible_hashes(runtime)
    if not eligible: raise ValueError("no training-data-verified clean-winner checkpoint in model manifest")
    rows = [r for r in develop["candidates"] if r["status"] == "ok"]
    selection = select_candidate(rows, sar_policy_status=policy["sar_policy_status"], sar_floor_db=float(policy["sar_floor_db"]),
                                 minimum_sdr_gain_db=float(policy.get("minimum_sdr_gain_db", .2)))
    selection["excluded"] = [{"name": r["name"], "status": r["status"]} for r in develop["candidates"] if r["status"] != "ok"]
    seal = seal_selection(selection, policy=policy, develop_sha256=sha256_file(develop_path), binding=binding,
                          model_hashes=eligible, baseline_model_sha256=model["sha256"])
    atomic_json(artifact / "selection-seal.json", seal)
    return {"seal": str(artifact / "selection-seal.json"), "sha256": seal["sha256"], "selected": selection["selected"]}

def _evaluate(args, config, runtime, artifact):
    seal = _read_json(args.seal); validate_seal(seal, split="evaluation")
    policy = _require_policy(config)
    if policy != seal.get("policy"): raise ValueError("selection policy changed since the seal")
    binding = provenance.current_binding(runtime, args.config)
    provenance.verify_binding(seal, binding, what="sealed input changed")
    model = _baseline_model(runtime, config)
    if model["sha256"] != seal.get("baseline_model_sha256"): raise ValueError("sealed baseline model changed")
    if _eligible_hashes(runtime) != seal.get("model_hashes"): raise ValueError("sealed eligible model set changed")
    selected = seal["selection"]["selected"]
    separator = _LazySeparator(config, model); registry = candidate_registry(separator)
    if selected not in registry: raise ValueError(f"unsupported sealed candidate: {selected}")
    spec = registry[selected]
    if spec["requires"] == "norbert" and not norbert_available(): raise RuntimeError("sealed candidate needs norbert, which is not installed")
    manifest = _read_json(provenance.manifest_paths(runtime)["dataset"])
    tracks = [t for t in manifest["tracks"] if t["split"] == "evaluation"]
    if len(tracks) != config["evaluation_tracks"]: raise ValueError(f"evaluation requires exactly {config['evaluation_tracks']} tracks, found {len(tracks)}")
    wet_key, est_key = _keys(binding, config, model); samples = config["samples"]
    # Preflight before the claim: every wet entry verified, every baseline estimate present, the candidate runnable.
    inputs = []
    for track in tracks:
        for room in ROOMS:
            x, _, wet_record = load_wet(runtime, split="evaluation", track=track["name"], room=room, cache_key=wet_key, samples=samples)
            _estimates(runtime, split="evaluation", track=track["name"], room=room, x=x, wet_record=wet_record, key=est_key, model=model, separator=separator)
            inputs.append((track["name"], room))
    probe = (np.random.default_rng(20261108).normal(0, .01, (2, samples))).astype(np.float32)
    probe_out = spec["fn"](probe, np.stack([probe / 4] * 4).astype(np.float32))
    validate_tensors(probe, probe_out)
    if not np.isfinite(probe_out).all(): raise ValueError("sealed candidate produced nonfinite preflight output")
    claims = runtime / "claims"; claims.mkdir(parents=True, exist_ok=True)
    results_path = runtime / "results/evaluation.json"; claim_path = claims / "evaluation-40.claim"
    with open(claims / "evaluation-40.lock", "w") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError("another evaluation run holds the holdout lock") from exc
        claim = claim_start(claim_path, seal_sha256=seal["sha256"], results_path=results_path, now=_now())
        if claim["action"] == "finalize":
            result = _read_json(results_path)
        else:
            records = []
            for track, room in inputs:
                x, targets, wet_record = load_wet(runtime, split="evaluation", track=track, room=room, cache_key=wet_key, samples=samples)
                estimates = _estimates(runtime, split="evaluation", track=track, room=room, x=x, wet_record=wet_record, key=est_key, model=model, separator=separator)
                out = spec["fn"](x, estimates); validate_tensors(x, out)
                before = {n: project_metrics(targets[i], estimates[i], targets) for i, n in enumerate(contracts.SOURCES)}
                after = {n: project_metrics(targets[i], out[i], targets) for i, n in enumerate(contracts.SOURCES)}
                records.append({"track": track, "room": room, "baseline": before, "selected": after, "mixture_residual": mixture_residual(x, out)})
            result = {"split": "evaluation", "selected": selected, "seal_sha256": seal["sha256"], "records": records, "tracks": len(tracks),
                      "claim": str(claim_path), "claim_attempt": claim["attempt"]}
            atomic_json(results_path, result); atomic_json(artifact / "evidence/evaluation.json", result)
        claim_complete(claim_path, seal_sha256=seal["sha256"], results_sha256=sha256_file(results_path), now=_now())
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
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    separator = _LazySeparator(config, model)()
    load_seconds = time.perf_counter() - start
    torch.cuda.synchronize(); cold_start = time.perf_counter(); estimates = separator(x); torch.cuda.synchronize(); cold_seconds = time.perf_counter() - cold_start
    warm_start = time.perf_counter(); separator(x); torch.cuda.synchronize(); warm_seconds = time.perf_counter() - warm_start
    donor_start = time.perf_counter(); candidate = donor_reseparation(x, estimates, separator, donors=("other",), strength=.5); torch.cuda.synchronize()
    donor_seconds = time.perf_counter() - donor_start
    if not np.isfinite(estimates).all() or not np.isfinite(candidate).all(): raise ValueError("smoke produced nonfinite outputs")
    result = {"track": track, "split": args.split, "condition": "R4", "seconds": args.seconds, "device": str(torch.cuda.current_device()),
              "device_name": torch.cuda.get_device_name(), "input_device": "cuda:0", "output_device": "cuda:0", "dtype": "float32",
              "shape": list(estimates.shape), "load_seconds": load_seconds, "cold_forward_seconds": cold_seconds, "warm_forward_seconds": warm_seconds,
              "donor_forward_seconds": donor_seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "output_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
              "model_state_sha256": state_dict_sha256(separator.model.state_dict()), "registered_model_sha256": model["sha256"],
              "dataset_manifest_sha256": binding["manifest_hash"], "binding": binding, "wet_cache_key": wet_key,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "code_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()}
    atomic_json(artifact / "reviews/smoke-attestation.json", result)
    return result

def _register_model(args, config, runtime, artifact):
    native, samplerate, checkpoints = args.native_source_order, args.samplerate, [Path(p).resolve() for p in args.checkpoint]
    if args.introspect:
        try: from demucs.pretrained import get_model
        except ImportError as exc: raise RuntimeError("--introspect needs demucs in the research environment; run bootstrap.sh") from exc
        loaded = get_model(args.model_id)
        members = len(getattr(loaded, "models", [loaded]))
        if members != len(checkpoints): raise ValueError(f"{args.model_id} has {members} bag members but {len(checkpoints)} checkpoints were given")
        found = list(loaded.sources)
        if native and native.split(",") != found: raise ValueError(f"declared source order {native} differs from loaded {found}")
        if samplerate and samplerate != loaded.samplerate: raise ValueError("declared samplerate differs from loaded model")
        native, samplerate = ",".join(found), loaded.samplerate
    if not native or not samplerate: raise ValueError("give --native-source-order and --samplerate, or use --introspect")
    record = model_record(model_id=args.model_id, checkpoints=checkpoints, native_source_order=native.split(","), samplerate=samplerate,
                          train_data_status=args.train_data_status, training_data_evidence=args.evidence, url=args.url, revision=args.revision)
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
        sources = None if args.offline_synthetic else rir.load_sources(args.sources)
        if sources is None and not (args.offline_synthetic or args.synthetic_fallback): raise ValueError("give --sources, or --offline-synthetic / --synthetic-fallback")
        manifest = rir.acquire_rirs(runtime, sources=sources, synthetic_fallback=args.synthetic_fallback or args.offline_synthetic,
                                    manifest_path=provenance.manifest_paths(runtime)["rirs"])
        atomic_json(artifact / "evidence/manifests/rirs.json", manifest)
        result = {"rir_manifest": str(provenance.manifest_paths(runtime)["rirs"]), "rooms": {r: v["kind"] for r, v in manifest["rooms"].items()},
                  "measured_t30": {r: {p: e["measured_t30_seconds"] for p, e in v["positions"].items()} for r, v in manifest["rooms"].items()}, "next": "build-wet"}
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
    elif args.command == "develop": result = _develop(args, config, runtime, artifact)
    elif args.command == "seal": result = _seal(args, config, runtime, artifact)
    else: result = _evaluate(args, config, runtime, artifact)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)); return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); raise SystemExit(2)

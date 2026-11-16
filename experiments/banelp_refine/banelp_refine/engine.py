"""Readiness checks stop before optimizer updates; training is a separate command."""
from __future__ import annotations

from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .cache import CachedCrops, gpu_guard, producer_binding
from .common import SOURCES, atomic_json, digest, emit, environment, exclusive, read_json, sha256, source_hash, verify_entry
from .losses import ReconstructionLoss
from .model import JointRefiner, infer_record, project_mixture
from .contracts import validate_execution_config


def state_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode() + b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".writing")
    torch.save(payload, temp)
    temp.replace(path)


def current_binding(root, config):
    root = Path(root)
    cache = read_json(root / "manifests/cache.json")
    accelerator = None
    if torch.cuda.is_available():
        accelerator = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                       "cudnn": torch.backends.cudnn.version(), "name": torch.cuda.get_device_name(0),
                       "capability": list(torch.cuda.get_device_capability(0)),
                       "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()}
    return {"config_sha256": digest(config), "source_sha256": source_hash(), "environment": environment(), "accelerator": accelerator,
            "dataset_sha256": sha256(root / "manifests/dataset.json"),
            "rirs_sha256": sha256(root / "manifests/rirs.json"),
            "cache_manifest_sha256": sha256(root / "manifests/cache.json"),
            "cache_binding_sha256": cache["binding_sha256"]}


def validate_inventory(root, config):
    root = Path(root)
    data = read_json(root / "manifests/dataset.json")
    rirs = read_json(root / "manifests/rirs.json")
    cache = read_json(root / "manifests/cache.json")
    if len(cache["records"]) != config["readiness"]["required_cache_records"]:
        raise ValueError("not all preparation records exist")
    if len({r["id"] for r in cache["records"]}) != len(cache["records"]):
        raise ValueError("duplicate record IDs")
    songs, rooms, counts = {}, {}, {}
    for t in data["tracks"]:
        if t["name"] in songs:
            raise ValueError("duplicate song identity")
        songs[t["name"]] = t["split"]
        for entry in t["files"].values():
            verify_entry(root, entry)
    for r in rirs["sets"]:
        previous = rooms.setdefault(r["room_id"], r["split"])
        if previous != r["split"]:
            raise ValueError("RIR room leaked across splits")
        for entry in r["sources"].values():
            verify_entry(root, entry)
    rir_splits = {r["id"]: r["split"] for r in rirs["sets"]}
    per_song = {}
    for r in cache["records"]:
        if songs.get(r["track"]) != r["split"] or (r["rir_set"] and rir_splits.get(r["rir_set"]) != r["split"]):
            raise ValueError("song/RIR record split mismatch")
        counts[r["split"]] = counts.get(r["split"], 0) + 1
        per_song.setdefault(r["track"], []).append(r["rir_set"])
        arrays = {}
        for name, entry in r["files"].items():
            p = verify_entry(root, entry)
            arrays[name] = np.load(p, mmap_mode="r", allow_pickle=False)
        x, y, z = arrays["mixture"], arrays["targets"], arrays["estimates"]
        if y.shape != (4, 2, 1323000) or z.shape != y.shape or x.shape != (2, 1323000):
            raise ValueError("cached record shape mismatch")
        if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (x, y, z)):
            raise ValueError("cached data is nonfinite or not FP32")
        if np.max(np.abs(y.astype(np.float64).sum(axis=0).astype(np.float32) - x)) > 1e-7:
            raise ValueError("mixture and supervised stems are inconsistent")
    if any(len(v) != 4 or v.count(None) != 1 or len(set(v)) != 4 for v in per_song.values()):
        raise ValueError("each song needs one dry and three distinct wet records")
    return {"songs": len(songs), "rooms": len(rooms), "rir_sets": len(rirs["sets"]), "records": counts,
            "all_file_hashes_shapes_finiteness_and_sums_verified": True,
            "short_songs_zero_padded": sum(t["pad_left_frames"] + t["pad_right_frames"] > 0 for t in data["tracks"])}


def _move(sample, device):
    return {name: value[None].to(device) for name, value in sample.items()}


def preflight(root, config, baseline_repo, baseline_runtime, device="cuda"):
    validate_execution_config(config)
    root = Path(root)
    with exclusive(root, "preflight"):
        torch.set_num_threads(config["training"]["cpu_threads"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA readiness cannot use a silent CPU fallback")
        binding, _ = producer_binding(config, root, baseline_repo, baseline_runtime)
        if digest(binding) != read_json(root / "manifests/cache.json")["binding_sha256"]:
            raise ValueError("frozen separator producer binding changed")
        inventory = validate_inventory(root, config)
        torch.manual_seed(config["seed"])
        model = JointRefiner(config["model"]).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != 325024:
            raise ValueError(f"architecture differs from its design: {parameters}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                      weight_decay=config["training"]["weight_decay"])
        dataset = CachedCrops(root, config, "train")
        # A wet training record tests the actual intended reconstruction path.
        index = next(i for i, r in enumerate(dataset.records) if r["rir_set"] is not None)
        sample, sample_info = dataset.load(index, step=0)
        sample = _move(sample, device)
        before = state_hash(model)
        if device == "cuda":
            gpu_guard(config)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.monotonic()
        model.train()
        prediction = model(sample["mixture"], sample["estimates"])
        expected = project_mixture(sample["mixture"], sample["estimates"])
        identity_error = float((prediction - expected).abs().max().detach())
        sum_error = float((prediction.sum(dim=1) - sample["mixture"]).abs().max().detach())
        if identity_error > 1e-5 or sum_error > 1e-5 or not torch.isfinite(prediction).all():
            raise ValueError("initial identity/sum contract failed")
        loss_fn = ReconstructionLoss(config["loss"])
        loss, terms = loss_fn(prediction, sample["targets"], sample["mixture"])
        if not torch.isfinite(loss):
            raise ValueError("loss is nonfinite")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("nonfinite or missing gradients")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_gradient_norm"]))
        if gradient_norm <= 0 or before != state_hash(model) or optimizer.state:
            raise ValueError("preflight changed parameters/optimizer or produced no learning signal")
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        allocated = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        reserved = torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else None
        if reserved is not None and reserved > config["readiness"]["peak_reserved_gib_limit"] * 1024:
            raise RuntimeError("measured memory exceeds the configured readiness budget")
        optimizer.zero_grad(set_to_none=True)
        initial_validation = validate(model, CachedCrops(root, config, "validation"), loss_fn, device)
        if state_hash(model) != before or optimizer.state:
            raise ValueError("initial validation changed model or optimizer state")
        atomic_json(root / "initial/validation-step-000000.json", initial_validation)
        binding = current_binding(root, config)
        initial_path = root / "initial/step-000000.pt"
        if initial_path.exists():
            old = torch.load(initial_path, map_location="cpu", weights_only=True)
            if old["binding"] != binding or old["model_state_sha256"] != before:
                raise ValueError("initial checkpoint already exists with different inputs; use a new runtime")
        else:
            atomic_checkpoint(initial_path, {"schema_version": 1, "global_step": 0, "optimizer_steps": 0,
                              "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "binding": binding, "config": config, "model_state_sha256": before,
                              "torch_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []})
        # Reload the file that the future training command will actually consume.
        loaded = torch.load(initial_path, map_location="cpu", weights_only=True)
        restored = JointRefiner(config["model"])
        restored.load_state_dict(loaded["model"], strict=True)
        if state_hash(restored) != before or loaded["optimizer_steps"] != 0:
            raise ValueError("saved initial state failed readback")
        report = {"schema_version": 1, "status": "ready" if device == "cuda" else "cpu-check-only",
                  "binding": binding, "inventory": inventory, "model_parameters": parameters,
                  "device": device, "optimizer_steps": 0, "weights_unchanged": True,
                  "initial_state_sha256": before, "initial_checkpoint_sha256": sha256(initial_path),
                  "identity_max_abs": identity_error, "mixture_sum_max_abs": sum_error,
                  "loss_terms": {k: float(v.detach()) for k, v in terms.items()},
                  "gradient_norm_before_clipping": gradient_norm,
                  "forward_backward_seconds_cold": elapsed, "peak_allocated_mib": allocated,
                  "peak_reserved_mib": reserved, "sample": sample_info,
                  "initial_validation": {"records": len(initial_validation["rows"]), "mean_loss": initial_validation["mean_loss"],
                                         "median_delta_si_sdr_db": initial_validation["median_delta_si_sdr_db"],
                                         "scope": initial_validation["scope"]},
                  "training_speed_measured": False, "backward_checked_without_optimizer_step": True}
        atomic_json(root / "readiness.json", report)
        emit("preflight_complete", status=report["status"], optimizer_steps=0, parameters=parameters,
             peak_reserved_mib=reserved, checkpoint=str(initial_path))
        return report


def verify_ready(root, config, check_files=True):
    root = Path(root)
    report = read_json(root / "readiness.json")
    if report["status"] != "ready" or report["optimizer_steps"] != 0 or report["binding"] != current_binding(root, config):
        raise ValueError("readiness is missing, CPU-only or stale; run preflight again")
    if sha256(root / "initial/step-000000.pt") != report["initial_checkpoint_sha256"]:
        raise ValueError("initial checkpoint changed after readiness")
    if check_files:
        validate_inventory(root, config)
    return report


def _scores(target, prediction):
    """SI-SDR only, matching the legacy uncentered scale projection for validation."""
    s, z = target.double().flatten(2), prediction.double().flatten(2)
    norm = s.square().sum(dim=-1)
    gain = (s * z).sum(dim=-1) / norm.clamp_min(1e-24)
    t = gain.unsqueeze(-1) * s
    numerator, denominator = t.square().sum(dim=-1), (z - t).square().sum(dim=-1)
    floor = 1e-12 * torch.maximum(norm, z.square().sum(dim=-1))
    values = 10 * torch.log10(torch.maximum(numerator, floor).clamp_min(1e-30) / denominator.clamp_min(1e-30))
    valid = (norm > 1e-24) & (denominator > floor)
    values = torch.clamp(values, -120, 120)
    return values, valid


@torch.no_grad()
def validate(model, dataset, loss_fn, device):
    model.eval()
    losses, deltas, rows = [], [], []
    for index in range(len(dataset.records)):
        sample, info = dataset.load(index)
        sample = _move(sample, device)
        predicted = model(sample["mixture"], sample["estimates"])
        loss, _ = loss_fn(predicted, sample["targets"], sample["mixture"])
        after, valid_after = _scores(sample["targets"], predicted)
        before, valid_before = _scores(sample["targets"], sample["estimates"])
        valid = valid_after & valid_before
        ds = (after - before)[valid]
        delta = float(ds.mean()) if ds.numel() else None
        if delta is not None:
            deltas.append(delta)
        losses.append(float(loss))
        rows.append({"id": info["id"], "track": info["track"], "condition": info["condition"],
                     "mean_stem_delta_si_sdr_db": delta, "valid_stems": int(valid.sum()), "loss": float(loss)})
    if len(deltas) != len(rows) or not np.isfinite(losses).all():
        raise ValueError("validation has invalid records; retain results and inspect before selecting")
    return {"mean_loss": float(np.mean(losses)), "median_delta_si_sdr_db": float(np.median(deltas)),
            "rows": rows, "scope": "fixed_center_4s_validation_not_final_30s_evaluation"}


def train(root, config, *, start_training=False, resume=None, stop_after=None, device="cuda"):
    if not start_training:
        raise ValueError("training requires the separate explicit --start-training flag")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop-after must be positive")
    validate_execution_config(config)
    root = Path(root)
    with exclusive(root, "train"):
        report = verify_ready(root, config)
        torch.set_num_threads(config["training"]["cpu_threads"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the prepared training contract requires CUDA")
        gpu_guard(config)
        checkpoint = torch.load(resume or root / "initial/step-000000.pt", map_location="cpu", weights_only=True)
        if checkpoint["binding"] != report["binding"] or checkpoint["global_step"] != checkpoint["optimizer_steps"]:
            raise ValueError("checkpoint binding or step counters differ")
        run = root / "runs" / f"seed-{config['seed']}-{config['loss']['variant']}"
        if run.exists() and not resume:
            raise ValueError("run already exists; use --resume with its last.pt")
        run.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config["seed"])
        model = JointRefiner(config["model"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                      weight_decay=config["training"]["weight_decay"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["max_steps"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        training, validation = CachedCrops(root, config, "train"), CachedCrops(root, config, "validation")
        objective = ReconstructionLoss(config["loss"])
        best = checkpoint.get("best_validation", float("-inf"))
        misses = checkpoint.get("nonimproving_validations", 0)
        step = checkpoint["global_step"]
        ceiling = min(stop_after or config["training"]["max_steps"], config["training"]["max_steps"])
        if ceiling <= step:
            raise ValueError("stop-after must exceed the checkpoint's global step")
        while step < ceiling:
            model.train()
            rng = np.random.default_rng(config["seed"] + step)
            sample, info = training.load(int(rng.integers(len(training.records))), step=step)
            sample = _move(sample, device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(sample["mixture"], sample["estimates"])
            loss, terms = objective(predicted, sample["targets"], sample["mixture"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss; no update performed")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_gradient_norm"], error_if_nonfinite=True)
            optimizer.step()  # The ONLY parameter update path; never called by preparation/preflight.
            scheduler.step()
            step += 1
            if step % 20 == 0:
                gpu_guard(config)
                emit("training_step", step=step, loss=float(loss.detach()), gradient_norm=float(grad))
            if step % config["training"]["validate_every"] == 0 or step == ceiling:
                summary = validate(model, validation, objective, device)
                atomic_json(run / f"validation-{step:06d}.json", summary)
                score = summary["median_delta_si_sdr_db"]
                better = score > best
                best, misses = (score, 0) if better else (best, misses + 1)
                payload = {"schema_version": 1, "global_step": step, "optimizer_steps": step,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "binding": report["binding"], "config": config,
                           "best_validation": best, "nonimproving_validations": misses,
                           "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
                atomic_checkpoint(run / "last.pt", payload)
                if better:
                    atomic_checkpoint(run / "best.pt", payload)
                emit("validation_complete", step=step, score_db=score, best_db=best)
                if misses >= config["training"]["early_stop_validations"]:
                    break
        status = "complete" if step == config["training"]["max_steps"] else ("early-stopped" if misses >= config["training"]["early_stop_validations"] else "stopped-at-budget")
        atomic_json(run / "run-state.json", {"status": status, "optimizer_steps": step, "best_validation": best,
                    "config_sha256": digest(config), "checkpoint": str(run / "best.pt")})

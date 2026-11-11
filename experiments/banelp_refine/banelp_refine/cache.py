"""Hash-bound frozen separator outputs and reproducible supervised crops."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

from .common import SOURCES, atomic_json, baseline_import, digest, emit, environment, exclusive, file_entry, read_json, sha256, source_hash, verify_entry


def gpu_guard(config):
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True)
    if int(result.strip().splitlines()[0]) >= config["readiness"]["gpu_temperature_c_limit"]:
        raise RuntimeError("GPU temperature exceeds the configured preparation limit")


def producer_binding(config, root, baseline_repo, baseline_runtime):
    repo = baseline_import(baseline_repo)
    from experiments.banelp.separators import verify_model_record
    entries = read_json(Path(baseline_runtime) / "manifests/models.json")["models"]
    record = next(e for e in entries if e["model_id"] == config["baseline"]["model_id"])
    verify_model_record(record)
    if len(record["members"]) != 1 or config["baseline"]["signature"] not in Path(record["members"][0]["path"]).name:
        raise ValueError("unexpected baseline checkpoint signature")
    binding = {"dataset_sha256": sha256(Path(root) / "manifests/dataset.json"),
               "rirs_sha256": sha256(Path(root) / "manifests/rirs.json"), "baseline": record,
               "baseline_code": {p.name: sha256(p) for p in sorted((repo / "experiments/banelp").glob("*.py"))},
               "producer_code_sha256": source_hash(["common.py", "prepare.py", "rirs.py", "cache.py"]),
               "environment": environment(),
               "settings": {k: config[k] for k in ("audio", "data", "baseline")}}
    return binding, record


def plan_records(dataset, rirs, config):
    result = []
    pools = {split: [s for s in rirs["sets"] if s["split"] == split] for split in config["data"]["split_counts"]}
    for track in dataset["tracks"]:
        seed = int(hashlib.sha256((config["data"]["split_salt"] + "\0" + track["name"]).encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        count = config["data"]["conditions_per_song"] - 1
        if len(pools[track["split"]]) < count:
            raise ValueError("not enough independent RIR sets")
        selected = rng.choice(len(pools[track["split"]]), size=count, replace=False)
        for condition in [None] + [pools[track["split"]][int(i)]["id"] for i in selected]:
            rid = digest({"song": track["name"], "condition": condition})[:24]
            result.append({"id": rid, "track": track["name"], "split": track["split"],
                           "rir_set": condition, "condition": condition or "no_added_rir",
                           "active_start_frame": track["pad_left_frames"],
                           "active_end_frame": track["pad_left_frames"] + track["crop_frames"]})
    return result


def save_array(path, array):
    path = Path(path)
    with path.with_suffix(".part").open("wb") as f:
        np.save(f, np.asarray(array, dtype=np.float32), allow_pickle=False)
    path.with_suffix(".part").replace(path)


def make_cache(root, config, baseline_repo, baseline_runtime):
    baseline_import(baseline_repo)
    from experiments.banelp.data import apply_rir
    from experiments.banelp.separators import HTDemucsCUDA
    root = Path(root)
    with exclusive(root, "cache"):
        binding, model_record = producer_binding(config, root, baseline_repo, baseline_runtime)
        key = digest(binding)
        output = root / "manifests/cache.json"
        if output.exists():
            m = read_json(output)
            if m["binding_sha256"] != key:
                raise ValueError("cache producer inputs changed; use a new runtime")
            for item in m["records"]:
                for e in item["files"].values():
                    verify_entry(root, e)
            emit("cache_reused", records=len(m["records"]))
            return m
        dataset, rir_manifest = read_json(root / "manifests/dataset.json"), read_json(root / "manifests/rirs.json")
        tracks = {t["name"]: t for t in dataset["tracks"]}
        rir_sets = {}
        for item in rir_manifest["sets"]:
            rir_sets[item["id"]] = {name: np.load(verify_entry(root, e), allow_pickle=False) for name, e in item["sources"].items()}
        records = plan_records(dataset, rir_manifest, config)
        separator, last_track, dry = None, None, None
        complete_records = []
        started = time.monotonic()
        for index, row in enumerate(records):
            folder = root / "cache" / row["id"]
            complete_path = folder / "complete.json"
            if complete_path.exists():
                item = read_json(complete_path)
                if item["binding_sha256"] != key or item["record"] != row:
                    raise ValueError(f"cache key/identity changed: {row['id']}")
                for e in item["files"].values():
                    verify_entry(root, e)
                complete_records.append({**row, "files": item["files"]})
                continue
            if index % 10 == 0:
                gpu_guard(config)
            track = tracks[row["track"]]
            if last_track != row["track"]:
                dry = {}
                for name in SOURCES:
                    path = verify_entry(root, track["files"][name])
                    audio, rate = sf.read(path, start=track["crop_start_frame"], frames=track["crop_frames"],
                                          dtype="float32", always_2d=True)
                    if rate != 44100 or audio.shape != (track["crop_frames"], 2):
                        raise ValueError("invalid source crop")
                    dry[name] = np.pad(audio, ((track["pad_left_frames"], track["pad_right_frames"]), (0, 0)))
                last_track = row["track"]
            if row["rir_set"] is None:
                targets = np.stack([dry[name].T for name in SOURCES])
                gains = None
            else:
                processed = [apply_rir(dry[name], rir_sets[row["rir_set"]][name]) for name in SOURCES]
                targets = np.stack([v[0] for v in processed])
                gains = [v[1] for v in processed]
            mixture = targets.astype(np.float64).sum(axis=0).astype(np.float32)
            if separator is None:
                separator = HTDemucsCUDA(model_name=config["baseline"]["model_id"],
                                           segment=config["baseline"]["segment_seconds"],
                                           overlap=config["baseline"]["overlap"], registered=model_record)
            estimates = separator(mixture)
            if targets.shape != estimates.shape or not np.isfinite(estimates).all():
                raise ValueError("separator shape/finiteness failure")
            folder.mkdir(parents=True, exist_ok=True)
            files = {}
            for name, value in (("mixture", mixture), ("targets", targets), ("estimates", estimates)):
                p = folder / f"{name}.npy"
                save_array(p, value)
                files[name] = file_entry(p, root)
            item = {"binding_sha256": key, "record": row, "files": files, "gains": gains,
                    "baseline_normalization": separator.last_normalization}
            atomic_json(complete_path, item)
            complete_records.append({**row, "files": files})
            if (index + 1) % 5 == 0:
                emit("cache_progress", done=index + 1, total=len(records), elapsed_seconds=time.monotonic() - started)
        final_binding, _ = producer_binding(config, root, baseline_repo, baseline_runtime)
        if digest(final_binding) != key:
            raise ValueError("producer code/config changed during cache production")
        manifest = {"schema_version": 1, "binding": binding, "binding_sha256": key, "records": complete_records,
                    "elapsed_seconds_this_invocation": time.monotonic() - started}
        atomic_json(output, manifest)
        emit("cache_complete", records=len(records), optimizer_steps=0)
        return manifest


class CachedCrops:
    def __init__(self, root, config, split):
        self.root, self.config = Path(root), config
        self.records = [r for r in read_json(self.root / "manifests/cache.json")["records"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"no records for {split}")

    def load(self, index, *, step=None, full=False):
        row = self.records[index]
        frames = int(self.config["audio"]["record_seconds"] * 44100)
        length = frames if full else int(self.config["audio"]["crop_seconds"] * 44100)
        if full:
            start = 0
        elif step is None:
            start = (frames - length) // 2
        else:
            rng = np.random.default_rng(self.config["seed"] + step * 1009 + index)
            edge = int(self.config["audio"]["edge_seconds"] * 44100)
            active_start, active_end = row["active_start_frame"], row["active_end_frame"]
            margin = min(edge, max(0, (active_end - active_start - length) // 2))
            low = max(edge, active_start + margin)
            high = min(frames - edge, active_end - margin) - length
            start = int(rng.integers(low, high + 1)) if high >= low else max(0, (active_start + active_end - length) // 2)
        tensors = {}
        for name, entry in row["files"].items():
            p = self.root / entry["path"]
            value = np.load(p, mmap_mode="r", allow_pickle=False)
            expected = (2, frames) if name == "mixture" else (4, 2, frames)
            if value.shape != expected or value.dtype != np.float32:
                raise ValueError(f"invalid cached tensor: {p}")
            tensors[name] = torch.from_numpy(np.array(value[..., start:start + length], copy=True))
        return tensors, {**row, "crop_start_frame": start, "crop_frames": length}

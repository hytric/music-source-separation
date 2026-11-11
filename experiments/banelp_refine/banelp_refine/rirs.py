"""Separate, bounded gpuRIR pools with disjoint room identities."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .common import SOURCES, atomic_json, baseline_import, digest, emit, environment, exclusive, file_entry, read_json, verify_entry


def make_rirs(root, config, baseline_repo):
    baseline_import(baseline_repo)
    from experiments.banelp.rir import load_gpurir, GPURIR_REFERENCE
    from experiments.banelp.data import prepare_rir
    root = Path(root)
    settings = config["rir"]
    with exclusive(root, "rirs"):
        manifest_path = root / "manifests/rirs.json"
        if manifest_path.exists():
            m = read_json(manifest_path)
            if m["settings_sha256"] != digest(settings):
                raise ValueError("RIR settings changed; use a new runtime")
            for item in m["sets"]:
                for e in item["sources"].values():
                    verify_entry(root, e)
            emit("rirs_reused", sets=len(m["sets"]))
            return m
        sim, simulator_info = load_gpurir()
        sim.activateMixedPrecision(False)
        sim.activateLUT(True)
        rng = np.random.default_rng(settings["seed"])
        sets, rejected = [], []
        ref = GPURIR_REFERENCE
        for split, count in settings["pools"].items():
            for room_index in range(count["rooms"]):
                room_id = f"{split}-room-{room_index:02d}"
                room = np.array([rng.uniform(a, b) for a, b in settings["dimensions_m"]])
                for set_index in range(count["sets_per_room"]):
                    set_id = f"{room_id}-set-{set_index:02d}"
                    for attempt in range(settings["max_sampling_attempts"]):
                        t60 = float(rng.uniform(*settings["target_t60_s"]))
                        beta = np.asarray(sim.beta_SabineEstimation(room.tolist(), t60,
                                          abs_weights=np.array(ref["abs_weights"], dtype=np.float64)))
                        diff = float(sim.att2t_SabineEstimator(settings["attenuation_diffuse_db"], t60))
                        end = float(sim.att2t_SabineEstimator(settings["attenuation_end_db"], t60))
                        images = [int(v) for v in sim.t2n(diff, room.tolist())]
                        if np.isfinite(beta).all() and np.all((beta >= 0) & (beta < 1)) and math.prod(images) <= settings["max_image_product"]:
                            break
                        rejected.append({"set_id": set_id, "attempt": attempt, "t60": t60,
                                         "image_product": math.prod(images), "reason": "invalid-absorption-or-image-budget"})
                    else:
                        raise RuntimeError(f"no admissible simulation for {set_id}")
                    margin = settings["wall_margin_m"]
                    mic = rng.uniform(margin, room - margin)
                    positions = {}
                    for name in SOURCES:
                        for _ in range(1000):
                            pos = rng.uniform(margin, room - margin)
                            if np.linalg.norm(pos - mic) >= settings["minimum_distance_m"]:
                                positions[name] = pos.tolist()
                                break
                        else:
                            raise ValueError("could not place a separated source")
                    geometry = {"room_id": room_id, "room_m": room.tolist(), "mic_m": mic.tolist(),
                                "source_m": positions, "target_t60_s": t60, "beta": beta.tolist(),
                                "images": images, "tdiff_s": diff, "tmax_s": end}
                    geometry_hash = digest(geometry)
                    folder = root / "rirs" / set_id
                    complete = folder / "complete.json"
                    if complete.exists():
                        item = read_json(complete)
                        if item["geometry_sha256"] != geometry_hash:
                            raise ValueError(f"RIR geometry changed: {set_id}")
                        for e in item["sources"].values():
                            verify_entry(root, e)
                        sets.append(item)
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    sources = {}
                    for name, pos in positions.items():
                        result = np.asarray(sim.simulateRIR(room.tolist(), beta, np.array([pos]), mic[None], images,
                                            end, 44100, Tdiff=diff, spkr_pattern="omni", mic_pattern="omni", c=343.0))
                        if result.ndim != 3 or result.shape[:2] != (1, 1) or not np.isfinite(result).all():
                            raise ValueError(f"invalid gpuRIR output: {set_id}/{name}")
                        h, info = prepare_rir(result[0, 0].astype(np.float64), 44100)
                        peak = float(np.max(np.abs(h)))
                        if peak <= 0:
                            raise ValueError("silent RIR")
                        direct = int(round(np.linalg.norm(np.array(pos) - mic) / 343.0 * 44100))
                        onset = int(np.flatnonzero(np.abs(h) >= 0.1 * peak)[0])
                        if abs(onset - direct) > 8:
                            raise ValueError(f"onset disagrees with geometry: {set_id}/{name}: {onset}/{direct}")
                        path = folder / f"{name}.npy"
                        with path.with_suffix(".part").open("wb") as stream:
                            np.save(stream, h, allow_pickle=False)
                        path.with_suffix(".part").replace(path)
                        sources[name] = {**file_entry(path, root), "onset_samples": onset, "direct_samples": direct,
                                         "measurement": info}
                    item = {"id": set_id, "split": split, "room_id": room_id, "geometry": geometry,
                            "geometry_sha256": geometry_hash, "sources": sources}
                    atomic_json(complete, item)
                    sets.append(item)
                    emit("rir_set_complete", id=set_id, done=len(sets), t60=t60)
        m = {"schema_version": 1, "generator": "gpuRIR", "settings": settings, "settings_sha256": digest(settings),
             "simulator": simulator_info, "reference": ref, "environment": environment(),
             "sets": sets, "rejected_proposals": rejected,
             "sampling_disclosure": "uniform bounded proposals conditioned on valid absorption and image-product resource budget",
             "room_partition": "room identity and all variants remain within their split"}
        atomic_json(manifest_path, m)
        emit("rirs_complete", sets=len(sets), files=len(sets) * 4, rejected=len(rejected))
        return m

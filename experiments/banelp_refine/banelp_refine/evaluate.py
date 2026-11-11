"""Full-record legacy metrics, clustered effects and a checkpoint-bound holdout claim."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .cache import CachedCrops
from .common import SOURCES, atomic_json, baseline_import, emit, exclusive, read_json, sha256
from .engine import verify_ready
from .model import JointRefiner, infer_record


def cluster_summary(rows, replicates, seed):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["track"], []).append(r["mean_delta_db"])
    if not grouped or any(len(v) != 4 for v in grouped.values()):
        raise ValueError("clustered summary requires four fixed conditions per song")
    matrix = np.array([grouped[k] for k in sorted(grouped)], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("cannot silently drop an invalid candidate record")
    rng = np.random.default_rng(seed)
    ids = rng.integers(len(matrix), size=(replicates, len(matrix)))
    resampled = np.median(matrix[ids].reshape(replicates, -1), axis=1)
    return {"median_delta_db": float(np.median(matrix)), "song_cluster_bootstrap_95_ci_db": np.percentile(resampled, [2.5, 97.5]).tolist(),
            "songs": len(matrix), "conditions": int(matrix.size), "bootstrap_replicates": replicates,
            "scope": "song variability conditional on the fixed RIR list"}


def evaluate(root, config, baseline_repo, baseline_runtime, checkpoint_path, split="confirmation", include_ensemble=False):
    root = Path(root)
    repo = baseline_import(baseline_repo)
    from experiments.banelp.metrics import project_metrics
    from experiments.banelp.separators import MSSTSeparator
    with exclusive(root, "evaluate"):
        report = verify_ready(root, config)
        recorded_code = read_json(root / "manifests/cache.json")["binding"]["baseline_code"]
        if {p.name: sha256(p) for p in sorted((repo / "experiments/banelp").glob("*.py"))} != recorded_code:
            raise ValueError("legacy scoring/separator code changed after cache preparation")
        checkpoint_path = Path(checkpoint_path)
        checkpoint_sha = sha256(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if sha256(checkpoint_path) != checkpoint_sha:
            raise ValueError("checkpoint changed while loading; freeze a file before evaluation")
        if ckpt["binding"] != report["binding"] or ckpt["optimizer_steps"] < 1:
            raise ValueError("evaluation needs a trained, bound checkpoint")
        selection = {"checkpoint_sha256": checkpoint_sha, "binding": report["binding"],
                     "split": split, "include_ensemble": include_ensemble}
        claim_path = root / "claims" / f"{split}.json"
        destination = root / "evaluations" / f"{split}.json"
        if claim_path.exists():
            claim = read_json(claim_path)
            if claim["selection"] != selection:
                raise ValueError("this split was already exposed to a different checkpoint/evaluation choice")
            if claim["status"] == "complete":
                if sha256(destination) != claim["result_sha256"]:
                    raise ValueError("completed evaluation changed")
                return read_json(destination)
        atomic_json(claim_path, {"status": "running", "selection": selection})
        model = JointRefiner(config["model"]).cuda().eval()
        model.load_state_dict(ckpt["model"], strict=True)
        donor = None
        if include_ensemble:
            models = read_json(Path(baseline_runtime) / "manifests/models.json")["models"]
            donor_record = next(r for r in models if r["model_id"] == "msst_bsroformer_ep17")
            donor = MSSTSeparator(registered=donor_record, runtime_dir=baseline_runtime, device="cuda")
        dataset = CachedCrops(root, config, split)
        comparison_names = ["htdemucs", "mixture_projection"] + (["htdemucs_bsroformer_average"] if donor else [])
        results = {name: [] for name in comparison_names}
        started = time.monotonic()
        for index, row in enumerate(dataset.records):
            sample, info = dataset.load(index, full=True)
            x, z, y = (sample[k].numpy() for k in ("mixture", "estimates", "targets"))
            prediction = infer_record(model, sample["mixture"], sample["estimates"],
                                      int(config["audio"]["crop_seconds"] * 44100), "cuda").numpy()
            if not np.isfinite(prediction).all() or prediction.shape != y.shape:
                raise ValueError("refined output is nonfinite or misaligned")
            references = {"htdemucs": z, "mixture_projection": z + (x - z.sum(axis=0))[None] / 4}
            if donor:
                references["htdemucs_bsroformer_average"] = (z + donor(x)) / 2
            refined = {s: project_metrics(y[k], prediction[k], y) for k, s in enumerate(SOURCES)}
            for name, reference in references.items():
                base = {s: project_metrics(y[k], reference[k], y) for k, s in enumerate(SOURCES)}
                deltas = []
                for stem in SOURCES:
                    a, b = refined[stem], base[stem]
                    if a["state"] == b["state"] == "silent-target":
                        continue
                    av, bv = a["si_sdr"], b["si_sdr"]
                    if av is None or bv is None or av["value_db"] is None or bv["value_db"] is None:
                        raise ValueError(f"undefined evaluation metric: {info['id']}/{stem}; claim retained for inspection")
                    deltas.append(av["value_db"] - bv["value_db"])
                if not deltas:
                    raise ValueError("no defined evaluation stems")
                results[name].append({"id": info["id"], "track": info["track"], "condition": info["condition"],
                                      "mean_delta_db": float(np.mean(deltas)), "baseline": base, "refined": refined})
            emit("evaluation_progress", done=index + 1, total=len(dataset.records))
        summary = {name: cluster_summary(rows, config["evaluation"]["bootstrap_replicates"],
                                         config["evaluation"]["bootstrap_seed"]) for name, rows in results.items()}
        value = {"selection": selection, "summary": summary, "records": results,
                 "elapsed_seconds": time.monotonic() - started,
                 "disclosure": "refiner-held-out official train songs; frozen separator pretraining overlap unconfirmed"}
        atomic_json(destination, value)
        atomic_json(claim_path, {"status": "complete", "selection": selection, "result_sha256": sha256(destination)})
        return value

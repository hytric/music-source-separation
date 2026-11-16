"""Reject settings the implementation would otherwise silently ignore."""
from __future__ import annotations

import math


def validate_execution_config(config):
    expected_baseline = {"model_id": "htdemucs", "signature": "955717e8", "precision": "fp32", "shifts": 0,
                         "segment_seconds": 7.0, "overlap": 0.25}
    if config["baseline"] != expected_baseline:
        raise ValueError("this version requires the pinned frozen HTDemucs inference contract")
    fixed_stft = {"n_fft": 2048, "hop": 512, "center": True, "pad_mode": "constant", "hann_periodic": True}
    if any(config["model"].get(k) != v for k, v in fixed_stft.items()):
        raise ValueError("unsupported STFT settings; center/window/padding cannot be silently changed")
    if config["model"]["dilations"] != [1, 2, 4, 8, 16] or config["model"]["common_rms_floor"] <= 0:
        raise ValueError("unsupported model context or normalization")
    if not math.isfinite(config["model"]["residual_spectral_scale"]) or config["model"]["residual_spectral_scale"] <= 0:
        raise ValueError("residual scale must be finite and positive")
    if config["data"]["conditions_per_song"] != 4 or config["data"]["dry_conditions"] != 1:
        raise ValueError("this experiment prepares one dry and three distinct wet conditions")
    if config["data"]["window"] != "center-30s-or-symmetric-zero-pad":
        raise ValueError("unsupported crop/padding policy")
    if config["training"]["num_workers"] != 0:
        raise ValueError("this reproducible crop loader does not launch data workers")
    for k in ("learning_rate", "weight_decay", "clip_gradient_norm"):
        if not math.isfinite(config["training"][k]) or config["training"][k] < 0:
            raise ValueError(f"invalid optimizer setting: {k}")
    if config["training"]["learning_rate"] <= 0 or config["training"]["clip_gradient_norm"] <= 0:
        raise ValueError("learning rate and gradient clip must be positive")

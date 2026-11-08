"""Projection metrics on the four-source target subspace."""
from __future__ import annotations
import numpy as np

def _db_ratio(num: float, den: float) -> dict[str, object]:
    if num <= 1e-24 and den <= 1e-24: return {"value_db": None, "state": "undefined-zero-over-zero"}
    if den <= 1e-24: return {"value_db": None, "state": "positive-infinity"}
    raw = 10.0*np.log10(max(num, 1e-300)/den)
    return {"value_db": float(np.clip(raw, -120, 120)), "state": "finite" if abs(raw) <= 120 else "capped"}

def project_metrics(target: np.ndarray, estimate: np.ndarray, targets: np.ndarray, *, rcond: float = 1e-10) -> dict[str, object]:
    s, z = np.asarray(target, dtype=np.float64).reshape(-1), np.asarray(estimate, dtype=np.float64).reshape(-1)
    matrix = np.asarray(targets, dtype=np.float64).reshape((4, -1)).T
    if s.shape != z.shape or matrix.shape[0] != s.size: raise ValueError("metric tensor lengths differ")
    norm2 = float(s@s)
    if norm2 <= 1e-24: return {"state":"silent-target", "si_sdr":None,"si_sir":None,"si_sar":None,"sd_sdr":None,"target_gain":None,"rank":int(np.linalg.matrix_rank(matrix))}
    gain = float(z@s/norm2); t = gain*s
    # One thin SVD gives both the pinv(rcond) projection onto span(targets) and the rank.
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > rcond*singular[0] if singular.size else np.zeros(0, dtype=bool)
    rank = int(keep.sum()); basis = u[:, keep]
    p = basis @ (basis.T @ z)
    return {"state":"ok", "si_sdr":_db_ratio(float(t@t),float((z-t)@(z-t))),
            "si_sir":_db_ratio(float(t@t),float((p-t)@(p-t))), "si_sar":_db_ratio(float(p@p),float((z-p)@(z-p))),
            "sd_sdr":_db_ratio(float(t@t),float((z-s)@(z-s))), "target_gain":gain,"rank":rank}

def mixture_residual(x: np.ndarray, estimates: np.ndarray) -> dict[str, float]:
    r=np.asarray(estimates,dtype=np.float64).sum(axis=0)-np.asarray(x,dtype=np.float64)
    return {"l2":float(np.linalg.norm(r)),"rms":float(np.sqrt(np.mean(r*r)))}

"""Projection metrics on the four-source target subspace."""
from __future__ import annotations
import numpy as np

SILENT_TARGET_ENERGY = 1e-24  # absolute: a target below this has no defined gain (state silent-target)

def _db_ratio(num: float, den: float, *, floor: float, db_cap: float) -> dict[str, object]:
    """dB ratio with a relative energy floor: `floor` = energy_epsilon x reference energy.

    Both energies at or below the floor -> undefined (0/0); only the denominator -> positive infinity; a numerator
    below the floor is displayed at the floor. The original state is kept, `energy_floored` marks the display floor."""
    if num <= floor and den <= floor: return {"value_db": None, "state": "undefined-zero-over-zero", "energy_floored": False}
    if den <= floor: return {"value_db": None, "state": "positive-infinity", "energy_floored": False}
    floored = num <= floor
    raw = 10.0*np.log10(max(num, floor)/den)
    return {"value_db": float(np.clip(raw, -db_cap, db_cap)), "state": "finite" if abs(raw) <= db_cap else "capped", "energy_floored": bool(floored)}

def project_metrics(target: np.ndarray, estimate: np.ndarray, targets: np.ndarray, *, rcond: float = 1e-10,
                    energy_epsilon: float = 1e-12, db_cap: float = 120.0) -> dict[str, object]:
    s, z = np.asarray(target, dtype=np.float64).reshape(-1), np.asarray(estimate, dtype=np.float64).reshape(-1)
    matrix = np.asarray(targets, dtype=np.float64).reshape((4, -1)).T
    if s.shape != z.shape or matrix.shape[0] != s.size: raise ValueError("metric tensor lengths differ")
    norm2 = float(s@s)
    if norm2 <= SILENT_TARGET_ENERGY: return {"state":"silent-target", "si_sdr":None,"si_sir":None,"si_sar":None,"sd_sdr":None,"target_gain":None,"rank":int(np.linalg.matrix_rank(matrix))}
    gain = float(z@s/norm2); t = gain*s
    # One thin SVD gives both the pinv(rcond) projection onto span(targets) and the rank.
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > rcond*singular[0] if singular.size else np.zeros(0, dtype=bool)
    rank = int(keep.sum()); basis = u[:, keep]
    p = basis @ (basis.T @ z)
    floor = energy_epsilon*max(norm2, float(z@z))
    def ratio(num, den): return _db_ratio(float(num), float(den), floor=floor, db_cap=db_cap)
    return {"state":"ok", "si_sdr":ratio(t@t,(z-t)@(z-t)),
            "si_sir":ratio(t@t,(p-t)@(p-t)), "si_sar":ratio(p@p,(z-p)@(z-p)),
            "sd_sdr":ratio(t@t,(z-s)@(z-s)), "target_gain":gain,"rank":rank}

def mixture_residual(x: np.ndarray, estimates: np.ndarray) -> dict[str, float]:
    r=np.asarray(estimates,dtype=np.float64).sum(axis=0)-np.asarray(x,dtype=np.float64)
    return {"l2":float(np.linalg.norm(r)),"rms":float(np.sqrt(np.mean(r*r)))}

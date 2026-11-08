"""Predeclared dev selection contract; holdout remains sealed until policy approval."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from .contracts import canonical_json

def paired_summary(deltas: np.ndarray, *, seed: int=20261108, bootstrap: int=10000) -> dict:
    """Rows are tracks; columns are fixed space/stem observations."""
    a=np.asarray(deltas,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError("deltas must be a finite track-by-condition matrix")
    per_track=a.mean(axis=1); statistic=float(np.median(per_track))
    rng=np.random.default_rng(seed); boots=np.empty(bootstrap)
    for i in range(bootstrap): boots[i]=np.median(per_track[rng.integers(0,len(per_track),len(per_track))])
    try: p=float(wilcoxon(per_track,zero_method="wilcox",alternative="two-sided",method="approx",correction=False).pvalue) if np.any(per_track) else 1.0
    except ValueError: p=1.0
    return {"track_median_delta_db":statistic,"track_bootstrap_95_ci":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],"wilcoxon_p":p,"tracks":len(per_track)}

def seal_selection(selection: dict, *, runtime_dir: str|Path, manifest_hash: str, rir_manifest_hash: str,
                   model_hashes: list[str], code_hash: str, lock_hash: str, config_hash: str) -> dict:
    if selection.get("sar_policy_status")!="confirmed": raise ValueError("selection seal blocked: SAR policy requires owner confirmation")
    if not selection.get("selected"): raise ValueError("selection seal requires a selected development candidate")
    seal={"schema_version":1,"selection":selection,"manifest_hash":manifest_hash,"rir_manifest_hash":rir_manifest_hash,
          "model_hashes":sorted(model_hashes),"code_hash":code_hash,"lock_hash":lock_hash,"config_hash":config_hash,"split":"dev","sealed":True}
    seal["sha256"]=hashlib.sha256(canonical_json(seal)).hexdigest()
    return seal

def validate_seal(seal: dict, *, split: str) -> None:
    if not seal.get("sealed") or split!="evaluation": raise ValueError("evaluation requires a valid dev selection seal")
    content={k:v for k,v in seal.items() if k!="sha256"}
    if hashlib.sha256(canonical_json(content)).hexdigest()!=seal.get("sha256"): raise ValueError("selection seal hash mismatch")

def holm_adjust(p_values: dict[str,float]) -> dict[str,float]:
    """Holm step-down family-wise adjusted p values in deterministic key order."""
    ordered=sorted(p_values.items(),key=lambda item:(item[1],item[0])); m=len(ordered); adjusted={}; running=0.0
    for rank,(name,p) in enumerate(ordered):
        if not 0<=p<=1: raise ValueError(f"invalid p value for {name}")
        running=max(running,min(1.0,(m-rank)*p)); adjusted[name]=running
    return adjusted

def select_candidate(results: list[dict], *, sar_policy_status: str, sar_floor_db: float|None,
                     minimum_sdr_gain_db: float=.2) -> dict:
    """Apply the preregistered dev rule and deterministic tie breakers."""
    if sar_policy_status!="confirmed" or sar_floor_db is None:
        raise ValueError("candidate selection blocked until owner confirms per-stem SAR policy")
    practical=[r for r in results if not r.get("oracle",False) and r.get("name")!="identity"]
    adjusted=holm_adjust({r["name"]:float(r["wilcoxon_p"]) for r in practical})
    eligible=[]
    for row in practical:
        summary=row["summary"]
        sar=row.get("stem_median_delta_sar",{})
        passes=(summary["track_median_delta_db"]>=minimum_sdr_gain_db and summary["track_bootstrap_95_ci"][0]>0
                and all(float(sar.get(stem,float("-inf")))>=sar_floor_db for stem in ("vocals","drums","bass","other")))
        enriched={**row,"holm_adjusted_p":adjusted[row["name"]],"eligible":passes}
        if passes: eligible.append(enriched)
    if not eligible: return {"selected":"identity","reason":"no-practical-candidate-passed-preregistered-gates","holm_adjusted_p":adjusted}
    eligible.sort(key=lambda r:(-float(r["summary"]["track_median_delta_db"]),int(r.get("additional_inference_calls",0)),str(r.get("config_hash",""))))
    return {"selected":eligible[0]["name"],"reason":"maximum-development-improvement-after-gates","candidate":eligible[0],"holm_adjusted_p":adjusted}

def training_cost_record(*, training_data_bytes: int|None=None, optimizer: str|None=None,
                         batch_size: int|None=None, sequence_seconds: float|None=None,
                         activation_bytes: int|None=None, gradient_bytes: int|None=None,
                         measured_backward_seconds: float|None=None) -> dict:
    """Keep missing training costs explicitly unknown; inference is not a proxy."""
    fields={"training_data_bytes":training_data_bytes,"optimizer":optimizer,"batch_size":batch_size,
            "sequence_seconds":sequence_seconds,"activation_bytes":activation_bytes,
            "gradient_bytes":gradient_bytes,"measured_backward_seconds":measured_backward_seconds}
    return {"status":"measured" if all(v is not None for v in fields.values()) else "unmeasured",
            "inference_smoke_is_training_evidence":False,"fields":fields}

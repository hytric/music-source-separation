"""Predeclared dev selection contract; the evaluation split opens only through a sealed dev selection."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from .contracts import canonical_json

SEED, BOOTSTRAP, CI_LEVEL = 20261108, 10000, 0.95
STATISTIC_WORDS = {
    "effect_estimand": "median over dev track x room samples of the 4-stem mean dSI-SDR",
    "ci": f"{int(CI_LEVEL*100)}% percentile track-cluster bootstrap of the same statistic, B={BOOTSTRAP}, seed={SEED}",
    "sar_estimand": "per-stem median over dev track x room samples of dSI-SAR",
    "wilcoxon": "per-track mean of the sample values; two-sided, zero_method=wilcox, method=approx, correction=False; all-zero -> p=1",
    "holm_family": "practical candidates in the lock versus baseline; identity and oracle excluded",
    "non_gating_fields": "track_median_delta_db and track_bootstrap_95_ci (median of per-track means) are reported only",
}

def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")

def paired_summary(samples: np.ndarray, *, seed: int=SEED, bootstrap: int=BOOTSTRAP) -> dict:
    """`samples`: (tracks, rooms) per-sample 4-stem-mean delta; NaN marks a dropped sample.

    Gating statistic = median over all finite samples. The CI resamples whole tracks (all their rooms) with replacement and
    recomputes that same median. The per-track-mean median (`track_*`) is a non-gating report field."""
    a=np.asarray(samples,dtype=np.float64)
    if a.ndim!=2 or np.isinf(a).any(): raise ValueError("samples must be a (tracks, rooms) matrix without infinities")
    if not np.isfinite(a).any(axis=1).all(): raise ValueError("every track needs at least one defined sample")
    tracks=a.shape[0]; counts=np.isfinite(a).sum(axis=1)
    per_track=np.where(np.isfinite(a),a,0.0).sum(axis=1)/counts
    rng=np.random.default_rng(seed); idx=rng.integers(0,tracks,(bootstrap,tracks))
    drawn=a[idx].reshape(bootstrap,-1)
    with np.errstate(all="ignore"):
        boots=np.nanmedian(drawn,axis=1); track_boots=np.median(per_track[idx],axis=1)
    lo,hi=(1-CI_LEVEL)/2,1-(1-CI_LEVEL)/2
    try: p=float(wilcoxon(per_track,zero_method="wilcox",alternative="two-sided",method="approx",correction=False).pvalue) if np.any(per_track) else 1.0
    except ValueError: p=1.0
    return {"sample_median_delta_db":_median(a.reshape(-1)),"sample_bootstrap_95_ci":[float(np.quantile(boots,lo)),float(np.quantile(boots,hi))],
            "track_median_delta_db":float(np.median(per_track)),"track_bootstrap_95_ci":[float(np.quantile(track_boots,lo)),float(np.quantile(track_boots,hi))],
            "wilcoxon_p":p,"tracks":tracks,"samples":int(np.isfinite(a).sum()),"statistic_words":STATISTIC_WORDS}

def seal_selection(selection: dict, *, policy: dict, policy_lock: dict, develop_sha256: str, binding: dict, model_hashes: list[dict],
                   baseline_model_sha256: str, baseline_disclosure: dict) -> dict:
    """Bind a `select_candidate` result to the dev results, the pre-registered lock and every input hash it depends on."""
    if policy.get("sar_policy_status")!="confirmed": raise ValueError("selection seal blocked: SAR policy requires owner confirmation")
    if not selection.get("selected") or not selection.get("reason"): raise ValueError("selection seal requires a select_candidate result")
    seal={"schema_version":4,"policy":policy,"policy_lock":policy_lock["content"],"policy_lock_sha256":policy_lock["sha256"],
          "statistic":STATISTIC_WORDS,"selection":selection,"develop_sha256":develop_sha256,**binding,
          "model_hashes":sorted(model_hashes,key=lambda m:m["model_id"]),"baseline_model_sha256":baseline_model_sha256,
          "baseline_disclosure":baseline_disclosure,"split":"dev","sealed":True}
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
    """Apply the pre-registered dev rule: sample-median gain >= minimum, sample-median CI lower bound > 0 and every
    stem's median dSI-SAR >= floor (an undefined stem fails). Rank by -sample median, extra inference calls, config hash."""
    if sar_policy_status!="confirmed" or sar_floor_db is None:
        raise ValueError("candidate selection blocked until owner confirms per-stem SAR policy")
    practical=[r for r in results if not r.get("oracle",False) and r.get("name")!="identity"]
    adjusted=holm_adjust({r["name"]:float(r["wilcoxon_p"]) for r in practical})
    eligible=[]
    for row in practical:
        summary=row["summary"]
        sar=row.get("stem_median_delta_sar",{})
        gain_ok=summary["sample_median_delta_db"] is not None and summary["sample_median_delta_db"]>=minimum_sdr_gain_db
        ci_ok=summary["sample_bootstrap_95_ci"][0]>0
        sar_ok=all(sar.get(stem) is not None and float(sar[stem])>=sar_floor_db for stem in ("vocals","drums","bass","other"))
        enriched={**row,"holm_adjusted_p":adjusted[row["name"]],"eligible":gain_ok and ci_ok and sar_ok,"gates":{"sample_median_gain":gain_ok,"sample_ci_lower_gt_0":ci_ok,"stem_sar_floor":sar_ok}}
        if enriched["eligible"]: eligible.append(enriched)
    if not eligible: return {"selected":"identity","reason":"no-practical-candidate-passed-preregistered-gates","holm_adjusted_p":adjusted}
    eligible.sort(key=lambda r:(-float(r["summary"]["sample_median_delta_db"]),int(r.get("additional_inference_calls",0)),str(r.get("config_hash",""))))
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

CLAIM_RETRY_RULE=("A claim stays `started` until results are written. A `started` claim for the same seal may be retried: "
                  "an interrupted run (OOM, corrupt cache, crash) persisted no holdout metrics, so it is not a look at results. "
                  "If results for that seal were already written, the retry only finalizes the claim and never recomputes. "
                  "A `completed` claim, or any claim bound to a different seal, is refused.")

def _write_claim(path: Path, claim: dict) -> None:
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent); os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(claim,sort_keys=True,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,path)
    finally: Path(tmp).unlink(missing_ok=True)

def claim_start(path: str|Path, *, seal_sha256: str, results_path: str|Path, now: str) -> dict:
    """Create or resume the single holdout claim; returns the claim with `action` run or finalize."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    claim={"seal_sha256":seal_sha256,"status":"started","attempt":1,"history":[{"attempt":1,"started_at":now}],"retry_rule":CLAIM_RETRY_RULE}
    try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        claim=json.loads(path.read_text(encoding="utf-8"))
        if claim.get("seal_sha256")!=seal_sha256: raise RuntimeError("holdout already claimed by a different seal; refusing")
        if claim.get("status")=="completed": raise RuntimeError("evaluation seal already completed; refusing duplicate holdout run")
        if claim.get("status")!="started": raise RuntimeError(f"unknown claim status {claim.get('status')!r}; refusing")
        results=Path(results_path)
        finalize=results.is_file() and json.loads(results.read_text(encoding="utf-8")).get("seal_sha256")==seal_sha256
        claim["attempt"]+=1; claim["history"].append({"attempt":claim["attempt"],"started_at":now,"action":"finalize" if finalize else "retry"})
        _write_claim(path,claim)
        return {**claim,"action":"finalize" if finalize else "run"}
    with os.fdopen(fd,"w",encoding="utf-8") as stream: stream.write(json.dumps(claim,sort_keys=True,indent=2)+"\n")
    return {**claim,"action":"run"}

def claim_complete(path: str|Path, *, seal_sha256: str, results_sha256: str, now: str) -> dict:
    path=Path(path); claim=json.loads(path.read_text(encoding="utf-8"))
    if claim.get("seal_sha256")!=seal_sha256 or claim.get("status")!="started": raise RuntimeError("claim is not an open claim for this seal")
    claim.update(status="completed",completed_at=now,results_sha256=results_sha256)
    _write_claim(path,claim)
    return claim

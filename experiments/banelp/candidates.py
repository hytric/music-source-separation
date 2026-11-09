"""Practical four-stem candidate adapters; no target input is accepted."""
from __future__ import annotations
import numpy as np
from scipy.signal import get_window, istft, stft
from .contracts import STFT_CONTRACT, validate_tensors
NFFT,HOP=STFT_CONTRACT["nperseg"],STFT_CONTRACT["hop"]

def _stft(audio: np.ndarray) -> np.ndarray:
    audio=np.asarray(audio)
    channels=audio[None,:] if audio.ndim==1 else audio
    if channels.ndim!=2: raise ValueError("STFT input must be (samples,) or (channels,samples)")
    win=get_window("hann",NFFT,fftbins=True)
    return np.stack([stft(ch,fs=44100,window=win,nperseg=NFFT,noverlap=NFFT-HOP,nfft=NFFT,
        boundary="zeros",padded=True,return_onesided=True,scaling="spectrum")[2] for ch in channels])

def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    """Invert (channels,freq,frames) or (sources,channels,freq,frames) spectrograms."""
    spec=np.asarray(spec)
    if spec.ndim not in (3,4): raise ValueError("ISTFT input must be (channels,freq,frames) or (sources,channels,freq,frames)")
    if spec.ndim==4: return np.stack([_istft(src,length) for src in spec])
    win=get_window("hann",NFFT,fftbins=True)
    channels=[]
    for ch in spec:
        _,wave=istft(ch,fs=44100,window=win,nperseg=NFFT,noverlap=NFFT-HOP,nfft=NFFT,
            input_onesided=True,boundary=True,scaling="spectrum")
        if len(wave)<length: raise ValueError(f"ISTFT shorter than requested length: {len(wave)} < {length}")
        channels.append(wave[:length])
    return np.asarray(channels)

def identity(x, estimates, config=None, separator=None):
    validate_tensors(x,estimates); return estimates.copy()

def mixture_consistency(x, estimates, weights="uniform"):
    validate_tensors(x,estimates); residual=x-estimates.sum(axis=0)
    if weights=="uniform": w=np.full(4,.25)
    elif weights=="energy":
        energy=np.sum(estimates.astype(np.float64)**2,axis=(1,2)); w=energy/energy.sum() if energy.sum() else np.full(4,.25)
    else: raise ValueError("weights must be uniform or energy")
    out=estimates+w[:,None,None]*residual[None]
    if not np.isfinite(out).all(): raise ValueError("nonfinite mixture consistency output")
    return out

def ratio_mask(x, estimates, alpha=2.0, *, psd=None):
    validate_tensors(x,estimates)
    if alpha<=0: raise ValueError("alpha must be positive")
    X=_stft(x); S=np.stack([_stft(src) for src in estimates]) # (source,channel,freq,frame)
    power=np.abs(S)**2 if psd is None else np.asarray(psd,dtype=np.float64)
    if power.shape!=S.shape: raise ValueError("PSD shape mismatch")
    strength=np.maximum(power,0)**(alpha/2); denom=strength.sum(axis=0,keepdims=True)
    mask=np.divide(strength,denom,out=np.full_like(strength,.25),where=denom>0)
    if not np.allclose(mask.sum(axis=0),1,atol=1e-7): raise RuntimeError("ratio masks do not sum to one")
    out=_istft(mask*X[None],x.shape[-1])
    if out.shape!=estimates.shape or not np.isfinite(out).all(): raise ValueError("invalid ratio-mask output")
    return out

def t1(x,estimates): return mixture_consistency(x,estimates,"uniform")
def t2(x,estimates,alpha=2.0): return ratio_mask(x,estimates,alpha)
def t5(x,estimates,alpha=1.0,blend=.25):
    if not 0<=blend<=1: raise ValueError("blend must be in [0,1]")
    return (1-blend)*estimates+blend*t2(x,estimates,alpha)

def norbert_wiener(estimates,x,iterations=1):
    """Invoke norbert using (frames,bins,channels,sources) tensors."""
    if iterations not in (0,1,2): raise ValueError("supported diagnostic iterations: 0, 1, 2")
    try: import norbert
    except ImportError as exc: raise RuntimeError("norbert dependency unavailable") from exc
    validate_tensors(x,estimates)
    X=_stft(x).transpose(2,1,0)
    mags=np.stack([np.abs(_stft(src)) for src in estimates])
    V=mags.transpose(3,2,1,0)
    Y=norbert.wiener(V,X,iterations=iterations)
    if Y.shape[-1]!=4: raise ValueError(f"norbert returned unexpected shape {Y.shape}")
    return _istft(Y.transpose(3,2,1,0),x.shape[-1])

def consistency_iteration(x,estimates,iterations=1):
    """MISI-like phase consistency with fixed initial magnitudes."""
    if iterations not in (0,1,2): raise ValueError("iterations must be 0, 1, or 2")
    validate_tensors(x,estimates)
    if iterations==0: return estimates.copy()
    specs=np.stack([_stft(src) for src in estimates]); X=_stft(x)
    mags=np.abs(specs); Y=specs.copy()
    for _ in range(iterations):
        wave=_istft(Y,x.shape[-1]); re=np.stack([_stft(src) for src in wave])
        corrected=re+(X-re.sum(axis=0,keepdims=True))/4
        Y=mags*np.exp(1j*np.angle(corrected))
    return _istft(Y,x.shape[-1])

def donor_reseparation(x, estimates, separator, *, donors=("other",), strength=.5):
    """Return bass recovered from selected donor re-inference, conserving estimate sum."""
    validate_tensors(x,estimates)
    if not 0<=strength<=1: raise ValueError("strength must be in [0,1]")
    indexes={"drums":1,"other":3}
    if not donors or any(d not in indexes for d in donors): raise ValueError("donors must be drums and/or other")
    out=estimates.copy()
    for donor in donors:
        donor_index=indexes[donor]
        separated=np.asarray(separator(estimates[donor_index]),dtype=np.float32)
        if separated.shape!=estimates.shape or not np.isfinite(separated).all(): raise ValueError("donor separator must return finite (4,2,N)")
        delta=strength*separated[2]
        out[2]+=delta; out[donor_index]-=delta
    return out

def norbert_available() -> bool:
    try: import norbert  # noqa: F401
    except ImportError: return False
    return True

DONOR_EVIDENCE_STATUS = ("exploratory — no direct precedent for 4-stem frozen music separators "
                         "(closest: speech OR-PIT, trained recursive models)")

def parse_model_candidate(name: str) -> tuple[str, list[str]]:
    """`replace:<model_id>` or `average:<id>+<id>[+...]` (ids sorted, unique, so one name = one candidate)."""
    kind, _, rest = str(name).partition(":")
    ids = rest.split("+") if rest else []
    if kind not in ("replace", "average") or not ids or any(not i for i in ids): raise ValueError(f"model candidate must be replace:<id> or average:<id>+<id>: {name!r}")
    if kind == "replace" and len(ids) != 1: raise ValueError("replace takes exactly one model id")
    if kind == "average" and len(ids) < 2: raise ValueError("average needs at least two model ids")
    if ids != sorted(set(ids)): raise ValueError(f"model ids in {name!r} must be unique and sorted")
    return kind, ids

def model_estimates_candidate(kind: str, ids: list[str]):
    """Equal-weight waveform average, or a plain replacement, of registered models' (4,2,N) estimates; mixture-only inputs."""
    def run(x, estimates, by_model):
        missing = [i for i in ids if i not in by_model]
        if missing: raise ValueError(f"model estimates missing for {missing}")
        stack = [np.asarray(by_model[i], dtype=np.float32) for i in ids]
        for e in stack: validate_tensors(x, e)
        out = stack[0].copy() if kind == "replace" else (np.mean(np.stack(stack).astype(np.float64), axis=0)).astype(np.float32)
        validate_tensors(x, out); return out
    return run

def run_candidate(spec: dict, x, estimates, by_model: dict | None = None):
    """Single dispatch point for develop/evaluate: model-estimate candidates also receive the other models' estimates."""
    if spec.get("requires") == "model-estimates": return spec["fn"](x, estimates, by_model or {})
    return spec["fn"](x, estimates)

def candidate_registry(separator_getter=None, *, model_candidates=(), baseline_id: str | None = None) -> dict[str,dict]:
    """The single name -> adapter table used by develop and evaluate.

    `separator_getter` is called lazily, only by re-separation candidates. `requires` names the dependency a candidate
    needs; `models` lists the registered checkpoints it depends on; `additional_inference_calls` is the selection tie breaker."""
    def donor(names):
        def run(x,e):
            if separator_getter is None: raise RuntimeError("re-separation candidate needs a separator")
            return donor_reseparation(x,e,separator_getter(),donors=names,strength=.5)
        return run
    base_models = [baseline_id] if baseline_id else []
    table={
        "identity":(lambda x,e:identity(x,e),0,None,{},[]),
        "T1-uniform":(lambda x,e:t1(x,e),0,None,{"weights":"uniform"},[]),
        "T2-alpha1":(lambda x,e:t2(x,e,1),0,None,{"alpha":1},[]),
        "T2-alpha2":(lambda x,e:t2(x,e,2),0,None,{"alpha":2},[]),
        "T5-alpha1-lambda025":(lambda x,e:t5(x,e,1,.25),0,None,{"alpha":1,"blend":.25},[]),
        "C2-iter1":(lambda x,e:consistency_iteration(x,e,1),0,None,{"iterations":1},[]),
        "C2-iter2":(lambda x,e:consistency_iteration(x,e,2),0,None,{"iterations":2},[]),
        **{f"C1-norbert-{n}":(lambda x,e,n=n:norbert_wiener(e,x,n),0,"norbert",{"iterations":n},[]) for n in (0,1,2)},
        "C-donor-other-lambda05":(donor(("other",)),1,"separator",{"donors":["other"],"strength":.5,"evidence_status":DONOR_EVIDENCE_STATUS},base_models),
        "C-donor-drums-lambda05":(donor(("drums",)),1,"separator",{"donors":["drums"],"strength":.5,"evidence_status":DONOR_EVIDENCE_STATUS},base_models),
    }
    for name in model_candidates:
        kind, ids = parse_model_candidate(name)
        if kind == "replace" and baseline_id and ids[0] == baseline_id: raise ValueError("replace:<baseline> equals identity; not a candidate")
        if name in table: raise ValueError(f"duplicate candidate {name}")
        table[name]=(model_estimates_candidate(kind,ids),len(ids),"model-estimates",{"kind":kind,"models":ids,"weights":"equal"},ids)
    return {name:{"fn":fn,"additional_inference_calls":calls,"requires":requires,"params":params,"models":list(models)}
            for name,(fn,calls,requires,params,models) in table.items()}

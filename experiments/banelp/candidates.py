"""Practical four-stem candidate adapters; no target input is accepted."""
from __future__ import annotations
import numpy as np
from scipy.signal import get_window, istft, stft
from .contracts import validate_tensors
NFFT,HOP=4096,1024

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

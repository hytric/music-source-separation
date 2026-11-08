"""Dev-only target-informed diagnostic helpers, kept apart from candidates."""
from __future__ import annotations
import numpy as np
from .candidates import _istft, _stft
from .contracts import validate_tensors

def target_psd_replacement(x,estimates,targets,stem_index=None,alpha=2.0):
    validate_tensors(x,estimates); validate_tensors(x,targets)
    powers=np.stack([np.abs(_stft(s))**2 for s in estimates]); truth=np.stack([np.abs(_stft(s))**2 for s in targets])
    if stem_index is None: powers=truth
    else:
        if stem_index not in range(4): raise ValueError("stem_index must be 0..3")
        powers[stem_index]=truth[stem_index]
    strength=np.maximum(powers,0)**(alpha/2); den=strength.sum(axis=0,keepdims=True)
    mask=np.divide(strength,den,out=np.full_like(strength,.25),where=den>0)
    return _istft(mask*_stft(x)[None],x.shape[-1])

def oracle_namespace(split: str) -> str:
    if split!="dev": raise ValueError("oracle diagnostics are restricted to dev split")
    return "oracle-dev"

def oracle_ratio_mask(x,estimates,targets,alpha=2.0):
    """Apply target-PSD ratio masks to the mixture; dev-only diagnostic."""
    validate_tensors(x,estimates); validate_tensors(x,targets)
    powers=np.stack([np.abs(_stft(s))**2 for s in targets])
    strength=np.maximum(powers,0)**(alpha/2); den=strength.sum(axis=0,keepdims=True)
    mask=np.divide(strength,den,out=np.full_like(strength,.25),where=den>0)
    return _istft(mask*_stft(x)[None],x.shape[-1])

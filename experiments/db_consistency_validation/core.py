"""Mixture consistency with vocals/other fixed and only drums/bass free."""
import numpy as np
from scipy import signal


def close_db(x,out):
    # The residual is only numerical after a consistent STFT projection.
    for _ in range(2):
        residual=np.asarray(x,np.float64)-out.sum(axis=0,dtype=np.float64)
        out[1]+=(.5*residual).astype(np.float32)
        out[2]+=(.5*residual).astype(np.float32)
    return out


def project(x,z,mode,stft=None,istft=None):
    if z.shape[:2]!=(4,2) or z.ndim!=3 or x.shape!=(2,z.shape[-1]):raise ValueError('expected stereo mixture and four sources')
    if z.dtype!=np.float32 or not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('finite FP32 estimates required')
    out=z.copy();residual=np.asarray(x,np.float64)-z.sum(axis=0,dtype=np.float64)
    if mode=='uniform':out[1:3]+=(.5*residual).astype(np.float32)[None]
    elif mode=='power':
        if stft is None or istft is None:raise ValueError('STFT callbacks required')
        db=stft(z[1:3]);r=stft(residual)
        power=db.real.astype(np.float64)**2+db.imag.astype(np.float64)**2
        alpha=np.exp(-1024/(.2*44100))
        power=signal.lfilter([1-alpha],[1,-alpha],power,axis=-1)
        denominator=power.sum(axis=0,keepdims=True)
        masks=np.divide(power,denominator,out=np.full_like(power,.5),where=denominator>1e-12*max(float(denominator.max()),1e-24))
        out[1:3]=istft(db+masks*r[None],x.shape[-1])
    else:raise ValueError('unknown projection')
    out=close_db(x,out)
    if out[[0,3]].tobytes()!=z[[0,3]].tobytes():raise ValueError('protected waveform changed')
    error=float(np.max(np.abs(np.asarray(x,np.float64)-out.sum(0,dtype=np.float64))))
    if error>3e-6 or not np.isfinite(out).all():raise ValueError(f'consistency failed: {error}')
    return out


def reconstruction(x,z):
    x=np.asarray(x,np.float64);residual=x-z.sum(0,dtype=np.float64)
    rr=float(np.sqrt(np.mean(residual**2)));rms=float(np.sqrt(np.mean(x**2)))
    return {'relative_residual_rms':rr/max(rms,1e-12),'residual_rms':rr,'max_abs':float(np.max(np.abs(residual)))}

"""Fixed stem combination and reconstruction diagnostics (CPU only)."""
import numpy as np


def combine(baseline,finetuned):
    if baseline.shape!=finetuned.shape or baseline.ndim!=3 or baseline.shape[:2]!=(4,2):
        raise ValueError('expected matching (4,2,N) estimates')
    if baseline.dtype!=np.float32 or finetuned.dtype!=np.float32:
        raise ValueError('FP32 estimates required')
    if not np.isfinite(baseline).all() or not np.isfinite(finetuned).all():
        raise ValueError('nonfinite estimate')
    out=baseline.copy();out[1:3]=finetuned[1:3]
    if not np.array_equal(out[[0,3]],baseline[[0,3]]):raise ValueError('protected stems changed')
    return out


def reconstruction(x,z):
    x=np.asarray(x,dtype=np.float64);estimate=np.asarray(z,dtype=np.float64).sum(axis=0)
    if x.shape!=estimate.shape:raise ValueError('reconstruction shape mismatch')
    residual=x-estimate
    rms=float(np.sqrt(np.mean(x*x)));rr=float(np.sqrt(np.mean(residual*residual)))
    return {'mixture_rms':rms,'residual_rms':rr,'relative_residual_rms':rr/max(rms,1e-12),'max_abs':float(np.max(np.abs(residual)))}


def mean_si(scores):
    values=[r.get('si_sdr') for r in scores['metrics']]
    values=[v for v in values if v is not None]
    if not values:raise ValueError('no valid stems')
    return float(np.mean(values))


def compare_metrics(actual,expected,tolerance=1e-7):
    errors=[]
    for a,b in zip(actual['metrics'],expected['metrics']):
        if a['stem']!=b['stem'] or a['state']!=b['state']:raise ValueError('metric state/order changed')
        for key in ('si_sdr','si_sir','si_sar','sd_sdr','target_gain','target_rms','output_rms'):
            av,bv=a.get(key),b.get(key)
            if (av is None)!=(bv is None):raise ValueError('metric availability changed')
            if av is not None:errors.append(abs(av-bv))
    error=max(errors,default=0.)
    if error>tolerance:raise ValueError(f'metric equivalence changed: {error}')
    return error

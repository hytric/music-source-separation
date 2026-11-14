"""Deterministic data roles and quality-gate numerical contracts."""
import hashlib
import numpy as np


def split_songs(names,salt,fit_count=64):
    if len(set(names))!=80 or len(names)!=80:raise ValueError('expected80 unique train songs')
    ordered=sorted(names,key=lambda n:(hashlib.sha256((salt+'\0'+n).encode()).hexdigest(),n))
    return {n:'fit' if i<fit_count else 'calibration' for i,n in enumerate(ordered)}


def quality_label(delta,rms_fraction,bass_B,bass_H,config):
    valid=[v for v in delta if v is not None]
    if not valid or not all(np.isfinite(v) for v in valid):raise ValueError('undefined quality record')
    q=config['quality_label'];threshold=config['activity']['target_rms_div_mixture_rms_min']
    active=[i for i in (1,2) if delta[i] is not None and rms_fraction[i]>=threshold]
    new=bass_B is not None and bass_H is not None and bass_B>=q['gain_threshold'] and bass_H<q['gain_threshold']
    return int(np.mean(valid)>=q['min_legacy_overall_gain_db'] and len(active)>=q['min_active_free_stems'] and all(delta[i]>=q['min_active_free_stem_delta_db'] for i in active) and not new)


def choose_policy(candidates,tolerance=1e-12):
    safe=[v for v in candidates if v['safe']]
    if not safe:raise ValueError('always-baseline candidate must be present and safe')
    best=max(v['wet_song_median'] for v in safe)
    tied=[v for v in safe if abs(v['wet_song_median']-best)<=tolerance]
    chosen=max(tied,key=lambda v:2. if v.get('policy')=='always_baseline' else v['threshold'])
    return chosen,('scientific_failed_no_effect_candidate' if best<=0 or chosen.get('policy')=='always_baseline' else 'calibrated')


def fit_linear(x,y,weights,config):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);w=np.asarray(weights,np.float64)
    if x.ndim!=2 or len(x)!=len(y) or y.shape!=w.shape or set(y)!={0.,1.}:raise ValueError('label_degenerate or invalid training arrays')
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w<=0).any() or abs(w.sum()-1)>1e-10:raise ValueError('invalid training weights')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale]);beta=np.zeros(design.shape[1]);cfg=config['fit']
    reg=np.diag([cfg['intercept_penalty']]+[cfg['l2']]*x.shape[1])
    def objective(b):
        a=design@b;return float(np.sum(w*(np.logaddexp(0,a)-y*a))+.5*b@reg@b)
    for iteration in range(1,cfg['max_newton_iterations']+1):
        prob=1/(1+np.exp(-np.clip(design@beta,-60,60)))
        gradient=design.T@(w*(prob-y))+reg@beta
        hessian=design.T@((w*prob*(1-prob))[:,None]*design)+reg+np.eye(len(beta))*1e-10
        step=np.linalg.solve(hessian,gradient);rate=1.;old=objective(beta)
        while rate>=cfg['line_search_min_rate'] and objective(beta-rate*step)>old+1e-14:rate*=.5
        if rate<cfg['line_search_min_rate']:raise ValueError('line search failed')
        beta-=rate*step
        if not np.isfinite(beta).all():raise ValueError('nonfinite coefficients')
        if np.max(abs(rate*step))<cfg['step_max_abs_convergence']:
            return {'mean':mean.tolist(),'scale':scale.tolist(),'beta':beta.tolist(),'iterations':iteration,'objective':objective(beta)}
    raise ValueError('optimizer did not converge')

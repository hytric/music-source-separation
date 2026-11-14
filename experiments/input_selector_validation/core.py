"""Input-only features and a frozen, training-standardized linear selector."""
import numpy as np
FS=44100


def features(wave,stft):
    if wave.ndim!=2 or wave.shape[0]!=2 or wave.shape[-1]<26*FS or not np.isfinite(wave).all():raise ValueError('finite stereo audio of at least26s required')
    begin=(wave.shape[-1]-26*FS)//2
    x=np.ascontiguousarray(wave[:,begin:begin+26*FS],dtype=np.float32)
    z=stft(x)[...,2:-2]
    p=np.abs(z).astype(np.float64)**2
    total=float(p.mean())
    if total<1e-20:return np.zeros(33,dtype=np.float64)
    eps=total*1e-10;frequency=np.fft.rfftfreq(4096,1/FS)
    edges=np.geomspace(40,18000,13);bands=[(frequency>=a)&(frequency<b) for a,b in zip(edges[:-1],edges[1:])]
    value=[float(np.log(max(p[:,mask].mean()/total,1e-10))) for mask in bands]
    for mask in bands[::2]:
        a=z[0,mask];b=z[1,mask]
        coherence=abs(np.sum(a*np.conj(b),dtype=np.complex128))/max(float(np.sqrt(np.sum(abs(a)**2,dtype=np.float64)*np.sum(abs(b)**2,dtype=np.float64))),eps)
        value.append(float(coherence))
    mono=p.mean(0);flat=np.exp(np.log(mono+eps).mean(0))/(mono.mean(0)+eps)
    value.extend(np.percentile(flat,[10,50,90]).tolist())
    norm=mono/(mono.sum(0,keepdims=True)+eps)
    flux=np.maximum(np.diff(norm,axis=1),0).sum(0)
    value.extend([float(np.mean(flux)),float(np.percentile(flux,90))])
    envelope=np.sqrt(mono.mean(0)/total);center=envelope-envelope.mean();energy=float(center@center)
    for seconds in (.05,.1,.25,.5,1.):
        lag=max(1,round(seconds*FS/1024));value.append(float(center[:-lag]@center[lag:]/max(energy,1e-12)))
    scale=max(float(envelope.std()),1e-10)
    value.extend([float(envelope.std()),float(np.mean((center/scale)**4)),float(np.mean(np.diff(envelope)**2)),float(np.percentile(envelope,95)/max(np.median(envelope),1e-6)),float(np.mean(envelope<.1))])
    result=np.asarray(value,np.float64)
    if result.shape!=(33,) or not np.isfinite(result).all():raise ValueError('invalid feature vector')
    return result


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))


def fit(x,y,l2=.01,max_iterations=100):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or set(y)!={0.,1.}:raise ValueError('binary training features required')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale]);weights=np.where(y==0,.5/np.sum(y==0),.5/np.sum(y==1))
    beta=np.zeros(design.shape[1]);penalty=np.diag([0.]+[l2]*x.shape[1])
    def objective(b):
        a=design@b
        return float(np.sum(weights*(np.logaddexp(0,a)-y*a))+.5*b@penalty@b)
    iterations=0
    for iterations in range(1,max_iterations+1):
        probability=sigmoid(design@beta)
        gradient=design.T@(weights*(probability-y))+penalty@beta
        hessian=design.T@((weights*probability*(1-probability))[:,None]*design)+penalty+np.eye(len(beta))*1e-10
        step=np.linalg.solve(hessian,gradient);rate=1.
        before=objective(beta)
        while rate>1e-8 and objective(beta-rate*step)>before:rate*=.5
        beta-=rate*step
        if np.max(abs(rate*step))<1e-8:break
    if not np.isfinite(beta).all():raise ValueError('nonfinite fitted selector')
    return {'mean':mean.tolist(),'scale':scale.tolist(),'beta':beta.tolist(),'iterations':iterations,'objective':objective(beta),'l2':l2}


def predict(x,model):
    a=(np.asarray(x,np.float64)-np.asarray(model['mean']))/np.asarray(model['scale'])
    return sigmoid(a@np.asarray(model['beta'][1:])+model['beta'][0])

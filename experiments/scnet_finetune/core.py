"""Training math for matched wet-target SCNet fine-tuning."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch

FS=44100
SOURCES=('vocals','drums','bass','other')


def tensor_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode()+b'\0');h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def normalized_pair(y):
    x=y.sum(dim=1)
    mono=x.mean(dim=1,keepdim=True)
    mean=mono.mean(dim=-1,keepdim=True);std=mono.std(dim=-1,keepdim=True)
    if not torch.isfinite(std).all() or (std<1e-6).any():raise ValueError('near-silent mixture')
    return (x-mean)/std,(y-mean[:,None])/std[:,None]


def reconstruction_loss(estimate,target,stft_config,guard=44100,vocal_weight=2.,wave_weight=.05):
    estimate=estimate[...,guard:-guard].contiguous();target=target[...,guard:-guard].contiguous()
    if estimate.shape!=target.shape:raise ValueError('target alignment failed')
    n=estimate.shape[-1]
    a=torch.stft(estimate.reshape(-1,n),**stft_config,return_complex=True)
    b=torch.stft(target.reshape(-1,n),**stft_config,return_complex=True)
    d=torch.view_as_real(a-b).square().reshape(*estimate.shape[:-1],*a.shape[-2:],2)
    spectral=(d.mean(dim=(2,3,4,5))+1e-8).sqrt().mean(0)
    wave=(estimate-target).abs().mean(dim=(0,2,3))
    weights=estimate.new_tensor([vocal_weight,1.,1.,1.])
    loss=((spectral+wave_weight*wave)*weights).sum()/weights.sum()
    return loss,{'spectral':spectral.detach().cpu().tolist(),'wave':wave.detach().cpu().tolist()}


def save_checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.writing');torch.save(payload,temp);temp.replace(path)


class PairedCrops:
    def __init__(self,root,records,tracks,verify):
        self.root=Path(root);self.tracks=tracks
        train_names={t['name'] for t in tracks if t['split']=='train'}
        self.by_song={name:[] for name in sorted(train_names)}
        for row in records:
            if row['split']!='train':continue
            if row['track'] not in train_names:raise ValueError('training song leaked')
            self.by_song[row['track']].append(row)
        for name,rows in self.by_song.items():
            if len(rows)!=4 or sum(r['rir_set'] is None for r in rows)!=1:raise ValueError('expected dry+three wet records')
        self.names=sorted(self.by_song);self.verify=verify;self.verified=set()
        if len(self.names)!=80:raise ValueError('expected 80 training songs')

    def sample(self,step,seed):
        rng=np.random.default_rng(seed+step*1009)
        name=self.names[int(rng.integers(len(self.names)))];rows=self.by_song[name]
        dry=next(r for r in rows if r['rir_set'] is None)
        wet=[r for r in rows if r['rir_set'] is not None];wet=wet[int(rng.integers(3))]
        frames=6*FS;edge=3*FS
        lo=max(edge,dry['active_start_frame']);hi=min(30*FS-edge,dry['active_end_frame'])-frames
        start=int(rng.integers(lo,hi+1)) if hi>=lo else max(0,min(30*FS-frames,(dry['active_start_frame']+dry['active_end_frame']-frames)//2))
        out=[]
        for row in (dry,wet):
            e=row['files']['targets']
            if e['path'] not in self.verified:self.verify(self.root,e);self.verified.add(e['path'])
            a=np.load(self.root/e['path'],mmap_mode='r',allow_pickle=False)
            if a.shape!=(4,2,30*FS) or a.dtype!=np.float32:raise ValueError('invalid cached target')
            crop=np.array(a[...,start:start+frames],copy=True)
            if not np.isfinite(crop).all():raise ValueError('nonfinite training target')
            out.append(torch.from_numpy(crop)[None])
        return out,{'track':name,'crop_start':start,'crop_frames':frames,'wet_rir_set':wet['rir_set']}

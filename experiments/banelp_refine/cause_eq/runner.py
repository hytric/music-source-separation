"""Resumeable GPU inference for the predeclared causal EQ experiment."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

from banelp_refine.common import SOURCES, atomic_json, digest, emit, environment, exclusive, file_entry, read_json, sha256, verify_entry
from .dsp import FS, CORE, core_slice, pad_core, apply_filter, factors, known_q, blind_q, log_power, rms, sum_stems, boundary_check
from .metrics import score_sets, equivalence_check


def source_digest():
    return digest({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name != 'report.py'})


def save_array(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.part').open('wb') as stream:
        np.save(stream, np.asarray(value, dtype=np.float32), allow_pickle=False)
    path.with_suffix('.part').replace(path)


class Experiment:
    def __init__(self, args):
        self.args = args
        self.base = Path(args.baseline_runtime).resolve()
        self.prepared = Path(args.prepared_runtime).resolve()
        self.config = read_json(args.config)
        if self.config['n_fft_block'] != 8388608 or self.config['sample_rate'] != FS or self.config['core_samples'] != CORE:
            raise ValueError('unsupported block/audio contract')
        self.n = self.config['n_fft_block'];self.core = core_slice(self.n)
        self.dsp_n = self.config['dsp_fft_length']
        sys.path.insert(0, str(Path(args.baseline_repo).resolve()))
        from experiments.banelp.separators import verify_model_record
        models = read_json(self.base/'manifests/models.json')['models']
        self.model_record = next(m for m in models if m['model_id'] == 'htdemucs')
        verify_model_record(self.model_record)
        if len(self.model_record['members']) != 1 or '955717e8' not in self.model_record['members'][0]['path']:
            raise ValueError('wrong checkpoint')
        self.dataset = read_json(self.base/'manifests/dataset.json')
        self.prepared_dataset = read_json(self.prepared/'manifests/dataset.json')
        self.rir_manifest = read_json(self.base/'manifests/rirs.json')
        self.rirs = {}
        for name, pos in zip(SOURCES, ('P1','P2','P3','P4')):
            entry = self.rir_manifest['rooms']['R4']['positions'][pos]
            p = self.base/entry['path']
            if sha256(p) != entry['sha256']:raise ValueError('RIR hash changed')
            self.rirs[name] = np.load(p,allow_pickle=False)
        self.h = self.rirs['bass']
        self.delay = self.rir_manifest['rooms']['R4']['positions']['P3']['direct_delay_samples']
        self.binding = {'config': self.config, 'code_sha256': source_digest(), 'environment': environment(),
                        'protocol_sha256':sha256(Path(__file__).parent.parent/'CAUSE_EXPERIMENT.md'),
                        'model': self.model_record,
                        'baseline_code': {p.name:sha256(p) for p in sorted((Path(args.baseline_repo)/'experiments/banelp').glob('*.py'))},
                        'input_manifests':{str(p):sha256(p) for p in [self.base/'manifests/dataset.json',self.base/'manifests/rirs.json',
                                          self.prepared/'manifests/dataset.json',self.prepared/'manifests/cache.json']}}
        root = Path(args.runtime).resolve()
        if root in [self.base,self.prepared] or self.base in root.parents or self.prepared in root.parents:
            raise ValueError('cause experiment must use a separate runtime')
        self.root = root / ('run-'+digest(self.binding)[:16]);self.root.mkdir(parents=True,exist_ok=True)
        if (self.root/'binding.json').exists() and read_json(self.root/'binding.json') != self.binding:
            raise ValueError('run binding differs')
        atomic_json(self.root/'binding.json',self.binding)
        atomic_json(root/'latest.json',{'run':str(self.root),'binding_sha256':digest(self.binding)})
        self.sep = None;self.calls = 0
        self.reference = self.reference_power()
        self.responses,self.magnitude,self.factor_check = factors(self.h,self.delay,self.dsp_n)
        self.q_known = known_q(self.magnitude,self.dsp_n)
        self.core_tracks = sorted([t for t in self.dataset['tracks'] if t['split']=='dev'],key=lambda t:t['name'])
        self.apply_tracks = sorted([t for t in self.prepared_dataset['tracks'] if t['split']=='validation'],key=lambda t:t['name'])
        if len(self.core_tracks)!=10 or len(self.apply_tracks)!=10:raise ValueError('expected ten songs per group')

    def reference_power(self):
        output = self.root/'reference.json'
        if output.exists():return np.asarray(read_json(output)['log_power'])
        cache = read_json(self.prepared/'manifests/cache.json')
        records = [r for r in cache['records'] if r['split']=='train' and r['rir_set'] is None]
        if len(records)!=80:raise ValueError('reference needs exactly 80 dry training mixtures')
        powers=[]
        for r in records:
            p=verify_entry(self.prepared,r['files']['mixture'])
            _,value=log_power(np.load(p,allow_pickle=False));powers.append(value)
        reference=np.median(np.asarray(powers),axis=0)
        atomic_json(output,{'tracks':len(records),'record_ids':[r['id'] for r in records],'log_power':reference.tolist()})
        return reference

    def read_dry(self, entry, group):
        out=[]
        for name in SOURCES:
            if group=='core':
                e=entry['sources'][name];p=Path(e['path'])
                if sha256(p)!=e['sha256']:raise ValueError('original source changed')
                start=(e['frames']-CORE)//2;length=CORE;left=right=0
            else:
                e=entry['files'][name];p=verify_entry(self.prepared,e)
                start=entry['crop_start_frame'];length=entry['crop_frames']
                left,right=entry['pad_left_frames'],entry['pad_right_frames']
            audio,sr=sf.read(p,start=start,frames=length,dtype='float32',always_2d=True)
            if sr!=FS or audio.shape!=(length,2):raise ValueError('invalid source crop')
            out.append(np.pad(audio.T,((0,0),(left,right))))
        return np.asarray(out,dtype=np.float32)

    def guard(self):
        temp=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
        if temp>=self.config['gpu_temperature_limit']:raise RuntimeError('GPU temperature limit reached')

    def inference(self, folder, name, mixture, q=None, gain=None):
        import torch
        path=folder/(name+'.npy');meta=folder/(name+'.json')
        descriptor={'name':name,'shape':list(mixture.shape),'binding':digest(self.binding),'q':bool(q is not None),'gain':gain}
        if meta.exists():
            old=read_json(meta)
            if old['descriptor']!=descriptor:raise ValueError('cached inference configuration changed')
            verify_entry(self.root,old['file'])
            return np.load(path,mmap_mode='r',allow_pickle=False)
        self.guard()
        if self.sep is None:
            from experiments.banelp.separators import HTDemucsCUDA
            torch.set_num_threads(self.config['cpu_threads'])
            self.sep=HTDemucsCUDA(model_name='htdemucs',segment=7.,overlap=.25,registered=self.model_record)
        started=time.monotonic()
        if q is not None:network_input=apply_filter(mixture,q)
        elif gain is not None:network_input=(mixture*gain).astype(np.float32)
        else:network_input=mixture
        input_sha=__import__('hashlib').sha256(network_input.tobytes()).hexdigest()
        result=self.sep(network_input)
        del network_input
        if q is not None:result=apply_filter(result,1/q)
        elif gain is not None:result=(result/gain).astype(np.float32)
        if result.shape!=(4,2,self.n) or not np.isfinite(result).all():raise ValueError('invalid GPU output')
        save_array(path,result);del result;gc.collect()
        elapsed=time.monotonic()-started;self.calls+=1
        atomic_json(meta,{'descriptor':descriptor,'file':file_entry(path,self.root),'input_sha256':input_sha,'elapsed_seconds':elapsed,
                          'gpu':'cuda:0','peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                          'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'normalization':self.sep.last_normalization})
        emit('cause_gpu_case',group=folder.parent.name,track=folder.name,condition=name,seconds=elapsed,calls_this_invocation=self.calls)
        atomic_json(self.root/'progress.json',{'last_case':str(meta.relative_to(self.root)),'calls_this_invocation':self.calls,'optimizer_updates':0})
        return np.load(path,mmap_mode='r',allow_pickle=False)

    def core_track(self, entry):
        name=entry['name'];folder=self.root/'core'/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'results.json').exists():return read_json(folder/'results.json')
        dry=self.read_dry(entry,'core')
        checks={}
        key=digest({'track':name,'dataset':self.binding['input_manifests'][str(self.base/'manifests/dataset.json')],
                    'rir':self.binding['input_manifests'][str(self.base/'manifests/rirs.json')],
                    'dsp_code':sha256(Path(__file__).parent/'dsp.py'),'dsp_n':self.dsp_n})
        boundary_path=self.root.parent/'boundary-cache'/(key+'.json')
        if boundary_path.exists():
            checks=read_json(boundary_path)['checks']
        else:
            for condition in ('M','A'):
                for i,stem in enumerate(SOURCES):
                    checks[condition+'_'+stem]=boundary_check(dry[i],self.h,self.delay,self.dsp_n,condition)
            atomic_json(boundary_path,{'checks':checks,'key':key})
        if len(checks)!=8 or any(v>1e-4 for v in checks.values()):raise ValueError('invalid reused boundary checks')
        atomic_json(folder/'boundary.json',checks)
        targets={};inputs={};outputs={};gains={}
        reference_rms=rms(sum_stems(dry)) * np.sqrt(CORE/self.n)
        for condition in ('D','M','A','H'):
            yp=np.lib.format.open_memmap(folder/'temporary-targets.npy',mode='w+',dtype='float32',shape=(4,2,self.n))
            for i in range(4):
                tail={}
                yp[i]=apply_filter(pad_core(dry[i],self.n),self.responses[condition],stats=tail)
                checks['discarded_'+condition+'_'+SOURCES[i]]=tail['maximum_discarded_energy_fraction']
                if tail['maximum_discarded_energy_fraction']>self.config['maximum_discarded_input_energy_fraction']:
                    raise ValueError('model guard discards too much transformed input energy')
            x=sum_stems(yp);g=reference_rms/max(rms(x),1e-12);gains[condition]=g
            for i in range(4):yp[i]*=g
            x=sum_stems(yp)
            targets[condition]=np.array(yp[...,self.core],copy=True)
            del yp
            (folder/'temporary-targets.npy').unlink()
            save_array(folder/('input-'+condition+'.npy'),x)
            inputs[condition]=np.load(folder/('input-'+condition+'.npy'),mmap_mode='r')
            outputs[condition]=self.inference(folder,condition,x);del x
        records={}
        for condition in ('D','M','A','H'):
            if condition=='D':order=np.array(outputs['D'][...,self.core],copy=True)
            else:
                order_full=apply_filter(outputs['D'],self.responses[condition]/self.responses['D'])
                order=np.array(order_full[...,self.core]*(gains[condition]/gains['D']),copy=True);del order_full
            actual=np.array(outputs[condition][...,self.core],copy=True)
            records[condition]=score_sets(targets[condition],{'base':actual,'order':order})
        metric_error=equivalence_check(targets['H'],np.array(outputs['H'][...,self.core],copy=True))
        eq_checks={}
        for condition in ('D','H'):
            x=inputs[condition]
            qx=apply_filter(x,self.q_known);gain=rms(qx)/max(rms(x),1e-12);del qx
            eq=self.inference(folder,condition+'_eq',x,q=self.q_known)
            scalar=self.inference(folder,condition+'_scalar',x,gain=gain)
            roundtrip=apply_filter(apply_filter(outputs[condition],self.q_known),1/self.q_known)
            error=rms(roundtrip[...,self.core].astype(np.float64)-outputs[condition][...,self.core])/max(rms(outputs[condition][...,self.core]),1e-12)
            del roundtrip
            if error>1e-6:raise ValueError('EQ sham roundtrip failed')
            eq_checks[condition]={'roundtrip_relative_rms':error,'scalar_gain':gain}
            records[condition].update(score_sets(targets[condition],{'eq':np.array(eq[...,self.core]),'scalar':np.array(scalar[...,self.core])}))
        value={'track':name,'group':'core','records':records,'gains':gains,'boundary':checks,'eq_checks':eq_checks,'metric_equivalence_max_db':metric_error}
        atomic_json(folder/'results.json',value)
        emit('cause_track_complete',group='core',track=name)
        return value

    def apply_track(self, entry):
        from experiments.banelp.data import apply_rir
        name=entry['name'];folder=self.root/'apply'/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'results.json').exists():return read_json(folder/'results.json')
        dry=self.read_dry(entry,'apply');records={};checks={}
        for condition in ('dry','wet'):
            if condition=='dry':targets=dry
            else:targets=np.stack([apply_rir(dry[i].T,self.rirs[s])[0] for i,s in enumerate(SOURCES)])
            x=pad_core(sum_stems(targets),self.n)
            q=blind_q(sum_stems(targets),self.reference,self.dsp_n)
            qx=apply_filter(x,q);gain=rms(qx)/max(rms(x),1e-12);del qx
            base=self.inference(folder,condition,x)
            eq=self.inference(folder,condition+'_eq',x,q=q)
            scalar=self.inference(folder,condition+'_scalar',x,gain=gain)
            records[condition]=score_sets(targets,{'base':np.array(base[...,self.core]),'eq':np.array(eq[...,self.core]),'scalar':np.array(scalar[...,self.core])})
            checks[condition]={'scalar_gain':gain,'q_min':float(q.min()),'q_max':float(q.max()),'q_sha256':__import__('hashlib').sha256(q.tobytes()).hexdigest()}
        value={'track':name,'group':'apply','records':records,'eq_checks':checks}
        atomic_json(folder/'results.json',value);emit('cause_track_complete',group='apply',track=name)
        return value

    def run(self, phase):
        started=time.monotonic()
        with exclusive(self.root,'inference'):
            if phase in ('pilot','core','all'):
                tracks=self.core_tracks[:1] if phase=='pilot' else self.core_tracks
                for t in tracks:self.core_track(t);gc.collect()
            if phase in ('pilot','apply','all'):
                tracks=self.apply_tracks[:1] if phase=='pilot' else self.apply_tracks
                for t in tracks:self.apply_track(t);gc.collect()
            if source_digest()!=self.binding['code_sha256']:raise ValueError('code changed during execution')
            atomic_json(self.root/f'phase-{phase}.json',{'phase':phase,'complete':True,'elapsed_seconds':time.monotonic()-started,
                        'new_gpu_calls':self.calls,'optimizer_updates':0,'factor_check':self.factor_check})
            emit('cause_phase_complete',phase=phase,run=str(self.root),new_gpu_calls=self.calls)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent.parent/'configs/cause_eq.json')
    parser.add_argument('--baseline-repo',required=True)
    parser.add_argument('--baseline-runtime',required=True)
    parser.add_argument('--prepared-runtime',required=True)
    parser.add_argument('--runtime',required=True)
    parser.add_argument('--phase',choices=('pilot','core','apply','all'),default='pilot')
    args=parser.parse_args();Experiment(args).run(args.phase)


if __name__=='__main__':main()

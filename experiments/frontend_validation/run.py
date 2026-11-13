"""Frozen-separator input processing validation against unchanged wet targets."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import fft,signal
import soundfile as sf

FS=44100;N=1323000;GUARD=220500;L=N+2*GUARD;SL=slice(GUARD,GUARD+N)


def stft(x):
    return signal.stft(x,fs=FS,window='hann',nperseg=4096,noverlap=3072,
                       boundary='zeros',padded=True,axis=-1)[2]


def istft(z,length):
    return signal.istft(z,fs=FS,window='hann',nperseg=4096,noverlap=3072,
                        input_onesided=True,boundary=True,time_axis=-1,freq_axis=-2)[1][...,:length].astype(np.float32)


def restore(x,z,weighted):
    Z=stft(z);X=stft(x)
    if weighted:
        power=(Z.real.astype(np.float64)**2+Z.imag.astype(np.float64)**2)
        alpha=np.exp(-1024/(.2*FS))
        power=signal.lfilter([1-alpha],[1,-alpha],power,axis=-1)
        den=power.sum(axis=0,keepdims=True)
        mask=np.divide(power,den,out=np.full_like(power,.25),where=den>1e-12*max(float(den.max()),1e-24))
    else:mask=.25
    result=istft(Z+mask*(X-Z.sum(axis=0)),x.shape[-1])
    result+=(x-result.sum(axis=0))[None]/4
    if not np.isfinite(result).all() or np.max(np.abs(result.sum(axis=0)-x))>3e-6:
        raise ValueError('residual restoration failed')
    return result


def wpe_front(x):
    from nara_wpe.wpe import wpe_v7
    Y=stft(x).transpose(1,0,2).astype(np.complex128)
    estimate=wpe_v7(Y,taps=10,delay=3,iterations=3,psd_context=0,statistics_mode='full')
    result=istft(estimate.transpose(1,0,2),x.shape[-1])
    if result.shape!=x.shape or not np.isfinite(result).all():raise ValueError('invalid WPE signal')
    return result


def paired_stat(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=np.float64)
    if not len(a):return {'median':None,'ci':[None,None],'n':0,'excluded':len(values)}
    ids=np.random.default_rng(20261113).integers(len(a),size=(10000,len(a)))
    boots=np.median(a[ids],axis=1)
    return {'median':float(np.median(a)),'ci':np.percentile(boots,[2.5,97.5]).tolist(),'n':len(a),'excluded':len(values)-len(a)}


class Experiment:
    def __init__(self,args):
        self.args=args;self.root=Path(args.runtime).resolve();self.base=Path(args.baseline_runtime).resolve();self.prep=Path(args.prepared_runtime).resolve()
        suite=Path(args.suite).resolve();sys.path.insert(0,str(suite));sys.path.insert(0,str(Path(args.baseline_repo).resolve()))
        sys.path.insert(0,str(self.root/'vendor'))
        from banelp_refine.common import SOURCES,atomic_json,read_json,sha256,digest,verify_entry,emit,environment
        from cause_eq.dsp import blind_q,apply_filter
        from cause_eq.metrics import score_sets,equivalence_check
        from experiments.banelp.separators import HTDemucsCUDA,verify_model_record
        self.sources=SOURCES;self.dump=atomic_json;self.read=read_json;self.sha=sha256;self.digest=digest;self.verify=verify_entry;self.emit=emit
        self.blind=blind_q;self.filter=apply_filter;self.score=score_sets;self.equiv=equivalence_check;self.separator_class=HTDemucsCUDA
        self.protocol=read_json(Path(__file__).with_name('protocol.json'))
        model=next(m for m in read_json(self.base/'manifests/models.json')['models'] if m['model_id']=='htdemucs');verify_model_record(model)
        self.model_record=model
        import nara_wpe.wpe as official_wpe
        binding={'protocol':self.protocol,'source':sha256(__file__),'environment':environment(),
                 'wpe_version':importlib.metadata.version('nara_wpe'),'click_version':importlib.metadata.version('click'),
                 'wpe_source':sha256(official_wpe.__file__),
                 'baseline_code':{p.name:sha256(p) for p in sorted((Path(args.baseline_repo)/'experiments/banelp').glob('*.py'))},
                 'shared_code':{str(p.relative_to(suite)):sha256(p) for p in [suite/'cause_eq/dsp.py',suite/'cause_eq/metrics.py',suite/'banelp_refine/common.py']},
                 'model':model,'manifests':{str(p):sha256(p) for p in [self.base/'manifests/rirs.json',self.prep/'manifests/dataset.json',self.prep/'manifests/cache.json']}}
        if binding['wpe_version']!='0.0.11':raise ValueError('wrong WPE implementation')
        self.binding=binding;self.run=self.root/('run-'+digest(binding)[:16]);self.run.mkdir(parents=True,exist_ok=True)
        atomic_json(self.run/'binding.json',binding);atomic_json(self.root/'latest.json',{'run':str(self.run)})
        self.tracks=sorted([t for t in read_json(self.prep/'manifests/dataset.json')['tracks'] if t['split']=='validation'],key=lambda t:t['name'])
        if len(self.tracks)!=10:raise ValueError('expected ten validation songs')
        rm=read_json(self.base/'manifests/rirs.json');self.rirs={}
        for s,p in zip(SOURCES,('P1','P2','P3','P4')):
            e=rm['rooms']['R4']['positions'][p];path=self.base/e['path']
            if sha256(path)!=e['sha256']:raise ValueError('RIR changed')
            self.rirs[s]=np.load(path,allow_pickle=False)
        self.reference=self.reference_power();self.sep=None;self.calls=0

    def reference_power(self):
        from cause_eq.dsp import log_power
        out=self.run/'reference.json'
        if out.exists():return np.asarray(self.read(out)['log_power'])
        rows=[r for r in self.read(self.prep/'manifests/cache.json')['records'] if r['split']=='train' and r['rir_set'] is None]
        if len(rows)!=80:raise ValueError('reference scope mismatch')
        powers=[log_power(np.load(self.verify(self.prep,r['files']['mixture']),allow_pickle=False))[1] for r in rows]
        value=np.median(powers,axis=0);self.dump(out,{'tracks':80,'log_power':value.tolist()});return value

    def dry_sources(self,t):
        out=[]
        for s in self.sources:
            p=self.verify(self.prep,t['files'][s]);x,sr=sf.read(p,start=t['crop_start_frame'],frames=t['crop_frames'],dtype='float32',always_2d=True)
            if sr!=FS or x.shape!=(t['crop_frames'],2):raise ValueError('source crop mismatch')
            out.append(np.pad(x.T,((0,0),(t['pad_left_frames'],t['pad_right_frames']))))
        return np.asarray(out,dtype=np.float32)

    def targets(self,dry,condition):
        from experiments.banelp.data import apply_rir
        if condition=='dry':return np.pad(dry,((0,0),(0,0),(GUARD,GUARD)))
        out=[]
        for i,s in enumerate(self.sources):
            legacy,meta=apply_rir(dry[i].T,self.rirs[s])
            channels=[]
            for ch in range(2):
                full=signal.fftconvolve(dry[i,ch].astype(np.float64),self.rirs[s],mode='full')*meta['gain']
                channels.append(np.pad(full,(GUARD,max(0,L-GUARD-len(full))))[:L])
            wet=np.asarray(channels,dtype=np.float32)
            if np.max(np.abs(wet[:,SL]-legacy))>1e-6:raise ValueError('wet core differs from original synthesis')
            out.append(wet)
        return np.asarray(out,dtype=np.float32)

    def infer(self,folder,name,x,q=None):
        p=folder/(name+'.npy');meta=folder/(name+'.json')
        if meta.exists():
            m=self.read(meta)
            if m['binding']!=self.digest(self.binding):raise ValueError('cached binding changed')
            self.verify(self.run,m['file']);return np.load(p,allow_pickle=False)
        temp=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
        if temp>=83:raise RuntimeError('GPU temperature limit')
        if self.sep is None:
            import torch
            torch.set_num_threads(2);self.sep=self.separator_class(model_name='htdemucs',segment=7.,overlap=.25,registered=self.model_record)
        started=time.monotonic();input_wave=self.filter(x,q) if q is not None else x
        result=self.sep(input_wave)
        if q is not None:result=self.filter(result,1/q)
        if not np.isfinite(result).all():raise ValueError('nonfinite separator result')
        with p.with_suffix('.part').open('wb') as f:np.save(f,result,allow_pickle=False)
        p.with_suffix('.part').replace(p)
        from banelp_refine.common import file_entry
        self.dump(meta,{'binding':self.digest(self.binding),'file':file_entry(p,self.run),'seconds':time.monotonic()-started})
        self.calls+=1;self.emit('frontend_gpu',track=folder.parent.name,condition=folder.name,input=name,new_gpu_calls=self.calls)
        return result

    def case(self,t,condition):
        folder=self.run/t['name']/condition;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'results.json').exists():return
        dry=self.dry_sources(t);y=self.targets(dry,condition);x=y.astype(np.float64).sum(axis=0).astype(np.float32)
        roundtrip=istft(stft(x),L);stft_error=float(np.max(np.abs(roundtrip-x)))
        if stft_error>1e-6:raise ValueError('STFT alignment/roundtrip failed')
        raw=self.infer(folder,'raw',x)
        q=self.blind(x[:,SL],self.reference,2097152)
        eq=self.infer(folder,'eq',x,q=q)
        eq_sham=self.filter(self.filter(raw,q),1/q)
        eq_error=float(np.sqrt(np.mean((eq_sham[...,SL].astype(np.float64)-raw[...,SL])**2))/max(np.sqrt(np.mean(raw[...,SL].astype(np.float64)**2)),1e-12));del eq_sham
        if eq_error>1e-6:raise ValueError(f'EQ crop roundtrip failed: {eq_error}')
        started=time.monotonic();front=wpe_front(x);wpe_seconds=time.monotonic()-started
        wpe=self.infer(folder,'wpe',front)
        gain=float(np.sqrt(np.mean(front.astype(np.float64)**2))/max(np.sqrt(np.mean(x.astype(np.float64)**2)),1e-12))
        gain_output=self.infer(folder,'gain',x*gain)
        estimates={'raw':raw[...,SL].copy(),'eq':eq[...,SL].copy(),'wpe':wpe[...,SL].copy()}
        estimates['raw_weighted']=restore(x,raw,True)[...,SL].copy()
        estimates['wpe_uniform']=restore(x,wpe,False)[...,SL].copy()
        estimates['wpe_weighted']=restore(x,wpe,True)[...,SL].copy()
        estimates['gain_weighted']=restore(x,gain_output,True)[...,SL].copy()
        scores=self.score(y[...,SL],estimates)
        equivalence=self.equiv(y[...,SL],estimates['wpe_weighted'])
        value={'track':t['name'],'condition':condition,'scores':scores,'checks':{'stft_max_abs':stft_error,'eq_roundtrip_relative_rms':eq_error,'legacy_metric_max_difference_db':equivalence},
               'wpe_seconds':wpe_seconds,'front_rms_ratio':float(np.sqrt(np.mean(front.astype(np.float64)**2))/max(np.sqrt(np.mean(x.astype(np.float64)**2)),1e-12))}
        self.dump(folder/'results.json',value);self.emit('frontend_case_complete',track=t['name'],condition=condition)

    def report(self):
        rows=[self.read(p) for p in sorted(self.run.glob('*/*/results.json'))]
        summaries={};candidate='wpe_weighted'
        def diff(r,a,b,stem):
            va=r['scores'][a]['metrics'][stem]['si_sdr'];vb=r['scores'][b]['metrics'][stem]['si_sdr']
            return None if va is None or vb is None else va-vb
        for c in ['dry','wet_R4']:
            group=[r for r in rows if r['condition']==c];summaries[c]={}
            for name in self.protocol['outputs']:
                if name=='raw':continue
                per={s:paired_stat([diff(r,name,'raw',i) for r in group]) for i,s in enumerate(self.sources)}
                overall=[]
                for r in group:
                    d=[diff(r,name,'raw',i) for i in range(4)];valid=[x for x in d if x is not None];overall.append(float(np.mean(valid)) if valid else None)
                bass=[(r['scores']['raw']['metrics'][2]['target_gain'],r['scores'][name]['metrics'][2]['target_gain']) for r in group]
                per['overall']=paired_stat(overall)
                per['bass_collapse']={'baseline':sum(a<.1 for a,b in bass if a is not None),'candidate':sum(b<.1 for a,b in bass if b is not None),'valid':sum(a is not None and b is not None for a,b in bass),'new':sum(a>=.1 and b<.1 for a,b in bass if a is not None and b is not None)}
                summaries[c][name]=per
        wet=[r for r in rows if r['condition']=='wet_R4'];added=[]
        for r in wet:
            d=[diff(r,candidate,'raw_weighted',i) for i in range(4)];d=[x for x in d if x is not None];added.append(float(np.mean(d)) if d else None)
        out={'complete':len(rows)==20,'cases':len(rows),'neural_parameter_updates':0,'summary':summaries,'wpe_added_vs_raw_weighted':paired_stat(added), 'wpe_added_vs_gain_weighted':paired_stat([float(np.mean([diff(r,candidate,'gain_weighted',i) for i in range(4) if diff(r,candidate,'gain_weighted',i) is not None])) for r in wet]),'primary_candidate':candidate,'binding':self.digest(self.binding)}
        self.dump(self.run/'summary.json',out)
        destination=Path(__file__).parent/'결과';destination.mkdir(exist_ok=True)
        self.dump(destination/'분석요약.json',out)
        text=['# 입력 전처리 재검증 결과','',f"완료 조건 {len(rows)}/20 · 고정 HTDemucs · 신경망 가중치 학습 없음",'',
              '동일 validation10곡의 dry/wet를 비교했다. 모든 방법은 40초 입력·중앙30초·원래 잔향 정답을 유지한다. 기존190초EQ결과와 직접 수치 합산하지 않는다. WPE는NARA0.0.11,taps10,delay3,3회반복으로고정했다.','',
              '| 조건 | 방법 | 전체 ΔSI-SDR | bass ΔSI-SDR | bass 소실 기준→후 |','|---|---|---|---|---|']
        def fmt(s):return '미정의' if s['median'] is None else f"{s['median']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]"
        for c in ['dry','wet_R4']:
            for name,v in summaries[c].items():
                b=v['bass_collapse'];text.append(f"| {c} | {name} | {fmt(v['overall'])} | {fmt(v['bass'])} | {b['baseline']}/{b['valid']}→{b['candidate']}/{b['valid']} |")
        primary=summaries['wet_R4'][candidate]['overall'];passed=out['complete'] and primary['median']>=.2 and primary['ci'][0]>0
        text+=['',f"사전 주요 후보 WPE+weighted의 기본 개선 기준(+0.2dB,CI하한>0)은 {('통과' if passed else '통과하지 못했다') if out['complete'] else '미완료로 판정 보류'}. 비전처리 weighted 대조 대비 추가 효과는 {fmt(out['wpe_added_vs_raw_weighted'])} dB다.",'',
               '음량 대조 gain_weighted 대비 추가 효과는 '+fmt(out['wpe_added_vs_gain_weighted'])+' dB다. 다른 음원의 손상·dry 손상·새 소실은 함께 검토해야 한다. 주 후보가 실패한 경우 다른 출력 중 가장 좋은 값으로 성공 선언을 바꾸지 않는다. 이 자료는 validation으로 이미 노출됐고 기초 모델 학습 중복도 미확인이므로 탐색적 결과다.','',
               '평가값은 같은 곡의 대응 차이 중앙값과10,000회곡bootstrap95%CI다. 무음정답은수치미정의로보존하고관측수에반영했다. WPE가제거한원래잔향을정확히복원했다고가정하지않는다.','',
               '[실험 계획](../PLAN.md) · [설정](../protocol.json)']
        (destination/'결과보고서.md').write_text('\n'.join(text)+'\n')
        csvrows=[]
        for r in rows:
            for name,scores in r['scores'].items():
                for m in scores['metrics']:csvrows.append({'track':r['track'],'condition':r['condition'],'output':name,**{k:v for k,v in m.items() if k!='metric_states'}})
        fields=['track','condition','output','stem','state','si_sdr','si_sir','si_sar','sd_sdr','target_gain','target_rms','output_rms','rank']
        with (destination/'곡별결과.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(csvrows)
        return out

    def execute(self,limit):
        from banelp_refine.common import exclusive
        with exclusive(self.run,'run'):
            for t in self.tracks[:limit]:
                for c in self.protocol['conditions']:self.case(t,c);gc.collect()
            if self.sha(__file__)!=self.binding['source']:raise ValueError('code changed during execution')
            result=self.report();self.emit('frontend_finished',complete=result['complete'],cases=result['cases'],new_gpu_calls=self.calls)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--suite',required=True);p.add_argument('--baseline-repo',required=True);p.add_argument('--baseline-runtime',required=True);p.add_argument('--prepared-runtime',required=True);p.add_argument('--limit',type=int,default=10)
    a=p.parse_args();Experiment(a).execute(a.limit)


if __name__=='__main__':main()

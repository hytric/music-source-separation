"""Matched validation of official PaRIRset checkpoints and existing baselines."""
from __future__ import annotations
import argparse,csv,gc,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from adapter import OfficialSCNet


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def wave_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x,dtype='<f4').tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',required=True);p.add_argument('--baseline-repo',required=True)
    p.add_argument('--frontend-source',required=True);p.add_argument('--runtime',required=True)
    p.add_argument('--frontend-run',required=True);p.add_argument('--official',required=True)
    p.add_argument('--baseline-runtime',required=True);p.add_argument('--prepared-runtime',required=True)
    p.add_argument('--stage',choices=['prepare','clean','rev','bsr','score'],required=True)
    args=p.parse_args();sys.path[:0]=[args.suite,args.baseline_repo]
    from banelp_refine.common import SOURCES,atomic_json,read_json,sha256,digest,verify_entry,file_entry,environment,exclusive,emit
    from cause_eq.metrics import score_sets,equivalence_check
    from experiments.banelp.separators import MSSTSeparator,verify_model_record
    torch.set_num_threads(2);torch.manual_seed(20261113)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(args.runtime);root.mkdir(parents=True,exist_ok=True)
    front=Path(args.frontend_run);old=read_json(front/'binding.json')
    frontend=module(args.frontend_source,'previous_frontend')
    prep=Path(args.prepared_runtime);base=Path(args.baseline_runtime);assets=Path(args.official)
    if sha256(args.frontend_source)!=old['source']:raise ValueError('frontend construction code changed')
    for path,h in old['manifests'].items():
        if sha256(path)!=h:raise ValueError('manifest changed')
    for path,h in old['shared_code'].items():
        if sha256(Path(args.suite)/path)!=h:raise ValueError('shared evaluation code changed')
    for name,h in old['baseline_code'].items():
        if sha256(Path(args.baseline_repo)/'experiments/banelp'/name)!=h:raise ValueError('baseline code changed')
    acquisition=read_json(assets/'acquisition.json')
    for f in acquisition['files']:
        if sha256(assets/f['path'])!=f['sha256']:raise ValueError('official assets changed')
    records=read_json(base/'manifests/models.json')['models']
    htd=next(m for m in records if m['model_id']=='htdemucs')
    bsr=next(m for m in records if m['model_id']=='msst_bsroformer_ep17')
    if htd!=old['model']:raise ValueError('cached HTDemucs model changed')
    verify_model_record(htd);verify_model_record(bsr)
    protocol=read_json(Path(__file__).with_name('protocol.json'))
    binding={'protocol':protocol,'environment':environment(),'official':acquisition,
        'official_source':{str(f.relative_to(assets)):sha256(f) for f in sorted((assets/'source').rglob('*.py'))},
        'code':{f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')},
        'frontend_binding_sha256':sha256(front/'binding.json'),'models':{'htdemucs':htd,'bsr':bsr}}
    binding_id=digest(binding)
    if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ValueError('run binding changed')
    atomic_json(root/'binding.json',binding)
    tracks=sorted([t for t in read_json(prep/'manifests/dataset.json')['tracks'] if t['split']=='validation'],key=lambda t:t['name'])
    if len(tracks)!=10:raise ValueError('wrong split')
    builder=frontend.Experiment.__new__(frontend.Experiment)
    builder.sources=SOURCES;builder.prep=prep;builder.verify=verify_entry;builder.rirs={}
    rm=read_json(base/'manifests/rirs.json')
    for s,pos in zip(SOURCES,('P1','P2','P3','P4')):
        e=rm['rooms']['R4']['positions'][pos];path=base/e['path']
        if sha256(path)!=e['sha256']:raise ValueError('RIR changed')
        builder.rirs[s]=np.load(path,allow_pickle=False)
    def cases():
        for t in tracks:
            dry=builder.dry_sources(t)
            for condition in ('dry','wet_R4'):
                y=builder.targets(dry,condition);x=y.astype(np.float64).sum(0).astype(np.float32)
                yield t,condition,x,y
    def folder(t,c):
        f=root/t['name']/c;f.mkdir(parents=True,exist_ok=True);return f
    def cached(t,c):
        meta=read_json(front/t['name']/c/'raw.json')
        if meta['binding']!=digest(old):raise ValueError('HTD cache binding mismatch')
        return np.load(verify_entry(front,meta['file']),allow_pickle=False)
    with exclusive(root,'run'):
        if args.stage=='prepare':
            for t,c,x,y in cases():
                raw=cached(t,c);r=score_sets(y[...,frontend.SL],{'htdemucs':raw[...,frontend.SL]})['htdemucs']
                previous=read_json(front/t['name']/c/'results.json')['scores']['raw']
                errors=[]
                for a,b in zip(r['metrics'],previous['metrics']):
                    for key in ('si_sdr','si_sir','si_sar','target_gain','target_rms','output_rms'):
                        if (a.get(key) is None)!=(b.get(key) is None):raise ValueError('cached metric state mismatch')
                        if a.get(key) is not None:errors.append(abs(a[key]-b[key]))
                error=max(errors)
                if error>1e-7:raise ValueError('reconstructed target/HTD cache score mismatch')
                atomic_json(folder(t,c)/'input.json',{'binding':binding_id,'input_sha256':wave_hash(x),'target_sha256':wave_hash(y),'htd_file':read_json(front/t['name']/c/'raw.json')['file'],'recomputed_metric_max_difference':error})
                emit('input_verified',track=t['name'],condition=c)
        elif args.stage in ('clean','rev','bsr'):
            sep=None
            for t,c,x,y in cases():
                f=folder(t,c);entry=read_json(f/'input.json')
                if entry['binding']!=binding_id or wave_hash(x)!=entry['input_sha256'] or wave_hash(y)!=entry['target_sha256']:raise ValueError('input changed')
                meta=f/(args.stage+'.json');out=f/(args.stage+'.npy')
                if meta.exists():
                    m=read_json(meta)
                    if m['binding']!=binding_id:raise ValueError('output binding mismatch')
                    verify_entry(root,m['file']);continue
                if sep is None:
                    sep=MSSTSeparator(registered=bsr,runtime_dir=base,device='cuda',batch_size=1) if args.stage=='bsr' else OfficialSCNet(assets,args.stage,segment=20)
                    atomic_json(root/(args.stage+'-attestation.json'),sep.attestation if args.stage=='bsr' else {'segment':20,'overlap':.5,'shifts':0,'strict':True,'fp32':True,'native_sources':sep.model.sources,'reorder':sep.order})
                torch.cuda.reset_peak_memory_stats();started=time.monotonic();z=sep(x)
                with out.with_suffix('.part').open('wb') as stream:np.save(stream,z,allow_pickle=False)
                out.with_suffix('.part').replace(out)
                m={'binding':binding_id,'input_sha256':entry['input_sha256'],'file':file_entry(out,root),'seconds':time.monotonic()-started,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
                atomic_json(meta,m);emit('inference_complete',model=args.stage,track=t['name'],condition=c,seconds=m['seconds']);del z;gc.collect()
        else:
            rows=[]
            for t,c,x,y in cases():
                f=folder(t,c);entry=read_json(f/'input.json')
                if entry['binding']!=binding_id or wave_hash(x)!=entry['input_sha256'] or wave_hash(y)!=entry['target_sha256']:raise ValueError('input changed')
                estimates={'htdemucs':cached(t,c)[...,frontend.SL]}
                for name in ('clean','rev','bsr'):
                    meta=read_json(f/(name+'.json'))
                    if meta['binding']!=binding_id or meta['input_sha256']!=entry['input_sha256']:raise ValueError('estimate binding mismatch')
                    estimates[name]=np.load(verify_entry(root,meta['file']),allow_pickle=False)[...,frontend.SL]
                estimates['ensemble']=(estimates['htdemucs']+estimates.pop('bsr'))*.5
                scores=score_sets(y[...,frontend.SL],estimates)
                check=equivalence_check(y[...,frontend.SL],estimates['rev'])
                row={'track':t['name'],'condition':c,'scores':scores,'legacy_metric_max_difference':check}
                atomic_json(f/'results.json',row);rows.append(row)
                emit('case_scored',track=t['name'],condition=c)
            summary={'complete':len(rows)==20,'cases':len(rows),'binding':binding_id,'comparisons':{},'legacy_metric_max_difference':max(r['legacy_metric_max_difference'] for r in rows)}
            for c in ('dry','wet_R4'):
                group=[r for r in rows if r['condition']==c];summary['comparisons'][c]={}
                for a,b in [('rev','clean'),('clean','htdemucs'),('rev','htdemucs'),('ensemble','htdemucs'),('rev','ensemble')]:
                    def delta(r,i):
                        va=r['scores'][a]['metrics'][i]['si_sdr'];vb=r['scores'][b]['metrics'][i]['si_sdr']
                        return None if va is None or vb is None else va-vb
                    per={s:frontend.paired_stat([delta(r,i) for r in group]) for i,s in enumerate(SOURCES)}
                    song=[]
                    for r in group:
                        d=[delta(r,i) for i in range(4)];v=[n for n in d if n is not None];song.append(float(np.mean(v)) if v else None)
                    per['overall']=frontend.paired_stat(song);per['improved_songs']=sum(v>0 for v in song if v is not None)
                    per['bass_collapse']={}
                    for name in (a,b):
                        gains=[r['scores'][name]['metrics'][2]['target_gain'] for r in group]
                        per['bass_collapse'][name]={'count':sum(g<.1 for g in gains if g is not None),'valid':sum(g is not None for g in gains)}
                    summary['comparisons'][c][a+'-minus-'+b]=per
            dest=Path(__file__).parent/'결과';dest.mkdir(exist_ok=True)
            atomic_json(dest/'분석요약.json',summary);atomic_json(root/'summary.json',summary)
            csvrows=[]
            for r in rows:
                for model,scores in r['scores'].items():
                    for m in scores['metrics']:csvrows.append({'track':r['track'],'condition':r['condition'],'model':model,**m})
            fields=['track','condition','model','stem','state','si_sdr','si_sir','si_sar','sd_sdr','target_gain','target_rms','output_rms','rank']
            with (dest/'곡별결과.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(csvrows)
            emit('comparison_complete',cases=len(rows))
    # Refuse provenance that changed during execution.
    for name,h in binding['code'].items():
        if sha256(Path(__file__).with_name(name))!=h:raise ValueError('source changed during run')

if __name__=='__main__':main()

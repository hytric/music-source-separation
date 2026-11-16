"""Evaluate fixed stem mixing using existing, hash-bound predictions only."""
from __future__ import annotations
import argparse,hashlib,importlib.util,sys,time,traceback
from pathlib import Path
import numpy as np
from core import combine,reconstruction,compare_metrics,mean_si


def wave_hash(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f4').tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('runtime','prepared','baseline-runtime','fine-runtime','official-run','frontend-run','suite','baseline-repo','frontend-source'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();sys.path[:0]=[args.suite,args.baseline_repo]
    from banelp_refine.common import read_json,atomic_json,sha256,digest,verify_entry,file_entry,environment,exclusive,emit,SOURCES
    from cause_eq.metrics import score_sets,equivalence_check
    from report import summarize,write_report
    root=Path(args.runtime);root.mkdir(parents=True,exist_ok=True)
    prep=Path(args.prepared);base=Path(args.baseline_runtime);fine=Path(args.fine_runtime);prior=Path(args.official_run);front=Path(args.frontend_run)
    config=read_json(Path(__file__).with_name('protocol.json'))
    fbind=read_json(fine/'binding.json');pbind=read_json(prior/'binding.json');bbind=read_json(front/'binding.json');state=read_json(fine/'status.json')
    if state['status']!='complete' or state['selected_step']!=1000 or state['binding']!=digest(fbind):raise ValueError('fine-tune is not the selected completed run')
    if sha256(front/'binding.json')!=pbind['frontend_binding_sha256']:raise ValueError('HTD cache provenance changed')
    for name,h in fbind['baseline_files'].items():
        if sha256(prior/name)!=h:raise ValueError('official baseline record changed')
    for name,h in fbind['manifests'].items():
        if sha256(prep/'manifests'/name)!=h:raise ValueError('dataset manifest changed')
    for name,h in fbind['shared_code'].items():
        if sha256(Path(args.suite)/name)!=h:raise ValueError('metric/helper definitions changed')
    if sha256(args.frontend_source)!=fbind['frontend_source'] or sha256(base/'manifests/rirs.json')!=fbind['baseline_rirs']:raise ValueError('signal construction changed')
    checkpoint_sha=sha256(fine/'step-1000.pt')
    spec=importlib.util.spec_from_file_location('hybrid_frontend',args.frontend_source);frontend=importlib.util.module_from_spec(spec);spec.loader.exec_module(frontend)
    builder=frontend.Experiment.__new__(frontend.Experiment);builder.sources=SOURCES;builder.prep=prep;builder.verify=verify_entry;builder.rirs={}
    rm=read_json(base/'manifests/rirs.json')
    for s,pos in zip(SOURCES,('P1','P2','P3','P4')):
        e=rm['rooms']['R4']['positions'][pos];path=base/e['path']
        if sha256(path)!=e['sha256']:raise ValueError('RIR bytes changed')
        builder.rirs[s]=np.load(path,allow_pickle=False)
    tracks=sorted([t for t in read_json(prep/'manifests/dataset.json')['tracks'] if t['split']=='validation'],key=lambda t:t['name'])
    if len(tracks)!=10:raise ValueError('wrong validation scope')
    input_files=sorted((fine/'evaluation/step-1000').glob('*/*/results.json'))
    if len(input_files)!=20:raise ValueError('fine predictions incomplete')
    binding={'protocol':config,'environment':environment(),'code':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'fine_binding_sha256':sha256(fine/'binding.json'),'prior_binding_sha256':sha256(prior/'binding.json'),'frontend_binding_sha256':sha256(front/'binding.json'),
        'fine_status_sha256':sha256(fine/'status.json'),'checkpoint_sha256':checkpoint_sha,'fine_result_files':{str(f.relative_to(fine)):sha256(f) for f in input_files},
        'manifests':fbind['manifests'],'shared_code':fbind['shared_code'],'frontend_source':sha256(args.frontend_source),'baseline_rirs_sha256':fbind['baseline_rirs']}
    binding_id=digest(binding)
    if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ValueError('run changed; use a new runtime')
    atomic_json(root/'binding.json',binding)
    def code_check():
        for name,h in binding['code'].items():
            if sha256(Path(__file__).with_name(name))!=h:raise ValueError('code changed during evaluation')
    def load_prediction(path,entry):
        z=np.load(verify_entry(path,entry),allow_pickle=False)
        if z.shape!=(4,2,1764000) or z.dtype!=np.float32 or not np.isfinite(z).all():raise ValueError('prediction contract changed')
        return z
    with exclusive(root,'evaluate'):
        started=time.monotonic();rows=[]
        atomic_json(root/'status.json',{'status':'running','cases':0,'binding':binding_id})
        try:
            for track in tracks:
                dry=builder.dry_sources(track)
                for c in config['conditions']:
                    folder=root/track['name']/c;folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'results.json').exists():
                        row=read_json(folder/'results.json')
                        if row['binding']!=binding_id:raise ValueError('cached hybrid changed')
                        verify_entry(root,row['prediction']);rows.append(row);continue
                    source=fine/'evaluation/step-1000'/track['name']/c
                    frow=read_json(source/'results.json');info=read_json(prior/track['name']/c/'input.json')
                    if frow['binding']!=digest(fbind) or frow['checkpoint_sha256']!=checkpoint_sha or frow['step']!=1000 or info['binding']!=digest(pbind):raise ValueError('prediction binding changed')
                    y=builder.targets(dry,c);x=y.astype(np.float64).sum(0).astype(np.float32)
                    if wave_hash(x)!=info['input_sha256'] or wave_hash(y)!=info['target_sha256'] or frow['input_sha256']!=info['input_sha256'] or frow['target_sha256']!=info['target_sha256']:raise ValueError('signal mismatch')
                    hm=read_json(front/track['name']/c/'raw.json');bm=read_json(prior/track['name']/c/'bsr.json')
                    if hm['binding']!=digest(bbind) or hm['file']!=info['htd_file'] or bm['binding']!=digest(pbind) or bm['input_sha256']!=info['input_sha256']:raise ValueError('baseline cache mismatch')
                    htd=load_prediction(front,hm['file']);bsr=load_prediction(prior,bm['file']);b=(htd+bsr)*.5;del htd,bsr
                    f=load_prediction(fine,frow['prediction']);h=combine(b,f)
                    protected_error=float(np.max(np.abs(h[[0,3]]-b[[0,3]])))
                    if protected_error!=0:raise ValueError('protected stem changed')
                    path=folder/'hybrid.npy'
                    with path.with_suffix('.part').open('wb') as stream:np.save(stream,h,allow_pickle=False)
                    path.with_suffix('.part').replace(path)
                    scores=score_sets(y[...,frontend.SL],{'baseline':b[...,frontend.SL],'finetuned':f[...,frontend.SL],'hybrid':h[...,frontend.SL]})
                    eb=compare_metrics(scores['baseline'],frow['scores']['ensemble']);ef=compare_metrics(scores['finetuned'],frow['scores']['candidate'])
                    expected={'metrics':[scores['baseline']['metrics'][0],scores['finetuned']['metrics'][1],scores['finetuned']['metrics'][2],scores['baseline']['metrics'][3]]}
                    eh=compare_metrics(scores['hybrid'],expected)
                    legacy=equivalence_check(y[...,frontend.SL],h[...,frontend.SL]) if not rows else None
                    diagnostics={name:reconstruction(x[...,frontend.SL],z[...,frontend.SL]) for name,z in [('baseline',b),('finetuned',f),('hybrid',h)]}
                    condition_choice='baseline' if c=='dry' else 'hybrid'
                    oracle_choice='hybrid' if mean_si(scores['hybrid'])>mean_si(scores['baseline']) else 'baseline'
                    scores['condition_label_route']=scores[condition_choice];scores['gt_case_oracle']=scores[oracle_choice]
                    diagnostics['condition_label_route']=diagnostics[condition_choice];diagnostics['gt_case_oracle']=diagnostics[oracle_choice]
                    row={'track':track['name'],'condition':c,'binding':binding_id,'input_sha256':info['input_sha256'],'target_sha256':info['target_sha256'],
                        'prediction':file_entry(path,root),'input_predictions':{'htdemucs':hm['file'],'bsr':bm['file'],'finetuned':frow['prediction']},'checkpoint_sha256':checkpoint_sha,
                        'scores':scores,'reconstruction':diagnostics,'diagnostic_choices':{'condition_label_route':condition_choice,'gt_case_oracle':oracle_choice},
                        'checks':{'protected_stem_max_abs':protected_error,'baseline_metric_max_error':eb,'finetuned_metric_max_error':ef,'spliced_metric_max_error':eh,'legacy_metric_max_error':legacy}}
                    atomic_json(folder/'results.json',row);rows.append(row)
                    atomic_json(root/'status.json',{'status':'running','cases':len(rows),'binding':binding_id})
                    emit('hybrid_case_complete',case=len(rows),track=track['name'],condition=c)
                    del y,x,b,f,h
            code_check();summary=summarize(rows,config);summary.update({'binding':binding_id,'cases':len(rows),'complete':len(rows)==20,'seconds':time.monotonic()-started,'new_gpu_inferences':0,'training_updates':0})
            atomic_json(root/'summary.json',summary)
            write_report(root,Path(__file__).parent,rows,summary,config)
            atomic_json(root/'status.json',{'status':'complete','cases':len(rows),'fixed_hybrid_accepted':summary['accepted'],'binding':binding_id,'seconds':time.monotonic()-started})
            emit('hybrid_complete',cases=len(rows),accepted=summary['accepted'],wet=summary['comparisons']['wet_R4']['hybrid']['overall'],dry=summary['comparisons']['dry']['hybrid']['overall'])
        except Exception as error:
            atomic_json(root/'status.json',{'status':'failed','cases':len(rows),'binding':binding_id,'error':str(error)})
            atomic_json(root/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()

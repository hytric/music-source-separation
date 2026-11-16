"""Evaluate two-source consistency projections on frozen routed outputs."""
from pathlib import Path
import argparse,hashlib,importlib.util,sys,time,traceback
import numpy as np
from core import project,reconstruction


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def wave_hash(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f4').tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('project','assets','runtime','baseline-repo'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();home=Path(args.project);assets=Path(args.assets);root=Path(args.runtime);root.mkdir(parents=True,exist_ok=True)
    suite=home/'학습실험_2026-11-11';sys.path[:0]=[str(suite),args.baseline_repo]
    from banelp_refine.common import read_json,atomic_json,sha256,digest,verify_entry,file_entry,environment,exclusive,emit,SOURCES
    from cause_eq.metrics import score_sets,equivalence_check
    from report import summarize,write_report
    config=read_json(Path(__file__).with_name('protocol.json'))
    hybrid=assets/'stem-hybrid-v1';selector=assets/'input-selector-v1';prep=assets/'banelp-refine-v1';base=assets/'banelp-v1';front=assets/'frontend-verify-v1/run-9441074cf871f26a';prior=assets/'parirset-compare-v1'
    hb=read_json(hybrid/'binding.json');sb=read_json(selector/'binding.json');model=read_json(selector/'model.json')
    for parent,binding in ((hybrid,hb),(selector,sb)):
        status=read_json(parent/'status.json')
        if status['status']!='complete' or status['binding']!=digest(binding):raise ValueError('parent run incomplete or changed')
    if sha256(hybrid/'binding.json')!=sb['parent_binding_sha256'] or sha256(hybrid/'status.json')!=sb['parent_status_sha256']:raise ValueError('selector parent changed')
    for path,h in sb['parent_results'].items():
        if sha256(hybrid/path)!=h:raise ValueError('hybrid scores changed')
    for path,h in hb['manifests'].items():
        if sha256(prep/'manifests'/path)!=h:raise ValueError('data manifest changed')
    for path,h in hb['shared_code'].items():
        if sha256(suite/path)!=h:raise ValueError('metric definitions changed')
    frontend_path=home/'입력전처리검증_2026-11-13/run.py'
    if sha256(frontend_path)!=hb['frontend_source'] or sha256(base/'manifests/rirs.json')!=hb['baseline_rirs_sha256']:raise ValueError('signal construction changed')
    old_core=home/'음원별결합검증_2026-11-14/core.py'
    if sha256(old_core)!=hb['code']['core.py']:raise ValueError('metric validation helper changed')
    frontend=module(frontend_path,'consistency_frontend');validation=module(old_core,'consistency_validation')
    selected=read_json(selector/'rows.json');choices={(r['track'],r['condition']):r for r in selected}
    if len(choices)!=20 or model['threshold']!=.8 or model['binding']!=digest(sb):raise ValueError('selector policy changed')
    for row in selected:
        if row['choice']!=('hybrid' if row['sigmoid_score']>=.8 else 'baseline'):raise ValueError('recorded route disagrees with policy')
    binding={'protocol':config,'environment':environment(),'code':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'selector_binding_sha256':sha256(selector/'binding.json'),'selector_model_sha256':sha256(selector/'model.json'),'selector_rows_sha256':sha256(selector/'rows.json'),
        'hybrid_binding_sha256':sha256(hybrid/'binding.json'),'hybrid_result_sha256':sb['parent_results'],'manifests':hb['manifests'],'shared_code':hb['shared_code'],
        'frontend_source_sha256':sha256(frontend_path),'validation_helper_sha256':sha256(old_core),'baseline_rirs_sha256':sha256(base/'manifests/rirs.json')}
    binding_id=digest(binding)
    if (root/'binding.json').exists():raise ValueError('use a fresh consistency runtime')
    atomic_json(root/'binding.json',binding)
    builder=frontend.Experiment.__new__(frontend.Experiment);builder.sources=SOURCES;builder.prep=prep;builder.verify=verify_entry;builder.rirs={}
    rm=read_json(base/'manifests/rirs.json')
    for s,pos in zip(SOURCES,('P1','P2','P3','P4')):
        e=rm['rooms']['R4']['positions'][pos];path=base/e['path']
        if sha256(path)!=e['sha256']:raise ValueError('evaluation RIR changed')
        builder.rirs[s]=np.load(path,allow_pickle=False)
    tracks=sorted([t for t in read_json(prep/'manifests/dataset.json')['tracks'] if t['split']=='validation'],key=lambda t:t['name'])
    if len(tracks)!=10:raise ValueError('validation scope changed')
    with exclusive(root,'evaluate'):
        rows=[];started=time.monotonic();atomic_json(root/'status.json',{'status':'running','cases':0,'binding':binding_id})
        try:
            for track in tracks:
                dry=builder.dry_sources(track)
                for condition in config['conditions']:
                    old=read_json(hybrid/track['name']/condition/'results.json');route=choices[(track['name'],condition)]
                    y=builder.targets(dry,condition);x=y.astype(np.float64).sum(0).astype(np.float32)
                    if wave_hash(x)!=old['input_sha256'] or wave_hash(y)!=old['target_sha256'] or route['input_sha256']!=old['input_sha256']:raise ValueError('evaluation waveform changed')
                    h=np.load(verify_entry(hybrid,old['prediction']),allow_pickle=False)
                    htd=np.load(verify_entry(front,old['input_predictions']['htdemucs']),allow_pickle=False);bsr=np.load(verify_entry(prior,old['input_predictions']['bsr']),allow_pickle=False)
                    b=(htd+bsr)*.5;del htd,bsr
                    raw=h if route['choice']=='hybrid' else b
                    bu=project(x,b,'uniform');bp=project(x,b,'power',frontend.stft,frontend.istft)
                    if route['choice']=='baseline':ru,rp=bu,bp
                    else:ru=project(x,raw,'uniform');rp=project(x,raw,'power',frontend.stft,frontend.istft)
                    estimates={'baseline':b,'selected_raw':raw,'selected_uniform':ru,'selected_power':rp,'baseline_uniform':bu,'baseline_power':bp}
                    folder=root/track['name']/condition;folder.mkdir(parents=True,exist_ok=True);files={};checks={}
                    for name in ('selected_uniform','selected_power','baseline_uniform','baseline_power'):
                        z=estimates[name];reference=raw if name.startswith('selected') else b
                        if z[[0,3]].tobytes()!=reference[[0,3]].tobytes():raise ValueError('protected stems changed')
                        checks[name]={'protected_max_abs':float(np.max(abs(z[[0,3]]-reference[[0,3]]))),'full40':reconstruction(x,z),'core30':reconstruction(x[...,frontend.SL],z[...,frontend.SL])}
                        path=folder/(name+'.npy')
                        with path.with_suffix('.part').open('wb') as stream:np.save(stream,z,allow_pickle=False)
                        path.with_suffix('.part').replace(path);files[name]=file_entry(path,root)
                    for name in ('baseline','selected_raw'):
                        checks[name]={'protected_max_abs':0.,'full40':reconstruction(x,estimates[name]),'core30':reconstruction(x[...,frontend.SL],estimates[name][...,frontend.SL])}
                    scores=score_sets(y[...,frontend.SL],{name:z[...,frontend.SL] for name,z in estimates.items()})
                    b_error=validation.compare_metrics(scores['baseline'],old['scores']['baseline']);r_error=validation.compare_metrics(scores['selected_raw'],route['selected_scores'])
                    legacy=equivalence_check(y[...,frontend.SL],rp[...,frontend.SL]) if not rows else None
                    row={'track':track['name'],'condition':condition,'route':route['choice'],'sigmoid_score':route['sigmoid_score'],'binding':binding_id,'input_sha256':old['input_sha256'],'target_sha256':old['target_sha256'],
                        'files':files,'scores':scores,'checks':checks,'baseline_metric_max_error':b_error,'selected_metric_max_error':r_error,'legacy_metric_max_error':legacy}
                    atomic_json(folder/'results.json',row);rows.append(row);atomic_json(root/'status.json',{'status':'running','cases':len(rows),'binding':binding_id})
                    emit('consistency_case_complete',case=len(rows),track=track['name'],condition=condition)
                    del x,y,b,h,raw,bu,bp,ru,rp,estimates,z,reference
            for name,h in binding['code'].items():
                if sha256(Path(__file__).with_name(name))!=h:raise ValueError('source changed')
            summary=summarize(rows,config);summary.update({'complete':len(rows)==20,'cases':len(rows),'binding':binding_id,'seconds':time.monotonic()-started,'new_gpu_inferences':0,'training_updates':0,'deployment_ready':False})
            atomic_json(root/'summary.json',summary);write_report(root,Path(__file__).parent,rows,summary,config)
            atomic_json(root/'status.json',{'status':'complete','cases':20,'binding':binding_id,'accepted':summary['accepted']})
            emit('consistency_complete',accepted=summary['accepted'],gates=summary['gates'])
        except Exception as e:
            atomic_json(root/'status.json',{'status':'failed','cases':len(rows),'binding':binding_id,'error':str(e)});atomic_json(root/'failure.json',{'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()

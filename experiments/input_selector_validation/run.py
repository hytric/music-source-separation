"""Fit a condition selector on train mixtures, then route exposed validation."""
from pathlib import Path
import argparse,importlib.util,sys,time,csv,json,hashlib
import numpy as np
from core import features,fit,predict


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser()
    for name in ('project','assets','runtime','baseline-repo'):p.add_argument('--'+name,required=True)
    args=p.parse_args();project=Path(args.project);assets=Path(args.assets);root=Path(args.runtime);root.mkdir(parents=True,exist_ok=True)
    suite=project/'학습실험_2026-11-11';sys.path[:0]=[str(suite),args.baseline_repo]
    from banelp_refine.common import read_json,atomic_json,sha256,digest,verify_entry,environment,exclusive,emit,SOURCES
    from evaluate import evaluate,write_report
    config=read_json(Path(__file__).with_name('protocol.json'));hybrid=assets/'stem-hybrid-v1';prep=assets/'banelp-refine-v1';base=assets/'banelp-v1'
    parent_status=read_json(hybrid/'status.json');parent_binding=read_json(hybrid/'binding.json')
    if parent_status['status']!='complete' or parent_status['binding']!=digest(parent_binding):raise ValueError('fixed hybrid provenance incomplete')
    cache=read_json(prep/'manifests/cache.json');dataset=read_json(prep/'manifests/dataset.json');rirs=read_json(prep/'manifests/rirs.json')
    for name,h in parent_binding['manifests'].items():
        if sha256(prep/'manifests'/name)!=h:raise ValueError('training manifest changed')
    train=[r for r in cache['records'] if r['split']=='train'];names={t['name'] for t in dataset['tracks'] if t['split']=='train'};rir_splits={r['id']:r['split'] for r in rirs['sets']}
    if len(train)!=320 or len(names)!=80:raise ValueError('wrong training scope')
    for r in train:
        if r['track'] not in names or (r['rir_set'] is not None and rir_splits[r['rir_set']]!='train'):raise ValueError('training leakage')
    front_path=project/'입력전처리검증_2026-11-13/run.py';frontend=load(front_path,'selector_frontend')
    if sha256(front_path)!=parent_binding['frontend_source']:raise ValueError('signal code changed')
    parent_files=sorted(hybrid.glob('*/*/results.json'))
    binding={'protocol':config,'environment':environment(),'code':{f.name:sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'parent_binding_sha256':sha256(hybrid/'binding.json'),'parent_status_sha256':sha256(hybrid/'status.json'),'parent_results':{str(f.relative_to(hybrid)):sha256(f) for f in parent_files},
        'manifests':parent_binding['manifests'],'frontend_source':sha256(front_path),'baseline_rirs_sha256':sha256(base/'manifests/rirs.json')}
    binding_id=digest(binding)
    if (root/'binding.json').exists():raise ValueError('use a fresh selector runtime')
    atomic_json(root/'binding.json',binding)
    with exclusive(root,'selector'):
        started=time.monotonic();atomic_json(root/'status.json',{'status':'training_features','records':0,'binding':binding_id})
        x=[];labels=[]
        for i,r in enumerate(train):
            wave=np.load(verify_entry(prep,r['files']['mixture']),allow_pickle=False)
            if wave.shape!=(2,1323000) or wave.dtype!=np.float32:raise ValueError('mixture format changed')
            x.append(features(wave,frontend.stft));labels.append(int(r['rir_set'] is not None))
            if (i+1)%40==0:emit('selector_train_features',done=i+1,total=320)
        model=fit(np.asarray(x),np.asarray(labels),l2=config['l2'],max_iterations=config['max_iterations']);model.update({'threshold':config['threshold'],'binding':binding_id,'training_records':320,'training_songs':80,'labels':'synthetic added-RIR status from training manifest only'})
        atomic_json(root/'model.json',model)
        training_probability=predict(np.asarray(x),model)
        atomic_json(root/'training-fit.json',{'training_accuracy':float(np.mean((training_probability>=config['threshold'])==np.asarray(labels))),'positive_selected':int(np.sum(training_probability>=config['threshold'])),'iterations':model['iterations']})
        builder=frontend.Experiment.__new__(frontend.Experiment);builder.sources=SOURCES;builder.prep=prep;builder.verify=verify_entry;builder.rirs={}
        rm=read_json(base/'manifests/rirs.json')
        for s,pos in zip(SOURCES,('P1','P2','P3','P4')):
            e=rm['rooms']['R4']['positions'][pos];f=base/e['path']
            if sha256(f)!=e['sha256']:raise ValueError('RIR changed')
            builder.rirs[s]=np.load(f,allow_pickle=False)
        val=sorted([t for t in dataset['tracks'] if t['split']=='validation'],key=lambda t:t['name']);rows=[]
        if len(val)!=10 or len(parent_files)!=20:raise ValueError('wrong validation scope')
        for t in val:
            dry=builder.dry_sources(t)
            for condition in ('dry','wet_R4'):
                y=builder.targets(dry,condition);wave=y.astype(np.float64).sum(0).astype(np.float32)
                prior=read_json(hybrid/t['name']/condition/'results.json')
                digest_wave=hashlib.sha256(np.ascontiguousarray(wave,dtype='<f4').tobytes()).hexdigest()
                if digest_wave!=prior['input_sha256']:raise ValueError('validation waveform changed')
                verify_entry(hybrid,prior['prediction'])
                # Decision consumes only audio features and frozen training parameters.
                score=float(predict(features(wave,frontend.stft),model));choice='hybrid' if score>=model['threshold'] else 'baseline'
                row={'track':t['name'],'condition':condition,'sigmoid_score':score,'choice':choice,'input_sha256':digest_wave,'selected_scores':prior['scores'][choice],
                    'baseline_scores':prior['scores']['baseline'],'fixed_hybrid_scores':prior['scores']['hybrid'],'reconstruction':prior['reconstruction'][choice]}
                rows.append(row);emit('selector_case_complete',case=len(rows),track=t['name'],condition=condition,choice=choice,score=score)
        summary=evaluate(rows,config);summary.update({'binding':binding_id,'complete':len(rows)==20,'cases':len(rows),'seconds':time.monotonic()-started,'separator_weight_updates':0,'new_gpu_inferences':0})
        atomic_json(root/'rows.json',rows);atomic_json(root/'summary.json',summary)
        for name,h in binding['code'].items():
            if sha256(Path(__file__).with_name(name))!=h:raise ValueError('code changed')
        write_report(root,Path(__file__).parent,summary,rows,model)
        atomic_json(root/'status.json',{'status':'complete','accepted':summary['accepted'],'cases':20,'binding':binding_id})
        emit('selector_complete',accepted=summary['accepted'],summary=summary['comparisons'])

if __name__=='__main__':main()

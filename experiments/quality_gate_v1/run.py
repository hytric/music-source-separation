"""QG1 milestone1: verified lock, split, full-convolution inputs and teacher stage."""
from pathlib import Path
import argparse,importlib.util,json,sys,time,shutil,subprocess
import numpy as np
from scipy import signal
from core import split_songs


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','prepare','cache'])
    p.add_argument('--project',required=True);p.add_argument('--runtime',required=True);p.add_argument('--limit',type=int,default=320)
    p.add_argument('--model',choices=['htdemucs','msst_bsroformer_ep17','finetuned_scnet']);args=p.parse_args()
    project=Path(args.project).resolve();root=Path(args.runtime).resolve();root.mkdir(parents=True,exist_ok=True)
    guide=project/'전체실험검수_2026-11-14/구현가이드';config=json.loads((guide/'protocol.template.json').read_text());assets=json.loads((guide/'assets.template.json').read_text())
    sys.path[:0]=[str(project/'학습실험_2026-11-11'),assets['baseline_library_repo']]
    from banelp_refine.common import read_json,atomic_json,sha256,digest,verify_entry,file_entry,exclusive,emit,environment
    dataset_path=Path(assets['manifests']['dataset.json']['path']);prep=dataset_path.parent.parent
    for record in list(assets['manifests'].values())+[assets['model_registry'],assets['baseline_RIR_manifest']]:
        if sha256(record['path'])!=record['sha256']:raise ValueError('asset manifest changed')
    for path,h in assets['source_file_checks'].items():
        if sha256(path)!=h:raise ValueError('source dependency changed')
    if sha256(assets['models']['F']['checkpoint'])!=assets['models']['F']['sha256']:raise ValueError('F checkpoint changed')
    registry=read_json(assets['model_registry']['path'])['models']
    for id in ('htdemucs','msst_bsroformer_ep17'):
        registered=next(r for r in registry if r['model_id']==id)
        if registered!=assets['models'][id]:raise ValueError('registered model differs')
        for member in registered['members']:
            if sha256(member['path'])!=member['sha256']:raise ValueError('model weights changed')
        if registered['config_file'] and sha256(registered['config_file']['path'])!=registered['config_file']['sha256']:raise ValueError('model config changed')
    tracks=read_json(dataset_path)['tracks'];by_name={t['name']:t for t in tracks};train_names=[t['name'] for t in tracks if t['split']=='train']
    roles=split_songs(train_names,config['split']['salt'],config['split']['fit_songs'])
    records=[r for r in read_json(prep/'manifests/cache.json')['records'] if r['split']=='train']
    if len(records)!=320:raise ValueError('expected320 train records')
    records=sorted(records,key=lambda r:(r['track'],r['condition']))
    rirs=read_json(prep/'manifests/rirs.json')['sets'];rir_by_id={r['id']:r for r in rirs}
    for name in train_names:
        rows=[r for r in records if r['track']==name]
        if len(rows)!=4 or sum(r['rir_set'] is None for r in rows)!=1 or len({r['rir_set'] for r in rows})!=4:raise ValueError('song conditions changed')
    for r in records:
        if r['track'] not in roles or r['rir_set'] is not None and rir_by_id[r['rir_set']]['split']!='train':raise ValueError('split leakage')
    binding={'protocol':config,'assets_sha256':sha256(guide/'assets.template.json'),'code':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},'environment':environment()}
    locked={**config,'document_state':'implementation_milestone1_locked','binding':digest(binding)}
    if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ValueError('code/config changed: new runtime required')
    atomic_json(root/'binding.json',binding);atomic_json(root/'protocol.lock.json',locked)
    atomic_json(root/'split.json',{'salt':config['split']['salt'],'songs':roles,'fit_records':256,'calibration_records':64,'scope':'selector layer only; F pretrained on train80'})
    def gpu():
        values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0].split(',')
        return {'free_bytes':int(values[0])*1024**2,'temperature_C':int(values[1])}
    with exclusive(root,'qg1'):
        if args.stage=='preflight':
            info={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'binding':digest(binding),'train_records':320,'fit_songs':64,'calibration_songs':16,'disk_free_bytes':shutil.disk_usage(root).free,'gpu':gpu(),'teacher_predictions_generated':0}
            info['disk_ok']=info['disk_free_bytes']>=config['resources']['minimum_free_disk_bytes'];info['gpu_smoke_ok']=info['gpu']['free_bytes']>=config['resources']['initial_smoke_min_free_gpu_bytes']
            atomic_json(root/'preflight.json',info);emit('preflight',**info);return
        if args.stage=='prepare':
            if not 1<=args.limit<=320:raise ValueError('limit1..320 required')
            if shutil.disk_usage(root).free<config['resources']['minimum_free_disk_bytes']:raise RuntimeError('insufficient disk')
            front=load(project/'입력전처리검증_2026-11-13/run.py','qg1_frontend');builder=front.Experiment.__new__(front.Experiment);builder.sources=config['audio']['sources'];builder.prep=prep;builder.verify=verify_entry
            cache={};done=[];started=time.monotonic();last=None;dry=None
            for r in records[:args.limit]:
                folder=root/'inputs'/r['id'];folder.mkdir(parents=True,exist_ok=True);meta=folder/'complete.json'
                if meta.exists():
                    old=read_json(meta)
                    if old['binding']!=digest(binding):raise ValueError('input binding changed')
                    verify_entry(root,old['mixture']);done.append(old);continue
                if last!=r['track']:dry=builder.dry_sources(by_name[r['track']]);last=r['track']
                if r['rir_set'] is None:y=np.pad(dry,((0,0),(0,0),(front.GUARD,front.GUARD)))
                else:
                    item=rir_by_id[r['rir_set']]
                    if r['rir_set'] not in cache:cache[r['rir_set']]={s:np.load(verify_entry(prep,e),allow_pickle=False) for s,e in item['sources'].items()}
                    builder.rirs=cache[r['rir_set']]
                    if max(len(h) for h in builder.rirs.values())>front.GUARD:raise ValueError('RIR tail exceeds guard')
                    y=builder.targets(dry,'wet_R4') # uses the supplied training set, not evaluation R4.
                target=np.load(verify_entry(prep,r['files']['targets']),mmap_mode='r',allow_pickle=False)
                error=float(np.max(abs(y[...,front.SL]-target)))
                if error>config['audio']['core_equivalence_max_abs']:raise ValueError('30s target core changed')
                x=y.astype(np.float64).sum(0).astype(np.float32);path=folder/'mixture.npy'
                with path.with_suffix('.part').open('wb') as stream:np.save(stream,x,allow_pickle=False)
                path.with_suffix('.part').replace(path)
                value={'record':r,'role':roles[r['track']],'binding':digest(binding),'mixture':file_entry(path,root),'target40_sha256':__import__('hashlib').sha256(y.tobytes()).hexdigest(),'core_max_abs':error}
                atomic_json(meta,value);done.append(value)
                if len(done)%8==0:emit('inputs_prepared',done=len(done),requested=args.limit)
            atomic_json(root/'inputs.json',{'binding':digest(binding),'records':done,'expected':320,'complete':len(done)==320,'seconds':time.monotonic()-started})
            atomic_json(root/'status.json',{'stage':'inputs_ready' if len(done)==320 else 'preparation_partial','prepared':len(done),'expected':320,'teacher_outputs':0,'selector_fit':False});emit('prepare_done',prepared=len(done),seconds=time.monotonic()-started);return
        if not args.model:raise ValueError('--model required')
        info=gpu()
        if info['free_bytes']<config['resources']['initial_smoke_min_free_gpu_bytes'] or info['temperature_C']>=config['resources']['gpu_temperature_limit_C']:
            atomic_json(root/'gpu_status.json',{'stage':'resource_blocked','model':args.model,'gpu':info,'required_free_bytes':config['resources']['initial_smoke_min_free_gpu_bytes'],'teacher_outputs':0});emit('resource_blocked',**info);return
        # Avoid loading any model before the full dataset contract is ready.
        manifest=read_json(root/'inputs.json')
        if not 1<=args.limit<=320 or len(manifest['records'])<args.limit:raise RuntimeError('prepare requested inputs before teacher caching')
        if args.limit==320 and not manifest['complete']:raise RuntimeError('full cache requires all320 inputs')
        from experiments.banelp.separators import HTDemucsCUDA,MSSTSeparator,verify_model_record
        if args.model=='finetuned_scnet':
            import torch
            adapter=load(assets['models']['F']['adapter'],'qg1_official_adapter')
            sep=adapter.OfficialSCNet(assets['models']['F']['official_assets'],'rev',segment=20)
            checkpoint=torch.load(assets['models']['F']['checkpoint'],map_location='cpu',weights_only=True)
            if checkpoint['step']!=1000:raise ValueError('wrong F step')
            sep.model.load_state_dict(checkpoint['model'],strict=True);sep.model.eval()
        else:
            record=next(r for r in registry if r['model_id']==args.model);verify_model_record(record)
            sep=HTDemucsCUDA(model_name='htdemucs',segment=7.,overlap=.25,registered=record) if args.model=='htdemucs' else MSSTSeparator(registered=record,runtime_dir=Path(assets['runtime_root'])/'banelp-v1',device='cuda',batch_size=1)
        import torch
        completed=0
        for row in manifest['records'][:args.limit]:
            folder=root/'teachers'/args.model/row['record']['id'];folder.mkdir(parents=True,exist_ok=True);meta=folder/'complete.json'
            if meta.exists():
                old=read_json(meta)
                if old['binding']!=digest(binding) or old['input_sha256']!=row['mixture']['sha256']:raise ValueError('teacher binding changed')
                verify_entry(root,old['output']);completed+=1;continue
            current=gpu()
            if current['temperature_C']>=config['resources']['gpu_temperature_limit_C']:raise RuntimeError('GPU thermal limit')
            x=np.load(verify_entry(root,row['mixture']),allow_pickle=False);torch.cuda.reset_peak_memory_stats();started=time.monotonic();z=sep(x)
            if z.shape!=(4,2,config['audio']['input_frames']) or z.dtype!=np.float32 or not np.isfinite(z).all():raise ValueError('teacher contract failed')
            path=folder/'prediction.npy'
            with path.with_suffix('.part').open('wb') as stream:np.save(stream,z,allow_pickle=False)
            path.with_suffix('.part').replace(path)
            value={'binding':digest(binding),'input_sha256':row['mixture']['sha256'],'model':args.model,'output':file_entry(path,root),'sources':config['audio']['sources'],'precision':'fp32','seconds':time.monotonic()-started,'peak_reserved':torch.cuda.max_memory_reserved()}
            atomic_json(meta,value);completed+=1;emit('teacher_record_complete',model=args.model,completed=completed)
        atomic_json(root/(args.model+'-status.json'),{'stage':'cache_complete' if completed==320 else 'cache_partial','completed':completed,'expected':320,'binding':digest(binding)})

if __name__=='__main__':main()

"""Reproduction entry point for validation, preparation, smoke and sealed runs."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from . import contracts
from .contracts import config_copy, validate_practical_config, validate_tensors, sha256_file
from .metrics import project_metrics, mixture_residual
from .data import validate_acquired_dataset, atomic_json
from .candidates import identity, t1, t2, t5, consistency_iteration, donor_reseparation
from .separators import HTDemucsCUDA
from .selection import seal_selection, validate_seal, paired_summary, select_candidate
from .diagnostics import oracle_ratio_mask
from .metrics import project_metrics

def _parser():
    p=argparse.ArgumentParser(prog='python -m experiments.banelp.cli'); sub=p.add_subparsers(dest='command',required=True)
    for name in ('validate','prepare','smoke','develop','seal','evaluate'):
        q=sub.add_parser(name); q.add_argument('--runtime-dir',required=True); q.add_argument('--artifact-dir',required=True); q.add_argument('--config',required=True)
        if name=='validate': q.add_argument('--synthetic',action='store_true')
        if name=='prepare': q.add_argument('--track-root',required=True)
        if name=='smoke':
            q.add_argument('--device',default='cuda'); q.add_argument('--require-cuda',action='store_true'); q.add_argument('--split',default='dev',choices=('dev','evaluation')); q.add_argument('--track-index',type=int,default=0); q.add_argument('--seconds',type=int,default=5)
        if name=='evaluate': q.add_argument('--seal',required=True)
    return p

def _load(args):
    config=json.loads(Path(args.config).read_text(encoding='utf-8')); validate_practical_config(config)
    runtime=Path(args.runtime_dir).resolve(); artifact=Path(args.artifact_dir).resolve()
    runtime.mkdir(parents=True,exist_ok=True); artifact.mkdir(parents=True,exist_ok=True)
    return config,runtime,artifact

def _synthetic():
    rng=np.random.default_rng(20261108); n=32768
    stems=rng.normal(0,.02,(4,2,n)).astype(np.float32); x=stems.sum(axis=0,dtype=np.float64).astype(np.float32)
    validate_tensors(x,stems)
    target=stems[0]; exact=project_metrics(target,target,stems)
    outputs={'identity':identity(x,stems),'T1':t1(x,stems),'T2':t2(x,stems),'T5':t5(x,stems),'C2-negative-control':consistency_iteration(x,stems,0)}
    for name,out in outputs.items():
        validate_tensors(x,out)
        if not np.isfinite(out).all(): raise ValueError(f'{name} produced nonfinite output')
    if not np.allclose(outputs['identity'],stems): raise AssertionError('identity changed output')
    if exact['target_gain'] is None or abs(exact['target_gain']-1)>1e-8: raise AssertionError('projection identity gain failed')
    return {'synthetic_samples':n,'outputs':list(outputs),'identity_projection':exact,'mixture_residual':mixture_residual(x,stems)}

def _smoke_input(mixture, seconds, sample_rate):
    """First `seconds` of the fixed 30 s manifest crop, so 5 s and 30 s smokes share one lineage."""
    if mixture.shape!=(2,contracts.SAMPLES): raise ValueError(f'wet cache mixture must have shape (2,{contracts.SAMPLES}), got {mixture.shape}')
    return np.ascontiguousarray(mixture[:,:int(seconds*sample_rate)])

def _sample_metrics(estimates, targets):
    return {name:project_metrics(targets[i],estimates[i],targets)['si_sdr']['value_db']
            for i,name in enumerate(contracts.SOURCES)}

def _run_development(runtime, artifact, manifest, track_index=0):
    tracks=[t for t in manifest['tracks'] if t['split']=='dev']
    if len(tracks)!=10: raise ValueError('development requires exactly 10 dev tracks')
    methods={'T1-uniform':lambda x,e:t1(x,e),'T2-alpha1':lambda x,e:t2(x,e,1),
             'T2-alpha2':lambda x,e:t2(x,e,2),'T5-alpha1-lambda025':lambda x,e:t5(x,e,1,.25),
             'C2-iter1':lambda x,e:consistency_iteration(x,e,1),'C2-iter2':lambda x,e:consistency_iteration(x,e,2)}
    adapter_status={}
    try:
        import norbert  # noqa: F401
        from .candidates import norbert_wiener
        for count in (0,1,2): methods[f'C1-norbert-{count}']=lambda x,e,count=count:norbert_wiener(e,x,count)
        adapter_status['C1-norbert']='available'
    except ImportError: adapter_status['C1-norbert']='unavailable-dependency'
    methods['C-donor-other-lambda05']=lambda x,e:donor_reseparation(x,e,separator,donors=('other',),strength=.5)
    methods['C-donor-drums-lambda05']=lambda x,e:donor_reseparation(x,e,separator,donors=('drums',),strength=.5)
    rows={name:[] for name in methods}; baselines=[]; oracle=[]; separator=None
    for track in tracks:
        track_values={name:[] for name in methods}; base_values=[]
        for room in ('R1','R2','R3','R4'):
            root=runtime/'cache/wet'/'dev'/track['name']/room
            x=np.load(root/'mixture.npy',allow_pickle=False).astype(np.float32)
            targets=np.load(root/'targets.npy',allow_pickle=False).astype(np.float32)
            estimate_path=runtime/'cache/separation/htdemucs/dev'/track['name']/room/'estimates.npy'
            if estimate_path.is_file(): estimates=np.load(estimate_path,allow_pickle=False).astype(np.float32)
            else: estimates=None
            validate_tensors(x,targets,samples=contracts.SAMPLES)
            if estimates is None:
                if not __import__('torch').cuda.is_available(): raise RuntimeError(f'baseline cache missing and CUDA unavailable: {estimate_path}')
                if separator is None: separator=HTDemucsCUDA()
                estimates=separator(x)
                estimate_path.parent.mkdir(parents=True,exist_ok=True); np.save(estimate_path,estimates,allow_pickle=False)
            validate_tensors(x,estimates,samples=x.shape[-1])
            base=_sample_metrics(estimates,targets); base_values.extend(v for v in base.values() if v is not None)
            if separator is None and any(name.startswith('C-donor') for name in methods):
                try: separator=HTDemucsCUDA()
                except Exception as exc: adapter_status['donor-reseparation']=f'unavailable:{type(exc).__name__}:{exc}'
            for name,fn in methods.items():
                if name.startswith('C-donor') and adapter_status.get('donor-reseparation','available')!='available':
                    track_values[name].extend([float('nan')]*16); continue
                out=fn(x,estimates); current=_sample_metrics(out,targets)
                track_values[name].extend(current[k]-base[k] for k in base if current[k] is not None and base[k] is not None)
            # Oracle results are diagnostics only and never enter candidate selection.
            oracle_out=oracle_ratio_mask(x,estimates,targets,2); oracle.append({'track':track['name'],'room':room,'si_sdr':_sample_metrics(oracle_out,targets)})
        baselines.append({'track':track['name'],'mean_si_sdr':float(np.mean(base_values))})
        for name in methods: rows[name].append(track_values[name])
    summaries={}
    for name,values in rows.items():
        array=np.asarray(values,dtype=np.float64)
        if np.isfinite(array).all(): summaries[name]=paired_summary(array)
        else: summaries[name]={'status':adapter_status.get(name.split('-')[0],adapter_status.get('donor-reseparation','unavailable'))}
    result={'split':'dev','tracks':[t['name'] for t in tracks],'baseline':baselines,'candidate_summaries':summaries,
            'oracle_diagnostics':oracle,'adapter_status':adapter_status,'selection_status':'not-sealed','metric_contract':'four-source wet-target projection'}
    runtime_path=runtime/'results/develop.json'; atomic_json(runtime_path,result); atomic_json(artifact/'evidence/develop.json',result)
    return result

def main(argv=None):
    args=_parser().parse_args(argv); config,runtime,artifact=_load(args)
    if args.command=='validate':
        result={'config_sha256':hashlib.sha256(contracts.canonical_json(config)).hexdigest(),'synthetic':_synthetic() if args.synthetic else None,'status':'PASS'}
    elif args.command=='prepare':
        manifest=validate_acquired_dataset(args.track_root,runtime,runtime/'manifests/dataset.json')
        result={'tracks':len(manifest['tracks']),'dev':sum(t['split']=='dev' for t in manifest['tracks']),'evaluation':sum(t['split']=='evaluation' for t in manifest['tracks']),'dataset_manifest':str(runtime/'manifests/dataset.json')}
    elif args.command=='smoke':
        if not args.require_cuda or args.device!='cuda': raise ValueError('smoke requires --device cuda --require-cuda')
        if args.seconds not in (5,30): raise ValueError('smoke duration must be 5 or 30 seconds')
        manifest_path=runtime/'manifests/dataset.json'
        if not manifest_path.is_file(): raise FileNotFoundError('host-run dataset manifest is missing')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        tracks=[t for t in manifest['tracks'] if t['split']==args.split]
        if not 0<=args.track_index<len(tracks): raise IndexError('track-index outside split')
        track=tracks[args.track_index]['name']; mix_path=runtime/'cache/wet'/args.split/track/'R4'/'mixture.npy'
        if not mix_path.is_file(): raise FileNotFoundError(f'wet cache input missing: {mix_path}')
        x=np.load(mix_path,allow_pickle=False).astype(np.float32)
        x=_smoke_input(x,args.seconds,config['sample_rate'])
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; smoke failed without CPU fallback')
        torch.cuda.reset_peak_memory_stats(); start=time.perf_counter(); separator=HTDemucsCUDA(segment=config['separator'].get('segment_seconds',7.0),overlap=config['separator']['overlap'])
        load_seconds=time.perf_counter()-start
        torch.cuda.synchronize(); cold_start=time.perf_counter(); estimates=separator(x); torch.cuda.synchronize(); cold_seconds=time.perf_counter()-cold_start
        warm_start=time.perf_counter(); warm=separator(x); torch.cuda.synchronize(); warm_seconds=time.perf_counter()-warm_start
        donor_start=time.perf_counter(); candidate=donor_reseparation(x,estimates,separator,donors=('other',),strength=.5); torch.cuda.synchronize(); donor_seconds=time.perf_counter()-donor_start
        if not np.isfinite(estimates).all() or not np.isfinite(candidate).all(): raise ValueError('smoke produced nonfinite outputs')
        state=separator.model.state_dict(); model_hash=hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for _,v in sorted(state.items()))).hexdigest()
        torch.cuda.synchronize(); result={'track':track,'split':args.split,'condition':'R4','seconds':args.seconds,'device':str(torch.cuda.current_device()),'device_name':torch.cuda.get_device_name(),'input_device':'cuda:0','output_device':'cuda:0','dtype':'float32','shape':list(estimates.shape),'load_seconds':load_seconds,'cold_forward_seconds':cold_seconds,'warm_forward_seconds':warm_seconds,'donor_forward_seconds':donor_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'output_sha256':hashlib.sha256(candidate.tobytes()).hexdigest(),'model_state_sha256':model_hash,'dataset_manifest_sha256':sha256_file(manifest_path),'config_sha256':sha256_file(args.config),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'code_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()}
        atomic_json(artifact/'reviews/smoke-attestation.json',result)
    elif args.command=='develop':
        manifest_path=runtime/'manifests/dataset.json'
        if not manifest_path.is_file(): raise FileNotFoundError('prepare the runtime dataset manifest first')
        result=_run_development(runtime,artifact,json.loads(manifest_path.read_text(encoding='utf-8')))
    elif args.command=='seal':
        selection=config.get('selection',{})
        if selection.get('sar_policy_status')!='confirmed': raise ValueError('selection seal blocked until owner confirms SAR policy')
        manifest_path=runtime/'manifests/dataset.json'; rir_path=runtime/'manifests/rirs.json'; models_path=runtime/'manifests/models.json'
        for required in (manifest_path,rir_path,models_path):
            if not required.is_file(): raise FileNotFoundError(f'required seal input missing: {required}')
        model_records=json.loads(models_path.read_text(encoding='utf-8'))
        model_hashes=[m['sha256'] for m in model_records.get('models',[]) if m.get('clean_winner_eligible')]
        if not model_hashes: raise ValueError('no training-data-verified clean-winner checkpoint in model manifest')
        code_files=sorted(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('config.json'),Path(__file__).with_name('requirements.lock')]
        code_hash=hashlib.sha256(b''.join(p.name.encode()+bytes.fromhex(sha256_file(p)) for p in code_files)).hexdigest()
        seal=seal_selection(selection,runtime_dir=runtime,manifest_hash=sha256_file(manifest_path),rir_manifest_hash=sha256_file(rir_path),
            model_hashes=model_hashes,code_hash=code_hash,lock_hash=sha256_file(Path(__file__).with_name('requirements.lock')),
            config_hash=sha256_file(args.config))
        atomic_json(artifact/'selection-seal.json',seal); result={'seal':str(artifact/'selection-seal.json'),'sha256':seal['sha256']}
    else:
        seal=json.loads(Path(args.seal).read_text(encoding='utf-8')); validate_seal(seal,split='evaluation')
        if config.get('selection',{}).get('sar_policy_status')!='confirmed': raise ValueError('evaluation blocked until owner confirms SAR policy')
        manifest_path=runtime/'manifests/dataset.json'; manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        tracks=[t for t in manifest['tracks'] if t['split']=='evaluation']
        if len(tracks)!=40: raise ValueError(f'evaluation requires exactly 40 tracks, found {len(tracks)}')
        selected=seal.get('selection',{}).get('selected')
        rir_path=runtime/'manifests/rirs.json'; models_path=runtime/'manifests/models.json'
        if seal.get('manifest_hash')!=sha256_file(manifest_path): raise ValueError('sealed dataset manifest changed')
        if not rir_path.is_file() or seal.get('rir_manifest_hash')!=sha256_file(rir_path): raise ValueError('sealed RIR manifest changed or missing')
        if seal.get('config_hash')!=sha256_file(args.config): raise ValueError('sealed configuration changed')
        model_records=json.loads(models_path.read_text(encoding='utf-8'))
        current_hashes=sorted(m['sha256'] for m in model_records.get('models',[]) if m.get('clean_winner_eligible'))
        if current_hashes!=seal.get('model_hashes'): raise ValueError('sealed eligible model set changed')
        candidate_map={'identity':lambda x,e:e.copy(),'T1-uniform':lambda x,e:t1(x,e),
                       'T2-alpha1':lambda x,e:t2(x,e,1),'T2-alpha2':lambda x,e:t2(x,e,2),
                       'T5-alpha1-lambda025':lambda x,e:t5(x,e,1,.25),
                       'C2-iter1':lambda x,e:consistency_iteration(x,e,1),'C2-iter2':lambda x,e:consistency_iteration(x,e,2),
                       'C1-norbert-1':lambda x,e:__import__('experiments.banelp.candidates',fromlist=['norbert_wiener']).norbert_wiener(e,x,1)}
        separator=None
        if selected in ('C-donor-other-lambda05','C-donor-drums-lambda05'):
            separator=HTDemucsCUDA()
            candidate_map[selected]=lambda x,e:donor_reseparation(x,e,separator,donors=('other' if selected.endswith('other-lambda05') else 'drums',),strength=.5)
        if selected not in candidate_map: raise ValueError(f'unsupported sealed candidate: {selected}')
        inputs=[]
        for track in tracks:
            for room in ('R1','R2','R3','R4'):
                wet=runtime/'cache/wet'/'evaluation'/track['name']/room
                base=runtime/'cache/separation/htdemucs/evaluation'/track['name']/room/'estimates.npy'
                for path in (wet/'mixture.npy',wet/'targets.npy',base):
                    if not path.is_file(): raise FileNotFoundError(f'evaluation input/cache missing: {path}')
                inputs.append((track['name'],room,wet/'mixture.npy',wet/'targets.npy',base))
        claim=runtime/'claims/evaluation-40.claim'; claim.parent.mkdir(parents=True,exist_ok=True)
        try: fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as exc: raise RuntimeError('evaluation seal already claimed; refusing duplicate holdout run') from exc
        with os.fdopen(fd,'w') as stream: stream.write(seal['sha256']+'\n')
        records=[]
        for track,room,xp,sp,ep in inputs:
            x=np.load(xp,allow_pickle=False).astype(np.float32); targets=np.load(sp,allow_pickle=False).astype(np.float32); estimates=np.load(ep,allow_pickle=False).astype(np.float32)
            validate_tensors(x,targets,samples=contracts.SAMPLES); validate_tensors(x,estimates,samples=contracts.SAMPLES)
            out=candidate_map[selected](x,estimates)
            before={n:project_metrics(targets[i],estimates[i],targets) for i,n in enumerate(contracts.SOURCES)}
            after={n:project_metrics(targets[i],out[i],targets) for i,n in enumerate(contracts.SOURCES)}
            records.append({'track':track,'room':room,'baseline':before,'selected':after,'mixture_residual':mixture_residual(x,out)})
        result={'split':'evaluation','selected':selected,'seal_sha256':seal['sha256'],'records':records,'tracks':len(tracks),'claim':str(claim)}
        atomic_json(runtime/'results/evaluation.json',result); atomic_json(artifact/'evidence/evaluation.json',result)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)); return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr); raise SystemExit(2)

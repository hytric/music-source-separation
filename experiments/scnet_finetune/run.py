"""Fine-tune the actual official SCNet with paired dry/wet supervised updates."""
from __future__ import annotations
import argparse,csv,gc,hashlib,importlib.util,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from core import PairedCrops,normalized_pair,reconstruction_loss,save_checkpoint,tensor_hash,SOURCES


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def wave_hash(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f4').tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('runtime','prepared','baseline-runtime','baseline-repo','official','previous-run','frontend-source','adapter-source','suite'):p.add_argument('--'+name,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    sys.path[:0]=[args.suite,args.baseline_repo,str(Path(args.adapter_source).parent)]
    from adapter import OfficialSCNet
    from banelp_refine.common import atomic_json,read_json,sha256,digest,verify_entry,file_entry,environment,exclusive,emit
    from cause_eq.metrics import score_sets,equivalence_check
    from report import make_report
    root=Path(args.runtime);root.mkdir(parents=True,exist_ok=True);prep=Path(args.prepared);previous=Path(args.previous_run);assets=Path(args.official);base=Path(args.baseline_runtime)
    config=read_json(Path(__file__).with_name('protocol.json'));train=config['training'];eval_config=config['evaluation']
    torch.set_num_threads(train['cpu_threads']);torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    def guard():
        temp=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
        if temp>=83:raise RuntimeError('GPU temperature limit reached')
    guard()
    old=read_json(previous/'binding.json');frontend=load_module(args.frontend_source,'finetune_frontend')
    for f in old['official']['files']:
        if sha256(assets/f['path'])!=f['sha256']:raise ValueError('official asset changed')
    for source,h in old['official_source'].items():
        if sha256(assets/source)!=h:raise ValueError('official source changed')
    if sha256(args.adapter_source)!=old['code']['adapter.py']:raise ValueError('official adapter changed')
    dataset=read_json(prep/'manifests/dataset.json');cache=read_json(prep/'manifests/cache.json');rirs=read_json(prep/'manifests/rirs.json')
    tracks=dataset['tracks'];train_names={t['name'] for t in tracks if t['split']=='train'}
    val=sorted([t for t in tracks if t['split']=='validation'],key=lambda t:t['name'])
    if len(train_names)!=80 or len(val)!=10:raise ValueError('song split changed')
    rooms={};rir_split={}
    for row in rirs['sets']:
        if rooms.setdefault(row['room_id'],row['split'])!=row['split']:raise ValueError('RIR room leakage')
        rir_split[row['id']]=row['split']
    train_rows=[r for r in cache['records'] if r['split']=='train']
    for row in train_rows:
        if row['track'] not in train_names or (row['rir_set'] is not None and rir_split[row['rir_set']]!='train'):raise ValueError('training leakage')
    baseline_files=[previous/'binding.json']+sorted(previous.glob('*/*/input.json'))+sorted(previous.glob('*/*/results.json'))
    binding={'protocol':config,'environment':environment(),'code':{f.name:sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'adapter':sha256(args.adapter_source),'frontend_source':sha256(args.frontend_source),
        'shared_code':{name:sha256(Path(args.suite)/name) for name in ('banelp_refine/common.py','cause_eq/metrics.py')},
        'manifests':{name:sha256(prep/'manifests'/name) for name in ('dataset.json','cache.json','rirs.json')},
        'baseline_rirs':sha256(base/'manifests/rirs.json'),'baseline_files':{str(f.relative_to(previous)):sha256(f) for f in baseline_files},
        'official':old['official'],'official_source':old['official_source'],'seed':config['seed'],'device':torch.cuda.get_device_name(0)}
    binding_id=digest(binding)
    if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ValueError('training binding changed')
    atomic_json(root/'binding.json',binding)
    def assert_code():
        for name,h in binding['code'].items():
            if sha256(Path(__file__).with_name(name))!=h:raise ValueError('source changed during execution')
    builder=frontend.Experiment.__new__(frontend.Experiment);builder.sources=SOURCES;builder.prep=prep;builder.verify=verify_entry;builder.rirs={}
    rm=read_json(base/'manifests/rirs.json')
    for s,pos in zip(SOURCES,('P1','P2','P3','P4')):
        e=rm['rooms']['R4']['positions'][pos];f=base/e['path']
        if sha256(f)!=e['sha256']:raise ValueError('evaluation RIR changed')
        builder.rirs[s]=np.load(f,allow_pickle=False)
    def cases():
        for t in val:
            dry=builder.dry_sources(t)
            for c in eval_config['conditions']:
                y=builder.targets(dry,c);x=y.astype(np.float64).sum(0).astype(np.float32)
                old_input=read_json(previous/t['name']/c/'input.json')
                if wave_hash(x)!=old_input['input_sha256'] or wave_hash(y)!=old_input['target_sha256']:raise ValueError('evaluation signal differs')
                yield t,c,x,y,old_input
    def save_state(step,opt,sep):
        assert_code();state=tensor_hash(sep.model)
        payload={'step':step,'binding':binding_id,'model':sep.model.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'state_sha256':state}
        save_checkpoint(root/'last.pt',payload)
        if step in train['evaluate_steps'] or step==0:save_checkpoint(root/f'step-{step:04d}.pt',payload)
        atomic_json(root/'status.json',{'status':'running','step':step,'max_steps':train['max_steps'],'binding':binding_id,'state_sha256':state})
    def evaluate(step,sep):
        sep.model.eval();checkpoint=root/f'step-{step:04d}.pt';checkpoint_sha=sha256(checkpoint);rows=[]
        folder=root/'evaluation'/f'step-{step:04d}';folder.mkdir(parents=True,exist_ok=True)
        for t,c,x,y,info in cases():
            f=folder/t['name']/c;f.mkdir(parents=True,exist_ok=True);saved=f/'results.json'
            if saved.exists():
                row=read_json(saved)
                if row['binding']!=binding_id or row['checkpoint_sha256']!=checkpoint_sha:raise ValueError('stale evaluation')
                verify_entry(root,row['prediction']);rows.append(row);continue
            guard();started=time.monotonic();out=sep(x);path=f/'prediction.npy'
            with path.with_suffix('.part').open('wb') as stream:np.save(stream,out,allow_pickle=False)
            path.with_suffix('.part').replace(path)
            score=score_sets(y[...,frontend.SL],{'candidate':out[...,frontend.SL]})['candidate']
            baseline=read_json(previous/t['name']/c/'results.json')['scores']
            check=equivalence_check(y[...,frontend.SL],out[...,frontend.SL]) if not rows else None
            row={'track':t['name'],'condition':c,'binding':binding_id,'step':step,'checkpoint_sha256':checkpoint_sha,'input_sha256':info['input_sha256'],'target_sha256':info['target_sha256'],'prediction':file_entry(path,root),'scores':{'candidate':score,'initial_rev':baseline['rev'],'clean':baseline['clean'],'htdemucs':baseline['htdemucs'],'ensemble':baseline['ensemble']},'seconds':time.monotonic()-started,'legacy_metric_max_difference':check}
            atomic_json(saved,row);rows.append(row);emit('finetune_case_scored',step=step,track=t['name'],condition=c)
        from report import summarize
        summary=summarize(rows,config,frontend.paired_stat);summary.update({'step':step,'binding':binding_id,'checkpoint_sha256':checkpoint_sha,'cases':len(rows)})
        atomic_json(folder/'summary.json',summary);emit('finetune_validation',step=step,wet_delta=summary['comparisons']['wet_R4']['candidate-minus-initial_rev']['overall'],passes=summary['passes'])
        sep.model.train();return summary
    with exclusive(root,'train'):
        started=time.monotonic();atomic_json(root/'status.json',{'status':'verifying_training_targets','binding':binding_id})
        for i,row in enumerate(train_rows):
            verify_entry(prep,row['files']['targets'])
            if (i+1)%80==0:emit('training_target_verified',done=i+1,total=len(train_rows))
        samples=PairedCrops(prep,cache['records'],tracks,verify_entry);samples.verified={r['files']['targets']['path'] for r in train_rows}
        sep=OfficialSCNet(assets,'rev',segment=20);model=sep.model
        if sum(p.numel() for p in model.parameters())!=10578768:raise ValueError('wrong architecture')
        opt=torch.optim.AdamW(model.parameters(),lr=train['learning_rate'],weight_decay=train['weight_decay']);begin=0
        if args.resume:
            ckpt=torch.load(root/'last.pt',map_location='cpu',weights_only=True)
            if ckpt['binding']!=binding_id:raise ValueError('resume binding mismatch')
            model.load_state_dict(ckpt['model'],strict=True)
            if tensor_hash(model)!=ckpt['state_sha256']:raise ValueError('checkpoint state changed')
            opt.load_state_dict(ckpt['optimizer']);torch.set_rng_state(ckpt['torch_rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng']);begin=ckpt['step']
            for group in opt.state.values():
                for k,v in group.items():
                    if torch.is_tensor(v):group[k]=v.to('cuda') if k!='step' else v
        else:
            if (root/'last.pt').exists():raise ValueError('existing training requires --resume')
            initial=tensor_hash(model)
            t,c,x,y,info=next(cases());out=sep(x)
            old_meta=read_json(previous/t['name']/c/'rev.json');old_out=np.load(verify_entry(previous,old_meta['file']),allow_pickle=False)
            error=float(np.max(np.abs(out-old_out)))
            if error>3e-6:raise ValueError(f'initial output differs from official rev: {error}')
            atomic_json(root/'initial-check.json',{'state_sha256':initial,'max_abs_output_difference':error,'shape':list(out.shape),'parameter_count':10578768,'production_optimizer_updates':0})
            del out,old_out,x,y;save_state(0,opt,sep)
            emit('training_started',max_steps=train['max_steps'],initial_output_error=error,binding=binding_id)
        model.train();last_info=None;validations=[]
        if begin in train['evaluate_steps'] and not (root/'evaluation'/f'step-{begin:04d}'/'summary.json').exists():evaluate(begin,sep)
        try:
            with (root/'training.jsonl').open('a',encoding='utf-8') as log:
                for step in range(begin+1,train['max_steps']+1):
                    crops,info=samples.sample(step,config['seed']);opt.zero_grad(set_to_none=True);losses=[]
                    for crop in crops:
                        target=crop.to('cuda');x,target=normalized_pair(target);out=model(x)[:,sep.order]
                        loss,terms=reconstruction_loss(out,target,model.stft_config,vocal_weight=config['loss']['stem_weights'][0],wave_weight=config['loss']['wave_l1_weight'])
                        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                        (loss/2).backward();losses.append(float(loss.detach()))
                        del out,target,x,loss
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),train['clip_norm'],error_if_nonfinite=True);opt.step()
                    value={'step':step,'dry_loss':losses[0],'wet_loss':losses[1],'gradient_norm':float(norm),'elapsed':time.monotonic()-started,**info};log.write(json.dumps(value,ensure_ascii=False)+'\n');log.flush()
                    if step%25==0:emit('finetune_progress',**value)
                    if step%train['save_every']==0:guard();save_state(step,opt,sep)
                    if step in train['evaluate_steps']:validations.append(evaluate(step,sep))
            for step in train['evaluate_steps']:
                f=root/'evaluation'/f'step-{step:04d}'/'summary.json'
                if not f.exists():raise ValueError('required validation missing')
            summaries=[read_json(root/'evaluation'/f'step-{step:04d}'/'summary.json') for step in train['evaluate_steps']]
            chosen=max(summaries,key=lambda v:(v['passes'],v['comparisons']['wet_R4']['candidate-minus-initial_rev']['overall']['median']))
            save={'status':'complete','step':train['max_steps'],'selected_step':chosen['step'],'accepted':chosen['passes'],'binding':binding_id,'seconds':time.monotonic()-started,'final_state_sha256':tensor_hash(model),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
            assert_code();atomic_json(root/'status.json',{**save,'status':'reporting'});make_report(root,Path(__file__).parent,config);atomic_json(root/'status.json',save)
            emit('finetune_complete',**save)
        except Exception as e:
            atomic_json(root/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'binding':binding_id});raise

if __name__=='__main__':main()

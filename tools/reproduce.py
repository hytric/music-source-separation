"""Portable evidence replay and hash-verified assets for separation research.

Raw assets are exported privately; no venv, credentials or runtime databases are copied.
"""
from pathlib import Path
import argparse,copy,csv,hashlib,importlib.util,io,json,os,shutil,subprocess,sys,time


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.part');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def safe(root,relative):
 p=(root/relative).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError('unsafe bundle path')
 return p


def catalog(runtime,profile):
 files={}
 def add(p,expected=None):
  p=Path(p).resolve()
  if not p.is_file() or not p.is_relative_to(runtime):raise ValueError(f'missing/outside runtime asset: {p}')
  rel='roots/'+str(p.relative_to(runtime));files[rel]={'source':str(p),'relative':rel,'bytes':p.stat().st_size,'expected_sha256':expected}
 def entry(root,e):add(root/e['path'],e.get('sha256'))
 for name in ('banelp-v1','banelp-refine-v1'):
  root=runtime/name
  for file in ('dataset.json','rirs.json'):
   add(root/'manifests'/file)
  d=read(root/'manifests/dataset.json')
  for t in d['tracks']:
   if profile=='validation' and (name=='banelp-v1' or t['split']!='validation'):continue
   for e in t.get('files',t.get('sources',{})).values():entry(root,e)
  rm=read(root/'manifests/rirs.json')
  if name=='banelp-v1':
   for room in rm['rooms'].values():
    for e in room['positions'].values():entry(root,e)
  elif profile=='all':
   for r in rm['sets']:
    for e in r['sources'].values():entry(root,e)
 base=runtime/'banelp-v1';add(base/'manifests/models.json')
 for m in read(base/'manifests/models.json')['models']:
  for e in m['members']:add(e['path'],e['sha256'])
  if m.get('config_file'):add(m['config_file']['path'],m['config_file']['sha256'])
 msst=base/'src/MSST'
 if subprocess.check_output(['git','-C',str(msst),'rev-parse','HEAD'],text=True).strip()!='f17924ddb1817cdbae6353706708c4518dd7cd14':raise ValueError('MSST commit changed')
 if subprocess.check_output(['git','-C',str(msst),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('MSST modified')
 for name in subprocess.check_output(['git','-C',str(msst),'ls-files','-z']).decode().split('\0'):
  if name:add(msst/name)
 # A minimal local .git enables the upstream loader without copying config/hooks/logs.
 for name in ('HEAD','shallow','packed-refs'):
  p=msst/'.git'/name
  if p.is_file():add(p)
 for directory in ('refs','objects'):
  for p in (msst/'.git'/directory).rglob('*'):
   if p.is_file():add(p)
 official=runtime/'parirset-official-v1'
 for name in ('source','conf','models'):
  for p in (official/name).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:add(p)
 for name in ('LICENSE','acquisition.json'):add(official/name)
 for name in ('step-0000.pt','step-1000.pt','binding.json','status.json'):add(runtime/'scnet-finetune-v1'/name)
 for name in ('model.json','binding.json','summary.json','rows.json'):add(runtime/'input-selector-v1'/name)
 for p in (runtime/'parirset-compare-v1').glob('*/*/input.json'):add(p)
 if profile=='all':
  for p in (runtime/'banelp-refine-v1/runs/seed-20261111-base').glob('*.pt'):add(p)
  for p in (runtime/'banelp-refine-v1/initial').glob('*.pt'):add(p)
  for name in ('cache.json',):add(runtime/'banelp-refine-v1/manifests'/name)
  qg=runtime/'quality-gate-v1-full'
  entries=list((qg/'inputs').glob('*/complete.json'))
  if len(entries)!=320:raise ValueError('full profile requires320 frozen train40 input metadata')
  for meta in entries:
   add(meta);e=read(meta)['mixture'];add(qg/e['path'],e['sha256'])
  add(qg/'inputs.json')
 return list(files.values())


def export(args):
 runtime=Path(args.runtime_root).resolve();out=Path(args.bundle).resolve();files=catalog(runtime,args.profile)
 total=sum(x['bytes'] for x in files);print(json.dumps({'files':len(files),'bytes':total,'GB':total/1e9,'profile':args.profile}),flush=True)
 if args.plan:return
 if out.exists() and any(out.iterdir()):raise ValueError('destination must be empty; never overwrite a bundle')
 out.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(out).free<total*1.05:raise ValueError('insufficient bundle disk space')
 for i,e in enumerate(files):
  dst=safe(out,e['relative']);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(e['source'],dst);shutil.copymode(e['source'],dst)
  h=sha(dst)
  if e['expected_sha256'] and h!=e['expected_sha256']:raise ValueError('source content changed: '+e['source'])
  e['sha256']=h
  if (i+1)%100==0:print(json.dumps({'copied':i+1,'of':len(files)}),flush=True)
 gitconfig=out/'roots/banelp-v1/src/MSST/.git/config'
 gitconfig.write_text('[core]\nrepositoryformatversion = 0\nbare = false\nfilemode = false\nautocrlf = false\n')
 subprocess.run(['git','-C',str(gitconfig.parent.parent),'read-tree','HEAD'],check=True)
 files.append({'source':None,'relative':str(gitconfig.relative_to(out)),'bytes':gitconfig.stat().st_size,'expected_sha256':None,'sha256':sha(gitconfig),'note':'new credential-free Git metadata; original config/hooks/logs excluded'})
 code={str(p.relative_to(REPO)):sha(p) for p in (REPO/'experiments').rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml') and '__pycache__' not in p.parts}
 write(out/'bundle.json',{'version':1,'profile':args.profile,'file_count':len(files),'bytes':sum(e['bytes'] for e in files),'files':files,'code_sha256':code,'runtime_original':str(runtime),'scope':'raw data, RIR and pinned models; inference caches not included; legacy bindings remain historical evidence','privacy':'private transfer only; not a public dataset/model redistribution grant'})
 print('EXPORT_COMPLETE',out,flush=True)


def verify(args):
 os.environ['GIT_OPTIONAL_LOCKS']='0'
 root=Path(args.bundle).resolve();manifest=read(root/'bundle.json')
 expected_path=REPO/'docs/experiments/10_reproduction/expected_assets.json'
 expected=read(expected_path) if expected_path.exists() else {}
 if manifest['profile']=='all' and set(expected)-{e['relative'] for e in manifest['files']}:raise ValueError('full bundle is missing expected assets; canonicalize after export')
 for i,e in enumerate(manifest['files']):
  if e['relative'] in expected and (e['sha256']!=expected[e['relative']]['sha256'] or e['bytes']!=expected[e['relative']]['bytes']):raise ValueError('bundle differs from published asset fingerprint: '+e['relative'])
  p=safe(root,e['relative'])
  if not p.is_file() or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError('bundle mismatch: '+e['relative'])
  if (i+1)%200==0:print('verified',i+1,flush=True)
 for name,h in manifest['code_sha256'].items():
  if sha(REPO/name)!=h:raise ValueError('experiment code differs: '+name)
 print('BUNDLE_VERIFIED',manifest['file_count'])


def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def context(args):
 import numpy as np
 os.environ['GIT_OPTIONAL_LOCKS']='0'
 root=Path(args.bundle).resolve();manifest=read(root/'bundle.json');mapping={e['source']:str(safe(root,e['relative'])) for e in manifest['files'] if e['source'] is not None}
 def relocate(value):
  if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
  if isinstance(value,list):return [relocate(v) for v in value]
  if isinstance(value,str) and value in mapping:return mapping[value]
  return value
 def check(p):
  rel=str(p.relative_to(root));e=next((e for e in manifest['files'] if e['relative']==rel),None)
  if e is None or sha(p)!=e['sha256']:raise ValueError('missing or changed bundle dependency: '+rel)
  return p
 os.environ['GIT_OPTIONAL_LOCKS']='0'
 msst=root/'roots/banelp-v1/src/MSST'
 if not (msst/'.git/index').exists():subprocess.run(['git','-C',str(msst),'read-tree','HEAD'],check=True)
 sys.path[:0]=[str(REPO/'experiments/banelp_refine'),str(REPO)]
 from banelp_refine.common import verify_entry
 front=module(REPO/'experiments/frontend_validation/run.py','portable_frontend')
 prep=root/'roots/banelp-refine-v1';base=root/'roots/banelp-v1'
 builder=front.Experiment.__new__(front.Experiment);builder.prep=prep;builder.sources=('vocals','drums','bass','other');builder.verify=verify_entry;builder.rirs={}
 rm=read(check(base/'manifests/rirs.json'))
 for stem,pos in zip(builder.sources,('P1','P2','P3','P4')):
  e=rm['rooms']['R4']['positions'][pos];p=Path(relocate(e['path'])) if Path(e['path']).is_absolute() else base/e['path'];builder.rirs[stem]=np.load(check(p),allow_pickle=False)
 all_tracks=read(check(prep/'manifests/dataset.json'))['tracks'];train_row=None
 if getattr(args,'record_id',None):
  train_row=next(r for r in read(check(prep/'manifests/cache.json'))['records'] if r['id']==args.record_id)
  if train_row['split']!='train':raise ValueError('teacher record must be in train80')
  args.split='train'
  args.condition='dry' if train_row['rir_set'] is None else train_row['condition']
  if train_row['rir_set'] is not None:
   rs=next(r for r in read(check(prep/'manifests/rirs.json'))['sets'] if r['id']==train_row['rir_set'])
   if rs['split']!='train':raise ValueError('RIR leakage')
   builder.rirs={stem:np.load(check(prep/e['path']),allow_pickle=False) for stem,e in rs['sources'].items()}
 tracks=[t for t in all_tracks if t['name']==train_row['track']] if train_row else [t for t in all_tracks if t['split']==args.split]
 tracks=sorted(tracks,key=lambda t:t['name'])
 if not tracks:raise ValueError('no tracks for split')
 t=tracks[0 if train_row else args.track_index];dry=builder.dry_sources(t);y=builder.targets(dry,args.condition);x=y.astype(np.float64).sum(0).astype(np.float32)
 canonical=root/'canonical'/t['name']/args.condition
 regeneration_error=0.
 if train_row:
  source_record=root/'roots/quality-gate-v1-full/inputs'/train_row['id']/'complete.json'
  original=read(check(source_record));cx=np.load(check(root/'roots/quality-gate-v1-full'/original['mixture']['path']),allow_pickle=False)
  regeneration_error=float(np.max(abs(cx-x)))
  if regeneration_error>1e-6 or hashlib.sha256(y.tobytes()).hexdigest()!=original['target40_sha256']:raise ValueError('training input or target reconstruction mismatch')
  x=cx
 if (canonical/'mixture.npy').exists():
  cx=np.load(check(canonical/'mixture.npy'),allow_pickle=False);cy=np.load(check(canonical/'targets.npy'),allow_pickle=False)
  regeneration_error=max(float(np.max(abs(cx-x))),float(np.max(abs(cy-y))))
  if regeneration_error>1e-6:raise ValueError('canonical synthesis mismatch')
  x,y=cx,cy
 info={'regeneration_max_abs':regeneration_error,'canonical_input_used':bool(train_row or (canonical/'mixture.npy').exists()),'track':t['name'],'split':args.split,'condition':args.condition,'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'target_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'shape':list(x.shape),'source_order':list(builder.sources)}
 expected=base.parent/'parirset-compare-v1'/t['name']/args.condition/'input.json'
 if args.split=='validation':
  e=read(check(expected));assert e['input_sha256']==info['input_sha256'] and e['target_sha256']==info['target_sha256'],'regenerated input/target differs'
 return root,relocate,check,front,x,y,info


def inference(args):
 import numpy as np
 root,relocate,check,front,x,y,info=context(args)
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise ValueError('output result already exists')
 write(out/'input-proof.json',info)
 if args.command=='input-check':print(json.dumps(info,ensure_ascii=False));return
 import torch
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU/FP16 fallback')
 free,total=torch.cuda.mem_get_info()
 required=(1.863 if args.model=='htdemucs' else 2.283 if args.model in ('clean','rev','finetuned') else 4)*1024**3
 if not hasattr(args,'_separator') and free<required:raise RuntimeError('insufficient free GPU memory for portable smoke')
 if hasattr(args,'_separator'):sep=args._separator
 else:
  if args.model in ('htdemucs','bsr'):
   from experiments.banelp.separators import HTDemucsCUDA,MSSTSeparator
   registry=relocate(read(check(root/'roots/banelp-v1/manifests/models.json')))['models'];id='htdemucs' if args.model=='htdemucs' else 'msst_bsroformer_ep17';r=next(r for r in registry if r['model_id']==id)
   for entry in r['members']+([r['config_file']] if r.get('config_file') else []):
    if not Path(entry['path']).resolve().is_relative_to(root):raise ValueError('model path did not relocate into bundle')
   sep=HTDemucsCUDA(model_name='htdemucs',segment=7.,overlap=.25,registered=r) if args.model=='htdemucs' else MSSTSeparator(registered=r,runtime_dir=root/'roots/banelp-v1',device='cuda',batch_size=1)
  else:
   adapter=module(REPO/'experiments/parirset_validation/adapter.py','portable_adapter');sep=adapter.OfficialSCNet(root/'roots/parirset-official-v1','clean' if args.model=='clean' else 'rev',segment=20)
   if args.model=='finetuned':
    checkpoint=torch.load(check(root/'roots/scnet-finetune-v1/step-1000.pt'),map_location='cpu',weights_only=True)
    assert checkpoint['step']==1000;sep.model.load_state_dict(checkpoint['model'],strict=True);sep.model.eval()
 started=time.monotonic();torch.cuda.reset_peak_memory_stats();z=sep(x);np.save(out/'prediction.npy',z,allow_pickle=False)
 from cause_eq.metrics import score_sets
 scores=score_sets(y[...,front.SL],{'prediction':z[...,front.SL]})['prediction']
 write(out/'result.json',{'input':info,'model':args.model,'scores':scores,'prediction_sha256':sha(out/'prediction.npy'),'seconds':time.monotonic()-started,'peak_reserved':torch.cuda.max_memory_reserved(),'scope':'new relocated inference, not an old binding reuse'})
 print('INFERENCE_COMPLETE',args.model,info['track'],flush=True)
 return sep


def results(args):
 import numpy as np
 def stat(values,seed):
  a=np.asarray([v for v in values if v is not None],float)
  if not len(a):return None
  ids=np.random.default_rng(seed).integers(len(a),size=(10000,len(a)))
  return [float(np.median(a)),*np.percentile(np.median(a[ids],1),[2.5,97.5]).tolist()]
 plan=[('frontend_validation','output','raw',20261113),('parirset_validation','model','htdemucs',20261113),('scnet_finetune','model','ensemble',20261113),('stem_hybrid_validation','method','baseline',20261114),('db_consistency_validation','method','baseline',20261114)]
 output={};tables={}
 for folder,column,reference,seed in plan:
  path=REPO/'experiments'/folder/'결과/곡별결과.csv'
  with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
  table={(r.get('step',''),r['track'],r['condition'],r[column],r['stem']):float(r['si_sdr']) if r['si_sdr'] else None for r in rows}
  tables[folder]=table
  for step in sorted({k[0] for k in table}):
   for condition in ('dry','wet_R4'):
    tracks=sorted({k[1] for k in table if k[0]==step and k[2]==condition})
    for method in sorted({k[3] for k in table}):
     values=[]
     for track in tracks:
      d=[]
      for stem in ('vocals','drums','bass','other'):
       a=table[(step,track,condition,method,stem)];b=table[(step,track,condition,reference,stem)]
       if a is not None and b is not None:d.append(a-b)
      values.append(float(np.mean(d)) if d else None)
     output['/'.join((folder,step,condition,method,reference))]=stat(values,seed)
 # The input selector has a decision CSV rather than a new stem CSV.
 with (REPO/'experiments/input_selector_validation/결과/선택결과.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 t=tables['stem_hybrid_validation']
 for c in ('dry','wet_R4'):
  values=[]
  for r in sorted((r for r in rows if r['condition']==c),key=lambda r:r['track']):
   assert r['choice']==('hybrid' if float(r['sigmoid_score'])>=.8 else 'baseline')
   d=[]
   for stem in ('vocals','drums','bass','other'):
    a=t['',r['track'],c,r['choice'],stem];b=t['',r['track'],c,'baseline',stem]
    if a is not None and b is not None:d.append(a-b)
   values.append(float(np.mean(d)))
  output['input_selector/'+c]=stat(values,20261114)
 expected=read(REPO/'docs/experiments/10_reproduction/expected_results.json')
 for key,value in expected.items():
  if max(abs(a-b) for a,b in zip(value,output[key]))>1e-9:raise ValueError('result mismatch: '+key)
 write(args.output,{'recomputed':output,'key_results_checked':len(expected),'scope':'saved-evidence arithmetic replay, not new model evaluation'})
 print('RESULTS_VERIFIED',len(output),'groups;',len(expected),'golden comparisons')


def compose(args):
 import numpy as np
 root,_,check,front,x,y,info=context(args);predictions={}
 for name in ('htdemucs','bsr','finetuned'):
  folder=Path(args.predictions)/name;record=read(folder/'result.json')
  if record['input']['input_sha256']!=info['input_sha256'] or record['model']!=name or sha(folder/'prediction.npy')!=record['prediction_sha256']:raise ValueError('case predictions do not share the requested input')
  predictions[name]=np.load(folder/'prediction.npy',allow_pickle=False)
 b=(predictions['htdemucs']+predictions['bsr'])*.5;f=predictions['finetuned'];h=b.copy();h[1:3]=f[1:3]
 selector=module(REPO/'experiments/input_selector_validation/core.py','portable_selector')
 model=read(check(root/'roots/input-selector-v1/model.json'));prob=float(selector.predict(selector.features(x,front.stft),model));r=h if prob>=model['threshold'] else b
 correction=module(REPO/'experiments/db_consistency_validation/core.py','portable_correction')
 methods={'baseline':b,'finetuned':f,'hybrid':h,'selector':r,'selector_uniform':correction.project(x,r,'uniform'),'selector_power':correction.project(x,r,'power',front.stft,front.istft)}
 from cause_eq.metrics import score_sets
 scores=score_sets(y[...,front.SL],{k:v[...,front.SL] for k,v in methods.items()})
 write(Path(args.output)/'comparison.json',{'input':info,'selector_score':prob,'choice':'hybrid' if prob>=model['threshold'] else 'baseline','scores':scores,'scope':'new relocated case, not independent generalization'})
 print('CASE_COMPARED',info['track'])



def teacher(args):
 if not 1<=args.limit<=320:raise ValueError('teacher limit must be1..320')
 root=Path(args.bundle).resolve();cache=read(root/'roots/banelp-refine-v1/manifests/cache.json');rows=sorted([r for r in cache['records'] if r['split']=='train'],key=lambda r:r['id'])
 base=Path(args.output).resolve()
 for r in rows[:args.limit]:
  args.record_id=r['id'];args.output=str(base/r['id']);args.split='train';args.track_index=0;args.condition='dry'
  if (Path(args.output)/'result.json').exists():raise ValueError('Existing teacher output requires explicit hash-aware resume; use fresh output')
  args._separator=inference(args)
 print('TEACHER_COMPLETE',args.model,min(args.limit,len(rows)))


def pack(args):
 import tarfile
 root=Path(args.bundle).resolve();m=read(root/'bundle.json');archive=Path(args.archive).resolve()
 if archive.exists():raise ValueError('archive already exists')
 archive.parent.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(archive.parent).free<m['bytes']*1.02:raise ValueError('insufficient archive disk')
 digest=hashlib.sha256()
 class Writer:
  def __init__(self,f):self.f=f
  def write(self,data):digest.update(data);return self.f.write(data)
  def tell(self):return self.f.tell()
 def metadata(info):info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;return info
 temp=archive.with_suffix('.part')
 with temp.open('wb') as raw:
  with tarfile.open(fileobj=Writer(raw),mode='w|',format=tarfile.PAX_FORMAT) as tar:
   tar.add(root/'bundle.json',arcname='mss-bundle/bundle.json',recursive=False,filter=metadata)
   for i,e in enumerate(m['files']):
    tar.add(safe(root,e['relative']),arcname='mss-bundle/'+e['relative'],recursive=False,filter=metadata)
    if (i+1)%300==0:print('packed',i+1,flush=True)
 temp.replace(archive);archive.with_suffix(archive.suffix+'.sha256').write_text(digest.hexdigest()+'  '+archive.name+'\n')
 print('ARCHIVE_COMPLETE',archive,digest.hexdigest(),flush=True)

def canonicalize(args):
 import numpy as np
 root=Path(args.bundle).resolve();m=read(root/'bundle.json');new=[]
 for i in range(10):
  for c in ('dry','wet_R4'):
   ns=argparse.Namespace(bundle=str(root),split='validation',track_index=i,condition=c)
   _,_,_,_,x,y,info=context(ns)
   for name,array in (('mixture',x),('targets',y)):
    p=root/'canonical'/info['track']/c/(name+'.npy');p.parent.mkdir(parents=True,exist_ok=True);np.save(p,array,allow_pickle=False)
    new.append({'source':None,'relative':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p),'expected_sha256':None,'note':'canonical val40 input/target matching original raw-byte SHA'})
   print('canonical',i+1,c,flush=True)
 m['files']=[e for e in m['files'] if not e['relative'].startswith('canonical/')]+new;m['file_count']=len(m['files']);m['bytes']=sum(e['bytes'] for e in m['files']);write(root/'bundle.json',m)


def hydrate(args):
 root=Path(args.bundle).resolve();m=read(root/'bundle.json');work=Path(args.workspace).resolve()
 if work.exists() and any(work.iterdir()):raise ValueError('workspace must be new/empty')
 work.mkdir(parents=True,exist_ok=True);runtimes=work/'runtimes';runtimes.mkdir()
 old=m['runtime_original']
 def rebase(v):
  if isinstance(v,dict):return {k:rebase(x) for k,x in v.items()}
  if isinstance(v,list):return [rebase(x) for x in v]
  if isinstance(v,str) and v.startswith(old+'/'):return str(runtimes/v[len(old)+1:])
  return v
 allowed={'banelp-v1','banelp-refine-v1','parirset-official-v1'};count=0
 for e in m['files']:
  parts=Path(e['relative']).parts
  if len(parts)<3 or parts[0]!='roots' or parts[1] not in allowed:continue
  if parts[2]=='manifests':continue
  if parts[1]=='banelp-refine-v1' and parts[2] not in ('data','rirs'):continue
  src=safe(root,e['relative']);dst=runtimes/Path(*parts[1:]);dst.parent.mkdir(parents=True,exist_ok=True)
  if args.copy:shutil.copy2(src,dst)
  else:
   try:os.link(src,dst)
   except OSError as error:raise RuntimeError('Hardlink requires same filesystem; use --copy with sufficient disk') from error
  count+=1
 for runtime in ('banelp-v1','banelp-refine-v1'):
  for name in ('dataset.json','rirs.json','models.json'):
   src=root/'roots'/runtime/'manifests'/name
   if src.exists():write(runtimes/runtime/'manifests'/name,rebase(read(src)))
 # Never import old cache/run completion states as a new scientific run.
 shutil.copytree(REPO/'experiments',work/'code/experiments',ignore=shutil.ignore_patterns('__pycache__'))
 aliases={'학습실험_2026-11-11':'banelp_refine','입력전처리검증_2026-11-13':'frontend_validation','공식SCNet검증_2026-11-13':'parirset_validation','SCNet미세조정_2026-11-13':'scnet_finetune','음원별결합검증_2026-11-14':'stem_hybrid_validation','입력선택검증_2026-11-14':'input_selector_validation','출력합보정검증_2026-11-14':'db_consistency_validation'}
 (work/'project').mkdir()
 for local,folder in aliases.items():
  alias=work/'project'/local
  try:alias.symlink_to(Path('../code/experiments')/folder,target_is_directory=True)
  except OSError:shutil.copytree(work/'code/experiments'/folder,alias)
 write(work/'workspace.json',{'bundle':str(root),'repo':str(REPO),'files_linked_or_copied':count,'copy_mode':args.copy,'old_manifests_rebased':True,'old_prediction_cache_imported':False,'note':'source bundle preserved; new bindings/results must be produced here'})
 subprocess.run(['git','-C',str(runtimes/'banelp-v1/src/MSST'),'read-tree','HEAD'],check=True)
 print('WORKSPACE_READY',work,'files',count)


def stage(args):
 work=Path(args.workspace).resolve();meta=read(work/'workspace.json');r=work/'runtimes';code=work/'code';suite=code/'experiments/banelp_refine';base=r/'banelp-v1';prep=r/'banelp-refine-v1';official=r/'parirset-official-v1';front=r/'frontend-verify-v1';compare=r/'parirset-compare-v1';fine=r/'scnet-finetune-v1';hybrid=r/'stem-hybrid-v1';selector=r/'input-selector-v1'
 common=[sys.executable,str(suite/'run.py'),'--runtime-dir',str(prep),'--baseline-repo',str(code),'--baseline-runtime',str(base)]
 calls=[]
 if args.name in ('cache-refiner','preflight-refiner','train-refiner'):
  command={'cache-refiner':'cache','preflight-refiner':'preflight','train-refiner':'train'}[args.name]
  calls=[common+[command]+(['--start-training'] if command=='train' else [])]
 elif args.name=='frontend':calls=[[sys.executable,str(code/'experiments/frontend_validation/run.py'),'--runtime',str(front),'--suite',str(suite),'--baseline-repo',str(code),'--baseline-runtime',str(base),'--prepared-runtime',str(prep),'--limit','10']]
 elif args.name=='official':
  fr=read(front/'latest.json')['run']
  a=[sys.executable,str(code/'experiments/parirset_validation/run.py'),'--suite',str(suite),'--baseline-repo',str(code),'--frontend-source',str(code/'experiments/frontend_validation/run.py'),'--runtime',str(compare),'--frontend-run',fr,'--official',str(official),'--baseline-runtime',str(base),'--prepared-runtime',str(prep)]
  calls=[a+['--stage',s] for s in ('prepare','clean','rev','bsr','score')]
 elif args.name=='finetune':calls=[[sys.executable,str(code/'experiments/scnet_finetune/run.py'),'--runtime',str(fine),'--prepared',str(prep),'--baseline-runtime',str(base),'--baseline-repo',str(code),'--official',str(official),'--previous-run',str(compare),'--frontend-source',str(code/'experiments/frontend_validation/run.py'),'--adapter-source',str(code/'experiments/parirset_validation/adapter.py'),'--suite',str(suite)]]
 elif args.name=='hybrid':calls=[[sys.executable,str(code/'experiments/stem_hybrid_validation/run.py'),'--runtime',str(hybrid),'--prepared',str(prep),'--baseline-runtime',str(base),'--fine-runtime',str(fine),'--official-run',str(compare),'--frontend-run',read(front/'latest.json')['run'],'--suite',str(suite),'--baseline-repo',str(code),'--frontend-source',str(code/'experiments/frontend_validation/run.py')]]
 elif args.name in ('selector','consistency'):
  folder='input_selector_validation' if args.name=='selector' else 'db_consistency_validation';output=selector if args.name=='selector' else r/'db-consistency-v1'
  calls=[[sys.executable,str(code/'experiments'/folder/'run.py'),'--project',str(work/'project'),'--assets',str(r),'--runtime',str(output),'--baseline-repo',str(code)]]
 elif args.name=='cause-eq':
  a=[sys.executable,str(suite/'run_cause.py'),'--baseline-repo',str(code),'--baseline-runtime',str(base),'--prepared-runtime',str(prep),'--runtime',str(r/'cause-eq-v1')]
  calls=[a+['--phase',phase] for phase in ('pilot','all')]
 elif args.name=='baseline21':
  common=['--runtime-dir',str(base),'--artifact-dir',str(work/'baseline-evidence'),'--config',str(REPO/'docs/experiments/01_baselines/config21.json')]
  calls=[[sys.executable,'-m','experiments.banelp.cli',step]+common for step in ('validate','lock-policy','build-wet','develop','seal')]
  calls.append([sys.executable,'-m','experiments.banelp.cli','evaluate']+common+['--seal',str(work/'baseline-evidence/selection-seal.json')])
 else:raise ValueError('unsupported stage')
 if args.name in ('train-refiner','finetune') and not args.allow_optimizer:raise ValueError('optimizer stages require --allow-optimizer')
 print(json.dumps({'commands':calls,'writes':'workspace only; archived bundle and repository results unchanged'},ensure_ascii=False,indent=2))
 if args.dry_run:return
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 for call in calls:subprocess.run(call,check=True,env=env,cwd=code)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));subs=parser.add_subparsers(dest='command',required=True)
 e=subs.add_parser('export');e.add_argument('--runtime-root',required=True);e.add_argument('--bundle',required=True);e.add_argument('--profile',choices=['validation','all'],default='all');e.add_argument('--plan',action='store_true')
 v=subs.add_parser('verify');v.add_argument('--bundle',required=True)
 v=subs.add_parser('canonicalize');v.add_argument('--bundle',required=True)
 v=subs.add_parser('pack');v.add_argument('--bundle',required=True);v.add_argument('--archive',required=True)
 v=subs.add_parser('hydrate');v.add_argument('--bundle',required=True);v.add_argument('--workspace',required=True);v.add_argument('--copy',action='store_true')
 v=subs.add_parser('stage');v.add_argument('--workspace',required=True);v.add_argument('--name',choices=['cache-refiner','preflight-refiner','train-refiner','frontend','official','finetune','hybrid','selector','consistency','cause-eq','baseline21'],required=True);v.add_argument('--allow-optimizer',action='store_true');v.add_argument('--dry-run',action='store_true')
 for name in ('input-check','infer','compose','teacher'):
  s=subs.add_parser(name);s.add_argument('--bundle',required=True);s.add_argument('--split',choices=['train','validation','confirmation'],default='validation');s.add_argument('--track-index',type=int,default=0);s.add_argument('--condition',choices=['dry','wet_R4'],default='dry');s.add_argument('--output',required=True)
  s.add_argument('--record-id')
  if name=='compose':s.add_argument('--predictions',required=True)
  if name=='teacher':s.add_argument('--limit',type=int,default=320)
  if name in ('infer','teacher'):s.add_argument('--model',choices=['htdemucs','bsr','clean','rev','finetuned'],required=True)
 s=subs.add_parser('results');s.add_argument('--output',required=True)
 args=parser.parse_args();REPO=Path(args.repo).resolve()
 if not (REPO/'experiments/banelp/separators.py').is_file():raise ValueError('--repo must point to the research repository')
 if args.command=='export':export(args)
 elif args.command=='verify':verify(args)
 elif args.command=='results':results(args)
 elif args.command=='canonicalize':canonicalize(args)
 elif args.command=='hydrate':hydrate(args)
 elif args.command=='pack':pack(args)
 elif args.command=='stage':stage(args)
 elif args.command=='compose':compose(args)
 elif args.command=='teacher':teacher(args)
 else:inference(args)

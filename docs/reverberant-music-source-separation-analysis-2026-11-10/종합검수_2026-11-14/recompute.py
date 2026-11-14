from pathlib import Path
import csv,json,hashlib
import numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--project',required=True);args=parser.parse_args()
P=Path(args.project).resolve();D=Path(__file__).resolve().parent
records=[];checks=[];sources=[]

def read(folder):
 path=P/folder/'결과/곡별결과.csv'
 with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 sources.append({'path':str(path.relative_to(P)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows)})
 table={}
 for row in rows:
  method=row.get('method',row.get('model',row.get('output')));step=row.get('step','')
  key=(step,row['track'],row['condition'],method,row['stem']);assert key not in table
  table[key]={k:(float(row[k]) if row.get(k) else None) for k in ('si_sdr','si_sar','target_gain','output_rms')}
 return table

def aggregate(table,folder,method,reference,step,seed,expected,kind='measured'):
 for cond in ('dry','wet_R4'):
  tracks=sorted({k[1] for k in table if k[0]==step and k[2]==cond})
  overall=[];bass=[];damage=[];collapsed=[]
  for name in tracks:
   deltas=[]
   for stem in ('vocals','drums','bass','other'):
    a=table[(step,name,cond,method,stem)];b=table[(step,name,cond,reference,stem)]
    d=None if a['si_sdr'] is None or b['si_sdr'] is None else a['si_sdr']-b['si_sdr']
    if d is not None:deltas.append(d)
    if stem=='bass':bass.append(d);collapsed.append((b['target_gain'],a['target_gain']))
   value=float(np.mean(deltas));overall.append(value);damage.append((value,name))
  def stat(values):
   a=np.array([v for v in values if v is not None],float);ids=np.random.default_rng(seed).integers(len(a),size=(10000,len(a)))
   return float(np.median(a)),np.percentile(np.median(a[ids],axis=1),[2.5,97.5]).tolist(),len(a)
  med,ci,n=stat(overall);bm,bci,bn=stat(bass)
  e=expected(cond);err=max(abs(med-e['median']),abs(ci[0]-e['ci'][0]),abs(ci[1]-e['ci'][1]));assert err<1e-9,(folder,method,cond,med,ci,e)
  checks.append({'experiment':folder,'method':method,'reference':reference,'condition':cond,'step':step,'max_abs_error':err})
  pairs=[(a,b) for a,b in collapsed if a is not None and b is not None]
  records.append({'experiment':folder,'method':method,'reference':reference,'step':step,'condition':cond,'kind':kind,'songs':n,'median_delta':med,'ci_low':ci[0],'ci_high':ci[1],'bass_median_delta':bm,'bass_ci_low':bci[0],'bass_ci_high':bci[1],'bass_n':bn,'bass_collapse_reference':sum(a<.1 for a,b in pairs),'bass_collapse_method':sum(b<.1 for a,b in pairs),'new_bass_collapse':sum(a>=.1 and b<.1 for a,b in pairs),'worse_than_minus_point2':sum(v<-.2 for v in overall),'worst_delta':min(damage)[0],'worst_song':min(damage)[1]})

folder='입력전처리검증_2026-11-13';t=read(folder);s=json.loads((P/folder/'결과/분석요약.json').read_text())
for method in ('raw_weighted','eq','wpe','wpe_uniform','wpe_weighted','gain_weighted'):aggregate(t,folder,method,'raw','',20261113,lambda c,m=method:s['summary'][c][m]['overall'])
folder='공식SCNet검증_2026-11-13';t=read(folder);s=json.loads((P/folder/'결과/분석요약.json').read_text())
for method,ref in [('clean','htdemucs'),('rev','htdemucs'),('rev','clean'),('rev','ensemble'),('ensemble','htdemucs')]:aggregate(t,folder,method,ref,'',20261113,lambda c,m=method,r=ref:s['comparisons'][c][m+'-minus-'+r]['overall'])
folder='SCNet미세조정_2026-11-13';t=read(folder);s=json.loads((P/folder/'결과/분석요약.json').read_text())
for step in ('500','1000'):
 e=next(v for v in s['checkpoints'] if v['step']==int(step))
 for ref in ('initial_rev','clean','htdemucs','ensemble'):aggregate(t,folder,'candidate',ref,step,20261113,lambda c,r=ref:e['comparisons'][c]['candidate-minus-'+r]['overall'])
folder='음원별결합검증_2026-11-14';hybrid_table=t=read(folder);s=json.loads((P/folder/'결과/분석요약.json').read_text())
for method in ('finetuned','hybrid','condition_label_route','gt_case_oracle'):aggregate(t,folder,method,'baseline','',20261114,lambda c,m=method:s['comparisons'][c][m]['overall'],'oracle' if method in ('condition_label_route','gt_case_oracle') else 'measured')
folder='입력선택검증_2026-11-14';path=P/folder/'결과/선택결과.csv'
with path.open(encoding='utf-8-sig') as f:decisions=list(csv.DictReader(f))
assert len(decisions)==20
sources.append({'path':str(path.relative_to(P)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':20})
t={}
for r in decisions:
 assert r['choice']==('hybrid' if float(r['sigmoid_score'])>=.8 else 'baseline')
 for stem in ('vocals','drums','bass','other'):
  for method,original in [('baseline','baseline'),('selector',r['choice'])]:t[('',r['track'],r['condition'],method,stem)]=hybrid_table[('',r['track'],r['condition'],original,stem)]
s=json.loads((P/folder/'결과/분석요약.json').read_text());aggregate(t,folder,'selector','baseline','',20261114,lambda c:s['comparisons'][c]['overall'])
folder='출력합보정검증_2026-11-14';t=read(folder);s=json.loads((P/folder/'결과/분석요약.json').read_text())
for method in ('selected_raw','selected_uniform','selected_power','baseline_uniform','baseline_power'):
 for ref in ('baseline','selected_raw'):aggregate(t,folder,method,ref,'',20261114,lambda c,m=method,r=ref:s['comparisons'][c][m][r]['overall'])
with (D/'최근실험_재계산.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(D/'정량검수근거.json').write_text(json.dumps({'source_files':sources,'source_rows':sum(v['rows'] for v in sources),'comparisons':len(checks),'max_abs_difference':max(c['max_abs_error'] for c in checks),'checks':checks,'scope':'Recomputed paired-song median/10000-bootstrap CI from existing CSV. No new inference/training. This is arithmetic consistency, not independent generalization or waveform re-audit.'},ensure_ascii=False,indent=2)+'\n')
print('Validated',len(checks),'condition/comparison rows from',sum(v['rows'] for v in sources),'CSV rows. Max abs error',max(c['max_abs_error'] for c in checks))

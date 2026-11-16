from pathlib import Path
import json,csv
import numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--project',required=True);parser.add_argument('--hybrid-runtime',required=True);args=parser.parse_args()
P=Path(args.project).resolve();R=Path(args.hybrid_runtime).resolve();D=Path(__file__).resolve().parent;out=[]
with (P/'출력합보정검증_2026-11-14/결과/곡별결과.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
t={(r['track'],r['condition'],r['method'],r['stem']):r for r in rows}
for fraction in (0.,.001,.01,.03162277660168379):
 for cond in ('dry','wet_R4'):
  for method in ('selected_raw','selected_uniform','selected_power'):
   vals=[];counts=[];excluded=[]
   for name in sorted({r['track'] for r in rows}):
    mix=json.loads((R/name/cond/'results.json').read_text())['reconstruction']['baseline']['mixture_rms'];ds=[]
    for stem in ('vocals','drums','bass','other'):
     a=t[name,cond,method,stem];b=t[name,cond,'baseline',stem]
     if not a['si_sdr'] or not b['si_sdr']:continue
     if float(b['target_rms'])<fraction*mix:excluded.append(name+' / '+stem);continue
     ds.append(float(a['si_sdr'])-float(b['si_sdr']))
    if ds:vals.append((float(np.mean(ds)),name));counts.append(len(ds))
   ar=np.asarray([v for v,n in vals]);ids=np.random.default_rng(20261114).integers(len(ar),size=(10000,len(ar)));ci=np.percentile(np.median(ar[ids],1),[2.5,97.5])
   out.append({'rms_fraction_threshold':fraction,'condition':cond,'method':method,'median_delta':float(np.median(ar)),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'song_count':len(vals),'valid_stem_song_count':sum(counts),'worst_song':min(vals)[1],'worst_delta':min(vals)[0],'excluded_nonzero_targets':'; '.join(excluded)})
with (D/'활동도_사후민감도.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
for r in out:
 if r['method']=='selected_power' and r['condition']=='dry':print(r)

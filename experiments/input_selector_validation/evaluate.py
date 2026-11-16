"""Report a genuinely input-only decision separately from label oracles."""
from pathlib import Path
import json,csv
import numpy as np


def stat(values):
    a=np.asarray([x for x in values if x is not None],np.float64)
    if not len(a):return {'median':None,'ci':[None,None],'n':0}
    ids=np.random.default_rng(20261114).integers(len(a),size=(10000,len(a)))
    return {'median':float(np.median(a)),'ci':np.percentile(np.median(a[ids],axis=1),[2.5,97.5]).tolist(),'n':len(a)}


def evaluate(rows,config):
    comparisons={};choices={};confusion={}
    for c in ('dry','wet_R4'):
        group=[r for r in rows if r['condition']==c];per={}
        def diff(r,i):
            a=r['selected_scores']['metrics'][i].get('si_sdr');b=r['baseline_scores']['metrics'][i].get('si_sdr')
            return None if a is None or b is None else a-b
        for i,s in enumerate(('vocals','drums','bass','other')):per[s]=stat([diff(r,i) for r in group])
        delta=[]
        for r in group:
            values=[diff(r,i) for i in range(4)];values=[v for v in values if v is not None];delta.append(float(np.mean(values)) if values else None)
        per['overall']=stat(delta)
        pairs=[(r['baseline_scores']['metrics'][2].get('target_gain'),r['selected_scores']['metrics'][2].get('target_gain')) for r in group];pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
        per['bass_collapse']={'baseline':sum(a<.1 for a,b in pairs),'candidate':sum(b<.1 for a,b in pairs),'new':sum(a>=.1 and b<.1 for a,b in pairs),'valid':len(pairs)}
        comparisons[c]=per;choices[c]={name:sum(r['choice']==name for r in group) for name in ('baseline','hybrid')}
        confusion[c]={'predicted_dry':choices[c]['baseline'],'predicted_wet':choices[c]['hybrid']}
    wet=comparisons['wet_R4'];dry=comparisons['dry'];primary=wet['overall'];margin=config['retention_ci_lower_margin_db']
    gates={'wet_primary':primary['median']>=.2 and primary['ci'][0]>0,'dry_overall':dry['overall']['ci'][0]>=margin,'wet_vocals':wet['vocals']['ci'][0]>=margin,'wet_other':wet['other']['ci'][0]>=margin,'no_new_bass_collapse':wet['bass_collapse']['new']==0 and dry['bass_collapse']['new']==0}
    return {'comparisons':comparisons,'choices':choices,'classification_confusion':confusion,'gates':gates,'accepted':len(rows)==20 and all(gates.values())}


def write_report(root,destination,summary,rows,model):
    dest=destination/'결과';dest.mkdir(exist_ok=True)
    fmt=lambda v:f"{v['median']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
    wet=summary['comparisons']['wet_R4'];dry=summary['comparisons']['dry']
    intro=f"입력만 사용하는 선택기의 B 대비 wet 전체 {fmt(wet['overall'])}dB, dry 전체 {fmt(dry['overall'])}dB다. 사전 개선·보호 기준은 {'통과' if summary['accepted'] else '실패'}했다. 선택기는train80곡320입력의추가RIR라벨로만적합했고검증곡정답/조건라벨은선택에사용하지않았다."
    text=['# 입력만 사용하는 선택기 검증 결과','', '2026-11-14 · train80곡320입력 · validation10곡20조건 · 분리기추가학습0/GPU추론0','',intro,'',
        '| 조건 | 전체 ΔSI-SDR [95% CI] | vocals | drums | bass | other | B/H 선택 |','|---|---|---|---|---|---|---|']
    for c,values in summary['comparisons'].items():
        choices=summary['choices'][c];text.append('| '+c+' | '+fmt(values['overall'])+' | '+' | '.join(fmt(values[s]) for s in ('vocals','drums','bass','other'))+f" | {choices['baseline']}/{choices['hybrid']} |")
    text+=['','## 채택과 해석','',
        '각 사전 기준: '+', '.join(f'{k}={v}' for k,v in summary['gates'].items())+'.','',
        '실제파형만으로판단하는결과이며이전조건라벨분기와GT oracle성능으로대체하지않는다. L2규제0.01,threshold0.8,특징33개를검증예측전에고정했고검증결과를보고바꾸지않았다. sigmoid score는교정된확률이아니다. train만으로특징표준화를적합했다. 중심26초를사용해합성guard와원래30초crop의가장자리2초를제외했다.','',
        'source정답과품질점수는선택기적합에사용하지않았다. 비교지표는고정결합에서동일입력/정답/예측SHA와실제파형결합을검증한B/H score를선택해계산했다. 같은validation을보고설계한탐색적실험이며새곡일반화,실제음악의잔향판별,청감우위를입증한것은아니다. 미사용confirmation도아직평가하지않았다.','',
        '[사전 계획](../PLAN.md) · [고정 설정](../protocol.json) · [고정결합결과](../../음원별결합검증_2026-11-14/결과/결과보고서.md) · [선택기 파라미터](선택기.json)','',
        '| 곡 | 조건 | sigmoid score | 실제선택 | 전체ΔSI-SDR |','|---|---|---|---|---|']
    for r in rows:
        diffs=[a['si_sdr']-b['si_sdr'] for a,b in zip(r['selected_scores']['metrics'],r['baseline_scores']['metrics']) if a.get('si_sdr') is not None and b.get('si_sdr') is not None]
        text.append(f"| {r['track']} | {r['condition']} | {r['sigmoid_score']:.6f} | {r['choice']} | {float(np.mean(diffs)):+.3f} |")
    (dest/'결과보고서.md').write_text('\n'.join(text)+'\n');(dest/'분석요약.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(dest/'선택기.json').write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
    with (dest/'선택결과.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['track','condition','sigmoid_score','choice']);writer.writeheader();writer.writerows({k:r[k] for k in ('track','condition','sigmoid_score','choice')} for r in rows)
    readme=destination/'README.md';body=readme.read_text();a=body.index('<!-- RESULT START -->');b=body.index('<!-- RESULT END -->');readme.write_text(body[:a]+'<!-- RESULT START -->\n'+intro+'\n\n[상세 결과](결과/결과보고서.md)\n'+body[b:])
    assert intro in readme.read_text() and len((dest/'결과보고서.md').read_text())>1000

"""Report source quality separately from algebraic reconstruction accuracy."""
from pathlib import Path
import csv,json
import numpy as np
SOURCES=('vocals','drums','bass','other')


def paired(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],np.float64)
    if not len(a):return {'median':None,'ci':[None,None],'n':0}
    ids=np.random.default_rng(20261114).integers(len(a),size=(10000,len(a)))
    return {'median':float(np.median(a)),'ci':np.percentile(np.median(a[ids],axis=1),[2.5,97.5]).tolist(),'n':len(a)}


def differences(row,method,reference,metric='si_sdr',stem=None):
    values=[]
    for i in range(4) if stem is None else [stem]:
        a=row['scores'][method]['metrics'][i].get(metric);b=row['scores'][reference]['metrics'][i].get(metric)
        if a is not None and b is not None:values.append(a-b)
    return float(np.mean(values)) if values else None


def summarize(rows,config):
    out={'comparisons':{},'reconstruction':{},'individual_damage':{},'silent_bass':[]}
    for c in config['conditions']:
        group=[r for r in rows if r['condition']==c];out['comparisons'][c]={};out['reconstruction'][c]={};out['individual_damage'][c]={}
        for method in config['methods']:
            out['comparisons'][c][method]={}
            for reference in ('baseline','selected_raw'):
                per={s:paired([differences(r,method,reference,stem=i) for r in group]) for i,s in enumerate(SOURCES)}
                per['overall']=paired([differences(r,method,reference) for r in group]);per['si_sar']=paired([differences(r,method,reference,'si_sar') for r in group])
                gains=[(r['scores'][reference]['metrics'][2].get('target_gain'),r['scores'][method]['metrics'][2].get('target_gain')) for r in group];gains=[(a,b) for a,b in gains if a is not None and b is not None]
                per['bass_collapse']={'reference':sum(a<.1 for a,b in gains),'candidate':sum(b<.1 for a,b in gains),'new':sum(a>=.1 and b<.1 for a,b in gains),'valid':len(gains)}
                out['comparisons'][c][method][reference]=per
            damages=[{'track':r['track'],'delta':differences(r,method,'baseline')} for r in group];damages=[v for v in damages if v['delta'] is not None]
            out['individual_damage'][c][method]={'worst':min(damages,key=lambda v:v['delta']),'below_minus_point2':sum(v['delta']<-.2 for v in damages),'songs':damages}
            vals=[r['checks'][method]['core30']['relative_residual_rms'] for r in group]
            out['reconstruction'][c][method]={'median_relative_rms':float(np.median(vals)),'max_relative_rms':float(max(vals)),
                'full40_max_abs':max(r['checks'][method]['full40']['max_abs'] for r in group),'protected_max_abs':max(r['checks'][method]['protected_max_abs'] for r in group)}
        for r in group:
            if r['scores']['baseline']['metrics'][2]['state']=='silent-target':
                out['silent_bass'].append({'track':r['track'],'condition':c,'output_rms':{m:r['scores'][m]['metrics'][2]['output_rms'] for m in config['methods']}})
    def gates(method):
        wet=out['comparisons']['wet_R4'][method];dry=out['comparisons']['dry'][method];wb=wet['baseline'];wr=wet['selected_raw'];limit=config['gates']
        low=lambda v:v['ci'][0] if v['ci'][0] is not None else -np.inf
        return {'mixture_sum':all(out['reconstruction'][c][method]['full40_max_abs']<=limit['full40_sum_max_abs'] for c in config['conditions']),
            'protected_waveforms':all(out['reconstruction'][c][method]['protected_max_abs']==0 for c in config['conditions']),
            'wet_gain_vs_baseline':wb['overall']['median']>=.2 and low(wb['overall'])>0,
            'wet_retention_vs_raw':low(wr['overall'])>=-.2,'wet_drums_retention':low(wr['drums'])>=-.2,'wet_bass_retention':low(wr['bass'])>=-.2,
            'dry_overall':low(dry['baseline']['overall'])>=-.2,'wet_si_sar':low(wr['si_sar'])>=-.5,
            'each_dry_song':out['individual_damage']['dry'][method]['worst']['delta']>=-.2,
            'no_new_bass_collapse':all(out['comparisons'][c][method][ref]['bass_collapse']['new']==0 for c in config['conditions'] for ref in ('baseline','selected_raw'))}
    out['candidate_gates']={m:gates(m) for m in ('selected_uniform','selected_power')};out['gates']=out['candidate_gates'][config['primary']]
    out['accepted']=len(rows)==20 and all(out['gates'].values())
    out['numeric_checks']={key:max((r[key] for r in rows if r[key] is not None),default=None) for key in ('baseline_metric_max_error','selected_metric_max_error','legacy_metric_max_error')}
    return out


def write_report(root,destination,rows,summary,config):
    if len(rows)!=20:raise ValueError('incomplete comparison')
    output=destination/'결과';output.mkdir(exist_ok=True)
    fmt=lambda s:'미정의' if s['median'] is None else f"{s['median']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]"
    main=summary['comparisons']['wet_R4']['selected_power'];dry=summary['comparisons']['dry']['selected_power']
    intro=f"주후보 에너지 배분의 B 대비 wet 전체는 {fmt(main['baseline']['overall'])}dB, dry 전체는 {fmt(dry['baseline']['overall'])}dB다. 미보정 선택기 대비 wet 변화는 {fmt(main['selected_raw']['overall'])}dB이며, 사전 성능·합·개별곡 보호 기준은 {'모두 통과했다' if summary['accepted'] else '동시에 통과하지 못했다'}. 합 오차 감소와 분리 성능 개선을 구분한다."
    text=['# vocals other 보존 출력합 보정 결과','', '2026-11-14 · 같은 validation10곡 dry/wet20조건 · 40초입력/중앙30초 · 추가학습/GPU추론0','',intro,'',
        '기존 선택기의 threshold0.8과20개B/H선택을변경하지않았다. 보정은B를고른경우도포함한모든출력에적용했다. vocals/other파형을그대로복사하고두가지drums/bass보정으로잔차를배분했다. selected_power가사전주후보이며selected_uniform은대조다. baseline_uniform/power는기존앙상블자체에동일한보정을적용한대조다.','',
        '| 조건 | 방법 | 전체 ΔSI-SDR 대 B | 전체 ΔSI-SDR 대 미보정선택 | bass ΔSI-SDR 대 미보정선택 |','|---|---|---|---|---|']
    for c in config['conditions']:
        for m in config['methods']:
            v=summary['comparisons'][c][m];text.append(f"| {c} | {m} | {fmt(v['baseline']['overall'])} | {fmt(v['selected_raw']['overall'])} | {fmt(v['selected_raw']['bass'])} |")
    text+=['','## 사전 기준과 음원별 변화','', '| 후보 | 기준 | 결과 |','|---|---|---|']
    for m,g in summary['candidate_gates'].items():
        for k,v in g.items():text.append(f"| {m} | {k} | {'통과' if v else '실패'} |")
    text+=['','균등 대조가 더 좋은 지표를 보여도 사전 주후보를 바꾸지 않는다. 각 dry 곡 손실−0.2dB 제한은 이번에 추가한 사전 기준이며 이전 실험의 기준을 사후 변경한 것이 아니다.','',
        '| 조건 | 방법 대 미보정선택 | vocals | drums | bass | other |','|---|---|---|---|---|']
    for c in config['conditions']:
        for m in ('selected_uniform','selected_power'):
            v=summary['comparisons'][c][m]['selected_raw'];text.append('| '+c+' | '+m+' | '+' | '.join(fmt(v[s]) for s in SOURCES)+' |')
    text+=['','## 출력 합과 보호 음원','', '| 조건 | 방법 | 상대잔차RMS 중앙값 | 전체40초 최대절대오차 | 보호음원 최대차이 |','|---|---|---|---|---|']
    for c in config['conditions']:
        for m,v in summary['reconstruction'][c].items():text.append(f"| {c} | {m} | {v['median_relative_rms']:.3g} | {v['full40_max_abs']:.3g} | {v['protected_max_abs']:.3g} |")
    text+=['','상대잔차RMS는RMS(X−음원합)/RMS(X)이며잔향량이나청감지표가아니다. 합이일치해도고정한vocals/other의오차를drums/bass에넘길수있다. 두자유음원이무음인bin도잔차를절반씩받으므로유령성분가능성은별도로검토해야한다.','',
        '## 베이스 소실과 무음 정답','', '| 조건 | 방법 | 소실 B→후보 | 새 소실 대 B | 새 소실 대 미보정선택 |','|---|---|---|---|---|']
    for c in config['conditions']:
        for m in ('selected_raw','selected_uniform','selected_power'):
            a=summary['comparisons'][c][m]['baseline']['bass_collapse'];b=summary['comparisons'][c][m]['selected_raw']['bass_collapse']
            text.append(f"| {c} | {m} | {a['reference']}/{a['valid']}→{a['candidate']}/{a['valid']} | {a['new']} | {b['new']} |")
    for r in summary['silent_bass']:
        text+=['',r['condition']+' 무음bass '+r['track']+': '+', '.join(m+' RMS='+f"{r['output_rms'][m]:.6g}" for m in ('baseline','selected_raw','selected_uniform','selected_power'))+'.']
    text+=['','## 개별 dry 실패','', '| 방법 | −0.2dB보다 악화된 곡 수 | 최악곡 | 변화 대 B |','|---|---|---|---|']
    for m in config['methods']:
        d=summary['individual_damage']['dry'][m];text.append(f"| {m} | {d['below_minus_point2']}/10 | {d['worst']['track']} | {d['worst']['delta']:+.3f} |")
    text+=['','## 범위와 재현성','',
        '전체값은곡별유효음원의대응차이를평균한후10곡중앙값과10,000회곡bootstrap95%CI를계산했다. 무음bass는소실/지표의분모에서제외하고RMS를따로표시했다. B/미보정선택기의지표재계산일치,원래입력/정답/제공파형/선택기/코드SHA,전체40초합오차와보호음원byte동일성을검사했다. 보정파형80개와음원별지표480행을runtime와결과파일에보존했다.','',
        '같은노출validation에서의탐색적결과다. 공개모델사전학습중복가능성과선택기설계의노출은남아독립일반화입증이나실사용채택은아니다. confirmation·새곡/RIR·청감평가는이번에수행하지않았다.','',
        '[사전 계획](../PLAN.md) · [고정 설정](../protocol.json) · [집계 JSON](분석요약.json) · [곡별 CSV](곡별결과.csv) · [입력선택기 결과](../../입력선택검증_2026-11-14/결과/결과보고서.md)']
    (output/'결과보고서.md').write_text('\n'.join(text)+'\n');(output/'분석요약.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    csvrows=[]
    for r in rows:
        for m,s in r['scores'].items():
            for row in s['metrics']:csvrows.append({'track':r['track'],'condition':r['condition'],'method':m,**row})
    with (output/'곡별결과.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['track','condition','method','stem','state','si_sdr','si_sir','si_sar','sd_sdr','target_gain','target_rms','output_rms','rank'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(csvrows)
    readme=destination/'README.md';body=readme.read_text();a=body.index('<!-- RESULT START -->');b=body.index('<!-- RESULT END -->');readme.write_text(body[:a]+'<!-- RESULT START -->\n'+intro+'\n\n[상세 결과](결과/결과보고서.md)\n'+body[b:])
    assert intro in readme.read_text() and len((output/'결과보고서.md').read_text())>2000

"""Paired-song statistics and a report that separates oracle diagnostics."""
from pathlib import Path
import csv,json
import numpy as np

SOURCES=('vocals','drums','bass','other')


def paired(values,config):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=np.float64)
    if not len(a):return {'median':None,'ci':[None,None],'n':0,'excluded':len(values)}
    rng=np.random.default_rng(config['bootstrap']['seed'])
    ids=rng.integers(len(a),size=(config['bootstrap']['iterations'],len(a)))
    boots=np.median(a[ids],axis=1)
    return {'median':float(np.median(a)),'ci':np.percentile(boots,config['bootstrap']['interval']).tolist(),'n':len(a),'excluded':len(values)-len(a)}


def summarize(rows,config):
    out={'comparisons':{},'reconstruction':{},'diagnostic_choices':{}}
    for c in config['conditions']:
        group=[r for r in rows if r['condition']==c];out['comparisons'][c]={};out['reconstruction'][c]={}
        for name in config['methods']:
            def deltas(metric,stem=None):
                values=[]
                for r in group:
                    d=[]
                    for i in range(4) if stem is None else [stem]:
                        a=r['scores'][name]['metrics'][i].get(metric);b=r['scores']['baseline']['metrics'][i].get(metric)
                        if a is not None and b is not None:d.append(a-b)
                    values.append(float(np.mean(d)) if d else None)
                return values
            per={s:paired(deltas('si_sdr',i),config) for i,s in enumerate(SOURCES)}
            per['overall']=paired(deltas('si_sdr'),config);per['overall_si_sar']=paired(deltas('si_sar'),config)
            per['improved_songs']=sum(v>1e-9 for v in deltas('si_sdr') if v is not None)
            gains=[(r['scores']['baseline']['metrics'][2].get('target_gain'),r['scores'][name]['metrics'][2].get('target_gain')) for r in group]
            gains=[(a,b) for a,b in gains if a is not None and b is not None]
            per['bass_collapse']={'baseline':sum(a<.1 for a,b in gains),'candidate':sum(b<.1 for a,b in gains),'new':sum(a>=.1 and b<.1 for a,b in gains),'valid':len(gains)}
            out['comparisons'][c][name]=per
            residual=[r['reconstruction'][name]['relative_residual_rms'] for r in group]
            out['reconstruction'][c][name]={'median_relative_rms':float(np.median(residual)),'max_relative_rms':float(max(residual))}
        out['diagnostic_choices'][c]={name:{choice:sum(r['diagnostic_choices'][name]==choice for r in group) for choice in ('baseline','hybrid')} for name in ('condition_label_route','gt_case_oracle')}
    wet=out['comparisons']['wet_R4']['hybrid'];dry=out['comparisons']['dry']['hybrid'];p=wet['overall'];margin=config['noninferiority_ci_lower_margin_db']
    gates={'wet_primary':p['median'] is not None and p['median']>=config['minimum_gain_db'] and p['ci'][0]>0,
           'dry_overall':dry['overall']['ci'][0] is not None and dry['overall']['ci'][0]>=margin,
           'wet_vocals':wet['vocals']['ci'][0] is not None and wet['vocals']['ci'][0]>=margin,
           'wet_other':wet['other']['ci'][0] is not None and wet['other']['ci'][0]>=margin,
           'no_new_bass_collapse':wet['bass_collapse']['new']==0 and dry['bass_collapse']['new']==0}
    out['gates']=gates;out['accepted']=len(rows)==20 and all(gates.values())
    keys=('protected_stem_max_abs','baseline_metric_max_error','finetuned_metric_max_error','spliced_metric_max_error','legacy_metric_max_error')
    out['checks']={key:max((r['checks'][key] for r in rows if r['checks'][key] is not None),default=None) for key in keys}
    return out


def write_report(root,destination,rows,summary,config):
    if len(rows)!=20:raise ValueError('incomplete report')
    dest=destination/'결과';dest.mkdir(exist_ok=True)
    fmt=lambda v:'미정의' if v['median'] is None else f"{v['median']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
    wet=summary['comparisons']['wet_R4']['hybrid'];dry=summary['comparisons']['dry']['hybrid']
    intro=f"고정 결합의 기존 앙상블 대비 wet 전체 변화는 {fmt(wet['overall'])}dB, dry 전체는 {fmt(dry['overall'])}dB다. 사전 개선·보호 기준은 {'모두 통과했다' if summary['accepted'] else '모두 통과하지 못했다'}. vocals/other는 기존 파형과 비트 단위로 같지만, 이것이 dry의 drums/bass 성능이나 출력합 보존을 보장하지 않는다."
    text=['# 음원별 고정 결합 검증 결과','',
          '2026-11-14 · validation10곡 dry/wet20조건 · 20파형 결합 · 추가 학습0회·신규 GPU추론0회','',intro,'',
          'H(vocals,drums,bass,other)=(B_vocals,F_drums,F_bass,B_other)로 고정했다. B는 HTDemucs+BS-RoFormer 50:50, F는 이미 선택된 미세조정 SCNet step1000이다. 40초 입력·중앙30초·원래 잔향 정답으로 재평가했다. 조건라벨과 정답을 사용하는 두 선택 대조는 진단값이며 실제 추론 결과가 아니다.','',
          '| 조건 | 방법 | 전체 ΔSI-SDR [95% CI] | vocals | drums | bass | other |','|---|---|---|---|---|---|---|']
    for c in config['conditions']:
        for name,v in summary['comparisons'][c].items():
            if name=='baseline':continue
            text.append('| '+c+' | '+name+' | '+fmt(v['overall'])+' | '+' | '.join(fmt(v[s]) for s in SOURCES)+' |')
    text+=['','## 채택 기준','']
    for k,v in summary['gates'].items():text.append('- '+k+': '+('통과' if v else '실패'))
    text+=['','주요 기준은 B 대비 wet 전체 중앙값≥+0.2dB/CI하한>0, dry전체 및wet vocals/other CI하한≥−0.2dB, 새bass소실0이다. 고정결합이실패하면진단oracle의좋은값으로성공판정을바꾸지않는다. 무음정답은제외했고bass는9곡이다. 전체값은곡별유효음원대응차이평균의10곡중앙값이며10,000회곡bootstrap95%CI를표시했다.','',
           '| 조건 | 방법 | bass 소실 B→후보 | 새 소실 |','|---|---|---|---|']
    for c in config['conditions']:
        for name in ('finetuned','hybrid','condition_label_route','gt_case_oracle'):
            b=summary['comparisons'][c][name]['bass_collapse'];text.append(f"| {c} | {name} | {b['baseline']}/{b['valid']}→{b['candidate']}/{b['valid']} | {b['new']} |")
    text+=['','## 음원합 진단','',
           '상대잔차RMS=RMS(X−음원합)/RMS(X)로 중앙30초에서 측정했다. 이는 잔향량·에너지비율·청감 점수가 아니다. 큰 값은 출력합과 입력의 불일치를 뜻하며, 어느 성분이 중복/누락됐는지 이 값 하나로 확정하지 않는다. vocals 보호를 바꾸는 잔차 재배분은 적용하지 않았다.','',
           '| 조건 | 방법 | 상대잔차RMS 중앙값 | 최대 |','|---|---|---|---|']
    for c in config['conditions']:
        for name in ('baseline','finetuned','hybrid'):
            v=summary['reconstruction'][c][name];text.append(f"| {c} | {name} | {v['median_relative_rms']:.4f} | {v['max_relative_rms']:.4f} |")
    text+=['','## 진단 선택과 실제 적용의 차이','',
           'condition_label_route는 합성시알고있는dry/wet라벨로dry에는B,wet에는H를쓴다. gt_case_oracle은정답으로곡전체의B/H중유효4음원평균SI-SDR가좋은출력을고른다. 둘다추가정보를사용하므로입력만으로선택하는시스템의실제성능이아니다. oracle상한은이두후보중곡전체선택단위에만해당한다. 아직정답없는선택기는학습/평가하지않았다.','']
    for c,choices in summary['diagnostic_choices'].items():
        for name,v in choices.items():text.append(f"- {c} {name}: B {v['baseline']}곡 / H {v['hybrid']}곡")
    text+=['','## 재현성·해석 범위','',
           '입력·정답SHA를이전공식비교와대조했고step1000 checkpoint SHA,저장된SCNet/HTDemucs/BS-RoFormer의60개출력파일SHA와binding을확인했다. 실제H파형20개를생성해지표를재계산했다. 기존B/F의지표재계산과H의음원별지표가제공모델의해당음원지표와일치하는지도모든조건에서검사했다. 보호음원최대절대차이는 '+str(summary['checks']['protected_stem_max_abs'])+'이다. 지표검사오차는 [실행근거](실행근거.json)에보존했다.','',
           '같은validation10곡의실패/성공을보고설계한후속실험으로탐색적이다. 공개백본사전학습중복과checkpoint선택의영향은남아있다. confirmation·새곡·새RIR·청감평가는이번에수행하지않았다.','',
           '[SCNet 미세조정 결과](../../SCNet미세조정_2026-11-13/결과/결과보고서.md)의음원별상보성과출력을가져왔고,이번에는고정결합의dry보호·복원합을새로검증했다. [공식 비교](../../공식SCNet검증_2026-11-13/결과/결과보고서.md)와[입력전처리검증](../../입력전처리검증_2026-11-13/결과/결과보고서.md)에서는입력합성과기준선cache만재사용했다.','',
           '[사전 계획](../PLAN.md) · [고정 설정](../protocol.json) · [집계 JSON](분석요약.json) · [곡별 CSV](곡별결과.csv)']
    (dest/'결과보고서.md').write_text('\n'.join(text)+'\n')
    (dest/'분석요약.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    proof={'binding':summary['binding'],'cases':len(rows),'input_prediction_files_verified':60,'hybrid_files':20,'training_updates':0,'new_gpu_inferences':0,'checks':summary['checks'],'checkpoint_sha256':rows[0]['checkpoint_sha256']}
    (dest/'실행근거.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    values=[]
    for r in rows:
        for name,scores in r['scores'].items():
            for m in scores['metrics']:values.append({'track':r['track'],'condition':r['condition'],'method':name,**m})
    with (dest/'곡별결과.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['track','condition','method','stem','state','si_sdr','si_sir','si_sar','sd_sdr','target_gain','target_rms','output_rms','rank']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(values)
    readme=destination/'README.md';body=readme.read_text();a=body.index('<!-- RESULT START -->');b=body.index('<!-- RESULT END -->')
    readme.write_text(body[:a]+'<!-- RESULT START -->\n'+intro+'\n\n[상세 결과](결과/결과보고서.md)\n'+body[b:])
    assert len((dest/'결과보고서.md').read_text())>2000 and intro in readme.read_text()

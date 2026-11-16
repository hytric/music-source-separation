"""Paired song bootstrap and a report that separates cause, intervention and use."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from banelp_refine.common import SOURCES,atomic_json,read_json,sha256


def item(row, condition, output, stem):
    return row['records'][condition][output]['metrics'][SOURCES.index(stem)]


def number(row, condition, output, stem, metric='si_sdr'):
    return item(row,condition,output,stem).get(metric)


def difference(row,condition,a,b,stem):
    x,y=number(row,condition,a,stem),number(row,condition,b,stem)
    return None if x is None or y is None else x-y


def stat(values,seed=20261112):
    valid=np.asarray([x for x in values if x is not None and np.isfinite(x)],dtype=np.float64)
    if not len(valid):return {'median':None,'ci':[None,None],'n':0,'excluded':len(values)}
    ids=np.random.default_rng(seed).integers(len(valid),size=(10000,len(valid)))
    boot=np.median(valid[ids],axis=1)
    return {'median':float(np.median(valid)),'ci':np.percentile(boot,[2.5,97.5]).tolist(),
            'n':len(valid),'excluded':len(values)-len(valid)}


def subtract(a,b):
    return None if a is None or b is None else a-b


def collapses(rows,condition,output):
    values=[number(r,condition,output,'bass','target_gain') for r in rows]
    return {'count':sum(v<.1 for v in values if v is not None),'valid':sum(v is not None for v in values)}


def summarize(run):
    core=[read_json(p) for p in sorted((run/'core').glob('*/results.json'))]
    apply=[read_json(p) for p in sorted((run/'apply').glob('*/results.json'))]
    mechanism={}
    for stem in SOURCES:
        effects={c:[difference(r,c,'base','order',stem) for r in core] for c in ('M','A','H')}
        mechanism[stem]={c:stat(v) for c,v in effects.items()}
        mechanism[stem]['M_minus_A']=stat([subtract(a,b) for a,b in zip(effects['M'],effects['A'])])
        mechanism[stem]['interaction']=stat([None if None in v else v[0]-v[1]-v[2]
                                            for v in zip(effects['H'],effects['M'],effects['A'])])
    eq={}
    for group,rows,conditions in [('core',core,('D','H')),('apply',apply,('dry','wet'))]:
        eq[group]={}
        for c in conditions:
            eq[group][c]={s:{'vs_base':stat([difference(r,c,'eq','base',s) for r in rows]),
                             'vs_scalar':stat([difference(r,c,'eq','scalar',s) for r in rows])} for s in SOURCES}
            eq[group][c]['collapse']={o:collapses(rows,c,o) for o in ('base','eq','scalar')}
            eq[group][c]['new_collapse']=sum(number(r,c,'base','bass','target_gain') is not None and
                 number(r,c,'eq','bass','target_gain') is not None and
                 number(r,c,'base','bass','target_gain')>=.1 and number(r,c,'eq','bass','target_gain')<.1 for r in rows)
        eq[group]['wet_minus_dry_gain']=stat([subtract(difference(r,conditions[1],'eq','base','bass'),
                                                   difference(r,conditions[0],'eq','base','bass')) for r in rows])
    maxmetric=max([r['metric_equivalence_max_db'] for r in core],default=None)
    return {'complete':len(core)==10 and len(apply)==10,'core_songs':len(core),'apply_songs':len(apply),
            'mechanism':mechanism,'eq':eq,'max_legacy_metric_difference_db':maxmetric,
            'bootstrap':'10000 paired song resamples; exploratory; each condition is retained in its song',
            'optimizer_updates':0,'report_code_sha256':sha256(__file__)}


def fmt(s):
    if s['median'] is None:return '미정의'
    return f"{s['median']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] (n={s['n']})"


def positive(s):
    return s['n']==10 and s['ci'][0] is not None and s['ci'][0]>0


def verdict(s):
    if not s['complete']:return ['실험 진행 중이다. 전체 집합의 결론을 내리지 않는다.']
    m=s['mechanism']['bass'];e=s['eq'];out=[]
    reproduced=m['H']['median']<=-.5 and m['H']['ci'][1]<0
    color=m['M']['median']<=-.5 and m['M']['ci'][1]<0 and m['M_minus_A']['ci'][1]<0
    phase=m['A']['ci'][1]<0
    if not reproduced:out.append('공통 RIR에서 사전 기준에 맞는 추가 베이스 악화가 재현되지 않았다. 이 단순화로 원래 실패의 원인을 설명할 수 없다.')
    elif color:out.append('공통 RIR 대조에서는 크기 변화의 추가 베이스 손실과 위상 변화 대비 더 큰 손실이 관찰돼 착색 기여 가설을 지지한다.')
    else:out.append('공통 RIR의 추가 악화는 재현됐지만 크기 변화가 위상 변화보다 주된 원인이라는 사전 기준은 통과하지 못했다.')
    if phase:out.append('위상 조건도 추가 손실을 보여 주파수 착색만으로 설명할 수 없다.')
    core=e['core']['H']['bass']
    if core['vs_base']['median']>=.5 and positive(core['vs_base']) and positive(core['vs_scalar']) and positive(e['core']['wet_minus_dry_gain']):
        out.append('알려진 RIR을 사용하는 EQ 개입은 공통 잔향 조건의 사전 개선 기준을 통과했다. 이는 oracle 개입이다.')
    else:out.append('알려진 RIR EQ의 공통 잔향 조건 개선은 사전에 정한 모든 기준을 통과하지 못했다.')
    practical=e['apply']['wet'];bass=practical['bass']['vs_base']
    protected=all(practical[k]['vs_base']['median'] is not None and practical[k]['vs_base']['median']>=-.2 for k in ('vocals','drums','other'))
    dry=e['apply']['dry']['bass']['vs_base']
    if bass['median']>=.5 and positive(bass) and protected and dry['median']>=-.2 and practical['new_collapse']==0:
        out.append('입력만 이용한 EQ는 이번 원래 음원별 RIR 집합에서 단순 보정 후보 기준을 통과했다. 새 자료와 청취 검증이 남아 있다.')
    else:out.append('입력만 이용한 EQ는 원래 음원별 RIR 조건의 실용 후보 기준을 통과하지 못했다. 현재 방법이 원래 문제를 해결했다고 말하지 않는다.')
    return out


def write_report(run,destination,failure=None):
    destination.mkdir(parents=True,exist_ok=True)
    summary=summarize(run)
    if failure:
        summary['complete']=False;summary['execution_failure']=failure
    atomic_json(destination/'분석요약.json',summary)
    sentences=verdict(summary)
    lines=['# 잔향 크기와 위상 원인 대조 및 EQ 실험 결과','',
           '2026년 11월 12일 · GTX 1660 · 고정 HTDemucs · 신경망 학습 없음','',
           '**'+('실험을 완료했다.' if summary['complete'] else ('실행이 중단돼 부분 집계만 남겼다.' if failure else '중간 집계다.'))+'** '+' '.join(sentences),'',
           '## 실험 범위','',
           f"원인 대조 {summary['core_songs']}곡과 원래 음원별 RIR 적용 확인 {summary['apply_songs']}곡이다. 두 집합은 이미 분석·준비에 노출된 자료로 탐색적 결과다. 기존 보정망의 confirmation10과 학습 가중치는 사용하거나 변경하지 않았다.",'',
           '모델 입력은 모든 조건에서 약 190.22초이며 같은 중앙 30초를 채점했다. 기존 95초 pilot은 위상 조건의 경계/제외 에너지 검사 실패로 중단됐고 D·M 두 출력은 집계에서 제외했다. DSP 계산은 약 760.87초로 확장했다. 이전 380초 run은 다음 곡의 A 경계 오차 1.69e-4로 중단됐으며 이전 결과와 합치지 않는다. 제외된 입력 에너지 상한 1e-5는 성능 관찰 전 추가 수치 기준으로 정했다. 원래 core 경계 기준 1e-4·EQ 왕복 1e-6과 효과 판정 기준은 유지했다.','',
           'M 조건은 크기 응답의 변화이며 비인과적 시간 퍼짐도 생길 수 있다. A는 단위 크기 응답의 잔여 위상 연산이다. 이를 물리적 초기/후기 잔향 또는 RT60의 완전한 분해라고 해석하지 않는다.','',
           '## 같은 정답의 순서 대조 대비 추가 손실','',
           'E는 조건 입력을 분리한 점수에서 기준 분리 출력을 같은 필터로 변환한 점수를 뺀 값이다. 음수면 분리 전에 그 변화가 들어갈 때 더 나빴다. 단위 dB, 중앙값 [곡 bootstrap 95% CI].','',
           '| 음원 | 크기 M | 위상 A | 전체 H | M−A |','|---|---|---|---|---|']
    for stem,m in summary['mechanism'].items():lines.append('| '+stem+' | '+' | '.join(fmt(m[c]) for c in ('M','A','H','M_minus_A'))+' |')
    lines+=['','## 같은 wet 정답에서 EQ 보정 효과','',
            '아래 변화는 역EQ까지 적용한 출력과 무보정 출력을 비교한다. 정답은 원래의 잔향 stem으로 유지했다. known-RIR EQ는 원인 개입용 oracle이며 입력 기반 EQ와 구분한다.','',
            '| 집합과 조건 | vocals | drums | bass | other | 베이스 소실 기준→EQ |','|---|---|---|---|---|---|']
    for group,conds in [('core',('D','H')),('apply',('dry','wet'))]:
        for c in conds:
            v=summary['eq'][group][c];base=v['collapse']['base'];after=v['collapse']['eq']
            lines.append('| '+group+'/'+c+' | '+' | '.join(fmt(v[k]['vs_base']) for k in SOURCES)+f" | {base['count']}/{base['valid']}→{after['count']}/{after['valid']} |")
    lines+=['','## EQ 특이성 대조','',
            '| 비교 | 베이스 SI-SDR 차이 |','|---|---|']
    for group,c in [('core','H'),('apply','wet')]:
        lines.append('| '+group+' EQ−scalar | '+fmt(summary['eq'][group][c]['bass']['vs_scalar'])+' |')
        lines.append('| '+group+' 잔향 개선−대조 개선 | '+fmt(summary['eq'][group]['wet_minus_dry_gain'])+' |')
    lines+=['','단순한 유의성 부재를 효과 없음으로 단정하지 않는다. 각 음원의 손상과 새 소실을 함께 확인한다. 투영 계수는 에너지 이동 비율이 아니다.','',
            '## 검증과 원자료','',
            f"기존 float64 평가 함수와 새 공통 계산 경로의 최대 차이는 {summary['max_legacy_metric_difference_db']} dB였다. 각 곡의 수치·경계·EQ 왕복 검사는 별도 runtime의 results.json과 boundary.json에 남겼다.",'',
            '설정과 구현은 [원인 실험 설계](../CAUSE_EXPERIMENT.md), configs/cause_eq.json, cause_eq/의 코드다. 원래 설계와 이번 실행의 DSP 보완을 구분한다. 원 음원·모델·95초 출력 배열은 Git에 포함하지 않는다.','',
            '이 결과의 조건부 원인 지지, EQ 개입 효과, 입력 기반 방법의 적용 가능성은 각각 위에서 판정했다. 새로운 곡·공간·청취 성능이나 학습 보정망의 성능으로 확대하지 않는다.']
    (destination/'결과보고서.md').write_text('\n'.join(lines)+'\n')
    state='완료' if summary['complete'] else ('중단' if failure else '진행 중')
    (destination/'README.md').write_text('# 원인 대조와 전후 EQ 실험 결과\n\n상태: '+state+'\n\n'+
        f"원인 대조 {summary['core_songs']}/10곡, 적용 확인 {summary['apply_songs']}/10곡을 집계했다. "+
        ' '.join(sentences)+'\n\n[상세 결과](결과보고서.md) · [집계 JSON](분석요약.json) · [곡별 CSV](곡별_조건별_결과.csv)\n\n'+
        '중앙 30초·원래 wet 정답으로 평가하며 고정 HTDemucs만 사용했다. 보정망 가중치 업데이트는 0회다.\n')
    rows=[]
    for group in ('core','apply'):
        for p in sorted((run/group).glob('*/results.json')):
            r=read_json(p)
            for condition,outputs in r['records'].items():
                for output,value in outputs.items():
                    for m in value['metrics']:
                        rows.append({'group':group,'track':r['track'],'condition':condition,'output':output,
                                     **{k:v for k,v in m.items() if k!='metric_states'}})
    fields=['group','track','condition','output','stem','state','si_sdr','si_sir','si_sar','sd_sdr','target_gain','target_rms','output_rms','rank']
    with (destination/'곡별_조건별_결과.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();s=write_report(a.run,a.output);print(s['complete'],s['core_songs'],s['apply_songs'])


if __name__=='__main__':main()

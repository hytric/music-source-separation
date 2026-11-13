"""Paired-song validation and retention gates, fixed before optimization."""
from pathlib import Path
import csv,json
import numpy as np
from core import SOURCES


def summarize(rows,config,paired_stat):
    result={'comparisons':{}}
    for condition in ('dry','wet_R4'):
        group=[r for r in rows if r['condition']==condition];result['comparisons'][condition]={}
        for baseline in ('initial_rev','clean','htdemucs','ensemble'):
            def differences(metric,stem=None):
                out=[]
                for r in group:
                    values=[]
                    for i in range(4) if stem is None else [stem]:
                        a=r['scores']['candidate']['metrics'][i].get(metric);b=r['scores'][baseline]['metrics'][i].get(metric)
                        if a is not None and b is not None:values.append(a-b)
                    out.append(float(np.mean(values)) if values else None)
                return out
            per={s:paired_stat(differences('si_sdr',i)) for i,s in enumerate(SOURCES)}
            per['overall']=paired_stat(differences('si_sdr'));per['overall_si_sar']=paired_stat(differences('si_sar'))
            per['improved_songs']=sum(v>0 for v in differences('si_sdr') if v is not None)
            gains=[(r['scores'][baseline]['metrics'][2].get('target_gain'),r['scores']['candidate']['metrics'][2].get('target_gain')) for r in group]
            gains=[(a,b) for a,b in gains if a is not None and b is not None]
            per['bass_collapse']={'baseline':sum(a<.1 for a,b in gains),'candidate':sum(b<.1 for a,b in gains),'valid':len(gains),'new':sum(a>=.1 and b<.1 for a,b in gains)}
            result['comparisons'][condition]['candidate-minus-'+baseline]=per
    cfg=config['evaluation'];wet=result['comparisons']['wet_R4']['candidate-minus-initial_rev'];dry=result['comparisons']['dry']['candidate-minus-initial_rev']
    primary=wet['overall'];margin=cfg['retention_ci_lower_margin_db']
    gates={'primary':primary['median'] is not None and primary['median']>=cfg['minimum_gain_db'] and primary['ci'][0]>0,
           'dry_overall':dry['overall']['ci'][0] is not None and dry['overall']['ci'][0]>=margin,
           'wet_si_sar':wet['overall_si_sar']['ci'][0] is not None and wet['overall_si_sar']['ci'][0]>=cfg['wet_si_sar_ci_lower_margin_db'],
           'no_new_bass_collapse':wet['bass_collapse']['new']==0 and dry['bass_collapse']['new']==0}
    for stem in ('vocals','drums','other'):gates['wet_'+stem]=wet[stem]['ci'][0] is not None and wet[stem]['ci'][0]>=margin
    result['gates']=gates;result['passes']=all(gates.values()) and len(rows)==20
    return result


def make_report(root,destination,config):
    status=json.loads((root/'status.json').read_text());summaries=[json.loads((root/'evaluation'/f'step-{s:04d}'/'summary.json').read_text()) for s in config['training']['evaluate_steps']]
    chosen=next(v for v in summaries if v['step']==status['selected_step'])
    rows=[json.loads(p.read_text()) for p in sorted((root/'evaluation'/f"step-{chosen['step']:04d}").glob('*/*/results.json'))]
    if len(rows)!=20:raise ValueError('incomplete selected evaluation')
    output=destination/'결과';output.mkdir(exist_ok=True)
    def fmt(v):return '미정의' if v['median'] is None else f"{v['median']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
    text=['# SCNet dry·wet 미세조정 결과','',f"2026-11-13 · 실제 optimizer갱신{status['step']:,}회 · 500/1,000회 validation 각20조건 · 선택 step{chosen['step']}",'',
          f"사전에 고정한 주요 개선·보호 기준을 모두 {'통과했다' if chosen['passes'] else '통과하지 못했다'}. 선택 후보의 초기 rev 대비 wet 전체 변화는 {fmt(chosen['comparisons']['wet_R4']['candidate-minus-initial_rev']['overall'])}dB다. 추가 학습을 실행했지만, validation 선택 결과를 독립 일반화 입증으로 표현하지 않는다.",'',
          '| 갱신 | wet 전체−초기rev | dry 전체−초기rev | wet vocals−초기rev | 개선·보호 모두 통과 |','|---|---|---|---|---|']
    for v in summaries:
        wet=v['comparisons']['wet_R4']['candidate-minus-initial_rev'];dry=v['comparisons']['dry']['candidate-minus-initial_rev']
        text.append(f"| {v['step']} | {fmt(wet['overall'])} | {fmt(dry['overall'])} | {fmt(wet['vocals'])} | {v['passes']} |")
    text+=['','선택은 사전 규칙(모든 기준 통과 우선, 같은 등급에서는 wet 전체 개선량)으로 정했다. 두 후보 모두 실패하면 선택 checkpoint는 미채택 후보다.','',
           '| 조건 | 선택후보−대조 | 전체 | vocals | drums | bass | other |','|---|---|---|---|---|---|---|']
    for c in ('dry','wet_R4'):
        for name,v in chosen['comparisons'][c].items():
            text.append('| '+c+' | '+name+' | '+fmt(v['overall'])+' | '+' | '.join(fmt(v[s]) for s in SOURCES)+' |')
    text+=['','## 베이스 소실과 보존 기준','',
           '| 조건 | 대조 | bass소실 대조→후보 | 새 소실 |','|---|---|---|---|']
    for c in ('dry','wet_R4'):
        for name,v in chosen['comparisons'][c].items():
            b=v['bass_collapse'];text.append(f"| {c} | {name} | {b['baseline']}/{b['valid']}→{b['candidate']}/{b['valid']} | {b['new']} |")
    text+=['','선택 후보의 각 사전 기준: '+', '.join(f'{k}={v}' for k,v in chosen['gates'].items())+'.','',
           '전체는 곡마다 유효한 음원의 대응 SI-SDR 차이를 평균한 뒤10곡 중앙값·10,000회 곡bootstrap95%CI를 계산했다. 무음 정답은 제외했고 bass는9곡이다. CI 하한으로 dry/vocals/drums/other −0.2dB 및 wet SI-SAR −0.5dB 비열등성을 확인했다. 기존 앙상블보다 전체·vocals·dry가 모두 좋다는 별도 결론으로 확대하지 않는다.','',
           '## 학습과 평가 범위','',
           '공식 rev 분리기의10,578,768개 전체 파라미터를 AdamW lr1e-5로 갱신했다. train80곡에서 매회 같은 곡·같은6초의 dry1/wet1을 microbatch1로 처리하고 gradient를 절반씩 누적했다. 앞뒤1초를 제외한 중앙4초의 complex STFT RMSE와0.05 waveform L1을 사용하고 vocals가중치를2로 했다. train pool의 RIR만 사용했다. 데이터 재생성·소형보정망·teacher학습은 없다.','',
           '평가는 기존 공식SCNet검증의 입력·정답 SHA와 같은40초 입력/중앙30초, dry/wet_R4다. 초기 출력 일치 검사 후 원본 rev부터 시작했다. 모든 checkpoint/출력은 코드·데이터·모델 binding에 연결했다. 500/1,000회에서 신규40 GPU추론을 실행했으며, baseline 값은 동일 정답의 기존 검증 점수를 재사용했다. 새 최적화에 validation/confirmation 음원은 넣지 않았다. 확인용 random benchmark3회 갱신 가중치는 본 학습에 사용하지 않았다.','',
           '공개 분리기가 이 validation곡을 사전 학습했을 수 있으며, 두 checkpoint 선택도 validation에 의존한다. confirmation은 사용하지 않았다. 짧은crop·혼합비·vocals가중치는 우리 추가 설계로, 공식 논문 전체 학습의 재현이나 성공 보장이 아니다. 청감 평가와 새 곡·새 RIR 검증은 남아 있다.','',
           '[사전 계획](../PLAN.md) · [고정 설정](../protocol.json) · [직전 공식 가중치 결과](../../공식SCNet검증_2026-11-13/결과/결과보고서.md)']
    (output/'결과보고서.md').write_text('\n'.join(text)+'\n')
    value={'complete':True,'training':status,'selected':chosen,'checkpoints':summaries}
    (output/'분석요약.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    data=[]
    for v in summaries:
        for path in sorted((root/'evaluation'/f"step-{v['step']:04d}").glob('*/*/results.json')):
            row=json.loads(path.read_text())
            for model,scores in row['scores'].items():
                for m in scores['metrics']:data.append({'step':v['step'],'track':row['track'],'condition':row['condition'],'model':model,**m})
    with (output/'곡별결과.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['step','track','condition','model','stem','state','si_sdr','si_sir','si_sar','sd_sdr','target_gain','target_rms','output_rms','rank']
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
    readme=destination/'README.md';body=readme.read_text();start=body.index('<!-- RESULT START -->');end=body.index('<!-- RESULT END -->')
    intro=f"실제1,000회 학습과500/1,000회 각20조건 평가를 완료했다. 선택 step{chosen['step']}, 초기 rev 대비 wet 전체 {fmt(chosen['comparisons']['wet_R4']['candidate-minus-initial_rev']['overall'])}dB, dry 전체 {fmt(chosen['comparisons']['dry']['candidate-minus-initial_rev']['overall'])}dB다. 사전 개선·보호 기준은 {'통과' if chosen['passes'] else '실패'}이며, 공개 사전 학습과 validation 선택에 의한 한계를 유지한다.\n\n[상세 결과](결과/결과보고서.md) · [곡별 CSV](결과/곡별결과.csv)\n"
    body=body[:start]+'<!-- RESULT START -->\n'+intro+body[end:];readme.write_text(body)
    for file in (readme,output/'결과보고서.md'):
        assert len(file.read_text())>500 and '1,000' in file.read_text()

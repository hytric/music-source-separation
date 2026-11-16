# 공식 SCNet 비교와 전체 미세조정

공식clean/rev를같은val20에서비교했다. rev−clean wet+4.397dB/dry−1.396dB다. M2에서전체10578768개파라미터를1000회미세조정한결과초기rev대비wet+.509dB/dry+.182dB였다. B=HTD+BSR대비wet+1.531dB이나dry−2.154dB로전입력교체는부적격이다.

## 범위와 판정

공식repo commit600585bc 및step1000을구분한다. OfficialSCNet rev만생성하면미세조정전M2이므로반드시checkpoint[model] strict적재가필요하다. 두모델의학습률등이달라RIR단독효과라고확정하지않는다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/scnet_finetune) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

[공식 checkpoint 비교 코드/결과](../../../experiments/parirset_validation)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

격리 workspace의 단계 실행은 `python tools/reproduce.py stage --workspace "$WORK" --name <단계>`다. 이 실험의 단계는 **official → finetune**이며 선행 cache를 먼저 만든다. optimizer가갱신되는train-refiner/finetune은`--allow-optimizer`를명시해야한다. 결과는workspace/code에기록되고Git에있는원래결과는유지된다. 전체신규GPU재실험을이번문서작업에서수행한것은아니다.

[전체 실험 문서 허브](../README.md)

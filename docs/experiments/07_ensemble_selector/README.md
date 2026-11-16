# 음원별 결합과 실제 입력 선택

H의vocals/other는B,drums/bass는F1000이다. 항상H는B대비wet+1.577/dry−1.253dB로실패했다. train80의입력특징33개로적합한G0는threshold.8에서wet+1.460 [0.080,2.062]dB,drymedian0이지만StVitus−.310/Titanium−2.202손상이남았다.

## 범위와 판정

조건라벨분기와GT oracle는실제입력선택기가아니다. G0의학습라벨도실제성능이득이아닌추가RIR유무였다. C/Q품질이득선택기는아직학습/평가하지않았다. 전체합불일치와개별실패로배포는보류다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/input_selector_validation) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

[고정 결합·oracle 대조](../../../experiments/stem_hybrid_validation)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

격리 workspace의 단계 실행은 `python tools/reproduce.py stage --workspace "$WORK" --name <단계>`다. 이 실험의 단계는 **hybrid → selector**이며 선행 cache를 먼저 만든다. optimizer가갱신되는train-refiner/finetune은`--allow-optimizer`를명시해야한다. 결과는workspace/code에기록되고Git에있는원래결과는유지된다. 전체신규GPU재실험을이번문서작업에서수행한것은아니다.

[전체 실험 문서 허브](../README.md)

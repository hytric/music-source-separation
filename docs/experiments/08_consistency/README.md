# 보호 음원 고정 출력합 보정

선택기와20개선택을고정해drums/bass에만잔차를균등/에너지가중배분했다. 전체40초합오차2.98e−8/VO차이0이지만power의미보정대응wet−.285/bass−.924dB로품질보존실패했다. B자체의보정대조도포함한다.

## 범위와 판정

Matthew의4stem평균−2.808dB는near-silent bass SI-SDR에크게좌우되며청감악화의직접척도가아니다. 기존지표판정은유지하고RMS/사후활동도민감도를별도로본다. 합일치는분리품질의충분조건이아니다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/db_consistency_validation) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

격리 workspace의 단계 실행은 `python tools/reproduce.py stage --workspace "$WORK" --name <단계>`다. 이 실험의 단계는 **consistency**이며 선행 cache를 먼저 만든다. optimizer가갱신되는train-refiner/finetune은`--allow-optimizer`를명시해야한다. 결과는workspace/code에기록되고Git에있는원래결과는유지된다. 전체신규GPU재실험을이번문서작업에서수행한것은아니다.

[전체 실험 문서 허브](../README.md)

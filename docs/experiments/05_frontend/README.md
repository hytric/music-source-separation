# 입력 EQ·WPE와 잔차 배분

내부val10의dry/wet20조건,40초입력/중앙30초다. HTD대비WPE단독wet−.498dB,WPE+weighted+1.185dB [−.127,2.637],dry−.139dB로사전성공기준은실패했다. raw weighted와음량대조도함께비교했다.

## 범위와 판정

NARA-WPE0.0.11, taps10/delay3/3iterations가고정됐다. 원래wet정답을유지하므로음성direct-target 잔향제거성공을그대로이전할수없다. 마스킹·EQ·잔차배분은각각다른출력이다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/frontend_validation) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

격리 workspace의 단계 실행은 `python tools/reproduce.py stage --workspace "$WORK" --name <단계>`다. 이 실험의 단계는 **frontend**이며 선행 cache를 먼저 만든다. optimizer가갱신되는train-refiner/finetune은`--allow-optimizer`를명시해야한다. 결과는workspace/code에기록되고Git에있는원래결과는유지된다. 전체신규GPU재실험을이번문서작업에서수행한것은아니다.

[전체 실험 문서 허브](../README.md)

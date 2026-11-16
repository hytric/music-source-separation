# 소형 보정망 준비와 학습

공식train100의내부80/10/10분할,325024파라미터복소잔차보정망을3000회학습했고best는2500이다. 고정4초val40조건에서best의HTD대비+.0461dB,초기projection대비record대응+.0179dB로충분한효과를확인하지못했다.

## 범위와 판정

초기readiness optimizer0은현재학습step이아니다. 확인평가는미실행이다. 30초cache는400조건이며새40초teacher로padding해재사용하면안된다. 새학습재현은기존best.pt재적재와다른의미다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/banelp_refine) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

[당시 상세 보고](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/02_검수와피드백.md)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

격리 workspace의 단계 실행은 `python tools/reproduce.py stage --workspace "$WORK" --name <단계>`다. 이 실험의 단계는 **cache-refiner → preflight-refiner → train-refiner**이며 선행 cache를 먼저 만든다. optimizer가갱신되는train-refiner/finetune은`--allow-optimizer`를명시해야한다. 결과는workspace/code에기록되고Git에있는원래결과는유지된다. 전체신규GPU재실험을이번문서작업에서수행한것은아니다.

[전체 실험 문서 허브](../README.md)

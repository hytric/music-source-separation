# QG1 구현과 teacher 데이터 생성

64곡fit/16곡교정의품질라벨선택기명세와첫구현을정리했다. 2026-11-15 GPU단계는train320조건×3teacher=960예측을완료했다. 이는teacher cache준비이며C/Q적합/교정/validation성과를뜻하지않는다.

## 범위와 판정

기존m1의teacher0/smoke24표시는당시snapshot이다. 전체모델학습/평가는아직미완료다. prototype에preflight/prepare/cache가있어도labels부터최종보고서CLI까지완성된것이아니다. 다른GPU작업중단이나precision묵시변경을허용하지않는다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/quality_gate_v1) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

초기30초기준선/190초원인대조/teacher960은서로다른프로토콜이다. 최신40초case명령을그대로쓰면동일실험이되지않는다. 아래원문코드·설정과전용재실행안내를사용한다.

[전체 실험 문서 허브](../README.md)

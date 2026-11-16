# 기준선과 개발 후보 비교

공식test50을dev10/eval40으로나눠음원분리후보를비교했다. 입력30초,음원별RIR를적용한wet정답,기준선HTDemucs다. 선정된HTD+BSR50:50은eval40×4RIR160조건의중앙값+1.652dB [1.518,2.000], bass저투영소실31/160→12/160이었다. 이평가곡도기존오류분석에노출됐으므로완전한새곡검증은아니다.

## 범위와 판정

T1/T2/T5, MISI, Wiener, donor 재분리, 5개모델대체/앙상블의개발21후보를포함한다. Wiener2는무효관측이고,개발Holm검정과효과크기조건은별개다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/banelp) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

[당시 상세 보고](../../reverberant-music-source-separation-analysis-2026-11-10/01_진행현황과_검수.md)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

초기30초기준선/190초원인대조/teacher960은서로다른프로토콜이다. 최신40초case명령을그대로쓰면동일실험이되지않는다. 아래원문코드·설정과전용재실행안내를사용한다.

[전체 실험 문서 허브](../README.md)

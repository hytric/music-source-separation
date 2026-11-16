# 잔향 영향과 성분별 원인 진단

test50에서추가RIR없는조건과wet200조건을비교했다. R4의bass변화는−14.51dB였다. dev10의delay/early/full대조에서bass는+.020/−4.326/−6.399dB의대응차이를보였다. 이는개선후처리성과가아니라오류발생조건의진단이다.

## 범위와 판정

모든조건의GT가같은것은아니다. A는원래GT,B/C는원래wetGT다. 같은wet정답의분리순서대조와조건별생성GT를구분한다. early절단은RT60하나만변경한인과실험이아니다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/banelp) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

[당시 상세 보고](../../reverberant-music-source-separation-analysis-2026-11-10/08_잔향성분_대조결과.md)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

초기30초기준선/190초원인대조/teacher960은서로다른프로토콜이다. 최신40초case명령을그대로쓰면동일실험이되지않는다. 아래원문코드·설정과전용재실행안내를사용한다.

[전체 실험 문서 허브](../README.md)

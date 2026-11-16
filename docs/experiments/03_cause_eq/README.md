# 크기·위상과 EQ 원인 대조

140GPU추론완료. 모델입력190.22초/중앙30초이며DSP grid760.87초다. bass E_M−7.557/E_A−7.159dB,M−A CI는0을포함해크기우세를확정하지못했다. known commonRIR EQ는bass+2.197dB였지만blind source별RIR EQ는−.668dB로유효성을확인하지못했다.

## 범위와 판정

95초입력/380초DSP2회와190초입력/380초DSP14회pilot은무효로분리한다. 원래30초/40초조건의새실험과이결과를합산하지않는다. 실제응답을아는oracle EQ와입력만사용하는EQ는다르다.

## 코드와 원자료

[실험 코드·설정·기존 결과](../../../experiments/banelp_refine) · [기준선/입력길이별 전체 성능표](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/05_실측성능표.md)

[당시 상세 보고](../../reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/01_전체실험목록.md)

## 다른 환경에서 재현

[공통 환경·번들 이전·실행 절차](../10_reproduction/README.md)를 먼저 수행한다. `python tools/reproduce.py results --output /새/results.json`은 저장된 지표를 재집계하며 새 추론이나 학습이 아니다. [검증 범위표](../10_reproduction/검증범위.md)에서 이 실험의 재실행 수준을 확인한다.

초기30초기준선/190초원인대조/teacher960은서로다른프로토콜이다. 최신40초case명령을그대로쓰면동일실험이되지않는다. 아래원문코드·설정과전용재실행안내를사용한다.

[전체 실험 문서 허브](../README.md)

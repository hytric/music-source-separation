# QG1 첫 구현 마일스톤

설계안에 따라 품질 이득 선택기의 구현을 시작했다. `core.py`에 곡 단위64/16분할·품질라벨·동일 sample weight를 받는logistic적합·안전threshold선택을구현했고, `run.py`에preflight/40초입력준비/모델별cache실행경로를추가했다.

**현재 검증 범위:** 단위테스트4개 통과,train8/320조건의실제40초입력합성과기존30초정답core동등성·파일SHA·재개확인. GPU는가용2471MiB로최초4096MiB조건을충족하지않아teacher미적재/출력0이다. 전체labels/features·모델적합·교정·동결·validation평가/보고서CLI는다음구현단계다. 새C/Q성능결과는없다.

[구현·실험 타임라인](타임라인.md) · [검증 기록](검증기록.json) · [설계 가이드](../../docs/reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/구현가이드/README.md)

기존실험/모델을수정하지않고새runtime `quality-gate-v1-m1`을사용했다. source변경후재개는binding검사로거절되며새version/import가필요하다. 현재상태를전체실험complete로표시하지않는다.

```bash
# 실제 현재 명령. 프로젝트/런타임 경로를 지정한다.
OPENBLAS_NUM_THREADS=2 python run.py preflight --project /seperation --runtime /새/runtime
OPENBLAS_NUM_THREADS=2 python run.py prepare --project /seperation --runtime /새/runtime --limit 8
OPENBLAS_NUM_THREADS=2 python run.py cache --project /seperation --runtime /새/runtime --model msst_bsroformer_ep17 --limit 8
```

Python은가이드의기존banelp-v1환경을사용한다. 30초예측을40초teacher로대체하지않으며train조건에는각record의trainRIR를사용한다. 소스/원자료/모델manifest를검증하고하나라도다르면중단한다. `cache`는한프로세스한모델이며실제GPU출력경로는아직이장비에서검증하지않았다. 누락산출물을0으로채우거나실제실행전성능수치를보고하지않는다.

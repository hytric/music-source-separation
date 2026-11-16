# 로컬 GPU 원인 실험 실행 상태

> 2026-11-14 검수: 아래는 당시 실행 snapshot이다. 최신 상태는 소형 refiner3,000회 완료(best2,500), 원인EQ140회 완료다. 준비 optimizer0이나 과거중단문구를 현재진행상태로읽지않는다. [종합검수](../../docs/reverberant-music-source-separation-analysis-2026-11-10/종합검수_2026-11-14/README.md)에서 최종 수치와 미실행 confirmation을 확인한다.

확인 시각 2026-11-12T10:16:37+09:00 · 상태 pilot-running

GTX 1660에서 고정 HTDemucs를 실행하는 원인 대조 실험을 시작했다. 먼저 14개 GPU 추론과 EQ 왕복·기존 지표 대조를 통과했지만, 전체 집합의 다음 곡에서 위상 조건의 경계 오차가 1.69e-4로 기준 1e-4를 넘어서 해당 run을 중단했다. 이 결과로 원인 가설의 성공을 주장하지 않는다.

현재는 DSP grid를 2^25(760.87초)로 확장한 별도 run으로 재시작했다. 모델 입력은 190.22초, 중앙 평가 구간은 같은 30초다. EQ 강도와 수치/효과 판정 기준은 바꾸지 않았다. 이전 14개 출력은 이전 run에 보존하고 새 집계에 합치지 않는다. 추가 DSP 단위 테스트 6개가 통과했다.

새 run은 한 곡 pilot을 먼저 검사하고 통과하면 핵심 80개·원래 조건 확인 60개, 총 140개 GPU 추론을 이어간다. 실패 시 자동으로 기준을 완화하거나 재시작하지 않고 부분 결과와 실패 상태를 남긴다. 보정망 optimizer 갱신은 0회다. GPU는 분리 추론에, CPU는 FFT·정렬·지표 검사에 사용하므로 사용률은 구간에 따라 달라진다.

## 실행과 결과 위치

- OS 작업: `banelp-cause-eq-20261112.service`, 확인 시 active/running.
- 실제 상태: 새 runtime의 `job_status.json`과 `latest.json`.
- 현재 run: `run-c2ea006a17708652`.
- 정상 완료 또는 처리 가능한 실행 실패 시 로컬 `원인대조결과_2026-11-12/`에 보고서·집계 JSON·CSV를 저장한다.
- 이 페이지는 실행 시점 기록이며 전체 결과가 아니다. 채팅 자동 완료 알림은 연결하지 않았다.

```bash
systemctl --user status banelp-cause-eq-20261112.service
journalctl --user -u banelp-cause-eq-20261112.service -n 30 --no-pager
```

[설계와 수치 변경](CAUSE_EXPERIMENT.md) · [실행 코드 안내](README.md)

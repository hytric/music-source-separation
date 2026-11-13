# 입력 전처리 재검증

2026년 11월 13일 · 20조건·80 GPU 추론 완료

입력 전처리가 음악 분리를 돕는지 고정 HTDemucs로 검증한다. 무보정·EQ·WPE와 잔차 배분 대조를 같은 입력 길이·원래 잔향 정답으로 비교한다. 주요 후보는 WPE+weighted이며, 조건을 실행 결과에 맞춰 선택하지 않는다.

공식 train에서 이미 분리한 validation10곡의 dry/wet20조건을 사용한다. 가중치 학습이나 최종 confirmation 사용은 없다. 별도 ASR GPU 작업은 중단하지 않고 단일 모델 추론을 순차 수행한다. NARA WPE0.0.11과click8.5.1은별도runtime/vendor에만설치했다.

- [실험 계획](PLAN.md)
- [고정 설정](protocol.json)
- 실행 코드: run.py
- 숫자 검증: tests.py

STFT·정렬·무음·출력 합·WPE 유한값·paired bootstrap 단위 테스트5개가통과했다. 최초1곡pilot의신호검사후전체시작전에음량만맞춘gain_weighted대조를추가해최종80GPU호출로고정했다. 초기6호출pilot과최종run은다른binding이며결과를섞지않는다.

```bash
python run.py --runtime /새/frontend-verify-v1 \
  --suite /학습실험_2026-11-11 --baseline-repo /music-source-separation \
  --baseline-runtime /기존/banelp-v1 --prepared-runtime /준비된/banelp-refine-v1 --limit 10
```

runtime/vendor에 nara_wpe==0.0.11과click==8.5.1이필요하다.기존torch/numpy/scipy/soundfile환경을그대로사용하고실제버전과원문구현hash는각run/binding.json에기록한다. 원자료·가중치·vendor·파형출력은Git에넣지않는다. 결과는 결과/폴더에 저장했다. 주요 후보의 성공 기준은 통과하지 못했으며, 개선 경향과 dry 손상을 함께 보고한다.

## 실제 검증 결과

[상세 결과](결과/결과보고서.md) · [집계 JSON](결과/분석요약.json) · [곡별 CSV](결과/곡별결과.csv)

WPE만의 wet 전체 변화는 −0.498 dB였다. WPE+weighted는 +1.185 dB [−0.127, +2.637], 베이스 소실은 3/9→2/9로 개선 경향이 있었지만 CI가 0을 포함해 성공 기준을 통과하지 못했다. dry 전체는 −0.139 dB로 악화됐다. 입력 전처리를 무조건 적용할 근거는 없고, 잔향 정도에 따른 우회와 실패곡·음원 손상 분석이 다음 과제다. 현재 자료의 탐색적 결과이며 새 곡 일반화와 학습 보정망의 성능으로 확대하지 않는다.

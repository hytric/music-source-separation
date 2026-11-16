# SCNet 분리기 자체 미세조정 계획

2026-11-13 · 학습·후보 점수 관찰 전 고정

직전 공식 SCNet 비교에서 rev는 wet에서 clean 대비 +4.397dB, HTDemucs 대비+3.198dB였지만 dry에서 clean 대비−1.396dB, wet vocals에서 기존 앙상블 대비−0.917dB였다. 이 결과를 출발점으로 분리기 전체를 dry·wet 혼합으로 추가 학습한다. 소형 보정망은 사용하지 않는다. 이번 혼합 비율과 vocals 가중치는 우리의 검증할 설계이며 논문이 보장한 성공 설정이라고 표현하지 않는다.

공식 PaRIRset commit600585bc0d50cb48dd64e4602e3c34fe48be7347의 rev 가중치에서 10,578,768개 전체 파라미터를 갱신한다. 기존 train80곡 cache만 사용한다. 곡마다 dry1/wet3조건이 있으며 각 갱신에서 같은 곡·같은 시간의 dry와 wet 하나를 순서대로 처리해 gradient를 절반씩 누적한다. 학습 RIR은 기존 train pool만 사용한다. validation 및 confirmation의 곡/RIR은 추가 최적화에 들어가지 않는다. 원래 wet 음원을 정답으로 유지한다.

6초 입력·앞뒤1초 guard·중앙4초 손실, FP32/microbatch1, AdamW lr1e-5/weight_decay1e-4, gradient clip5, 1,000회 갱신이다. 공식 complex STFT RMSE에 waveform L1 0.05를 더하고 vocals:drums:bass:other=2:1:1:1로 고정한다. sqrt의1e-8은 완전 일치 때 유한 gradient를 유지한다. 별도 random 입력 benchmark에서3회 갱신과 상태 변화·유한 gradient를 확인했고 최대 예약3.23GiB였다. 이 가중치는 폐기하고 본 학습은 공식 rev 원본부터 다시 시작한다.

500/1,000회에서 각각 validation10곡 dry/wet20조건을 평가한다. 입력은 기존 공식SCNet검증과 같은40초·중앙30초, 같은 정답과 입력 SHA를 사용한다. 비교는 초기 rev, clean, HTDemucs, 기존 50:50 앙상블과 수행한다. 첫 초기 출력은 기존 rev cache와 숫자를 대조한다. checkpoint/입력/출력/코드를 binding으로 연결한다. optimizer/RNG 상태를50회마다 원자적으로 저장하고 재개 시 엄격히 검증한다. 실패는 완료로 표시하지 않는다.

주 검정은 초기 rev 대비 wet 전체 +0.2dB 이상·곡 bootstrap95%CI하한>0이다. 보존 기준은 초기 rev 대비 dry 전체 및 wet vocals/drums/other의 CI하한≥−0.2dB, wet 전체 SI-SAR CI하한≥−0.5dB, 새 bass소실0이다. 통과 후보를 우선하고 같은 등급에서는 wet 전체 중앙값이 큰 후보를 선택한다. 모두 실패하면 미채택 후보로 보고한다. 기존 앙상블 대비 dry/vocals 격차와 실패곡은 별도로 보고하며, 학습곡·checkpoint 선택으로 생긴 낙관성을 독립 검증으로 바꾸어 말하지 않는다. 최종 배포 전환과 confirmation 사용은 이번 범위에 포함하지 않는다.

[직전 검증 결과](../parirset_validation/결과/결과보고서.md)의 구성·수치와 cache를 가져오고, 이번에는 실제 추가 optimizer 갱신의 효과를 새로 측정한다. [공식 train_clean.py](https://github.com/enricguso/guso_ismir26/blob/600585bc0d50cb48dd64e4602e3c34fe48be7347/scnet/train_clean.py)의 wet target 학습과 [loss.py](https://github.com/enricguso/guso_ismir26/blob/600585bc0d50cb48dd64e4602e3c34fe48be7347/scnet/loss.py)의 complex spectral loss를 참고했다. 짧은 crop·dry/wet pairing·vocals 가중치는 이번 실험의 변경점이다. 공개 모델의 사전 학습 중복 가능성 때문에 새 곡 일반화 입증은 아니다.

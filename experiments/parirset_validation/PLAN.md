# 공식 SCNet 비교 계획

2026-11-13 · 점수 관찰 전에 고정

PaRIRset 공식 M1_clean와 M2_reverberant를 같은 validation10곡의 dry/wet20조건에서 비교한다. 주요 검정은 wet의 곡별 4음원 평균 SI-SDR에서 rev−clean 차이의 중앙값이다. +0.2dB 이상이면서 10,000회 곡 bootstrap 95% CI 하한이 0보다 큰지 확인한다. dry 손상·음원별 손상·bass target_gain<0.1 소실도 함께 공개한다. 주 검정 성공만으로 우리 HTDemucs 기준선 대체를 선언하지 않고 rev−HTDemucs 및 rev−기존 50:50 앙상블도 평가한다.

입력과 정답 합성은 기존 입력전처리검증_2026-11-13/run.py의 dry_sources/targets를 그대로 사용한다. 40초 입력의 중앙30초만 평가하며 wet의 원래 잔향 정답을 유지한다. 저장된 HTDemucs는 원본 코드·모델·manifest·출력 SHA를 검증하고, 재구성 정답에 대한 점수 일치까지 확인해 재사용한다. 이전 cache에는 입력 해시가 없으므로 과거 입력 해시를 직접 대조했다고 표현하지 않는다. 이번부터 재구성 입력·정답 배열 SHA를 저장한다. 각 출력을 같은 입력 해시에 연결한다.

공식 commit600585bc0d50cb48dd64e4602e3c34fe48be7347의 원문 구조와 apply_model을 직접 불러오고 state_dict를 strict=True로 적재한다. CLI의 잘못된 인자 참조는 우회하고 tensor_chunk의 누락된 th 별칭만 연결한다. 원문의 사각 STFT 창을 보존한다. 원문과 같은 mono평균/std 정규화, 원문 삼각창 overlap-add, 원래 음원 순서 drums/bass/other/vocals를 vocals/drums/bass/other로 재배열한다. 20초 chunk·50% overlap·batch1·FP32·shifts0을 두 모델에 동일 적용한다. shifts0은 원문 CLI 기본1과 다르며 무작위 시간 이동을 제외한다. PyTorch2.6.0 환경은 원문 요구2.8과 달라 strict 적재 및 실제 CUDA forward로 호환을 확인한다. 점수 이전 5초/20초 random smoke는 성공했고 최대 할당 약1.18GB였다.

BS-RoFormer는 기존 등록 설정을 쓰되 공유 GPU를 위해 batch1로 고정한다. 모델을 한 번에 하나씩 올리고 다른 ASR를 중단하지 않는다. batch1도 실패하면 실패를 기록하고 평가를 완료한 것처럼 보고하지 않는다. 새 학습·confirmation 사용은 없다.

[공식 구현](https://github.com/enricguso/guso_ismir26/tree/600585bc0d50cb48dd64e4602e3c34fe48be7347)의 구조·가중치를 사용한 로컬 재평가다. 공식 모델의 사전 학습에 우리 validation곡이 포함됐을 수 있으므로 일반화 성능이나 순수 RIR 단일 요인 인과 검증으로 해석하지 않는다. clean/rev 설정의 학습률도 다르다. 논문 표의 noisy-reverberant SDR와 이 실험의 pure-reverb SI-SDR를 합치지 않는다.

# 잔향 음악 공동 보정 학습 실험 기획

2026년 11월 11일 · 구현 기준 `configs/pilot.json`

## 목적과 가설

고정 HTDemucs의 네 출력에 흩어진 음원 단서를 함께 사용하면 잔향 입력에서 발생한 베이스·드럼 손실과 음원 간 누출을 줄일 수 있는지 확인한다. 현재 코드의 출력과 정답은 모두 잔향이 포함된 stem이다. 목표는 잔향을 보존한 분리이며, 잔향을 제거한 직접음 복원이 아니다.

기존 50곡 사후 진단에서는 가장 긴 잔향 조건의 원본 대비 SI-SDR 대응 차이가 베이스 −14.51 dB·드럼 −8.88 dB였고, 소실 사례의 베이스 투영 성분은 other 출력에서 증가했다. 개발 10곡 성분 대조에서는 지연만의 베이스 변화는 +0.020 dB, 초기 R4−지연은 −4.326 dB, 전체 R4−초기는 −6.399 dB였다. 이 관찰을 근거로 혼합과 모든 출력, 시간 문맥, 복소 보정을 사용한다. 이 수치는 새 보정망의 성능이 아니다.

## 비교할 구성

| 구성 | 목적 |
|---|---|
| 고정 HTDemucs 원 출력 | 분리 기준선 |
| 출력 합 projection만 적용 | 학습 없이 합을 맞추는 효과 분리 |
| 소형 공동 보정망과 기본 손실 | 첫 학습 대상 |
| 공동 보정망과 확장 손실 | 위상·저역 감독의 추가 가치 |
| HTDemucs+BS-RoFormer 50:50 | 최종 실용 품질 비교. 평가 명령의 `--include-ensemble`로 선택 |

기본과 확장 손실은 별도 config/runtime으로 고정한다. 모델·손실을 평가 결과에 맞춰 바꾼 뒤 같은 confirmation을 새 검증처럼 반복 사용하지 않는다. 혼합만 입력하거나 한 음원만 보는 ablation은 본 pilot의 결과 이후 별도 실험으로 구현·검증한다. 이번 버전이 이 모든 ablation을 구현했다고 주장하지 않는다.

## 데이터와 공간 분할

기존 ZIP의 공식 train 100곡을 전체 WAV 추출·CRC·SHA256·오디오 디코딩으로 검증한다. 곡 이름을 NFC 정규화한 뒤 `training-design-v1` salt와 SHA256으로 정렬하고 학습 80곡·validation 10곡·confirmation 10곡을 고정한다. 이는 보정망을 위한 분할이다. 기초 분리기는 이 곡들을 과거 학습에 사용했을 수 있다. 기존 test 50곡은 이번 학습과 후보 선택에 쓰지 않는다.

RIR은 별도 경로에서 gpuRIR로 생성한다. 학습 16개 방·64개 설정, validation 4개 방·16개 설정, confirmation 4개 방·16개 설정으로 나눈다. 한 설정은 네 음원 위치의 RIR 묶음이며 총 96묶음·384개 파일이다. 같은 방의 배치·흡수 설정은 같은 분할에 둔다. 목표 T60과 측정값, geometry, GPU 확장 모듈 hash를 함께 보존한다.

제안된 방·잔향시간 범위에서 샘플링하되, 유효한 흡수계수와 image 수 예산 5,000,000을 만족하는 제안만 생성한다. 이는 자원 예산에 조건부인 표본이며 무조건 균등한 공간 표본이라고 표현하지 않는다. 거절한 제안은 manifest에 남긴다. GPU 실패 시 다른 생성기로 조용히 대체하지 않는다.

각 곡에 추가 잔향 없음 1조건·분할에 맞는 서로 다른 전체 RIR 3조건을 적용한다. 총 학습 320·validation 40·confirmation 40개, 전체 400개 30초 record다. 같은 음원별 mono RIR을 좌우에 적용하고 원 stem RMS를 복원한 뒤 합산한다. 기존 crop→convolution→끝 절단 절차를 유지한다.

원곡이 30초보다 짧으면 중앙을 맞춰 양쪽에 0을 채운다. 반복 재생하거나 곡을 제외하지 않는다. 패딩 위치는 manifest에 저장하며 학습 crop은 실제 신호 구간 안에서 선택한다. 실제 신호가 4초보다 짧은 경우에만 이를 포함하는 중앙 4초를 사용한다. 초기 계획의 30초 가정에 추가한 명시적 처리다.

## 모델과 학습 목표

혼합과 네 추정치를 STFT로 변환해 실수·허수·log magnitude 30채널 특징을 구성한다. 24/48/96채널 encoder, 시간 dilation 1/2/4/8/16의 bottleneck, skip decoder를 사용한다. 마지막 head는 네 음원×두 채널×실수·허수의 16채널을 출력한다. 총 파라미터는 325,024개다.

먼저 네 기존 출력의 합을 혼합과 맞춘다. 보정망이 예측한 네 복소 잔차에서는 평균을 빼 합을 0으로 만든다. 이를 기존 출력에 더하고 파형으로 복원한다. 0으로 초기화한 head의 결과는 원 HTDemucs가 아니라 projection 기준선과 같다. STFT는 FFT 2048·hop 512·periodic Hann·center=true·constant padding이며 길이를 명시해 복원한다. 보정 스펙트럼에는 sqrt(2048) 배율을 고정해 사용한다.

모든 입력·정답은 혼합에서 구한 같은 RMS scale을 사용한다. 기본 손실은 stem별 안정화 파형 L1 + 0.5×다중 해상도 스펙트럼 손실이다. 확장 손실은 복소 L1과 네 음원 모두의 20~1000 Hz 복소 오차를 각각 0.25 가중치로 추가한다. 무음 정답에는 불안정한 로그 비율 대신 출력 에너지 억제 항을 사용한다.

학습 설정은 44.1 kHz stereo·4초 crop·batch 1·FP32·AdamW lr 1e-4·weight decay 1e-4·gradient clip 5다. pilot 예산은 최대 3,000 step이다. 처음에는 `--stop-after 100`으로 동일 실험의 앞 100 step만 확인할 수 있다. CosineAnnealing의 전체 기간은 항상 고정된 3,000 step이며 100 step 중단으로 바뀌지 않는다.

## 검증과 후보 고정

학습 중 validation은 고정 중앙 4초 구간의 4-stem 평균 ΔSI-SDR 중앙값으로 best checkpoint를 고른다. 원 HTDemucs와 비교하며 정확한 무음 정답 등 정의되지 않은 stem은 사유를 유지한다. 이 값은 최종 30초 평가 결과와 구분한다. 500 step마다 또는 실행 예산 끝에서 validation하며, 5회 연속 개선이 없으면 중단한다.

최종 confirmation은 선택한 checkpoint를 hash로 고정하고 30초 전체를 같은 4초 공통-window overlap-add로 추론한다. 기존 `project_metrics`를 그대로 사용해 SI-SDR·SI-SIR·SI-SAR·SD-SDR·target gain을 저장한다. 주 효과는 조건별 네 음원 평균 ΔSI-SDR의 중앙값이며, 10,000회 곡 bootstrap은 같은 곡의 네 조건을 함께 재표집한다. 방 모집단의 변동까지 포함한 CI는 아니다.

현재 evaluator는 수치 원자료와 주 효과/CI를 산출한다. 베이스 소실 전이·음원별 비열등성·청취 판단의 종합 채택은 저장된 결과를 이용한 후속 검수다. 그 검수가 끝나기 전 자동으로 최종 성공 판정을 내리지 않는다.

프로젝트 목표는 HTDemucs 대비 ≥+0.2 dB와 CI 하한 >0, 베이스 소실 감소, 다른 음원·추가 잔향 없는 조건의 손상 억제다. 이는 제안 임계값이다. 기존 두 모델 평균보다 품질이 높은지와, 비슷한 품질을 더 낮은 계산 비용으로 내는지는 별도로 평가한다. 내부 confirmation도 기초 분리기의 학습 중복 한계가 있어, 완전한 새 곡·공간 일반화는 새 녹음으로 추가 검증한다.

## 학습 시작 전의 정지점

`prepare`는 원자료·RIR·cache 생성 후 preflight에서 끝난다. preflight는 모든 파일 hash와 데이터 합을 확인하고, 실제 학습 자료로 forward·loss·backward를 수행한다. optimizer는 생성하되 step을 호출하지 않는다. 실제 validation 40개도 업데이트 없이 읽고, 모델 hash 불변과 optimizer state가 비어 있음을 확인한다. step 0 checkpoint를 다시 읽어 검증한 뒤 readiness를 기록한다.

따라서 preflight의 gradient와 loss가 정상이라는 사실은 성능 개선의 증거가 아니다. 학습은 별도 `train --start-training` 명령에서만 수행한다. config·코드·입력 manifest·환경·초기 checkpoint가 바뀌면 기존 readiness로 학습을 시작할 수 없다.

## 참고한 선행연구

- [PaRIRset 2026](https://arxiv.org/html/2607.27828v1)의 RIR 증강과 잔향 보존 정답을 참고한다. 전체 SCNet 재학습을 그대로 재현한 실험은 아니다.
- [MIMO Model Extension 2026](https://arxiv.org/html/2609.07226v1)의 공동 음원 입력과 mixture projection을 참고한다. 첫 모델은 한 번만 잔차를 보정한다.
- [MSG 2022](https://archives.ismir.net/ismir2022/paper/000093.pdf)는 고정 음악 분리기 뒤 학습형 보정의 선례다. 현재 구현은 GAN을 사용하지 않는다.
- [PyTorch STFT](https://docs.pytorch.org/docs/stable/generated/torch.stft.html)와 [ISTFT](https://docs.pytorch.org/docs/stable/generated/torch.istft.html)의 창·center·length 규칙을 따른다. 실제 검증 환경은 torch 2.6.0+cu126이다.

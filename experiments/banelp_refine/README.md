# 잔향 음악 공동 보정 학습 준비

고정 HTDemucs의 네 출력과 혼합을 함께 보는 325,024 파라미터의 복소 잔차 보정망이다. 베이스·드럼 손실과 음원 간 누출을 잔향이 포함된 정답으로 감독한다. [실험 기획](EXPERIMENT.md)과 [실행 설정](configs/pilot.json)에 분할·손실·후보 선택·해석 범위를 명시했다.

**현재 컴퓨터에서 준비 완료를 확인했다.** train 100곡·500 WAV 검증, RIR 96묶음, HTDemucs cache 400개, 실제 GPU forward/backward·초기 validation 40개를 완료했다. 최대 reserved 메모리는 2,006 MiB였고 가중치 갱신은 0회다.

`prepare`와 `preflight`는 **학습 직전에서 멈춘다.** optimizer.step을 호출하지 않고 초기 checkpoint를 저장한다. 학습 시작에는 별도의 `train --start-training`이 필요하다. 이 저장소에는 음원·모델·cache·학습 환경을 포함하지 않는다.

## 원인 검증을 위한 추가 실험

[주파수 착색 원인 검증과 전후 EQ 대조](CAUSE_EXPERIMENT.md)를 별도 설계했다. 주파수 크기와 위상 변화를 나눠 어떤 조건에서 베이스 오류가 생기는지 확인하고, 분리 전 EQ와 분리 후 역EQ가 같은 잔향 정답에서 오류를 줄이는지 비교한다. 알려진 RIR을 쓰는 원인 진단과 입력만 쓰는 적용 확인을 구분한다. 현재는 설계 단계이며 새 추론이나 학습은 실행하지 않았다.

## 현재 컴퓨터에서 실행

현재 프로젝트 폴더의 `run_local.sh`는 검증된 기존 Python 환경과 새 runtime을 지정한다. 대용량 입력과 출력은 기존 완료 실험과 별도 경로에 둔다. 준비 상태와 실제 검증값은 [준비 완료 보고서](준비완료보고서.md)에서 확인한다.

```bash
# 원자료·RIR·cache·readiness 준비. 가중치 갱신 없음.
bash run_local.sh prepare

# 존재하는 준비 산출물 확인
bash run_local.sh status

# 아래 명령부터 실제 학습이다. 이번 준비 작업에서는 실행하지 않는다.
bash run_local.sh train --start-training --stop-after 100

# 앞선 100 step checkpoint에서 3,000 step pilot을 이어갈 때
bash run_local.sh train --start-training \
  --resume /실제/runtime/runs/seed-20261111-base/last.pt
```

학습 중단은 global step 기준이다. 100 step 실행은 고정된 3,000 step 실험의 앞부분이며 scheduler 기간을 바꾸지 않는다. 기존 run이 있으면 새 학습으로 덮어쓰지 않고 `--resume`을 요구한다.

## 다른 컴퓨터와 GitHub 코드

Python은 `requirements.txt`의 패키지뿐 아니라 기존 `experiments/banelp`의 Demucs·gpuRIR 환경이 필요하다. 첫 컴퓨터의 전체 package 목록·GPU·CUDA·cuDNN·driver·모델 hash는 runtime 기록을 기준으로 한다. requirements 파일만으로 CUDA 확장 빌드까지 재현됐다고 보지 않는다. CLI는 모델이나 데이터 다운로드를 자동 실행하지 않는다.

```bash
python run.py \
  --runtime-dir /새/학습/runtime \
  --baseline-repo /기존/music-source-separation \
  --baseline-runtime /기존/banelp-v1 \
  prepare
```

baseline runtime에는 `data/musdb18hq.zip`, `manifests/dataset.json`, `manifests/models.json`, hash가 맞는 기존 HTDemucs checkpoint와 gpuRIR 환경이 있어야 한다. 원 ZIP에서 공식 train만 추출한다. 기존 test 50곡은 이번 학습에 쓰지 않는다.

`run_local.sh`는 이 컴퓨터용 실행 편의 파일이며 GitHub 코드 묶음에는 포함하지 않는다. 다른 환경에서는 위 CLI의 경로 인자를 명시한다.

## 개별 준비 단계와 평가

명령은 `prepare-data`, `rirs`, `cache`, `preflight`, `status`, `train`, `evaluate`다. 준비 중 중단되면 원본 입력과 설정이 같은 경우 완료된 파일·record의 hash를 검증해 재사용한다. 다른 내용의 기존 결과를 자동 삭제해 새 실험처럼 만들지 않는다.

```bash
# 학습 후 best checkpoint를 고정한 내부 confirmation 평가
python run.py --runtime-dir /새/학습/runtime \
  --baseline-repo /기존/music-source-separation \
  --baseline-runtime /기존/banelp-v1 \
  evaluate --checkpoint /새/학습/runtime/runs/seed-20261111-base/best.pt \
  --split confirmation --include-ensemble
```

confirmation은 checkpoint·평가 옵션을 claim에 고정한다. 다른 checkpoint로 이미 노출된 confirmation을 재평가하려 하면 거부한다. 내부 확인은 보정망 학습과 분리됐다는 뜻이며, 기초 분리기의 학습 중복까지 배제했다는 뜻은 아니다.

## 코드와 검증

| 파일 | 역할 |
|---|---|
| `prepare.py` | 공식 train 추출·CRC/디코딩/hash·곡 분할 |
| `rirs.py` | 공간 분할과 새 gpuRIR pool 생성 |
| `cache.py` | 실제 잔향 혼합 분리, hash를 고정한 파형 cache, 정렬된 crop |
| `model.py` | 복소 잔차 모델, 출력 합 보존, 공통-window 추론 |
| `losses.py` | 무음에 안정적인 기본·확장 손실 |
| `engine.py` | 업데이트 없는 preflight와 별도 학습·재개 |
| `evaluate.py` | 기존 지표를 사용한 30초 평가와 곡 bootstrap |

```bash
python -m unittest discover -s tests -v
```

검증은 곡/공간 분할, 파일 변조 거부, STFT 왕복·길이, 영 초기화, 출력 합, 무음, 손실 gradient, overlap-add, 준비 과정에서 optimizer update가 없음을 다룬다. 모델 가중치를 갱신하는 학습 테스트를 실행해 준비 상태를 학습 결과로 바꾸지 않는다.

# Music Source Separation

음원을 **보컬 / 반주**로 분리하는 웹 데모와 CLI입니다. CPU로 기본 실행하며, NVIDIA CUDA GPU 사용 시 장치가 없거나 VRAM이 부족하면 알림 후 CPU로 전환합니다. 데이터셋 없이 파일 업로드로 시작할 수 있습니다.

## Git으로 받아서 실행

Git과 Docker/Compose를 설치한 뒤 아래 URL을 실제 원격 저장소 주소로 바꿔 실행합니다.

```bash
git clone https://github.com/YOUR_ACCOUNT/music-source-separation.git
cd music-source-separation
docker compose up -d --build
```

접속 주소는 http://localhost:7860 입니다. 원격 저장소 연결, 첫 push, 서버 업데이트와 버전 고정 절차는 [Git 배포 문서](docs/git-deployment.md)에 정리했습니다. 음원·모델·실제 .env는 Git에 포함하지 않으며, 데이터 없이 웹 업로드로 사용할 수 있습니다.

## 가장 간단한 시작: Docker

Docker Engine 또는 Docker Desktop과 Compose 플러그인을 설치한 뒤 프로젝트 폴더에서 실행합니다.

```bash
docker compose up --build
```

브라우저에서 http://localhost:7860 에 접속합니다. 첫 설치와 모델 다운로드에는 인터넷과 디스크 공간이 필요합니다. 기본 Compose 구성은 CPU 전용이며 로컬 컴퓨터에서만 접속할 수 있습니다. 모델 캐시는 Docker 볼륨에 보관합니다.

## Python으로 시작: Windows / Linux / macOS

**Python 3.11 또는 3.12, 64비트와 FFmpeg**가 필요합니다. `ffmpeg -version`이 실행되는지 확인하세요.

| OS | FFmpeg 설치 예 |
|---|---|
| Windows PowerShell | `winget install --id Gyan.FFmpeg -e` 후 터미널 다시 열기 |
| Ubuntu / Debian | `sudo apt-get update && sudo apt-get install ffmpeg python3-venv build-essential cmake libsamplerate0-dev` |
| macOS (Homebrew) | `brew install ffmpeg` |

```bash
python start.py
```

Linux/macOS에서 명령명이 `python3`이면 `python3 start.py`를 사용합니다. Windows에서 기본 Python 버전이 다르면 `py -3.11 start.py`를 사용하세요. 스크립트가 `.venv/`를 만들고 설치한 후 웹 데모를 실행합니다. 두 번째 실행부터는 의존성이 바뀌지 않았으면 설치를 생략합니다. 설치만 하려면 `python start.py --install-only`, 재설치는 `--reinstall`입니다.

기본 설치는 Linux/Windows에서 PyTorch CPU 빌드를 사용합니다. 패키지의 OS/CPU용 바이너리가 없으면 컴파일러가 필요할 수 있습니다. Windows 빌드 오류가 발생하면 Visual Studio C++ Build Tools를 설치하거나 Docker 경로를 사용하세요. 실제 검증 환경은 Linux x86_64 / Python 3.11·3.12입니다. 모든 OS·하드웨어 조합의 동작을 보장하지는 않습니다.

### GPU 사용

NVIDIA GPU와 호환되는 드라이버가 있는 컴퓨터에서:

```bash
python start.py --gpu
```

별도 `.venv-gpu/`에 CUDA용 패키지를 설치하고 `MSS_USE_GPU=1`로 실행합니다. CPU와 GPU용 ONNX Runtime을 같은 환경에 섞지 마세요. 이미 GPU 환경을 구성했다면 `MSS_USE_GPU=1 python -m inference.app`도 가능합니다. Windows PowerShell은 `$env:MSS_USE_GPU="1"; python -m inference.app`입니다.

- CUDA GPU 없음 / ONNX CUDA 지원 없음 → 사유 안내 후 CPU.
- 여유 VRAM이 기준 미만 → CPU. 기본 기준은 MDX 2GB, RoFormer 6GB이며 정확한 최대 사용량 예측은 아닙니다.
- 모델 로딩 또는 추론 중 GPU 메모리 부족 / CUDA 실행 오류 → GPU 모델 해제 후 CPU로 **한 번** 재시도.
- GPU 실행에 실패한 모델은 프로세스를 다시 시작할 때까지 CPU를 사용합니다.
- 상태창과 CLI에 실제 사용 장치를 표시합니다. Apple MPS와 AMD GPU 가속은 이번 구현 범위에 포함하지 않고 CPU로 실행합니다.

GPU별 CUDA/cuDNN 호환 조건은 [audio-separator 공식 설치 문서](https://github.com/nomadkaraoke/python-audio-separator#gpu--cuda-specific-installation-steps-with-pip)를 확인하세요.

## CLI 및 테스트 음원

가상환경 활성화 후 실행합니다.

```bash
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

python scripts/make_test_audio.py
python -m inference.pretrained_infer data/raw/test_mix.wav
python -m inference.pretrained_infer input.wav -o results/my-song
python -m inference.pretrained_infer input.wav --model model_bs_roformer_ep_317_sdr_12.9755.ckpt
python -m inference.pretrained_infer --list
python scripts/bench_rtf.py data/raw/test_mix.wav --warm
```

기본 모델은 경량 MDX-Net Inst HQ3입니다. CLI 결과는 `results/`에 저장되며 동일 입력·모델·출력 폴더를 다시 사용하면 결과를 덮어쓸 수 있습니다. 웹 결과는 요청별로 별도 임시 폴더에 저장합니다. 벤치마크의 cold 시간에는 모델 로딩·다운로드·CPU 재시도가 포함되므로 warm 결과와 구분하세요.

## 폴더 구조

```text
inference/             웹/CLI, 공통 장치 선택·CPU 전환, 경로 설정
scripts/               합성 음원, 선택적 MUSDB 샘플 준비, 벤치마크, 배포 ZIP 생성
tests/                 GPU 전환, 웹 입력·경로, 배포 파일 검사
start.py               OS별 가상환경 생성·설치·웹 시작
requirements*.txt      기본 CPU / 선택적 GPU / 데이터 도구 설치 목록
Dockerfile             CPU 배포 이미지
compose.yaml           로컬 실행, 모델 캐시 볼륨, 임시 오디오 저장
.env.example           Compose용 설정 예시 (실제 비밀번호 없음)
docs/deployment.md     설정, 데이터 보관, 공개 배포 준비와 검증 기록
docs/git-deployment.md Git 업로드·clone·서버 업데이트·평가 데이터 재현
docs/00-*, 01-*        기존 연구 계획 (Git·배포 ZIP 제외)
data/, results/        로컬 음원·결과 (배포 제외)
checkpoints/           다운로드 모델 캐시 (배포 제외)
src/, configs/,
notebooks/             기존 연구용 빈 디렉터리 (배포 제외)
```

학습 코드, 자작 U-Net, SDR 평가, ONNX 내보내기는 기존 연구 문서의 계획이며 현재 구현된 기능이 아닙니다.

## 배포 묶음 및 검증

```bash
python -m unittest discover -s tests -v
python scripts/package_release.py
```

파일로 전달할 때는 **`dist/music-source-separation.zip`만 전달**하세요. Git으로 배포할 때는 추적된 소스만 push합니다. 생성기가 허용한 파일만 넣고 알려진 키 형태·개인 절대경로·실제 `MSS_AUTH` 설정을 검사합니다. `.gitignore`는 폴더 전체를 압축할 때 적용되지 않습니다. 생성 후 소스를 수정했다면 ZIP을 다시 생성하세요.

배포 방식은 로컬 사용자를 위한 ZIP + Docker, 발표용 Gradio 공유 링크, 상시 운영용 Linux 서버 + Docker + Caddy를 제안합니다. 공개 배포 시 인증, HTTPS, 음원·모델 이용 조건, 자원 제한과 데이터 보관 정책을 확인하세요. 자세한 설정과 남은 개선 제안은 [배포 문서](docs/deployment.md)에 정리했습니다.

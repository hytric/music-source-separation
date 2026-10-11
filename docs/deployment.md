# 배포 설정 및 점검

## 설정

Python 직접 실행은 셸 환경변수를 읽습니다. `.env`를 자동 로드하지 않습니다. Docker Compose는 프로젝트 루트의 `.env`를 읽으므로 `.env.example`을 복사해 실제 값을 설정할 수 있습니다. `.env`는 배포 ZIP과 Git에서 제외됩니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MSS_USE_GPU` | `0` | `1`이면 CUDA 시도, 불가능하거나 실패하면 CPU |
| `MSS_MIN_VRAM_GB` | MDX 2 / 기타 6 | 실행 전 여유 VRAM 기준; 양의 유한한 숫자 |
| `MSS_DATA_DIR` | `data/raw` | 선택적 로컬 데이터셋; 없으면 업로드만 사용 |
| `MSS_MODEL_DIR` | `checkpoints/pretrained` | 모델 캐시; 쓰기 권한 필요 |
| `MSS_OUTPUT_DIR` | `results` | CLI 출력 기본 경로 |
| `MSS_CACHE_DIR` | `.cache` | 웹 결과와 Gradio 임시 파일 |
| `MSS_HOST` | `127.0.0.1` | 직접 실행의 접속 주소; 컨테이너 내부는 `0.0.0.0` |
| `MSS_PORT` | `7860` | 직접 실행 포트 / Compose 호스트 포트 |
| `MSS_SHARE` | `0` | Gradio 임시 공유 링크; 상시 운영용 배포 대신 사용하지 않기 |
| `MSS_AUTH` | 없음 | `사용자명:비밀번호`; 공개하기 전에 실제 인증값 설정 |
| `MSS_MAX_UPLOAD_MB` | `100` | 업로드·데이터셋 파일 크기 제한 |
| `MSS_MAX_AUDIO_SECONDS` | `600` | 디코딩된 오디오 길이 제한 |
| `MSS_RETENTION_SECONDS` | `3600` | 웹 임시 결과 만료 기준 |

상대 설정 경로는 현재 터미널 디렉터리와 관계없이 프로젝트 루트를 기준으로 해석합니다. CLI가 받는 입력 경로와 명시적 `-o` 상대경로는 일반 명령처럼 현재 디렉터리를 기준으로 합니다. 웹은 과거 `results/` 전체를 파일 제공 허용 경로로 등록하지 않습니다.

Compose는 CPU 전용입니다. `MSS_USE_GPU`와 `MSS_SHARE`를 `.env`에 써도 Compose에서는 `0`으로 고정됩니다. GPU 서버 컨테이너는 NVIDIA Container Toolkit, GPU용 이미지와 호환 CUDA/cuDNN 런타임을 별도로 구성해야 합니다. 이번에는 Python의 `start.py --gpu` 경로를 제공합니다.

## 배포 방법 추천

이 프로젝트는 음원 전체를 서버에서 처리하므로, 배포 목적을 먼저 구분하는 편이 좋습니다. **사용자 컴퓨터에서 실행하도록 배포하려면 ZIP + Docker**, **웹 주소로 상시 제공하려면 Linux 서버 + Docker + Caddy**를 우선 추천합니다. 클라우드에 업로드하는 대신 아래 문서와 실행 구성을 준비했으며, 외부 서비스에는 아직 배포하지 않았습니다.

| 목적 | 방식 | 현재 코드에서 필요한 작업 |
|---|---|---|
| 다른 사람이 자기 PC에서 사용 | 배포 ZIP / GitHub Release + Docker | ZIP 압축 해제, Docker 설치, Compose 실행 |
| 발표·짧은 시연 | 내 PC + Gradio 공유 링크 | 인증 설정 후 Python 실행; PC와 프로세스 유지 |
| 소수 사용자에게 상시 제공 | Linux 서버 + Docker + Caddy | 서버·도메인 준비, 인증, HTTPS 프록시 구성 |
| 플랫폼이 서버를 관리하는 데모 | Hugging Face Docker Spaces | 플랜 확인, README 메타데이터와 컨테이너 사용자 권한 조정 |

### 1. 사용자 PC용 배포: 현재 바로 사용할 수 있는 방식

`dist/music-source-separation.zip`을 전달하거나 GitHub Release의 첨부 파일로 올립니다. 전체 작업 폴더 대신 이 ZIP만 배포하세요. 사용자는 새 폴더에 압축을 풀고 Docker Desktop/Engine 및 Compose를 설치한 뒤 실행합니다.

```bash
docker compose up -d --build
docker compose logs -f mss
```

접속 주소는 `http://localhost:7860`입니다. 종료는 `docker compose down`입니다. 모델 캐시를 지우지 않으려면 `down --volumes`는 사용하지 마세요. 업데이트할 때는 새 ZIP을 소스 파일이 들어 있는 같은 프로젝트 폴더에 반영한 뒤 다시 `up -d --build`를 실행합니다. `.env`를 덮어쓰지 마세요. 다른 폴더로 옮기면 Compose 프로젝트 이름이 바뀌어 모델 볼륨이 새로 만들어질 수 있습니다.

이 방식은 서버 임대가 필요하지 않고 각 PC의 음원을 로컬에서 처리합니다. 첫 이미지 설치와 모델 다운로드에는 인터넷이 필요합니다. Docker가 없는 사용자는 Python/FFmpeg 설치 후 `python start.py`를 사용할 수 있습니다.

### 2. 발표·임시 공개: Gradio 공유 링크

인증값은 셸 환경변수 `MSS_AUTH`로 지정합니다. 실제 비밀번호를 소스·공유 문서에 넣지 마세요. Linux/macOS Bash에서는 터미널에서 숨겨 입력할 수 있습니다.

```bash
read -rsp '접속 인증값 (사용자명:비밀번호): ' MSS_AUTH
export MSS_AUTH
MSS_SHARE=1 python start.py
```

GPU로 시연하려면 마지막 명령에 `--gpu`를 추가합니다. Windows PowerShell은 `$env:MSS_AUTH`와 `$env:MSS_SHARE` 환경변수를 지정한 뒤 `python start.py`를 실행합니다. 직접 실행은 `.env`를 읽지 않습니다. 기존 Compose는 `MSS_SHARE=0`을 고정하므로 이 시연 방식에는 Python 실행을 사용하세요.

출력된 `https://…gradio.live` 주소를 참가자에게 전달합니다. 처리 서버는 내 PC이므로 PC를 켜두고 프로세스를 유지해야 합니다. 공식 공유 서버의 링크는 72시간 제한이 있으므로 발표·피드백용으로 사용하세요. [Gradio 공유 링크 문서](https://gradio.app/guides/understanding-gradio-share-links)

### 3. 상시 웹 서비스: Linux 서버 + Docker + Caddy (추천)

서버 운영을 할 수 있고 고정 웹 주소가 필요할 때 추천합니다. 학교·연구실 서버나 임대한 Linux 서버에 같은 Docker 구성을 올립니다. **CPU 4코어, RAM 16GB, 여유 디스크 20GB 이상**을 초기 시험 구성으로 제안합니다. 이는 측정한 최소 사양이 아니며, 실제 음악 길이·선택 모델에 따라 RAM과 처리 시간을 확인해 조정해야 합니다. 처음에는 경량 MDX-Net과 짧은 음원, 소수 사용자로 시작하세요.

1. 서버에 Docker Engine/Compose와 Caddy를 설치합니다. 서버로 배포 ZIP을 옮겨 새 프로젝트 폴더에 압축을 풉니다.
2. `.env.example`을 `.env`로 복사하고 `MSS_AUTH`에 실제 접속 인증값을 설정합니다. 필요하면 음원 길이 제한을 먼저 낮춥니다. `.env` 파일은 배포하거나 Git에 올리지 마세요.
3. 아래 명령으로 앱을 시작하고 서버 내부에서 응답을 확인합니다.

```bash
cp .env.example .env
chmod 600 .env
# .env를 편집해 실제 MSS_AUTH를 설정한 다음 실행
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 mss
curl -I http://127.0.0.1:7860/
```

4. 실제 도메인의 DNS A/AAAA 레코드를 서버에 맞춥니다. 서버의 외부 포트 80/443을 Caddy로 연결하고 SSH는 운영 정책에 맞게 제한합니다. 앱 포트 7860은 현재의 `127.0.0.1` 바인딩을 유지합니다.
5. **서버 호스트에 설치한 Caddy**의 `/etc/caddy/Caddyfile`에 아래 내용을 넣고 `mss.example.com`을 실제 도메인으로 바꿉니다.

```caddyfile
mss.example.com {
    reverse_proxy 127.0.0.1:7860
}
```

이 예시는 Caddy를 앱과 같은 서버 호스트에서 실행하는 경우입니다. 별도 Caddy 컨테이너의 `127.0.0.1`은 앱 컨테이너가 아니므로 그대로 사용하지 마세요. DNS와 포트 조건이 맞으면 Caddy가 HTTPS 인증서 발급·갱신을 처리합니다. [Caddy 자동 HTTPS 문서](https://caddyserver.com/docs/automatic-https)

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable --now caddy
sudo systemctl reload caddy
```

6. 외부 브라우저에서 `https://실제도메인` 접속, 로그인, 실제 음원 분리·다운로드를 확인합니다. 서버 재부팅 후 Docker와 앱이 다시 시작하는지도 확인합니다.

기본 Compose는 CPU 전용이며 공개 무제한 서비스의 완성 구성이 아닙니다. 운영 전에 요청 속도 제한, 작업별 시간·RAM 제한, 사용자별 파일 접근 정책을 보완하세요. 이미지 빌드에 컴파일러와 패키지 설치가 필요하므로 빌드 시에도 RAM·디스크 여유가 필요합니다. GPU 서버는 현재 `start.py --gpu` 경로를 사용하거나 별도의 CUDA Docker 이미지를 준비해야 합니다.

### 4. 서버 관리를 줄이는 대안: Hugging Face Docker Spaces

2026-10-11 확인한 공식 안내에 따르면 Gradio/Docker Space 생성에는 유료 계정 플랜이 필요합니다. CPU Basic은 별도 시간당 하드웨어 요금이 없지만, 이를 모든 계정에서 무료 생성 가능한 서비스로 해석하면 안 됩니다. [Spaces 개요·계정 조건](https://huggingface.co/docs/hub/spaces-overview)

Docker SDK의 Space를 만들고 배포 ZIP에서 추출한 소스를 올립니다. ZIP 파일 자체만 올리면 앱으로 실행되지 않습니다. Space용 README 맨 위에는 다음 메타데이터를 추가합니다.

```yaml
---
title: Music Source Separation
sdk: docker
app_port: 7860
---
```

현재 Dockerfile은 UID 10001을 사용하지만 Spaces 안내는 UID 1000 기준이므로 Space용 Dockerfile의 `useradd --uid 10001`을 `--uid 1000`으로 조정하고 캐시·결과 경로의 쓰기 권한을 확인해야 합니다. 접속 인증값은 Space Settings의 **Secrets**에 `MSS_AUTH`로 등록합니다. Secrets를 소스나 공개 Variables에 넣지 마세요. [Docker Spaces 설정·권한·Secrets](https://huggingface.co/docs/hub/spaces-sdks-docker)

`compose.yaml`의 모델 볼륨·tmpfs·재시작 설정은 Docker Space에서 자동 적용되지 않습니다. 재시작 시 모델 재다운로드와 임시 파일 저장 정책을 플랫폼에 맞게 검증해야 합니다. GPU 하드웨어를 선택하더라도 현재 CPU Dockerfile이 GPU 이미지로 자동 바뀌지는 않습니다. 이 경로는 아직 해당 플랫폼에서 빌드·실행 검증하지 않았습니다.

## 음원·결과 보관

- 원래 `data/`와 `results/`의 음원은 사용자 작업 자료이므로 보존하고 배포에서 제외했습니다. 음원 내용 자체의 개인정보 유무는 검사하지 않았습니다.
- 웹의 요청별 결과는 이 프로세스가 만든 `.cache/web-*/job-*`에만 저장합니다. 정상 종료 시 프로세스 결과 폴더를 제거합니다.
- 실행 중 만료한 원본 결과는 다음 분리 요청 때 정리합니다. 요청이 없으면 만료 시각에 즉시 삭제되는 방식은 아닙니다.
- Gradio가 복사한 업로드·재생 파일은 Gradio의 `delete_cache` 설정으로 주기적으로 정리합니다. 강제 종료로 남은 캐시는 다음 시작 시 자동으로 모두 삭제하지 않습니다.
- 기본 Compose는 오디오 캐시를 2GB tmpfs에 보관합니다. 컨테이너 재생성 시 사라지며, 메모리 사용과 저장 크기를 운영 환경에 맞게 조정해야 합니다. 컨테이너가 잠시 정지해 있는 상태를 영구 삭제로 간주하지 마세요.
- 모델 파일만 Docker 볼륨에 계속 보관합니다. 입력 음원은 모델 서버로 전송하지 않지만, 모델 다운로드에는 외부 네트워크를 사용합니다. `MSS_SHARE=1` 사용 시 Gradio의 외부 공유 경로를 사용합니다.
- 공개 서비스의 사용자별 접근 격리는 별도 구현이 필요합니다. 임시 파일 URL은 서비스 인증 경계를 공유하므로 엄격한 계정별 비공개 저장소로 간주하지 마세요.

## 배포 파일과 개인정보 점검

`python scripts/package_release.py`는 명시한 소스·설정 예시·배포 문서·테스트만 ZIP에 포함합니다. 새 파일을 추가했다면 `ROOT_FILES` 또는 `SOURCE_FILES` 목록을 검토 후 갱신하세요. 키 검사 패턴은 흔한 키 형태를 검사하며 모든 종류의 개인정보·비밀번호 탐지를 보장하지 않습니다.

최초 점검 시에는 Git 저장소가 없었으며, 이후 배포용 소스만 담는 로컬 Git 저장소를 준비했습니다. 이전 Git 이력은 없어 확인할 대상이 없었습니다. 연결·업로드·서버 업데이트 절차는 [Git 배포 문서](git-deployment.md)에 정리했습니다. 소스·문서 텍스트에서는 개인 홈 경로, 이메일, 실제 비밀번호나 토큰을 발견하지 않았습니다. `.gradio/certificate.pem`은 공개 ISRG Root X1 인증서로 개인 비밀키가 아니며, 불필요한 실행 캐시로 제거했습니다. 약 512MB의 기존 음원·결과와 연구 계획·빈 연구 디렉터리는 배포 ZIP에서 제외합니다.

Git 초기 커밋은 검사한 배포 파일 목록만 명시적으로 추가합니다. 배포 ZIP과 Docker build context는 각각 허용 목록을 사용합니다. 폴더 전체를 직접 압축하지 마세요. Git에 올릴 경우 `.gitignore`가 보호하는 파일을 `git add -f`로 추가하지 마세요.

## 추가로 고려할 사항

1. **공개 운영 인증과 HTTPS**: 기본은 로컬 접속입니다. 외부에 공개할 때 `MSS_AUTH`와 HTTPS reverse proxy를 구성하고 로그인 실패 제한을 추가하세요. 별도의 계정 시스템이 필요하면 Gradio 공통 비밀번호 대신 서버 인증과 사용자별 저장소를 붙이는 것이 다음 작업입니다.
2. **음원·모델 이용 조건과 프로젝트 라이선스**: 패키지 라이선스와 각 모델 가중치·데이터의 이용 조건은 별도로 검토하세요. 소유하지 않은 샘플 음원을 배포 묶음에 넣지 않았습니다. 프로젝트 자체의 LICENSE 선택은 소유자가 결정해야 하므로 임의로 추가하지 않았습니다.
3. **자원과 시간 제한**: 현재 큐 8건, 공통 오디오 작업 동시 실행 1건, 크기·길이 제한이 있습니다. 이것은 여러 악의적 업로드나 비정상 디코더에 대한 강한 자원 격리는 아닙니다. 공개 서비스에는 worker 프로세스별 시간 제한, RAM/CPU 제한, 프록시 업로드 제한과 요청 속도 제한을 추가하세요. CPU 폴백은 실행 가능성을 높이지만 속도나 RAM 충분 여부를 보장하지 않습니다.
4. **데이터 보관과 삭제**: 운영의 삭제 시점·사용자별 접근 범위를 정하고 정기 정리 작업을 구성하세요. 디스크/메모리 캐시의 사용량을 감시하고 만료 결과의 다운로드 실패도 사용자에게 안내해야 합니다.
5. **설치 재현성과 오프라인 준비**: 직접 의존성은 고정했으나 전이 의존성까지 전부 잠근 OS별 lockfile은 아닙니다. 검증한 이미지에 태그·digest를 지정해 배포하고, 모델을 사전에 다운로드하면 첫 요청 지연을 줄일 수 있습니다. 설치·모델 다운로드에 인터넷이 필요합니다. 모델 목록 갱신 등 upstream 네트워크 호출도 있으므로 완전 오프라인 동작은 별도 검증이 필요합니다.
6. **지원 환경 자동 검증**: Windows/macOS와 새 NVIDIA CUDA 설치 환경은 CI에서 추가 검증하세요. Apple MPS와 AMD GPU 가속은 현재 자동 사용하지 않습니다. 오래된 CPU 명령어 지원·32비트·패키지 wheel이 없는 환경은 지원 범위 밖일 수 있습니다.

## 검증 기록

2026-10-11, Linux x86_64에서 검증했습니다.

| 검증 | 결과 |
|---|---|
| 공통 회귀 테스트 | 29개 통과: GPU 없음, VRAM 부족, ONNX CUDA 미지원·세션 폴백, 로딩·추론 중 OOM, 일반 오류의 재시도 방지 |
| 웹 입력·출력 | 경로 순회·외부 심볼릭 링크 차단, 외부 데이터셋의 선택 음원만 임시 복사, 크기·길이 제한, 요청별 결과 분리와 오류 메시지 검사 통과 |
| 배포 생성기 | 허용 파일만 ZIP 생성, 키·개인 경로 검사, 데이터/결과/캐시/실제 인증 설정 제외 검사 통과 |
| 새 CPU Docker 이미지 | Python 3.11, PyTorch 2.14.0+cpu / torchvision 0.29.1+cpu; 설치·pip check·모듈 import 통과 |
| Compose 실제 실행 | 설정 검사, 이미지 빌드, 일반 사용자 실행, localhost HTTP 200 및 healthcheck 통과 |
| 웹 업로드 실제 분리 | 2초 합성 WAV → MDX-Net CPU 분리 → 두 WAV와 스펙트로그램 다운로드 성공; 44.1kHz, 스테레오, 길이와 유한 샘플 확인 |
| 새 Python 직접 설치 | start.py로 Python 3.12의 새 .venv 생성·설치·모듈 import 통과; 두 번째 실행의 설치 생략 확인 |
| 직접 실행의 데이터셋 흐름 | 외부 폴더 음원 선택 → 미리보기 → CPU 분리 → 두 WAV 다운로드 성공 |
| GPU 실제 분리 | 기존 Python 3.11 CUDA 환경에서 BS-RoFormer로 2초 합성 WAV 분리 성공; 실제 장치 CUDA 표시 |
| GPU 없음 실제 전환 | CPU 빌드에서 MSS_USE_GPU=1 → GPU 없음 안내 → MDX-Net CPU 분리 성공 |
| VRAM 기준 실제 전환 | CUDA 환경에서 여유 VRAM보다 큰 100GB 기준 지정 → 사유 안내 → BS-RoFormer CPU 로딩·분리 성공 |

Windows/macOS는 실제 실행하지 않았습니다. 실제 GPU VRAM을 고갈시키는 검사는 하지 않았고, 모델 로딩·추론 중 OOM은 회귀 테스트로 재현했습니다. 합성 음원 테스트는 기능 검증이며 음악 분리 품질의 SDR 평가가 아닙니다. 짧은 클립의 cold RTF에는 모델 로딩 비용이 크게 반영되므로 장곡 처리 속도로 해석하지 마세요.

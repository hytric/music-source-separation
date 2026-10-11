# Git 저장소와 배포 절차

## 현재 준비 상태

프로젝트 루트에 `main` 브랜치의 로컬 Git 저장소와 초기 커밋을 준비했습니다. 원격 저장소 주소를 받기 전에는 원격 연결이나 업로드를 하지 않습니다. Git은 소스 변경을 관리하고 전달하는 도구이며, 서버에서 앱을 실행하려면 별도로 Docker/Compose 또는 Python 실행이 필요합니다.

소스, 설치·실행 설정, 테스트, 배포 문서만 추적합니다. `data/`, `results/`, `checkpoints/`, 가상환경, 캐시, 실제 `.env`, 개인 인증서, 대용량 음원·모델, `dist/`는 Git에서 제외합니다. 기존 연구 계획 문서 2개는 로컬 보관 자료로 제외합니다. 새 파일을 커밋하기 전에는 `git diff --cached`를 확인하고, 배포 ZIP에 포함할 소스라면 `scripts/package_release.py`의 파일 목록도 갱신하세요.

개인 이름·메일을 커밋 이력에 넣지 않도록 초기 커밋에는 프로젝트 작성자명 `Music Source Separation`과 임시 비개인 주소 `mss@example.invalid`를 사용합니다. 이 설정은 이 저장소에만 적용하며 다른 저장소의 Git 설정은 바꾸지 않습니다. GitHub 계정과 커밋을 연결하려면 계정 설정의 공식 noreply 주소를 확인해 저장소의 `user.email`을 변경하세요. 임시 주소는 실제 계정의 noreply 주소가 아닙니다. [GitHub 커밋 이메일 안내](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)

## GitHub/GitLab에 첫 업로드

1. 계정에 빈 원격 저장소를 만듭니다. 처음에는 Private로 시작해 업로드 내용을 확인하는 것을 권합니다. 저장소명은 `music-source-separation` 등을 사용할 수 있습니다.
2. README, .gitignore, LICENSE를 원격 생성 화면에서 추가하지 않습니다. 이 프로젝트에는 README와 초기 커밋이 있으므로 빈 저장소로 연결합니다. 프로젝트 자체의 라이선스는 소유자가 선택해야 하므로 아직 임의로 추가하지 않았습니다.
3. 원격 URL을 복사합니다. URL에 비밀번호나 토큰을 포함하지 마세요. HTTPS는 플랫폼의 인증 도구를 사용하거나, SSH URL과 SSH 키를 사용합니다.
4. 아래 URL을 실제 저장소 주소로 바꿔 프로젝트 폴더에서 실행합니다.

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/music-source-separation.git
git remote -v
git push -u origin main
```

원격 URL과 플랫폼 로그인은 사용자 계정에 맞춰 설정해야 합니다. `origin`이 이미 있다면 추가 명령 대신 `git remote set-url origin 실제주소`를 사용합니다. 원격에 다른 커밋이 있다면 이력 차이를 검토하세요. 이 문서의 절차는 빈 저장소 기준이며 강제 푸시는 사용하지 않습니다. [GitHub 공식 첫 업로드 절차](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

## 다른 컴퓨터에서 실행

Git과 Docker/Compose 설치 후 실제 URL로 바꿔 실행합니다.

```bash
git clone https://github.com/YOUR_ACCOUNT/music-source-separation.git
cd music-source-separation
docker compose up -d --build
```

접속 주소는 `http://localhost:7860`입니다. 첫 실행에는 이미지 설치와 모델 다운로드가 필요합니다. Git에 없는 음원은 웹에서 업로드하면 됩니다. NVIDIA GPU를 사용하려면 Python 3.11/3.12와 FFmpeg를 설치한 뒤 `python start.py --gpu`를 사용하세요. 현재 Compose는 CPU 전용입니다.

Private 저장소는 각 사용자/배포 서버에 저장소 읽기 권한이 필요합니다. 서버에는 가능하면 읽기 전용 deploy key 또는 최소 권한 인증을 사용하고 자격 증명 파일은 저장소에 넣지 마세요.

## 서버 최초 배포

서버에서 Git으로 소스를 받은 뒤 [상시 웹 배포 문서](deployment.md#3-상시-웹-서비스-linux-서버--docker--caddy-추천)의 인증·HTTPS 절차를 진행합니다.

```bash
git clone https://github.com/YOUR_ACCOUNT/music-source-separation.git
cd music-source-separation
cp .env.example .env
chmod 600 .env
# .env를 편집해 실제 접속 인증값 MSS_AUTH를 지정

docker compose up -d --build
docker compose ps
docker compose logs --tail=100 mss
```

서버 외부에는 Caddy의 80/443만 연결하고 앱의 localhost:7860 바인딩을 유지합니다. 저장소를 GitHub에 올리는 것만으로 HTTPS 웹 서비스가 자동 생성되는 것은 아닙니다. CI/CD로 서버 배포를 자동화하려면 서버·도메인·원격 저장소가 결정된 후 별도 설정해야 합니다.

## 변경 반영과 서버 업데이트

개발 컴퓨터에서 변경 파일을 명시적으로 추가하고 검토합니다.

```bash
python -m unittest discover -s tests -v
python scripts/package_release.py
git add 변경한파일
git diff --cached --check
git diff --cached
git commit -m "Describe the change"
git push
```

테스트 명령은 프로젝트 가상환경을 활성화한 상태에서 실행합니다. `git add 변경한파일`은 예시이며 실제 수정한 경로로 바꿔야 합니다. 비밀키·음원·결과를 `git add -f`로 넣지 마세요. ZIP은 Git 추적 대상이 아니므로 배포 버전의 GitHub/GitLab Release에 첨부할 수 있습니다.

서버에서는 로컬 코드 수정이 없는 상태에서 배포할 브랜치를 확인하고 업데이트합니다.

```bash
git status --short
git pull --ff-only
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 mss
```

`.env`는 Git이 관리하지 않으므로 서버 설정을 그대로 유지합니다. 모델 캐시도 같은 Compose 프로젝트의 볼륨에 보관됩니다. 앱 재시작 시 처리 중인 작업이나 임시 결과는 사라질 수 있으므로 사용자에게 점검 시간을 안내하세요. 종료는 `docker compose down`이며, 모델 볼륨을 유지하려면 `--volumes`를 붙이지 않습니다.

## 버전 고정 배포와 되돌리기

첫 배포가 실제로 검증되면 개발 컴퓨터에서 태그를 만들어 함께 업로드합니다. 아직 검증되지 않은 운영 배포를 완료했다고 표시하는 태그는 만들지 않았습니다.

```bash
git tag -a v0.1.0 -m "First verified release"
git push origin v0.1.0
```

서버에서 태그로 배포하려면:

```bash
git fetch --tags origin
git switch --detach v0.1.0
docker compose up -d --build
```

이 상태는 브랜치가 아닌 특정 버전에 고정돼 있으므로 `git pull`로 다음 버전을 받는 대신 새 태그를 fetch 후 선택합니다. 이전에 검증한 태그로 동일 절차를 실행하면 코드를 되돌릴 수 있습니다. 브랜치 배포로 돌아가려면 변경 없는 상태에서 `git switch main`을 사용합니다. 매번 재빌드할 경우 전이 의존성까지 완전히 고정된 것은 아니므로 운영에서는 검증한 컨테이너 이미지 태그/digest도 함께 보관하는 편이 좋습니다.

## 평가 데이터 범위와 재현

2026-10-11 현재 로컬에는 MUSDB의 짧은 샘플 원본 144곡(train 94 / test 50), export한 믹스 WAV 144개, 정답 보컬 WAV 144개가 있습니다. `data/raw/test_mix.wav`는 별도의 합성 테스트 음원입니다. test 샘플 50곡은 믹스와 정답 보컬 파일이 모두 존재하는 것을 확인했습니다. 전체 MUSDB18-HQ 데이터셋은 없고 정식 SDR/SIR/SAR 자동 평가 코드도 아직 구현하지 않았습니다.

데이터는 Git 저장소와 배포 ZIP에 포함하지 않습니다. 다른 컴퓨터에서 합성 테스트 음원은 아래 명령으로 생성할 수 있습니다.

```bash
python scripts/make_test_audio.py
```

MUSDB 샘플을 받으려면 가상환경을 활성화하고 선택적 데이터 의존성을 설치한 뒤 준비 스크립트를 실행합니다.

```bash
python -m pip install -r requirements-data.txt
python scripts/prepare_musdb_sample.py
```

스크립트는 musdb 라이브러리의 샘플 다운로드와 믹스/보컬 export를 수행합니다. FFmpeg와 인터넷이 필요하며, 외부에서 제공하는 샘플 내용·곡 수가 바뀔 수 있으므로 위 로컬 144곡과 완전히 동일한 구성을 보장하지는 않습니다. 정식 평가에는 데이터의 이용 조건을 확인해 전체 데이터셋과 split을 별도로 확보하고 평가 코드를 추가해야 합니다.

## 로컬 Git 준비 검증 기록

2026-10-11 검증 결과입니다.

- 추적 파일 26개를 배포 허용 목록과 대조하고 staged 파일의 비밀키 형태·개인 절대경로를 검사했습니다.
- 음원·결과·모델·실제 .env·가상환경·캐시·인증서·배포 ZIP·기존 연구 계획에 대한 제외 경로 14개를 검사했습니다.
- 기존 작업 폴더와 새로 clone한 폴더에서 회귀 테스트 29개가 각각 통과했습니다.
- 새 clone에 데이터·모델·가상환경이 포함되지 않으며, 데이터 없이 웹 UI 21개 컴포넌트가 생성됨을 확인했습니다.
- 새 clone에서 만든 배포 ZIP의 파일 목록이 Git 추적 목록과 일치하고 ZIP 무결성 검사도 통과했습니다.
- 테스트와 ZIP 생성 이후에도 Git 작업 트리가 깨끗한 것을 확인했습니다.
- 원격 저장소는 아직 연결하지 않았고 push하지 않았습니다. 실제 원격 URL을 받아 연결하는 단계가 남아 있습니다.

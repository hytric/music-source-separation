# 다른 환경에서 재현 시작하기

재현 도구는 저장된 **결과 재집계**, 이전된 자산으로 **입력 재생성·고정 모델 재추론**, 격리 workspace에서 **새 실험 실행**을 구분한다. 기존 컴퓨터의 절대경로를 새 컴퓨터에 만들 필요가 없다. 코드와작은결과는Git,원음원/RIR/큰가중치는개인용번들로분리했다. 기존절대경로는번들의과거provenance로만남고실제모델경로는새bundle/workspace에매핑한다.

## 1. 코드와 연구 환경

저장소를clone하고해당문서가추가된commit을checkout한다. 저장소루트에서아래명령을실행한다. 대상은Linux x86_64/Python3.12/NVIDIA CUDA 환경이다. Windows에서는WSL2/Linux환경을권장하며네이티브Windows를검증한것은아니다.

```bash
python3.12 -m venv .venv-research
. .venv-research/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0+cu126 torchaudio==2.6.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r docs/experiments/10_reproduction/requirements.research.resolved.txt
python -c 'import torch,numpy,scipy,soundfile; print(torch.__version__,torch.cuda.is_available())'
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

Torch2.6/CUDA12.6 설치 조합은 [PyTorch 공식 이전 버전 안내](https://pytorch.org/get-started/previous-versions/#v260)를 참조했다. 이pin은실제연구환경에서추출했다. 웹데모의루트requirements(torch2.14계열)를대신설치하지않는다. 새OS의휠가용성/드라이버까지모두검증한설치이미지는아니므로설치실패시임의버전업으로계속하지말고실제오류/환경을기록한다. [환경 명세](environment.json)의fresh-install검증여부를확인한다. gpuRIR의비이식local설치한항목은freeze에서제외했다. 고정실험에는실제생성RIR배열을사용하므로gpuRIR재컴파일이필요없다. 새RIR재생성은원래bootstrap_gpurir/build-record절차를따른다.

## 2. Git만으로 숫자 재현

```bash
python tools/reproduce.py results --output /원하는/새/results.json
```

NumPy만있으면최근6실험의66개그룹을계산하고12개대표수치[중앙값,CI]를`expected_results.json`과대조한다. dataset이나GPU는필요없다. 이것은이미저장된관측의산술검증이며새추론/학습재현이아니다. 전체검수의68비교와는선택한대조군목록이다르다.

## 3. 재추론 자산을 이전

[실제 전송파일 지문](asset_transfer.json)의단일tar와sha256파일을소유자에게전달받을수있다. 공개다운로드URL은없고Git에는메타데이터만있다.

현재마스터에서**실제내보낸개인용번들**은원자료·고정RIR·등록가중치·선택기·원래validation40입력/정답·train40입력으로약36.5GB다. 결과예측960개/venv/credential/원HWPX는포함하지않는다. raw MUSDB자료와가중치의이용·이전권한이있는환경에서만직접전달한다. 공개Git/GitLFS에음원·이큰번들을올리지않는다.

원래마스터에서새로내보낼경우:

```bash
python tools/reproduce.py export --runtime-root "$SOURCE_RUNTIME" \
  --bundle /data/mss-bundle --profile all --plan
python tools/reproduce.py export --runtime-root "$SOURCE_RUNTIME" \
  --bundle /data/mss-bundle --profile all
python tools/reproduce.py canonicalize --bundle /data/mss-bundle
```

`SOURCE_RUNTIME`은마스터의banelp-v1/banelp-refine-v1등이있는상위폴더다. 대상bundle은새빈폴더여야한다. 원자료에대해등록된SHA가있으면복사후확인한다. canonicalize는마스터의검증환경에서실행해val20의x/y를원래input/targetSHA와대조한후고정한다. all profile은이미완료된QG1의320개40초입력도포함한다. `--profile validation`은val20과고정모델의축소용이며전체train재실행용이아니다.

다른컴퓨터로전송:

```bash
# 송신측: manifest에있는파일만포함하고,mss-bundle/ prefix로묶는다.
python tools/reproduce.py pack --bundle /data/mss-bundle --archive /transfer/mss-assets-v1.tar
# 수신측: 체크섬검사후새상위폴더에추출한다.
cd /transfer
sha256sum -c mss-assets-v1.tar.sha256
tar -C /data -xf mss-assets-v1.tar
cd /path/to/music-source-separation
python tools/reproduce.py verify --bundle /data/mss-bundle
```

[expected_assets.json](expected_assets.json)은이실험자산의상대경로/bytes/SHA지문이다. verify는번들내용과공개지문·실험코드도확인한다. 기존경로가남은JSON파일을새실험의완료binding으로조작하지않는다. MSST는원commit과tracked소스/필수Git객체만보존했고원Gitconfig/hooks/logs는복사하지않았다. 자격증명없는config와재생성가능index를쓴다. index는git status가갱신할수있는cache라과학자산SHA에서제외한다.

## 4. 이동한 경로에서 입력 확인과 GPU 재추론

```bash
python tools/reproduce.py input-check --bundle /data/mss-bundle \
  --condition wet_R4 --output /data/check-input
python tools/reproduce.py infer --bundle /data/mss-bundle \
  --condition wet_R4 --model finetuned --output /data/case/finetuned
```

기본track-index0은validation의Clara Berry…Waltz다. 입력40초/정답원wet/중앙30초/순서VO-DR-BA-OT를유지한다. canonical입력이있으면hash검증후사용하고재합성결과도최대1e−6안에서일치하는지확인한다. 다른track은`--track-index 0..9`,조건은`dry/wet_R4`다. F는step1000의model키를strict적재한다. CPU/FP16로조용히전환하지않는다. 원래예측SHA의bit동일성은다른GPU/라이브러리에서보장하지않으며새실행SHA/시간/점수를남긴다.

세모델과현재방법비교:

```bash
python tools/reproduce.py infer --bundle /data/mss-bundle --condition wet_R4 --model htdemucs --output /data/case/htdemucs
python tools/reproduce.py infer --bundle /data/mss-bundle --condition wet_R4 --model bsr --output /data/case/bsr
python tools/reproduce.py infer --bundle /data/mss-bundle --condition wet_R4 --model finetuned --output /data/case/finetuned
python tools/reproduce.py compose --bundle /data/mss-bundle --condition wet_R4 \
  --predictions /data/case --output /data/case/comparison
```

compose는B/F/H/G0/DB균등/DB에너지보정의실제파형·동일정답지표를계산한다. 별도`--model clean/rev`로공식SCNet을비교할수있다. 같은case명령을다른condition/track과섞으면input SHA검사에서거절한다. 기존result가있는출력폴더는덮어쓰지않는다.

## 5. train40 teacher 재생성

full번들의train record를지정하면그record의train RIR·원래40초입력·targetSHA를사용한다. R4를train에대신적용하지않는다.

```bash
python tools/reproduce.py input-check --bundle /data/mss-bundle \
  --record-id a89b6ff1386d6eb578e53d6a --output /data/check-train
python tools/reproduce.py teacher --bundle /data/mss-bundle \
  --model finetuned --limit 320 --output /data/train40-F
```

HTD/BSR도각각별도프로세스로실행한다. 모델은프로세스안에서한번적재해재사용한다. 이teacher명령은현재새출력에만실행하며기존출력의자동resume는지원하지않는다. 중단재개가필요한전체QG1운영은원래명세의cache계약과신규코드검증을따른다. QG1의C/Qfit·교정·최종평가는아직완성된재현명령이없으며실행했다고표현하지않는다.

## 6. 원래 실험을 격리해서 새로 실행

```bash
# hardlink 미지원/exFAT/다른filesystem이면--copy를사용한다.
python tools/reproduce.py hydrate --bundle /data/mss-bundle --workspace /data/mss-run --copy
python tools/reproduce.py stage --workspace /data/mss-run --name cache-refiner --dry-run
```

hydrate는rawdata/RIR/model파일을새workspace에연결또는복사하고manifest의경로를새루트로바꾼다. **원래archive는보존하고새binding을생성**한다. 이전cache.json/oldcomplete를새실험의성공증거로복사하지않는다. 원본repo의결과파일을덮어쓰지않도록코드복사본도workspace/code에만든다. --copy는약30GB추가공간이필요하며새추론cache는별도다. full학습cache와teacher960을모두만들때는수백GB여유공간을준비한다.

[단계별 명령](단계별실행.md)과[검증 범위](검증범위.md)를읽고해당단계만실행한다. stage는명령을출력하며`--dry-run`이없으면실제로실행한다. optimizer단계는별도`--allow-optimizer`가필요하다. 새로운학습checkpoint가원래파일과같을것을보장하지않으며새결과/환경/실패를보고해야한다.

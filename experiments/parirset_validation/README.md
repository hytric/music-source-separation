# 공식 PaRIRset SCNet 비교

2026-11-13 · 20조건 최종 평가 완료

공식 clean/reverberant 가중치를 기존 HTDemucs 및 HTDemucs+BS-RoFormer 50:50과 비교한다. 새 가중치 학습은 하지 않는다. validation10곡의 dry/wet20조건, 40초 입력/중앙30초, 원래 잔향 정답을 고정한다. 공식 코드·가중치 commit은600585bc0d50cb48dd64e4602e3c34fe48be7347이다. 이번 결과의 목적은 논문의 잔향 학습이 우리 설정에서도 이득인지 확인하는 것이며 독립 테스트 일반화 검증은 아니다.

20조건·신규 CUDA 추론60회·320개 음원별 지표를 완료했다. wet에서 rev−clean은 +4.397 dB [+2.475, +9.137], rev−HTDemucs는 +3.198 dB [+1.651, +4.707], rev−50:50 앙상블은 +0.862 dB [-0.187, +1.731]였다. dry의 rev−clean은 -1.396 dB [-1.938, -0.968]다. 공식 rev 모델의 clean 대비 잔향 조건 개선은 사전 기준을 통과했다. 기존 앙상블 대비 확실한 우위도 확인하지 못했다. dry 손상도 관찰돼 무조건 교체는 권하지 않는다. wet의 vocals도 기존 앙상블 대비 −0.917 dB [−1.565, −0.141]로 악화됐다.

[상세 결과](결과/결과보고서.md) · [곡별 CSV](결과/곡별결과.csv)

[사전 계획](PLAN.md) · [고정 설정](protocol.json)

adapter.py는 공식 구조와 overlap-add를 직접 호출한다. run.py는 입력/정답/출력 SHA를 고정하고 prepare→clean→rev→bsr→score 순서로 실행한다. tests.py는 overlap 경계·마지막 샘플 보존과 패딩 정렬을 확인한다. smoke.py는 점수 이전 CUDA 메모리 시험이다. 공식 파일은 runtime에만 두며 Git에는 가중치·파형·vendor를 올리지 않는다.

```bash
python run.py --stage prepare \
  --suite /학습실험_2026-11-11 \
  --baseline-repo /music-source-separation \
  --frontend-source /입력전처리검증_2026-11-13/run.py \
  --runtime /새/parirset-compare-v1 \
  --frontend-run /frontend-verify-v1/run-9441074cf871f26a \
  --official /parirset-official-v1 \
  --baseline-runtime /banelp-v1 \
  --prepared-runtime /banelp-refine-v1
```

다음 단계는 같은 인자로 --stage clean, rev, bsr, score를 순서대로 실행한다. 로컬 프로젝트 경로를 인자로 전달한다. 공유 모듈은 기존 학습실험_2026-11-11/banelp_refine과cause_eq이며 내용 해시를 기존 frontend binding과 대조한다. 원본 frontend 파일을 수정한 사본으로 대체하면 검사에서 실패한다. 동일한 기존 데이터와 모델 registry를 필요로 한다.

공식 소스는 [고정 저장소](https://github.com/enricguso/guso_ismir26/tree/600585bc0d50cb48dd64e4602e3c34fe48be7347)의 scnet/*.py를 runtime/source/scnet에, 두 conf 파일과 models/M1_clean/model.th 및 models/M2_reverberant/model.th를 원래 상대 경로에 준비한다. acquisition.json의 SHA로 파일을 검사한다. Python3.12/torch2.6.0+cu126, numpy2.2.6, scipy1.15.3, soundfile0.13.1과 기존 baseline 환경을 썼다. 원문의 torch2.8 요구와 다른 점, CLI 오류 수정 및 shifts0 적용 범위는 계획에 명시했다.

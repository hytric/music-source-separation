# SCNet dry·wet 혼합 미세조정

공식 rev 모델에서 출발해 분리기 전체를 train80곡의 dry/wet로 추가 학습한다. 직전 검증에서 확인된 wet 개선과 dry/vocals 손상 사이의 균형을 검증한다. train RIR만 사용하고 validation10곡은 checkpoint500/1,000의 평가에 사용한다. confirmation은 사용하지 않는다. 같은40초 입력·중앙30초·원래 잔향 정답으로 초기 rev, clean, HTDemucs, 앙상블과 비교한다.

<!-- RESULT START -->
공식 rev 분리기를 train80곡의 dry/wet 대응 쌍으로 실제1,000회 갱신하고500/1,000회에서 각각20조건을 평가했다. 선택 step1000의 초기 rev 대비 wet 전체는 +0.509 dB [+0.286, +0.902], dry 전체는 +0.182 dB [+0.057, +0.239], wet vocals는 +0.097 dB [+0.063, +0.536]다. 기존 앙상블 대비 wet 전체는 +1.531 dB [+0.173, +2.243]였다. 사전 개선·보호 기준은 모두 통과했다. 기존 앙상블 대비 dry 전체는 -2.154 dB [-2.561, -1.777], wet vocals는 -0.403 dB [-1.135, +0.074]로 격차가 남아 전 입력 교체는 권하지 않는다.

[상세 결과](결과/결과보고서.md) · [곡별 CSV](결과/곡별결과.csv)
<!-- RESULT END -->

[사전 계획](PLAN.md) · [설정](protocol.json)

core.py는 affine 정규화·complex spectral loss·dry/wet 대응crop을 구현한다. run.py는 기존 manifest/입력/정답/코드를 해시로 고정하고 1,000회 갱신·중간평가·최종보고서를 생성한다. report.py는 사전 기준과 대응 차이·곡bootstrap을 계산한다. tests.py는 정답 정규화 역변환·무음·guard·완전일치gradient·vocals 가중치를 검사한다. benchmark.py의 임시 random 학습 상태는 본 실험에 사용하지 않는다.

```bash
python run.py --runtime /새/scnet-finetune-v1 \
  --prepared /banelp-refine-v1 --baseline-runtime /banelp-v1 \
  --baseline-repo /music-source-separation --official /parirset-official-v1 \
  --previous-run /parirset-compare-v1 \
  --frontend-source /입력전처리검증_2026-11-13/run.py \
  --adapter-source /공식SCNet검증_2026-11-13/adapter.py \
  --suite /학습실험_2026-11-11
```

중단 후에는 동일 인자에 --resume을 추가한다. 새로 실행할 때 기존 last.pt를 덮어쓰지 않는다. 원자료·checkpoint·파형은 runtime에만 보존하고 Git에는 코드/설정/작은 결과만 게시한다. PyTorch2.6.0+cu126의 기존 CUDA 환경을 사용하며 공개 코드의2.8 요구와 다르다. 공식 architecture와 정규화·출력 순서는 직전 검증된 adapter를 재사용한다.

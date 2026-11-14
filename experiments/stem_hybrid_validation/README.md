# 음원별 고정 결합 검증

2026-11-14 · 미세조정 SCNet과 기존 앙상블의 상보성 검증

고정 결합 H는 vocals/other를 기존 앙상블 B에서, drums/bass를 미세조정 SCNet F에서 가져온다. 동일 validation10곡 dry/wet20조건·40초입력·중앙30초·원래wet정답으로 실제 파형을 결합해 재평가한다. 주요 후보는 H 한 가지이며 결과를 보고 가중치나 음원 배분을 바꾸지 않는다. 추가학습·신규GPU추론·confirmation사용은없다.

<!-- RESULT START -->
고정 결합은 B 대비 wet 전체 +1.577 dB [+0.725, +2.079], dry 전체 -1.253 dB [-1.424, -0.846]로 dry 보호 기준을 실패했다. vocals/other 파형은 B와 정확히 같고 새 bass 소실은 없다. 음원합의 상대잔차RMS 중앙값은 dry에서 B 0.0230→H 0.0922, wet에서 0.0545→0.1919로 커졌다.

[상세 결과](결과/결과보고서.md)
<!-- RESULT END -->

[사전 계획](PLAN.md) · [설정](protocol.json)

core.py는 고정결합과음원합잔차,report.py는곡bootstrap과사전채택기준을구현한다. run.py는이전binding/manifest/예측/checkpoint/입력/정답SHA를검증한뒤CPU에서파형결합과지표계산을실행한다. raw음원합을강제보정하지않아vocals/other의기준선파형이그대로보존되는지확인할수있다. 원자료·checkpoint·파형은runtime에만둔다.

```bash
python run.py --runtime /새/stem-hybrid-v1 \
  --prepared /banelp-refine-v1 --baseline-runtime /banelp-v1 \
  --fine-runtime /scnet-finetune-v1 --official-run /parirset-compare-v1 \
  --frontend-run /frontend-verify-v1/run-9441074cf871f26a \
  --suite /학습실험_2026-11-11 --baseline-repo /music-source-separation \
  --frontend-source /입력전처리검증_2026-11-13/run.py
```

기존 SCNet미세조정 결과의step1000과이전baselinecache가필요하다. Python3.12/numpy2.2.6/scipy1.15.3/soundfile0.13.1의기존환경을사용하고BLAS스레드는2로제한한다. PyTorch를사용하는분리기추론은호출하지않는다. 실제정답없는선택기와미사용곡일반화는이번범위가아니다.

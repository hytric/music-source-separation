# vocals other 보존 출력합 보정 검증

선택기와20개선택결과를고정하고,네음원합과입력의차이를drums/bass에만배분한다. 기존vocals/other를그대로보존하면서미보정/균등배분/에너지가중배분을비교한다. 주후보는에너지배분이고균등배분과기존앙상블자체의보정은대조다. 같은validation10곡20조건·40초입력·중앙30초·원래wet정답으로추가학습/GPU추론없이평가한다.

<!-- RESULT START -->
선택기와threshold0.8을고정한drums/bass에너지가중보정은전체40초출력합최대오차2.98e-08, vocals/other차이0을달성했다. B대비wet전체 +1.037 dB [+0.066, +1.423], dry전체 -0.026 dB [-1.064, -0.014], 미보정선택기대비wet변화 -0.285 dB [-0.970, -0.061]였다. 사전성능·합·개별곡기준은 동시에통과하지못했다; 실패기준은 dry_overall, each_dry_song, wet_bass_retention, wet_drums_retention, wet_retention_vs_raw, wet_si_sar이다. 합일치만으로성능개선을판정하지않으며실사용채택은보류한다.

기존 앙상블 자체에 에너지 보정을 적용한 대조에서도 Matthew dry가 −2.808dB 악화됐다. 이 실패는 선택기의 오선택에만 국한되지 않는다. vocals/other를 고정한 채 drums/bass만으로 모든 잔차를 흡수하는 이번 두 배분은 채택하지 않는다. 미보정 선택기의 기존 결과와 파형은 보존한다.

[상세 결과](결과/결과보고서.md)
<!-- RESULT END -->

[사전 계획](PLAN.md) · [고정 설정](protocol.json)

core.py는두음원projection,report.py는사전채택기준·개별곡손상·곡bootstrap을구현한다. run.py는이전입력선택/선택기/제공파형의SHA를검증하고새파형80개를생성한다. 에너지가중치는학습된불확실성이아닌고정proxy이며합일치를강제해도성능은나빠질수있다. 무음bass출력RMS와새소실을함께보고한다.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run.py \
 --project /seperation --assets /music-source-separation/.agent_runtime \
 --runtime /새/db-consistency-v1 --baseline-repo /music-source-separation
```

기존stem-hybrid-v1/input-selector-v1과HTDemucs/BS-RoFormer파형이필요하다. Hann STFT4096/hop1024는이전입력전처리코드와동일구현을재사용한다. 전과같은노출validation으로,미사용곡/RIR·청감·실사용배포까지보장하지않는다. 파형/원자료는runtime에만두고Git에는코드/설정/작은지표만게시한다.

# vocals other 보존 출력합 보정 계획

2026-11-14 · 결과 계산 전 고정

현재 입력 선택기는 기존 앙상블 B 대비 wet 전체 +1.460dB, dry 중앙값0으로 정량 기준을 통과했지만 dry2곡의 손실과 출력합 오차가 남았다. 선택기·threshold0.8·기존20개 선택 결과를 그대로 고정한 R을 미보정/균등/에너지 가중 보정으로 비교한다. 주후보는 selected_power, selected_uniform은 대조이며 결과를 본 뒤 주후보를 바꾸지 않는다. B에도 같은 두 보정을 적용해 보정 자체의 효과를 구분한다.

입력 X와 네 음원 합의 차이 R=X−sum(Z)를 drums/bass 두 음원에만 배분하고 vocals/other 파형은 그대로 복사한다. 균등 배분은 파형 잔차를 절반씩 더한다. 에너지 배분은 Hann STFT4096/hop1024에서 두 음원의제곱크기를0.2초causal평활하고정규화해complex잔차를배분한다. 저에너지bin은0.5씩배분한다. 역변환후수치rounding잔차도두음원에만균등배분한다. 두 방식 모두 선택기가B를고른경우까지포함해20조건전체출력에적용한다.

40초입력의전체합오차≤3e-6과vocals/other의byte동일성을검사하고중앙30초의기존wet정답으로평가한다. 기존앙상블대비wet전체+0.2dB/95%CI하한>0, 미보정선택기대비wet전체/drums/bass의CI하한≥−0.2dB, dry전체B대비CI하한≥−0.2dB, wet SI-SAR미보정대비CI하한≥−0.5dB, 새bass소실0도동시에확인한다. 이번에는dry개별곡의최악손실도−0.2dB이내인지새보호기준으로사전에고정해,중앙값이소수실패를가리지않도록한다. 무음bass는소실분모에서제외하고출력RMS를따로보고한다.

출력합을맞춰도vocals/other의오차를drums/bass에넣어유령성분을만들거나기존bass이득을되돌릴수있다. 합오차감소만으로성능개선이라고하지않는다. 기존10곡20조건의입력/정답/제공예측/선택기/코드SHA를검증하고추가학습·GPU추론·confirmation·재선택은하지않는다. 새곡일반화나배포채택은판정하지않는다.

[Wisdom 등의 mixture consistency 연구](https://research.google/pubs/differentiable-consistency-constraints-for-improved-deep-speech-enhancement/)에서음원합을입력에맞추는projection원리를참고했다. 논문은음성향상실험이고vocals/other를고정한두음원배분·에너지가중치는우리음악분리후속설계다. 이가중치를학습된불확실성이나우리과제에서보장된성능으로표현하지않는다. [입력선택기결과](../input_selector_validation/결과/결과보고서.md)와[고정결합결과](../stem_hybrid_validation/결과/결과보고서.md)의파형·조건·score를재사용하고이번에는출력합과성능보존을새로검증한다.

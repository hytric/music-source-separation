# 입력 선택기 검증 계획

2026-11-14 · 검증곡의선택점수관찰전고정

고정결합H는wet+1.577dB이지만dry−1.253dB로실패했다. 조건라벨을아는dry우회는dry0/wet+1.577dB이지만실사용결과가아니다. 다음단계로입력파형특징만사용하는보수적선택기를고정해검증한다. 이는선행모델선택개념을우리자료에적용한추가설계이며성공보장이아니다.

train80곡의기존320 mixture cache에서만표준화와balanced L2 logistic 회귀를학습한다. 학습라벨은추가RIR없음0/있음1이며source정답·예측품질점수·validation음원/라벨을적합에사용하지않는다. 중심26초에서정규화대역에너지·stereo coherence·spectral flatness/flux·envelope상관/shape33개를추출한다. train30초/검증40초의중심26초를사용해합성guard와원래crop앞뒤2초를제외한다. 파일명·곡ID·신호길이·추가padding꼬리를모델특징으로넣지않는다.

규제0.01/최대100Newton반복,train만으로표준화,선택threshold0.8을고정한다. sigmoid score≥0.8이면H,그외B를곡전체에사용한다. score를교정된확률이나실제RT60으로해석하지않는다. 검증예측을본뒤threshold나특징을바꾸지않는다. 실제검증라벨은예측이완료된후분류혼동행렬에만사용한다. 실제출력평가는고정결합에서검증한동일파형/정답score를선택한결과로계산한다.

기존앙상블B대비wet전체+0.2dB/곡bootstrap95%CI하한>0, dry전체및wetvocals/other CI하한≥−0.2dB,새bass소실0을모두만족해야한다. 같은validation10곡20조건이므로탐색적이며진짜새곡·새RIR·청감평가는남는다. 실제선택기와조건라벨/GT oracle을구분해보고한다. 추가분리기학습·GPU추론·confirmation은없다.

[고정결합결과](../stem_hybrid_validation/결과/결과보고서.md)의파형·score를재사용하고이번에는추론에정답이필요없는선택기를새로적합한다.

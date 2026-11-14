# 입력만 사용하는 출력 선택기

고정결합은wet이좋지만dry손상으로실패했다. 이번에는실제추론에서정답을사용하지않는선택기를train80곡320mixture로적합하고validation20조건에적용한다. 원래분리기의추가학습이나GPU추론은없다. 선택기는입력중심26초의33개특징만보고기존앙상블B/고정결합H중하나를선택한다.

<!-- RESULT START -->
train80곡320입력만으로적합한실제입력선택기는B 대비 wet 전체 +1.460 dB [+0.080, +2.062], dry 전체 +0.000 dB [-0.155, +0.000]였다. 사전 기준은 통과했다. dry에서B/H=8/2, wet에서B/H=2/8를선택했다. 검증곡정답/조건라벨은선택에사용하지않았다. 다만 dry 2곡을 잘못 선택해 St Vitus −0.310dB, Titanium −2.202dB 손상이 남았다. dry 중앙값0을 모든 곡의 무손상으로 해석하지 않는다.

[상세 결과](결과/결과보고서.md)
<!-- RESULT END -->

[사전 계획](PLAN.md) · [설정](protocol.json)

core.py는특징33개와L2 logistic 회귀,run.py는학습mixture SHA검사·적합·입력만사용하는선택을구현한다. evaluate.py는선택한B/H의검증된score로곡bootstrap과보호기준을계산한다. 중심26초와정규화특징은입력길이/합성padding꼬리를분류단서로쓰지않기위한설계다. 공개백본사전학습중복과검증셋설계노출은여전히남는다.

```bash
OPENBLAS_NUM_THREADS=2 python run.py --project /seperation \
 --assets /music-source-separation/.agent_runtime \
 --runtime /새/input-selector-v1 --baseline-repo /music-source-separation
```

선택기의sigmoid score는교정된확률이아니며0.8은고정한의사결정threshold다. 검증라벨은예측이후혼동행렬에만사용한다. 실제음악·새RIR·미사용곡에대한성공을보장하지않는다. 원음원과분리기checkpoint는Git에넣지않고작은선택기계수와지표만보존한다.

# Separation 전체 실험 문서와 재현 허브

지금까지의실험을주제별11개폴더로정리했다. 각페이지에범위·조건·주요수치·판정·소스·재현경로가있다. 기존원보고서/코드/원시CSV는보존하고이허브에서연결한다. 마스킹하나만의문서가아니다.

| 폴더 | 내용 |
|---|---|
| [00_overview](00_overview/README.md) | 최종목표·기준선·검수·서베이·해석범위 |
| [01_baselines](01_baselines/README.md) | 기준선과 개발 후보 비교 |
| [02_reverb_diagnostics](02_reverb_diagnostics/README.md) | 잔향 영향과 성분별 원인 진단 |
| [03_cause_eq](03_cause_eq/README.md) | 크기·위상과 EQ 원인 대조 |
| [04_refiner](04_refiner/README.md) | 소형 보정망 준비와 학습 |
| [05_frontend](05_frontend/README.md) | 입력 EQ·WPE와 잔차 배분 |
| [06_scnet](06_scnet/README.md) | 공식 SCNet 비교와 전체 미세조정 |
| [07_ensemble_selector](07_ensemble_selector/README.md) | 음원별 결합과 실제 입력 선택 |
| [08_consistency](08_consistency/README.md) | 보호 음원 고정 출력합 보정 |
| [09_qg1](09_qg1/README.md) | QG1 구현과 teacher 데이터 생성 |
| [10_reproduction](10_reproduction/README.md) | 연구환경·실제이전번들·재집계/재추론/격리재실행 도구 |

## 재현의 시작

Git clone만으로는음원과큰가중치가없으므로새추론을실행할수없다. 저장된결과의재집계는Git+NumPy만으로가능하다. 새추론은[재현절차](10_reproduction/README.md)의개인용이전번들또는같은SHA의정식자산을준비한다. 번들에는기존환경venv/credential/원HWPX개인정보를넣지않았다.

현재연구환경은Python3.12/Torch2.6.0+cu126/NumPy2.2.6이다. 웹데모의루트requirements와혼합하지않는다. 경로는`--repo/--bundle/--workspace`로지정하고과거절대경로를새머신에생성하지않는다.

## 이번 정리에서 실행한 것

기존결과의산술검증과자산이전/경로재현을수행하며새모델학습은하지않는다. 실제검증수준은[검증범위](10_reproduction/검증범위.md)와기계판독기록에남긴다. 새QG1선택기성과는아직없고960teacher완료는준비단계다.

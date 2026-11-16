# 프로젝트 문서

[전체 실험과 다른 환경 재현 허브](experiments/README.md)를먼저확인한다. 기준선·잔향진단·원인EQ·소형보정망·전처리·SCNet·결합/선택·출력합·QG1·재현을주제별11폴더로나눴다. 각폴더에는조건과주요실측치·판정·원코드/결과·재현경로가있다.

원보고서와검수/논문서베이/목표지는 [기존 연구 문서](reverberant-music-source-separation-analysis-2026-11-10/README.md)에보존했다. [배포 문서](deployment.md)와[Git 배포](git-deployment.md)는웹데모운영용이며연구환경의Torch2.6/CUDA12.6와혼용하지않는다.

코드clone만으로원음원/큰모델이가져와지는것은아니다. [다른 환경 재현](experiments/10_reproduction/README.md)의자산번들·SHA·설치·이동경로검증을따른다. 저장된숫자replay는Git과NumPy만으로실행할수있다. 새추론/새학습에대한실제검증범위도별도로명시했다.

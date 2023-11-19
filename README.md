# FL-MLOps: Development of an Open Source-based MLOps Platform for Federated Learning
서버-클라이언트 아키텍처를 기반으로 연합학습을 위한 MLOps 플랫폼을 개발하였다. 

연합학습을 담당하는 Federated Learning Part와 MLOps를 담당하는 MLOps Part로 구성된다. 

FedML 라이브러리와 MLflow를 이용하여 연합학습과 MLOps를 통합하여 플랫폼을 개발하였다.

<br>

## 소프트웨어 아키텍처
<img width="520" alt="스크린샷 2023-11-20 오전 8 43 35" src="https://github.com/Dayoung-Jung/FL-MLOps/assets/68275740/da568130-ad71-4320-b07f-175cf19048c2">

### 컴포넌트 정의

(1) Upload Dataset: Data Repository를 통해 실험을 진행할 데 이터셋을 클라이언트에 업로드한다.

(2) Train (Experiment Tracking): 데이터셋을 기반으로 학습이 진행되며, MLOps tool을 통해 실험을 추적한다.

(3) Update Model: 학습을 통해 모델의 파라미터를 업데이트한다.

(4) Upload ML Model: 업데이트된 모델을 server로 업로드 한다.

(5) Model Aggregation: 클라이언트들로부터 업로드된 모델의 파라미터에 대해 Aggregation을 진행한다.

(6) Test (Experiment Tracking): Aggregation된 파라미터를 기반으로 업데이트된 모델을 테스트한다. 동시에, MLOps tool을 통 해 실험을 추적한다.

(7) Download ML Model: Aggregation을 진행한 모델을 클라이 언트에서 다운로드한다.

(2)부터 (7)까지의 컴포넌트를 연합학습에서 지정된 라운드 값 에 따라 반복 진행한다.

(8) Download ML Model: 서버의 모델을 ML Model Repository에서 다운로드한다.

(9) Download ML Model: 클라이언트의 모델을 ML Model Repository에서 다운로드한다.

(10) ML Model Monitoring: 서버와 클라이언트에서 다운로드한 모델에 대해 모니터링을 진행하며, 모델의 라이프사이클을 관리한다.


<br>


## 사용 예제
중앙 서버 실행을 위해, 터미널에 다음 명령어 입력
```bash
sh run_server.sh 1   
```

첫 번째 클라이언트 실행을 위해, 터미널에 다음 명령어 입력
```bash
sh run_client_01.sh 1 1   
```

두 번째 클라이언트 실행을 위해, 터미널에 다음 명령어 입력
```bash
sh run_client_02.sh 2 1       
```



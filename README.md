# Wi-Fi CSI 기반 비접촉 실내 이상상황 감지 시스템

Wi-Fi 신호의 CSI(Channel State Information)를 활용하여 카메라·웨어러블 장치 없이 실내에서 사람의 행동을 감지하고 이상상황(낙상)을 판별하는 저비용 프로토타입 시스템입니다.

## 주요 특징

- **비접촉 감지**: 별도 착용 장치 없이 Wi-Fi 신호만으로 행동 감지
- **프라이버시 보호**: 카메라 미사용
- **저비용**: ESP32-S3 + 일반 Wi-Fi 환경 활용
- **4종 행동 분류**: Walking, Sitting, Lying, Falling

## 시스템 구성

```
낙상감지/
├── config.py                   # 전역 설정
├── train.py                    # 모델 학습 원스탑 스크립트
├── requirements.txt            # Python 의존성
├── data_collection/            # 데이터 수집
│   ├── esp32_receiver.py       # ESP32-S3 시리얼 CSI 수신
│   ├── csi_simulator.py        # CSI 시뮬레이션 데이터 생성
│   └── data_logger.py          # 데이터 저장 유틸리티
├── preprocessing/              # 전처리
│   ├── signal_processor.py     # 노이즈 제거, 필터링, 정규화
│   ├── feature_extractor.py    # 시간/주파수 도메인 특징 추출
│   └── dataset.py              # 데이터셋 로드·분할
├── models/                     # 분류 모델
│   ├── classical_classifier.py # SVM, Random Forest
│   └── lstm_classifier.py      # LSTM, CNN-LSTM
├── realtime/                   # 실시간 감지
│   ├── detector.py             # 실시간 분류 파이프라인
│   └── alerting.py             # 알림 시스템
├── dashboard/                  # 웹 대시보드
│   ├── app.py                  # Flask + SocketIO 서버
│   └── templates/index.html    # 모니터링 UI
└── evaluation/                 # 평가
    └── evaluator.py            # 성능 평가 및 보고서
```

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 시뮬레이션 데이터 생성 → 모델 학습 → 평가 (원스탑)

```bash
python train.py --model all --samples 200
```

### 3. 개별 실행

```bash
# 시뮬레이션 데이터 생성
python data_collection/csi_simulator.py --samples 200 --output data/raw/

# 특정 모델만 학습
python train.py --model lstm --skip-generate

# 실시간 감지 (시뮬레이션 모드)
python realtime/detector.py --model lstm --simulate

# 대시보드 실행
python dashboard/app.py --simulate
# → http://localhost:5000 접속
```

### 4. ESP32-S3 실제 사용

```bash
# 사용 가능한 시리얼 포트 확인
python data_collection/esp32_receiver.py --list-ports

# 데이터 수집 (레이블별)
python data_collection/data_logger.py --label walking --duration 30 --port COM3
python data_collection/data_logger.py --label sitting --duration 30 --port COM3
python data_collection/data_logger.py --label lying --duration 30 --port COM3
python data_collection/data_logger.py --label falling --duration 30 --port COM3

# 실시간 감지 (ESP32 연결)
python realtime/detector.py --model lstm --port COM3
```

## 하드웨어

- **ESP32-S3** (CSI 데이터 수집용)
- Wi-Fi AP (일반 공유기)
- PC/노트북 (Python 실행)

## 기술 스택

- Python 3.8+
- TensorFlow/Keras (LSTM, CNN-LSTM)
- scikit-learn (SVM, Random Forest)
- SciPy (신호처리)
- Flask + SocketIO (대시보드)
- Chart.js (시각화)

## 목표 성능

| 지표 | 목표 |
|------|------|
| 분류 정확도 | ≥ 85% |
| 오탐률 | ≤ 15% |
| 미탐률 | ≤ 15% |
| 판별 시간 | ≤ 3초 |

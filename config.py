"""
Wi-Fi CSI 기반 비접촉 실내 이상상황 감지 시스템 - 전역 설정
"""
import os

# ===== 경로 설정 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ===== 행동 클래스 =====
ACTIVITY_CLASSES = ["walking", "sitting", "lying", "falling"]
NUM_CLASSES = len(ACTIVITY_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ACTIVITY_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(ACTIVITY_CLASSES)}

# 이상 상황 클래스 (falling은 이상 상황)
ANOMALY_CLASSES = ["falling"]

# ===== ESP32 CSI 설정 =====
ESP32_SERIAL_PORT = "COM3"          # Windows 기본 시리얼 포트
ESP32_BAUD_RATE = 921600            # ESP32-S3 기본 baud rate
CSI_NUM_SUBCARRIERS = 64            # 서브캐리어 수
CSI_SAMPLE_RATE = 100               # 초당 CSI 패킷 수 (Hz)

# ===== 전처리 설정 =====
WINDOW_SIZE = 100                   # 슬라이딩 윈도우 크기 (샘플 수)
WINDOW_STRIDE = 50                  # 슬라이딩 윈도우 이동 간격
LOWPASS_CUTOFF = 20.0               # 저역통과 필터 컷오프 주파수 (Hz)
LOWPASS_ORDER = 4                   # 버터워스 필터 차수

# ===== 모델 설정 =====
LSTM_HIDDEN_SIZE = 64               # LSTM 은닉 유닛 수
LSTM_NUM_LAYERS = 2                 # LSTM 레이어 수
LEARNING_RATE = 0.001               # 학습률
BATCH_SIZE = 32                     # 배치 크기
EPOCHS = 50                         # 최대 에포크 수
EARLY_STOPPING_PATIENCE = 10        # 조기 종료 인내 에포크

# ===== 실시간 감지 설정 =====
DETECTION_THRESHOLD = 0.7           # 이상상황 판별 임계값 (확률)
ALERT_COOLDOWN = 5                  # 경고 반복 방지 쿨다운 (초)
RESULT_TIMEOUT = 3.0                # 판별 시간 제한 (초)

# ===== 대시보드 설정 =====
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
SOCKETIO_ASYNC_MODE = "threading"

# 디렉토리 자동 생성
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

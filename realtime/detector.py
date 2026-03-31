"""
실시간 CSI 감지 및 분류
학습된 모델로 실시간·시뮬레이션 스트리밍 CSI 데이터를 분류
"""
import sys
import os
import time
import logging
from datetime import datetime
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ACTIVITY_CLASSES, IDX_TO_CLASS, ANOMALY_CLASSES,
    WINDOW_SIZE, CSI_NUM_SUBCARRIERS,
    DETECTION_THRESHOLD, ALERT_COOLDOWN, RESULT_TIMEOUT
)
from preprocessing.signal_processor import SignalProcessor
from realtime.alerting import AlertManager

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """실시간 CSI 기반 이상상황 감지기"""

    def __init__(self, model_type="lstm"):
        self.model_type = model_type
        self.model = None
        self.processor = SignalProcessor()
        self.alert_manager = AlertManager()

        # 슬라이딩 윈도우 버퍼
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.is_running = False

        # 결과 이력
        self.detection_history = []
        
        # 확률 스무딩 (EMA)
        self.prev_proba = None
        self.alpha = 0.5  # 전환 속도 상향 (신속한 반영)

    def load_model(self, model_type=None):
        """학습된 모델 로드"""
        model_type = model_type or self.model_type

        if model_type in ("lstm", "cnn_lstm"):
            from models.lstm_classifier import LSTMClassifier
            self.model = LSTMClassifier()
            self.model.load()
        elif model_type in ("svm", "rf"):
            from models.classical_classifier import ClassicalClassifier
            self.model = ClassicalClassifier(model_type=model_type)
            self.model.load()
        else:
            raise ValueError(f"지원하지 않는 모델: {model_type}")

        self.model_type = model_type
        logger.info(f"{model_type} 모델 로드 완료")

    def process_packet(self, packet):
        """
        CSI 패킷 실시간 처리
        
        1. 버퍼에 추가
        2. 버퍼가 가득 차면 분류 수행
        3. 이상상황 감지 시 알림
        
        Returns:
            dict or None: 분류 결과
        """
        amplitudes = packet.get("amplitudes", np.zeros(CSI_NUM_SUBCARRIERS))
        self.buffer.append(amplitudes)

        if len(self.buffer) < WINDOW_SIZE:
            return None

        # 윈도우 데이터 추출
        window = np.array(list(self.buffer))

        # 전처리
        start_time = time.time()
        processed = self.processor.preprocess_pipeline(window)

        # 분류
        if self.model_type in ("lstm", "cnn_lstm"):
            # 딥러닝: (1, window_size, n_subcarriers)
            X = processed.reshape(1, *processed.shape)
            proba = self.model.predict_proba(X)[0]
        else:
            # ML: 특징 추출 필요
            from preprocessing.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor()
            features = extractor.extract_all_features(processed)
            X = features.reshape(1, -1)
            proba = self.model.predict_proba(X)[0]

        # 확률값 EMA 스무딩 
        if self.prev_proba is None:
            self.prev_proba = proba
        else:
            proba = self.alpha * proba + (1 - self.alpha) * self.prev_proba
            self.prev_proba = proba

        pred_idx = np.argmax(proba)
        pred_class = IDX_TO_CLASS[pred_idx]
        confidence = float(proba[pred_idx])

        elapsed = time.time() - start_time

        result = {
            "timestamp": datetime.now().isoformat(),
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(proba)},
            "is_anomaly": pred_class in ANOMALY_CLASSES and confidence >= DETECTION_THRESHOLD,
            "processing_time": elapsed
        }

        self.detection_history.append(result)

        # 이상상황 감지 → 알림
        if result["is_anomaly"]:
            self.alert_manager.trigger_alert(result)

        return result

    def run_with_esp32(self, port=None):
        """ESP32 실시간 감지"""
        from data_collection.esp32_receiver import ESP32CSIReceiver

        receiver = ESP32CSIReceiver(port=port)
        self.is_running = True

        print("\n" + "=" * 60)
        print("  [LIVE] 실시간 CSI 이상상황 감지 시작")
        print("  Ctrl+C로 종료")
        print("=" * 60 + "\n")

        try:
            for packet in receiver.receive_stream():
                if not self.is_running:
                    break
                result = self.process_packet(packet)
                if result:
                    self._print_result(result)
        except KeyboardInterrupt:
            print("\n감지 종료")
        finally:
            self.is_running = False

    def run_simulation(self, activity_sequence=None, duration_per_activity=10):
        """시뮬레이션 모드로 실행 (테스트용)"""
        from data_collection.csi_simulator import CSISimulator

        if activity_sequence is None:
            activity_sequence = ["walking", "sitting", "lying", "falling", "sitting"]

        sim = CSISimulator(seed=None)
        self.is_running = True

        print("\n" + "=" * 60)
        print("  [SIM] 시뮬레이션 모드 실시간 감지 시작")
        print(f"  시나리오: {' -> '.join(activity_sequence)}")
        print("=" * 60 + "\n")

        try:
            for activity in activity_sequence:
                print(f"\n--- 시뮬레이션: '{activity}' ({duration_per_activity}초) ---")
                stream = sim.stream_simulate(activity=activity)
                start = time.time()

                for packet in stream:
                    if not self.is_running or (time.time() - start) >= duration_per_activity:
                        break
                    result = self.process_packet(packet)
                    if result:
                        self._print_result(result)

        except KeyboardInterrupt:
            print("\n시뮬레이션 종료")
        finally:
            self.is_running = False

    def _print_result(self, result):
        """감지 결과 콘솔 출력"""
        cls = result["predicted_class"]
        conf = result["confidence"]
        t = result["processing_time"]

        if result["is_anomaly"]:
            icon = "[ALERT]"
            color_start = "\033[91m"  # 빨강
            color_end = "\033[0m"
        elif cls == "walking":
            icon = "[WALK]"
            color_start = "\033[92m"
            color_end = "\033[0m"
        elif cls == "sitting":
            icon = "[SIT]"
            color_start = "\033[94m"
            color_end = "\033[0m"
        elif cls == "lying":
            icon = "[LIE]"
            color_start = "\033[93m"
            color_end = "\033[0m"
        else:
            icon = "[UNK]"
            color_start = ""
            color_end = ""

        print(f"  {icon} {color_start}{cls:10s}{color_end} | "
              f"신뢰도: {conf:.2%} | 처리시간: {t*1000:.1f}ms"
              f"{'  [WARN] 이상감지!' if result['is_anomaly'] else ''}")

    def get_stats(self):
        """감지 통계 반환"""
        if not self.detection_history:
            return {}

        classes = [r["predicted_class"] for r in self.detection_history]
        anomalies = [r for r in self.detection_history if r["is_anomaly"]]
        times = [r["processing_time"] for r in self.detection_history]

        return {
            "total_detections": len(self.detection_history),
            "anomaly_count": len(anomalies),
            "class_distribution": {c: classes.count(c) for c in ACTIVITY_CLASSES},
            "avg_processing_time": np.mean(times),
            "max_processing_time": np.max(times),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="실시간 CSI 감지 시스템")
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["svm", "rf", "lstm", "cnn_lstm"])
    parser.add_argument("--simulate", action="store_true",
                        help="시뮬레이션 모드 실행")
    parser.add_argument("--port", type=str, default=None,
                        help="ESP32 시리얼 포트")
    parser.add_argument("--duration", type=int, default=10,
                        help="시뮬레이션 활동당 시간(초)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    detector = RealtimeDetector(model_type=args.model)
    detector.load_model()

    if args.simulate:
        detector.run_simulation(duration_per_activity=args.duration)
    else:
        detector.run_with_esp32(port=args.port)

    stats = detector.get_stats()
    print(f"\n[STATS] 통계: {stats}")

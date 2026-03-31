"""
실시간 모니터링 대시보드
Flask + SocketIO 기반 웹 대시보드
"""
import sys
import os
import json
import time
import logging
import threading
from datetime import datetime

import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DASHBOARD_HOST, DASHBOARD_PORT, SOCKETIO_ASYNC_MODE,
    ACTIVITY_CLASSES, IDX_TO_CLASS, CSI_NUM_SUBCARRIERS
)

logger = logging.getLogger(__name__)

_base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(_base_dir, "templates"),
            static_folder=os.path.join(_base_dir, "static"))
app.config["SECRET_KEY"] = "csi-fall-detection-secret"
app.config["TEMPLATES_AUTO_RELOAD"] = True
socketio = SocketIO(app, async_mode=SOCKETIO_ASYNC_MODE, cors_allowed_origins="*")

# 전역 상태
dashboard_state = {
    "is_running": False,
    "detection_history": [],
    "current_status": "대기 중",
    "current_class": "unknown",
    "current_confidence": 0.0,
    "stats": {
        "total": 0,
        "anomaly_count": 0,
        "class_counts": {c: 0 for c in ACTIVITY_CLASSES}
    }
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(dashboard_state)


@app.route("/api/history")
def api_history():
    return jsonify(dashboard_state["detection_history"][-100:])


def emit_detection(result):
    """감지 결과를 웹소켓으로 전송"""
    dashboard_state["current_class"] = result["predicted_class"]
    dashboard_state["current_confidence"] = result["confidence"]
    dashboard_state["current_status"] = (
        "🚨 이상 감지!" if result["is_anomaly"] else "정상 모니터링 중"
    )
    dashboard_state["detection_history"].append(result)
    dashboard_state["stats"]["total"] += 1
    dashboard_state["stats"]["class_counts"][result["predicted_class"]] += 1
    if result["is_anomaly"]:
        dashboard_state["stats"]["anomaly_count"] += 1

    # 최근 500개만 유지
    if len(dashboard_state["detection_history"]) > 500:
        dashboard_state["detection_history"] = dashboard_state["detection_history"][-500:]

    socketio.emit("detection", result)


def run_simulation_stream():
    """시뮬레이션 스트림을 백그라운드에서 실행"""
    from data_collection.csi_simulator import CSISimulator

    # 모델 로드 시도 (실패해도 더미 시뮬레이션으로 진행)
    detector = None
    try:
        from realtime.detector import RealtimeDetector
        detector = RealtimeDetector(model_type="svm")
        detector.load_model()
        logger.info("모델 로드 성공 - 실제 분류 사용")
    except Exception as e:
        logger.warning(f"모델 로드 실패({e}) - 더미 시뮬레이션으로 전환")
        detector = None

    sim = CSISimulator(seed=None)
    activities = ["walking", "sitting", "lying", "falling", "walking", "sitting"]
    dashboard_state["is_running"] = True
    dashboard_state["current_status"] = "시뮬레이션 실행 중"

    rng = np.random.default_rng()
    ACTIVITY_DURATION = 8  # 활동당 8초
    EMIT_INTERVAL = 0.5    # 0.5초마다 결과 전송

    try:
        while dashboard_state["is_running"]:
            for activity in activities:
                if not dashboard_state["is_running"]:
                    break
                logger.info(f"시뮬레이션 전환: {activity}")

                # 활동 전환 시 detector 버퍼 초기화
                if detector:
                    detector.buffer.clear()

                stream = sim.stream_simulate(activity=activity)
                start = time.time()
                packet_count = 0

                for packet in stream:
                    if not dashboard_state["is_running"]:
                        break
                    if time.time() - start > ACTIVITY_DURATION:
                        break

                    # detector에 패킷을 모아서 처리 (버퍼 채우기)
                    if detector:
                        result = detector.process_packet(packet)
                        packet_count += 1
                        # 100패킷(윈도우 크기)마다 한 번 emit + sleep
                        if result:
                            emit_detection(result)
                            time.sleep(EMIT_INTERVAL)
                        elif packet_count % 10 == 0:
                            # 버퍼가 아직 안 찼으면 짧은 대기
                            time.sleep(0.01)
                    else:
                        # 더미 결과 생성
                        base_conf = rng.uniform(0.65, 0.95)
                        probs = {}
                        remaining = 1.0 - base_conf
                        other_classes = [c for c in ACTIVITY_CLASSES if c != activity]
                        for c in other_classes:
                            p = rng.uniform(0, remaining / len(other_classes) * 2)
                            probs[c] = round(p, 4)
                            remaining -= p
                        probs[activity] = round(base_conf, 4)

                        is_anomaly = activity == "falling" and base_conf > 0.7
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "predicted_class": activity,
                            "confidence": round(base_conf, 4),
                            "probabilities": probs,
                            "is_anomaly": is_anomaly,
                            "processing_time": rng.uniform(0.01, 0.05)
                        }
                        emit_detection(result)
                        time.sleep(EMIT_INTERVAL)

    except Exception as e:
        logger.error(f"시뮬레이션 오류: {e}")
    finally:
        dashboard_state["is_running"] = False
        dashboard_state["current_status"] = "중지됨"


@socketio.on("start_simulation")
def handle_start_simulation():
    """시뮬레이션 시작 이벤트"""
    if not dashboard_state["is_running"]:
        thread = threading.Thread(target=run_simulation_stream, daemon=True)
        thread.start()
        socketio.emit("status", {"message": "시뮬레이션 시작"})


@socketio.on("stop")
def handle_stop():
    """중지 이벤트"""
    dashboard_state["is_running"] = False
    socketio.emit("status", {"message": "중지"})


def run_dashboard(host=None, port=None, simulate=False):
    """대시보드 실행"""
    host = host or DASHBOARD_HOST
    port = port or DASHBOARD_PORT

    print(f"\n{'='*60}")
    print(f"  [DASHBOARD] CSI 이상상황 감지 대시보드")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}\n")

    if simulate:
        thread = threading.Thread(target=run_simulation_stream, daemon=True)
        thread.start()

    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSI 모니터링 대시보드")
    parser.add_argument("--host", default=DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    parser.add_argument("--simulate", action="store_true",
                        help="시뮬레이션 모드로 시작")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_dashboard(host=args.host, port=args.port, simulate=args.simulate)

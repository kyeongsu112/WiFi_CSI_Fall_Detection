"""
CSI 데이터 로거
수집된 CSI 데이터를 레이블과 함께 CSV 파일로 저장
"""
import sys
import os
import csv
import json
import logging
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, CSI_NUM_SUBCARRIERS, ACTIVITY_CLASSES

logger = logging.getLogger(__name__)


class CSIDataLogger:
    """CSI 데이터를 레이블과 함께 저장하는 로거"""

    def __init__(self, output_dir=None, label=None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.label = label
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.packet_count = 0
        self.csv_writer = None
        self.csv_file = None

        if self.label:
            self.label_dir = os.path.join(self.output_dir, self.label)
            os.makedirs(self.label_dir, exist_ok=True)

    def start_recording(self, label=None):
        """레이블을 지정하고 새로운 녹화 세션 시작"""
        if label:
            self.label = label
            self.label_dir = os.path.join(self.output_dir, self.label)
            os.makedirs(self.label_dir, exist_ok=True)

        if not self.label:
            raise ValueError("레이블을 지정해주세요.")

        filename = f"recording_{self.session_id}_{self.label}.csv"
        filepath = os.path.join(self.label_dir, filename)

        self.csv_file = open(filepath, "w", newline="", encoding="utf-8")
        columns = ["timestamp", "rssi", "channel"] + \
                  [f"sc_{i}" for i in range(CSI_NUM_SUBCARRIERS)]
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(columns)
        self.packet_count = 0

        logger.info(f"녹화 시작: {filepath} (레이블: {self.label})")
        return filepath

    def log_packet(self, packet):
        """CSI 패킷 한 줄 기록"""
        if self.csv_writer is None:
            raise RuntimeError("녹화가 시작되지 않았습니다. start_recording()을 먼저 호출하세요.")

        row = [
            packet.get("timestamp", ""),
            packet.get("rssi", ""),
            packet.get("channel", "")
        ]
        amplitudes = packet.get("amplitudes", np.zeros(CSI_NUM_SUBCARRIERS))
        row.extend(amplitudes.tolist())

        self.csv_writer.writerow(row)
        self.packet_count += 1

        if self.packet_count % 100 == 0:
            self.csv_file.flush()
            logger.debug(f"  {self.packet_count}개 패킷 기록 완료")

    def stop_recording(self):
        """녹화 종료"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            logger.info(f"녹화 종료: {self.packet_count}개 패킷 저장")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()


def collect_labeled_data(receiver, logger_obj, label, duration_sec, sample_rate=100):
    """
    레이블을 지정하여 특정 시간 동안 데이터 수집

    사용법:
        receiver = ESP32CSIReceiver()
        logger = CSIDataLogger()
        collect_labeled_data(receiver, logger, "walking", 30)
    """
    import time

    filepath = logger_obj.start_recording(label)
    print(f"\n📡 '{label}' 데이터 수집 시작! ({duration_sec}초)")
    print(f"   저장 위치: {filepath}")

    start_time = time.time()
    try:
        for packet in receiver.receive_stream():
            elapsed = time.time() - start_time
            if elapsed >= duration_sec:
                break
            logger_obj.log_packet(packet)

            remaining = duration_sec - elapsed
            if int(elapsed) % 5 == 0 and logger_obj.packet_count % sample_rate == 0:
                print(f"   잔여 시간: {remaining:.0f}초 "
                      f"| 수집: {logger_obj.packet_count}개")
    finally:
        logger_obj.stop_recording()

    print(f"✅ '{label}' 수집 완료: {logger_obj.packet_count}개 패킷")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSI 데이터 수집 도구")
    parser.add_argument("--label", type=str, required=True,
                        choices=ACTIVITY_CLASSES,
                        help="행동 레이블")
    parser.add_argument("--duration", type=int, default=30,
                        help="수집 시간(초) (기본: 30)")
    parser.add_argument("--port", type=str, default=None,
                        help="ESP32 시리얼 포트")
    parser.add_argument("--output", type=str, default=None,
                        help="저장 디렉토리")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from esp32_receiver import ESP32CSIReceiver

    receiver = ESP32CSIReceiver(port=args.port)
    with CSIDataLogger(output_dir=args.output, label=args.label) as data_logger:
        collect_labeled_data(receiver, data_logger, args.label, args.duration)

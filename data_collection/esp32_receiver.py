"""
ESP32-S3 CSI 데이터 수신기
시리얼 포트를 통해 ESP32-S3에서 전송하는 CSI 데이터를 실시간 수신·파싱
"""
import sys
import os
import time
import json
import logging
from datetime import datetime

import serial
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ESP32_SERIAL_PORT, ESP32_BAUD_RATE,
    CSI_NUM_SUBCARRIERS, RAW_DATA_DIR
)

logger = logging.getLogger(__name__)


class ESP32CSIReceiver:
    """ESP32-S3 시리얼 포트에서 CSI 데이터를 수신하는 클래스"""

    def __init__(self, port=None, baud_rate=None, num_subcarriers=None):
        self.port = port or ESP32_SERIAL_PORT
        self.baud_rate = baud_rate or ESP32_BAUD_RATE
        self.num_subcarriers = num_subcarriers or CSI_NUM_SUBCARRIERS
        self.serial_conn = None
        self.is_connected = False
        self.packet_count = 0

    def connect(self):
        """시리얼 포트 연결"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.is_connected = True
            logger.info(f"ESP32-S3 연결 성공: {self.port} @ {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            logger.error(f"ESP32-S3 연결 실패: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """시리얼 포트 연결 해제"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            logger.info("ESP32-S3 연결 해제")

    def parse_csi_line(self, line):
        """
        ESP32-S3 CSI 출력 라인 파싱
        
        ESP32 CSI 데이터 형식 (esp-csi 프로젝트 기준):
        CSI_DATA,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<cwb>,<smoothing>,
        <not_sounding>,<aggregation>,<stbc>,<fec_coding>,<sgi>,<noise_floor>,
        <ampdu_cnt>,<channel>,<secondary_channel>,<local_timestamp>,<ant>,
        <sig_len>,<rx_state>,<len>,<first_word_invalid>,[csi_data]
        """
        try:
            if not line.startswith("CSI_DATA"):
                return None

            parts = line.split(",")
            if len(parts) < 25:
                return None

            # 메타데이터 추출
            mac = parts[1]
            rssi = int(parts[2])
            channel = int(parts[15])
            timestamp = int(parts[17])

            # CSI 데이터 추출 (마지막 부분의 대괄호 안 데이터)
            csi_str = line.split("[")[-1].rstrip("]").strip()
            csi_raw = [int(x) for x in csi_str.split(",") if x.strip()]

            # I/Q 데이터 → 진폭 변환
            # CSI 데이터는 [I0, Q0, I1, Q1, ...] 형태
            if len(csi_raw) < 2:
                return None

            amplitudes = []
            phases = []
            for i in range(0, len(csi_raw) - 1, 2):
                real = csi_raw[i]
                imag = csi_raw[i + 1]
                amp = np.sqrt(real**2 + imag**2)
                phase = np.arctan2(imag, real)
                amplitudes.append(amp)
                phases.append(phase)

            # 서브캐리어 수 맞추기 (패딩 or 자르기)
            amplitudes = np.array(amplitudes[:self.num_subcarriers])
            phases = np.array(phases[:self.num_subcarriers])

            if len(amplitudes) < self.num_subcarriers:
                amplitudes = np.pad(amplitudes,
                                    (0, self.num_subcarriers - len(amplitudes)))
                phases = np.pad(phases,
                                (0, self.num_subcarriers - len(phases)))

            self.packet_count += 1
            return {
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "mac": mac,
                "rssi": rssi,
                "channel": channel,
                "amplitudes": amplitudes,
                "phases": phases,
                "packet_id": self.packet_count
            }

        except (ValueError, IndexError) as e:
            logger.debug(f"CSI 파싱 오류: {e}")
            return None

    def receive_stream(self, callback=None, max_packets=None):
        """
        실시간 CSI 데이터 스트리밍 수신
        
        Args:
            callback: 각 CSI 패킷에 대해 호출할 콜백 함수
            max_packets: 최대 수신 패킷 수 (None이면 무한)
        
        Yields:
            dict: 파싱된 CSI 데이터 딕셔너리
        """
        if not self.is_connected:
            if not self.connect():
                return

        logger.info("CSI 데이터 수신 시작...")
        count = 0

        try:
            while True:
                if max_packets and count >= max_packets:
                    break

                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        parsed = self.parse_csi_line(line)
                        if parsed:
                            count += 1
                            if callback:
                                callback(parsed)
                            yield parsed
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            logger.info(f"수신 중단. 총 {count}개 패킷 수신")
        finally:
            self.disconnect()


def list_serial_ports():
    """사용 가능한 시리얼 포트 목록 반환"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    return [(p.device, p.description) for p in ports]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESP32-S3 CSI 데이터 수신기")
    parser.add_argument("--port", type=str, default=ESP32_SERIAL_PORT,
                        help=f"시리얼 포트 (기본: {ESP32_SERIAL_PORT})")
    parser.add_argument("--baud", type=int, default=ESP32_BAUD_RATE,
                        help=f"전송 속도 (기본: {ESP32_BAUD_RATE})")
    parser.add_argument("--list-ports", action="store_true",
                        help="사용 가능한 시리얼 포트 목록 표시")
    parser.add_argument("--max-packets", type=int, default=None,
                        help="최대 수신 패킷 수")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.list_ports:
        print("\n사용 가능한 시리얼 포트:")
        for device, desc in list_serial_ports():
            print(f"  {device}: {desc}")
        sys.exit(0)

    receiver = ESP32CSIReceiver(port=args.port, baud_rate=args.baud)
    for packet in receiver.receive_stream(max_packets=args.max_packets):
        print(f"[{packet['packet_id']:05d}] RSSI={packet['rssi']:4d} | "
              f"Amp mean={np.mean(packet['amplitudes']):.2f} | "
              f"Ch={packet['channel']}")

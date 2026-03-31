"""
CSI 데이터 시뮬레이터
ESP32 장치 없이 행동별 CSI 데이터를 생성하여 ML 파이프라인 개발·테스트에 활용
"""
import sys
import os
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ACTIVITY_CLASSES, CSI_NUM_SUBCARRIERS,
    CSI_SAMPLE_RATE, RAW_DATA_DIR, WINDOW_SIZE
)

logger = logging.getLogger(__name__)


class CSISimulator:
    """
    행동별 현실적 CSI 데이터 시뮬레이션 생성기
    
    각 행동은 고유한 주파수·진폭 특성을 가진다:
    - walking: 주기적 진폭 변화 (1-2Hz 보행 주기), 높은 분산
    - sitting: 낮은 변동, 안정적 진폭, 미세한 호흡 패턴
    - lying: 매우 낮은 변동, sitting보다 더 안정적, 미약한 체동
    - falling: 급격한 진폭 스파이크 → 이후 안정화 (충격 + 정지)
    """

    def __init__(self, num_subcarriers=None, sample_rate=None, seed=None):
        self.num_subcarriers = num_subcarriers or CSI_NUM_SUBCARRIERS
        self.sample_rate = sample_rate or CSI_SAMPLE_RATE
        self.rng = np.random.default_rng(seed)

        # 서브캐리어별 기본 진폭 (주파수 선택적 페이딩 반영)
        self.base_amplitudes = self._generate_base_profile()

    def _generate_base_profile(self):
        """서브캐리어별 기본 진폭 프로파일"""
        freqs = np.linspace(0, 1, self.num_subcarriers)
        profile = 20 + 10 * np.sin(2 * np.pi * freqs * 3)
        profile += self.rng.normal(0, 1, self.num_subcarriers)
        return np.abs(profile)

    def generate_walking(self, duration_sec):
        """
        보행 CSI 시뮬레이션
        - 1-2Hz 주기적 진폭 변동 (발걸음)
        - 도플러 효과로 인한 서브캐리어 위상 변화
        - 높은 분산
        """
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples)
        data = np.zeros((n_samples, self.num_subcarriers))

        for sc in range(self.num_subcarriers):
            base = self.base_amplitudes[sc]
            # 보행 주기 (1.5-2Hz)
            walk_freq = 1.5 + 0.5 * self.rng.random()
            walk_amp = 5.0 + 3.0 * self.rng.random()

            # 도플러 시프트 (서브캐리어마다 다른 위상)
            doppler_phase = self.rng.uniform(0, 2 * np.pi)

            signal = (base
                      + walk_amp * np.sin(2 * np.pi * walk_freq * t + doppler_phase)
                      + 1.5 * np.sin(2 * np.pi * walk_freq * 2 * t)  # 2차 고조파
                      + self.rng.normal(0, 1.5, n_samples))  # 노이즈

            data[:, sc] = np.abs(signal)

        return data

    def generate_sitting(self, duration_sec):
        """
        앉기 CSI 시뮬레이션
        - 낮은 변동, 안정적 진폭
        - 미세한 호흡 패턴 (0.2-0.4Hz)
        - 간헐적 소동작 (팔 움직임 등)
        """
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples)
        data = np.zeros((n_samples, self.num_subcarriers))

        for sc in range(self.num_subcarriers):
            base = self.base_amplitudes[sc]
            # 호흡 패턴
            breath_freq = 0.25 + 0.15 * self.rng.random()
            breath_amp = 0.8 + 0.5 * self.rng.random()

            signal = (base
                      + breath_amp * np.sin(2 * np.pi * breath_freq * t)
                      + self.rng.normal(0, 0.5, n_samples))

            # 간헐적 소동작 (불규칙한 단발 이벤트)
            for _ in range(int(duration_sec * 0.3)):
                event_pos = self.rng.integers(0, n_samples)
                event_width = self.rng.integers(5, 20)
                event_amp = self.rng.uniform(1.0, 3.0)
                start = max(0, event_pos - event_width // 2)
                end = min(n_samples, event_pos + event_width // 2)
                window = np.hanning(end - start)
                signal[start:end] += event_amp * window

            data[:, sc] = np.abs(signal)

        return data

    def generate_lying(self, duration_sec):
        """
        누워있기 CSI 시뮬레이션
        - 매우 안정적 (sitting보다 더 낮은 변동)
        - 호흡 미약하게 반영
        - 가끔 뒤척임
        """
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples)
        data = np.zeros((n_samples, self.num_subcarriers))

        for sc in range(self.num_subcarriers):
            base = self.base_amplitudes[sc]
            # 호흡 (누운 상태에서 약간 다른 주파수)
            breath_freq = 0.2 + 0.1 * self.rng.random()
            breath_amp = 0.5 + 0.3 * self.rng.random()

            signal = (base
                      + breath_amp * np.sin(2 * np.pi * breath_freq * t)
                      + self.rng.normal(0, 0.3, n_samples))

            # 간헐적 뒤척임 (매우 드물게)
            for _ in range(int(duration_sec * 0.1)):
                event_pos = self.rng.integers(0, n_samples)
                event_width = self.rng.integers(20, 50)
                event_amp = self.rng.uniform(2.0, 5.0)
                start = max(0, event_pos - event_width // 2)
                end = min(n_samples, event_pos + event_width // 2)
                window = np.hanning(end - start)
                signal[start:end] += event_amp * window

            data[:, sc] = np.abs(signal)

        return data

    def generate_falling(self, duration_sec):
        """
        낙상 CSI 시뮬레이션
        - 초반: 급격한 진폭 스파이크 (충격)
        - 직전: 약간의 불안정한 움직임 (비틀거림)
        - 이후: 매우 안정적 (바닥에 누운 상태)
        """
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples)
        data = np.zeros((n_samples, self.num_subcarriers))

        # 낙상 발생 시점 (전체 구간의 30-50% 지점)
        fall_point = int(n_samples * self.rng.uniform(0.3, 0.5))
        # 스파이크 지속 시간
        spike_width = int(self.sample_rate * 0.3)  # 0.3초
        # 전환 구간
        transition = int(self.sample_rate * 0.5)     # 0.5초

        for sc in range(self.num_subcarriers):
            base = self.base_amplitudes[sc]
            signal = np.ones(n_samples) * base

            # 1단계: 낙상 전 - 약간의 불안정 (비틀거림)
            pre_fall = fall_point
            if pre_fall > 0:
                wobble_freq = 3.0 + 2.0 * self.rng.random()
                wobble_amp = 3.0 + 2.0 * self.rng.random()
                signal[:pre_fall] += wobble_amp * np.sin(
                    2 * np.pi * wobble_freq * t[:pre_fall]
                )
                # 점진적 증가
                ramp = np.linspace(0.5, 1.5, pre_fall)
                signal[:pre_fall] *= ramp

            # 2단계: 낙상 스파이크
            spike_start = fall_point
            spike_end = min(n_samples, fall_point + spike_width)
            spike_amp = 15.0 + 10.0 * self.rng.random()
            spike_signal = spike_amp * np.exp(
                -3 * np.linspace(0, 1, spike_end - spike_start)
            )
            signal[spike_start:spike_end] += spike_signal

            # 3단계: 낙상 후 - 바닥에서 안정 (lying과 유사하지만 더 급격한 전환)
            post_start = spike_end
            if post_start < n_samples:
                lying_level = base * 0.8  # 바닥에서 약간 낮은 진폭
                trans_end = min(n_samples, post_start + transition)
                # 급격한 전환
                if trans_end > post_start:
                    trans = np.linspace(signal[post_start - 1] if post_start > 0 else base,
                                        lying_level, trans_end - post_start)
                    signal[post_start:trans_end] = trans
                # 안정 상태
                if trans_end < n_samples:
                    signal[trans_end:] = lying_level
                    # 미세한 호흡만
                    signal[trans_end:] += 0.3 * np.sin(
                        2 * np.pi * 0.2 * t[trans_end:]
                    )

            signal += self.rng.normal(0, 0.8, n_samples)
            data[:, sc] = np.abs(signal)

        return data

    def generate_dataset(self, samples_per_class=100, duration_per_sample=3.0):
        """
        전체 클래스에 대한 데이터셋 생성

        Args:
            samples_per_class: 클래스당 샘플 수
            duration_per_sample: 샘플당 시간 (초)

        Returns:
            list[dict]: 각 항목은 {'data': np.array, 'label': str, 'sample_id': int}
        """
        generators = {
            "walking": self.generate_walking,
            "sitting": self.generate_sitting,
            "lying": self.generate_lying,
            "falling": self.generate_falling,
        }

        dataset = []
        for cls_name in ACTIVITY_CLASSES:
            gen_func = generators[cls_name]
            logger.info(f"'{cls_name}' 데이터 생성 중... ({samples_per_class}개)")

            for i in range(samples_per_class):
                data = gen_func(duration_per_sample)
                dataset.append({
                    "data": data,
                    "label": cls_name,
                    "sample_id": i
                })

        return dataset

    def save_dataset(self, dataset, output_dir=None):
        """
        데이터셋을 CSV 파일로 저장
        
        파일 구조:
            output_dir/
                walking/
                    sample_000.csv
                    sample_001.csv
                    ...
                sitting/
                    ...
        """
        output_dir = output_dir or RAW_DATA_DIR

        for item in dataset:
            cls_dir = os.path.join(output_dir, item["label"])
            os.makedirs(cls_dir, exist_ok=True)

            filename = f"sample_{item['sample_id']:04d}.csv"
            filepath = os.path.join(cls_dir, filename)

            # 컬럼: subcarrier_0, subcarrier_1, ..., subcarrier_63
            columns = [f"sc_{i}" for i in range(self.num_subcarriers)]
            df = pd.DataFrame(item["data"], columns=columns)
            df.to_csv(filepath, index=False)

        logger.info(f"데이터셋 저장 완료: {output_dir}")

    def stream_simulate(self, activity="walking", callback=None):
        """실시간 스트리밍 시뮬레이션 (1초 단위 생성 반복)"""
        generators = {
            "walking": self.generate_walking,
            "sitting": self.generate_sitting,
            "lying": self.generate_lying,
            "falling": self.generate_falling,
        }
        gen_func = generators.get(activity, self.generate_walking)

        try:
            while True:
                data = gen_func(1.0)  # 1초 분량
                for row in data:
                    packet = {
                        "timestamp": int(datetime.now().timestamp() * 1000),
                        "datetime": datetime.now().isoformat(),
                        "amplitudes": row,
                        "rssi": -50 + int(self.rng.normal(0, 5)),
                        "channel": 6
                    }
                    if callback:
                        callback(packet)
                    yield packet
                    import time
                    time.sleep(1.0 / self.sample_rate)
        except KeyboardInterrupt:
            logger.info("시뮬레이션 중단")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSI 데이터 시뮬레이터")
    parser.add_argument("--samples", type=int, default=200,
                        help="클래스당 생성할 샘플 수 (기본: 200)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="샘플당 시간(초) (기본: 3.0)")
    parser.add_argument("--output", type=str, default=None,
                        help=f"출력 디렉토리 (기본: {RAW_DATA_DIR})")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본: 42)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sim = CSISimulator(seed=args.seed)
    dataset = sim.generate_dataset(
        samples_per_class=args.samples,
        duration_per_sample=args.duration
    )
    sim.save_dataset(dataset, output_dir=args.output)

    print(f"\n[완료] 데이터셋 생성 완료!")
    print(f"  총 샘플 수: {len(dataset)}")
    print(f"  클래스: {ACTIVITY_CLASSES}")
    print(f"  저장 위치: {args.output or RAW_DATA_DIR}")

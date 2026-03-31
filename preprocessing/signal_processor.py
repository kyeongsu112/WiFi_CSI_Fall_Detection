"""
CSI 신호 전처리
노이즈 제거, 정규화, 필터링 등 CSI 데이터 전처리 파이프라인
"""
import sys
import os

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CSI_SAMPLE_RATE, LOWPASS_CUTOFF, LOWPASS_ORDER
)


class SignalProcessor:
    """CSI 신호 전처리 클래스"""

    def __init__(self, sample_rate=None, cutoff_freq=None, filter_order=None):
        self.sample_rate = sample_rate or CSI_SAMPLE_RATE
        self.cutoff_freq = cutoff_freq or LOWPASS_CUTOFF
        self.filter_order = filter_order or LOWPASS_ORDER

    def butterworth_lowpass(self, data, cutoff=None, order=None):
        """
        버터워스 저역통과 필터
        
        Args:
            data: 1D or 2D 배열 (n_samples,) or (n_samples, n_subcarriers)
            cutoff: 컷오프 주파수 (Hz)
            order: 필터 차수
        
        Returns:
            필터링된 데이터
        """
        cutoff = cutoff or self.cutoff_freq
        order = order or self.filter_order
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = cutoff / nyquist

        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99

        b, a = butter(order, normalized_cutoff, btype='low')

        if data.ndim == 1:
            return filtfilt(b, a, data)
        else:
            filtered = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered[:, i] = filtfilt(b, a, data[:, i])
            return filtered

    def median_filter(self, data, kernel_size=5):
        """
        중앙값 필터 (스파이크 노이즈 제거)
        
        Args:
            data: 1D or 2D 배열
            kernel_size: 필터 커널 크기 (홀수)
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        if data.ndim == 1:
            return medfilt(data, kernel_size)
        else:
            filtered = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered[:, i] = medfilt(data[:, i], kernel_size)
            return filtered

    def moving_average(self, data, window_size=5):
        """
        이동 평균 스무딩
        
        Args:
            data: 1D or 2D 배열
            window_size: 윈도우 크기
        """
        if data.ndim == 1:
            return uniform_filter1d(data, size=window_size)
        else:
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                smoothed[:, i] = uniform_filter1d(data[:, i], size=window_size)
            return smoothed

    def normalize_minmax(self, data):
        """
        Min-Max 정규화 [0, 1]
        """
        if data.ndim == 1:
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val == 0:
                return np.zeros_like(data)
            return (data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(data, dtype=np.float64)
            for i in range(data.shape[1]):
                col = data[:, i]
                min_val = np.min(col)
                max_val = np.max(col)
                if max_val - min_val == 0:
                    normalized[:, i] = 0
                else:
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
            return normalized

    def normalize_zscore(self, data):
        """
        Z-score 정규화 (평균 0, 분산 1)
        """
        if data.ndim == 1:
            std = np.std(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - np.mean(data)) / std
        else:
            normalized = np.zeros_like(data, dtype=np.float64)
            for i in range(data.shape[1]):
                col = data[:, i]
                std = np.std(col)
                if std == 0:
                    normalized[:, i] = 0
                else:
                    normalized[:, i] = (col - np.mean(col)) / std
            return normalized

    def remove_outliers(self, data, threshold=3.0):
        """
        Z-score 기반 이상치 제거 (보간으로 대체)
        
        Args:
            data: 1D or 2D 배열
            threshold: Z-score 임계값
        """
        if data.ndim == 1:
            z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-8))
            mask = z_scores > threshold
            cleaned = data.copy()
            # 이상치를 인접 값의 평균으로 대체
            for idx in np.where(mask)[0]:
                left = max(0, idx - 1)
                right = min(len(data) - 1, idx + 1)
                cleaned[idx] = (cleaned[left] + cleaned[right]) / 2
            return cleaned
        else:
            cleaned = data.copy()
            for i in range(data.shape[1]):
                cleaned[:, i] = self.remove_outliers(data[:, i], threshold)
            return cleaned

    def preprocess_pipeline(self, data, normalize_method="minmax"):
        """
        전체 전처리 파이프라인 (권장 순서)
        
        1. 중앙값 필터 (스파이크 제거)
        2. 버터워스 저역통과 필터
        3. 이동 평균 스무딩
        4. 정규화
        
        Args:
            data: (n_samples, n_subcarriers) 배열
            normalize_method: "minmax" 또는 "zscore"
        
        Returns:
            전처리된 데이터
        """
        # 1. 스파이크 제거
        data = self.median_filter(data, kernel_size=5)

        # 2. 저역통과 필터
        data = self.butterworth_lowpass(data)

        # 3. 스무딩
        data = self.moving_average(data, window_size=3)

        # 4. 정규화
        if normalize_method == "minmax":
            data = self.normalize_minmax(data)
        elif normalize_method == "zscore":
            data = self.normalize_zscore(data)

        return data


if __name__ == "__main__":
    # 간단한 데모
    print("=== 신호 전처리 데모 ===")
    processor = SignalProcessor()

    # 테스트 데이터 생성
    np.random.seed(42)
    raw = np.random.randn(300, 64) * 10 + 20
    # 스파이크 노이즈 추가
    raw[50, 10] = 200
    raw[150, 30] = -100

    processed = processor.preprocess_pipeline(raw, normalize_method="minmax")
    print(f"입력 형상: {raw.shape}")
    print(f"출력 형상: {processed.shape}")
    print(f"입력 범위: [{raw.min():.2f}, {raw.max():.2f}]")
    print(f"출력 범위: [{processed.min():.2f}, {processed.max():.2f}]")
    print("✅ 전처리 파이프라인 정상 동작")

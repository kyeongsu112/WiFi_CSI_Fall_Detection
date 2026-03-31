"""
RuView 기반 고급 신호 처리 모듈
RuView(https://github.com/ruvnet/RuView) 프로젝트의 신호처리 기법을 Python으로 구현

포함 기법:
1. Hampel 필터 (이상치 제거)
2. Fresnel Zone 기반 움직임 감지
3. SpotFi 기반 AoA(도달 각도) 추정
4. BVP(Body-coordinate Velocity Profile) 추출
5. CSI 비율(CSI Ratio) 기법 - 위상 보정
6. Spectrogram 기반 시간-주파수 분석
"""
import sys
import os

import numpy as np
from scipy.signal import stft, butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CSI_SAMPLE_RATE, CSI_NUM_SUBCARRIERS


class AdvancedSignalProcessor:
    """RuView 기반 고급 CSI 신호 처리"""

    def __init__(self, sample_rate=None, num_subcarriers=None):
        self.sample_rate = sample_rate or CSI_SAMPLE_RATE
        self.num_subcarriers = num_subcarriers or CSI_NUM_SUBCARRIERS

    # ========================
    # 1. Hampel 필터
    # ========================
    def hampel_filter(self, data, window_size=7, threshold=3.0):
        """
        Hampel 필터 (RuView의 핵심 노이즈 제거 기법)

        중앙값 기반 이상치 탐지 → 보간 대체
        일반적인 z-score 대비 로버스트한 이상치 처리

        Args:
            data: 1D 또는 2D 배열
            window_size: 윈도우 크기 (양측)
            threshold: MAD(Median Absolute Deviation) 기반 임계값
        """
        if data.ndim == 1:
            return self._hampel_1d(data, window_size, threshold)
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                result[:, i] = self._hampel_1d(data[:, i], window_size, threshold)
            return result

    def _hampel_1d(self, signal, window_size, threshold):
        """1D Hampel 필터 구현"""
        n = len(signal)
        filtered = signal.copy()
        k = 1.4826  # MAD 상수 (정규분포 가정)

        for i in range(n):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            window = signal[start:end]

            median = np.median(window)
            mad = k * np.median(np.abs(window - median))

            if mad > 0 and np.abs(signal[i] - median) > threshold * mad:
                filtered[i] = median

        return filtered

    # ========================
    # 2. Fresnel Zone 움직임 감지
    # ========================
    def fresnel_zone_detection(self, csi_data, lambda_val=0.125):
        """
        Fresnel Zone 기반 움직임 감지 (RuView 핵심 기법)

        Wi-Fi 신호의 Fresnel Zone 이론을 적용:
        사람이 1차 Fresnel Zone 내에서 움직이면 CSI 변화가 극대화됨

        Args:
            csi_data: (n_samples, n_subcarriers)
            lambda_val: Wi-Fi 파장 (2.4GHz → ~0.125m)

        Returns:
            dict: {
                'motion_index': 시간별 움직임 지수,
                'is_moving': bool 배열,
                'fresnel_energy': Fresnel 에너지
            }
        """
        n_samples, n_sc = csi_data.shape

        # 서브캐리어 간 상관관계 변화 → 움직임 지표
        # 인접 서브캐리어 쌍의 CSI 비율 변화
        motion_index = np.zeros(n_samples)

        for t in range(1, n_samples):
            # 현재 vs 이전 시점의 서브캐리어 프로파일 차이
            diff = np.abs(csi_data[t] - csi_data[t - 1])
            # 가중 합 (중간 서브캐리어에 높은 가중치)
            weights = np.exp(-((np.arange(n_sc) - n_sc / 2) ** 2) / (2 * (n_sc / 4) ** 2))
            motion_index[t] = np.sum(diff * weights)

        # Fresnel 에너지 (서브캐리어 상관 행렬의 고유값 기반)
        if n_samples > n_sc:
            # 공분산 행렬의 최대 고유값 변화
            window = min(50, n_samples // 4)
            fresnel_energy = np.zeros(n_samples)

            for t in range(window, n_samples):
                segment = csi_data[t - window:t]
                cov = np.cov(segment.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                fresnel_energy[t] = eigenvalues[-1] / (np.sum(eigenvalues) + 1e-8)
        else:
            fresnel_energy = motion_index.copy()

        # 움직임 판별 (적응적 임계값)
        threshold = np.mean(motion_index) + 2 * np.std(motion_index)
        is_moving = motion_index > threshold

        return {
            "motion_index": motion_index,
            "is_moving": is_moving,
            "fresnel_energy": fresnel_energy
        }

    # ========================
    # 3. SpotFi AoA 추정
    # ========================
    def spotfi_aoa(self, csi_matrix, antenna_spacing=0.025):
        """
        SpotFi 기반 AoA (Angle of Arrival) 추정 (간소화 버전)

        다중 안테나 CSI에서 MUSIC 알고리즘 기반 도달 각도 추정
        단일 안테나에서는 서브캐리어 위상 차이를 활용

        Args:
            csi_matrix: (n_samples, n_subcarriers)
            antenna_spacing: 안테나 간격 (m)

        Returns:
            angle_profile: 시간별 추정 도달 각도 프로파일
        """
        n_samples, n_sc = csi_matrix.shape
        angles = np.linspace(-90, 90, 181)  # -90° ~ +90°
        angle_profile = np.zeros((n_samples, len(angles)))

        for t in range(n_samples):
            # 서브캐리어 위상 차이 기반 (단일 안테나 근사)
            if n_sc >= 2:
                phase_diff = np.diff(csi_matrix[t])
                # 각 각도에 대한 스티어링 벡터와의 상관
                for a_idx, angle in enumerate(angles):
                    steering = np.exp(
                        -1j * 2 * np.pi * antenna_spacing *
                        np.sin(np.radians(angle)) * np.arange(len(phase_diff))
                        / 0.125
                    )
                    angle_profile[t, a_idx] = np.abs(
                        np.sum(phase_diff * np.real(steering))
                    )

        return angle_profile

    # ========================
    # 4. BVP 추출
    # ========================
    def extract_bvp(self, csi_data, n_velocity_bins=20):
        """
        BVP (Body-coordinate Velocity Profile) 추출
        Widar 3.0 논문의 핵심 특징 - CSI에서 도플러 속도 프로파일 추출

        Args:
            csi_data: (n_samples, n_subcarriers)
            n_velocity_bins: 속도 빈 수

        Returns:
            bvp: (n_time_bins, n_velocity_bins) 도플러 속도 프로파일
        """
        n_samples, n_sc = csi_data.shape

        # STFT로 시간-주파수 분해
        nperseg = min(64, n_samples // 2)
        if nperseg < 4:
            return np.zeros((1, n_velocity_bins))

        # 각 서브캐리어의 도플러 스펙트럼 추출
        all_spectra = []
        for sc in range(n_sc):
            f, t_stft, Zxx = stft(csi_data[:, sc],
                                   fs=self.sample_rate,
                                   nperseg=nperseg,
                                   noverlap=nperseg // 2)
            all_spectra.append(np.abs(Zxx))

        # 서브캐리어별 스펙트럼 평균
        avg_spectrum = np.mean(all_spectra, axis=0)

        # 속도 빈으로 리사이즈
        from scipy.ndimage import zoom
        if avg_spectrum.shape[0] != n_velocity_bins:
            zoom_factor = (n_velocity_bins / avg_spectrum.shape[0],
                           1 if avg_spectrum.shape[1] == 1
                           else avg_spectrum.shape[1] / avg_spectrum.shape[1])
            try:
                bvp = zoom(avg_spectrum, (n_velocity_bins / avg_spectrum.shape[0],
                                          1), order=1)
            except Exception:
                bvp = avg_spectrum[:n_velocity_bins] if avg_spectrum.shape[0] >= n_velocity_bins \
                    else np.pad(avg_spectrum,
                                ((0, n_velocity_bins - avg_spectrum.shape[0]), (0, 0)))
        else:
            bvp = avg_spectrum

        return bvp.T  # (n_time_bins, n_velocity_bins)

    # ========================
    # 5. CSI 비율 기법
    # ========================
    def csi_ratio(self, csi_data, reference_sc=0):
        """
        CSI Ratio 기법 - 위상 오프셋 제거

        두 서브캐리어의 CSI 비율을 사용하여
        하드웨어 의존적 위상 오프셋을 제거하는 기법

        Args:
            csi_data: (n_samples, n_subcarriers)
            reference_sc: 참조 서브캐리어 인덱스

        Returns:
            corrected: 위상 보정된 CSI 데이터
        """
        reference = csi_data[:, reference_sc:reference_sc + 1]
        # 0 나누기 방지
        reference = np.where(np.abs(reference) < 1e-8, 1e-8, reference)
        corrected = csi_data / reference
        return corrected

    # ========================
    # 6. 스펙트로그램 특징
    # ========================
    def compute_spectrogram(self, csi_data, nperseg=64):
        """
        CSI 데이터의 시간-주파수 스펙트로그램 계산

        Args:
            csi_data: (n_samples, n_subcarriers)
            nperseg: STFT 세그먼트 크기

        Returns:
            spectrogram: (n_freq, n_time, n_subcarriers) 스펙트로그램
        """
        n_samples, n_sc = csi_data.shape
        nperseg = min(nperseg, n_samples // 2)
        if nperseg < 4:
            nperseg = 4

        spectrograms = []
        for sc in range(n_sc):
            f, t, Zxx = stft(csi_data[:, sc], fs=self.sample_rate,
                              nperseg=nperseg, noverlap=nperseg // 2)
            spectrograms.append(np.abs(Zxx))

        return np.stack(spectrograms, axis=-1)

    def extract_spectrogram_features(self, csi_data, nperseg=64):
        """
        스펙트로그램에서 통계 특징 추출 (딥러닝 입력 대안)

        Returns:
            features: 1D 특징 벡터
        """
        spec = self.compute_spectrogram(csi_data, nperseg)
        features = []

        # 전체 통계
        features.append(np.mean(spec))
        features.append(np.std(spec))
        features.append(np.max(spec))

        # 주파수 대역별 에너지
        n_freq = spec.shape[0]
        n_bands = min(5, n_freq)
        band_size = n_freq // n_bands
        for b in range(n_bands):
            band = spec[b * band_size:(b + 1) * band_size]
            features.append(np.mean(band))
            features.append(np.std(band))

        # 시간별 에너지 변화
        time_energy = np.mean(spec, axis=(0, 2))
        features.append(np.mean(time_energy))
        features.append(np.std(time_energy))
        features.append(np.max(time_energy) / (np.mean(time_energy) + 1e-8))

        return np.array(features)

    # ========================
    # 통합 파이프라인
    # ========================
    def advanced_pipeline(self, csi_data):
        """
        RuView 기반 고급 전처리 파이프라인

        순서:
        1. Hampel 필터 (이상치 제거)
        2. CSI Ratio (위상 보정)
        3. 저역통과 필터
        4. Fresnel Zone 움직임 감지
        5. BVP 추출

        Args:
            csi_data: (n_samples, n_subcarriers)

        Returns:
            dict: {
                'cleaned': Hampel 필터링된 데이터,
                'motion': Fresnel 움직임 지수,
                'bvp': Body Velocity Profile,
                'spectrogram_features': 스펙트로그램 특징
            }
        """
        # 1. Hampel 필터 (RuView 핵심)
        cleaned = self.hampel_filter(csi_data, window_size=5, threshold=3.0)

        # 2. CSI Ratio (위상 보정)
        corrected = self.csi_ratio(cleaned)

        # 3. 저역통과 필터
        nyq = self.sample_rate / 2.0
        cutoff = min(20.0 / nyq, 0.99)
        b, a = butter(4, cutoff, btype='low')
        filtered = np.zeros_like(corrected)
        for i in range(corrected.shape[1]):
            try:
                filtered[:, i] = filtfilt(b, a, corrected[:, i])
            except ValueError:
                filtered[:, i] = corrected[:, i]

        # 4. Fresnel Zone 움직임 감지
        motion_result = self.fresnel_zone_detection(filtered)

        # 5. BVP 추출
        bvp = self.extract_bvp(filtered)

        # 6. 스펙트로그램 특징
        spec_features = self.extract_spectrogram_features(filtered)

        return {
            "cleaned": filtered,
            "motion": motion_result,
            "bvp": bvp,
            "spectrogram_features": spec_features,
            "is_motion_detected": np.any(motion_result["is_moving"])
        }


if __name__ == "__main__":
    print("=== RuView 기반 고급 신호 처리 데모 ===")
    processor = AdvancedSignalProcessor()

    # 테스트 데이터
    np.random.seed(42)
    t = np.linspace(0, 3, 300)
    data = np.zeros((300, 64))
    for sc in range(64):
        data[:, sc] = (20 + 5 * np.sin(2 * np.pi * 1.5 * t + sc * 0.1)
                       + np.random.randn(300) * 1.5)
    # 이상치 주입
    data[50, 10] = 200
    data[150, 30] = -100

    result = processor.advanced_pipeline(data)

    print(f"입력: {data.shape}")
    print(f"Hampel 필터 후: {result['cleaned'].shape}")
    print(f"움직임 감지: {np.sum(result['motion']['is_moving'])}개 시점")
    print(f"BVP: {result['bvp'].shape}")
    print(f"스펙트로그램 특징: {result['spectrogram_features'].shape}")
    print(f"움직임 감지됨: {result['is_motion_detected']}")
    print("[완료] 고급 신호 처리 파이프라인 정상 동작")

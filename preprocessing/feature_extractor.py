"""
특징 추출 모듈
CSI 시계열 데이터에서 시간 도메인 및 주파수 도메인 특징을 추출
"""
import sys
import os
import argparse
import json
import logging

import numpy as np
from scipy.fft import fft
from scipy.stats import kurtosis, skew

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WINDOW_SIZE, WINDOW_STRIDE, CSI_SAMPLE_RATE, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """CSI 데이터에서 특징을 추출하는 클래스"""

    def __init__(self, window_size=None, window_stride=None, sample_rate=None):
        self.window_size = window_size or WINDOW_SIZE
        self.window_stride = window_stride or WINDOW_STRIDE
        self.sample_rate = sample_rate or CSI_SAMPLE_RATE

    def sliding_window(self, data):
        """
        슬라이딩 윈도우로 데이터 분할
        
        Args:
            data: (n_samples, n_subcarriers) 배열
        
        Returns:
            list of (window_size, n_subcarriers) 배열
        """
        windows = []
        n_samples = data.shape[0]

        for start in range(0, n_samples - self.window_size + 1, self.window_stride):
            end = start + self.window_size
            windows.append(data[start:end])

        return windows

    def extract_time_features(self, window):
        """
        시간 도메인 특징 추출 (윈도우 단위)
        
        Args:
            window: (window_size, n_subcarriers) 배열
        
        Returns:
            1D 특징 벡터
        """
        features = []
        n_sc = window.shape[1]

        for sc in range(n_sc):
            col = window[:, sc]

            # 기본 통계량
            features.append(np.mean(col))           # 평균
            features.append(np.std(col))             # 표준편차
            features.append(np.var(col))             # 분산
            features.append(np.max(col) - np.min(col))  # 범위
            features.append(np.median(col))          # 중앙값

            # 고차 통계량
            features.append(skew(col))               # 왜도
            features.append(kurtosis(col))           # 첨도

            # 에너지
            features.append(np.sum(col**2))          # 에너지

            # 이동량 관련
            diff = np.diff(col)
            features.append(np.mean(np.abs(diff)))   # 평균 변화량
            features.append(np.std(diff))            # 변화량 표준편차

            # 피크 (zero-crossing rate의 변형)
            zero_crossings = np.sum(np.diff(np.sign(col - np.mean(col))) != 0)
            features.append(zero_crossings)          # 영교차 수

        return np.array(features)

    def extract_freq_features(self, window):
        """
        주파수 도메인 특징 추출 (FFT 기반)
        
        Args:
            window: (window_size, n_subcarriers) 배열
        
        Returns:
            1D 특징 벡터
        """
        features = []
        n_sc = window.shape[1]

        for sc in range(n_sc):
            col = window[:, sc]

            # FFT 수행
            fft_vals = np.abs(fft(col))
            freqs = np.fft.fftfreq(len(col), d=1.0 / self.sample_rate)

            # 양의 주파수만
            positive_mask = freqs > 0
            fft_positive = fft_vals[positive_mask]
            freqs_positive = freqs[positive_mask]

            if len(fft_positive) == 0:
                features.extend([0] * 5)
                continue

            # 주파수 특징
            features.append(np.max(fft_positive))           # 최대 주파수 성분
            features.append(np.mean(fft_positive))          # 평균 주파수 에너지

            # 주요 주파수 (dominant frequency)
            dominant_idx = np.argmax(fft_positive)
            features.append(freqs_positive[dominant_idx])   # 주요 주파수

            # 스펙트럼 엔트로피
            psd = fft_positive**2
            psd_norm = psd / (np.sum(psd) + 1e-8)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
            features.append(spectral_entropy)

            # 스펙트럼 중심 (spectral centroid)
            spectral_centroid = np.sum(freqs_positive * fft_positive) / (np.sum(fft_positive) + 1e-8)
            features.append(spectral_centroid)

        return np.array(features)

    def extract_all_features(self, window):
        """
        시간 + 주파수 도메인 특징 결합
        
        Args:
            window: (window_size, n_subcarriers) 배열
        
        Returns:
            1D 특징 벡터
        """
        time_feats = self.extract_time_features(window)
        freq_feats = self.extract_freq_features(window)
        return np.concatenate([time_feats, freq_feats])

    def extract_from_sample(self, data, mode="all"):
        """
        전체 샘플에 대해 슬라이딩 윈도우 + 특징 추출
        
        Args:
            data: (n_samples, n_subcarriers) 배열
            mode: "time", "freq", "all"
        
        Returns:
            (n_windows, n_features) 배열
        """
        windows = self.sliding_window(data)

        if mode == "time":
            extract_func = self.extract_time_features
        elif mode == "freq":
            extract_func = self.extract_freq_features
        else:
            extract_func = self.extract_all_features

        features = [extract_func(w) for w in windows]
        return np.array(features)

    def get_raw_windows(self, data):
        """
        특징 추출 없이 슬라이딩 윈도우만 반환 (딥러닝용)
        
        Args:
            data: (n_samples, n_subcarriers) 배열
        
        Returns:
            (n_windows, window_size, n_subcarriers) 배열
        """
        windows = self.sliding_window(data)
        return np.array(windows)

    def get_feature_names(self, n_subcarriers=64, mode="all"):
        """특징 이름 목록 반환 (디버깅/시각화용)"""
        time_names = [
            "mean", "std", "var", "range", "median",
            "skew", "kurtosis", "energy",
            "mean_diff", "std_diff", "zero_crossings"
        ]
        freq_names = [
            "fft_max", "fft_mean", "dominant_freq",
            "spectral_entropy", "spectral_centroid"
        ]

        names = []
        if mode in ("time", "all"):
            for sc in range(n_subcarriers):
                for feat in time_names:
                    names.append(f"sc{sc}_{feat}")
        if mode in ("freq", "all"):
            for sc in range(n_subcarriers):
                for feat in freq_names:
                    names.append(f"sc{sc}_{feat}")

        return names


def run_feature_pipeline(input_dir, output_dir, mode="all",
                         normalize_method="minmax", prefix="features"):
    """Process raw CSI CSV files and save extracted features to disk."""
    from preprocessing.dataset import CSIDataset

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    dataset = CSIDataset(raw_dir=input_dir, processed_dir=output_dir)
    samples, labels = dataset.load_raw_data()

    if not samples:
        raise RuntimeError(f"No raw CSI samples found under: {input_dir}")

    processed = dataset.preprocess_all(samples, normalize_method=normalize_method)
    X, y = dataset.prepare_ml_dataset(processed, labels, mode=mode)
    dataset.save_processed(X, y, prefix=prefix)

    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names(mode=mode)

    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "input_dir": os.path.abspath(input_dir),
        "output_dir": os.path.abspath(output_dir),
        "mode": mode,
        "normalize_method": normalize_method,
        "samples": len(samples),
        "windows": int(len(y)),
        "num_features": int(X.shape[1]) if X.ndim == 2 and len(X) > 0 else 0,
        "classes": {label: labels.count(label) for label in sorted(set(labels))},
        "x_file": os.path.join(os.path.abspath(output_dir), f"{prefix}_X.npy"),
        "y_file": os.path.join(os.path.abspath(output_dir), f"{prefix}_y.npy"),
    }

    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    names_path = os.path.join(output_dir, f"{prefix}_feature_names.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    return summary, summary_path, names_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract CSI features from raw CSV files")
    parser.add_argument("--input", type=str, default=RAW_DATA_DIR,
                        help="input raw CSI directory")
    parser.add_argument("--output", type=str, default=PROCESSED_DATA_DIR,
                        help="output processed directory")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["time", "freq", "all"],
                        help="feature extraction mode")
    parser.add_argument("--normalize", type=str, default="minmax",
                        choices=["minmax", "zscore"],
                        help="normalization method")
    parser.add_argument("--prefix", type=str, default="features",
                        help="output filename prefix")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    summary, summary_path, names_path = run_feature_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        mode=args.mode,
        normalize_method=args.normalize,
        prefix=args.prefix,
    )

    print("=" * 60)
    print("Feature extraction completed")
    print("=" * 60)
    print(f"Input directory : {summary['input_dir']}")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Samples         : {summary['samples']}")
    print(f"Windows         : {summary['windows']}")
    print(f"Features/window : {summary['num_features']}")
    print(f"Saved summary   : {summary_path}")
    print(f"Saved names     : {names_path}")
    print(f"Saved features  : {summary['x_file']}")
    print(f"Saved labels    : {summary['y_file']}")


if __name__ == "__main__":
    main()


if False and __name__ == "__main__":
    print("=== 특징 추출 데모 ===")
    extractor = FeatureExtractor()

    # 테스트 데이터
    np.random.seed(42)
    data = np.random.randn(300, 64)

    # 슬라이딩 윈도우
    windows = extractor.sliding_window(data)
    print(f"입력: {data.shape} → 윈도우 {len(windows)}개 (크기 {windows[0].shape})")

    # 특징 추출
    features = extractor.extract_from_sample(data, mode="all")
    print(f"특징 벡터: {features.shape}")

    # 딥러닝용 원시 윈도우
    raw_windows = extractor.get_raw_windows(data)
    print(f"딥러닝용 원시 윈도우: {raw_windows.shape}")

    print("✅ 특징 추출 정상 동작")

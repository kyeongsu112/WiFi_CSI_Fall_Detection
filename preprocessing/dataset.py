"""
데이터셋 관리 모듈
CSV 파일 로드, 레이블 인코딩, 훈련/검증/테스트 분할
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ACTIVITY_CLASSES, CLASS_TO_IDX, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    WINDOW_SIZE, WINDOW_STRIDE, CSI_NUM_SUBCARRIERS
)
from preprocessing.signal_processor import SignalProcessor
from preprocessing.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class CSIDataset:
    """CSI 데이터셋 로드·처리·분할 클래스"""

    def __init__(self, raw_dir=None, processed_dir=None):
        self.raw_dir = raw_dir or RAW_DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DATA_DIR
        self.processor = SignalProcessor()
        self.extractor = FeatureExtractor()

    def load_raw_data(self):
        """
        원시 CSI 데이터 로드 (디렉토리 구조 기반)
        
        구조:
            raw_dir/
                walking/sample_000.csv
                sitting/sample_001.csv
                ...
        
        Returns:
            samples: list of np.array (n_samples, n_subcarriers)
            labels: list of str
        """
        samples = []
        labels = []

        for cls_name in ACTIVITY_CLASSES:
            cls_dir = os.path.join(self.raw_dir, cls_name)
            if not os.path.exists(cls_dir):
                logger.warning(f"디렉토리 없음: {cls_dir}")
                continue

            csv_files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".csv")])
            logger.info(f"  '{cls_name}': {len(csv_files)}개 파일")

            for fname in csv_files:
                filepath = os.path.join(cls_dir, fname)
                try:
                    df = pd.read_csv(filepath)
                    # sc_ 로 시작하는 컬럼만 추출 (서브캐리어 데이터)
                    sc_cols = [c for c in df.columns if c.startswith("sc_")]
                    if not sc_cols:
                        sc_cols = df.columns.tolist()
                    data = df[sc_cols].values.astype(np.float64)
                    samples.append(data)
                    labels.append(cls_name)
                except Exception as e:
                    logger.error(f"파일 로드 실패: {filepath} - {e}")

        logger.info(f"총 {len(samples)}개 샘플 로드 완료")
        return samples, labels

    def preprocess_all(self, samples, normalize_method="minmax"):
        """
        전체 샘플 전처리
        
        Args:
            samples: list of np.array
            normalize_method: "minmax" 또는 "zscore"
        
        Returns:
            list of 전처리된 np.array
        """
        processed = []
        for i, sample in enumerate(samples):
            try:
                proc = self.processor.preprocess_pipeline(sample, normalize_method)
                processed.append(proc)
            except Exception as e:
                logger.warning(f"전처리 실패 (샘플 {i}): {e}")
                processed.append(sample)  # 원본 유지
        return processed

    def prepare_ml_dataset(self, samples, labels, mode="all"):
        """
        전통 ML 분류기용 데이터셋 준비 (특징 추출)
        
        Args:
            samples: list of np.array (전처리 완료)
            labels: list of str
            mode: "time", "freq", "all"
        
        Returns:
            X: (n_total_windows, n_features)
            y: (n_total_windows,) 정수 레이블
        """
        all_features = []
        all_labels = []

        for sample, label in zip(samples, labels):
            features = self.extractor.extract_from_sample(sample, mode=mode)
            label_idx = CLASS_TO_IDX[label]

            for feat_vec in features:
                all_features.append(feat_vec)
                all_labels.append(label_idx)

        X = np.array(all_features)
        y = np.array(all_labels)

        # NaN/Inf 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"ML 데이터셋: X={X.shape}, y={y.shape}")
        return X, y

    def prepare_dl_dataset(self, samples, labels):
        """
        딥러닝 분류기용 데이터셋 준비 (원시 윈도우)
        
        Returns:
            X: (n_total_windows, window_size, n_subcarriers)
            y: (n_total_windows,) 정수 레이블
        """
        all_windows = []
        all_labels = []

        for sample, label in zip(samples, labels):
            windows = self.extractor.get_raw_windows(sample)
            label_idx = CLASS_TO_IDX[label]

            for window in windows:
                all_windows.append(window)
                all_labels.append(label_idx)

        X = np.array(all_windows)
        y = np.array(all_labels)

        logger.info(f"DL 데이터셋: X={X.shape}, y={y.shape}")
        return X, y

    def split_dataset(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
        """
        훈련/검증/테스트 분할 (70/15/15)
        
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # 먼저 훈련+검증 vs 테스트 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 훈련 vs 검증 분할
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )

        logger.info(f"데이터 분할: 훈련={len(y_train)}, 검증={len(y_val)}, 테스트={len(y_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_processed(self, X, y, prefix="dataset"):
        """전처리된 데이터 저장"""
        os.makedirs(self.processed_dir, exist_ok=True)
        np.save(os.path.join(self.processed_dir, f"{prefix}_X.npy"), X)
        np.save(os.path.join(self.processed_dir, f"{prefix}_y.npy"), y)
        logger.info(f"데이터 저장: {self.processed_dir}/{prefix}_*.npy")

    def load_processed(self, prefix="dataset"):
        """전처리된 데이터 로드"""
        X = np.load(os.path.join(self.processed_dir, f"{prefix}_X.npy"))
        y = np.load(os.path.join(self.processed_dir, f"{prefix}_y.npy"))
        return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = CSIDataset()
    samples, labels = dataset.load_raw_data()

    if len(samples) > 0:
        print(f"\n전체 샘플 수: {len(samples)}")
        for cls in ACTIVITY_CLASSES:
            count = labels.count(cls)
            print(f"  {cls}: {count}개")

        # 전처리
        processed = dataset.preprocess_all(samples)

        # ML 데이터셋
        X_ml, y_ml = dataset.prepare_ml_dataset(processed, labels)
        print(f"\nML 데이터셋: {X_ml.shape}")

        # DL 데이터셋
        X_dl, y_dl = dataset.prepare_dl_dataset(processed, labels)
        print(f"DL 데이터셋: {X_dl.shape}")

        # 분할
        splits = dataset.split_dataset(X_ml, y_ml)
        print("✅ 데이터셋 준비 완료")
    else:
        print("⚠️ 데이터가 없습니다. 먼저 csi_simulator.py로 데이터를 생성하세요.")

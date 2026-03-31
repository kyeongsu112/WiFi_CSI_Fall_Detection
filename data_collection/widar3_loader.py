"""
Widar 3.0 데이터셋 로더 (실제 구조 대응)

Widar 3.0 BVP 실제 구조:
    BVP/
        20181109-VS/
            6-link/
                user1/
                    user1-{room}-{orient}-{gesture}-{trial}-...-L0.mat
                user2/
                    ...

.mat 파일 내부:
    velocity_spectrum_ro: (20, 20, 20) — 3D BVP 텐서
    - 축 0: 시간 (20 프레임)
    - 축 1: 속도 빈 X (20)
    - 축 2: 속도 빈 Y (20)

Widar 3.0 Gesture ID 매핑:
    1: Push, 2: Pull, 3: Sweep(좌→우), 4: Clap
    5: Slide, 6: Draw-O, 7: Draw-Zigzag, 8: Zoom-in, 9: Zoom-out

참조: http://tns.thss.tsinghua.edu.cn/widar3.0/
"""
import sys
import os
import glob
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ACTIVITY_CLASSES, CLASS_TO_IDX, CSI_NUM_SUBCARRIERS,
    RAW_DATA_DIR
)

logger = logging.getLogger(__name__)

# Widar 3.0 Gesture ID → 우리 행동 클래스 매핑
# gesture ID는 파일명의 4번째 필드
GESTURE_TO_ACTIVITY = {
    1: "walking",   # Push (전진 동작 → 보행과 유사)
    2: "walking",   # Pull (후퇴 동작 → 보행과 유사)
    3: "walking",   # Sweep (팔 좌→우 → 큰 동작)
    4: "sitting",   # Clap (정적 상태 소동작)
    5: "walking",   # Slide (미끄러지듯 이동)
    6: "sitting",   # Draw-O (정적 상태, 손 원 그리기)
    7: "sitting",   # Draw-Zigzag (정적 상태, 손 지그재그)
    8: "lying",     # Zoom-in (매우 미세한 동작)
    9: "lying",     # Zoom-out (매우 미세한 동작)
}


class Widar3Loader:
    """Widar 3.0 BVP 데이터셋 로더"""

    def __init__(self, dataset_dir=None):
        self.dataset_dir = dataset_dir
        self.bvp_dir = None

        if dataset_dir and os.path.exists(dataset_dir):
            # BVP 디렉토리 찾기
            bvp_path = os.path.join(dataset_dir, "BVP")
            if os.path.exists(bvp_path):
                self.bvp_dir = bvp_path
            else:
                self.bvp_dir = dataset_dir  # 직접 BVP 디렉토리일 수도

    def load_bvp_mat_files(self, max_total=2000):
        """
        BVP .mat 파일 로드

        Args:
            max_total: 최대 로드 파일 수

        Returns:
            samples: list of np.array (n, 64) — 2D 변환된 BVP
            labels: list of str
            metadata: list of dict
        """
        import scipy.io

        if not self.bvp_dir:
            logger.error("BVP 디렉토리를 찾을 수 없습니다")
            return [], [], []

        mat_files = glob.glob(os.path.join(self.bvp_dir, "**", "*.mat"), recursive=True)
        logger.info(f"발견된 .mat 파일: {len(mat_files)}개")

        if not mat_files:
            return [], [], []

        # 셔플 (다양한 사용자/날짜에서 추출)
        rng = np.random.default_rng(42)
        rng.shuffle(mat_files)

        if max_total and len(mat_files) > max_total:
            mat_files = mat_files[:max_total]

        samples = []
        labels = []
        metadata = []

        for filepath in mat_files:
            try:
                # 파일명에서 gesture ID 추출
                fname = os.path.basename(filepath)
                gesture_id = self._extract_gesture_id(fname)
                if gesture_id is None:
                    continue

                # 행동 클래스 매핑
                label = GESTURE_TO_ACTIVITY.get(gesture_id, None)
                if label is None:
                    continue

                # .mat 파일 로드
                mat_data = scipy.io.loadmat(filepath)
                bvp_3d = mat_data.get("velocity_spectrum_ro", None)
                if bvp_3d is None:
                    continue

                # 3D BVP (20,20,20) → 2D (20, 400) → 리사이즈 (20, 64)
                data_2d = self._bvp_3d_to_2d(bvp_3d)
                if data_2d is not None:
                    samples.append(data_2d)
                    labels.append(label)
                    metadata.append({
                        "file": fname,
                        "gesture_id": gesture_id,
                        "source": "widar3_bvp"
                    })

            except Exception as e:
                logger.debug(f"로드 실패: {filepath} - {e}")

        logger.info(f"BVP 데이터 로드 완료: {len(samples)}개")
        for cls in ACTIVITY_CLASSES:
            cnt = labels.count(cls)
            logger.info(f"  {cls}: {cnt}개")

        return samples, labels, metadata

    def _extract_gesture_id(self, filename):
        """
        파일명에서 gesture ID 추출
        형식: user{ID}-{room}-{orient}-{gesture}-{trial}-...
        """
        parts = filename.replace('.mat', '').split('-')
        if len(parts) >= 4:
            try:
                return int(parts[3])
            except ValueError:
                return None
        return None

    def _bvp_3d_to_2d(self, bvp_3d):
        """
        3D BVP (20, 20, 20) → 2D (n_time, n_features)
        CSI 형식과 호환되도록 변환

        변환 전략:
        - 시간 축 유지 (20 프레임)
        - 속도 평면(20x20)을 flatten → 400차원
        - 서브캐리어 수(64)에 맞게 PCA/다운샘플
        """
        if bvp_3d.ndim != 3:
            return None

        n_time = bvp_3d.shape[0]

        # (20, 20, 20) → (20, 400) flatten
        data_flat = bvp_3d.reshape(n_time, -1)

        # 400 → 64 다운샘플 (균등 간격 선택)
        if data_flat.shape[1] > CSI_NUM_SUBCARRIERS:
            indices = np.linspace(0, data_flat.shape[1] - 1,
                                  CSI_NUM_SUBCARRIERS, dtype=int)
            data_out = data_flat[:, indices]
        elif data_flat.shape[1] < CSI_NUM_SUBCARRIERS:
            # 패딩
            data_out = np.pad(data_flat,
                              ((0, 0),
                               (0, CSI_NUM_SUBCARRIERS - data_flat.shape[1])))
        else:
            data_out = data_flat

        # 시간 축 확장 (20 → WINDOW_SIZE=100)
        # 보간으로 100 포인트로 확장
        from scipy.ndimage import zoom
        time_scale = 100 / n_time  # 20 → 100 (5배)
        data_out = zoom(data_out, (time_scale, 1), order=1)

        return data_out.astype(np.float64)

    def synthesize_falling(self, samples, labels, target_count=200):
        """
        정상 데이터에서 낙상(falling) 데이터 합성

        방법: 기존 동작 데이터에 급격한 스파이크 + 이후 정적를 주입
        """
        existing_fall = labels.count("falling")
        need = max(0, target_count - existing_fall)
        if need == 0:
            return samples, labels

        rng = np.random.default_rng(42)
        non_fall = [s for s, l in zip(samples, labels) if l != "falling"]

        if not non_fall:
            return samples, labels

        logger.info(f"합성 낙상 데이터 {need}개 생성 중...")

        for i in range(need):
            base = non_fall[rng.integers(0, len(non_fall))].copy()
            n_rows = base.shape[0]

            # 낙상 시점 (30-50%)
            fall_pt = int(n_rows * rng.uniform(0.3, 0.5))
            spike_len = max(1, int(n_rows * 0.1))

            # 스파이크 주입
            spike_end = min(n_rows, fall_pt + spike_len)
            base[fall_pt:spike_end] *= rng.uniform(3.0, 8.0)

            # 이후 정적
            if spike_end < n_rows:
                static = np.mean(base[:fall_pt]) * 0.2
                base[spike_end:] = static + rng.normal(0, 0.05, base[spike_end:].shape)

            samples.append(base)
            labels.append("falling")

        return samples, labels

    def prepare_dataset(self, max_per_class=200, max_total=2000):
        """
        Widar 3.0 → 프로젝트 형식 데이터셋 준비

        Returns:
            samples, labels
        """
        samples, labels, _ = self.load_bvp_mat_files(max_total=max_total)

        if not samples:
            logger.error("데이터를 로드할 수 없습니다")
            return [], []

        # 낙상 데이터 합성
        samples, labels = self.synthesize_falling(samples, labels, target_count=max_per_class)

        # 클래스 균형 맞추기
        balanced_s, balanced_l = [], []
        counts = {cls: 0 for cls in ACTIVITY_CLASSES}

        # 셔플
        indices = list(range(len(samples)))
        np.random.default_rng(42).shuffle(indices)

        for idx in indices:
            l = labels[idx]
            if l in ACTIVITY_CLASSES and counts[l] < max_per_class:
                balanced_s.append(samples[idx])
                balanced_l.append(l)
                counts[l] += 1

        logger.info(f"최종 데이터셋: {len(balanced_s)}개")
        for cls, cnt in counts.items():
            logger.info(f"  {cls}: {cnt}개")

        return balanced_s, balanced_l

    def save_as_project_format(self, samples, labels, output_dir=None):
        """프로젝트 CSV 형식으로 저장"""
        import pandas as pd

        output_dir = output_dir or os.path.join(RAW_DATA_DIR, "widar3")

        for i, (sample, label) in enumerate(zip(samples, labels)):
            cls_dir = os.path.join(output_dir, label)
            os.makedirs(cls_dir, exist_ok=True)

            columns = [f"sc_{j}" for j in range(sample.shape[1])]
            df = pd.DataFrame(sample, columns=columns)
            df.to_csv(os.path.join(cls_dir, f"widar_{i:04d}.csv"), index=False)

        logger.info(f"저장 완료: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Widar 3.0 데이터 로더")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    loader = Widar3Loader(dataset_dir=args.dataset_dir)
    samples, labels = loader.prepare_dataset(max_per_class=args.max_per_class)

    if samples:
        loader.save_as_project_format(samples, labels, output_dir=args.output)
        print(f"\n[완료] Widar 3.0 변환 완료: {len(samples)}개")
        for cls in ACTIVITY_CLASSES:
            print(f"  {cls}: {labels.count(cls)}개")

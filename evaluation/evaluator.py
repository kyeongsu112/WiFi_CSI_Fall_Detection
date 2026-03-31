"""
모델 평가 모듈
정확도, F1, 혼동 행렬, ROC 커브, 응답 시간 등 종합 평가
"""
import sys
import os
import time
import logging
import json
import tempfile

import numpy as np
_mpl_config_dir = tempfile.mkdtemp(prefix="codex-mpl-")
os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVITY_CLASSES, NUM_CLASSES, IDX_TO_CLASS, LOG_DIR

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


class ModelEvaluator:
    """모델 성능 종합 평가"""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(LOG_DIR, "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self, y_true, y_pred, y_proba=None, model_name="model"):
        """
        종합 평가 수행
        
        Returns:
            dict: 평가 결과
        """
        results = {}

        # 기본 메트릭
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
        results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
        results["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        results["precision"] = float(precision_score(y_true, y_pred, average="weighted"))
        results["recall"] = float(recall_score(y_true, y_pred, average="weighted"))

        # 클래스별 메트릭
        report = classification_report(y_true, y_pred,
                                        target_names=ACTIVITY_CLASSES,
                                        output_dict=True)
        results["class_report"] = report

        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()

        # 오탐률/미탐률 (falling 클래스 기준)
        falling_idx = ACTIVITY_CLASSES.index("falling")
        tp = cm[falling_idx, falling_idx]
        fn = sum(cm[falling_idx, :]) - tp
        fp = sum(cm[:, falling_idx]) - tp
        tn = cm.sum() - tp - fn - fp

        results["falling_fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0  # 오탐률
        results["falling_fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0  # 미탐률

        # 출력
        print(f"\n{'='*60}")
        print(f"  [EVAL] {model_name} 평가 결과")
        print(f"{'='*60}")
        print(f"  정확도:     {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
        print(f"  F1 (가중):  {results['f1_weighted']:.4f}")
        print(f"  F1 (매크):  {results['f1_macro']:.4f}")
        print(f"  정밀도:     {results['precision']:.4f}")
        print(f"  재현율:     {results['recall']:.4f}")
        print(f"  낙상 오탐률: {results['falling_fpr']:.4f} ({results['falling_fpr']*100:.1f}%)")
        print(f"  낙상 미탐률: {results['falling_fnr']:.4f} ({results['falling_fnr']*100:.1f}%)")

        # 시각화
        self.plot_confusion_matrix(cm, model_name)
        if y_proba is not None:
            self.plot_roc_curve(y_true, y_proba, model_name)

        # 결과 저장
        result_path = os.path.join(self.output_dir, f"{model_name}_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        return results

    def plot_confusion_matrix(self, cm, model_name="model"):
        """혼동 행렬 시각화"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=ACTIVITY_CLASSES,
                    yticklabels=ACTIVITY_CLASSES,
                    ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_xlabel("예측값", fontsize=12)
        ax.set_ylabel("실제값", fontsize=12)
        ax.set_title(f"{model_name} 혼동 행렬", fontsize=14)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        logger.info(f"혼동 행렬 저장: {filepath}")

    def plot_roc_curve(self, y_true, y_proba, model_name="model"):
        """ROC 커브 시각화"""
        y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#00e676", "#4f8cff", "#ffa726", "#ff4757"]

        for i, (cls, color) in enumerate(zip(ACTIVITY_CLASSES, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{cls} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"{model_name} ROC 커브", fontsize=14)
        ax.legend(loc="lower right")
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f"{model_name}_roc_curve.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        logger.info(f"ROC 커브 저장: {filepath}")

    def measure_inference_time(self, model, X_sample, n_iterations=100):
        """추론 시간 측정"""
        times = []
        for _ in range(n_iterations):
            start = time.time()
            if hasattr(model, "predict_proba"):
                model.predict_proba(X_sample)
            else:
                model.predict(X_sample)
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # ms

        result = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "within_3sec": bool(np.mean(times) < 3000)
        }

        print(f"\n[TIME] 추론 시간 ({n_iterations}회)")
        print(f"  평균: {result['mean_ms']:.2f}ms")
        print(f"  P95:  {result['p95_ms']:.2f}ms")
        print(f"  3초 이내: {'PASS' if result['within_3sec'] else 'FAIL'}")

        return result

    def generate_report(self, all_results, output_file=None):
        """평가 보고서 생성"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "evaluation_report.md")

        lines = [
            "# 모델 평가 보고서",
            f"",
            f"## 모델 비교",
            "",
            "| 모델 | 정확도 | F1 (가중) | 오탐률 | 미탐률 |",
            "|------|--------|----------|--------|--------|",
        ]

        for name, res in all_results.items():
            lines.append(
                f"| {name} | {res['accuracy']:.4f} | {res['f1_weighted']:.4f} | "
                f"{res.get('falling_fpr', 0):.4f} | {res.get('falling_fnr', 0):.4f} |"
            )

        lines.append("")
        lines.append("## 목표 달성 여부")
        lines.append("")

        for name, res in all_results.items():
            acc_ok = "✅" if res["accuracy"] >= 0.85 else "❌"
            fpr_ok = "✅" if res.get("falling_fpr", 1) <= 0.15 else "❌"
            fnr_ok = "✅" if res.get("falling_fnr", 1) <= 0.15 else "❌"
            lines.append(f"### {name}")
            lines.append(f"- {acc_ok} 정확도 ≥ 85%: {res['accuracy']*100:.1f}%")
            lines.append(f"- {fpr_ok} 오탐률 ≤ 15%: {res.get('falling_fpr', 0)*100:.1f}%")
            lines.append(f"- {fnr_ok} 미탐률 ≤ 15%: {res.get('falling_fnr', 0)*100:.1f}%")
            lines.append("")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"보고서 저장: {output_file}")
        return output_file

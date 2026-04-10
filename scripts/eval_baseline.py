from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import yaml

from training.model import load_model
from training.trainer import build_datasets, evaluate


def _save_confusion_matrix_png(cm: list[list[int]], path: Path) -> None:
    """Save a simple confusion-matrix heatmap using matplotlib only."""
    cm_array = np.array(cm, dtype=int)
    labels = ["non_fall (0)", "fall (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    thresh = cm_array.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm_array[i, j]),
                ha="center", va="center",
                color="white" if cm_array[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained fall detector baseline")
    parser.add_argument(
        "--config",
        default="configs/training_baseline.yaml",
        help="Path to baseline training config YAML (default: configs/training_baseline.yaml)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model checkpoint (overrides config output.model_path)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_path = Path(args.model) if args.model else Path(cfg["output"]["model_path"])
    metrics_path = Path(cfg["output"]["metrics_path"])
    predictions_path = Path(cfg["output"]["predictions_path"])
    confusion_matrix_path = Path(cfg["output"]["confusion_matrix_path"])

    # Ensure output dirs
    for p in (metrics_path, predictions_path, confusion_matrix_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}")
    model, metadata = load_model(model_path)

    # Build test dataset (target_subject)
    _, test_dataset = build_datasets(cfg)
    print(f"Test set size: {len(test_dataset)} windows")

    # Evaluate
    metrics = evaluate(model, test_dataset, cfg)

    # Print metrics table
    print("\n--- Evaluation Metrics ---")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    # Resolve target_subject via split YAML (single source of truth) or inline key
    _split_yaml_path = cfg["data"].get("split_yaml_path")
    if _split_yaml_path:
        import yaml as _yaml
        with open(_split_yaml_path, encoding="utf-8") as _f:
            target_subject = _yaml.safe_load(_f)["target_subject"]
    else:
        target_subject = cfg["data"]["target_subject"]

    # Save metrics JSON
    metrics_to_save = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "model_path": str(model_path),
        "target_subject": target_subject,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Save predictions CSV
    raw_preds = metrics["raw_predictions"]
    if raw_preds:
        fieldnames = ["source_file", "window_index", "true_label", "pred_prob", "pred_label"]
        with predictions_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_preds)
        print(f"Predictions saved to {predictions_path}")

    # Save confusion matrix PNG
    _save_confusion_matrix_png(cm, confusion_matrix_path)
    print(f"Confusion matrix plot saved to {confusion_matrix_path}")


if __name__ == "__main__":
    main()

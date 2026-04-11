"""Evaluate a trained WiFall baseline model.

Outputs
-------
Per-run artifacts (paths set in configs/training_baseline.yaml):
  wifall_baseline_metrics.json          overall metrics at default + best threshold
  wifall_baseline_predictions.csv       per-window scores and hard labels
  wifall_baseline_confusion_matrix.png  confusion matrix heatmap at default threshold
  baseline_threshold_sweep.csv          precision / recall / F1 at 99 thresholds
  baseline_roc_curve.png                ROC curve with AUC annotation
  baseline_pr_curve.png                 Precision-Recall curve with AP annotation
  baseline_score_distribution.csv       raw (prob, label) rows for external analysis
  baseline_score_histogram.png          fall vs non-fall score histograms

Usage
-----
    python scripts/eval_baseline.py
    python scripts/eval_baseline.py --config configs/training_baseline.yaml
    python scripts/eval_baseline.py --model artifacts/models/wifall_baseline.pt
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on Windows without a display
import matplotlib.pyplot as plt
import numpy as np
import yaml

from training.model import load_model
from training.trainer import build_datasets, evaluate


# ── small pure-Python / numpy helpers ────────────────────────────────────────

def _metrics_at_threshold(
    probs: list[float],
    labels: list[int],
    threshold: float,
) -> dict[str, Any]:
    """Compute binary classification metrics at a single threshold."""
    tn = fp = fn = tp = 0
    for p, y in zip(probs, labels):
        pred = 1 if p >= threshold else 0
        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    total = len(labels)
    acc  = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "threshold": round(threshold, 4),
        "precision": round(prec, 6),
        "recall":    round(rec,  6),
        "f1":        round(f1,   6),
        "accuracy":  round(acc,  6),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _threshold_sweep(
    probs: list[float],
    labels: list[int],
    n_steps: int = 99,
) -> list[dict[str, Any]]:
    """Evaluate metrics at n_steps evenly-spaced thresholds in [0.01, 0.99]."""
    thresholds = np.linspace(0.01, 0.99, n_steps).tolist()
    return [_metrics_at_threshold(probs, labels, t) for t in thresholds]


def _best_threshold_row(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the sweep row with highest F1.

    Ties broken by lowest threshold to favour recall — useful when the model
    produces very low fall probabilities (exactly the weak-recall scenario).

    Note: this selection is performed on the *test* subject. Use it to guide
    threshold choices for Step 3, not as a final unbiased estimate.
    """
    return max(sweep, key=lambda r: (r["f1"], -r["threshold"]))


def _compute_roc(
    probs: list[float],
    labels: list[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve using sorted probability values as thresholds.

    Returns (fpr, tpr, auc_roc).
    """
    pa = np.asarray(probs, dtype=np.float64)
    la = np.asarray(labels, dtype=np.int32)
    n_pos = int(la.sum())
    n_neg = len(la) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    sort_idx   = np.argsort(-pa)          # descending probability
    sl         = la[sort_idx]
    cum_tp     = np.cumsum(sl).astype(float)
    cum_fp     = np.cumsum(1 - sl).astype(float)

    tpr = np.concatenate([[0.0], cum_tp / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], cum_fp / n_neg, [1.0]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _compute_pr(
    probs: list[float],
    labels: list[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Precision-Recall curve using sorted probability values as thresholds.

    Returns (recall, precision, average_precision).
    """
    pa = np.asarray(probs, dtype=np.float64)
    la = np.asarray(labels, dtype=np.int32)
    n_pos = int(la.sum())
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    sort_idx = np.argsort(-pa)
    sl       = la[sort_idx]
    cum_tp   = np.cumsum(sl).astype(float)
    cum_all  = np.arange(1, len(sl) + 1, dtype=float)

    prec = cum_tp / cum_all           # precision at each rank
    rec  = cum_tp / n_pos             # recall at each rank

    # Prepend interpolation boundary (recall=0, precision=1)
    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])

    # Average Precision = area under PR curve (recall is monotone increasing)
    ap = float(np.trapezoid(prec, rec))
    return rec, prec, ap


# ── plot helpers ──────────────────────────────────────────────────────────────

def _save_confusion_matrix_png(
    cm: list[list[int]],
    path: Path,
    threshold: float,
) -> None:
    cm_arr = np.array(cm, dtype=int)
    labels = ["non_fall (0)", "fall (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=15)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix  (threshold={threshold:.3f})")
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm_arr[i, j] > thresh else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_roc_curve_png(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — WiFall Baseline")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_pr_curve_png(
    recall: np.ndarray,
    precision: np.ndarray,
    ap: float,
    n_pos: int,
    n_total: int,
    path: Path,
) -> None:
    baseline = n_pos / n_total if n_total > 0 else 0.0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(baseline, color="k", ls="--", lw=1,
               label=f"Random ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — WiFall Baseline")
    ax.legend(loc="upper right")
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_score_histogram_png(
    probs: list[float],
    labels: list[int],
    default_threshold: float,
    best_threshold: float,
    path: Path,
) -> None:
    fall_scores     = [p for p, y in zip(probs, labels) if y == 1]
    non_fall_scores = [p for p, y in zip(probs, labels) if y == 0]
    bins = np.linspace(0.0, 1.0, 51)   # 50 bins of width 0.02
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(non_fall_scores, bins=bins, alpha=0.6, color="steelblue",
            label=f"non_fall  (n={len(non_fall_scores)})", density=True)
    ax.hist(fall_scores, bins=bins, alpha=0.6, color="tomato",
            label=f"fall  (n={len(fall_scores)})", density=True)
    ax.axvline(default_threshold, color="navy", ls="--", lw=1.5,
               label=f"default threshold ({default_threshold:.3f})")
    ax.axvline(best_threshold, color="darkred", ls=":", lw=1.5,
               label=f"best F1 threshold ({best_threshold:.3f})")
    ax.set_xlabel("Predicted probability (fall)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by True Label")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── CSV writers ───────────────────────────────────────────────────────────────

def _save_threshold_sweep_csv(
    sweep: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "precision", "recall", "f1",
                        "accuracy", "tp", "fp", "tn", "fn"],
        )
        writer.writeheader()
        writer.writerows(sweep)


def _save_score_distribution_csv(
    probs: list[float],
    labels: list[int],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_prob", "true_label"])
        writer.writeheader()
        for p, y in zip(probs, labels):
            writer.writerow({"pred_prob": round(p, 8), "true_label": y})


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained fall detector baseline with full diagnostics"
    )
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

    out = cfg["output"]
    model_path           = Path(args.model) if args.model else Path(out["model_path"])
    metrics_path         = Path(out["metrics_path"])
    predictions_path     = Path(out["predictions_path"])
    cm_path              = Path(out["confusion_matrix_path"])
    sweep_path           = Path(out["threshold_sweep_path"])
    roc_path             = Path(out["roc_curve_path"])
    pr_path              = Path(out["pr_curve_path"])
    score_dist_path      = Path(out["score_distribution_path"])
    score_hist_path      = Path(out["score_histogram_path"])

    default_threshold = cfg["training"].get("threshold", 0.5)

    # ── ensure output directories exist ──────────────────────────────────────
    for p in (metrics_path, predictions_path, cm_path, sweep_path,
              roc_path, pr_path, score_dist_path, score_hist_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    # ── resolve target_subject ────────────────────────────────────────────────
    split_yaml_path = cfg["data"].get("split_yaml_path")
    if split_yaml_path:
        with open(split_yaml_path, encoding="utf-8") as _f:
            target_subject = yaml.safe_load(_f)["target_subject"]
    else:
        target_subject = cfg["data"]["target_subject"]

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {model_path}")
    model, metadata = load_model(model_path)

    # ── build test dataset ────────────────────────────────────────────────────
    _, test_dataset = build_datasets(cfg)
    n_test = len(test_dataset)
    print(f"Test set size: {n_test} windows  (subject {target_subject})")

    # ── run inference once — reuse scores for all diagnostics ─────────────────
    eval_result   = evaluate(model, test_dataset, cfg)
    all_probs     = eval_result["all_probs"]
    all_labels    = eval_result["all_labels"]

    n_pos = sum(all_labels)
    n_neg = n_test - n_pos
    print(f"  fall windows     : {n_pos}")
    print(f"  non-fall windows : {n_neg}")

    # ── metrics at default threshold ──────────────────────────────────────────
    metrics_default = _metrics_at_threshold(all_probs, all_labels, default_threshold)
    print(f"\n--- Metrics at default threshold ({default_threshold:.3f}) ---")
    print(f"  Recall    : {metrics_default['recall']:.4f}")
    print(f"  Precision : {metrics_default['precision']:.4f}")
    print(f"  F1        : {metrics_default['f1']:.4f}")
    print(f"  Accuracy  : {metrics_default['accuracy']:.4f}")
    cm = eval_result["confusion_matrix"]
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    # ── score summary (quick diagnostic) ─────────────────────────────────────
    fall_probs     = [p for p, y in zip(all_probs, all_labels) if y == 1]
    nf_probs       = [p for p, y in zip(all_probs, all_labels) if y == 0]
    if fall_probs:
        print(f"\n  Fall score stats     - "
              f"min={min(fall_probs):.4f}  "
              f"median={float(np.median(fall_probs)):.4f}  "
              f"max={max(fall_probs):.4f}")
    if nf_probs:
        print(f"  Non-fall score stats - "
              f"min={min(nf_probs):.4f}  "
              f"median={float(np.median(nf_probs)):.4f}  "
              f"max={max(nf_probs):.4f}")

    # ── threshold sweep ───────────────────────────────────────────────────────
    print("\nRunning threshold sweep (0.01 -> 0.99, 99 steps)...")
    sweep = _threshold_sweep(all_probs, all_labels)
    best_row = _best_threshold_row(sweep)
    best_threshold = best_row["threshold"]

    print(f"\n--- Best threshold by F1 (test-set diagnostic) ---")
    print(f"  Threshold : {best_threshold:.4f}")
    print(f"  Recall    : {best_row['recall']:.4f}")
    print(f"  Precision : {best_row['precision']:.4f}")
    print(f"  F1        : {best_row['f1']:.4f}")
    print(f"  TN={best_row['tn']}  FP={best_row['fp']}  "
          f"FN={best_row['fn']}  TP={best_row['tp']}")
    print(f"  (Note: threshold selected on test subject - use as a diagnostic guide,")
    print(f"   not as an unbiased estimate. Consider a dedicated val split for Step 3.)")

    _save_threshold_sweep_csv(sweep, sweep_path)
    print(f"\nThreshold sweep  -> {sweep_path}")

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, auc_roc = _compute_roc(all_probs, all_labels)
    print(f"AUC-ROC          : {auc_roc:.4f}")
    _save_roc_curve_png(fpr, tpr, auc_roc, roc_path)
    print(f"ROC curve        -> {roc_path}")

    # ── PR curve ──────────────────────────────────────────────────────────────
    rec_arr, prec_arr, ap = _compute_pr(all_probs, all_labels)
    print(f"Average Precision: {ap:.4f}")
    _save_pr_curve_png(rec_arr, prec_arr, ap, n_pos, n_test, pr_path)
    print(f"PR curve         -> {pr_path}")

    # ── score distribution ────────────────────────────────────────────────────
    _save_score_distribution_csv(all_probs, all_labels, score_dist_path)
    print(f"Score dist CSV   -> {score_dist_path}")
    _save_score_histogram_png(
        all_probs, all_labels,
        default_threshold, best_threshold,
        score_hist_path,
    )
    print(f"Score histogram  -> {score_hist_path}")

    # ── confusion matrix at default threshold ─────────────────────────────────
    _save_confusion_matrix_png(cm, cm_path, default_threshold)
    print(f"Confusion matrix -> {cm_path}")

    # ── per-window predictions CSV ────────────────────────────────────────────
    raw_preds = eval_result["raw_predictions"]
    if raw_preds:
        fieldnames = ["source_file", "window_index", "true_label",
                      "pred_prob", "pred_label"]
        with predictions_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_preds)
        print(f"Predictions CSV  -> {predictions_path}")

    # ── metrics JSON (both threshold operating points) ────────────────────────
    metrics_best = _metrics_at_threshold(all_probs, all_labels, best_threshold)

    metrics_to_save = {
        "target_subject":     target_subject,
        "model_path":         str(model_path),
        "n_test_windows":     n_test,
        "n_fall_windows":     n_pos,
        "n_non_fall_windows": n_neg,
        # ── curve metrics ────────────────────────────────────────────────────
        "auc_roc":             round(auc_roc, 6),
        "average_precision":   round(ap,      6),
        # ── default threshold ─────────────────────────────────────────────────
        "threshold_default":  default_threshold,
        "metrics_at_default": {
            "recall":    metrics_default["recall"],
            "precision": metrics_default["precision"],
            "f1":        metrics_default["f1"],
            "accuracy":  metrics_default["accuracy"],
            "confusion_matrix": cm,
        },
        # ── best threshold (test-set diagnostic) ─────────────────────────────
        "threshold_selected":      best_threshold,
        "threshold_selection_note": (
            "Best F1 on the test subject. "
            "Use as a diagnostic guide only — not an unbiased estimate."
        ),
        "metrics_at_selected": {
            "recall":    metrics_best["recall"],
            "precision": metrics_best["precision"],
            "f1":        metrics_best["f1"],
            "accuracy":  metrics_best["accuracy"],
            "confusion_matrix": [
                [metrics_best["tn"], metrics_best["fp"]],
                [metrics_best["fn"], metrics_best["tp"]],
            ],
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    print(f"Metrics JSON     -> {metrics_path}")

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n========== Evaluation Summary ==========")
    print(f"  AUC-ROC           : {auc_roc:.4f}")
    print(f"  Avg Precision     : {ap:.4f}")
    print(f"  Default threshold : {default_threshold:.3f}  "
          f"->  F1={metrics_default['f1']:.4f}  "
          f"recall={metrics_default['recall']:.4f}")
    print(f"  Best F1 threshold : {best_threshold:.3f}  "
          f"->  F1={best_row['f1']:.4f}  "
          f"recall={best_row['recall']:.4f}")
    print("=========================================")


if __name__ == "__main__":
    main()

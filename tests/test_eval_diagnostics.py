"""Unit tests for eval_baseline diagnostic helpers.

Covers the pure-numeric functions: _metrics_at_threshold, _threshold_sweep,
_best_threshold_row, _compute_roc, _compute_pr.  No model, no ZIP, no disk I/O.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prepare_wifall  # noqa: F401 — ensures project root on sys.path
from eval_baseline import (
    _best_threshold_row,
    _compute_pr,
    _compute_roc,
    _metrics_at_threshold,
    _threshold_sweep,
)

import numpy as np
import pytest


# ── _metrics_at_threshold ─────────────────────────────────────────────────────

class TestMetricsAtThreshold:
    PROBS  = [0.9, 0.8, 0.2, 0.1]
    LABELS = [1,   1,   0,   0  ]

    def test_perfect_classifier(self):
        m = _metrics_at_threshold(self.PROBS, self.LABELS, 0.5)
        assert m["tp"] == 2 and m["tn"] == 2
        assert m["fp"] == 0 and m["fn"] == 0
        assert m["recall"]    == 1.0
        assert m["precision"] == 1.0
        assert m["f1"]        == 1.0

    def test_threshold_below_all_scores_predicts_all_positive(self):
        m = _metrics_at_threshold(self.PROBS, self.LABELS, 0.05)
        assert m["tp"] == 2 and m["fp"] == 2
        assert m["tn"] == 0 and m["fn"] == 0
        assert m["recall"]    == 1.0
        assert m["precision"] == 0.5

    def test_threshold_above_all_scores_predicts_all_negative(self):
        m = _metrics_at_threshold(self.PROBS, self.LABELS, 0.99)
        assert m["tp"] == 0 and m["fp"] == 0
        assert m["tn"] == 2 and m["fn"] == 2
        assert m["recall"]    == 0.0
        assert m["precision"] == 0.0
        assert m["f1"]        == 0.0

    def test_output_keys_present(self):
        m = _metrics_at_threshold(self.PROBS, self.LABELS, 0.5)
        for key in ("threshold", "precision", "recall", "f1",
                    "accuracy", "tp", "fp", "tn", "fn"):
            assert key in m, f"Missing key: {key}"

    def test_confusion_matrix_sums_to_total(self):
        m = _metrics_at_threshold(self.PROBS, self.LABELS, 0.5)
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == len(self.LABELS)


# ── _threshold_sweep ──────────────────────────────────────────────────────────

class TestThresholdSweep:
    PROBS  = [0.9, 0.8, 0.2, 0.1]
    LABELS = [1,   1,   0,   0  ]

    def test_returns_99_rows_by_default(self):
        sweep = _threshold_sweep(self.PROBS, self.LABELS)
        assert len(sweep) == 99

    def test_thresholds_increasing(self):
        sweep = _threshold_sweep(self.PROBS, self.LABELS)
        thresholds = [r["threshold"] for r in sweep]
        assert thresholds == sorted(thresholds)

    def test_first_threshold_approx_0_01(self):
        sweep = _threshold_sweep(self.PROBS, self.LABELS)
        assert abs(sweep[0]["threshold"] - 0.01) < 0.001

    def test_last_threshold_approx_0_99(self):
        sweep = _threshold_sweep(self.PROBS, self.LABELS)
        assert abs(sweep[-1]["threshold"] - 0.99) < 0.001

    def test_each_row_has_required_keys(self):
        sweep = _threshold_sweep(self.PROBS, self.LABELS)
        for row in sweep:
            for key in ("threshold", "precision", "recall", "f1",
                        "accuracy", "tp", "fp", "tn", "fn"):
                assert key in row


# ── _best_threshold_row ───────────────────────────────────────────────────────

class TestBestThresholdRow:
    def test_returns_highest_f1(self):
        sweep = [
            {"threshold": 0.1, "f1": 0.5, "recall": 1.0, "precision": 0.33},
            {"threshold": 0.5, "f1": 0.8, "recall": 0.8, "precision": 0.8},
            {"threshold": 0.9, "f1": 0.4, "recall": 0.3, "precision": 1.0},
        ]
        best = _best_threshold_row(sweep)
        assert best["threshold"] == 0.5
        assert best["f1"] == 0.8

    def test_ties_broken_by_lower_threshold(self):
        # Two rows with equal F1 — prefer the lower threshold (higher recall)
        sweep = [
            {"threshold": 0.1, "f1": 0.8},
            {"threshold": 0.5, "f1": 0.8},
        ]
        best = _best_threshold_row(sweep)
        assert best["threshold"] == 0.1

    def test_works_with_full_sweep(self):
        probs  = [0.9, 0.8, 0.2, 0.1]
        labels = [1,   1,   0,   0  ]
        sweep = _threshold_sweep(probs, labels)
        best  = _best_threshold_row(sweep)
        assert best["f1"] == 1.0


# ── _compute_roc ──────────────────────────────────────────────────────────────

class TestComputeRoc:
    def test_perfect_classifier_auc_1(self):
        probs  = [0.9, 0.8, 0.2, 0.1]
        labels = [1,   1,   0,   0  ]
        _, _, auc = _compute_roc(probs, labels)
        assert abs(auc - 1.0) < 1e-9

    def test_worst_classifier_auc_0(self):
        # Model ranks every positive below every negative
        probs  = [0.1, 0.2, 0.8, 0.9]
        labels = [1,   1,   0,   0  ]
        _, _, auc = _compute_roc(probs, labels)
        assert abs(auc - 0.0) < 1e-9

    def test_auc_between_0_and_1(self):
        rng = np.random.default_rng(42)
        probs  = rng.random(100).tolist()
        labels = (rng.random(100) > 0.7).astype(int).tolist()
        _, _, auc = _compute_roc(probs, labels)
        assert 0.0 <= auc <= 1.0

    def test_zero_positives_returns_fallback(self):
        fpr, tpr, auc = _compute_roc([0.3, 0.7], [0, 0])
        assert auc == 0.5

    def test_zero_negatives_returns_fallback(self):
        fpr, tpr, auc = _compute_roc([0.3, 0.7], [1, 1])
        assert auc == 0.5

    def test_output_starts_at_origin(self):
        probs  = [0.9, 0.8, 0.2, 0.1]
        labels = [1,   1,   0,   0  ]
        fpr, tpr, _ = _compute_roc(probs, labels)
        assert fpr[0] == 0.0 and tpr[0] == 0.0

    def test_output_ends_at_one_one(self):
        probs  = [0.9, 0.8, 0.2, 0.1]
        labels = [1,   1,   0,   0  ]
        fpr, tpr, _ = _compute_roc(probs, labels)
        assert fpr[-1] == 1.0 and tpr[-1] == 1.0


# ── _compute_pr ───────────────────────────────────────────────────────────────

class TestComputePr:
    def test_perfect_classifier_ap_1(self):
        probs  = [0.9, 0.8, 0.2, 0.1]
        labels = [1,   1,   0,   0  ]
        _, _, ap = _compute_pr(probs, labels)
        assert abs(ap - 1.0) < 1e-9

    def test_ap_between_0_and_1(self):
        rng = np.random.default_rng(0)
        probs  = rng.random(80).tolist()
        labels = (rng.random(80) > 0.75).astype(int).tolist()
        _, _, ap = _compute_pr(probs, labels)
        assert 0.0 <= ap <= 1.0

    def test_zero_positives_returns_fallback(self):
        _, _, ap = _compute_pr([0.3, 0.7], [0, 0])
        assert ap == 0.0

    def test_output_starts_at_zero_recall(self):
        probs  = [0.9, 0.8, 0.2, 0.1]
        labels = [1,   1,   0,   0  ]
        rec, prec, _ = _compute_pr(probs, labels)
        assert rec[0] == 0.0 and prec[0] == 1.0

    def test_recall_is_monotone_increasing(self):
        rng = np.random.default_rng(7)
        probs  = rng.random(50).tolist()
        labels = (rng.random(50) > 0.6).astype(int).tolist()
        rec, _, _ = _compute_pr(probs, labels)
        assert all(rec[i] <= rec[i + 1] for i in range(len(rec) - 1))

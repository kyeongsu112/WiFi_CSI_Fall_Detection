from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json

import yaml

from datasets.loader import subject_window_distribution
from training.model import save_model
from training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D-CNN fall detector baseline")
    parser.add_argument(
        "--config",
        default="configs/training_baseline.yaml",
        help="Path to baseline training config YAML (default: configs/training_baseline.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ensure output directories exist
    model_path = Path(cfg["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg["output"]["metrics_path"]).parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Print subject/window distribution
    manifest_path = cfg["data"]["manifest_path"]
    print("Subject / window distribution")
    print("-" * 50)
    dist_df = subject_window_distribution(manifest_path)
    print(dist_df.to_string(index=False))
    print()

    # Save distribution CSV
    dist_csv_path = Path(cfg["output"]["distribution_path"])
    dist_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"Distribution saved to {dist_csv_path}")

    # Train
    print("\nStarting training...")
    results = train(cfg)
    model = results["model"]

    # Save model
    save_model(model, model_path, cfg)
    print(f"\nModel saved to {model_path}")

    # Print final summary
    loss_history = results["train_loss_history"]
    best_epoch = results["best_epoch"]
    if loss_history:
        final_loss = loss_history[-1]
        best_loss = min(loss_history)
        print("\n--- Training Summary ---")
        print(f"  Epochs completed : {len(loss_history)}")
        print(f"  Final train loss : {final_loss:.4f}")
        print(f"  Best train loss  : {best_loss:.4f}  (epoch {best_epoch})")
    else:
        print("\nTraining did not produce any loss values.")

    print("\nDone. Run scripts/eval_baseline.py to evaluate on the test subject.")


if __name__ == "__main__":
    main()

"""Parse WiFall.zip and produce a window-level manifest CSV, summary JSON, and split YAML.

Outputs
-------
artifacts/processed/wifall_manifest.csv
    One row per 1-second window. Columns:
      subject_id, activity_label, binary_label, source_file,
      window_index, start_row, end_row, num_rows

artifacts/processed/wifall_summary.json
    Dataset statistics. n_windows_total == len(manifest).

configs/splits/wifall_subject_split.yaml
    Subject-out split config consumed by training and KNN-MMD scripts.

Usage
-----
    python scripts/prepare_wifall.py
    python scripts/prepare_wifall.py --zip data/WiFall.zip --split-target ID1
    python scripts/prepare_wifall.py --config configs/dataset.yaml --output-dir artifacts/processed
"""
from __future__ import annotations

# ── path bootstrap (allows running as `python scripts/prepare_wifall.py`
#    even when the package is not installed via pip) ─────────────────────
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml

from datasets.loader import count_csi_rows, N_SUBCARRIERS
from datasets.manifest import parse_zip_entry, write_manifest
from preprocessing.windowing import compute_window_count

# ── constants ──────────────────────────────────────────────────────────────
WINDOW_SIZE: int = 100   # 1 second at 100 Hz
STRIDE: int = 100        # non-overlapping (stride == window_size)
SAMPLE_RATE_HZ: int = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── argument parsing ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse WiFall.zip and build window-level manifest, summary, and split config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/dataset.yaml",
                   help="Dataset config YAML.")
    p.add_argument("--zip", default=None, metavar="PATH",
                   help="Path to WiFall.zip (overrides config zip_path).")
    p.add_argument("--output-dir", default="artifacts/processed", metavar="DIR",
                   help="Destination for manifest CSV and summary JSON.")
    p.add_argument("--splits-dir", default="configs/splits", metavar="DIR",
                   help="Destination for the subject split YAML.")
    p.add_argument("--split-target", default=None, metavar="SUBJECT",
                   help="Subject used as target (test) set, e.g. ID0 (overrides config).")
    return p


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: '{cfg_path}'. "
            "Run from the project root or pass --config explicitly."
        )
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── core processing ─────────────────────────────────────────────────────────

def process_zip(
    zip_path: Path,
    binary_mapping: dict[str, str],
) -> list[dict]:
    """Walk WiFall.zip and emit one manifest row per 1-second window.

    Each source CSV is split into non-overlapping windows of WINDOW_SIZE rows.
    start_row and end_row are 0-indexed data-row offsets (after the CSV header),
    following Python slice convention: csi[start_row:end_row].

    Args:
        zip_path:       path to WiFall.zip.
        binary_mapping: lowercase activity → 'fall' | 'non_fall'.

    Returns:
        List of window-level manifest dicts. len() equals total window count.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"WiFall.zip not found at '{zip_path}'.\n"
            "  Download the dataset and place it at that path,\n"
            "  or pass --zip <path> to override."
        )

    rows: list[dict] = []
    skipped_unmatched = 0
    skipped_error = 0
    skipped_short = 0

    with zipfile.ZipFile(zip_path) as zf:
        csv_entries = [n for n in zf.namelist() if n.endswith(".csv")]
        log.info("Scanning %d CSV entries in %s ...", len(csv_entries), zip_path.name)

        for entry in csv_entries:
            # ── parse path ──────────────────────────────────────────────
            parsed = parse_zip_entry(entry)
            if parsed is None:
                skipped_unmatched += 1
                continue
            subject_id, activity_label = parsed   # activity_label is lowercased
            label = binary_mapping.get(activity_label, "non_fall")

            # ── count rows (fast path — no I/Q parsing needed here) ────
            try:
                with zf.open(entry) as f:
                    n_samples = count_csi_rows(f)
            except Exception as exc:
                log.warning("  SKIP %s -- count error: %s", entry, exc)
                skipped_error += 1
                continue

            # ── compute window count ────────────────────────────────────
            n_windows = compute_window_count(n_samples, WINDOW_SIZE, STRIDE)
            if n_windows == 0:
                log.warning(
                    "  SKIP %s -- only %d samples, need at least %d for one window",
                    entry, n_samples, WINDOW_SIZE,
                )
                skipped_short += 1
                continue

            # ── emit one row per window ─────────────────────────────────
            for win_idx in range(n_windows):
                start_row = win_idx * STRIDE
                end_row   = start_row + WINDOW_SIZE
                rows.append({
                    "subject_id":     subject_id,
                    "activity_label": activity_label,
                    "binary_label":   label,
                    "source_file":    entry,          # internal zip path
                    "window_index":   win_idx,
                    "start_row":      start_row,
                    "end_row":        end_row,
                    "num_rows":       WINDOW_SIZE,    # always 100 for full windows
                })

    log.info(
        "Results: %d windows from %d CSV entries | "
        "%d skipped (unmatched path) | %d skipped (error) | %d skipped (too short)",
        len(rows),
        len(csv_entries) - skipped_unmatched - skipped_error - skipped_short,
        skipped_unmatched, skipped_error, skipped_short,
    )
    return rows


def build_summary(rows: list[dict], zip_path: Path) -> dict:
    """Compute dataset statistics from window-level manifest rows.

    n_windows_total is derived directly from len(rows), so it always matches
    the manifest length exactly.
    """
    subjects       = sorted({r["subject_id"]     for r in rows})
    source_files   = {r["source_file"]           for r in rows}
    action_counts: dict[str, int] = defaultdict(int)
    label_counts:  dict[str, int] = defaultdict(int)

    for r in rows:
        action_counts[r["activity_label"]] += 1
        label_counts[r["binary_label"]]    += 1

    return {
        "zip_path":            str(zip_path),
        "n_subjects":          len(subjects),
        "subjects":            subjects,
        "n_files":             len(source_files),   # unique CSI files
        "n_windows_total":     len(rows),           # == len(manifest)
        "action_window_counts": dict(sorted(action_counts.items())),
        "label_distribution":  dict(label_counts),
        "n_subcarriers":       N_SUBCARRIERS,
        "window_size":         WINDOW_SIZE,
        "stride":              STRIDE,
        "sample_rate_hz":      SAMPLE_RATE_HZ,
    }


def write_split_yaml(
    rows: list[dict],
    target_subject: str,
    splits_dir: Path,
    support_shot_n: int,
    random_seed: int,
) -> Path:
    """Write the subject-out split configuration to YAML."""
    subjects = sorted({r["subject_id"] for r in rows})

    if target_subject not in subjects:
        raise ValueError(
            f"--split-target '{target_subject}' is not in the dataset. "
            f"Available subjects: {subjects}"
        )

    source_subjects = [s for s in subjects if s != target_subject]
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_path = splits_dir / "wifall_subject_split.yaml"

    split_doc = {
        "target_subject":  target_subject,
        "source_subjects": source_subjects,
        "support_shot_n":  support_shot_n,
        "random_seed":     random_seed,
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(split_doc, f, default_flow_style=False, allow_unicode=True)

    return out_path


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    split_cfg   = cfg.get("split", {})

    zip_path       = Path(args.zip) if args.zip else Path(dataset_cfg.get("zip_path", "data/WiFall.zip"))
    output_dir     = Path(args.output_dir)
    splits_dir     = Path(args.splits_dir)
    target_subject = args.split_target or split_cfg.get("target_subject", "ID0")
    support_shot_n = int(split_cfg.get("support_shot_n", 1))
    random_seed    = int(split_cfg.get("random_seed", 42))

    # Normalize binary_mapping keys to lowercase to match parse_zip_entry output
    raw_mapping    = dataset_cfg.get("binary_mapping", {})
    binary_mapping = {k.lower(): str(v) for k, v in raw_mapping.items()}
    log.info("Binary mapping: %s", binary_mapping)

    rows = process_zip(zip_path, binary_mapping)
    if not rows:
        log.error(
            "No windows generated from '%s'. "
            "Check that the zip has the expected WiFall/ID{n}/action/*.csv structure.",
            zip_path,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "wifall_manifest.csv"
    write_manifest(rows, manifest_path)
    log.info("Manifest  -> %s  (%d rows)", manifest_path, len(rows))

    summary      = build_summary(rows, zip_path)
    summary_path = output_dir / "wifall_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Summary   -> %s", summary_path)

    split_out = write_split_yaml(rows, target_subject, splits_dir, support_shot_n, random_seed)
    log.info("Split cfg -> %s", split_out)

    print()
    print("-- WiFall Preparation Complete --")
    print(f"  Source files    : {summary['n_files']}")
    print(f"  Subjects        : {summary['subjects']}")
    print(f"  Windows total   : {summary['n_windows_total']}  (== manifest rows)")
    print(f"  Label dist.     : {summary['label_distribution']}")
    print(f"  Activity counts : {summary['action_window_counts']}")
    print(f"  Target subject  : {target_subject}")
    print()
    print(f"  Manifest  -> {manifest_path}")
    print(f"  Summary   -> {summary_path}")
    print(f"  Split cfg -> {split_out}")


if __name__ == "__main__":
    main()

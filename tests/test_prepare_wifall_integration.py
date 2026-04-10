"""End-to-end integration test for the Step 1 data-preparation pipeline.

Creates a tiny synthetic WiFall-style ZIP in a temp directory, runs
process_zip() and build_summary() through their full logic, and asserts
that the output manifest rows and summary stats are correct.

No real WiFall.zip required.
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

import pytest

# Make project root importable (scripts/ is NOT a package — load via sys.path).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import prepare_wifall  # loaded from scripts/prepare_wifall.py
from prepare_wifall import WINDOW_SIZE, STRIDE, build_summary, process_zip


# ── helpers ──────────────────────────────────────────────────────────────────

_WIFALL_COLUMNS = [
    "type", "seq", "timestamp", "taget_seq", "taget",
    "mac", "rssi", "rate", "sig_mode", "mcs", "cwb",
    "smoothing", "not_sounding", "aggregation", "stbc",
    "fec_coding", "sgi", "noise_floor", "ampdu_cnt",
    "channel_primary", "channel_secondary", "local_timestamp",
    "ant", "sig_len", "rx_state", "len", "first_word_invalid",
    "data",
]

_N_SUBCARRIERS = 52
_FLAT_IQ = [3, 4] * _N_SUBCARRIERS  # valid 104-element I/Q vector


def _make_csv_bytes(n_rows: int) -> bytes:
    """Return bytes of a minimal WiFall-format CSV with n_rows data rows."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_WIFALL_COLUMNS)
    writer.writeheader()
    for _ in range(n_rows):
        row = {col: "0" for col in _WIFALL_COLUMNS}
        row["type"] = "CSI_DATA"
        row["data"] = str(_FLAT_IQ)
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def _make_wifall_zip(tmp_path: Path, entries: dict[str, int]) -> Path:
    """Write a synthetic WiFall ZIP to tmp_path.

    Args:
        entries: mapping of internal zip path → number of CSI rows.
                 Paths must follow WiFall/ID{n}/{action}/{name}.csv format.

    Returns:
        Path to the written ZIP file.
    """
    zip_path = tmp_path / "WiFall_test.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for entry_name, n_rows in entries.items():
            zf.writestr(entry_name, _make_csv_bytes(n_rows))
    return zip_path


# ── fixtures ─────────────────────────────────────────────────────────────────

BINARY_MAPPING = {
    "fall":  "fall",
    "walk":  "non_fall",
    "sit":   "non_fall",
    "stand": "non_fall",
    "jump":  "non_fall",
}

# Synthetic ZIP contents: subject → action → n_rows
_ZIP_ENTRIES = {
    "WiFall/ID0/fall/a.csv":  200,   # 2 windows
    "WiFall/ID0/walk/b.csv":  100,   # 1 window
    "WiFall/ID1/fall/c.csv":  253,   # 2 windows  (253//100 = 2)
    "WiFall/ID1/sit/d.csv":   99,    # 0 windows  (too short)
    "WiFall/ID2/stand/e.csv": 150,   # 1 window
    # entries that must be skipped by the parser
    "not_wifall/ID0/fall/f.csv": 100,  # wrong root prefix
    "WiFall/ID3/fall/README.txt": 0,   # non-CSV — won't appear in CSV scan
}


@pytest.fixture
def zip_path(tmp_path: Path) -> Path:
    return _make_wifall_zip(tmp_path, _ZIP_ENTRIES)


# ── tests ────────────────────────────────────────────────────────────────────

class TestProcessZipEndToEnd:
    def test_returns_list_of_dicts(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_window_count_correct(self, zip_path: Path) -> None:
        # Expected: 2 + 1 + 2 + 0 + 1 = 6  (ID1/sit skipped; wrong-prefix entry skipped)
        rows = process_zip(zip_path, BINARY_MAPPING)
        assert len(rows) == 6

    def test_required_columns_present(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        required = {
            "subject_id", "activity_label", "binary_label", "source_file",
            "window_index", "start_row", "end_row", "num_rows",
        }
        for row in rows:
            assert required == set(row.keys()), f"Unexpected columns in row: {set(row.keys())}"

    def test_binary_labels_are_valid(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        for row in rows:
            assert row["binary_label"] in ("fall", "non_fall"), row

    def test_fall_entries_labelled_fall(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        fall_rows = [r for r in rows if r["activity_label"] == "fall"]
        assert all(r["binary_label"] == "fall" for r in fall_rows)

    def test_non_fall_entries_labelled_non_fall(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        non_fall_rows = [r for r in rows if r["activity_label"] != "fall"]
        assert all(r["binary_label"] == "non_fall" for r in non_fall_rows)

    def test_window_spans_are_valid(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        for row in rows:
            assert row["end_row"] == row["start_row"] + WINDOW_SIZE
            assert row["num_rows"] == WINDOW_SIZE
            assert row["start_row"] == row["window_index"] * STRIDE

    def test_source_files_use_forward_slashes(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        for row in rows:
            assert "\\" not in row["source_file"], (
                f"source_file must use forward slashes: {row['source_file']}"
            )

    def test_wrong_prefix_entry_excluded(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        sources = {r["source_file"] for r in rows}
        assert "not_wifall/ID0/fall/f.csv" not in sources

    def test_too_short_entry_excluded(self, zip_path: Path) -> None:
        """ID1/sit has only 99 rows — must produce zero windows."""
        rows = process_zip(zip_path, BINARY_MAPPING)
        sources = {r["source_file"] for r in rows}
        assert "WiFall/ID1/sit/d.csv" not in sources

    def test_subjects_detected(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        subjects = {r["subject_id"] for r in rows}
        assert subjects == {"ID0", "ID1", "ID2"}

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            process_zip(tmp_path / "nonexistent.zip", BINARY_MAPPING)


class TestBuildSummaryEndToEnd:
    def test_n_windows_total_equals_row_count(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        summary = build_summary(rows, zip_path)
        assert summary["n_windows_total"] == len(rows)

    def test_n_subjects_correct(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        summary = build_summary(rows, zip_path)
        assert summary["n_subjects"] == 3
        assert set(summary["subjects"]) == {"ID0", "ID1", "ID2"}

    def test_label_distribution_sums_to_total(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        summary = build_summary(rows, zip_path)
        dist = summary["label_distribution"]
        assert sum(dist.values()) == summary["n_windows_total"]

    def test_action_window_counts_sums_to_total(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        summary = build_summary(rows, zip_path)
        assert sum(summary["action_window_counts"].values()) == summary["n_windows_total"]

    def test_schema_keys_present(self, zip_path: Path) -> None:
        rows = process_zip(zip_path, BINARY_MAPPING)
        summary = build_summary(rows, zip_path)
        for key in (
            "zip_path", "n_subjects", "subjects", "n_files",
            "n_windows_total", "action_window_counts", "label_distribution",
            "n_subcarriers", "window_size", "stride", "sample_rate_hz",
        ):
            assert key in summary, f"Missing key in summary: {key}"

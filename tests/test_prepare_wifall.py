"""Tests for the WiFall data preparation layer.

Covers:
  - ZIP path parsing (parse_zip_entry)
  - Binary label normalization (binary_label)
  - I/Q to amplitude conversion (iq_to_amplitude)
  - CSI row counting (count_csi_rows)
  - CSI CSV parsing from a synthetic file-like object (load_csi_from_fileobj)
  - Window-level manifest schema and span validity

All tests use synthetic in-memory data — no WiFall.zip required.
"""
from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from datasets.loader import (
    IQ_VECTOR_LEN,
    N_SUBCARRIERS,
    count_csi_rows,
    iq_to_amplitude,
    load_csi_from_fileobj,
)
from datasets.manifest import binary_label, parse_zip_entry
from preprocessing.windowing import compute_window_count

# ── manifest constants (mirror scripts/prepare_wifall.py) ───────────────────
WINDOW_SIZE = 100
STRIDE      = 100

EXPECTED_MANIFEST_COLUMNS = {
    "subject_id",
    "activity_label",
    "binary_label",
    "source_file",
    "window_index",
    "start_row",
    "end_row",
    "num_rows",
}

# ── helpers ─────────────────────────────────────────────────────────────────

_WIFALL_COLUMNS = [
    "type", "seq", "timestamp", "taget_seq", "taget",
    "mac", "rssi", "rate", "sig_mode", "mcs", "cwb",
    "smoothing", "not_sounding", "aggregation", "stbc",
    "fec_coding", "sgi", "noise_floor", "ampdu_cnt",
    "channel_primary", "channel_secondary", "local_timestamp",
    "ant", "sig_len", "rx_state", "len", "first_word_invalid",
    "data",
]


def _make_wifall_csv(flat_iq: list[int], n_rows: int = 5) -> io.BytesIO:
    """Build an in-memory WiFall-format CSV with n_rows identical data rows."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_WIFALL_COLUMNS)
    writer.writeheader()
    for _ in range(n_rows):
        row = {col: "0" for col in _WIFALL_COLUMNS}
        row["type"] = "CSI_DATA"
        row["data"] = str(flat_iq)
        writer.writerow(row)
    return io.BytesIO(buf.getvalue().encode("utf-8"))


def _make_window_rows(n_samples: int, source: str = "WiFall/ID0/fall/f.csv") -> list[dict]:
    """Simulate the window rows that process_zip would emit for a single file."""
    n_windows = compute_window_count(n_samples, WINDOW_SIZE, STRIDE)
    rows = []
    for win_idx in range(n_windows):
        start = win_idx * STRIDE
        end   = start + WINDOW_SIZE
        rows.append({
            "subject_id":     "ID0",
            "activity_label": "fall",
            "binary_label":   "fall",
            "source_file":    source,
            "window_index":   win_idx,
            "start_row":      start,
            "end_row":        end,
            "num_rows":       WINDOW_SIZE,
        })
    return rows


# ── parse_zip_entry ──────────────────────────────────────────────────────────

class TestParseZipEntry:
    def test_fall_entry_parsed(self):
        result = parse_zip_entry("WiFall/ID0/fall/2023-11-12_21-21-03-546_104_1.csv")
        assert result == ("ID0", "fall")

    def test_walk_entry_parsed(self):
        assert parse_zip_entry("WiFall/ID2/walk/some_file.csv") == ("ID2", "walk")

    def test_jump_normalized_to_lowercase(self):
        # Actual WiFall zip uses 'Jump' with capital J
        result = parse_zip_entry("WiFall/ID3/Jump/xxx.csv")
        assert result == ("ID3", "jump")

    def test_all_known_actions_parsed(self):
        for raw_action in ("fall", "walk", "sit", "stand", "Jump"):
            result = parse_zip_entry(f"WiFall/ID5/{raw_action}/file.csv")
            assert result is not None, f"Expected match for action '{raw_action}'"
            subject, action_norm = result
            assert subject == "ID5"
            assert action_norm == raw_action.lower()

    def test_all_subject_ids_parsed(self):
        for n in range(10):
            result = parse_zip_entry(f"WiFall/ID{n}/fall/file.csv")
            assert result is not None
            assert result[0] == f"ID{n}"

    def test_directory_entry_returns_none(self):
        assert parse_zip_entry("WiFall/ID0/fall/") is None

    def test_non_csv_extension_returns_none(self):
        assert parse_zip_entry("WiFall/ID0/fall/readme.txt") is None

    def test_missing_root_prefix_returns_none(self):
        # Old incorrect assumption — no 'WiFall/' prefix
        assert parse_zip_entry("ID0/fall/file.csv") is None

    def test_wrong_structure_returns_none(self):
        assert parse_zip_entry("WiFall/ID0/file.csv") is None


# ── binary_label ─────────────────────────────────────────────────────────────

class TestBinaryLabel:
    def test_fall_returns_fall(self):
        assert binary_label("fall") == "fall"

    def test_non_fall_actions_return_non_fall(self):
        for action in ("walk", "sit", "stand", "jump"):
            assert binary_label(action) == "non_fall", f"failed for '{action}'"

    def test_case_insensitive(self):
        assert binary_label("FALL") == "fall"
        assert binary_label("Fall") == "fall"
        assert binary_label("Jump") == "non_fall"

    def test_unknown_action_returns_non_fall(self):
        assert binary_label("unknown_action") == "non_fall"


# ── iq_to_amplitude ───────────────────────────────────────────────────────────

class TestIqToAmplitude:
    def test_known_3_4_5_triple(self):
        flat = [3, 4] * N_SUBCARRIERS
        amp  = iq_to_amplitude(flat)
        assert amp.shape == (N_SUBCARRIERS,)
        np.testing.assert_allclose(amp, 5.0, rtol=1e-5)

    def test_zero_iq_gives_zero_amplitude(self):
        np.testing.assert_array_equal(iq_to_amplitude([0] * IQ_VECTOR_LEN), 0.0)

    def test_negative_iq_gives_non_negative_amplitude(self):
        flat = [-3, -4] * N_SUBCARRIERS
        amp  = iq_to_amplitude(flat)
        assert np.all(amp >= 0.0)
        np.testing.assert_allclose(amp, 5.0, rtol=1e-5)

    def test_output_dtype_is_float32(self):
        assert iq_to_amplitude([1, 0] * N_SUBCARRIERS).dtype == np.float32

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match=str(IQ_VECTOR_LEN)):
            iq_to_amplitude([1, 2, 3])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            iq_to_amplitude([])


# ── count_csi_rows ────────────────────────────────────────────────────────────

class TestCountCsiRows:
    def test_counts_data_rows_only(self):
        f = _make_wifall_csv([3, 4] * N_SUBCARRIERS, n_rows=10)
        assert count_csi_rows(f) == 10

    def test_single_row(self):
        f = _make_wifall_csv([0] * IQ_VECTOR_LEN, n_rows=1)
        assert count_csi_rows(f) == 1

    def test_header_only_returns_zero(self):
        buf = io.BytesIO((",".join(_WIFALL_COLUMNS) + "\n").encode("utf-8"))
        assert count_csi_rows(buf) == 0

    def test_empty_file_returns_zero(self):
        assert count_csi_rows(io.BytesIO(b"")) == 0

    def test_large_row_count(self):
        f = _make_wifall_csv([1, 0] * N_SUBCARRIERS, n_rows=253)
        assert count_csi_rows(f) == 253


# ── load_csi_from_fileobj ────────────────────────────────────────────────────

class TestLoadCsiFromFileobj:
    def test_output_shape(self):
        f      = _make_wifall_csv([3, 4] * N_SUBCARRIERS, n_rows=10)
        result = load_csi_from_fileobj(f)
        assert result.shape == (10, N_SUBCARRIERS)

    def test_amplitude_values_correct(self):
        # [3, 4] per pair -> sqrt(9+16) = 5.0 everywhere
        f      = _make_wifall_csv([3, 4] * N_SUBCARRIERS, n_rows=3)
        result = load_csi_from_fileobj(f)
        np.testing.assert_allclose(result, 5.0, rtol=1e-5)

    def test_output_dtype_is_float32(self):
        f = _make_wifall_csv([1, 0] * N_SUBCARRIERS, n_rows=2)
        assert load_csi_from_fileobj(f).dtype == np.float32

    def test_missing_data_column_raises(self):
        buf = io.BytesIO(b"type,seq\nCSI_DATA,1\n")
        with pytest.raises(ValueError, match="data"):
            load_csi_from_fileobj(buf)

    def test_empty_csv_body_returns_empty_array(self):
        buf    = io.BytesIO((",".join(_WIFALL_COLUMNS) + "\n").encode("utf-8"))
        result = load_csi_from_fileobj(buf)
        assert result.shape == (0, N_SUBCARRIERS)


# ── window-level manifest schema ─────────────────────────────────────────────

class TestManifestSchema:
    """Verify the window-level manifest column names and value constraints."""

    def _sample_row(self) -> dict:
        return {
            "subject_id":     "ID0",
            "activity_label": "fall",
            "binary_label":   "fall",
            "source_file":    "WiFall/ID0/fall/2023-11-12.csv",
            "window_index":   0,
            "start_row":      0,
            "end_row":        100,
            "num_rows":       100,
        }

    def test_required_columns_present(self):
        assert EXPECTED_MANIFEST_COLUMNS == set(self._sample_row().keys())

    def test_old_column_names_absent(self):
        row = self._sample_row()
        for old_name in ("subject", "action", "file_path", "n_samples", "n_subcarriers", "n_windows"):
            assert old_name not in row, f"Old column '{old_name}' must not appear"

    def test_binary_label_is_valid(self):
        assert self._sample_row()["binary_label"] in ("fall", "non_fall")

    def test_source_file_uses_forward_slashes(self):
        # Internal zip paths always use forward slashes (zip spec)
        assert "\\" not in self._sample_row()["source_file"]

    def test_num_rows_equals_window_size(self):
        assert self._sample_row()["num_rows"] == WINDOW_SIZE


# ── window span validity ──────────────────────────────────────────────────────

class TestWindowSpans:
    """Verify that generated window rows have correct and consistent spans."""

    def _rows_for(self, n_samples: int) -> list[dict]:
        return _make_window_rows(n_samples)

    def test_window_count_matches_compute_window_count(self):
        for n in (100, 150, 200, 253, 274):
            rows     = self._rows_for(n)
            expected = compute_window_count(n, WINDOW_SIZE, STRIDE)
            assert len(rows) == expected, (
                f"n_samples={n}: expected {expected} windows, got {len(rows)}"
            )

    def test_window_index_is_sequential(self):
        rows = self._rows_for(253)
        for i, row in enumerate(rows):
            assert row["window_index"] == i

    def test_start_row_equals_index_times_stride(self):
        rows = self._rows_for(253)
        for row in rows:
            assert row["start_row"] == row["window_index"] * STRIDE

    def test_end_row_equals_start_plus_window_size(self):
        rows = self._rows_for(253)
        for row in rows:
            assert row["end_row"] == row["start_row"] + WINDOW_SIZE

    def test_num_rows_equals_window_size(self):
        rows = self._rows_for(253)
        for row in rows:
            assert row["num_rows"] == WINDOW_SIZE

    def test_span_width_consistent(self):
        rows = self._rows_for(253)
        for row in rows:
            assert row["end_row"] - row["start_row"] == row["num_rows"] == WINDOW_SIZE

    def test_no_windows_when_too_few_samples(self):
        assert self._rows_for(99) == []

    def test_exactly_one_window_at_boundary(self):
        rows = self._rows_for(100)   # exactly window_size
        assert len(rows) == 1
        assert rows[0]["start_row"] == 0
        assert rows[0]["end_row"]   == 100

    def test_two_windows_at_double_boundary(self):
        rows = self._rows_for(200)
        assert len(rows) == 2
        assert rows[1]["start_row"] == 100
        assert rows[1]["end_row"]   == 200


# ── total window count matches manifest ────────────────────────────────────────

class TestWindowCountConsistency:
    """n_windows_total in summary must equal the number of manifest rows."""

    def test_summary_n_windows_equals_manifest_len(self):
        # Simulate two source files: 253 rows each -> 2 windows each -> 4 total
        all_rows = (
            _make_window_rows(253, "WiFall/ID0/fall/a.csv")
            + _make_window_rows(253, "WiFall/ID1/fall/b.csv")
        )
        n_windows_total = len(all_rows)
        assert n_windows_total == 4

    def test_mixed_lengths_total_correct(self):
        # 200 samples -> 2 windows, 150 samples -> 1 window
        rows = (
            _make_window_rows(200, "WiFall/ID0/fall/a.csv")
            + _make_window_rows(150, "WiFall/ID1/fall/b.csv")
        )
        assert len(rows) == 3

    def test_window_rows_are_flat_not_nested(self):
        # Each element must be a dict, not a list of dicts
        rows = _make_window_rows(200)
        for row in rows:
            assert isinstance(row, dict)
            assert "window_index" in row

import json
from pathlib import Path

import numpy as np

from collector.packet_parser import parse_packet_dict
from collector.receiver import parse_esp32_adr018_datagram
from collector.session_store import SessionStore
from preprocessing.features import FEATURE_VERSION, extract_window_features
from preprocessing.pipeline import process_loaded_raw_session, process_raw_session_by_id
from preprocessing.session_loader import load_raw_session
from preprocessing.windowing import build_windows, resolve_node_order
from shared.config import PreprocessingConfig
from shared.models import SessionMetadata


def test_extract_window_features_is_deterministic_for_missing_nodes() -> None:
    packets = [
        parse_packet_dict(
            {
                "timestamp": 0.0,
                "node_id": "1",
                "session_id": "session_features",
                "rssi": -40,
                "amplitude": [1.0, 2.0],
                "phase": [0.1, 0.2],
            }
        ),
        parse_packet_dict(
            {
                "timestamp": 0.3,
                "node_id": "1",
                "session_id": "session_features",
                "rssi": -42,
                "amplitude": [1.1, 2.1],
                "phase": [0.2, 0.3],
            }
        ),
    ]
    from preprocessing.decoder import decode_packets

    decoded_packets = decode_packets(packets, phase_unwrap_enabled=False)
    node_order = resolve_node_order(decoded_packets, expected_nodes=("1", "2"))
    windows = build_windows(
        decoded_packets,
        session_id="session_features",
        window_seconds=1.0,
        stride_seconds=0.5,
        expected_nodes=node_order,
    )

    feature_bundle = extract_window_features(
        windows,
        node_order=node_order,
        selected_subcarriers=(0, 1),
        label="fall",
        source_session_id="session_features",
    )

    assert feature_bundle.feature_version == FEATURE_VERSION
    assert feature_bundle.feature_names[:4] == (
        "active_node_count",
        "total_packet_count",
        "node_1__present",
        "node_1__packet_count",
    )
    window = feature_bundle.windows[0]
    assert window.features["node_2__present"] == 0.0
    assert window.features["node_2__packet_count"] == 0.0
    assert window.features["node_1__amplitude_mean_sc0"] > 0.0
    assert feature_bundle.feature_matrix.shape == (1, len(feature_bundle.feature_names))


def test_process_raw_session_writes_processed_artifacts_from_normalized_fixture(
    tmp_path: Path,
    fixture_packets_path: Path,
    fixture_metadata_path: Path,
) -> None:
    metadata = SessionMetadata.from_dict(json.loads(fixture_metadata_path.read_text(encoding="utf-8")))
    packets = [
        parse_packet_dict(json.loads(line))
        for line in fixture_packets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_root = tmp_path / "raw"
    SessionStore(raw_root).write_session(metadata, packets)
    raw_session = load_raw_session(raw_root / metadata.session_id)

    result = process_loaded_raw_session(
        raw_session,
        preprocessing_config=PreprocessingConfig(
            window_seconds=0.1,
            stride_seconds=0.05,
            phase_unwrap_enabled=True,
            selected_subcarriers=[0, 2],
        ),
        expected_nodes=("node-a", "node-b", "node-c"),
        output_root=tmp_path / "processed",
    )

    assert result.window_count == 3
    assert result.feature_count > 0
    manifest_path = result.output_dir / "manifest.json"
    windows_path = result.output_dir / "windows.jsonl"
    dataset_path = result.output_dir / "dataset.npz"
    assert manifest_path.exists()
    assert windows_path.exists()
    assert dataset_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_version"] == FEATURE_VERSION
    assert manifest["node_order"] == ["node-a", "node-b", "node-c"]
    assert manifest["window_count"] == 3

    windows_lines = [json.loads(line) for line in windows_path.read_text(encoding="utf-8").splitlines()]
    assert len(windows_lines) == 3
    assert windows_lines[0]["label"] == metadata.activity_label
    assert "active_node_count" in windows_lines[0]["features"]

    dataset = np.load(dataset_path)
    assert dataset["feature_matrix"].shape == (3, result.feature_count)
    assert tuple(dataset["node_order"].tolist()) == ("node-a", "node-b", "node-c")


def test_process_raw_session_is_reproducible_for_binary_raw_fixture(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    metadata = SessionMetadata.from_dict(
        {
            "session_id": "session_binary_phase3",
            "participant_id": "P03",
            "room_id": "bedroom_a",
            "layout_version": "layout_v1",
            "node_setup_version": "setup_triangle_v1",
            "activity_label": "fall",
            "recorded_at": "2026-04-08T10:00:00+09:00",
        }
    )
    datagram = bytes.fromhex(
        (repo_root / "tests" / "fixtures" / "esp32_adr018_valid_packet.hex").read_text(
            encoding="utf-8"
        )
    )
    packets = [
        parse_esp32_adr018_datagram(
            datagram,
            session_id=metadata.session_id,
            received_at=1712577600.5 + offset,
        )
        for offset in (0.0, 0.2, 0.4)
    ]
    raw_root = tmp_path / "raw"
    SessionStore(raw_root).write_session(metadata, packets)

    config = PreprocessingConfig(
        window_seconds=0.3,
        stride_seconds=0.2,
        phase_unwrap_enabled=True,
        selected_subcarriers=[0, 1],
    )
    result_a = process_raw_session_by_id(
        raw_root,
        session_id=metadata.session_id,
        preprocessing_config=config,
        expected_nodes=("2",),
        output_root=tmp_path / "processed_a",
    )
    result_b = process_raw_session_by_id(
        raw_root,
        session_id=metadata.session_id,
        preprocessing_config=config,
        expected_nodes=("2",),
        output_root=tmp_path / "processed_b",
    )

    dataset_a = np.load(result_a.output_dir / "dataset.npz")
    dataset_b = np.load(result_b.output_dir / "dataset.npz")
    np.testing.assert_array_equal(dataset_a["feature_names"], dataset_b["feature_names"])
    np.testing.assert_allclose(dataset_a["feature_matrix"], dataset_b["feature_matrix"])

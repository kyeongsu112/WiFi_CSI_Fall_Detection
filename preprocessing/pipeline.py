from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from preprocessing.decoder import decode_packets
from preprocessing.features import FEATURE_VERSION, FeatureMatrixBundle, extract_window_features
from preprocessing.filters import filter_decoded_packets
from preprocessing.session_loader import RawSession, load_raw_session, load_raw_session_by_id
from preprocessing.windowing import build_windows, resolve_node_order
from shared.config import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class ProcessedWriteResult:
    session_id: str
    output_dir: Path
    window_count: int
    feature_count: int
    node_order: tuple[str, ...]


def process_raw_session(
    session_dir: Path | str,
    *,
    preprocessing_config: PreprocessingConfig,
    expected_nodes: Sequence[str] = (),
    output_root: Path | str = "artifacts/processed",
) -> ProcessedWriteResult:
    raw_session = load_raw_session(session_dir)
    return process_loaded_raw_session(
        raw_session,
        preprocessing_config=preprocessing_config,
        expected_nodes=expected_nodes,
        output_root=output_root,
    )


def process_raw_session_by_id(
    raw_root: Path | str,
    *,
    session_id: str,
    preprocessing_config: PreprocessingConfig,
    expected_nodes: Sequence[str] = (),
    output_root: Path | str = "artifacts/processed",
) -> ProcessedWriteResult:
    raw_session = load_raw_session_by_id(raw_root, session_id)
    return process_loaded_raw_session(
        raw_session,
        preprocessing_config=preprocessing_config,
        expected_nodes=expected_nodes,
        output_root=output_root,
    )


def process_loaded_raw_session(
    raw_session: RawSession,
    *,
    preprocessing_config: PreprocessingConfig,
    expected_nodes: Sequence[str] = (),
    output_root: Path | str = "artifacts/processed",
) -> ProcessedWriteResult:
    decoded_packets = decode_packets(
        raw_session.packets,
        phase_unwrap_enabled=preprocessing_config.phase_unwrap_enabled,
    )
    filtered_packets = filter_decoded_packets(
        decoded_packets,
        outlier_zscore_threshold=preprocessing_config.outlier_zscore_threshold,
        median_filter_kernel_size=preprocessing_config.median_filter_kernel_size,
        smoothing_window_size=preprocessing_config.smoothing_window_size,
    )
    node_order = resolve_node_order(filtered_packets, expected_nodes=expected_nodes)
    windows = build_windows(
        filtered_packets,
        session_id=raw_session.metadata.session_id,
        window_seconds=preprocessing_config.window_seconds,
        stride_seconds=preprocessing_config.stride_seconds,
        expected_nodes=node_order,
    )
    feature_bundle = extract_window_features(
        windows,
        node_order=node_order,
        selected_subcarriers=preprocessing_config.selected_subcarriers,
        label=raw_session.metadata.activity_label,
        source_session_id=raw_session.metadata.session_id,
        feature_version=FEATURE_VERSION,
    )

    output_dir = Path(output_root) / raw_session.metadata.session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(
        output_dir / "manifest.json",
        raw_session=raw_session,
        preprocessing_config=preprocessing_config,
        feature_bundle=feature_bundle,
    )
    _write_windows_jsonl(output_dir / "windows.jsonl", feature_bundle)
    _write_dataset_npz(output_dir / "dataset.npz", feature_bundle)

    return ProcessedWriteResult(
        session_id=raw_session.metadata.session_id,
        output_dir=output_dir,
        window_count=len(feature_bundle.windows),
        feature_count=len(feature_bundle.feature_names),
        node_order=feature_bundle.node_order,
    )


def _write_manifest(
    manifest_path: Path,
    *,
    raw_session: RawSession,
    preprocessing_config: PreprocessingConfig,
    feature_bundle: FeatureMatrixBundle,
) -> None:
    manifest = {
        "session_id": raw_session.metadata.session_id,
        "metadata": raw_session.metadata.to_dict(),
        "preprocessing_config": preprocessing_config.model_dump(mode="python"),
        "node_order": list(feature_bundle.node_order),
        "selected_subcarriers": list(preprocessing_config.selected_subcarriers),
        "feature_version": feature_bundle.feature_version,
        "feature_names": list(feature_bundle.feature_names),
        "decoder_assumptions": {
            "supported_raw_formats": [
                "esp32_adr018",
                "derived_amplitude_phase",
                "normalized_fixture",
            ],
            "adr018_iq_interpretation": "row-major int8 I/Q pair stream",
            "phase_unwrap_axis": "bin" if preprocessing_config.phase_unwrap_enabled else "disabled",
            "timestamp_source": "collector_receive_time_or_stored_packet_timestamp",
        },
        "window_count": len(feature_bundle.windows),
        "packet_count": len(raw_session.packets),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def _write_windows_jsonl(
    windows_path: Path,
    feature_bundle: FeatureMatrixBundle,
) -> None:
    with windows_path.open("w", encoding="utf-8") as handle:
        for window in feature_bundle.windows:
            payload = {
                "window_id": window.window_id,
                "start_ts": window.start_ts,
                "end_ts": window.end_ts,
                "node_ids": list(window.node_ids),
                "label": window.label,
                "feature_version": window.feature_version,
                "source_session_ids": list(window.source_session_ids),
                "features": dict(window.features),
            }
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")


def _write_dataset_npz(
    dataset_path: Path,
    feature_bundle: FeatureMatrixBundle,
) -> None:
    np.savez_compressed(
        dataset_path,
        feature_matrix=feature_bundle.feature_matrix,
        labels=np.asarray(feature_bundle.labels),
        window_ids=np.asarray([window.window_id for window in feature_bundle.windows]),
        feature_names=np.asarray(feature_bundle.feature_names),
        start_ts=np.asarray([window.start_ts for window in feature_bundle.windows], dtype=np.float64),
        end_ts=np.asarray([window.end_ts for window in feature_bundle.windows], dtype=np.float64),
        node_order=np.asarray(feature_bundle.node_order),
    )

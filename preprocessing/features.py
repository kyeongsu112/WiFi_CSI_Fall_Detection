from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from preprocessing.windowing import DecodedWindow
from shared.models import CsiWindow


FEATURE_VERSION = "baseline_v1"


@dataclass(frozen=True, slots=True)
class FeatureMatrixBundle:
    windows: tuple[CsiWindow, ...]
    feature_names: tuple[str, ...]
    feature_matrix: np.ndarray
    labels: tuple[str, ...]
    node_order: tuple[str, ...]
    feature_version: str


def build_feature_names(
    *,
    node_order: Sequence[str],
    selected_subcarriers: Sequence[int],
) -> tuple[str, ...]:
    feature_names = ["active_node_count", "total_packet_count"]
    sorted_subcarriers = tuple(sorted(dict.fromkeys(int(index) for index in selected_subcarriers)))

    for node_id in node_order:
        prefix = f"node_{node_id}__"
        feature_names.extend(
            [
                prefix + "present",
                prefix + "packet_count",
                prefix + "rssi_mean",
                prefix + "rssi_std",
                prefix + "amplitude_mean",
                prefix + "amplitude_std",
                prefix + "phase_mean",
                prefix + "phase_std",
            ]
        )
        for subcarrier_index in sorted_subcarriers:
            feature_names.append(prefix + f"amplitude_mean_sc{subcarrier_index}")
            feature_names.append(prefix + f"phase_mean_sc{subcarrier_index}")

    return tuple(feature_names)


def extract_window_features(
    windows: Sequence[DecodedWindow],
    *,
    node_order: Sequence[str],
    selected_subcarriers: Sequence[int],
    label: str,
    source_session_id: str,
    feature_version: str = FEATURE_VERSION,
) -> FeatureMatrixBundle:
    normalized_node_order = tuple(str(node_id) for node_id in node_order)
    feature_names = build_feature_names(
        node_order=normalized_node_order,
        selected_subcarriers=selected_subcarriers,
    )
    selected_bins = tuple(sorted(dict.fromkeys(int(index) for index in selected_subcarriers)))

    processed_windows: list[CsiWindow] = []
    matrix_rows: list[list[float]] = []
    labels: list[str] = []

    for window in windows:
        row: dict[str, float] = {}
        total_packet_count = sum(
            len(window.packets_by_node.get(node_id, ())) for node_id in normalized_node_order
        )
        active_node_count = sum(
            1 for node_id in normalized_node_order if window.packets_by_node.get(node_id, ())
        )
        row["active_node_count"] = float(active_node_count)
        row["total_packet_count"] = float(total_packet_count)

        present_nodes: list[str] = []
        for node_id in normalized_node_order:
            packets = window.packets_by_node.get(node_id, ())
            if packets:
                present_nodes.append(node_id)
            row.update(
                _compute_node_features(
                    node_id=node_id,
                    packets=packets,
                    selected_subcarriers=selected_bins,
                )
            )

        processed_window = CsiWindow(
            window_id=window.window_id,
            start_ts=window.start_ts,
            end_ts=window.end_ts,
            node_ids=tuple(present_nodes),
            features=row,
            label=label,
            feature_version=feature_version,
            source_session_ids=(source_session_id,),
        )
        processed_windows.append(processed_window)
        matrix_rows.append([float(processed_window.features[name]) for name in feature_names])
        labels.append(label)

    feature_matrix = np.asarray(matrix_rows, dtype=np.float64)
    if feature_matrix.size == 0:
        feature_matrix = np.zeros((0, len(feature_names)), dtype=np.float64)
    feature_matrix.setflags(write=False)

    return FeatureMatrixBundle(
        windows=tuple(processed_windows),
        feature_names=feature_names,
        feature_matrix=feature_matrix,
        labels=tuple(labels),
        node_order=normalized_node_order,
        feature_version=feature_version,
    )


def _compute_node_features(
    *,
    node_id: str,
    packets: Sequence,
    selected_subcarriers: Sequence[int],
) -> dict[str, float]:
    prefix = f"node_{node_id}__"
    row = {
        prefix + "present": 0.0,
        prefix + "packet_count": 0.0,
        prefix + "rssi_mean": 0.0,
        prefix + "rssi_std": 0.0,
        prefix + "amplitude_mean": 0.0,
        prefix + "amplitude_std": 0.0,
        prefix + "phase_mean": 0.0,
        prefix + "phase_std": 0.0,
    }
    for subcarrier_index in selected_subcarriers:
        row[prefix + f"amplitude_mean_sc{subcarrier_index}"] = 0.0
        row[prefix + f"phase_mean_sc{subcarrier_index}"] = 0.0

    if not packets:
        return row

    row[prefix + "present"] = 1.0
    row[prefix + "packet_count"] = float(len(packets))

    rssi_values = np.asarray(
        [packet.rssi for packet in packets if packet.rssi is not None],
        dtype=np.float64,
    )
    amplitude_values = np.concatenate([packet.flattened_amplitude() for packet in packets])
    phase_values = np.concatenate([packet.flattened_phase() for packet in packets])

    row[prefix + "rssi_mean"], row[prefix + "rssi_std"] = _mean_std(rssi_values)
    row[prefix + "amplitude_mean"], row[prefix + "amplitude_std"] = _mean_std(amplitude_values)
    row[prefix + "phase_mean"], row[prefix + "phase_std"] = _mean_std(phase_values)

    for subcarrier_index in selected_subcarriers:
        amplitude_bin_values = [
            packet.flattened_amplitude()[subcarrier_index]
            for packet in packets
            if packet.flattened_amplitude().shape[0] > subcarrier_index
        ]
        phase_bin_values = [
            packet.flattened_phase()[subcarrier_index]
            for packet in packets
            if packet.flattened_phase().shape[0] > subcarrier_index
        ]

        if amplitude_bin_values:
            row[prefix + f"amplitude_mean_sc{subcarrier_index}"] = float(
                np.mean(np.asarray(amplitude_bin_values, dtype=np.float64))
            )
        if phase_bin_values:
            row[prefix + f"phase_mean_sc{subcarrier_index}"] = float(
                np.mean(np.asarray(phase_bin_values, dtype=np.float64))
            )

    return row


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=0))

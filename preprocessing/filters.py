from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from preprocessing.decoder import DecodedCsiPacket


def filter_decoded_packets(
    packets: Sequence[DecodedCsiPacket],
    *,
    outlier_zscore_threshold: float,
    median_filter_kernel_size: int,
    smoothing_window_size: int,
) -> tuple[DecodedCsiPacket, ...]:
    return tuple(
        filter_decoded_packet(
            packet,
            outlier_zscore_threshold=outlier_zscore_threshold,
            median_filter_kernel_size=median_filter_kernel_size,
            smoothing_window_size=smoothing_window_size,
        )
        for packet in packets
    )


def filter_decoded_packet(
    packet: DecodedCsiPacket,
    *,
    outlier_zscore_threshold: float,
    median_filter_kernel_size: int,
    smoothing_window_size: int,
) -> DecodedCsiPacket:
    amplitude = apply_signal_filters(
        packet.amplitude,
        outlier_zscore_threshold=outlier_zscore_threshold,
        median_filter_kernel_size=median_filter_kernel_size,
        smoothing_window_size=smoothing_window_size,
    )
    phase = apply_signal_filters(
        packet.phase,
        outlier_zscore_threshold=outlier_zscore_threshold,
        median_filter_kernel_size=median_filter_kernel_size,
        smoothing_window_size=smoothing_window_size,
    )
    real = amplitude * np.cos(phase)
    imag = amplitude * np.sin(phase)

    return replace(
        packet,
        real=_freeze_array(real),
        imag=_freeze_array(imag),
        amplitude=_freeze_array(amplitude),
        phase=_freeze_array(phase),
    )


def apply_signal_filters(
    values: np.ndarray,
    *,
    outlier_zscore_threshold: float,
    median_filter_kernel_size: int,
    smoothing_window_size: int,
) -> np.ndarray:
    filtered = np.asarray(values, dtype=np.float64)
    if filtered.ndim == 1:
        filtered = filtered.reshape(1, -1)

    filtered = _clip_outliers_modified_zscore(filtered, outlier_zscore_threshold)
    filtered = _median_filter(filtered, median_filter_kernel_size)
    filtered = _moving_average_filter(filtered, smoothing_window_size)
    return filtered


def _clip_outliers_modified_zscore(
    values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    clipped_rows = [_clip_row_modified_zscore(row, threshold) for row in values]
    return np.vstack(clipped_rows)


def _clip_row_modified_zscore(
    row: np.ndarray,
    threshold: float,
) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    if row.size == 0 or threshold <= 0:
        return row.copy()

    median = float(np.median(row))
    absolute_deviation = np.abs(row - median)
    mad = float(np.median(absolute_deviation))
    if mad <= 1e-9:
        return row.copy()

    scale = mad / 0.6745
    lower = median - threshold * scale
    upper = median + threshold * scale
    return np.clip(row, lower, upper)


def _median_filter(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size == 1:
        return values.copy()

    pad = kernel_size // 2
    filtered_rows = []
    for row in values:
        padded = np.pad(row, pad_width=pad, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
        filtered_rows.append(np.median(windows, axis=-1))
    return np.vstack(filtered_rows)


def _moving_average_filter(values: np.ndarray, window_size: int) -> np.ndarray:
    window_size = _normalize_kernel_size(window_size)
    if window_size == 1:
        return values.copy()

    kernel = np.full(window_size, 1.0 / window_size, dtype=np.float64)
    pad = window_size // 2
    smoothed_rows = []
    for row in values:
        padded = np.pad(row, pad_width=pad, mode="edge")
        smoothed_rows.append(np.convolve(padded, kernel, mode="valid"))
    return np.vstack(smoothed_rows)


def _normalize_kernel_size(value: int) -> int:
    if value <= 1:
        return 1
    if value % 2 == 0:
        return value + 1
    return value


def _freeze_array(array: np.ndarray) -> np.ndarray:
    frozen = np.asarray(array, dtype=np.float64)
    frozen.setflags(write=False)
    return frozen

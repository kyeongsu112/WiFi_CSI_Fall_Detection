import numpy as np

from collector.packet_parser import parse_packet_dict
from preprocessing.decoder import decode_packet
from preprocessing.filters import apply_signal_filters, filter_decoded_packet


def test_apply_signal_filters_reduces_spike_outlier() -> None:
    signal = np.asarray([[1.0, 1.0, 50.0, 1.0, 1.0]], dtype=np.float64)

    filtered = apply_signal_filters(
        signal,
        outlier_zscore_threshold=3.5,
        median_filter_kernel_size=3,
        smoothing_window_size=3,
    )

    assert filtered.shape == signal.shape
    assert filtered[0, 2] < 20.0


def test_filter_decoded_packet_preserves_packet_shape() -> None:
    packet = parse_packet_dict(
        {
            "timestamp": 0.0,
            "node_id": "node-a",
            "session_id": "session_filters",
            "rssi": -45,
            "amplitude": [1.0, 2.0, 100.0, 2.0, 1.0],
            "phase": [0.0, 0.1, 2.5, 0.1, 0.0],
        }
    )
    decoded = decode_packet(packet, phase_unwrap_enabled=False)

    filtered = filter_decoded_packet(
        decoded,
        outlier_zscore_threshold=3.5,
        median_filter_kernel_size=3,
        smoothing_window_size=3,
    )

    assert filtered.amplitude.shape == decoded.amplitude.shape
    assert filtered.phase.shape == decoded.phase.shape
    assert filtered.amplitude[0, 2] < decoded.amplitude[0, 2]

from pathlib import Path

import numpy as np
import pytest

from collector.packet_parser import parse_packet_dict
from collector.receiver import parse_esp32_adr018_datagram
from preprocessing.decoder import PacketDecodeError, decode_packet


def test_decode_packet_preserves_existing_amplitude_phase() -> None:
    packet = parse_packet_dict(
        {
            "timestamp": 1712491800.1,
            "node_id": "node-a",
            "session_id": "session_2026_04_07_001",
            "amplitude": [1.0, 2.0, 3.0],
            "phase": [0.0, np.pi / 2, np.pi],
            "csi_raw": {"format": "normalized_fixture"},
        }
    )

    decoded = decode_packet(packet, phase_unwrap_enabled=False)

    assert decoded.source_format == "normalized_fixture"
    assert decoded.amplitude.shape == (1, 3)
    assert decoded.phase.shape == (1, 3)
    assert decoded.real.shape == (1, 3)
    assert decoded.imag.shape == (1, 3)
    np.testing.assert_allclose(decoded.amplitude, np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(decoded.real, np.array([[1.0, 0.0, -3.0]]), atol=1e-6)


def test_decode_binary_adr018_packet_is_deterministic(repo_root: Path) -> None:
    datagram = bytes.fromhex(
        (repo_root / "tests" / "fixtures" / "esp32_adr018_valid_packet.hex").read_text(
            encoding="utf-8"
        )
    )
    packet = parse_esp32_adr018_datagram(
        datagram,
        session_id="session_2026_04_08_binary",
        received_at=1712577600.5,
    )

    decoded = decode_packet(packet, phase_unwrap_enabled=False)

    np.testing.assert_array_equal(decoded.real, np.array([[1.0, 3.0, 5.0, 7.0]]))
    np.testing.assert_array_equal(decoded.imag, np.array([[2.0, 4.0, 6.0, 8.0]]))
    np.testing.assert_allclose(
        decoded.amplitude,
        np.sqrt(np.array([[1.0, 3.0, 5.0, 7.0]]) ** 2 + np.array([[2.0, 4.0, 6.0, 8.0]]) ** 2),
    )
    np.testing.assert_allclose(
        decoded.phase,
        np.arctan2(np.array([[2.0, 4.0, 6.0, 8.0]]), np.array([[1.0, 3.0, 5.0, 7.0]])),
    )


def test_decode_packet_rejects_malformed_hex_payload() -> None:
    packet = parse_packet_dict(
        {
            "timestamp": 1712491800.1,
            "node_id": "2",
            "session_id": "session_2026_04_08_binary",
            "csi_raw": {
                "format": "esp32_adr018",
                "antenna_count": 1,
                "subcarrier_count": 4,
                "iq_bytes_hex": "NOT_HEX",
            },
        }
    )

    with pytest.raises(PacketDecodeError, match="not valid hexadecimal"):
        decode_packet(packet, phase_unwrap_enabled=False)


def test_decode_packet_rejects_unsupported_raw_format() -> None:
    packet = parse_packet_dict(
        {
            "timestamp": 1712491800.1,
            "node_id": "2",
            "session_id": "session_2026_04_08_binary",
            "csi_raw": {
                "format": "vendor_xyz",
            },
        }
    )

    with pytest.raises(PacketDecodeError, match="Unsupported raw CSI format"):
        decode_packet(packet, phase_unwrap_enabled=False)

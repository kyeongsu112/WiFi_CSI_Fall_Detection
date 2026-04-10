import pytest

from collector.packet_parser import PacketParseError, parse_packet_dict


def test_parse_valid_packet() -> None:
    packet = parse_packet_dict(
        {
            "timestamp": 1712491800.1,
            "node_id": "node-a",
            "session_id": "session_2026_04_07_001",
            "seq": 1,
            "rssi": -42,
            "channel": 6,
            "amplitude": [0.1, 0.2, 0.3],
            "phase": [1.0, 1.1, 1.2],
            "csi_raw": {"format": "normalized_fixture"},
            "vendor_field": "preserved",
        }
    )

    assert packet.node_id == "node-a"
    assert packet.seq == 1
    assert packet.amplitude == (0.1, 0.2, 0.3)
    assert packet.phase == (1.0, 1.1, 1.2)
    assert packet.to_dict()["raw_payload"] == {
        "csi_raw": {"format": "normalized_fixture"},
        "vendor_field": "preserved",
    }


def test_missing_required_field_raises() -> None:
    with pytest.raises(PacketParseError):
        parse_packet_dict(
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
            }
        )


def test_optional_amplitude_and_phase_are_allowed() -> None:
    packet = parse_packet_dict(
        {
            "timestamp": 1712491800.1,
            "node_id": "node-a",
            "session_id": "session_2026_04_07_001",
            "csi_raw": {"opaque": True},
        }
    )

    assert packet.amplitude is None
    assert packet.phase is None
    assert packet.to_dict()["raw_payload"] == {"csi_raw": {"opaque": True}}


def test_fractional_seq_and_channel_are_rejected() -> None:
    with pytest.raises(PacketParseError):
        parse_packet_dict(
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": "session_2026_04_07_001",
                "seq": 1.5,
            }
        )

    with pytest.raises(PacketParseError):
        parse_packet_dict(
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": "session_2026_04_07_001",
                "channel": 6.25,
            }
        )


def test_whole_number_numeric_seq_and_channel_are_accepted() -> None:
    packet = parse_packet_dict(
        {
            "timestamp": 1712491800.1,
            "node_id": "node-a",
            "session_id": "session_2026_04_07_001",
            "seq": 2.0,
            "channel": 6.0,
            "csi_raw": {"opaque": True},
        }
    )

    assert packet.seq == 2
    assert packet.channel == 6


def test_raw_payload_is_frozen_after_parsing() -> None:
    packet = parse_packet_dict(
        {
            "timestamp": 1712491800.1,
            "node_id": "node-a",
            "session_id": "session_2026_04_07_001",
            "csi_raw": {"format": "normalized_fixture"},
        }
    )

    with pytest.raises(TypeError):
        packet.raw_payload["extra"] = "blocked"  # type: ignore[index]

    with pytest.raises(TypeError):
        packet.raw_payload["csi_raw"]["format"] = "mutated"  # type: ignore[index]

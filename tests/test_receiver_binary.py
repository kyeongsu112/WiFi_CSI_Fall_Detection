import json
import socket
import threading
import time
from pathlib import Path

import pytest

from collector.receiver import (
    ReceiverError,
    UdpEsp32BinaryPacketSource,
    parse_esp32_adr018_datagram,
)
from collector.session_store import SessionStore
from scripts.collect import run_collection
from shared.models import SessionMetadata


def test_parse_valid_esp32_adr018_datagram(repo_root: Path) -> None:
    packet = parse_esp32_adr018_datagram(
        _load_hex_fixture(repo_root, "esp32_adr018_valid_packet.hex"),
        session_id="session_2026_04_08_binary",
        received_at=1712577600.5,
    )

    assert packet.timestamp == 1712577600.5
    assert packet.node_id == "2"
    assert packet.session_id == "session_2026_04_08_binary"
    assert packet.seq == 42
    assert packet.rssi == -41.0
    assert packet.channel == 6
    assert packet.to_dict()["raw_payload"] == {
        "csi_raw": {
            "format": "esp32_adr018",
            "magic": "0xC5110001",
            "antenna_count": 1,
            "subcarrier_count": 4,
            "frequency_mhz": 2437,
            "noise_floor": -95,
            "iq_bytes_hex": "0102030405060708",
        }
    }


def test_parse_truncated_esp32_adr018_datagram_raises(repo_root: Path) -> None:
    with pytest.raises(ReceiverError, match="payload length mismatch"):
        parse_esp32_adr018_datagram(
            _load_hex_fixture(repo_root, "esp32_adr018_truncated_packet.hex"),
            session_id="session_2026_04_08_binary",
            received_at=1712577600.5,
        )


def test_parse_unsupported_esp32_binary_magic_raises(repo_root: Path) -> None:
    with pytest.raises(ReceiverError, match="Unsupported ESP32 binary packet magic"):
        parse_esp32_adr018_datagram(
            _load_hex_fixture(repo_root, "esp32_adr018_unsupported_magic.hex"),
            session_id="session_2026_04_08_binary",
            received_at=1712577600.5,
        )


def test_binary_live_collection_persists_valid_packets(
    tmp_path: Path, repo_root: Path
) -> None:
    host, port = _reserve_udp_endpoint()
    warnings: list[str] = []
    messages: list[str] = []
    metadata = SessionMetadata.from_dict(
        {
            "session_id": "session_2026_04_08_binary_collect",
            "participant_id": "P03",
            "room_id": "bedroom_a",
            "layout_version": "layout_v1",
            "node_setup_version": "setup_triangle_v1",
            "activity_label": "fall",
            "recorded_at": "2026-04-08T10:00:00+09:00",
        }
    )
    source = UdpEsp32BinaryPacketSource(
        host,
        port,
        session_id=metadata.session_id,
        timeout_seconds=0.05,
        timestamp_fn=lambda: 1712577600.5,
        skip_invalid_datagrams=True,
        error_reporter=warnings.append,
    )
    session_store = SessionStore(tmp_path / "raw")

    sender = threading.Thread(
        target=_send_raw_udp_datagram,
        args=(host, port, _load_hex_fixture(repo_root, "esp32_adr018_valid_packet.hex")),
    )
    stopper = threading.Thread(target=_request_stop_later, args=(source, 0.10))
    now_values = iter([0.0, 0.1])

    sender.start()
    stopper.start()
    result = run_collection(
        session_store=session_store,
        metadata=metadata,
        packet_source=source,
        source_type="live",
        expected_nodes=("2", "3"),
        health_timeout_seconds=1.0,
        reporter=messages.append,
        now_fn=lambda: next(now_values),
    )
    sender.join(timeout=1.0)
    stopper.join(timeout=1.0)

    assert warnings == []
    assert result.packet_count == 1
    assert result.node_ids == ("2",)
    assert messages == [
        f"live event=start host={host} port={port} duration_seconds=none",
        "live event=warning status=degraded active=2 stale=none missing=3",
        "live event=stop reason=manual",
    ]

    stored_packets = [
        json.loads(line)
        for line in (result.session_dir / "packets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert stored_packets[0]["node_id"] == "2"
    assert stored_packets[0]["session_id"] == metadata.session_id
    assert stored_packets[0]["seq"] == 42
    assert stored_packets[0]["channel"] == 6
    assert stored_packets[0]["raw_payload"]["csi_raw"]["format"] == "esp32_adr018"


def test_binary_udp_receiver_can_skip_malformed_datagram_and_continue(
    repo_root: Path,
) -> None:
    host, port = _reserve_udp_endpoint()
    warnings: list[str] = []
    source = UdpEsp32BinaryPacketSource(
        host,
        port,
        session_id="session_2026_04_08_binary_skip",
        timestamp_fn=lambda: 1712577600.5,
        timeout_seconds=1.0,
        skip_invalid_datagrams=True,
        error_reporter=warnings.append,
    )
    sender = threading.Thread(
        target=_send_datagram_sequence,
        args=(
            host,
            port,
            [
                _load_hex_fixture(repo_root, "esp32_adr018_truncated_packet.hex"),
                _load_hex_fixture(repo_root, "esp32_adr018_valid_packet.hex"),
            ],
        ),
    )

    try:
        sender.start()
        packet = next(iter(source))
    finally:
        sender.join(timeout=1.0)
        source.close()

    assert packet.node_id == "2"
    assert packet.seq == 42
    assert warnings == [
        "ESP32 ADR-018 datagram payload length mismatch: expected 8 I/Q bytes, got 6."
    ]


def _load_hex_fixture(repo_root: Path, name: str) -> bytes:
    payload = (repo_root / "tests" / "fixtures" / name).read_text(encoding="utf-8")
    return bytes.fromhex(payload)


def _reserve_udp_endpoint() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()
    return str(host), int(port)


def _send_raw_udp_datagram(host: str, port: int, payload: bytes) -> None:
    time.sleep(0.02)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(payload, (host, port))


def _send_datagram_sequence(host: str, port: int, payloads: list[bytes]) -> None:
    time.sleep(0.02)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        for payload in payloads:
            sender.sendto(payload, (host, port))
            time.sleep(0.02)


def _request_stop_later(source: UdpEsp32BinaryPacketSource, delay_seconds: float) -> None:
    time.sleep(delay_seconds)
    source.request_stop()

import json
import socket
import threading
import time

import pytest

from collector.receiver import ReceiverError, UdpJsonPacketSource


def test_udp_receiver_yields_normalized_packet() -> None:
    host, port = _reserve_udp_endpoint()
    source = UdpJsonPacketSource(host, port, timeout_seconds=1.0)
    sender = threading.Thread(
        target=_send_udp_datagram,
        args=(
            host,
            port,
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": "session_2026_04_07_001",
                "seq": 1,
                "channel": 6,
                "rssi": -42,
                "amplitude": [0.1, 0.2, 0.3],
                "phase": [1.0, 1.1, 1.2],
                "csi_raw": {"source": "udp-test"},
            },
        ),
    )

    try:
        sender.start()
        packet = next(iter(source))
    finally:
        sender.join(timeout=1.0)
        source.close()

    assert packet.node_id == "node-a"
    assert packet.seq == 1
    assert packet.channel == 6
    assert packet.to_dict()["raw_payload"] == {"csi_raw": {"source": "udp-test"}}


def test_udp_receiver_raises_on_malformed_datagram() -> None:
    host, port = _reserve_udp_endpoint()
    source = UdpJsonPacketSource(host, port, timeout_seconds=1.0)
    sender = threading.Thread(
        target=_send_raw_udp_datagram,
        args=(host, port, b"not-json"),
    )

    try:
        sender.start()
        with pytest.raises(ReceiverError):
            next(iter(source))
    finally:
        sender.join(timeout=1.0)
        source.close()


def test_udp_receiver_can_skip_malformed_datagram_and_continue() -> None:
    host, port = _reserve_udp_endpoint()
    warnings: list[str] = []
    source = UdpJsonPacketSource(
        host,
        port,
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
                b"not-json",
                json.dumps(
                    {
                        "timestamp": 1712491800.1,
                        "node_id": "node-a",
                        "session_id": "session_2026_04_07_001",
                        "seq": 7,
                    }
                ).encode("utf-8"),
            ],
        ),
    )

    try:
        sender.start()
        packet = next(iter(source))
    finally:
        sender.join(timeout=1.0)
        source.close()

    assert packet.node_id == "node-a"
    assert packet.seq == 7
    assert warnings == ["UDP datagram must contain exactly one JSON object."]


def _reserve_udp_endpoint() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()
    return str(host), int(port)


def _send_udp_datagram(host: str, port: int, payload: dict[str, object]) -> None:
    _send_raw_udp_datagram(host, port, json.dumps(payload).encode("utf-8"))


def _send_raw_udp_datagram(host: str, port: int, payload: bytes) -> None:
    time.sleep(0.05)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(payload, (host, port))


def _send_datagram_sequence(host: str, port: int, payloads: list[bytes]) -> None:
    time.sleep(0.05)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        for payload in payloads:
            sender.sendto(payload, (host, port))
            time.sleep(0.02)

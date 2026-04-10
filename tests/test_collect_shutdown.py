import json
import socket
import threading
import time
from pathlib import Path

from collector.receiver import UdpJsonPacketSource
from collector.session_store import SessionStore
from scripts.collect import run_collection
from shared.models import SessionMetadata


def test_run_collection_stops_live_session_cleanly(tmp_path: Path) -> None:
    host, port = _reserve_udp_endpoint()
    source = UdpJsonPacketSource(host, port, timeout_seconds=0.05)
    messages: list[str] = []
    metadata = SessionMetadata.from_dict(
        {
            "session_id": "session_2026_04_07_live_shutdown",
            "participant_id": "P03",
            "room_id": "bedroom_a",
            "layout_version": "layout_v1",
            "node_setup_version": "setup_triangle_v1",
            "activity_label": "fall",
            "recorded_at": "2026-04-07T21:10:00+09:00",
        }
    )
    session_store = SessionStore(tmp_path / "raw")

    sender = threading.Thread(
        target=_send_udp_datagram,
        args=(
            host,
            port,
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": metadata.session_id,
                "seq": 1,
                "channel": 6,
                "rssi": -42,
                "amplitude": [0.1, 0.2, 0.3],
                "phase": [1.0, 1.1, 1.2],
                "csi_raw": {"source": "udp-test"},
            },
        ),
    )
    stopper = threading.Thread(target=_request_stop_later, args=(source, 0.10))

    sender.start()
    stopper.start()
    result = run_collection(
        session_store=session_store,
        metadata=metadata,
        packet_source=source,
        source_type="live",
        reporter=messages.append,
    )
    sender.join(timeout=1.0)
    stopper.join(timeout=1.0)

    assert result.packet_count == 1
    assert result.node_ids == ("node-a",)
    assert source._socket is None
    assert messages == [
        f"live event=start host={host} port={port} duration_seconds=none",
        "live event=stop reason=manual",
    ]

    metadata_path = result.session_dir / "metadata.json"
    packets_path = result.session_dir / "packets.jsonl"
    assert metadata_path.exists()
    assert packets_path.exists()

    stored_packets = [
        json.loads(line)
        for line in packets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stored_packets) == 1
    assert stored_packets[0]["session_id"] == metadata.session_id


def _reserve_udp_endpoint() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()
    return str(host), int(port)


def _send_udp_datagram(host: str, port: int, payload: dict[str, object]) -> None:
    time.sleep(0.02)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(json.dumps(payload).encode("utf-8"), (host, port))


def _request_stop_later(source: UdpJsonPacketSource, delay_seconds: float) -> None:
    time.sleep(delay_seconds)
    source.request_stop()

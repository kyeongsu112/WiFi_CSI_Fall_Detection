from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.packet_parser import parse_packet_dict
from collector.replay import JsonlReplaySource
from collector.receiver import UdpEsp32BinaryPacketSource, UdpJsonPacketSource
from collector.session_store import SessionStore
from scripts.collect import build_packet_source
from shared.config import CollectionConfig
from shared.models import SessionMetadata


def test_build_packet_source_selects_replay(repo_root: Path) -> None:
    source = build_packet_source(
        CollectionConfig(
            source_type="replay",
            replay_input_path="tests/fixtures/mock_session_packets.jsonl",
            udp_host="127.0.0.1",
            udp_port=9000,
            expected_nodes=["node-a", "node-b", "node-c"],
            session_output_dir="artifacts/raw",
            flush_interval_ms=1000,
        )
    )

    assert isinstance(source, JsonlReplaySource)
    assert source.path == repo_root / "tests" / "fixtures" / "mock_session_packets.jsonl"


def test_build_packet_source_selects_live_udp() -> None:
    source = build_packet_source(
        CollectionConfig(
            source_type="live",
            replay_input_path="tests/fixtures/mock_session_packets.jsonl",
            udp_host="127.0.0.1",
            udp_port=9100,
            expected_nodes=["node-a", "node-b", "node-c"],
            session_output_dir="artifacts/raw",
            flush_interval_ms=1000,
        )
    )

    try:
        assert isinstance(source, UdpJsonPacketSource)
        assert source.host == "127.0.0.1"
        assert source.port == 9100
    finally:
        source.close()


def test_build_packet_source_selects_live_binary_udp() -> None:
    source = build_packet_source(
        CollectionConfig(
            source_type="live",
            live_udp_format="esp32_adr018",
            replay_input_path="tests/fixtures/mock_session_packets.jsonl",
            udp_host="127.0.0.1",
            udp_port=9101,
            expected_nodes=["1", "2", "3"],
            session_output_dir="artifacts/raw",
            flush_interval_ms=1000,
        ),
        session_id="session_2026_04_08_binary_source",
    )

    try:
        assert isinstance(source, UdpEsp32BinaryPacketSource)
        assert source.host == "127.0.0.1"
        assert source.port == 9101
        assert source.session_id == "session_2026_04_08_binary_source"
    finally:
        source.close()


def test_build_packet_source_rejects_unsupported_source_type() -> None:
    invalid_config = SimpleNamespace(
        source_type="serial",
        replay_input_path="tests/fixtures/mock_session_packets.jsonl",
        udp_host="127.0.0.1",
        udp_port=9000,
    )

    with pytest.raises(ValueError, match="Unsupported collection source_type"):
        build_packet_source(invalid_config)


def test_build_packet_source_rejects_binary_live_without_session_id() -> None:
    with pytest.raises(ValueError, match="requires a session_id"):
        build_packet_source(
            CollectionConfig(
                source_type="live",
                live_udp_format="esp32_adr018",
                replay_input_path="tests/fixtures/mock_session_packets.jsonl",
                udp_host="127.0.0.1",
                udp_port=9102,
                expected_nodes=["1", "2", "3"],
                session_output_dir="artifacts/raw",
                flush_interval_ms=1000,
            )
        )


def test_run_collection_reports_live_health_messages_consistently(
    tmp_path: Path,
) -> None:
    from scripts.collect import run_collection

    messages: list[str] = []
    metadata = SessionMetadata.from_dict(
        {
            "session_id": "session_2026_04_07_live_health_logging",
            "participant_id": "P03",
            "room_id": "bedroom_a",
            "layout_version": "layout_v1",
            "node_setup_version": "setup_triangle_v1",
            "activity_label": "fall",
            "recorded_at": "2026-04-07T21:10:00+09:00",
        }
    )
    session_store = SessionStore(tmp_path / "raw")
    source = SequenceLivePacketSource(
        [
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": metadata.session_id,
            },
            {
                "timestamp": 1712491801.5,
                "node_id": "node-a",
                "session_id": metadata.session_id,
            },
            {
                "timestamp": 1712491801.7,
                "node_id": "node-b",
                "session_id": metadata.session_id,
            },
        ]
    )
    now_values = iter([0.0, 0.1, 1.6, 1.7])

    result = run_collection(
        session_store=session_store,
        metadata=metadata,
        packet_source=source,
        source_type="live",
        expected_nodes=("node-a", "node-b"),
        health_timeout_seconds=1.0,
        reporter=messages.append,
        now_fn=lambda: next(now_values),
    )

    assert result.packet_count == 3
    assert messages == [
        "live event=start host=127.0.0.1 port=9999 duration_seconds=none",
        "live event=warning status=degraded active=node-a stale=none missing=node-b",
        "live event=health status=live active=node-a,node-b stale=none missing=none",
        "live event=stop reason=completed",
    ]


class SequenceLivePacketSource:
    def __init__(self, packets: list[dict[str, object]]) -> None:
        self.host = "127.0.0.1"
        self.port = 9999
        self._packets = [parse_packet_dict(packet) for packet in packets]
        self.closed = False
        self._stop_requested = False

    def __iter__(self):
        for packet in self._packets:
            if self._stop_requested:
                break
            yield packet

    def request_stop(self) -> None:
        self._stop_requested = True

    def close(self) -> None:
        self.closed = True
        self._stop_requested = True

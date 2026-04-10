import json
import threading
import time
from pathlib import Path

from collector.replay import JsonlReplaySource
from collector.session_store import SessionStore
from collector.packet_parser import parse_packet_dict
from scripts.collect import run_collection
from shared.models import SessionMetadata


class CountingLivePacketSource:
    def __init__(self, session_id: str, *, sleep_seconds: float = 0.01) -> None:
        self.session_id = session_id
        self.sleep_seconds = sleep_seconds
        self.closed = False
        self.request_stop_calls = 0
        self._stop_requested = False
        self._seq = 0

    def __iter__(self):
        while not self._stop_requested:
            self._seq += 1
            yield parse_packet_dict(
                {
                    "timestamp": float(self._seq),
                    "node_id": "node-a",
                    "session_id": self.session_id,
                    "seq": self._seq,
                    "channel": 6,
                }
            )
            time.sleep(self.sleep_seconds)

    def request_stop(self) -> None:
        self.request_stop_calls += 1
        self._stop_requested = True

    def close(self) -> None:
        self.closed = True
        self._stop_requested = True


def test_live_collection_stops_after_configured_duration(tmp_path: Path) -> None:
    metadata = _build_metadata("session_2026_04_07_duration_stop")
    source = CountingLivePacketSource(metadata.session_id)
    session_store = SessionStore(tmp_path / "raw")
    messages: list[str] = []

    result = run_collection(
        session_store=session_store,
        metadata=metadata,
        packet_source=source,
        source_type="live",
        live_session_duration_seconds=0.05,
        reporter=messages.append,
    )

    assert result.packet_count >= 1
    assert source.request_stop_calls == 1
    assert source.closed is True
    assert messages == [
        "live event=start duration_seconds=0.05",
        "live event=stop reason=duration duration_seconds=0.05",
    ]


def test_live_collection_continues_until_manual_stop_when_duration_is_unset(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata("session_2026_04_07_duration_unset")
    source = CountingLivePacketSource(metadata.session_id)
    session_store = SessionStore(tmp_path / "raw")
    stopper = threading.Thread(target=_request_stop_later, args=(source, 0.05))
    messages: list[str] = []

    stopper.start()
    result = run_collection(
        session_store=session_store,
        metadata=metadata,
        packet_source=source,
        source_type="live",
        live_session_duration_seconds=None,
        reporter=messages.append,
    )
    stopper.join(timeout=1.0)

    assert result.packet_count >= 1
    assert source.request_stop_calls == 1
    assert source.closed is True
    assert messages == [
        "live event=start duration_seconds=none",
        "live event=stop reason=manual",
    ]


def test_replay_collection_is_unchanged_when_live_duration_is_set(
    tmp_path: Path, fixture_packets_path: Path, fixture_metadata_path: Path
) -> None:
    metadata_payload = json.loads(fixture_metadata_path.read_text(encoding="utf-8"))
    metadata = SessionMetadata.from_dict(metadata_payload)
    session_store = SessionStore(tmp_path / "raw")

    result = run_collection(
        session_store=session_store,
        metadata=metadata,
        packet_source=JsonlReplaySource(fixture_packets_path),
        source_type="replay",
        live_session_duration_seconds=0.01,
    )

    assert result.packet_count == 6
    assert result.node_ids == ("node-a", "node-b", "node-c")


def _build_metadata(session_id: str) -> SessionMetadata:
    return SessionMetadata.from_dict(
        {
            "session_id": session_id,
            "participant_id": "P03",
            "room_id": "bedroom_a",
            "layout_version": "layout_v1",
            "node_setup_version": "setup_triangle_v1",
            "activity_label": "fall",
            "recorded_at": "2026-04-07T21:10:00+09:00",
        }
    )


def _request_stop_later(source: CountingLivePacketSource, delay_seconds: float) -> None:
    time.sleep(delay_seconds)
    source.request_stop()

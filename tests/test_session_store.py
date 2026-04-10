import json
from pathlib import Path

import pytest

from collector.replay import JsonlReplaySource
from collector.session_store import SessionStore
from shared.models import SessionMetadata


def test_session_store_persists_metadata_and_packets(
    tmp_path: Path, fixture_packets_path: Path, fixture_metadata_path: Path
) -> None:
    metadata_payload = json.loads(fixture_metadata_path.read_text(encoding="utf-8"))
    metadata = SessionMetadata.from_dict(metadata_payload)
    store = SessionStore(tmp_path / "raw")

    result = store.write_session(metadata, JsonlReplaySource(fixture_packets_path))

    assert result.packet_count == 6
    assert result.node_ids == ("node-a", "node-b", "node-c")

    metadata_path = result.session_dir / "metadata.json"
    packets_path = result.session_dir / "packets.jsonl"

    assert metadata_path.exists()
    assert packets_path.exists()

    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stored_packets = [
        json.loads(line)
        for line in packets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert stored_metadata["session_id"] == metadata.session_id
    assert len(stored_packets) == 6
    assert stored_packets[0]["node_id"] == "node-a"
    assert stored_packets[0]["raw_payload"] == {
        "csi_raw": {"format": "normalized_fixture", "packet": 1}
    }


def test_session_metadata_extra_is_frozen() -> None:
    metadata = SessionMetadata.from_dict(
        {
            "session_id": "session_2026_04_07_001",
            "participant_id": "P03",
            "room_id": "bedroom_a",
            "layout_version": "layout_v1",
            "node_setup_version": "setup_triangle_v1",
            "activity_label": "fall",
            "recorded_at": "2026-04-07T21:10:00+09:00",
            "custom": {"tags": ["fixture", "phase1"]},
        }
    )

    with pytest.raises(TypeError):
        metadata.extra["new_key"] = "blocked"  # type: ignore[index]

    with pytest.raises(TypeError):
        metadata.extra["custom"]["tags"] = []  # type: ignore[index]

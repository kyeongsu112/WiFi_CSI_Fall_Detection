import json
from pathlib import Path

import pytest

from preprocessing.session_loader import RawSessionLoadError, load_raw_session
from shared.models import SessionMetadata


def test_load_raw_session_roundtrip(
    tmp_path: Path,
    fixture_packets_path: Path,
    fixture_metadata_path: Path,
) -> None:
    session_dir = tmp_path / "raw" / "session_2026_04_07_001"
    session_dir.mkdir(parents=True)
    session_dir.joinpath("metadata.json").write_text(
        fixture_metadata_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    payload_lines = fixture_packets_path.read_text(encoding="utf-8").splitlines()
    session_dir.joinpath("packets.jsonl").write_text(
        "\n".join(reversed(payload_lines)) + "\n",
        encoding="utf-8",
    )

    raw_session = load_raw_session(session_dir)

    assert raw_session.metadata.session_id == "session_2026_04_07_001"
    assert len(raw_session.packets) == 6
    assert raw_session.packets[0].timestamp == 1712491800.1
    assert raw_session.packets[-1].timestamp == 1712491800.22


def test_load_raw_session_missing_packets_file_fails(tmp_path: Path) -> None:
    session_dir = tmp_path / "raw" / "session_missing_packets"
    session_dir.mkdir(parents=True)
    session_dir.joinpath("metadata.json").write_text(
        json.dumps(
            SessionMetadata.from_dict(
                {
                    "session_id": "session_missing_packets",
                    "participant_id": "P03",
                    "room_id": "bedroom_a",
                    "layout_version": "layout_v1",
                    "node_setup_version": "setup_triangle_v1",
                    "activity_label": "fall",
                    "recorded_at": "2026-04-07T21:10:00+09:00",
                }
            ).to_dict()
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="packets.jsonl"):
        load_raw_session(session_dir)


def test_load_raw_session_rejects_packet_session_mismatch(tmp_path: Path) -> None:
    session_dir = tmp_path / "raw" / "session_mismatch"
    session_dir.mkdir(parents=True)
    session_dir.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "session_id": "session_mismatch",
                "participant_id": "P03",
                "room_id": "bedroom_a",
                "layout_version": "layout_v1",
                "node_setup_version": "setup_triangle_v1",
                "activity_label": "fall",
                "recorded_at": "2026-04-07T21:10:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    session_dir.joinpath("packets.jsonl").write_text(
        json.dumps(
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": "different_session",
                "seq": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RawSessionLoadError, match="does not match metadata session_id"):
        load_raw_session(session_dir)

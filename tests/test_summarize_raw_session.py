from pathlib import Path

from collector.packet_parser import parse_packet_dict
from collector.session_store import SessionStore
from scripts.summarize_raw_session import summarize_raw_session
from shared.models import SessionMetadata


def test_summarize_raw_session_reports_expected_nodes_and_timestamps(tmp_path: Path) -> None:
    metadata = SessionMetadata.from_dict(
        {
            "session_id": "session_2026_04_08_summary",
            "participant_id": "P03",
            "room_id": "bedroom_a",
            "layout_version": "layout_v1",
            "node_setup_version": "setup_triangle_v1",
            "activity_label": "fall",
            "recorded_at": "2026-04-08T10:00:00+09:00",
        }
    )
    packets = [
        parse_packet_dict(
            {
                "timestamp": 1712577600.5,
                "node_id": "1",
                "session_id": metadata.session_id,
                "seq": 1,
            }
        ),
        parse_packet_dict(
            {
                "timestamp": 1712577601.5,
                "node_id": "2",
                "session_id": metadata.session_id,
                "seq": 2,
            }
        ),
        parse_packet_dict(
            {
                "timestamp": 1712577602.5,
                "node_id": "1",
                "session_id": metadata.session_id,
                "seq": 3,
            }
        ),
    ]

    result = SessionStore(tmp_path / "raw").write_session(metadata, packets)
    summary = summarize_raw_session(result.session_dir, expected_nodes=("1", "2", "3"))

    assert summary.packet_count == 3
    assert summary.seen_nodes == ("1", "2")
    assert summary.missing_expected_nodes == ("3",)
    assert summary.first_timestamp == 1712577600.5
    assert summary.last_timestamp == 1712577602.5
    assert summary.node_packet_counts == {"1": 2, "2": 1}

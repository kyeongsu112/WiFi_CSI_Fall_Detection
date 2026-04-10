from pathlib import Path

import pytest

from collector.replay import JsonlReplaySource, ReplaySourceError


def test_replay_reads_fixture_session(fixture_packets_path: Path) -> None:
    packets = list(JsonlReplaySource(fixture_packets_path))

    assert len(packets) == 6
    assert {packet.node_id for packet in packets} == {"node-a", "node-b", "node-c"}
    assert {packet.session_id for packet in packets} == {"session_2026_04_07_001"}


def test_replay_raises_on_malformed_jsonl(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"timestamp": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ReplaySourceError):
        list(JsonlReplaySource(bad_path))

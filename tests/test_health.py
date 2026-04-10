from collector.health import (
    LiveNodeHealthTracker,
    format_live_status,
    observe_live_packet_source,
)
from collector.packet_parser import parse_packet_dict


def test_live_health_reports_healthy_activity() -> None:
    messages: list[str] = []
    packets = [
        parse_packet_dict(
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": "session_2026_04_07_001",
            }
        ),
        parse_packet_dict(
            {
                "timestamp": 1712491800.2,
                "node_id": "node-b",
                "session_id": "session_2026_04_07_001",
            }
        ),
    ]
    now_values = iter([0.0, 0.1, 0.2])

    observed_packets = list(
        observe_live_packet_source(
            packets,
            expected_nodes=("node-a", "node-b"),
            timeout_seconds=1.0,
            reporter=messages.append,
            now_fn=lambda: next(now_values),
        )
    )

    assert len(observed_packets) == 2
    assert messages == [
        "status=degraded active=node-a stale=none missing=node-b",
        "status=live active=node-a,node-b stale=none missing=none",
    ]


def test_live_health_reports_degraded_when_expected_node_times_out() -> None:
    messages: list[str] = []
    packets = [
        parse_packet_dict(
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": "session_2026_04_07_001",
            }
        ),
        parse_packet_dict(
            {
                "timestamp": 1712491801.5,
                "node_id": "node-a",
                "session_id": "session_2026_04_07_001",
            }
        ),
    ]
    now_values = iter([0.0, 0.1, 1.6])

    list(
        observe_live_packet_source(
            packets,
            expected_nodes=("node-a", "node-b"),
            timeout_seconds=1.0,
            reporter=messages.append,
            now_fn=lambda: next(now_values),
        )
    )

    assert messages == [
        "status=degraded active=node-a stale=none missing=node-b",
    ]


def test_live_health_recovers_to_live_when_timed_out_node_resumes() -> None:
    messages: list[str] = []
    packets = [
        parse_packet_dict(
            {
                "timestamp": 1712491800.1,
                "node_id": "node-a",
                "session_id": "session_2026_04_07_001",
            }
        ),
        parse_packet_dict(
            {
                "timestamp": 1712491801.5,
                "node_id": "node-a",
                "session_id": "session_2026_04_07_001",
            }
        ),
        parse_packet_dict(
            {
                "timestamp": 1712491801.7,
                "node_id": "node-b",
                "session_id": "session_2026_04_07_001",
            }
        ),
    ]
    now_values = iter([0.0, 0.1, 1.6, 1.7])

    list(
        observe_live_packet_source(
            packets,
            expected_nodes=("node-a", "node-b"),
            timeout_seconds=1.0,
            reporter=messages.append,
            now_fn=lambda: next(now_values),
        )
    )

    assert messages == [
        "status=degraded active=node-a stale=none missing=node-b",
        "status=live active=node-a,node-b stale=none missing=none",
    ]


def test_live_health_tracker_snapshot_format_is_deterministic() -> None:
    tracker = LiveNodeHealthTracker(
        expected_nodes=("node-b", "node-a"),
        timeout_seconds=1.0,
        started_at=0.0,
    )

    tracker.record_packet("node-b", 0.2)
    status = tracker.snapshot(1.5)

    assert format_live_status(status) == (
        "status=degraded active=none stale=node-b missing=node-a"
    )

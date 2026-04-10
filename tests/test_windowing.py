from collector.packet_parser import parse_packet_dict
from preprocessing.decoder import decode_packets
from preprocessing.windowing import build_windows


def test_build_windows_is_deterministic_for_multi_node_packets() -> None:
    packets = decode_packets(
        [
            parse_packet_dict(
                {
                    "timestamp": 0.0,
                    "node_id": "1",
                    "session_id": "session_w",
                    "amplitude": [1.0, 2.0],
                    "phase": [0.1, 0.2],
                }
            ),
            parse_packet_dict(
                {
                    "timestamp": 0.4,
                    "node_id": "2",
                    "session_id": "session_w",
                    "amplitude": [1.2, 2.2],
                    "phase": [0.2, 0.3],
                }
            ),
            parse_packet_dict(
                {
                    "timestamp": 0.9,
                    "node_id": "1",
                    "session_id": "session_w",
                    "amplitude": [1.1, 2.1],
                    "phase": [0.2, 0.3],
                }
            ),
            parse_packet_dict(
                {
                    "timestamp": 1.1,
                    "node_id": "2",
                    "session_id": "session_w",
                    "amplitude": [1.3, 2.3],
                    "phase": [0.4, 0.5],
                }
            ),
        ],
        phase_unwrap_enabled=False,
    )

    windows = build_windows(
        packets,
        session_id="session_w",
        window_seconds=1.0,
        stride_seconds=0.5,
        expected_nodes=("1", "2", "3"),
    )

    assert [window.window_id for window in windows] == [
        "session_w_w00000",
        "session_w_w00001",
        "session_w_w00002",
    ]
    assert [len(windows[0].packets_by_node["1"]), len(windows[0].packets_by_node["2"])] == [2, 1]
    assert [len(windows[1].packets_by_node["1"]), len(windows[1].packets_by_node["2"])] == [1, 1]
    assert [len(windows[2].packets_by_node["1"]), len(windows[2].packets_by_node["2"])] == [0, 1]
    assert windows[0].node_order == ("1", "2", "3")


def test_build_windows_short_session_still_yields_one_window() -> None:
    packets = decode_packets(
        [
            parse_packet_dict(
                {
                    "timestamp": 10.0,
                    "node_id": "1",
                    "session_id": "session_short",
                    "amplitude": [1.0],
                    "phase": [0.1],
                }
            )
        ],
        phase_unwrap_enabled=False,
    )

    windows = build_windows(
        packets,
        session_id="session_short",
        window_seconds=2.0,
        stride_seconds=0.5,
        expected_nodes=("1",),
    )

    assert len(windows) == 1
    assert windows[0].start_ts == 10.0
    assert windows[0].end_ts == 12.0


def test_build_windows_skips_globally_empty_ranges() -> None:
    packets = decode_packets(
        [
            parse_packet_dict(
                {
                    "timestamp": 0.0,
                    "node_id": "1",
                    "session_id": "session_gap",
                    "amplitude": [1.0],
                    "phase": [0.1],
                }
            ),
            parse_packet_dict(
                {
                    "timestamp": 2.0,
                    "node_id": "1",
                    "session_id": "session_gap",
                    "amplitude": [1.1],
                    "phase": [0.2],
                }
            ),
        ],
        phase_unwrap_enabled=False,
    )

    windows = build_windows(
        packets,
        session_id="session_gap",
        window_seconds=0.4,
        stride_seconds=0.5,
        expected_nodes=("1",),
    )

    assert [window.window_id for window in windows] == [
        "session_gap_w00000",
        "session_gap_w00004",
    ]

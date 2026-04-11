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


def test_no_degenerate_trailing_window_when_last_ts_already_covered() -> None:
    """No trailing window is generated when last_ts falls inside the last stride window.

    Policy: ``_build_window_starts`` uses exclusive ``<`` to avoid degenerate
    sparse trailing windows.  A trailing window at ``last_ts`` is added ONLY
    when ``last_ts >= last_window_start + window_seconds``, i.e., when
    ``last_ts`` is NOT already covered.

    Setup: packets at 0.0, 0.5, 1.0 with window=1.0, stride=0.5.
      - Stride windows (exclusive <): [0.0, 1.0), [0.5, 1.5)
      - last_ts=1.0 is inside [0.5, 1.5) → no trailing window at 1.0
      - Packet at ts=1.0 is captured in window [0.5, 1.5) without a new window.
    """
    packets = decode_packets(
        [
            parse_packet_dict({
                "timestamp": 0.0,
                "node_id": "1",
                "session_id": "sess",
                "amplitude": [1.0],
                "phase": [0.0],
            }),
            parse_packet_dict({
                "timestamp": 0.5,
                "node_id": "1",
                "session_id": "sess",
                "amplitude": [1.0],
                "phase": [0.0],
            }),
            # last_ts == 1.0, already covered by window [0.5, 1.5)
            parse_packet_dict({
                "timestamp": 1.0,
                "node_id": "1",
                "session_id": "sess",
                "amplitude": [1.0],
                "phase": [0.0],
            }),
        ],
        phase_unwrap_enabled=False,
    )

    windows = build_windows(
        packets,
        session_id="sess",
        window_seconds=1.0,
        stride_seconds=0.5,
        expected_nodes=("1",),
    )

    start_times = [w.start_ts for w in windows]

    # No degenerate trailing window at 1.0 — it would contain only 1 packet
    # and last_ts is already covered by the preceding window [0.5, 1.5).
    assert 1.0 not in start_times, (
        "Degenerate trailing window at last_ts must NOT be generated when "
        "last_ts is already covered by the preceding window"
    )
    # Normal stride windows are still generated.
    assert 0.0 in start_times
    assert 0.5 in start_times

    # The packet at ts=1.0 is captured inside window [0.5, 1.5).
    w05 = next(w for w in windows if w.start_ts == 0.5)
    assert len(w05.packets_by_node["1"]) == 2, (
        "Window [0.5, 1.5) must contain packets at ts=0.5 AND ts=1.0"
    )


def test_trailing_window_added_when_last_ts_not_covered() -> None:
    """When last_ts is not covered by any stride window, a trailing window IS added.

    This prevents data loss for isolated packets that arrive after a data gap
    wider than window_seconds (common when stride >= window_seconds).

    Setup: packets at 0.0 and 2.0, window=0.4, stride=0.5.
      - Stride windows (exclusive <): 0.0, 0.5, 1.0, 1.5
      - Last stride window end = 1.5 + 0.4 = 1.9
      - last_ts=2.0 >= 1.9 → trailing window at 2.0 IS added
      - Packet at ts=2.0 captured in window [2.0, 2.4).
    """
    packets = decode_packets(
        [
            parse_packet_dict({
                "timestamp": 0.0,
                "node_id": "1",
                "session_id": "trail",
                "amplitude": [1.0],
                "phase": [0.0],
            }),
            parse_packet_dict({
                "timestamp": 2.0,
                "node_id": "1",
                "session_id": "trail",
                "amplitude": [1.1],
                "phase": [0.1],
            }),
        ],
        phase_unwrap_enabled=False,
    )

    windows = build_windows(
        packets,
        session_id="trail",
        window_seconds=0.4,
        stride_seconds=0.5,
        expected_nodes=("1",),
    )

    start_times = [w.start_ts for w in windows]
    assert 2.0 in start_times, (
        "Trailing window at last_ts must be generated when last_ts is not "
        "covered by any stride window (i.e., after a data gap)"
    )

    trailing = next(w for w in windows if w.start_ts == 2.0)
    assert len(trailing.packets_by_node["1"]) == 1, (
        "Packet at ts=2.0 must be captured in the trailing window"
    )


def test_single_packet_session_always_yields_one_window() -> None:
    """Short-session path: one packet → exactly one window (even if sparse)."""
    packets = decode_packets(
        [
            parse_packet_dict({
                "timestamp": 5.0,
                "node_id": "1",
                "session_id": "single",
                "amplitude": [1.0],
                "phase": [0.0],
            }),
        ],
        phase_unwrap_enabled=False,
    )

    windows = build_windows(
        packets,
        session_id="single",
        window_seconds=1.0,
        stride_seconds=1.0,
        expected_nodes=("1",),
    )

    assert len(windows) == 1, "Single-packet session must yield exactly one window"
    assert windows[0].start_ts == 5.0

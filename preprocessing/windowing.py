from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from preprocessing.decoder import DecodedCsiPacket


@dataclass(frozen=True, slots=True)
class DecodedWindow:
    window_id: str
    session_id: str
    start_ts: float
    end_ts: float
    node_order: tuple[str, ...]
    packets_by_node: Mapping[str, tuple[DecodedCsiPacket, ...]]

    def __post_init__(self) -> None:
        frozen_mapping = {
            str(node_id): tuple(packets)
            for node_id, packets in self.packets_by_node.items()
        }
        object.__setattr__(self, "packets_by_node", MappingProxyType(frozen_mapping))


def resolve_node_order(
    packets: Sequence[DecodedCsiPacket],
    *,
    expected_nodes: Sequence[str] = (),
) -> tuple[str, ...]:
    node_order: list[str] = []
    for node_id in expected_nodes:
        normalized = str(node_id)
        if normalized not in node_order:
            node_order.append(normalized)

    observed_nodes = sorted({packet.node_id for packet in packets})
    for node_id in observed_nodes:
        if node_id not in node_order:
            node_order.append(node_id)
    return tuple(node_order)


def build_windows(
    packets: Sequence[DecodedCsiPacket],
    *,
    session_id: str,
    window_seconds: float,
    stride_seconds: float,
    expected_nodes: Sequence[str] = (),
) -> tuple[DecodedWindow, ...]:
    if not packets:
        return ()

    ordered_packets = tuple(
        sorted(
            packets,
            key=lambda packet: (
                packet.timestamp,
                packet.node_id,
                packet.seq if packet.seq is not None else -1,
            ),
        )
    )
    node_order = resolve_node_order(ordered_packets, expected_nodes=expected_nodes)

    first_ts = ordered_packets[0].timestamp
    last_ts = ordered_packets[-1].timestamp
    window_starts = _build_window_starts(
        first_ts=first_ts,
        last_ts=last_ts,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )

    windows: list[DecodedWindow] = []
    for window_index, start_ts in enumerate(window_starts):
        end_ts = start_ts + window_seconds
        grouped_packets = {node_id: [] for node_id in node_order}
        for packet in ordered_packets:
            if packet.timestamp < start_ts:
                continue
            if packet.timestamp >= end_ts:
                continue
            grouped_packets.setdefault(packet.node_id, []).append(packet)

        if not any(grouped_packets.values()):
            continue

        windows.append(
            DecodedWindow(
                window_id=f"{session_id}_w{window_index:05d}",
                session_id=session_id,
                start_ts=start_ts,
                end_ts=end_ts,
                node_order=node_order,
                packets_by_node={
                    node_id: tuple(grouped_packets.get(node_id, ()))
                    for node_id in node_order
                },
            )
        )

    return tuple(windows)


def _build_window_starts(
    *,
    first_ts: float,
    last_ts: float,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[float, ...]:
    """Return the sequence of window start timestamps.

    Policy
    ------
    Short sessions (``last_ts - first_ts < window_seconds``):
        Exactly one window starting at ``first_ts``.  The window will be
        sparse but is always emitted so callers get at least one sample.

    Normal sessions:
        Windows are placed at ``first_ts, first_ts + stride, …`` while
        ``start_ts < last_ts`` (exclusive upper bound).

        A trailing window at ``last_ts`` is appended ONLY when
        ``last_ts >= last_window_start + window_seconds`` — i.e., when
        ``last_ts`` is NOT already covered by the final stride window.  This
        happens when ``stride >= window_size`` (non-overlapping windows) and
        the session ends after a data gap wider than one window, leaving an
        isolated packet that would otherwise be lost entirely.

        In the common case where windows overlap (``stride < window_size``),
        ``last_ts`` always falls inside the preceding window's range so no
        trailing window is generated.  This eliminates degenerate 1-packet
        trailing windows that would produce unreliable inference results.

    Examples
    --------
    Packets at [0.0, 0.5, 1.0], window=1.0, stride=0.5 (overlapping):
        Windows at 0.0 [0.0,1.0) and 0.5 [0.5,1.5).
        ``last_ts=1.0`` is inside [0.5, 1.5) → no trailing window.
        Packet at 1.0 captured in window [0.5, 1.5).

    Packets at [0.0, 2.0], window=0.4, stride=0.5 (gap, non-overlapping):
        Windows at 0.0, 0.5, 1.0, 1.5 from the ``<`` loop.
        ``last_ts=2.0 >= 1.5 + 0.4 = 1.9`` → trailing window at 2.0 added.
        Packet at 2.0 captured in window [2.0, 2.4).
    """
    if last_ts - first_ts < window_seconds:
        return (first_ts,)

    window_starts: list[float] = []
    start_ts = first_ts
    while start_ts < last_ts:   # exclusive: prevent degenerate window at last_ts
        window_starts.append(start_ts)
        start_ts += stride_seconds

    # Add a trailing window only when last_ts is not covered by the last
    # stride window.  This preserves isolated packets after a data gap while
    # avoiding sparse trailing windows in the normal overlapping-window case.
    if not window_starts or last_ts >= window_starts[-1] + window_seconds:
        window_starts.append(last_ts)

    return tuple(window_starts)


# ── WiFall row-count windowing ───────────────────────────────────────────────

def compute_window_count(n_samples: int, window_size: int, stride: int) -> int:
    """Return the number of complete windows that fit in n_samples rows.

    Only full windows are counted; partial trailing rows are dropped.

    Args:
        n_samples:   total number of data rows in the source CSV.
        window_size: number of rows per window (e.g. 100 for 1 s at 100 Hz).
        stride:      row offset between consecutive window starts.

    Returns:
        Number of complete windows (>= 0).

    Examples:
        >>> compute_window_count(100, 100, 100)
        1
        >>> compute_window_count(253, 100, 100)
        2
        >>> compute_window_count(99, 100, 100)
        0
    """
    if n_samples < window_size:
        return 0
    return (n_samples - window_size) // stride + 1

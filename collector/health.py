from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

from shared.models import CsiPacket


Reporter = Callable[[str], None]
NowFunction = Callable[[], float]


@dataclass(frozen=True, slots=True)
class LiveNodeStatus:
    status: str
    active_nodes: tuple[str, ...]
    stale_nodes: tuple[str, ...]
    missing_nodes: tuple[str, ...]


class LiveNodeHealthTracker:
    """Tracks expected-node activity for live collection visibility."""

    def __init__(
        self,
        expected_nodes: Sequence[str],
        timeout_seconds: float,
        *,
        started_at: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("health timeout_seconds must be positive.")

        self.expected_nodes = tuple(dict.fromkeys(expected_nodes))
        self.timeout_seconds = timeout_seconds
        self.started_at = started_at
        self._last_seen: dict[str, float] = {}

    def record_packet(self, node_id: str, observed_at: float) -> LiveNodeStatus:
        if node_id in self.expected_nodes:
            self._last_seen[node_id] = observed_at
        return self.snapshot(observed_at)

    def snapshot(self, observed_at: float) -> LiveNodeStatus:
        active_nodes: list[str] = []
        stale_nodes: list[str] = []
        missing_nodes: list[str] = []

        for node_id in self.expected_nodes:
            if node_id not in self._last_seen:
                missing_nodes.append(node_id)
                continue

            last_seen = self._last_seen[node_id]
            if observed_at - last_seen > self.timeout_seconds:
                stale_nodes.append(node_id)
            else:
                active_nodes.append(node_id)

        status = "degraded" if stale_nodes or missing_nodes else "live"
        return LiveNodeStatus(
            status=status,
            active_nodes=tuple(active_nodes),
            stale_nodes=tuple(stale_nodes),
            missing_nodes=tuple(missing_nodes),
        )


def observe_live_packet_source(
    packet_source: Iterable[CsiPacket],
    *,
    expected_nodes: Sequence[str],
    timeout_seconds: float,
    reporter: Reporter,
    now_fn: NowFunction,
) -> Iterator[CsiPacket]:
    tracker = LiveNodeHealthTracker(
        expected_nodes=expected_nodes,
        timeout_seconds=timeout_seconds,
        started_at=now_fn(),
    )
    previous_message: str | None = None

    for packet in packet_source:
        status = tracker.record_packet(packet.node_id, now_fn())
        message = format_live_status(status)
        if message != previous_message:
            reporter(message)
            previous_message = message
        yield packet


def format_live_status(status: LiveNodeStatus) -> str:
    active = ",".join(status.active_nodes) if status.active_nodes else "none"
    stale = ",".join(status.stale_nodes) if status.stale_nodes else "none"
    missing = ",".join(status.missing_nodes) if status.missing_nodes else "none"
    return f"status={status.status} active={active} stale={stale} missing={missing}"

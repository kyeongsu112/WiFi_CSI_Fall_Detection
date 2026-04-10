from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence


def bootstrap_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = bootstrap_repo_root()


@dataclass(frozen=True, slots=True)
class RawSessionSummary:
    session_dir: Path
    packet_count: int
    seen_nodes: tuple[str, ...]
    missing_expected_nodes: tuple[str, ...]
    first_timestamp: float | None
    last_timestamp: float | None
    node_packet_counts: dict[str, int]


def summarize_raw_session(
    session_dir: Path,
    *,
    expected_nodes: Sequence[str] = (),
) -> RawSessionSummary:
    session_dir = Path(session_dir)
    packets_path = session_dir / "packets.jsonl"
    if not packets_path.exists():
        raise FileNotFoundError(f"Raw session packets file not found: {packets_path}")

    packet_count = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    node_packet_counts: OrderedDict[str, int] = OrderedDict()

    with packets_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            packet_count += 1

            node_id = str(payload["node_id"])
            node_packet_counts[node_id] = node_packet_counts.get(node_id, 0) + 1

            timestamp = float(payload["timestamp"])
            if first_timestamp is None or timestamp < first_timestamp:
                first_timestamp = timestamp
            if last_timestamp is None or timestamp > last_timestamp:
                last_timestamp = timestamp

    seen_nodes = tuple(node_packet_counts.keys())
    expected = tuple(str(node_id) for node_id in expected_nodes)
    missing_expected_nodes = tuple(node_id for node_id in expected if node_id not in node_packet_counts)

    return RawSessionSummary(
        session_dir=session_dir,
        packet_count=packet_count,
        seen_nodes=seen_nodes,
        missing_expected_nodes=missing_expected_nodes,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        node_packet_counts=dict(node_packet_counts),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a captured raw session directory."
    )
    parser.add_argument(
        "--session-dir",
        required=True,
        help="Path to artifacts/raw/<session_id>.",
    )
    parser.add_argument(
        "--expected-nodes",
        default="",
        help="Optional comma-separated expected node IDs, e.g. 1,2,3",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected_nodes = _parse_expected_nodes(args.expected_nodes)
    summary = summarize_raw_session(
        resolve_repo_path(args.session_dir),
        expected_nodes=expected_nodes,
    )

    print(f"session_dir={summary.session_dir}")
    print(f"packet_count={summary.packet_count}")
    print("seen_nodes=" + _format_csv(summary.seen_nodes))
    print("missing_expected_nodes=" + _format_csv(summary.missing_expected_nodes))
    print(f"first_timestamp={_format_optional_float(summary.first_timestamp)}")
    print(f"last_timestamp={_format_optional_float(summary.last_timestamp)}")
    print(
        "node_packet_counts="
        + ",".join(f"{node}:{count}" for node, count in summary.node_packet_counts.items())
    )
    return 0


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _parse_expected_nodes(raw_value: str) -> tuple[str, ...]:
    if not raw_value.strip():
        return ()
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _format_csv(values: Iterable[str]) -> str:
    values = tuple(values)
    if not values:
        return "none"
    return ",".join(values)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())

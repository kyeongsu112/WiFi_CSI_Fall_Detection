from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from collector.packet_parser import PacketParseError, parse_packet_dict
from shared.models import CsiPacket, SessionMetadata


class RawSessionLoadError(RuntimeError):
    """Raised when a stored raw session cannot be reloaded safely."""


@dataclass(frozen=True, slots=True)
class RawSession:
    session_dir: Path
    metadata: SessionMetadata
    packets: tuple[CsiPacket, ...]


def load_raw_session(session_dir: Path | str) -> RawSession:
    session_dir = Path(session_dir)
    metadata_path = session_dir / "metadata.json"
    packets_path = session_dir / "packets.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Raw session metadata file not found: {metadata_path}")
    if not packets_path.exists():
        raise FileNotFoundError(f"Raw session packets file not found: {packets_path}")

    metadata = _load_metadata(metadata_path)
    packets = _load_packets(packets_path, expected_session_id=metadata.session_id)

    return RawSession(
        session_dir=session_dir,
        metadata=metadata,
        packets=tuple(
            sorted(
                packets,
                key=lambda packet: (
                    packet.timestamp,
                    packet.node_id,
                    packet.seq if packet.seq is not None else -1,
                ),
            )
        ),
    )


def load_raw_session_by_id(raw_root: Path | str, session_id: str) -> RawSession:
    return load_raw_session(Path(raw_root) / session_id)


def _load_metadata(metadata_path: Path) -> SessionMetadata:
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RawSessionLoadError(
            f"Raw session metadata is not valid JSON: {metadata_path}"
        ) from exc
    return SessionMetadata.from_dict(payload)


def _load_packets(
    packets_path: Path,
    *,
    expected_session_id: str,
) -> list[CsiPacket]:
    packets: list[CsiPacket] = []

    with packets_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RawSessionLoadError(
                    f"Raw session packet line {line_number} is not valid JSON: {packets_path}"
                ) from exc
            try:
                packet = parse_packet_dict(payload)
            except PacketParseError as exc:
                raise RawSessionLoadError(
                    f"Raw session packet line {line_number} is invalid: {exc}"
                ) from exc

            if packet.session_id != expected_session_id:
                raise RawSessionLoadError(
                    "Raw session packet session_id does not match metadata session_id: "
                    f"{packet.session_id} != {expected_session_id}"
                )
            packets.append(packet)

    return packets

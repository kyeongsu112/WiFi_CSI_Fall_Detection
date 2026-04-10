from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Union

from collector.interfaces import PacketSource
from collector.packet_parser import PacketParseError, parse_packet_dict
from shared.models import CsiPacket


class ReplaySourceError(RuntimeError):
    """Raised when a replay file cannot be read or normalized."""


class JsonlReplaySource(PacketSource):
    """Yields normalized packets from a JSONL replay file."""

    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)

    def __iter__(self) -> Iterator[CsiPacket]:
        if not self.path.exists():
            raise ReplaySourceError(f"Replay input does not exist: {self.path}")

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ReplaySourceError(
                        f"Invalid JSON on line {line_number} of {self.path}"
                    ) from exc

                try:
                    yield parse_packet_dict(payload)
                except PacketParseError as exc:
                    raise ReplaySourceError(
                        f"Invalid packet on line {line_number} of {self.path}: {exc}"
                    ) from exc

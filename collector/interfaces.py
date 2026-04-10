from __future__ import annotations

from typing import Iterator, Protocol

from shared.models import CsiPacket


class PacketSource(Protocol):
    """Minimal iterator contract for normalized packet producers."""

    def __iter__(self) -> Iterator[CsiPacket]:
        ...

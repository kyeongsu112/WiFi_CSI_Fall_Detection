from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Union

from shared.models import CsiPacket, SessionMetadata


@dataclass(frozen=True, slots=True)
class SessionWriteResult:
    session_dir: Path
    packet_count: int
    node_ids: Tuple[str, ...]


class SessionStore:
    """Persists raw session metadata and normalized packets to disk."""

    def __init__(self, root_dir: Union[Path, str]):
        self.root_dir = Path(root_dir)

    def write_session(
        self, metadata: SessionMetadata, packets: Iterable[CsiPacket]
    ) -> SessionWriteResult:
        session_dir = self.root_dir / metadata.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = session_dir / "metadata.json"
        packets_path = session_dir / "packets.jsonl"

        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata.to_dict(), handle, indent=2, ensure_ascii=False)

        node_ids: List[str] = []
        node_id_set: Set[str] = set()
        packet_count = 0

        with packets_path.open("w", encoding="utf-8") as handle:
            for packet in packets:
                json.dump(packet.to_dict(), handle, ensure_ascii=False)
                handle.write("\n")
                packet_count += 1
                if packet.node_id not in node_id_set:
                    node_id_set.add(packet.node_id)
                    node_ids.append(packet.node_id)

        return SessionWriteResult(
            session_dir=session_dir,
            packet_count=packet_count,
            node_ids=tuple(node_ids),
        )

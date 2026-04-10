from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple, Union


NumericSeries = Tuple[float, ...]
FeatureValue = Union[float, NumericSeries]


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    session_id: str
    participant_id: str
    activity_label: str
    recorded_at: str
    room_id: str
    layout_version: str
    node_setup_version: str
    fall_direction: Optional[str] = None
    notes: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionMetadata":
        known_keys = {
            "session_id",
            "participant_id",
            "activity_label",
            "recorded_at",
            "room_id",
            "layout_version",
            "node_setup_version",
            "fall_direction",
            "notes",
        }
        missing = sorted(
            key
            for key in (
                "session_id",
                "participant_id",
                "activity_label",
                "recorded_at",
                "room_id",
                "layout_version",
                "node_setup_version",
            )
            if not payload.get(key)
        )
        if missing:
            raise ValueError("Missing required session metadata fields: " + ", ".join(missing))

        extra = {
            str(key): value
            for key, value in payload.items()
            if str(key) not in known_keys
        }
        return cls(
            session_id=str(payload["session_id"]),
            participant_id=str(payload["participant_id"]),
            activity_label=str(payload["activity_label"]),
            recorded_at=str(payload["recorded_at"]),
            room_id=str(payload["room_id"]),
            layout_version=str(payload["layout_version"]),
            node_setup_version=str(payload["node_setup_version"]),
            fall_direction=_optional_string(payload.get("fall_direction")),
            notes=_optional_string(payload.get("notes")),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "activity_label": self.activity_label,
            "recorded_at": self.recorded_at,
            "room_id": self.room_id,
            "layout_version": self.layout_version,
            "node_setup_version": self.node_setup_version,
            "fall_direction": self.fall_direction,
            "notes": self.notes,
        }
        payload.update(_thaw_mapping(self.extra))
        return payload


@dataclass(frozen=True, slots=True)
class CsiPacket:
    timestamp: float
    node_id: str
    session_id: str
    seq: Optional[int] = None
    rssi: Optional[float] = None
    channel: Optional[int] = None
    amplitude: Optional[NumericSeries] = None
    phase: Optional[NumericSeries] = None
    raw_payload: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.raw_payload is None:
            return
        object.__setattr__(self, "raw_payload", _freeze_mapping(self.raw_payload))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "rssi": self.rssi,
            "channel": self.channel,
            "amplitude": list(self.amplitude) if self.amplitude is not None else None,
            "phase": list(self.phase) if self.phase is not None else None,
            "raw_payload": _thaw_mapping(self.raw_payload) if self.raw_payload is not None else None,
        }


@dataclass(frozen=True, slots=True)
class CsiWindow:
    window_id: str
    start_ts: float
    end_ts: float
    node_ids: Tuple[str, ...]
    features: Mapping[str, FeatureValue]
    label: Optional[str] = None
    feature_version: Optional[str] = None
    source_session_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _freeze_mapping(self.features))


@dataclass(frozen=True, slots=True)
class Prediction:
    timestamp: float
    fall_prob: float
    motion_score: float
    state: str
    reasons: Tuple[str, ...] = ()
    confidence: Optional[float] = None


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(mapping, MappingABC):
        raise TypeError("Expected a mapping for immutable payload storage.")
    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in mapping.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): _thaw_value(value) for key, value in mapping.items()}


def _thaw_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return _thaw_mapping(value)
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value

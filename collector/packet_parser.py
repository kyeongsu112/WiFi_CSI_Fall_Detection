from __future__ import annotations

from numbers import Real
from typing import Any, Dict, Mapping, Optional

from shared.models import CsiPacket, NumericSeries


class PacketParseError(ValueError):
    """Raised when a packet payload cannot be normalized safely."""


def parse_packet_dict(payload: Mapping[str, Any]) -> CsiPacket:
    if not isinstance(payload, Mapping):
        raise PacketParseError("Packet payload must be a mapping.")

    timestamp = _required_float(payload, "timestamp")
    node_id = _required_string(payload, "node_id")
    session_id = _required_string(payload, "session_id")

    return CsiPacket(
        timestamp=timestamp,
        node_id=node_id,
        session_id=session_id,
        seq=_optional_int(payload.get("seq"), field_name="seq"),
        rssi=_optional_float(payload.get("rssi")),
        channel=_optional_int(payload.get("channel"), field_name="channel"),
        amplitude=_optional_series(payload.get("amplitude"), field_name="amplitude"),
        phase=_optional_series(payload.get("phase"), field_name="phase"),
        raw_payload=_normalize_raw_payload(payload),
    )


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None:
        raise PacketParseError(f"Missing required field: {field_name}")
    value = str(value).strip()
    if not value:
        raise PacketParseError(f"Field must not be empty: {field_name}")
    return value


def _required_float(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    if value is None:
        raise PacketParseError(f"Missing required field: {field_name}")
    return _coerce_float(value, field_name)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _coerce_float(value, "float")


def _optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PacketParseError(f"Field '{field_name}' must be an integer.")

    numeric_value = float(value)
    if not numeric_value.is_integer():
        raise PacketParseError(
            f"Field '{field_name}' must be an integer without a fractional component."
        )
    return int(numeric_value)


def _optional_series(value: Any, field_name: str) -> Optional[NumericSeries]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise PacketParseError(f"Field '{field_name}' must be a numeric sequence.")

    normalized = []
    for item in value:
        normalized.append(_coerce_float(item, field_name))
    return tuple(normalized)


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise PacketParseError(f"Field '{field_name}' must be numeric.")
    if isinstance(value, Real):
        return float(value)
    raise PacketParseError(f"Field '{field_name}' must be numeric.")


def _normalize_raw_payload(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    known_keys = {
        "timestamp",
        "node_id",
        "session_id",
        "seq",
        "rssi",
        "channel",
        "amplitude",
        "phase",
        "csi_raw",
        "raw_payload",
    }
    raw_payload: Dict[str, Any] = {}

    raw_value = payload.get("raw_payload")
    if raw_value is not None:
        if isinstance(raw_value, Mapping):
            raw_payload.update(dict(raw_value))
        else:
            raw_payload["raw_payload"] = raw_value

    if "csi_raw" in payload:
        raw_payload["csi_raw"] = payload["csi_raw"]

    extra_fields = {
        str(key): value for key, value in payload.items() if str(key) not in known_keys
    }
    raw_payload.update(extra_fields)
    return raw_payload or None

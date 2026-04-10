from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from shared.models import CsiPacket


class PacketDecodeError(RuntimeError):
    """Raised when a normalized packet cannot be decoded to numeric arrays."""


@dataclass(frozen=True, slots=True)
class DecodedCsiPacket:
    timestamp: float
    node_id: str
    session_id: str
    seq: int | None
    rssi: float | None
    channel: int | None
    real: np.ndarray
    imag: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    source_format: str

    def flattened_real(self) -> np.ndarray:
        return self.real.reshape(-1)

    def flattened_imag(self) -> np.ndarray:
        return self.imag.reshape(-1)

    def flattened_amplitude(self) -> np.ndarray:
        return self.amplitude.reshape(-1)

    def flattened_phase(self) -> np.ndarray:
        return self.phase.reshape(-1)


def decode_packets(
    packets: list[CsiPacket] | tuple[CsiPacket, ...],
    *,
    phase_unwrap_enabled: bool,
) -> tuple[DecodedCsiPacket, ...]:
    return tuple(
        decode_packet(packet, phase_unwrap_enabled=phase_unwrap_enabled)
        for packet in packets
    )


def decode_packet(
    packet: CsiPacket,
    *,
    phase_unwrap_enabled: bool,
) -> DecodedCsiPacket:
    if packet.amplitude is not None or packet.phase is not None:
        if packet.amplitude is None or packet.phase is None:
            raise PacketDecodeError(
                "Packets with derived CSI values must contain both amplitude and phase."
            )

        amplitude = _freeze_array(np.asarray(packet.amplitude, dtype=np.float64).reshape(1, -1))
        phase = np.asarray(packet.phase, dtype=np.float64).reshape(1, -1)
        if phase_unwrap_enabled:
            phase = np.unwrap(phase, axis=-1)
        phase = _freeze_array(phase)
        real = _freeze_array(amplitude * np.cos(phase))
        imag = _freeze_array(amplitude * np.sin(phase))

        return DecodedCsiPacket(
            timestamp=packet.timestamp,
            node_id=packet.node_id,
            session_id=packet.session_id,
            seq=packet.seq,
            rssi=packet.rssi,
            channel=packet.channel,
            real=real,
            imag=imag,
            amplitude=amplitude,
            phase=phase,
            source_format=_derive_packet_source_format(packet),
        )

    csi_raw = _extract_csi_raw(packet)
    raw_format = str(csi_raw.get("format", "")).strip()
    if raw_format != "esp32_adr018":
        raise PacketDecodeError(
            f"Unsupported raw CSI format for preprocessing: {raw_format or 'missing'}"
        )

    antenna_count = _require_positive_int(csi_raw, "antenna_count")
    subcarrier_count = _require_positive_int(csi_raw, "subcarrier_count")
    iq_bytes_hex = csi_raw.get("iq_bytes_hex")
    if not isinstance(iq_bytes_hex, str) or not iq_bytes_hex.strip():
        raise PacketDecodeError("ESP32 ADR-018 payload is missing iq_bytes_hex.")

    try:
        iq_bytes = bytes.fromhex(iq_bytes_hex)
    except ValueError as exc:
        raise PacketDecodeError("ESP32 ADR-018 iq_bytes_hex is not valid hexadecimal.") from exc

    expected_iq_bytes = antenna_count * subcarrier_count * 2
    if len(iq_bytes) != expected_iq_bytes:
        raise PacketDecodeError(
            "ESP32 ADR-018 iq_bytes_hex length mismatch: "
            f"expected {expected_iq_bytes} bytes, got {len(iq_bytes)}."
        )

    iq_values = np.frombuffer(iq_bytes, dtype=np.int8).astype(np.float64)
    real = iq_values[0::2].reshape(antenna_count, subcarrier_count)
    imag = iq_values[1::2].reshape(antenna_count, subcarrier_count)
    amplitude = np.sqrt(real**2 + imag**2)
    phase = np.arctan2(imag, real)
    if phase_unwrap_enabled:
        phase = np.unwrap(phase, axis=-1)

    return DecodedCsiPacket(
        timestamp=packet.timestamp,
        node_id=packet.node_id,
        session_id=packet.session_id,
        seq=packet.seq,
        rssi=packet.rssi,
        channel=packet.channel,
        real=_freeze_array(real),
        imag=_freeze_array(imag),
        amplitude=_freeze_array(amplitude),
        phase=_freeze_array(phase),
        source_format="esp32_adr018",
    )


def _extract_csi_raw(packet: CsiPacket) -> Mapping[str, Any]:
    if packet.raw_payload is None:
        raise PacketDecodeError("Packet does not contain raw_payload for preprocessing.")
    csi_raw = packet.raw_payload.get("csi_raw")
    if not isinstance(csi_raw, Mapping):
        raise PacketDecodeError("Packet raw_payload does not contain a csi_raw mapping.")
    return csi_raw


def _require_positive_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or value < 1:
        raise PacketDecodeError(f"ESP32 ADR-018 field '{field_name}' must be a positive integer.")
    return value


def _derive_packet_source_format(packet: CsiPacket) -> str:
    if packet.raw_payload is None:
        return "derived_amplitude_phase"
    csi_raw = packet.raw_payload.get("csi_raw")
    if isinstance(csi_raw, Mapping):
        raw_format = str(csi_raw.get("format", "")).strip()
        if raw_format:
            return raw_format
    return "derived_amplitude_phase"


def _freeze_array(array: np.ndarray) -> np.ndarray:
    frozen = np.asarray(array, dtype=np.float64)
    frozen.setflags(write=False)
    return frozen

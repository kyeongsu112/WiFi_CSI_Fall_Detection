from __future__ import annotations

import json
import socket
import struct
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional

from collector.interfaces import PacketSource
from collector.packet_parser import PacketParseError, parse_packet_dict
from shared.models import CsiPacket


SocketFactory = Callable[[int, int], socket.socket]
ErrorReporter = Callable[[str], None]
TimestampFunction = Callable[[], float]

ADR018_CSI_MAGIC = 0xC5110001
ADR018_HEADER_SIZE = 20
ADR018_HEADER_STRUCT = struct.Struct("<IBBHIIbbH")


class ReceiverError(RuntimeError):
    """Raised when a live UDP datagram cannot be received or normalized."""


class _BaseUdpPacketSource(PacketSource, ABC):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        buffer_size: int = 65535,
        timeout_seconds: Optional[float] = None,
        socket_factory: SocketFactory = socket.socket,
        skip_invalid_datagrams: bool = False,
        error_reporter: ErrorReporter | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.timeout_seconds = timeout_seconds
        self.socket_factory = socket_factory
        self.skip_invalid_datagrams = skip_invalid_datagrams
        self.error_reporter = error_reporter
        self._socket: Optional[socket.socket] = None
        self._stop_requested = False

    def __iter__(self) -> Iterator[CsiPacket]:
        udp_socket = self._ensure_socket()
        while not self._stop_requested:
            try:
                datagram, _address = udp_socket.recvfrom(self.buffer_size)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._stop_requested:
                    break
                raise ReceiverError(
                    f"Failed to receive UDP datagram on {self.host}:{self.port}"
                ) from exc

            try:
                yield self._parse_datagram(datagram)
            except ReceiverError as exc:
                if not self.skip_invalid_datagrams:
                    raise
                if self.error_reporter is not None:
                    self.error_reporter(str(exc))
                continue

    def request_stop(self) -> None:
        self._stop_requested = True

    def close(self) -> None:
        self._stop_requested = True
        if self._socket is None:
            return
        self._socket.close()
        self._socket = None

    def _ensure_socket(self) -> socket.socket:
        if self._socket is not None:
            return self._socket

        try:
            udp_socket = self.socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.bind((self.host, self.port))
            if self.timeout_seconds is not None:
                udp_socket.settimeout(self.timeout_seconds)
        except OSError as exc:
            raise ReceiverError(
                f"Failed to bind UDP receiver on {self.host}:{self.port}"
            ) from exc

        self._socket = udp_socket
        return udp_socket

    @abstractmethod
    def _parse_datagram(self, datagram: bytes) -> CsiPacket:
        raise NotImplementedError


class UdpJsonPacketSource(_BaseUdpPacketSource):
    """Receives one normalized JSON object per UDP datagram.

    Phase 2 keeps the JSON live wire format intentionally conservative: one UTF-8
    JSON object per datagram that can already be normalized by
    ``parse_packet_dict``.
    """

    def _parse_datagram(self, datagram: bytes) -> CsiPacket:
        try:
            decoded = datagram.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReceiverError("UDP datagram must be valid UTF-8 JSON.") from exc

        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ReceiverError("UDP datagram must contain exactly one JSON object.") from exc

        try:
            return parse_packet_dict(payload)
        except PacketParseError as exc:
            raise ReceiverError(f"UDP datagram contained an invalid packet: {exc}") from exc


class UdpEsp32BinaryPacketSource(_BaseUdpPacketSource):
    """Receives ESP32 ADR-018 CSI frames and normalizes them to ``CsiPacket``.

    Supported binary format in this baseline is intentionally narrow:
    - magic ``0xC5110001`` only
    - 20-byte ADR-018 header
    - exact I/Q payload length implied by ``antenna_count`` and ``subcarrier_count``

    Other packet types such as vitals/WASM output or unknown binary layouts are
    rejected loudly instead of being guessed.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        session_id: str,
        timestamp_fn: TimestampFunction = time.time,
        buffer_size: int = 65535,
        timeout_seconds: Optional[float] = None,
        socket_factory: SocketFactory = socket.socket,
        skip_invalid_datagrams: bool = False,
        error_reporter: ErrorReporter | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("Binary live packet source requires a non-empty session_id.")

        super().__init__(
            host,
            port,
            buffer_size=buffer_size,
            timeout_seconds=timeout_seconds,
            socket_factory=socket_factory,
            skip_invalid_datagrams=skip_invalid_datagrams,
            error_reporter=error_reporter,
        )
        self.session_id = session_id
        self.timestamp_fn = timestamp_fn

    def _parse_datagram(self, datagram: bytes) -> CsiPacket:
        return parse_esp32_adr018_datagram(
            datagram,
            session_id=self.session_id,
            received_at=self.timestamp_fn(),
        )


def parse_esp32_adr018_datagram(
    datagram: bytes,
    *,
    session_id: str,
    received_at: float,
) -> CsiPacket:
    if len(datagram) < ADR018_HEADER_SIZE:
        raise ReceiverError(
            f"ESP32 ADR-018 datagram is truncated: expected at least "
            f"{ADR018_HEADER_SIZE} bytes, got {len(datagram)}."
        )

    (
        magic,
        node_id,
        antenna_count,
        subcarrier_count,
        frequency_mhz,
        sequence,
        rssi,
        noise_floor,
        _reserved,
    ) = ADR018_HEADER_STRUCT.unpack_from(datagram)

    if magic != ADR018_CSI_MAGIC:
        raise ReceiverError(
            f"Unsupported ESP32 binary packet magic: 0x{magic:08X}. "
            f"Only ADR-018 CSI frames (0x{ADR018_CSI_MAGIC:08X}) are supported."
        )

    if antenna_count < 1:
        raise ReceiverError("ESP32 ADR-018 datagram must report at least one antenna.")
    if subcarrier_count < 1:
        raise ReceiverError("ESP32 ADR-018 datagram must report at least one subcarrier.")

    expected_iq_bytes = antenna_count * subcarrier_count * 2
    actual_iq_bytes = len(datagram) - ADR018_HEADER_SIZE
    if actual_iq_bytes != expected_iq_bytes:
        raise ReceiverError(
            "ESP32 ADR-018 datagram payload length mismatch: "
            f"expected {expected_iq_bytes} I/Q bytes, got {actual_iq_bytes}."
        )

    payload = {
        "timestamp": received_at,
        "node_id": str(node_id),
        "session_id": session_id,
        "seq": sequence,
        "rssi": rssi,
        "channel": _frequency_to_channel(frequency_mhz),
        "csi_raw": {
            "format": "esp32_adr018",
            "magic": f"0x{magic:08X}",
            "antenna_count": antenna_count,
            "subcarrier_count": subcarrier_count,
            "frequency_mhz": frequency_mhz,
            "noise_floor": noise_floor,
            "iq_bytes_hex": datagram[ADR018_HEADER_SIZE:].hex(),
        },
    }

    try:
        return parse_packet_dict(payload)
    except PacketParseError as exc:
        raise ReceiverError(f"ESP32 ADR-018 datagram normalization failed: {exc}") from exc


def _frequency_to_channel(frequency_mhz: int) -> int | None:
    if frequency_mhz == 2484:
        return 14
    if 2412 <= frequency_mhz <= 2472 and (frequency_mhz - 2407) % 5 == 0:
        return (frequency_mhz - 2407) // 5
    if 5000 <= frequency_mhz <= 5900 and frequency_mhz % 5 == 0:
        return (frequency_mhz - 5000) // 5
    return None

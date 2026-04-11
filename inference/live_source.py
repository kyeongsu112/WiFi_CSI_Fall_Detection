"""Unified CSI source abstraction for the WiFall inference pipeline.

Step 4 introduces a common interface so the dashboard can run with any of:

    replay    — windows from the WiFall manifest + zip (Step 3 behaviour, unchanged)
    mock_live — synthetic random windows for end-to-end pipeline testing without hardware
    esp32     — JSON v1 UDP live source for ESP32/local senders

Window contract (matches FallDetector in training/model.py):
    CsiWindow.data  shape (52, 100)  float32
                    axis-0 = n_subcarriers, axis-1 = window_size

Model contract (unchanged from Step 2 / Step 3):
    Model returns raw logits.
    Sigmoid is applied here in InferencePipeline, NOT inside the model.

Usage
-----
    from inference.live_source import build_source, InferencePipeline

    source   = build_source(source_mode="replay", manifest_path=..., zip_path=...,
                            config_path=..., step_delay=None)
    pipeline = InferencePipeline.from_config("configs/inference.yaml")

    stop = threading.Event()
    for step_idx, window in enumerate(source.windows(stop)):
        event = pipeline.step(window, step_idx)
        ...   # ReplayEvent with .alert_state, .probability, etc.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import zipfile

import numpy as np
import torch

from inference.replay import (
    ReplayConfig,
    ReplayEvent,
    load_replay_config,
)
from inference.confirmation import (
    ConfirmationConfig,
    ConfirmationEngine,
    compute_motion_score,
)
from training.model import FallDetector, load_model


# ---------------------------------------------------------------------------
# Per-process model cache
# ---------------------------------------------------------------------------
#
# Loading a FallDetector checkpoint from disk takes ~100 ms on CPU and
# allocates a copy of all weights.  On a typical server, the model is the
# same for every SSE connection — loading it once and sharing it across
# InferencePipeline instances eliminates that per-connection overhead.
#
# Safety: FallDetector.forward() with torch.no_grad() in eval mode is
# thread-safe on CPU.  BatchNorm in eval mode reads (never writes)
# running_mean / running_var, so concurrent inference from different
# worker threads is safe without any additional locking.

_MODEL_CACHE: dict[str, FallDetector] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_cached_model(model_path: str) -> FallDetector:
    """Return a cached FallDetector, loading from disk only on first call per path.

    Thread-safe: the lock is held only during the initial load; subsequent
    callers read from the dict without blocking each other.
    """
    with _MODEL_CACHE_LOCK:
        if model_path not in _MODEL_CACHE:
            model, _ = load_model(model_path)
            _MODEL_CACHE[model_path] = model
        return _MODEL_CACHE[model_path]


# ---------------------------------------------------------------------------
# Window wrapper
# ---------------------------------------------------------------------------

@dataclass
class CsiWindow:
    """One normalised CSI window ready for model inference.

    Attributes:
        data:         float32 ndarray of shape (52, 100).
        source_file:  Human-readable origin label ("" for generated windows).
        window_index: Position within the originating session or stream.
        duration_seconds: Logical duration represented by the window.
    """
    data: np.ndarray   # (n_subcarriers=52, window_size=100), float32
    source_file: str = ""
    window_index: int = 0
    duration_seconds: float = 1.0


# ---------------------------------------------------------------------------
# Abstract source
# ---------------------------------------------------------------------------

class CsiSource(ABC):
    """Abstract CSI window source.

    All sources honour stop_event so the SSE server can cancel the worker
    thread cleanly when a browser client disconnects.
    """

    @abstractmethod
    def windows(
        self, stop_event: threading.Event | None = None
    ) -> Generator[CsiWindow, None, None]:
        """Yield normalised CsiWindow objects until exhausted or stop_event is set."""

    def get_runtime_status(self) -> dict[str, object] | None:
        return None


# ---------------------------------------------------------------------------
# Replay source
# ---------------------------------------------------------------------------

_WINDOW_SIZE = 100      # canonical; matches FallDetector checkpoint
_N_SUBCARRIERS = 52     # canonical; matches FallDetector checkpoint


class ReplaySource(CsiSource):
    """CSI source that reads windows from the WiFall manifest (replay mode).

    Loads windows from the WiFall zip in manifest order.  Padding/trimming
    to the canonical (52, 100) shape is applied here; the model and alert
    state machine are handled by InferencePipeline.

    The underlying inference/replay.py (replay_manifest function) is left
    completely unchanged — its Step 3 interface and tests remain valid.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        zip_path: str | Path = "data/WiFall.zip",
        config_path: str | Path = "configs/inference.yaml",
        step_delay: float | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.zip_path = zip_path
        self.config_path = config_path
        self.step_delay = step_delay
        self._windows_emitted = 0

    def windows(
        self, stop_event: threading.Event | None = None
    ) -> Generator[CsiWindow, None, None]:
        """Yield one CsiWindow per manifest row, in manifest order.

        I/O efficiency: the ZIP file is opened once for the lifetime of this
        generator, and each session CSV is parsed and cached after its first
        access.  Subsequent windows from the same session file are served by
        slicing the cached array — no repeated ZIP or CSV I/O.
        """
        import pandas as pd
        from datasets.loader import load_csi_from_fileobj

        cfg = load_replay_config(self.config_path)
        delay = self.step_delay if self.step_delay is not None else cfg.step_delay_seconds

        manifest = pd.read_csv(self.manifest_path)

        # Per-session CSI cache: source_file → full (n_rows, 52) array.
        _csi_cache: dict[str, np.ndarray] = {}

        with zipfile.ZipFile(self.zip_path) as zf:
            for row in manifest.itertuples(index=False):
                if stop_event is not None and stop_event.is_set():
                    return

                source_file = str(row.source_file)
                if source_file not in _csi_cache:
                    with zf.open(source_file) as f:
                        _csi_cache[source_file] = load_csi_from_fileobj(f)

                arr = _csi_cache[source_file]
                raw = arr[int(row.start_row):int(row.end_row)]  # (T, 52)

                t = raw.shape[0]
                if t < _WINDOW_SIZE:
                    raw = np.pad(raw, ((0, _WINDOW_SIZE - t), (0, 0)))
                else:
                    raw = raw[:_WINDOW_SIZE]

                # (window_size, n_subcarriers) → (n_subcarriers, window_size)
                self._windows_emitted = int(row.window_index) + 1
                yield CsiWindow(
                    data=raw.T.astype(np.float32),
                    source_file=source_file,
                    window_index=int(row.window_index),
                )

                if delay > 0:
                    time.sleep(delay)

                if stop_event is not None and stop_event.is_set():
                    return

    def get_runtime_status(self) -> dict[str, object]:
        return {
            "mode": "replay",
            "transport_state": "offline",
            "active_sid": None,
            "packets_received": 0,
            "packets_dropped": 0,
            "windows_emitted": self._windows_emitted,
        }


# ---------------------------------------------------------------------------
# Mock live source
# ---------------------------------------------------------------------------

class MockLiveSource(CsiSource):
    """CSI source that generates synthetic random windows for pipeline testing.

    Produces white-noise arrays scaled to a realistic CSI amplitude range.
    The model will score these as non-fall (low probability) since they lack
    the structured amplitude patterns present in WiFall data.

    Purpose: exercise the full inference pipeline (source → model → state
    machine → SSE) without WiFall.zip or ESP32 hardware.  The source runs
    indefinitely until stop_event is set.
    """

    def __init__(
        self,
        step_delay_seconds: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.step_delay_seconds = step_delay_seconds
        self._rng = np.random.default_rng(seed)
        self._windows_emitted = 0

    def windows(
        self, stop_event: threading.Event | None = None
    ) -> Generator[CsiWindow, None, None]:
        """Yield random (52, 100) windows indefinitely at step_delay_seconds rate."""
        step = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                return

            # Gaussian noise scaled to typical WiFall CSI amplitude range
            data = self._rng.normal(
                loc=50.0, scale=15.0, size=(_N_SUBCARRIERS, _WINDOW_SIZE)
            ).astype(np.float32)
            data = np.clip(data, 0.0, None)

            self._windows_emitted = step + 1
            yield CsiWindow(
                data=data,
                source_file="mock://live",
                window_index=step,
            )

            step += 1

            if self.step_delay_seconds > 0:
                time.sleep(self.step_delay_seconds)

            if stop_event is not None and stop_event.is_set():
                return

    def get_runtime_status(self) -> dict[str, object]:
        return {
            "mode": "mock_live",
            "transport_state": "synthetic",
            "active_sid": "mock",
            "packets_received": 0,
            "packets_dropped": 0,
            "windows_emitted": self._windows_emitted,
        }


# ---------------------------------------------------------------------------
# ESP32 UDP packet parsing
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


def _parse_esp32_packet(
    raw: bytes,
    n_subcarriers: int = _N_SUBCARRIERS,
) -> tuple[np.ndarray, str] | None:
    """Parse a v1 ESP32 UDP packet.

    Expected format — single-line JSON, all fields except ``amp`` are optional::

        {
          "ts":  <float>,              # Unix epoch seconds
          "fi":  <int>,                # frame index (monotonically increasing)
          "amp": [<float> × 52],       # CSI subcarrier amplitudes  ← REQUIRED
          "sid": "<str>"               # device / session id  (default "esp32")
        }

    Args:
        raw:           Raw UDP payload bytes.
        n_subcarriers: Expected length of the ``amp`` array (default 52).

    Returns:
        ``(amplitudes, sid)`` on success, where ``amplitudes`` is a float32
        ndarray of shape ``(n_subcarriers,)`` and ``sid`` is a string.
        Returns ``None`` when the packet is malformed or contains non-finite
        amplitude values (NaN, +Infinity, -Infinity).  Non-finite values must
        never reach model inference.
    """
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None

    if not isinstance(obj, dict):
        return None

    amp = obj.get("amp")
    if not isinstance(amp, list) or len(amp) != n_subcarriers:
        return None

    try:
        arr = np.array(amp, dtype=np.float32)
    except (ValueError, TypeError):
        return None

    # Reject NaN / ±Infinity.  These arise from firmware float overflows,
    # non-standard JSON encoders that emit the bare tokens NaN / Infinity,
    # or IEEE 754 edge cases in the CSI measurement path.
    if not np.all(np.isfinite(arr)):
        return None

    sid = str(obj.get("sid", "esp32"))
    return arr, sid


# ---------------------------------------------------------------------------
# ESP32 source
# ---------------------------------------------------------------------------

class Esp32Source(CsiSource):
    """Live CSI source that receives UDP packets from an ESP32-S3 device.

    The ESP32 and the laptop must be on the same Wi-Fi network.
    SSID and password are stored in the ``.env`` file (untracked).
    See ``.env.example`` for the expected format.

    Packet contract — v1 JSON over UDP
    ------------------------------------
    Each UDP datagram is a single-line JSON object::

        {"ts": 1712700000.123, "fi": 42, "amp": [f0, f1, …, f51], "sid": "dev-01"}

    Fields:
        ts  (float)      Unix epoch seconds.                          Optional.
        fi  (int)        Frame index, monotonically increasing.       Optional.
        amp (list[float])  52 CSI subcarrier amplitudes.            **Required.**
        sid (str)        Device / session identifier.                 Optional (default "esp32").

    Malformed packets (wrong JSON, missing ``amp``, wrong length, or any
    non-finite amplitude value) are logged at WARNING level and dropped without
    interrupting the stream.

    Sid lock policy (v1 — single-sender mode)
    -----------------------------------------
    The source locks onto the ``sid`` field of the **first valid packet** it
    receives.  All subsequent packets whose ``sid`` differs from the locked
    value are dropped with a WARNING log entry.  This prevents frames from
    two concurrent senders from being silently interleaved into the same model
    window.

    Reset: create a new :class:`Esp32Source` instance (or restart the server)
    to unlock and accept a different sender.

    Buffering
    ---------
    Frames are accumulated in a FIFO buffer.  Once ``window_size`` (100) frames
    are ready, one :class:`CsiWindow` of shape ``(52, 100)`` is emitted and the
    consumed frames are removed.  No overlap in v1 (stride == window_size).

    Window shape
    ------------
    ``CsiWindow.data`` is ``float32 ndarray`` of shape ``(n_subcarriers=52, window_size=100)``,
    matching the contract expected by :class:`InferencePipeline`.

    Logging
    -------
    All log output goes to the ``inference.live_source`` logger.  Set the log
    level to ``DEBUG`` to see per-packet counts; ``INFO`` shows bind/window events.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        window_size: int = _WINDOW_SIZE,
        n_subcarriers: int = _N_SUBCARRIERS,
        socket_timeout: float = 0.5,
        log_every_n_packets: int = 100,
    ) -> None:
        """
        Args:
            host:                 IP address to bind the UDP socket on.
                                  ``"0.0.0.0"`` accepts from any interface.
            port:                 UDP port to listen on (default 5005).
            window_size:          Frames per model window (default 100).
            n_subcarriers:        Expected amplitude array length (default 52).
            socket_timeout:       ``recvfrom`` timeout in seconds; controls how
                                  quickly ``stop_event`` is checked (default 0.5 s).
            log_every_n_packets:  Log a DEBUG packet-count summary every N packets
                                  (default 100).
        """
        self.host = host
        self.port = port
        self.window_size = window_size
        self.n_subcarriers = n_subcarriers
        self.socket_timeout = socket_timeout
        self.log_every_n_packets = log_every_n_packets
        self._status_lock = threading.Lock()
        self._transport_state = "idle"
        self._active_sid: str | None = None
        self._packets_received = 0
        self._packets_dropped = 0
        self._windows_emitted = 0

    def windows(
        self, stop_event: threading.Event | None = None
    ) -> Generator[CsiWindow, None, None]:
        """Bind UDP socket, receive frames, buffer and yield CsiWindows.

        The socket is bound lazily when this generator is first iterated, so
        :func:`build_source` and :class:`Esp32Source` construction are free
        from network side-effects.

        The generator runs indefinitely until ``stop_event`` is set or the
        calling thread is cancelled.  The socket is always closed in the
        ``finally`` block.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(self.socket_timeout)
        sock.bind((self.host, self.port))
        _log.info("Esp32Source: UDP socket bound on %s:%d", self.host, self.port)
        self._update_status(transport_state="listening")

        frame_buffer: list[np.ndarray] = []
        window_index = 0
        packets_received = 0
        packets_dropped = 0
        # Single-sender sid lock: set on first valid packet, immutable until reset.
        active_sid: str | None = None

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return

                try:
                    raw, addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if stop_event is not None and stop_event.is_set():
                        return
                    packets_dropped += 1
                    self._update_status(
                        transport_state="warning",
                        packets_dropped=packets_dropped,
                    )
                    _log.warning("Esp32Source: UDP receive error: %s", exc)
                    time.sleep(min(self.socket_timeout, 0.1))
                    continue

                packets_received += 1
                self._update_status(
                    transport_state="streaming" if active_sid is not None else "listening",
                    packets_received=packets_received,
                )
                if packets_received % self.log_every_n_packets == 0:
                    _log.debug(
                        "Esp32Source: %d packets received, %d dropped",
                        packets_received,
                        packets_dropped,
                    )

                parsed = _parse_esp32_packet(raw, self.n_subcarriers)
                if parsed is None:
                    packets_dropped += 1
                    self._update_status(
                        transport_state="warning",
                        packets_received=packets_received,
                        packets_dropped=packets_dropped,
                    )
                    _log.warning(
                        "Esp32Source: malformed packet #%d from %s — dropped",
                        packets_received,
                        addr,
                    )
                    continue

                amp, sid = parsed

                # Sid lock: lock onto the first observed sender and reject others.
                if active_sid is None:
                    active_sid = sid
                    self._update_status(
                        transport_state="streaming",
                        active_sid=active_sid,
                        packets_received=packets_received,
                        packets_dropped=packets_dropped,
                    )
                    _log.info("Esp32Source: locked onto sender sid=%r", active_sid)
                elif sid != active_sid:
                    packets_dropped += 1
                    self._update_status(
                        transport_state="warning",
                        active_sid=active_sid,
                        packets_received=packets_received,
                        packets_dropped=packets_dropped,
                    )
                    _log.warning(
                        "Esp32Source: packet from sid=%r dropped "
                        "(locked onto sid=%r; mixed-sender streams not supported in v1)",
                        sid,
                        active_sid,
                    )
                    continue

                frame_buffer.append(amp)

                if len(frame_buffer) >= self.window_size:
                    frames = np.stack(frame_buffer[: self.window_size], axis=0)  # (100, 52)
                    csi_window = CsiWindow(
                        data=frames.T.astype(np.float32),  # (52, 100)
                        source_file=f"esp32://{sid}",
                        window_index=window_index,
                    )
                    _log.info(
                        "Esp32Source: window #%d emitted (source=%s)",
                        window_index,
                        csi_window.source_file,
                    )
                    self._update_status(
                        transport_state="streaming",
                        active_sid=active_sid,
                        packets_received=packets_received,
                        packets_dropped=packets_dropped,
                        windows_emitted=window_index + 1,
                    )
                    yield csi_window
                    window_index += 1
                    frame_buffer = frame_buffer[self.window_size :]  # v1: no overlap
        finally:
            sock.close()
            self._update_status(
                transport_state="closed",
                active_sid=active_sid,
                packets_received=packets_received,
                packets_dropped=packets_dropped,
                windows_emitted=window_index,
            )
            _log.info(
                "Esp32Source: socket closed — received=%d dropped=%d windows=%d",
                packets_received,
                packets_dropped,
                window_index,
            )

    def get_runtime_status(self) -> dict[str, object]:
        with self._status_lock:
            return {
                "mode": "esp32",
                "transport_state": self._transport_state,
                "active_sid": self._active_sid,
                "packets_received": self._packets_received,
                "packets_dropped": self._packets_dropped,
                "windows_emitted": self._windows_emitted,
            }

    def _update_status(
        self,
        *,
        transport_state: str | None = None,
        active_sid: str | None = None,
        packets_received: int | None = None,
        packets_dropped: int | None = None,
        windows_emitted: int | None = None,
    ) -> None:
        with self._status_lock:
            if transport_state is not None:
                self._transport_state = transport_state
            if active_sid is not None:
                self._active_sid = active_sid
            if packets_received is not None:
                self._packets_received = packets_received
            if packets_dropped is not None:
                self._packets_dropped = packets_dropped
            if windows_emitted is not None:
                self._windows_emitted = windows_emitted


# ---------------------------------------------------------------------------
# Inference pipeline (model + sigmoid + state machine)
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Runs model inference and alert-state tracking on CsiWindow objects.

    Owns the model, threshold, and ConfirmationEngine.  Consumes one
    CsiWindow per call to step() and returns a fully-populated ReplayEvent.

    The model contract is unchanged from Step 2/3:
      - FallDetector returns raw logits.
      - Sigmoid is applied here, not inside the model.

    The FallDetector is shared across pipeline instances via a per-process
    cache (_get_cached_model) — it is loaded from disk only once per model
    path, then reused by every connection.  The ConfirmationEngine is
    per-instance (stateful per replay/live session).
    """

    def __init__(self, cfg: ReplayConfig) -> None:
        self.cfg = cfg
        self._model = _get_cached_model(cfg.model_path)
        self._engine = ConfirmationEngine(
            ConfirmationConfig(
                candidate_threshold=cfg.candidate_threshold,
                inactivity_seconds=cfg.post_fall_inactivity_seconds,
                motion_threshold=cfg.motion_floor_threshold,
                confirm_window_seconds=cfg.confirm_window_seconds,
                cooldown_seconds=cfg.cooldown_seconds,
            )
        )
        self._previous_window: np.ndarray | None = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "InferencePipeline":
        """Construct from an inference.yaml path."""
        return cls(load_replay_config(config_path))

    def step(self, window: CsiWindow, step_idx: int) -> ReplayEvent:
        """Score one window and return a ReplayEvent with current alert state.

        Args:
            window:   A CsiWindow with .data of shape (52, 100).
            step_idx: Monotonically increasing step counter (set by caller).

        Returns:
            ReplayEvent with probability, predicted_label, and alert_state.
        """
        # (52, 100) → (1, 52, 100)
        x = torch.tensor(window.data[np.newaxis], dtype=torch.float32)

        with torch.no_grad():
            logit: float = self._model(x).item()

        prob: float = float(torch.sigmoid(torch.tensor(logit)))
        motion_score = compute_motion_score(window.data, self._previous_window)
        self._previous_window = window.data
        predicted_label = "fall" if prob >= self.cfg.candidate_threshold else "non_fall"
        alert_state = self._engine.step(prob, motion_score, window.duration_seconds)

        return ReplayEvent(
            step=step_idx,
            source_file=window.source_file,
            window_index=window.window_index,
            probability=prob,
            predicted_label=predicted_label,
            alert_state=alert_state,
            motion_score=motion_score,
        )


# ---------------------------------------------------------------------------
# Source factory
# ---------------------------------------------------------------------------

def build_source(
    source_mode: str,
    *,
    manifest_path: str | Path = "artifacts/processed/wifall_manifest.csv",
    zip_path: str | Path = "data/WiFall.zip",
    config_path: str | Path = "configs/inference.yaml",
    step_delay: float | None = None,
) -> CsiSource:
    """Instantiate the correct CsiSource for the given mode.

    Args:
        source_mode:   One of "replay", "mock_live", "esp32".
        manifest_path: Path to wifall_manifest.csv (replay only).
        zip_path:      Path to WiFall.zip (replay only).
        config_path:   Path to inference.yaml (all modes).
        step_delay:    Override inter-window sleep (None = use config value).

    Returns:
        A CsiSource ready for use in the inference pipeline.

    Raises:
        ValueError: Unknown source_mode.
    """
    if source_mode == "replay":
        return ReplaySource(
            manifest_path=manifest_path,
            zip_path=zip_path,
            config_path=config_path,
            step_delay=step_delay,
        )
    if source_mode == "mock_live":
        cfg = load_replay_config(config_path)
        delay = step_delay if step_delay is not None else cfg.step_delay_seconds
        return MockLiveSource(step_delay_seconds=delay)
    if source_mode == "esp32":
        try:
            from shared.config import load_inference_config
            icfg = load_inference_config(Path(config_path))
            host = icfg.esp32_udp_host
            port = icfg.esp32_udp_port
        except (FileNotFoundError, OSError):
            host = "0.0.0.0"
            port = 5005
        return Esp32Source(host=host, port=port)
    raise ValueError(
        f"Unknown source_mode: {source_mode!r}. "
        "Valid choices: 'replay', 'mock_live', 'esp32'."
    )


# ---------------------------------------------------------------------------
# Source mode resolution (precedence helper used by script and server)
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE_MODE = "replay"


def resolve_source_mode(
    cli_source: str | None,
    config_path: str | Path = "configs/inference.yaml",
) -> str:
    """Return the effective source mode, applying the canonical precedence rule.

    Precedence (highest → lowest):
      1. cli_source  — explicit --source CLI argument (not None).
      2. source_mode field in the inference config file.
      3. "replay"    — safe hardcoded default only when the config file is
                       absent (FileNotFoundError / OSError).

    Silent fallback is intentionally narrow.  The following always propagate
    so that misconfiguration is visible immediately:
      - Malformed YAML             → yaml.YAMLError (or a subclass)
      - Invalid source_mode value  → pydantic.ValidationError
      - Any other unexpected error → re-raised as-is

    Args:
        cli_source:   Value of --source from argparse, or
                      os.environ.get("WIFALL_SOURCE_MODE").
                      Pass None when the flag / env var was not supplied.
        config_path:  Path to inference.yaml.

    Returns:
        One of "replay", "mock_live", or "esp32".

    Raises:
        yaml.YAMLError:          config file exists but is not valid YAML.
        pydantic.ValidationError: config parses but contains an invalid
                                  source_mode value.
    """
    if cli_source is not None:
        return cli_source

    try:
        from shared.config import load_inference_config
        return load_inference_config(Path(config_path)).source_mode
    except (FileNotFoundError, OSError):
        # Config file is genuinely absent — use the safe default.
        return _DEFAULT_SOURCE_MODE

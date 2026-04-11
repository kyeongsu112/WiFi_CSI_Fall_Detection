from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from collector.interfaces import PacketSource
    from collector.session_store import SessionStore, SessionWriteResult
    from shared.config import CollectionConfig
    from shared.models import SessionMetadata


def bootstrap_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = bootstrap_repo_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a raw CSI session from a replay or live UDP source."
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML config files.",
    )
    parser.add_argument(
        "--metadata-path",
        default="tests/fixtures/mock_session_metadata.json",
        help="Path to the session metadata JSON file.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session ID override for the loaded metadata.",
    )
    return parser


def load_session_metadata(path: Path) -> "SessionMetadata":
    from shared.models import SessionMetadata

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SessionMetadata.from_dict(payload)


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_packet_source(
    collection: "CollectionConfig",
    *,
    session_id: str | None = None,
    live_reporter: Callable[[str], None] | None = None,
) -> "PacketSource":
    from collector.replay import JsonlReplaySource
    from collector.receiver import UdpEsp32BinaryPacketSource, UdpJsonPacketSource

    if collection.source_type == "replay":
        replay_path = resolve_repo_path(collection.replay_input_path)
        return JsonlReplaySource(replay_path)

    if collection.source_type == "live":
        common_kwargs = {
            "host": collection.udp_host,
            "port": collection.udp_port,
            "timeout_seconds": 0.25,
            "skip_invalid_datagrams": live_reporter is not None,
            "error_reporter": (
                (lambda message: live_reporter(format_live_warning_message(message)))
                if live_reporter is not None
                else None
            ),
        }

        if collection.live_udp_format == "json":
            return UdpJsonPacketSource(**common_kwargs)
        if collection.live_udp_format == "esp32_adr018":
            if session_id is None or not session_id.strip():
                raise ValueError(
                    "Binary live collection requires a session_id for packet normalization."
                )
            return UdpEsp32BinaryPacketSource(
                **common_kwargs,
                session_id=session_id,
            )
        raise ValueError(
            f"Unsupported live UDP format: {collection.live_udp_format}"
        )

    raise ValueError(f"Unsupported collection source_type: {collection.source_type}")


def run_collection(
    *,
    session_store: "SessionStore",
    metadata: "SessionMetadata",
    packet_source: "PacketSource",
    source_type: str,
    expected_nodes: tuple[str, ...] = (),
    health_timeout_seconds: float | None = None,
    live_session_duration_seconds: float | None = None,
    reporter: Callable[[str], None] | None = None,
    now_fn: Callable[[], float] = time.monotonic,
) -> "SessionWriteResult":
    stop_state = {"reason": "completed", "signal": None}
    original_request_stop = install_live_stop_tracking(packet_source, source_type, stop_state)
    previous_handlers = install_live_shutdown_handlers(
        packet_source,
        source_type,
        on_shutdown=lambda signum: record_interrupted_stop(stop_state, signum),
    )
    duration_timer = install_live_duration_timer(
        packet_source=packet_source,
        source_type=source_type,
        duration_seconds=live_session_duration_seconds,
        on_timeout=lambda: record_duration_stop(stop_state),
    )
    completed = False
    if source_type == "live" and reporter is not None:
        reporter(format_live_start_message(packet_source, live_session_duration_seconds))
    try:
        tracked_packet_source = packet_source
        if source_type == "live" and health_timeout_seconds is not None and reporter is not None:
            from collector.health import observe_live_packet_source

            tracked_packet_source = observe_live_packet_source(
                packet_source,
                expected_nodes=expected_nodes,
                timeout_seconds=health_timeout_seconds,
                reporter=lambda message: reporter(format_live_health_message(message)),
                now_fn=now_fn,
            )
        result = session_store.write_session(metadata, tracked_packet_source)
        completed = True
        return result
    finally:
        cancel_duration_timer(duration_timer)
        restore_signal_handlers(previous_handlers)
        restore_live_stop_tracking(packet_source, original_request_stop)
        close_packet_source(packet_source)
        if source_type == "live" and reporter is not None and completed:
            reporter(format_live_stop_message(stop_state, live_session_duration_seconds))


def install_live_stop_tracking(
    packet_source: Any,
    source_type: str,
    stop_state: dict[str, Any],
) -> Callable[[], None] | None:
    if source_type != "live":
        return None

    request_stop = getattr(packet_source, "request_stop", None)
    if not callable(request_stop):
        raise ValueError("Live packet source must support explicit shutdown.")

    def _tracked_request_stop() -> None:
        record_manual_stop(stop_state)
        request_stop()

    setattr(packet_source, "request_stop", _tracked_request_stop)
    return request_stop


def restore_live_stop_tracking(
    packet_source: Any,
    original_request_stop: Callable[[], None] | None,
) -> None:
    if original_request_stop is None:
        return
    setattr(packet_source, "request_stop", original_request_stop)


def install_live_shutdown_handlers(
    packet_source: Any,
    source_type: str,
    *,
    on_shutdown: Callable[[int], None] | None = None,
) -> dict[int, Any]:
    if source_type != "live":
        return {}

    request_stop = getattr(packet_source, "request_stop", None)
    if not callable(request_stop):
        raise ValueError("Live packet source must support explicit shutdown.")

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        if on_shutdown is not None:
            on_shutdown(signum)
        request_stop()

    previous_handlers: dict[int, Any] = {}
    for sig in _supported_shutdown_signals():
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_shutdown)
    return previous_handlers


def install_live_duration_timer(
    *,
    packet_source: Any,
    source_type: str,
    duration_seconds: float | None,
    on_timeout: Callable[[], None] | None = None,
) -> threading.Timer | None:
    if source_type != "live" or duration_seconds is None:
        return None

    request_stop = getattr(packet_source, "request_stop", None)
    if not callable(request_stop):
        raise ValueError("Live packet source must support explicit shutdown.")

    def _handle_timeout() -> None:
        if on_timeout is not None:
            on_timeout()
        request_stop()

    timer = threading.Timer(duration_seconds, _handle_timeout)
    timer.daemon = True
    timer.start()
    return timer


def cancel_duration_timer(timer: threading.Timer | None) -> None:
    if timer is None:
        return
    timer.cancel()


def restore_signal_handlers(previous_handlers: dict[int, Any]) -> None:
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)


def close_packet_source(packet_source: Any) -> None:
    close = getattr(packet_source, "close", None)
    if callable(close):
        close()


def _supported_shutdown_signals() -> tuple[int, ...]:
    supported = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        supported.append(signal.SIGTERM)
    return tuple(supported)


def record_manual_stop(stop_state: dict[str, Any]) -> None:
    if stop_state["reason"] == "completed":
        stop_state["reason"] = "manual"


def record_interrupted_stop(stop_state: dict[str, Any], signum: int) -> None:
    stop_state["reason"] = "interrupted"
    stop_state["signal"] = _signal_name(signum)


def record_duration_stop(stop_state: dict[str, Any]) -> None:
    if stop_state["reason"] != "interrupted":
        stop_state["reason"] = "duration"


def format_live_start_message(packet_source: Any, duration_seconds: float | None) -> str:
    fields: list[tuple[str, str]] = [("event", "start")]
    host = getattr(packet_source, "host", None)
    port = getattr(packet_source, "port", None)
    if host is not None:
        fields.append(("host", str(host)))
    if port is not None:
        fields.append(("port", str(port)))
    fields.append(("duration_seconds", _format_optional_float(duration_seconds)))
    return _format_live_message(fields)


def format_live_health_message(message: str) -> str:
    event_name = "warning" if message.startswith("status=degraded") else "health"
    return _format_live_message([("event", event_name)]) + f" {message}"


def format_live_warning_message(message: str) -> str:
    return _format_live_message([("event", "warning")]) + f" message={message}"


def format_live_stop_message(
    stop_state: dict[str, Any], duration_seconds: float | None
) -> str:
    fields: list[tuple[str, str]] = [("event", "stop"), ("reason", stop_state["reason"])]
    if stop_state["reason"] == "interrupted" and stop_state["signal"] is not None:
        fields.append(("signal", str(stop_state["signal"])))
    if stop_state["reason"] == "duration":
        fields.append(("duration_seconds", _format_optional_float(duration_seconds)))
    return _format_live_message(fields)


def _format_live_message(fields: list[tuple[str, str]]) -> str:
    return "live " + " ".join(f"{key}={value}" for key, value in fields)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}"


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


_DEFAULT_HEALTH_TIMEOUT_SECONDS = 3.0


def _load_health_timeout(config_dir: Path) -> float:
    """Return health_timeout_seconds from inference.yaml, or a safe default.

    inference.yaml is optional for collection — a missing, malformed, or
    invalid config must never prevent replay or live collection from running.
    All parsing errors are caught here so the caller never sees them.
    """
    try:
        from shared.config import load_inference_config
        return float(
            load_inference_config(config_dir / "inference.yaml").health_timeout_seconds
        )
    except (FileNotFoundError, OSError):
        pass  # file simply absent — use default silently
    except Exception as exc:
        # Malformed YAML, Pydantic ValidationError, or unexpected error.
        # Warn so the operator knows the config is broken, but do not abort.
        print(
            f"Warning: could not read health_timeout from inference.yaml "
            f"({type(exc).__name__}: {exc}); "
            f"using default {_DEFAULT_HEALTH_TIMEOUT_SECONDS} s",
            file=sys.stderr,
        )
    return _DEFAULT_HEALTH_TIMEOUT_SECONDS


def main() -> int:
    from collector.session_store import SessionStore
    from shared.config import load_collection_config

    args = build_parser().parse_args()
    config_dir = resolve_repo_path(args.config_dir)

    # Load only what collect needs: collection config is mandatory;
    # inference config is optional (only health_timeout_seconds is used for
    # live collection health checks — replay collection does not need it).
    collection_cfg = load_collection_config(config_dir / "collection.yaml")
    health_timeout = _load_health_timeout(config_dir)

    metadata_path = resolve_repo_path(args.metadata_path)
    metadata = load_session_metadata(metadata_path)
    if args.session_id:
        metadata = replace(metadata, session_id=args.session_id)

    output_dir = resolve_repo_path(collection_cfg.session_output_dir)
    session_store = SessionStore(output_dir)

    try:
        packet_source = build_packet_source(
            collection_cfg,
            session_id=metadata.session_id,
            live_reporter=_report_live_status,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    result = run_collection(
        session_store=session_store,
        metadata=metadata,
        packet_source=packet_source,
        source_type=collection_cfg.source_type,
        expected_nodes=tuple(collection_cfg.expected_nodes),
        health_timeout_seconds=health_timeout,
        live_session_duration_seconds=collection_cfg.live_session_duration_seconds,
        reporter=_report_live_status,
    )

    print(f"session_id={metadata.session_id}")
    print(f"packet_count={result.packet_count}")
    print("node_ids=" + ",".join(result.node_ids))
    print(f"output_path={result.session_dir}")
    return 0


def _report_live_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

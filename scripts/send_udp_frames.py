"""Local UDP loopback sender for Esp32Source development testing.

Sends synthetic v1 JSON packets to a UDP port to simulate an ESP32-S3 device.
Useful for exercising the full inference pipeline (source → model → SSE dashboard)
without real hardware.

Packet format (v1)
------------------
    {"ts": <float>, "fi": <int>, "amp": [<float> × 52], "sid": "<str>"}

Usage examples
--------------
    # Send 500 background frames at 20 fps (no fall):
    python scripts/send_udp_frames.py --count 500

    # Simulate a fall burst at frames 200–299 (elevated amplitudes):
    python scripts/send_udp_frames.py --count 500 --fall-start 200 --fall-end 299

    # Custom host/port (must match esp32_udp_host/port in inference config):
    python scripts/send_udp_frames.py --host 127.0.0.1 --port 5005 --count 200

    # Continuous stream (Ctrl-C to stop):
    python scripts/send_udp_frames.py --count 0

    # Fast send for buffering tests (no inter-frame delay):
    python scripts/send_udp_frames.py --count 100 --delay 0

Loopback test workflow
----------------------
Terminal 1 — start the dashboard in esp32 mode:
    python scripts/replay_dashboard.py --source esp32 --config configs/inference.live_esp32.example.yaml

Terminal 2 — send synthetic frames:
    python scripts/send_udp_frames.py --count 500 --fall-start 200 --fall-end 250

Then open http://127.0.0.1:8000 and watch the alert state transitions.

Real ESP32 workflow
-------------------
1. Flash firmware that sends v1 JSON packets to LAPTOP_IP:5005.
2. Connect ESP32-S3 and laptop to the same Wi-Fi network.
   SSID and password are in the .env file (untracked). See .env.example.
3. Run: python scripts/replay_dashboard.py --source esp32
4. Open: http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_N_SUBCARRIERS = 52
_DEFAULT_BACKGROUND_LOC = 20.0   # mean amplitude for background (non-fall) frames
_DEFAULT_FALL_LOC = 80.0         # mean amplitude for simulated fall frames
_DEFAULT_SCALE = 10.0            # amplitude noise std-dev


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send synthetic v1 UDP frames to Esp32Source",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Destination host (must match esp32_udp_host bind address)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="Destination UDP port (must match esp32_udp_port in inference config)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of frames to send.  0 = run indefinitely until Ctrl-C.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Sleep between frames in seconds (20 fps default).  0 = as fast as possible.",
    )
    parser.add_argument(
        "--sid",
        default="loopback",
        help="Session/device id embedded in each packet (sid field).",
    )
    parser.add_argument(
        "--fall-start",
        type=int,
        default=None,
        dest="fall_start",
        help="First frame index (inclusive) of the simulated fall burst.",
    )
    parser.add_argument(
        "--fall-end",
        type=int,
        default=None,
        dest="fall_end",
        help="Last frame index (inclusive) of the simulated fall burst.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible amplitude noise.",
    )
    return parser.parse_args()


def _is_fall_frame(frame_index: int, fall_start: int | None, fall_end: int | None) -> bool:
    if fall_start is None or fall_end is None:
        return False
    return fall_start <= frame_index <= fall_end


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    if args.fall_start is not None and args.fall_end is None:
        args.fall_end = args.fall_start + 99  # default: one full window of fall

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    count_str = str(args.count) if args.count > 0 else "∞"
    fps_str = f"{1.0 / args.delay:.1f} fps" if args.delay > 0 else "max speed"
    print(f"Sending {count_str} frames → udp://{args.host}:{args.port}")
    print(f"  SID  : {args.sid}")
    print(f"  Rate : {fps_str}  (delay={args.delay}s)")
    if args.fall_start is not None:
        print(f"  Fall : frames {args.fall_start}–{args.fall_end} (elevated amplitudes)")
    print("  Press Ctrl-C to stop.\n")

    frame_index = 0
    try:
        while args.count == 0 or frame_index < args.count:
            is_fall = _is_fall_frame(frame_index, args.fall_start, args.fall_end)
            loc = _DEFAULT_FALL_LOC if is_fall else _DEFAULT_BACKGROUND_LOC
            amp = rng.normal(loc=loc, scale=_DEFAULT_SCALE, size=_N_SUBCARRIERS)
            amp = np.clip(amp, 0.0, None).tolist()

            pkt = json.dumps(
                {
                    "ts": time.time(),
                    "fi": frame_index,
                    "amp": amp,
                    "sid": args.sid,
                }
            ).encode("utf-8")
            sock.sendto(pkt, target)

            frame_index += 1
            if frame_index % 100 == 0:
                label = "FALL" if is_fall else "bg  "
                print(f"  [{label}] {frame_index} frames sent", end="\r", flush=True)

            if args.delay > 0:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print(f"\nDone. Sent {frame_index} frame(s).")


if __name__ == "__main__":
    main()

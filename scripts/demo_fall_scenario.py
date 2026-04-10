from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np


N_SUBCARRIERS = 52


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a scripted fall scenario to the Esp32Source UDP listener.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Destination UDP host.")
    parser.add_argument("--port", type=int, default=5005, help="Destination UDP port.")
    parser.add_argument("--sid", default="demo-scenario", help="sid field embedded in packets.")
    parser.add_argument("--frame-delay", type=float, default=0.05, help="Delay between frames.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    return parser


def _phase_frames(
    sock: socket.socket,
    target: tuple[str, int],
    *,
    sid: str,
    start_index: int,
    count: int,
    amplitude_loc: float,
    amplitude_scale: float,
    delay: float,
    rng: np.random.Generator,
    label: str,
) -> int:
    for frame_index in range(start_index, start_index + count):
        amplitude = rng.normal(
            loc=amplitude_loc,
            scale=amplitude_scale,
            size=N_SUBCARRIERS,
        )
        amplitude = np.clip(amplitude, 0.0, None)
        packet = json.dumps(
            {
                "ts": time.time(),
                "fi": frame_index,
                "amp": amplitude.tolist(),
                "sid": sid,
            }
        ).encode("utf-8")
        sock.sendto(packet, target)
        if delay > 0:
            time.sleep(delay)

    print(f"{label}: {count} frame(s)")
    return start_index + count


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)
    target = (args.host, args.port)
    frame_index = 0

    print(f"Sending scripted demo scenario to udp://{args.host}:{args.port}")
    print("  phases: warmup -> fall burst -> low motion -> recovery")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        frame_index = _phase_frames(
            sock,
            target,
            sid=args.sid,
            start_index=frame_index,
            count=120,
            amplitude_loc=18.0,
            amplitude_scale=4.0,
            delay=args.frame_delay,
            rng=rng,
            label="warmup",
        )
        frame_index = _phase_frames(
            sock,
            target,
            sid=args.sid,
            start_index=frame_index,
            count=40,
            amplitude_loc=95.0,
            amplitude_scale=18.0,
            delay=args.frame_delay,
            rng=rng,
            label="fall burst",
        )
        frame_index = _phase_frames(
            sock,
            target,
            sid=args.sid,
            start_index=frame_index,
            count=220,
            amplitude_loc=10.0,
            amplitude_scale=1.2,
            delay=args.frame_delay,
            rng=rng,
            label="low motion",
        )
        _phase_frames(
            sock,
            target,
            sid=args.sid,
            start_index=frame_index,
            count=120,
            amplitude_loc=22.0,
            amplitude_scale=5.0,
            delay=args.frame_delay,
            rng=rng,
            label="recovery",
        )

    print("Scenario complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

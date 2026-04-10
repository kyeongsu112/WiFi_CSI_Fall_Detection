"""Launch the WiFall inference dashboard.

Sets environment variables consumed by app/server.py, then starts uvicorn.

Source modes
------------
    replay     Read windows from the WiFall manifest + zip (default).
               Requires --manifest and --zip to exist.

    mock_live  Generate synthetic random windows indefinitely.
               No manifest or zip file needed.  Use this to test the full
               SSE pipeline without WiFall data or hardware.

    esp32      Placeholder for future live ESP32 UDP integration.
               Raises NotImplementedError until implemented.

Usage examples
--------------
    # replay mode (default):
    python scripts/replay_dashboard.py

    # mock live mode:
    python scripts/replay_dashboard.py --source mock_live

    # custom replay with slower pacing:
    python scripts/replay_dashboard.py \\
        --source replay \\
        --manifest artifacts/processed/wifall_manifest.csv \\
        --zip data/WiFall.zip \\
        --config configs/inference.yaml \\
        --delay 0.1 \\
        --port 8000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_VALID_SOURCES = ("replay", "mock_live", "esp32")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WiFall inference dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=None,
        choices=_VALID_SOURCES,
        help=(
            "CSI source type.  If omitted, the value is read from "
            "source_mode in the inference config file (default: replay)."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/processed/wifall_manifest.csv",
        help="Path to wifall_manifest.csv  (replay mode only)",
    )
    parser.add_argument(
        "--zip",
        default="data/WiFall.zip",
        help="Path to WiFall.zip  (replay mode only)",
    )
    parser.add_argument(
        "--config",
        default="configs/inference.yaml",
        help="Path to inference.yaml",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Inter-window sleep in seconds (overrides config value)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable uvicorn auto-reload (dev mode only)",
    )
    return parser.parse_args()


def _check_paths(source_mode: str, args: argparse.Namespace) -> None:
    required = [("config", args.config)]
    if source_mode == "replay":
        required += [("manifest", args.manifest), ("zip", args.zip)]

    missing = [
        f"  {label}: {path}"
        for label, path in required
        if not Path(path).exists()
    ]
    if missing:
        print("ERROR: the following required files are missing:")
        print("\n".join(missing))
        if source_mode == "replay":
            print("\nHave you run the data-prep and training steps?")
            print("  python scripts/prepare_wifall.py --config configs/dataset.yaml")
            print("  python scripts/train_baseline.py --config configs/training_baseline.yaml")
        sys.exit(1)


def main() -> None:
    args = _parse_args()

    # Resolve source mode: CLI --source > config source_mode > "replay"
    from inference.live_source import resolve_source_mode
    source_mode = resolve_source_mode(args.source, args.config)

    _check_paths(source_mode, args)

    os.environ["WIFALL_SOURCE_MODE"] = source_mode
    os.environ["WIFALL_CONFIG_PATH"] = str(args.config)
    if source_mode == "replay":
        os.environ["WIFALL_MANIFEST_PATH"] = str(args.manifest)
        os.environ["WIFALL_ZIP_PATH"]      = str(args.zip)
    if args.delay is not None:
        os.environ["WIFALL_STEP_DELAY"] = str(args.delay)

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed.  Run:  pip install uvicorn[standard]")
        sys.exit(1)

    print(f"Dashboard : http://{args.host}:{args.port}")
    print(f"Source    : {source_mode}")
    if source_mode == "replay":
        print(f"Manifest  : {args.manifest}")
    print(f"Config    : {args.config}")
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

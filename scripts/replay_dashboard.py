"""Launch the WiFall replay dashboard.

Sets environment variables read by app/server.py, then starts uvicorn.

Usage examples:
    # default paths (manifest + zip + config must exist):
    python scripts/replay_dashboard.py

    # custom paths:
    python scripts/replay_dashboard.py \\
        --manifest artifacts/processed/wifall_manifest.csv \\
        --zip data/WiFall.zip \\
        --config configs/inference.yaml \\
        --delay 0.05 \\
        --port 8000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WiFall replay dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/processed/wifall_manifest.csv",
        help="Path to wifall_manifest.csv",
    )
    parser.add_argument(
        "--zip",
        default="data/WiFall.zip",
        help="Path to WiFall.zip",
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


def _check_paths(args: argparse.Namespace) -> None:
    missing = []
    for label, path in [
        ("manifest", args.manifest),
        ("zip",      args.zip),
        ("config",   args.config),
    ]:
        if not Path(path).exists():
            missing.append(f"  {label}: {path}")
    if missing:
        print("ERROR: the following required files are missing:")
        print("\n".join(missing))
        print("\nHave you run the data-prep and training steps?")
        print("  python scripts/prepare_wifall.py --config configs/dataset.yaml")
        print("  python scripts/train_baseline.py --config configs/training_baseline.yaml")
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    _check_paths(args)

    # Pass paths to the server via environment variables
    os.environ["WIFALL_MANIFEST_PATH"] = str(args.manifest)
    os.environ["WIFALL_ZIP_PATH"]      = str(args.zip)
    os.environ["WIFALL_CONFIG_PATH"]   = str(args.config)
    if args.delay is not None:
        os.environ["WIFALL_STEP_DELAY"] = str(args.delay)

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed.  Run:  pip install uvicorn[standard]")
        sys.exit(1)

    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Manifest : {args.manifest}")
    print(f"Model    : loaded from {args.config}")
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

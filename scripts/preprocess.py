from __future__ import annotations

import argparse
from pathlib import Path
import sys


def bootstrap_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = bootstrap_repo_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a raw CSI session into processed windows and baseline features."
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML config files.",
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Session ID under artifacts/raw/ to preprocess.",
    )
    return parser


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    from preprocessing.pipeline import process_raw_session_by_id
    from shared.config import load_collection_config, load_preprocessing_config

    args = build_parser().parse_args(argv)
    config_dir = resolve_repo_path(args.config_dir)

    # Load only what preprocess needs — avoids failures due to unrelated
    # subsystem configs (training.yaml, inference.yaml) being absent or invalid.
    collection_cfg = load_collection_config(config_dir / "collection.yaml")
    preprocessing_cfg = load_preprocessing_config(config_dir / "preprocessing.yaml")

    raw_root = resolve_repo_path(collection_cfg.session_output_dir)
    output_root = raw_root.parent / "processed"

    result = process_raw_session_by_id(
        raw_root,
        session_id=args.session_id,
        preprocessing_config=preprocessing_cfg,
        expected_nodes=collection_cfg.expected_nodes,
        output_root=output_root,
    )

    print(f"session_id={result.session_id}")
    print(f"window_count={result.window_count}")
    print(f"feature_count={result.feature_count}")
    print("node_order=" + ",".join(result.node_order) if result.node_order else "node_order=none")
    print(f"output_path={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

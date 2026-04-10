from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandSpec:
    module_name: str
    help_text: str


COMMANDS: dict[str, CommandSpec] = {
    "collect": CommandSpec("scripts.collect", "Collect a raw replay/live CSI session."),
    "preprocess": CommandSpec("scripts.preprocess", "Convert a raw session into processed windows."),
    "prepare": CommandSpec("scripts.prepare_wifall", "Build the WiFall manifest and split config."),
    "train": CommandSpec("scripts.train_baseline", "Train the baseline fall detector."),
    "eval": CommandSpec("scripts.eval_baseline", "Evaluate the baseline checkpoint."),
    "dashboard": CommandSpec("scripts.replay_dashboard", "Run the local dashboard."),
    "summarize": CommandSpec("scripts.summarize_raw_session", "Summarize a captured raw session."),
    "send-udp": CommandSpec("scripts.send_udp_frames", "Send synthetic UDP frames to Esp32Source."),
    "demo-scenario": CommandSpec("scripts.demo_fall_scenario", "Run a scripted UDP fall demo scenario."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified WiFall CLI entrypoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=tuple(COMMANDS.keys()),
        help="Subcommand to run.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected subcommand.",
    )
    return parser


def run_command(command: str, args: list[str]) -> int:
    spec = COMMANDS[command]
    module = importlib.import_module(spec.module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise RuntimeError(f"Module {spec.module_name!r} does not expose main().")

    signature = inspect.signature(main)
    previous_argv = sys.argv
    sys.argv = [f"python -m app {command}", *args]
    try:
        if signature.parameters:
            result = main(args)
        else:
            result = main()
    finally:
        sys.argv = previous_argv

    if result is None:
        return 0
    return int(result)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(args.command, list(args.args))


if __name__ == "__main__":
    raise SystemExit(main())

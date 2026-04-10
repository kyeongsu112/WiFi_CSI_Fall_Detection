from __future__ import annotations

from types import SimpleNamespace

from app.cli import main, run_command


def test_run_command_passes_args_to_main_with_argv(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(
        "app.cli.importlib.import_module",
        lambda _name: SimpleNamespace(main=fake_main),
    )

    assert run_command("dashboard", ["--port", "8123"]) == 0
    assert captured["argv"] == ["--port", "8123"]


def test_run_command_supports_main_without_argv(monkeypatch) -> None:
    captured = {"called": False}

    def fake_main() -> int:
        captured["called"] = True
        return 7

    monkeypatch.setattr(
        "app.cli.importlib.import_module",
        lambda _name: SimpleNamespace(main=fake_main),
    )

    assert run_command("collect", []) == 7
    assert captured["called"] is True


def test_main_parses_command_and_remainder(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(command: str, args: list[str]) -> int:
        captured["command"] = command
        captured["args"] = args
        return 0

    monkeypatch.setattr("app.cli.run_command", fake_run_command)

    assert main(["demo-scenario", "--host", "127.0.0.1"]) == 0
    assert captured["command"] == "demo-scenario"
    assert captured["args"] == ["--host", "127.0.0.1"]

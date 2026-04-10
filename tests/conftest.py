from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def config_dir(repo_root: Path) -> Path:
    return repo_root / "configs"


@pytest.fixture
def fixture_packets_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "mock_session_packets.jsonl"


@pytest.fixture
def fixture_metadata_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "mock_session_metadata.json"

from pathlib import Path

from pydantic import ValidationError
import pytest
import yaml

from shared.config import load_all_configs, load_collection_config, load_inference_config


def test_load_all_configs(config_dir: Path) -> None:
    bundle = load_all_configs(config_dir)

    assert bundle.collection.source_type in {"replay", "live"}
    assert bundle.collection.live_udp_format in {"json", "esp32_adr018"}
    assert bundle.collection.udp_host
    assert 1 <= bundle.collection.udp_port <= 65535
    assert bundle.collection.expected_nodes
    assert bundle.inference.candidate_threshold == 0.50
    assert bundle.inference.post_fall_inactivity_seconds == 6
    assert bundle.inference.confirm_window_seconds == 8
    assert bundle.preprocessing.window_seconds > 0
    assert bundle.preprocessing.stride_seconds > 0
    assert bundle.training.runtime_labels == ["non_fall", "fall"]


def test_invalid_source_type_fails_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "collection.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "source_type": "bluetooth",
                "live_udp_format": "json",
                "replay_input_path": "mock.jsonl",
                "udp_host": "127.0.0.1",
                "udp_port": 9000,
                "live_session_duration_seconds": None,
                "expected_nodes": ["node-a", "node-b", "node-c"],
                "session_output_dir": "artifacts/raw",
                "flush_interval_ms": 1000,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_collection_config(config_path)


def test_live_esp32_example_configs_load(repo_root: Path) -> None:
    collection = load_collection_config(
        repo_root / "configs" / "collection.live_esp32.example.yaml"
    )
    inference = load_inference_config(
        repo_root / "configs" / "inference.live_esp32.example.yaml"
    )

    assert collection.source_type == "live"
    assert collection.live_udp_format == "esp32_adr018"
    assert collection.udp_host == "0.0.0.0"
    assert collection.udp_port == 5005
    assert collection.expected_nodes == ["1", "2", "3"]
    assert inference.health_timeout_seconds == 5

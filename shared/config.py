from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    log_level: str = "INFO"
    output_mode: Literal["cli", "json"] = "cli"


class CollectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: Literal["replay", "live"] = "replay"
    replay_input_path: str = "tests/fixtures/mock_session_packets.jsonl"
    live_udp_format: Literal["json", "esp32_adr018"] = "json"
    udp_host: str = "127.0.0.1"
    udp_port: int = Field(default=9000, ge=1, le=65535)
    live_session_duration_seconds: float | None = Field(default=None, gt=0)
    expected_nodes: List[str] = Field(
        default_factory=lambda: ["node-a", "node-b", "node-c"]
    )
    session_output_dir: str = "artifacts/raw"
    flush_interval_ms: int = Field(default=1000, ge=1)


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window_seconds: float = Field(default=2.0, gt=0)
    stride_seconds: float = Field(default=0.5, gt=0)
    phase_unwrap_enabled: bool = True
    outlier_zscore_threshold: float = Field(default=3.5, gt=0)
    median_filter_kernel_size: int = Field(default=3, ge=1)
    smoothing_window_size: int = Field(default=3, ge=1)
    selected_subcarriers: List[int] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    labels: List[str] = Field(
        default_factory=lambda: [
            "empty",
            "normal_walk",
            "sit_or_stand_transition",
            "lie_down_intentional",
            "fall",
        ]
    )
    runtime_labels: List[str] = Field(
        default_factory=lambda: ["non_fall", "fall"]
    )
    split_strategy: str = "participant_wise"
    model_type: str = "logistic_regression"
    feature_set: str = "baseline_statistical"
    random_seed: int = 42


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = "artifacts/models/wifall_baseline.pt"
    candidate_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    # Source mode — determines which CsiSource is used by the dashboard.
    # "replay"    : WiFall manifest + zip (default)
    # "mock_live" : synthetic random windows for pipeline testing
    # "esp32"     : placeholder for future live UDP integration
    source_mode: Literal["replay", "mock_live", "esp32"] = "replay"

    # --- Live ESP32 mode (time-based ConfirmationEngine) ---
    post_fall_inactivity_seconds: int = Field(default=6, ge=1)
    motion_floor_threshold: float = Field(default=0.15, ge=0.0)
    confirm_window_seconds: int = Field(default=8, ge=1)
    cooldown_seconds: int = Field(default=5, ge=0)
    health_timeout_seconds: int = Field(default=3, ge=1)

    # --- Replay / mock_live pacing ---
    step_delay_seconds: float = Field(default=0.05, ge=0.0)

    # These fields are accepted from inference.yaml for forward-compatibility
    # but are not currently wired to the active ConfirmationEngine.  The
    # running inference path uses the time-based fields above (post_fall_*,
    # confirm_window_seconds, cooldown_seconds) for all source modes.
    confirm_n_windows: int = Field(default=3, ge=1)
    cooldown_windows: int = Field(default=10, ge=0)

    # --- ESP32 UDP transport (esp32 source_mode only) ---
    # Host/port the laptop listens on for incoming CSI UDP packets.
    # Set esp32_udp_host to "0.0.0.0" to accept from any interface (typical for
    # Wi-Fi hotspot setups), or to a specific NIC IP to restrict to one network.
    # Override these in a local, untracked config file (e.g. configs/inference.local.yaml).
    esp32_udp_host: str = "0.0.0.0"
    esp32_udp_port: int = Field(default=5005, ge=1, le=65535)


@dataclass(frozen=True, slots=True)
class AppConfigBundle:
    app: AppConfig
    collection: CollectionConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    inference: InferenceConfig


def load_app_config(path: Path) -> AppConfig:
    return AppConfig.model_validate(_load_yaml(path))


def load_collection_config(path: Path) -> CollectionConfig:
    return CollectionConfig.model_validate(_load_yaml(path))


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    return PreprocessingConfig.model_validate(_load_yaml(path))


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(_load_yaml(path))


def load_inference_config(path: Path) -> InferenceConfig:
    return InferenceConfig.model_validate(_load_yaml(path))


def load_all_configs(config_dir: Path) -> AppConfigBundle:
    config_dir = Path(config_dir)
    return AppConfigBundle(
        app=load_app_config(config_dir / "app.yaml"),
        collection=load_collection_config(config_dir / "collection.yaml"),
        preprocessing=load_preprocessing_config(config_dir / "preprocessing.yaml"),
        training=load_training_config(config_dir / "training.yaml"),
        inference=load_inference_config(config_dir / "inference.yaml"),
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return payload

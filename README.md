# WiFi Fall Detection MVP

This repository contains a modular Python scaffold for a one-room WiFi CSI fall detection MVP built around three ESP32-S3 nodes.

## Scope

- Detect falls only. This project does not do pose estimation, skeleton extraction, or general human activity understanding.
- Keep the installation limited to a single room and a fixed three-node setup for the MVP.
- Use a two-stage decision flow in later phases: `candidate fall -> confirmed fall`.
- Reduce false alarms by confirming post-fall low-motion behavior after the initial candidate event.

## Phase 1 Contents

Phase 1 establishes the local development scaffold:

- YAML configuration files
- Shared data models and config loader
- Collector interfaces
- Normalized packet parser
- JSONL replay source for mock CSI development
- Raw session store
- Focused tests for the scaffolded logic

Live network collection is intentionally out of scope for Phase 1.

## Python Target

The official project target is Python 3.11 or newer.

Create a virtual environment and install the project with development tools:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Phase 1 Usage

Run the replay-backed collection script with the included fixture metadata:

```powershell
python scripts/collect.py --config-dir configs --metadata-path tests/fixtures/mock_session_metadata.json
```

The script reads normalized JSONL packets from the replay path configured in `configs/collection.yaml`, writes a raw session under `artifacts/raw/<session_id>/`, and prints a short collection summary.

If `source_type` is changed to `live`, the script exits with a clear Phase 1 limitation message instead of guessing hardware behavior.

## Repo Hygiene

Local development state is intentionally kept out of the repository. The project ignores the virtual environment, Python cache directories, pytest and Ruff caches, editable install metadata, and generated runtime outputs under `artifacts/`.

## Repository Shape

The codebase keeps the MVP layers separate:

- `collector/` for packet ingestion and raw session storage
- `preprocessing/` for later CSI cleanup and windowing
- `training/` for later offline model training
- `inference/` for later realtime prediction
- `app/` for later CLI and local API surfaces
- `shared/` for cross-layer contracts only

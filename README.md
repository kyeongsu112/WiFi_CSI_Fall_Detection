# WiFi Fall Detection MVP

ESP32-S3 3대를 사용하는 단일 방 WiFi CSI 기반 낙상 감지 MVP입니다.

핵심 목표는 포즈 추정이 아니라 `candidate fall -> confirmed fall` 2단계 낙상 판정입니다. 최종 확정은 모델 score만으로 내리지 않고, 낙상 후보 이후의 `저움직임(low motion)` 지속 여부를 함께 확인합니다.

## 범위

- 단일 실내 공간, 고정 3노드 배치
- collector / preprocessing / training / inference / app 계층 분리
- 오프라인 WiFall replay와 ESP32 JSON UDP live source 둘 다 지원
- baseline 학습, 평가 리포트, 대시보드 데모 포함

## 설치

공식 타깃은 Python 3.11+ 입니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

추가로 대시보드와 학습/평가를 쓰려면 현재 환경에 `fastapi`, `jinja2`, `uvicorn`, `pandas`, `torch`, `matplotlib` 가 필요합니다.

## 빠른 시작

### 1. WiFall manifest 준비

```powershell
python -m app prepare --config configs/dataset.yaml
```

생성물:

- `artifacts/processed/wifall_manifest.csv`
- `artifacts/processed/wifall_summary.json`
- `configs/splits/wifall_subject_split.yaml`

### 2. baseline 학습

```powershell
python -m app train --config configs/training_baseline.yaml
```

생성물:

- `artifacts/models/wifall_baseline.pt`
- `artifacts/reports/wifall_baseline_subject_distribution.csv`

### 3. baseline 평가

```powershell
python -m app eval --config configs/training_baseline.yaml
```

생성물:

- `artifacts/reports/wifall_baseline_metrics.json`
- ROC / PR / confusion matrix / threshold sweep 리포트

### 4. replay 대시보드 실행

```powershell
python -m app dashboard --source replay --config configs/inference.yaml
```

브라우저에서 `http://127.0.0.1:8000` 을 열면 fall probability, motion score, alert state, source status를 볼 수 있습니다.

## 통합 CLI

기존 개별 스크립트는 그대로 유지되며, 이제 아래 통합 진입점도 사용할 수 있습니다.

```powershell
python -m app collect
python -m app preprocess --session-id session_2026_04_07_001
python -m app train
python -m app eval
python -m app dashboard
python -m app summarize --session-dir artifacts/raw/session_2026_04_07_001
python -m app send-udp --count 500
python -m app demo-scenario
```

## 수집과 전처리

### replay 기반 raw session 수집

```powershell
python -m app collect --config-dir configs --metadata-path tests/fixtures/mock_session_metadata.json
```

### raw session 전처리

```powershell
python -m app preprocess --config-dir configs --session-id session_2026_04_07_001
```

전처리 파이프라인은 다음을 수행합니다.

- amplitude / phase 디코딩
- phase unwrap
- robust z-score 기반 outlier clipping
- median filter
- moving-average smoothing
- windowing
- baseline statistical feature 추출

## ESP32 live 모드

collector live 수집과 inference live 대시보드는 서로 다른 wire format을 사용합니다.

- collector live 수집: `esp32_adr018` binary UDP
- inference live 대시보드: JSON v1 UDP

이 관계는 `docs/SYSTEM_ARCHITECTURE.md` 에 정리되어 있습니다.

### collector smoke test

```powershell
Copy-Item configs\collection.live_esp32.example.yaml configs\collection.yaml -Force
python -m app collect --config-dir configs --metadata-path tests/fixtures/mock_session_metadata.json --session-id session_live_smoke_001
```

### inference live dashboard

로컬 전용 설정 파일을 권장합니다.

```powershell
Copy-Item configs\inference.live_esp32.example.yaml configs\inference.local.yaml
python -m app dashboard --source esp32 --config configs/inference.local.yaml
```

`configs/inference.local.yaml` 은 `.gitignore` 에 포함되어 있으므로 Wi-Fi 관련 로컬 값이나 호스트/포트를 커밋하지 않아도 됩니다.

## 시연용 UDP 시나리오

실기기 없이도 live 대시보드를 데모할 수 있습니다.

### 단순 synthetic sender

```powershell
python -m app send-udp --host 127.0.0.1 --port 5005 --count 500 --fall-start 200 --fall-end 260
```

### 낙상 시나리오 래퍼

```powershell
python -m app demo-scenario --host 127.0.0.1 --port 5005
```

이 시나리오는 `warmup -> fall burst -> low motion -> recovery` 순서로 프레임을 보냅니다.

## 대시보드 특징

- alert state: `idle`, `candidate`, `confirmed`, `cooldown`
- fall probability gauge
- motion score gauge
- source transport status / active sender / packet counters
- confirmed 전환 시 시각 플래시 + 비프음

## 테스트

```powershell
.\.venv313\Scripts\python.exe -m pytest -q
```

현재 로컬 검증 기준 전체 테스트가 통과합니다.

## 저장소 구조

```text
collector/        # 노드 데이터 수신 및 저장
preprocessing/    # CSI 정제, 필터, 윈도우, feature 추출
training/         # 데이터셋 로드, 학습, 평가
inference/        # 실시간 추론, 2단계 confirmation
app/              # 통합 CLI, 대시보드 서버
configs/          # 설정 파일
artifacts/        # 모델, 리포트, 로그
docs/             # 아키텍처, 테스트, 데이터 문서
scripts/          # 개별 실행 스크립트
tests/            # 단위/통합 테스트
```

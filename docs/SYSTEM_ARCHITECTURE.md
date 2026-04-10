# SYSTEM_ARCHITECTURE.md

## 1. 개요
이 문서는 WiFi CSI 기반 낙상 감지 MVP의 권장 소프트웨어 구조와 데이터 흐름을 정의한다.

목표는 다음 세 가지다.
- collector, preprocessing, training, inference, app을 명확히 분리
- raw data -> processed data -> model -> alerts 흐름을 일관되게 유지
- 추후 노드 수 증가나 모델 교체가 가능하도록 모듈화

---

## 2. 권장 저장소 구조

```text
wifi-fall-detection/
  AGENTS.md
  PRD.md
  README.md
  pyproject.toml
  configs/
    app.yaml
    collection.yaml
    preprocessing.yaml
    training.yaml
    inference.yaml
  collector/
    __init__.py
    receiver.py
    packet_parser.py
    session_store.py
    replay.py
    health.py
  preprocessing/
    __init__.py
    phase.py
    filters.py
    windowing.py
    features.py
    pipeline.py
  training/
    __init__.py
    dataset.py
    split.py
    train_baseline.py
    evaluate.py
    metrics.py
  inference/
    __init__.py
    online_buffer.py
    predictor.py
    confirmation.py
    events.py
  app/
    __init__.py
    __main__.py
    cli.py
    server.py
    templates/
      index.html
  tests/
    fixtures/
    test_packet_parser.py
    test_windowing.py
    test_confirmation.py
  artifacts/
    raw/
    processed/
    models/
    reports/
    events/
  scripts/
    collect.py
    prepare_wifall.py
    preprocess.py
    train_baseline.py
    eval_baseline.py
    replay_dashboard.py
``` 

---

## 3. 런타임 데이터 흐름

```text
ESP32 nodes
  -> receiver
  -> packet_parser
  -> session_store
  -> preprocessing pipeline
  -> feature/window generation
  -> predictor
  -> confirmation engine
  -> event logger / alert UI
```

### 단계별 설명
1. **receiver** 가 UDP 또는 socket으로 노드 데이터를 받는다.
2. **packet_parser** 가 표준 내부 포맷으로 정규화한다.
3. **session_store** 가 raw 이벤트를 파일로 기록한다.
4. **preprocessing pipeline** 이 amplitude/phase 처리와 필터링을 수행한다.
5. **windowing** 이 고정 길이 시퀀스를 만든다.
6. **predictor** 가 모델 추론을 수행한다.
7. **confirmation engine** 이 candidate fall을 confirmed fall로 승격할지 결정한다.
8. 결과는 **event log** 와 **UI** 로 전달된다.

---

## 3.1 ESP32 live 포맷 관계

현재 저장소에는 ESP32 관련 wire format이 두 가지 공존한다.

### collector live 수집용
- 위치: `collector/receiver.py`
- 포맷: `esp32_adr018` binary UDP
- 목적: raw CSI 수집과 세션 저장
- 출력: `CsiPacket.raw_payload.csi_raw.iq_bytes_hex`

### inference live 대시보드용
- 위치: `inference/live_source.py`
- 포맷: JSON v1 UDP
- 목적: 빠른 실시간 데모와 대시보드 확인
- 예시:

```json
{"ts": 1712700000.123, "fi": 42, "amp": [1.0, 2.0], "sid": "dev-01"}
```

### 왜 둘 다 있는가
- binary ADR-018은 collector에서 raw 재생성과 저장에 유리하다.
- JSON v1은 inference live 데모에서 파싱과 시뮬레이션이 단순하다.
- 둘은 역할이 다르며, collector와 inference 계층을 직접 섞지 않는다.

운영자는 binary raw 수집과 JSON live inference를 서로 다른 경로로 이해해야 한다.

---

## 4. 핵심 인터페이스 설계

## 4.1 내부 표준 패킷
가능하면 노드별 입력 형식이 달라도 아래 내부 스키마로 변환한다.

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class CsiPacket:
    timestamp: float
    node_id: str
    session_id: str
    seq: int | None
    rssi: float | None
    channel: int | None
    amplitude: list[float] | None
    phase: list[float] | None
    raw_payload: dict[str, Any] | None
```

## 4.2 윈도우 객체
```python
@dataclass
class CsiWindow:
    window_id: str
    start_ts: float
    end_ts: float
    node_ids: list[str]
    features: dict[str, float] | dict[str, list[float]]
    label: str | None = None
```

## 4.3 예측 결과 객체
```python
@dataclass
class Prediction:
    timestamp: float
    fall_prob: float
    motion_score: float
    state: str  # idle | candidate | confirmed
    reasons: list[str]
```

---

## 5. 모듈별 책임

## 5.1 collector/
### 책임
- 네트워크 수신
- 패킷 파싱
- 세션 시작/종료
- raw 로그 저장
- 노드 health 추적

### 하지 말아야 할 일
- 학습 코드 포함 금지
- 모델 추론 로직 혼합 금지

## 5.2 preprocessing/
### 책임
- amplitude/phase 분리
- phase unwrap
- denoise
- smoothing
- drift correction
- windowing
- feature extraction

### 구현 팁
- batch와 online에서 공유 가능한 순수 함수 중심으로 작성
- config-driven 파이프라인 유지

## 5.3 training/
### 책임
- processed dataset 로드
- split 생성
- baseline 학습
- 성능 평가
- report 저장

### 권장 baseline
- feature 기반: RandomForest, XGBoost, Logistic Regression
- sequence 기반: 작은 1D CNN 또는 가벼운 ResNet 변형

## 5.4 inference/
### 책임
- online buffer 관리
- 실시간 window 생성
- 모델 로드
- 추론 실행
- 후보/확정 이벤트 생성

## 5.5 app/
### 책임
- CLI 진입점
- optional API/server
- dashboard
- alert 출력

---

## 6. 2단계 낙상 판정 엔진

낙상 최종 경보는 단일 모델 확률만으로 내리지 않는다.

### 상태 머신
```text
idle
  -> candidate   (fall_prob > threshold_candidate 또는 급격한 motion spike)
  -> confirmed   (candidate 후 inactivity window 충족)
  -> idle        (조건 미충족 또는 cooldown 후 복귀)
```

### 권장 규칙
- `fall_prob >= candidate_threshold`
- 후보 발생 후 `inactivity_seconds >= post_fall_inactivity_threshold`
- confirm 후 `cooldown_seconds` 동안 중복 경보 방지

### 예시 설정
```yaml
candidate_threshold: 0.75
post_fall_inactivity_seconds: 6
motion_floor_threshold: 0.15
confirm_window_seconds: 8
cooldown_seconds: 5
```

---

## 7. 구성 파일 전략
모든 주요 파라미터는 config에서 조절 가능해야 한다.

### collection.yaml
- udp_port
- expected_nodes
- session_output_dir
- flush_interval_ms

### preprocessing.yaml
- smoothing_window
- phase_unwrap_enabled
- outlier_method
- window_seconds
- stride_seconds
- selected_subcarriers

### training.yaml
- labels
- split_strategy
- model_type
- feature_set
- random_seed

### inference.yaml
- model_path
- candidate_threshold
- inactivity_threshold
- cooldown_seconds
- health_timeout_seconds

---

## 8. 로그 및 아티팩트

### raw 로그
- jsonl 또는 parquet 권장
- 세션별 디렉터리 저장

### processed 로그
- window 단위 feature 테이블
- train/val/test split 파일 별도 저장

### event 로그
- candidate/confirmed event 저장
- 사후 디버깅 가능하도록 관련 score 포함

---

## 9. 실패 처리
- 노드 일부 누락 시 degraded mode 허용
- expected node count 미달이면 warning 상태로 표시
- 파싱 실패 패킷은 드롭하되 카운트 기록
- 예측 오류 시 프로세스 전체 중단 대신 skip + log 우선

---

## 10. 설계 원칙 요약
- baseline 먼저
- pure function 선호
- config 우선
- raw와 processed 분리
- candidate와 confirmed 분리
- replay 가능성 확보

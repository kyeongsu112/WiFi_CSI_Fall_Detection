# AGENTS.md

이 저장소는 **ESP32-S3 3대를 이용한 WiFi CSI 기반 낙상 감지 MVP** 를 구현한다.
Codex는 이 저장소에서 작업할 때 아래 지침을 반드시 따른다.

## 1. 제품 목표
- 목표는 **포즈 추정이 아니라 낙상 감지**다.
- 범위는 **실내 1개 방**이다.
- 실시간 파이프라인은 **candidate fall -> confirmed fall** 의 2단계 판정을 사용한다.
- 오탐 감소를 위해 **낙상 후 5~10초 저움직임 확인** 로직을 반드시 포함한다.

## 2. 절대 하지 말 것
- 포즈 추정, skeleton, keypoint 추출 기능을 넣지 말 것.
- 의료기기 수준의 안전성이나 임상적 문구를 쓰지 말 것.
- 하드코딩된 절대 경로를 넣지 말 것.
- 수집 로직, 학습 로직, 추론 로직을 한 파일에 섞지 말 것.
- 외부 서비스 의존성을 불필요하게 늘리지 말 것.
- 설명 없이 큰 구조 개편을 하지 말 것.

## 3. 선호 기술 스택
- Python 3.11+
- FastAPI 또는 Flask는 필요할 때만 사용
- CLI 우선, 웹 UI는 경량으로
- 모델 baseline은 scikit-learn 또는 PyTorch 중 간단한 쪽부터 시작
- 설정은 YAML 또는 TOML 또는 JSON으로 분리
- 데이터 저장은 parquet, jsonl, npy, csv 중 목적에 맞게 선택

## 4. 기본 아키텍처 원칙
코드는 아래 계층을 유지한다.

```text
collector/        # 노드 데이터 수신 및 저장
preprocessing/    # CSI 정제, 윈도우, feature 추출
training/         # 데이터셋 로드, 학습, 평가
inference/        # 실시간 추론
app/              # CLI, dashboard, alert orchestration
configs/          # 설정 파일
artifacts/        # 모델, 리포트, 로그
```

한 계층은 다른 계층의 내부 구현을 직접 침범하지 않는다.

## 5. 작업 방식
복잡한 작업은 바로 코딩하지 말고 다음 순서를 따른다.
1. 현재 구조 확인
2. 변경 범위 요약
3. 구현 계획 제안
4. 필요한 경우 TODO를 나눠 단계별 진행
5. 코드 작성
6. 테스트/정적 검사 실행
7. 결과와 남은 리스크 요약

## 6. 프롬프트를 받았을 때의 기본 행동
사용자 요청이 불명확하더라도, 아래 네 요소를 스스로 구조화해서 해석한다.
- Goal: 무엇을 만들거나 바꾸는가
- Context: 어떤 파일, 모듈, 요구사항이 관련되는가
- Constraints: 반드시 지켜야 할 제한은 무엇인가
- Done when: 무엇이 만족되면 완료인가

## 7. 설계 판단 원칙
- 처음에는 **작동하는 baseline** 을 우선한다.
- fancy한 모델보다 **재현 가능성** 과 **디버깅 용이성** 을 우선한다.
- 모델보다 **데이터 파이프라인, 라벨 일관성, 평가 리포트** 를 더 중요하게 본다.
- 낙상 최종 판정은 항상 **모델 score + 후속 저움직임 규칙** 으로 구성한다.
- feature engineering과 learned model을 교체 가능한 형태로 유지한다.

## 8. 구현 우선순위
1. 데이터 수집 안정화
2. 저장 포맷 정의
3. 리플레이 가능 파이프라인
4. baseline feature 추출
5. baseline model 학습
6. 실시간 추론
7. 경고 규칙 튜닝
8. 대시보드/UX 개선

## 9. 데이터 계약 원칙
각 수집 레코드는 최소한 아래 필드를 갖도록 설계한다.
- timestamp
- node_id
- session_id
- rssi
- channel
- csi_raw or derived values
- packet_seq if available

전처리 산출물은 아래 메타를 유지한다.
- window_id
- start_ts
- end_ts
- label
- feature_version
- source_session_ids

## 10. 테스트 원칙
새 기능을 만들면 아래 중 하나 이상을 반드시 추가한다.
- 단위 테스트
- 리플레이 기반 통합 테스트
- 샘플 입력/출력 fixture
- 평가 리포트 자동 생성

추론/전처리/규칙 로직은 가능하면 pure function 위주로 작성한다.

## 11. 완료 정의
작업 완료로 보려면 아래를 만족해야 한다.
- 코드가 실행 가능하다.
- 최소한 하나 이상의 검증 수단이 추가되었다.
- 변경 이유와 한계가 문서 또는 주석에 남아 있다.
- 설정값(threshold 등)은 코드 밖으로 분리되었다.
- 기존 구조를 깨지 않는다.

## 12. 커밋/PR 스타일
변경은 가능한 한 작은 단위로 나눈다.
PR 설명 또는 요약에는 아래를 포함한다.
- 무엇을 바꿨는가
- 왜 바꿨는가
- 어떻게 검증했는가
- 아직 남은 리스크는 무엇인가

## 13. 성능 관련 주의
- 실시간 추론 경로에서는 불필요한 pandas heavy operation을 피한다.
- 큰 배열 복사를 최소화한다.
- 로그는 verbose/debug 레벨을 분리한다.
- 네트워크 수신 실패 시 exponential backoff 또는 재시도 전략을 둔다.

## 14. 문서 우선순위
구현 중 판단이 필요하면 아래 순서로 따른다.
1. PRD.md
2. docs/SYSTEM_ARCHITECTURE.md
3. docs/DATASET_AND_LABELING.md
4. docs/TEST_PLAN.md
5. 이 AGENTS.md

충돌 시 PRD.md 의 제품 범위를 우선한다.

# TEST_PLAN.md

## 1. 목적
이 문서는 낙상 감지 MVP 구현 시 필요한 테스트 전략과 완료 기준을 정의한다.

핵심은 다음이다.
- collector가 안정적으로 데이터를 받는가
- preprocessing이 일관된 결과를 만드는가
- model + confirmation rule이 낙상을 분리하는가
- 실시간 루프가 죽지 않고 상태를 유지하는가

---

## 2. 테스트 레벨

## 2.1 단위 테스트
대상:
- packet_parser
- phase unwrap
- outlier filter
- windowing
- confirmation state machine
- config loader

### 예시
- 잘못된 패킷 입력 시 parser가 안전하게 실패하는가
- 2초 window와 0.5초 stride가 정확히 계산되는가
- confirmation engine이 candidate -> confirmed -> cooldown 흐름을 지키는가

## 2.2 통합 테스트
대상:
- recorded raw session replay
- preprocessing -> feature -> prediction end-to-end
- CLI 명령 실행

### 예시
- fixture session을 재생하면 예측 이벤트가 생성되는가
- node 하나가 누락되어도 프로세스가 유지되는가

## 2.3 평가 테스트
대상:
- baseline model metrics
- confusion matrix generation
- per-class report generation

---

## 3. 우선 작성할 테스트

### 1순위
- `test_packet_parser.py`
- `test_windowing.py`
- `test_confirmation.py`

### 2순위
- `test_feature_pipeline.py`
- `test_replay_inference.py`

### 3순위
- `test_cli_commands.py`
- `test_report_generation.py`

---

## 4. 완료 기준
작업은 아래 조건을 만족할 때 완료로 판단한다.

### collector 관련
- malformed packet 처리 테스트 존재
- session 저장 테스트 존재
- replay 테스트 존재

### preprocessing 관련
- feature shape 안정성 테스트 존재
- window 수 계산 테스트 존재
- 설정 변경 시 파이프라인이 예측 가능하게 반응

### inference 관련
- candidate/confirmed 구분 테스트 존재
- cooldown 테스트 존재
- threshold 변경 테스트 존재

### training 관련
- split manifest 생성 테스트 존재
- metrics report 생성 테스트 존재

---

## 5. 수동 데모 체크리스트

### 설치 전
- [ ] 노드 3대 전원 및 네트워크 연결 확인
- [ ] config의 expected_nodes 확인
- [ ] output 디렉터리 권한 확인

### 수집 중
- [ ] 모든 노드 heartbeat 확인
- [ ] raw session 파일 생성 확인
- [ ] timestamps 증가 확인

### 데모 중
- [ ] normal walk에서 confirmed fall이 발생하지 않음
- [ ] fall 시 candidate 이벤트가 먼저 보임
- [ ] 저움직임 후 confirmed fall 발생
- [ ] cooldown 중 중복 알림이 억제됨

---

## 6. 리그레션 방지 규칙
- parsing 포맷을 바꾸면 fixture도 같이 갱신
- windowing 로직 변경 시 golden count 테스트 갱신
- threshold 기본값을 바꾸면 문서와 config sample 동기화
- 모델 입출력 스키마 변경 시 backward compatibility 점검

---

## 7. Codex에게 요구할 검증 루틴
새 기능을 구현할 때 Codex는 아래를 같이 수행한다.
- 필요한 테스트 파일 추가 또는 수정
- 최소 1개의 실행 가능한 검증 명령 제시
- 결과 해석 요약
- 남은 수동 검증 항목 명시

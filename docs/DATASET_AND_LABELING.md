# DATASET_AND_LABELING.md

## 1. 목적
이 문서는 WiFi CSI 기반 낙상 감지 MVP를 위한 데이터 수집, 라벨링, split, 실험 관리 원칙을 정의한다.

이 프로젝트의 성패는 모델 구조보다 **데이터 품질** 에 더 크게 좌우된다.

---

## 2. 기본 원칙
- 반드시 **실제 설치 대상 방** 에서 데이터를 수집한다.
- `fall` 과 `lie_down_intentional` 을 명확히 분리한다.
- 참가자와 세션 메타데이터를 남긴다.
- 학습/평가 분리는 **사람 기준 split** 을 우선한다.
- 세션 재현 가능성을 위해 raw와 processed를 모두 보존한다.

---

## 3. 권장 클래스
학습용 세부 클래스:
- `empty`
- `normal_walk`
- `sit_or_stand_transition`
- `lie_down_intentional`
- `fall`

운영용 최종 출력:
- `non_fall`
- `fall`

### 이유
- `empty` 는 배경 노이즈를 분리하는 데 필요하다.
- `normal_walk` 는 가장 흔한 오탐 후보다.
- `sit_or_stand_transition` 은 급격한 자세 변화와 혼동될 수 있다.
- `lie_down_intentional` 은 실제 낙상과 가장 구분이 어려운 음성 클래스다.

---

## 4. 최소 수집 목표

### 초기 목표
- 참가자 수: **5~8명**
- 클래스당 반복: **30~50회**
- fall은 방향별 수집: 앞/뒤/좌/우
- 시간대: 낮/밤 최소 2회
- 의도적 눕기: 침대/바닥 각각 포함 가능하면 포함

### 최소 기준 예시
- empty: 20 세션
- normal_walk: 50 세션
- sit_or_stand_transition: 40 세션
- lie_down_intentional: 40 세션
- fall: 40 세션

---

## 5. 세션 단위 메타데이터
각 세션마다 아래 메타를 저장한다.

```json
{
  "session_id": "session_2026_04_07_001",
  "participant_id": "P03",
  "room_id": "bedroom_a",
  "layout_version": "layout_v1",
  "node_setup_version": "setup_triangle_v1",
  "activity_label": "fall",
  "fall_direction": "left",
  "notes": "started near bed edge",
  "recorded_at": "2026-04-07T21:10:00+09:00"
}
```

### 필수 필드
- session_id
- participant_id
- activity_label
- recorded_at
- room_id
- layout_version
- node_setup_version

---

## 6. 라벨링 규칙

### 6.1 이벤트 기준
- `fall`: 의도하지 않은 급격한 넘어짐으로, 충격성 변화와 이후 저움직임이 동반되는 동작
- `lie_down_intentional`: 사용자가 스스로 눕는 동작
- `sit_or_stand_transition`: 앉기/일어나기 또는 자세 전환
- `normal_walk`: 방 안 보행
- `empty`: 사람이 없거나 거의 없는 상태

### 6.2 경계 규칙
- 넘어질 듯하다가 회복하면 `fall` 이 아님
- 빠르게 바닥에 눕더라도 의도된 눕기면 `lie_down_intentional`
- 침대에 털썩 앉는 동작은 우선 `sit_or_stand_transition` 으로 처리

### 6.3 윈도우 라벨 규칙
- 중심 시점 기준 라벨링
- fall이 포함된 윈도우는 positive 가능
- 애매한 구간은 `ambiguous` 또는 제외 처리

---

## 7. 데이터 수집 프로토콜

## 7.1 세션 공통 절차
1. 장비 및 노드 연결 확인
2. 세션 메타데이터 생성
3. recording 시작
4. 동작 수행
5. recording 종료
6. 메모/특이사항 입력
7. 필요 시 비디오 타임스탬프 동기화 기록

## 7.2 fall 세션 프로토콜
- 안전 매트 준비
- 보조자 1명 이상 배치
- 낙상 동작은 안전 범위에서만 수행
- 과장된 액션보다 재현 가능한 자연스러운 동작 우선
- 같은 방향 낙상도 시작 위치를 조금씩 바꿔 수집

## 7.3 normal 생활 세션 프로토콜
- 천천히 걷기
- 빠르게 걷기
- 침대 주변 이동
- 앉기/일어나기
- 이불 정리 또는 소형 물체 줍기 등을 추가할 수 있음

---

## 8. split 전략

### 추천 1순위
**participant-wise split**
- train: 참가자 60~70%
- val: 10~20%
- test: 20~30%

### 이유
동일 참가자가 train/test에 동시에 들어가면 성능이 부풀려질 수 있다.

### 추가 평가
- session-wise split
- layout change test
- leave-one-person-out

---

## 9. 평가 리포트에 반드시 포함할 것
- confusion matrix
- fall precision / recall / F1
- non_fall false positive 사례
- lie_down_intentional 과의 혼동 분석
- 참가자별 성능 차이
- 방 배치가 바뀐 경우 성능 변화

---

## 10. 저장 포맷 권장

### raw
- `.jsonl` 또는 `.parquet`

### processed
- window features: `.parquet`
- numpy tensor: `.npy` 또는 `.npz`
- split manifest: `.json`

### reports
- metrics summary: `.json`
- human-readable report: `.md`

---

## 11. 권장 디렉터리
```text
artifacts/
  raw/
    session_*/
      packets.jsonl
      metadata.json
  processed/
    feature_table.parquet
    tensor_dataset.npz
    splits/
      split_v1.json
  reports/
    baseline_rf.md
    baseline_cnn.md
```

---

## 12. 자주 생기는 실수
- fall 데이터만 너무 많이 모으고 normal 데이터를 적게 모음
- intentional lying 라벨이 불명확함
- 노드 배치를 바꿔놓고 layout version을 남기지 않음
- 사람이 없는 구간을 수집하지 않음
- train/test가 세션만 다르고 참가자가 같음

---

## 13. Codex 구현 시 우선사항
Codex는 데이터셋 관련 기능을 만들 때 아래를 우선 구현한다.
1. 세션 메타데이터 스키마
2. raw session 저장기
3. processed feature export
4. split manifest 생성기
5. metrics report 생성기

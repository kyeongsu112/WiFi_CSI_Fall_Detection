# REALTIME_DASHBOARD_AND_ALERTS.md

## 1. 목적
이 문서는 WiFall 오프라인 세션을 실시간처럼 재생하면서 낙상 이벤트를 보여주는 웹 대시보드 설계를 정의한다.

---

## 2. 실시간의 의미
초기 버전의 실시간은 **live ESP32 stream** 이 아니라 **dataset replay** 다.
즉, 테스트 세션을 일정 배속으로 흘려보내며 online inference와 동일한 인터페이스를 검증한다.

---

## 3. 상태 정의
- `idle`: 평상시
- `candidate`: 낙상 가능성이 높아진 상태
- `confirmed`: 저움직임 확인 후 낙상 확정
- `cooldown`: 중복 알림 방지 시간

---

## 4. 이벤트 규칙

### 4.1 candidate 발생
다음 조건 중 기본 버전:
- `fall_probability >= candidate_threshold`
- 또는 최근 N개 window 평균이 threshold 이상

### 4.2 confirmed 발생
candidate 이후 아래가 성립해야 한다.
- `low_motion_score <= motion_threshold`
- 상태가 `inactivity_seconds` 이상 유지

### 4.3 cooldown
confirmed 이후 `cooldown_seconds` 동안 추가 confirmed를 막는다.

---

## 5. 대시보드 최소 구성

### 상단 카드
- 현재 상태
- 현재 세션
- 현재 fall score
- 최근 confirmed 시간

### 차트
- 시간축 fall probability line
- low-motion score line
- state timeline band

### 이벤트 패널
- candidate_fall
- confirmed_fall
- cooldown_started
- cooldown_ended

---

## 6. API/내부 인터페이스
권장 내부 구조:
- `ReplayStreamer`: 다음 window 제공
- `Predictor`: window -> probability
- `ConfirmationEngine`: probability + motion -> state
- `EventBus`: 이벤트 발행
- `DashboardStore`: 최신 상태 보관

---

## 7. 구현 순서
1. CLI replay
2. confirmation engine
3. event log
4. FastAPI websocket 또는 polling endpoint
5. 간단한 HTML/Jinja 대시보드

---

## 8. 데모 성공 기준
- replay 시작/정지 가능
- fall 세션에서 candidate와 confirmed가 보임
- normal 세션에서 confirmed가 거의 나오지 않음
- 설정값 변경이 UI 또는 config에 반영됨

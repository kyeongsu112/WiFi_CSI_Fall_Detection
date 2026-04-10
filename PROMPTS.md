# PROMPTS.md

이 문서는 Codex에 바로 넣어 사용할 수 있는 프롬프트 모음이다.
프롬프트는 가능한 한 **Goal / Context / Constraints / Done when** 구조를 유지한다.

---

# 1. 전체 프로젝트 부트스트랩 프롬프트

```text
너는 이 저장소의 첫 구현을 맡은 시니어 AI 소프트웨어 엔지니어다.

Goal:
ESP32-S3 3대를 사용한 WiFi CSI 기반 낙상 감지 MVP 저장소를 초기 구축해라. 목표는 실내 1개 방에서 낙상만 감지하는 것이다. 포즈 추정은 하지 않는다.

Context:
- 저장소에는 PRD.md, AGENTS.md, docs/SYSTEM_ARCHITECTURE.md, docs/DATASET_AND_LABELING.md, docs/TEST_PLAN.md가 있다.
- 이 프로젝트는 collector -> preprocessing -> training -> inference -> app 구조를 따른다.
- 낙상 판정은 반드시 2단계여야 한다: candidate fall 검출 후 5~10초 저움직임 확인 뒤 confirmed fall.
- baseline은 복잡한 모델보다 재현 가능한 구조를 우선한다.
- 처음부터 실시간 운영 구조와 오프라인 학습 구조를 분리해라.

Constraints:
- 포즈 추정, skeleton, keypoint 기능은 추가하지 마라.
- 하드코딩된 로컬 경로를 넣지 마라.
- 수집, 전처리, 학습, 추론 코드를 한 파일에 섞지 마라.
- 테스트 가능하고 리플레이 가능한 구조를 만들어라.
- 설정은 config 파일로 분리해라.
- 작은 단계로 작업하되, 먼저 계획을 제시하고 그 다음 구현하라.

What to produce:
1. 저장소 기본 디렉터리 구조
2. pyproject.toml 또는 requirements 정리
3. 최소 실행 가능한 collector, preprocessing, training, inference, app 스캐폴드
4. 샘플 config 파일
5. 최소 단위 테스트 3개 이상
6. README 초안

Done when:
- 프로젝트가 설치 가능하다.
- 기본 CLI 엔트리포인트가 있다.
- 테스트가 최소 일부 실행 가능하다.
- 다음 구현 단계가 명확히 이어질 수 있다.

작업 방식:
1. 먼저 현재 파일을 읽고 구현 계획을 요약해라.
2. 그 다음 단계별로 구현해라.
3. 마지막에 변경 사항, 검증 방법, 남은 TODO를 요약해라.
```

---

# 2. collector 구현 프롬프트

```text
Goal:
ESP32 CSI 데이터를 수신하고 세션 단위로 저장하는 collector 레이어를 구현해라.

Context:
- 내부 표준 패킷 스키마는 timestamp, node_id, session_id, seq, rssi, channel, amplitude, phase, raw_payload 를 포함해야 한다.
- receiver, packet_parser, session_store, replay 모듈이 필요하다.
- 이후 preprocessing 과 inference 가 raw session을 재사용해야 한다.

Constraints:
- 네트워크 포맷이 바뀌어도 내부 표준 스키마는 유지해라.
- parser는 malformed packet에 대해 안전하게 실패해야 한다.
- session 저장 포맷은 jsonl 또는 parquet 중 구현이 단순한 방식부터 선택해라.
- replay 기능을 같이 만들어라.

Done when:
- 샘플 패킷을 받아 표준 객체로 변환할 수 있다.
- 세션 시작/종료가 가능하다.
- 저장된 세션을 재생할 수 있다.
- 관련 테스트가 있다.

추가 요구:
- 먼저 설계 요약과 파일 생성 계획을 보여준 뒤 구현해라.
- 구현 후 실행 예시 명령도 제시해라.
```

---

# 3. preprocessing 구현 프롬프트

```text
Goal:
CSI 전처리 파이프라인을 구현해라. amplitude/phase 처리, phase unwrap, smoothing, windowing, feature extraction이 포함되어야 한다.

Context:
- 입력은 collector가 저장한 raw session이다.
- 출력은 학습과 실시간 추론에서 공통으로 사용할 수 있는 window 및 feature 표현이다.
- 초기 feature는 해석 가능한 형태를 우선한다.

Constraints:
- batch와 online이 공유 가능한 순수 함수 중심으로 설계해라.
- config-driven으로 파라미터를 제어하게 해라.
- fancy한 DSP보다 먼저 안정적으로 재현 가능한 baseline을 만들어라.
- feature shape 안정성을 검증하는 테스트를 넣어라.

추천 baseline feature:
- amplitude mean/std
- phase variance
- motion energy
- node 간 RSSI 통계
- short-term change score

Done when:
- raw session -> feature windows 변환이 된다.
- window_seconds 와 stride_seconds 가 config로 조절된다.
- 테스트와 샘플 실행 스크립트가 있다.
```

---

# 4. baseline model 구현 프롬프트

```text
Goal:
낙상 감지를 위한 baseline 학습 및 평가 파이프라인을 구현해라.

Context:
- 학습용 클래스는 empty, normal_walk, sit_or_stand_transition, lie_down_intentional, fall 이다.
- 운영용 출력은 non_fall / fall 로 매핑 가능해야 한다.
- baseline은 feature 기반 모델부터 시작한다.

Constraints:
- 먼저 RandomForest 또는 XGBoost 같은 간단한 baseline을 구현해라.
- 데이터 split은 participant-wise split을 우선 고려해라.
- metrics report와 confusion matrix 생성 코드를 포함해라.
- 모델 아티팩트와 리포트를 artifacts/ 아래에 저장해라.

Done when:
- train.py 로 학습 가능하다.
- evaluate.py 로 성능 요약을 만든다.
- 추론용으로 로드 가능한 모델 파일이 생성된다.
- 리포트가 markdown 또는 json 형태로 남는다.
```

---

# 5. 실시간 추론 + 2단계 판정 프롬프트

```text
Goal:
실시간 CSI 입력을 받아 낙상 후보와 최종 confirmed fall을 판정하는 inference 레이어를 구현해라.

Context:
- predictor는 모델의 fall_prob를 계산한다.
- confirmation engine은 candidate -> confirmed 상태 전이를 관리한다.
- confirmed fall은 모델 점수만으로 확정하지 않고, 후속 저움직임 구간 확인이 필요하다.

Constraints:
- 상태 머신을 명시적으로 구현해라.
- idle / candidate / confirmed / cooldown 상태를 분리해라.
- threshold와 cooldown은 config로 분리해라.
- 규칙 로직을 테스트 가능한 pure function에 가깝게 작성해라.

Done when:
- 실시간 또는 replay 입력으로 상태 전이가 동작한다.
- candidate와 confirmed가 별도 로그로 남는다.
- confirmation state machine 테스트가 존재한다.
```

---

# 6. CLI / 데모 앱 프롬프트

```text
Goal:
수집, 전처리, 학습, 추론을 실행할 수 있는 CLI를 만들어라.

Context:
- 사용자는 터미널에서 collect, preprocess, train, infer, replay 명령을 실행하고 싶다.
- 웹 UI는 필수가 아니지만, 상태 확인이 쉬운 출력은 필요하다.

Constraints:
- Typer 또는 argparse 중 단순한 쪽을 선택해라.
- 명령별 config 파일 경로를 받을 수 있게 해라.
- long-running process 는 graceful shutdown을 고려해라.
- CLI 도움말을 친절하게 제공해라.

Done when:
- `collect`, `preprocess`, `train`, `infer`, `replay` 명령이 동작한다.
- README에 예시 명령이 정리된다.
```

---

# 7. 코드 리뷰 프롬프트

```text
이 저장소의 변경 사항을 리뷰해라.

리뷰 기준:
- PRD 범위를 벗어나는 기능이 들어갔는가?
- collector / preprocessing / training / inference 계층이 섞이지 않았는가?
- threshold, cooldown, inactivity_seconds 가 하드코딩되지 않았는가?
- candidate 와 confirmed 판정이 분리되어 있는가?
- 테스트가 충분한가?
- 향후 디버깅 가능한 로그와 아티팩트가 남는가?

출력 형식:
1. 치명적 이슈
2. 수정 권장 사항
3. 지금 머지 가능한 부분
4. 빠진 테스트
5. 다음 단계 제안
```

---

# 8. 디버깅 프롬프트

```text
현재 시스템에서 normal walk 또는 intentional lying 중에도 fall 경보가 과도하게 발생한다.

다음 관점으로 원인을 분석하고 수정해라.
- candidate threshold 가 너무 낮은가?
- inactivity 조건이 너무 약한가?
- feature가 fall 과 lie_down 을 잘 구분하지 못하는가?
- 특정 node의 노이즈가 전체 판단을 오염시키는가?
- cooldown 이 너무 짧은가?

요구사항:
- 먼저 가능한 원인 가설을 우선순위로 정리해라.
- 그 다음 로그/테스트/코드 기준으로 검증 계획을 제시해라.
- 마지막으로 가장 작은 수정부터 적용해라.
- 수정 후 어떤 오탐이 줄어들지 설명해라.
```

---

# 9. 문서 동기화 프롬프트

```text
코드와 문서가 어긋나지 않도록 아래 작업을 수행해라.
- 현재 config 기본값과 README 예시를 대조해라.
- AGENTS.md, PRD.md, docs/*.md 와 구현 코드의 차이를 찾아라.
- 차이가 있으면 코드 또는 문서 중 어느 쪽을 맞춰야 하는지 판단해라.
- 마지막에 불일치 목록과 수정 결과를 요약해라.
```

---

# 10. 추천 운영 방식

1. 먼저 Codex에 전체 부트스트랩 프롬프트를 넣는다.
2. 이후 collector -> preprocessing -> training -> inference -> CLI 순서로 쪼개서 진행한다.
3. 각 단계가 끝날 때마다 코드 리뷰 프롬프트를 돌린다.
4. 오탐이 많아지면 디버깅 프롬프트를 사용한다.
5. 문서와 코드가 틀어지면 문서 동기화 프롬프트를 사용한다.

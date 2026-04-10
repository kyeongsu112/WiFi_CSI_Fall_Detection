# IMPLEMENTATION_PLAN.md

## Phase 1 — Data foundation
- WiFall.zip 구조 파악
- manifest 생성기
- subject/action 통계 리포트
- binary label mapping

## Phase 2 — Baseline model
- 1초 window loader
- baseline feature 또는 tensor pipeline
- binary classifier
- metrics/report 저장

## Phase 3 — Replay inference
- 세션 replay 엔진
- predictor wrapper
- candidate/confirmed/cooldown 상태머신
- 이벤트 로그 저장

## Phase 4 — Web dashboard
- FastAPI app
- 상태 polling 또는 websocket
- timeline / state cards

## Phase 5 — KNN-MMD adaptation
- source/support/test split
- UMAP + KNN help set
- local/global MMD
- support-based early stopping

## Phase 6 — Hardening
- config 정리
- 테스트 확대
- demo script
- README 보강

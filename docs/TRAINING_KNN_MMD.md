# TRAINING_KNN_MMD.md

## 1. 목적
이 문서는 KNN-MMD 논문 구조를 이 저장소에서 어떻게 구현할지 정리한다.

---

## 2. 논문에서 가져올 핵심 아이디어
- cross-domain few-shot 설정
- target support set으로 testing set에 대한 pseudo-label 생성
- confidence가 높은 target sample들로 help set 생성
- source train set과 help set을 **class-wise local alignment**
- support set은 training loss에 직접 넣지 않고 early stopping validation에 활용

즉, 이 저장소의 KNN-MMD 구현은 단순 분류기가 아니라 **source/target adaptation 실험 프레임워크** 다.

---

## 3. 권장 구현 단계

### 단계 1 — baseline
- binary fall detector부터 구현
- 입력: 1초 window tensor 또는 hand-crafted features
- 출력: fall probability

### 단계 2 — few-shot split
- target subject에서 클래스당 n개 샘플을 support set으로 선택
- 나머지는 test set

### 단계 3 — preliminary classification
- target support를 이용해 test 샘플을 UMAP + KNN으로 pseudo-labeling
- confidence 상위 p%를 help set으로 선택

### 단계 4 — local alignment training
loss:
```text
L = L_cls + alpha_local * L_local_MMD + alpha_global * L_global_MMD
```

설명:
- `L_cls`: source train cross-entropy
- `L_local_MMD`: 클래스별 source/help 임베딩 정렬
- `L_global_MMD`: 전체 source/help 임베딩 정렬 보조항

### 단계 5 — early stopping
- validation은 support set으로 수행
- support accuracy/loss 기준으로 best checkpoint 선택

---

## 4. config 항목
권장 config:
- `shot_n`
- `umap_dim`
- `knn_k`
- `help_set_ratio_p`
- `alpha_local`
- `alpha_global`
- `kernel_type`
- `kernel_sigmas`
- `min_epochs`
- `max_epochs`
- `patience`

---

## 5. 산출물
- best checkpoint
- training curve
- support validation curve
- target test metrics
- confusion matrix
- run config snapshot

---

## 6. 이 저장소의 현실적 목표
논문 구조를 최대한 따르되, 처음에는 **완전한 SOTA 복제보다 실행 가능한 버전** 을 우선한다.

구체적으로는:
1. UMAP + KNN help set 생성
2. source/help local MMD loss 연결
3. support-based early stopping
4. target test 평가

여기까지 되면 문서 목적은 달성이다.

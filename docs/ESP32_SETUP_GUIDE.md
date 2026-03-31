# ESP32-S3 CSI 데이터 수집 환경 구축 가이드

## 전체 구성도

```
[Wi-Fi AP (공유기)]  ←── Wi-Fi 신호 ──→  [ESP32-S3 (수신)]
                                              │ USB (시리얼)
                                              ↓
                                        [PC / 노트북]
                                         Python 스크립트
```

> 사람이 AP와 ESP32 사이를 지나가면 CSI 값이 변하고, 이를 분석하여 행동을 분류합니다.

---

## 1단계: ESP-IDF 개발 환경 설치

ESP32-S3의 CSI 기능을 사용하려면 **ESP-IDF** (Espressif IoT Development Framework)가 필요합니다.

### Windows 설치

1. **ESP-IDF 설치 프로그램** 다운로드:
   - https://dl.espressif.com/dl/esp-idf/ 에서 **ESP-IDF Tools Installer** 다운로드
   - `esp-idf-tools-setup-offline-x.x.exe` 실행
   - 설치 시 **ESP-IDF v5.1 이상** 선택
   - 설치 경로 예: `C:\Espressif`

2. 설치가 끝나면 시작 메뉴에서 **"ESP-IDF 5.x CMD"** 또는 **"ESP-IDF 5.x PowerShell"** 실행

3. 설치 확인:
   ```bash
   idf.py --version
   ```

---

## 2단계: ESP32-CSI 펌웨어 준비

Espressif 공식 CSI 수집 프로젝트를 사용합니다.

### 다운로드

```bash
# ESP-IDF 터미널에서 실행
cd C:\Espressif
git clone https://github.com/espressif/esp-csi.git
cd esp-csi
git submodule update --init --recursive
```

### CSI 수집용 예제 선택

```bash
cd examples/get-started/csi_recv
```

### Wi-Fi 설정 (menuconfig)

```bash
idf.py set-target esp32s3
idf.py menuconfig
```

menuconfig에서 아래 항목을 설정합니다:

```
→ Example Configuration
   → WiFi SSID: [공유기 SSID 입력]
   → WiFi Password: [공유기 비밀번호 입력]

→ Component config → Wi-Fi
   → WiFi CSI (Channel State Information): Enable
```

> **중요**: ESP32-S3는 **2.4GHz Wi-Fi만** 지원합니다. 공유기에서 2.4GHz 대역을 활성화하세요.

---

## 3단계: 펌웨어 빌드 및 플래싱

### ESP32-S3를 USB로 PC에 연결

- USB-C 케이블로 ESP32-S3를 PC에 연결
- 장치 관리자에서 **COM 포트 번호** 확인 (예: COM3)

### 빌드

```bash
idf.py build
```

### 플래싱 (업로드)

```bash
idf.py -p COM3 flash
```

> `COM3`을 실제 포트 번호로 변경하세요.

### 시리얼 모니터로 확인

```bash
idf.py -p COM3 monitor
```

정상적으로 동작하면 아래와 같은 CSI 데이터가 출력됩니다:

```
CSI_DATA,aa:bb:cc:dd:ee:ff,-45,11,1,0,0,1,0,0,0,0,0,-90,0,6,0,123456,0,128,0,128,0,[4,2,-3,5,8,-1,...]
CSI_DATA,aa:bb:cc:dd:ee:ff,-43,11,1,0,0,1,0,0,0,0,0,-90,0,6,0,123457,0,128,0,128,0,[3,1,-2,6,7,-2,...]
```

**Ctrl+]** 로 모니터 종료

---

## 4단계: config.py 포트 설정

프로젝트의 `config.py`에서 ESP32 시리얼 포트를 설정합니다:

```python
# config.py (23번째 줄 부근)
ESP32_SERIAL_PORT = "COM3"   # ← 실제 포트 번호로 변경
ESP32_BAUD_RATE = 921600     # ESP-IDF 기본 baud rate
```

---

## 5단계: 데이터 수집 실행

### 시리얼 포트 확인

```bash
python data_collection/esp32_receiver.py --list-ports
```

출력 예:
```
사용 가능한 시리얼 포트:
  COM3: Silicon Labs CP210x USB to UART Bridge (COM3)
```

### 행동별 데이터 수집

각 행동을 **30초씩** 수집합니다. 수집 중에는 해당 행동을 반복 수행하세요.

```bash
# 1) Walking - ESP32와 공유기 사이를 왔다 갔다
python data_collection/data_logger.py --label walking --duration 30 --port COM3

# 2) Sitting - ESP32와 공유기 사이에서 의자에 앉아있기
python data_collection/data_logger.py --label sitting --duration 30 --port COM3

# 3) Lying - 바닥에 누워있기
python data_collection/data_logger.py --label lying --duration 30 --port COM3

# 4) Falling - 넘어지는 동작 반복 (주의해서!)
python data_collection/data_logger.py --label falling --duration 30 --port COM3
```

> **팁**: falling 데이터를 수집할 때는 쿠션 등을 깔고 안전하게 진행하세요. 30초 동안 여러 번 낙상 동작을 반복합니다.

### 수집된 데이터 확인

```bash
# data/raw/ 디렉토리에 행동별 CSV 파일 생성됨
ls data/raw/walking/
ls data/raw/sitting/
ls data/raw/lying/
ls data/raw/falling/
```

---

## 6단계: 실제 데이터로 모델 학습

시뮬레이션 데이터 대신 실제 수집 데이터를 사용하여 학습합니다:

```bash
# 데이터 생성 건너뛰고 기존 data/raw/ 데이터로 학습
python -X utf8 train.py --model all --skip-generate
```

---

## 7단계: 실시간 감지 실행

학습이 끝나면 실시간으로 행동을 감지할 수 있습니다:

```bash
# ESP32 연결 상태에서 실시간 감지
python -X utf8 realtime/detector.py --model lstm --port COM3

# 대시보드와 함께 실행
python -X utf8 dashboard/app.py
# → http://localhost:5000 에서 모니터링
```

---

## 물리적 배치 팁

```
최적의 배치:

     [공유기/AP]
         │
    2~5m 거리
         │
     활동 영역 ← 사람이 여기서 행동
         │
    1~3m 거리
         │
     [ESP32-S3] ──USB──→ [PC]
```

- AP와 ESP32-S3 사이 거리: **3~8m** 권장
- 사람은 **AP와 ESP32 사이**에서 활동
- 장애물이 적은 공간이 좋음
- ESP32 안테나가 AP 방향을 향하도록 배치

---

## 문제 해결

| 증상 | 해결 방법 |
|------|-----------|
| COM 포트를 못 찾음 | USB 드라이버 설치 (CP210x 또는 CH340) |
| CSI_DATA가 안 나옴 | menuconfig에서 CSI Enable 확인, Wi-Fi 연결 확인 |
| 데이터가 끊김 | baud rate 확인 (921600), USB 케이블 교체 |
| 빌드 오류 | ESP-IDF 버전 5.1 이상 확인, `idf.py set-target esp32s3` 재실행 |
| 정확도가 낮음 | 수집 시간 늘리기, 환경 일관성 유지, 더 많은 샘플 수집 |

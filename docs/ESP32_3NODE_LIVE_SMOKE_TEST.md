# ESP32 3-Node Live Smoke Test

## Goal

3대의 ESP32-S3 노드가 ADR-018 binary UDP를 collector로 보내고, collector가 이를 수신/저장/health 추적하는지 빠르게 확인한다.

범위는 **실시간 collector baseline smoke test**만이다. 전처리, 학습, 추론, 대시보드 검증은 이번 체크리스트 범위가 아니다.

## Current Supported Binary Format

현재 collector가 지원하는 binary live format은 아래 하나뿐이다.

- `live_udp_format: esp32_adr018`
- packet magic: `0xC5110001`
- 20-byte ADR-018 header + I/Q payload
- header fields:
  - `magic` (`u32`, LE)
  - `node_id` (`u8`)
  - `antenna_count` (`u8`)
  - `subcarrier_count` (`u16`, LE)
  - `frequency_mhz` (`u32`, LE)
  - `seq` (`u32`, LE)
  - `rssi` (`i8`)
  - `noise_floor` (`i8`)
  - reserved (`u16`)
- payload size must be exactly `antenna_count * subcarrier_count * 2`

지원하지 않는 binary packet type은 loud failure 한다.

- vitals packet (`0xC5110002`)
- WASM output (`0xC5110004`)
- truncated payload
- unknown magic
- undocumented binary layout

## Explicit Assumptions

- 실기기 firmware는 ADR-018 CSI frame을 UDP로 전송한다.
- binary frame 안에는 wall-clock timestamp가 없으므로, collector는 **수신 시각**을 `timestamp`로 기록한다.
- binary frame 안의 `node_id`는 collector에서 문자열로 정규화된다.
  - 예: `1` -> `"1"`
- smoke test에서는 `expected_nodes`를 `["1", "2", "3"]`로 맞춘다.
- raw I/Q는 이번 baseline에서 amplitude/phase로 해석하지 않고 `raw_payload.csi_raw.iq_bytes_hex`로 보존한다.

## Config Setup

실기기 smoke test 전에는 sample config를 active config 위에 복사한다.

필수 sample files:

- [collection.live_esp32.example.yaml](D:\미프\wifi-fall-detection\configs\collection.live_esp32.example.yaml)
- [inference.live_esp32.example.yaml](D:\미프\wifi-fall-detection\configs\inference.live_esp32.example.yaml)

PowerShell:

```powershell
Copy-Item configs\collection.live_esp32.example.yaml configs\collection.yaml -Force
Copy-Item configs\inference.live_esp32.example.yaml configs\inference.yaml -Force
```

## Pre-flight

- ESP32 3대가 모두 같은 collector IP와 UDP port `5005`로 전송하도록 provision되어 있어야 한다.
- collector PC와 ESP32가 같은 네트워크 또는 같은 핫스팟에 있어야 한다.
- Windows 방화벽이 UDP `5005`를 막지 않는지 확인한다.
- `configs\collection.yaml`의 `expected_nodes`가 `"1"`, `"2"`, `"3"`인지 확인한다.
- 첫 smoke test는 `live_session_duration_seconds: 30` 그대로 두는 것을 권장한다.

## Start Collector

metadata fixture를 재사용하고, 실행마다 `--session-id`만 새로 준다.

```powershell
.\.venv\Scripts\python.exe scripts\collect.py --config-dir configs --metadata-path tests\fixtures\mock_session_metadata.json --session-id session_2026_04_08_live_smoke_001
```

## Expected Collector Logs

시작 직후:

```text
live event=start host=0.0.0.0 port=5005 duration_seconds=30
```

처음 한두 노드만 보일 때:

```text
live event=warning status=degraded active=1 stale=none missing=2,3
```

3대가 모두 들어오면:

```text
live event=health status=live active=1,2,3 stale=none missing=none
```

한 노드가 끊기면:

```text
live event=warning status=degraded active=1,3 stale=2 missing=none
```

bounded session 종료:

```text
live event=stop reason=duration duration_seconds=30
```

수동 종료:

```text
live event=stop reason=interrupted signal=SIGINT
```

malformed 또는 unsupported binary packet:

```text
live event=warning message=Unsupported ESP32 binary packet magic: ...
```

또는

```text
live event=warning message=ESP32 ADR-018 datagram payload length mismatch: ...
```

## Operator Steps

### 1. Single node sanity

- node `1`만 켠다.
- collector에 degraded warning이 뜨는지 본다.
- `active=1`, `missing=2,3`가 보이면 정상이다.

### 2. 3-node convergence

- node `2`, `3`을 차례로 켠다.
- `status=live active=1,2,3 stale=none missing=none`이 보이면 정상이다.

### 3. Dropout check

- collector가 live 상태일 때 node 하나의 전원을 끈다.
- `health_timeout_seconds` 이후 degraded warning이 뜨는지 본다.
- 남은 노드는 active, 끊긴 노드는 stale로 보여야 한다.

### 4. Raw session persistence

collector 종료 후 stdout 마지막 요약에서 `output_path=...`를 확인한다.

예:

```text
session_id=session_2026_04_08_live_smoke_001
packet_count=123
node_ids=1,2,3
output_path=D:\미프\wifi-fall-detection\artifacts\raw\session_2026_04_08_live_smoke_001
```

아래 helper로 저장 결과를 요약한다.

```powershell
.\.venv\Scripts\python.exe scripts\summarize_raw_session.py --session-dir artifacts\raw\session_2026_04_08_live_smoke_001 --expected-nodes 1,2,3
```

예상 출력:

```text
session_dir=D:\미프\wifi-fall-detection\artifacts\raw\session_2026_04_08_live_smoke_001
packet_count=123
seen_nodes=1,2,3
missing_expected_nodes=none
first_timestamp=...
last_timestamp=...
node_packet_counts=1:40,2:41,3:42
```

## Pass Criteria

- 3대 중 각 노드가 적어도 한 번 이상 보인다.
- 최종적으로 `status=live`에 도달한다.
- temporary dropout이 process crash 없이 degraded warning으로만 드러난다.
- `artifacts/raw/<session_id>/metadata.json`
- `artifacts/raw/<session_id>/packets.jsonl`
  가 생성된다.
- `summarize_raw_session.py` 출력에서 `missing_expected_nodes=none`이 확인된다.

## Troubleshooting

### No packets arriving

- ESP32 firmware의 target IP가 collector PC IP와 같은지 다시 확인한다.
- collector가 `udp_host: 0.0.0.0`, `udp_port: 5005`로 떠 있는지 확인한다.
- 방화벽이 UDP 5005를 막지 않는지 확인한다.
- collector 시작 로그가 바로 socket bind error 없이 떴는지 확인한다.

### Only one or two nodes visible

- provision된 `node_id`가 실제로 서로 다른지 확인한다.
- `expected_nodes`가 `"1"`, `"2"`, `"3"`인지 확인한다.
- 일부 노드만 다른 target IP/port를 쓰고 있지 않은지 확인한다.
- Wi-Fi 연결은 됐지만 UDP 송신이 끊긴 경우 serial log에서 `CSI streaming active` 이후 오류를 확인한다.

### Malformed or unsupported packets

- 현재 collector는 `0xC5110001` ADR-018 CSI frame만 받는다.
- vitals packet이나 다른 magic이면 warning이 나고 skip된다.
- payload 길이 mismatch가 반복되면 firmware binary format과 collector 지원 format이 어긋난 것이다.

### Degraded mode staying active

- 아직 한 번도 안 보인 노드는 `missing`
- 보였다가 timeout 지난 노드는 `stale`
- 계속 degraded면 실제로 3노드가 모두 송신 중인지 serial log와 `node_id`를 다시 확인한다.
- `health_timeout_seconds`가 너무 짧으면 brief Wi-Fi jitter에도 warning이 잦아질 수 있다.

## Recommended First Command Sequence

```powershell
Copy-Item configs\collection.live_esp32.example.yaml configs\collection.yaml -Force
Copy-Item configs\inference.live_esp32.example.yaml configs\inference.yaml -Force
.\.venv\Scripts\python.exe scripts\collect.py --config-dir configs --metadata-path tests\fixtures\mock_session_metadata.json --session-id session_2026_04_08_live_smoke_001
.\.venv\Scripts\python.exe scripts\summarize_raw_session.py --session-dir artifacts\raw\session_2026_04_08_live_smoke_001 --expected-nodes 1,2,3
```

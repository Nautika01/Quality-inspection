# 아키텍처

## 데이터 흐름

```
이미지(합성/카메라/파일/HTTP)
  │
  ▼
vision.measure_geometry ── 부품/포켓 분할 → 서브픽셀 원 피팅 → Geometry
  │
  ▼
vision.detect_defects ──── 반경 프로파일 + 조명 평면 제거 → 잔차 → 히스테리시스 → 형상 분류 → [Defect]
  │
  ▼
inspector.decide ───────── 가드밴드 공차 판정 + 결함 등급(치명/중/경) → InspectionResult(PASS/NG, reasons)
  │
  ├─ anomaly.AnomalyModel.predict (선택) → inspector.apply_anomaly: 설명되지 않는 이상만 NG 추가
  │
  ▼
db.InspectionDB.add_inspection ─ Lot/Serial/측정/결함/버전/해시 저장, prev_hash → record_hash 체인
  │
  ├─ line.Station: SpcMonitor(X̄-R) → 알람 → bus.publish("qinspect/<station>/alarm")
  ├─ api: /inspect /lots/{id}/summary /serials/{sn} /audit/verify
  └─ dashboard: KPI, 파레토, 관리도, 알람, 분포
```

## 모듈 책임

| 모듈 | 입력 | 출력 | 외부 의존 |
|---|---|---|---|
| `spec` | JSON | `PartSpec` (공차, 가드밴드, 등급 기준) | – |
| `synth` | `PartSpec`, seed | PNG + `ground_truth.csv` | numpy, pillow |
| `vision` | uint8 이미지 | `Geometry`, `[Defect]` | scipy.ndimage |
| `inspector` | `VisionResult` | `InspectionResult` | – |
| `anomaly` | 양품 이미지 | `.npz` 모델, 점수/맵 | scipy.spatial.cKDTree |
| `spc` | 1D 측정열 | 관리한계, 위반, Cp/Cpk | – |
| `msa` | 반복 측정표 | `GageRR` | scipy.stats |
| `db` | `InspectionResult` | SQLite, 해시 체인 | sqlite3 |
| `line` | 카메라/시뮬레이터 | DB 레코드, 버스 메시지, 알람 | paho-mqtt(선택) |
| `api` | HTTP | JSON | fastapi |
| `dashboard` | SQLite | 웹 UI | streamlit |
| `sampling` | Lot 크기, AQL/LQ | `SamplingPlan`, OC 곡선 | scipy.stats |
| `specdoc` | `PartSpec`, Gage 결과 | Markdown 검사 기준서 | – |
| `datasets` | 사진 폴더, MVTec 폴더 | 캐노니컬 PNG, 데이터셋 객체 | pillow |

## 왜 이렇게 설계했나

### 전통 비전 + 메모리뱅크 이상탐지 (하이브리드)
- 치수는 규격 대비 판정이라 **설명 가능한 측정**이 필수 → 전통 비전.
- 외관은 "무엇이 불량인지" 미리 다 열거할 수 없음 → 양품만으로 학습하는 **one-class** 접근.
- 둘의 결합 규칙은 `inspector.apply_anomaly` 한 곳에만 있음: 이상 점수는 **알려진 결함을 재판정하지 않고**, 전통 비전이 아무것도 못 찾은 부품의 이상만 NG로 만든다.

### 메시지 버스 추상화
`InMemoryBus` / `MqttBus` 가 같은 `publish(topic, payload)` 인터페이스. 테스트는 메모리 버스로 메시지를 검증하고, 현장은 Mosquitto/EMQX 등 브로커로 PLC·MES·SCADA에 전달. OPC UA가 필요하면 같은 인터페이스로 `OpcUaBus` 추가.

### 감사 체인
```
record_hash[i] = SHA256( record_hash[i-1] || canonical_json(record[i]) )
record[0]의 prev = "000…0"
```
canonical_json 에는 판정·측정·결함·버전·규격 해시·모델 해시·스테이션이 포함된다. 체인 머리 해시를 주기적으로 외부(출력물, 타임스탬프 서버, WORM 저장소)에 기록하면 DB 전체 교체도 탐지 가능.

### 토픽 규약
```
qinspect/<station>/result    검사 1건 (InspectionResult.to_row + lot_id)
qinspect/<station>/alarm     critical_defect | spc_violation | yield_drop
qinspect/<station>/heartbeat
```

## 확장 포인트
- 새 치수 특성: `PartSpec`에 `DimensionSpec` 추가 → `vision.measure_geometry`에서 측정 → `inspector.decide`에 판정 한 줄.
- CNN 임베딩: `anomaly.patch_features` 교체. 모델 포맷·임계값·해시 체계는 그대로.
- 다중 스테이션/고속: `InspectionDB`를 PostgreSQL 구현으로 교체(인터페이스 동일), 대시보드는 DB만 바라봄.

# 아키텍처

## 데이터 흐름

```
이미지(합성/카메라/파일/HTTP)
  │  vision.to_gray8 : RGB·RGBA·16-bit·float → uint8 (비트 깊이 추정)
  ▼
vision.segment_part ── Otsu 분할(부품이 밝든 어둡든) → 최대 성분 → 경계 걸림·대비 검증 (VisionError 슬러그)
vision._subpixel_disk ─ 섹터별(16) 안/밖 기준레벨로 경계 픽셀 커버리지 가중 → Circle (조명 기울기 보정)
vision._find_pocket ── 면 내부 Otsu → 중심 근처 성분 → 면/포켓 대비·반지름 비·중심 거리 검증 (선택적)
  ▼
vision.detect_defects ── 반경 프로파일 + 조명 평면 제거 → 잔차 → 히스테리시스 → 극성(어두움/밝음)·형상 분류, 길이=주축 extent
  ▼
inspector.decide ───── 특성별 가드밴드 공차 + 결함 등급 → PASS / NG / ERROR, reasons(코드)
  ├─ inspector.apply_anomaly (선택): 기하 실패 프레임은 건너뜀, 설명되지 않는 이상만 NG, 모델 예외는 anomaly_error
  ▼
db.InspectionDB.add_inspection ─ BEGIN IMMEDIATE { ensure_lot(규격 바인딩) → 머리 해시 → INSERT } COMMIT
  ├─ line.Station: SpcMonitor×특성(동결 한계 or Phase I) → 알람(critical/spc/yield/vision_error) → bus.publish
  ├─ api: /inspect /lots/{id}/summary /serials/{sn} /inspections/{id}/disposition /lots/{id}/close /events/{id}/ack
  │       /spc/limits/freeze /reports/shift /audit/verify /audit/anchor
  └─ dashboard: Lot 규격 기준 KPI, 파레토(한글 사유), X̄·R(특성 선택), 알람 확인 폼, 미처분 NG, 분포
```

## 판정 의미

| 판정 | 뜻 | 수율/SPC | MES 동작 |
|---|---|---|---|
| PASS | 모든 특성 가드밴드 안, 불합격 결함 없음 | 포함 | 통과 |
| NG | 사유 코드 ≥1 (`reasons.py`) | 포함 | 배출, 처분 필요 |
| ERROR | 측정 불가 (`vision:*`) — 빈 지그, 경계 걸림, 저대비, 이미지 형식 | **제외** | 재촬영 |

## 저장소 스키마 (SQLite, WAL)

| 테이블 | 역할 |
|---|---|
| `lots` | Lot ↔ 품번·규격 해시·규격 JSON 바인딩, 상태(open/closed) |
| `inspections` | 시리얼별 판정·측정·등급·이상 점수·버전·해시·스테이션·작업자·영상 SHA-256·증거 경로·재검사 링크, `prev_hash`(UNIQUE)·`record_hash` |
| `defects` | 결함 종류·등급·길이·면적·bbox (해시에 포함) |
| `dispositions` | NG 처분: rework / scrap / use_as_is / mrb / reinspect, 처분자·시각·MRB 번호 |
| `lot_dispositions` | Lot 마감: 샘플링 계획·샘플·결과·판정자·당시 체인 머리 |
| `control_limits` | 특성별 동결 X̄-R 한계(활성 1개), 출처 Lot, 동결자 |
| `calibrations` | 교정 기록: px/mm, 게이지, 작업자, 영상 해시 |
| `events` | 알람·시스템 이벤트, 확인(ack) 필드 |
| `anchors` | 체인 머리 고정(레코드 수·해시·시각) |
| `meta` | `chain_genesis`(로테이션 시 이전 DB의 head), 이전 DB 경로·해시, 로테이션 기록 |
| `spec_approvals` | 규격 개정 승인(해시·이전 해시·승인자·규격 JSON) |

## 감사 체인

```
record_hash[i] = SHA256( record_hash[i-1] || canonical(record[i]) )
record[0].prev = "000…0"
```
- `canonical()`은 SQLite 저장 표현과 동일하게 정규화한다: 실수는 float, NaN/inf → null, -0.0 → 0.0, numpy 스칼라 → 파이썬 값. 그래서 재검증 시 메모리 값과 DB 값이 항상 같다.
- 쓰기는 `BEGIN IMMEDIATE` 트랜잭션 안에서 머리를 읽고 삽입한다. 여러 프로세스/스레드가 같은 파일에 써도 체인이 분기하지 않으며, 분기는 `UNIQUE(prev_hash)`로 저장 자체가 불가능하다.
- 중간 행 수정·삭제는 체인이 잡는다. **최신 행 삭제**는 체인만으로는 안 보이므로 `anchors`에 머리를 고정하고(`audit --anchor`, Lot 마감, 배치 종료, 라인 종료) `verify_chain()`이 마지막 앵커와 대조한다. 앵커 값을 외부(MES, 출력물, 타임스탬프 서버)에 복사해 두면 DB 전체 교체도 탐지된다.

## 동시성

- 한 `InspectionDB` 인스턴스는 `RLock`으로 모든 읽기·쓰기를 직렬화한다(FastAPI 스레드풀과 안전).
- 여러 인스턴스/프로세스는 WAL + `busy_timeout=30s`로 공존한다. 대시보드가 읽는 동안 스테이션 쓰기가 막히지 않는다.

## 하이브리드 판정 규칙 (한 곳: `inspector.apply_anomaly`)

1. 기하 측정 실패(ERROR) → 이상탐지 실행 안 함.
2. 모델 예외 → 판정 유지, `anomaly_error` 기록.
3. 점수 > 임계값 이고 전통 비전이 **아무 결함도** 찾지 못했고 PASS 였다면 → NG(`anomaly_score`).
4. 전통 비전이 (경결함이라도) 찾았다면 → 등급 판정이 우선.

## 메시지 버스

`InMemoryBus` / `MqttBus` / `OpcUaBus`가 같은 `publish(topic, payload) -> bool` 인터페이스. 연결 실패는 시작 시 `ConnectionError`로 즉시 드러나고, 실행 중 오류는 `errors` 카운터(라인은 멈추지 않음). OPC UA는 `Objects/<root>/<station>/<kind>`(JSON)와 `Serial/Verdict/Reasons/OdMm/PocketMm/ConcentricityMm` 태그에 씁니다. 태그는 PLC가 미리 갖고 있어야 하며(익명 클라이언트는 보통 노드 생성 불가), URL은 `opc.tcp://host:port/endpoint?root=<object>&ns=<namespace-uri>`.

```
qinspect/<station>/result    검사 1건 (InspectionResult.to_row + lot_id)
qinspect/<station>/alarm     critical_defect | spc_violation | yield_drop | vision_error (+ event_id)
qinspect/<station>/heartbeat
```

## 체인 로테이션

`rotate`: 체인 검증 → 모든 Lot 마감 확인 → 앵커 → SQLite 백업 + CSV/`chain.json` 내보내기 → 라이브 파일을 비운 뒤 `meta.chain_genesis = 이전 head`, `previous_db_sha256` 기록. 새 DB의 첫 레코드는 이전 head에 연결되므로 아카이브 N개가 하나의 체인을 이룹니다.

## 인증 역할

| 역할 | 허용 |
|---|---|
| inspector | `POST /inspect` |
| quality | inspector + 처분, Lot 마감, 관리한계 동결, 알람 확인 |
| auditor | `POST /audit/anchor` |
| admin | 전부 (`QINSPECT_API_KEY` 단일 키는 admin) |

## 확장 포인트
- 새 치수 특성: `PartSpec`에 `DimensionSpec` 추가 → `vision.measure_geometry` 측정 → `inspector.decide` 판정 → `line.CHARACTERISTICS`.
- CNN 임베딩: `anomaly.patch_features` 교체. 모델 포맷·임계값·해시 체계는 그대로.
- 다중 스테이션/고속: `InspectionDB`를 PostgreSQL 구현으로 교체(인터페이스 동일), 대시보드는 DB만 바라봄.
- 인증 강화: `api.require_key`를 역할 기반(검사원/품질/감사)으로 교체.

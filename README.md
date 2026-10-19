# Quality-inspection

**스마트 팩토리 방산부품 품질 검사** — 1인 프로젝트.
정밀 가공 부품(예시: 탄피 바닥면)의 **비전 검사 → 합/불/검사불가 판정 → 공정 관리(SPC) → 추적성(DB·감사 체인·처분) → 설비 연동(API·MQTT·대시보드) → 품질 표준(샘플링·검사 기준서) → 폐쇄망 배포**를 하나의 Python 패키지 `qinspect`로 구현했습니다.

> 실제 방산 도면·영상은 보안 대상이므로, 정답(ground truth)이 정확히 알려진 **합성 이미지**로 전 구간을 검증합니다.
> `PartSpec`의 모든 치수·공차·등급 기준은 가상의 예시 값입니다. 실제 사진은 `qinspect import` / `calibrate` / `anomaly-eval` 로 바로 넣을 수 있습니다.

![SPC](docs/img/spc_example.png)
![NG overlays](docs/img/overlays_example.png)

## 한 줄 실행

```bash
pip install -e ".[dev]"
qinspect demo --out runs/demo        # 생성 → 이상탐지 학습 → 하이브리드 검사 → Gage R&R → 검사 기준서 (약 1분)
pytest                               # 111개 테스트 (~70 s)
```

`runs/demo/` 에 `results.csv`, `summary.json`, `spc_od_mm.png`, `spc_pocket_mm.png`, `overlays/`, `qinspect.db`(감사 체인·앵커 포함), `anomaly_model.npz`, `gage_rr.json`, `inspection_standard.md` 가 생깁니다.

### 데모 결과 (200개, seed 0 — 다른 시드 1~3도 유출 0)

| 지표 | 값 |
|---|---|
| 판정 정확도 / 유출 / 과검 | **99.5 % / 0 / 1** — 과검 1건은 실제 외경 12.0900(USL 12.10)으로 가드밴드 0.008 안쪽, 즉 설계대로 거른 경계 부품 |
| 결함 유형·등급 분류 | 100 % / 100 % (경결함 14건은 PASS 처리 + 기록, 밝은 점은 불합격 사유 아님) |
| 측정 오차 (σ / 최대) | 외경 0.0036 / 0.0067 mm, 포켓 0.0047 / 0.0098 mm, 동심도 0.002 mm (공차폭 0.2 / 0.16 mm) |
| 공구 마모 드리프트 | 드리프트 시작(part 121) 후 subgroup 29에서 첫 X̄ 규칙 위반, 첫 규격 이탈 부품(164)보다 4 subgroup 앞섬 |
| Gage R&R (%GRR/공차) | 외경 10.0 % · 포켓 20.1 % (조건부) → 제안 가드밴드 0.0067 / 0.0107 mm → 규격에 0.008 / 0.012 적용 |
| 이상탐지 (양품 40장 학습) | AUROC 스크래치 1.0 · 찍힘 1.0 · pit 0.96, 양품 오탐 0, 54 ms/장 |

## 구성

```
 camera / synth / file / HTTP
        │  to_gray8: RGB·16bit·float 정규화
        ▼
 vision  ── Otsu 분할(명암 극성 무관) → 경계·대비 검증 → 섹터별 기준레벨 서브픽셀 원 → 포켓(선택) → 잔차 결함(어두움/밝음)
        ▼
 inspector ── 가드밴드 공차(특성별) · 치명/중/경 등급 · PASS / NG / ERROR
        │           ▲ anomaly(메모리뱅크): 전통 비전이 설명 못 한 이상만 NG 추가, 모델 오류는 기록만
        ▼
 db ── SQLite(WAL) · BEGIN IMMEDIATE 원자적 해시체인 · UNIQUE(prev_hash) · 앵커 · Lot↔규격 바인딩 · 처분 · 교정 · 관리한계 동결
        ├─ line: 카메라→검사→DB→MQTT, 알람(치명/SPC/수율/비전오류), 동결 한계 로드, 증거 이미지 보존
        ├─ api : FastAPI (X-API-Key 쓰기 보호, 업로드 제한, 처분·Lot 마감·알람 확인·교대 리포트)
        └─ dashboard : Streamlit (Lot의 실제 규격, KST 표시, X̄·R, 특성 선택, 알람 확인, 미처분 NG)
 spc · msa · sampling · specdoc · datasets
```

| Phase | 모듈 | 내용 |
|---|---|---|
| 0 데이터 | `synth.py` | 가공 흔적·조명 기울기·노이즈, 스크래치/찍힘/pit, 치수 불량, 공구 마모 드리프트, 정답 CSV, 포켓 없는 부품 |
| 0 비전 | `vision.py` | 서브픽셀 외경·포켓·동심도(조명 기울기 보정), 잔차 기반 결함 검출·극성 구분, 입력 형식 정규화, 오류 슬러그 |
| 1 판정 | `spec.py` `inspector.py` `reasons.py` | **특성별 가드밴드**(Gage R&R 기반), **치명/중/경**, **ERROR 판정**(빈 지그·경계 걸림은 NG가 아님), 사유 코드 카탈로그(한/영) |
| 1 MSA | `msa.py` | **Gage R&R (ANOVA, AIAG)** — 균형 설계 검증, 재고정 조건을 작업자 인자로, 가드밴드 제안 |
| 2 실데이터 | `datasets.py` | 사진 import(해시 매니페스트, 재실행 안전, 레터박스), 16-bit TIFF, 기준 디스크 **교정**(포켓 불필요), **MVTec 레이아웃** 평가 |
| 3 딥러닝식 | `anomaly.py` | **PatchCore 방식 메모리뱅크**(양품만 학습, 중복 제거, 임계값 하한, 2-fold 보류 점수) + 하이브리드 규칙 |
| 0 SPC | `spc.py` | X̄-R, Western Electric 1~4(시작 구간 포함), Phase I 25 subgroup, **동결 한계 평가**, EWMA, Cp/Cpk/Pp/Ppk |
| 4 추적성 | `db.py` | Lot/Serial, 작업자, 영상 SHA-256, 규격·모델 해시, **원자적 SHA-256 해시 체인 + 앵커**, 재검사 연결, **처분(rework/scrap/use_as_is/mrb/reinspect)**, Lot 마감, 관리한계 동결, 교정 기록, 알람 확인 |
| 4 연동 | `line.py` `api.py` `dashboard.py` | 라인 시뮬레이터(**MQTT** 또는 메모리 버스), **FastAPI REST**(인증·업로드 제한·동시성 안전), **Streamlit 대시보드** |
| 5 표준 | `sampling.py` `specdoc.py` | **KS Q ISO 2859-1** 1회 샘플링(표 구조 인코딩, 상·하 화살표), **c=0** 정확 초기하, OC 곡선, 치명결함 Lot 불합격 규칙, **검사 기준서 자동 생성**(사유 코드·관리한계 포함) |
| 5 배포 | `Dockerfile` `docker-compose.yml` `requirements-lock.txt` `scripts/` | 폐쇄망용 이미지(태그 고정)/휠하우스, 잠금 파일, 헬스체크, 비루트 실행 |

## 명령어

```bash
qinspect generate --out data/lot1 --n 300 --seed 1                   # 합성 Lot + ground_truth.csv
qinspect train-anomaly --images data/lot1/images --truth data/lot1/ground_truth.csv --limit 40   # 결함 없는 양품만 학습
qinspect inspect --images data/lot1/images --truth data/lot1/ground_truth.csv \
                 --model anomaly_model.npz --db qinspect.db --operator kim    # 하이브리드 검사 + DB 기록
qinspect gage --parts 10 --operators 3 --trials 3 --out-json gage.json       # Gage R&R → 가드밴드 제안
qinspect run-line --n 300 --db qinspect.db --bus mqtt://localhost:1883 --freeze-by qa -v   # 라인 시뮬레이션, 관리한계 동결
qinspect serve --db qinspect.db --api-key secret --image-dir evidence/      # REST API → /docs
qinspect dashboard --db qinspect.db --tz Asia/Seoul                         # Streamlit
qinspect disposition --db qinspect.db --list --lot LOT-SIM                  # 미처분 NG
qinspect disposition --db qinspect.db --inspection-id 17 --decision rework --by qa
qinspect lot-close --db qinspect.db --lot LOT-SIM --by qa --aql 1.0         # 샘플링 + Lot 처분 기록 + 앵커
qinspect export --db qinspect.db --lot LOT-SIM --since 2026-10-14T00:00:00Z --until 2026-10-14T12:00:00Z --out shift.csv
qinspect sampling --lot-size 1000 --aql 1.0 --defectives 1 --criticals 0   # 샘플링 계획 + 판정
qinspect specdoc --gage gage.json --model anomaly_model.npz --db qinspect.db --out 검사기준서.md
qinspect audit --db qinspect.db --anchor                                    # 해시 체인 검증 + 앵커 (깨지면 exit 2)
qinspect import --src ~/photos --dst data/real                              # 실제 사진 투입 (매니페스트 유지)
qinspect calibrate --image gauge.png --diameter-mm 12.0 --out spec_real.json --db qinspect.db --gauge-id G-1
qinspect anomaly-eval --dataset ~/mvtec/metal_nut                           # MVTec 형식 데이터셋 평가 (AUROC)
```

REST 예시:
```bash
curl -H "X-API-Key: secret" -F file=@part.png -F lot_id=LOT-07 -F operator=kim http://localhost:8000/inspect
curl http://localhost:8000/lots/LOT-07/summary            # 수율(초회/최종), 미처분 NG, 파레토, 특성별 SPC·Cpk
curl -H "X-API-Key: secret" -X POST http://localhost:8000/inspections/17/disposition -d '{"decision":"rework","decided_by":"qa"}' -H 'content-type: application/json'
curl -H "X-API-Key: secret" -X POST http://localhost:8000/lots/LOT-07/close -d '{"decided_by":"qa","aql":1.0}' -H 'content-type: application/json'
curl "http://localhost:8000/reports/shift?lot_id=LOT-07&since=2026-10-14T00:00:00Z&until=2026-10-14T12:00:00Z"
curl http://localhost:8000/audit/verify
```

## 설계 포인트

- **PASS / NG / ERROR**: 빈 지그·경계에 걸린 부품·조명 불량은 부품 불량이 아니라 "검사 불가"입니다. ERROR는 수율·SPC·파레토에서 제외되고 `vision_error` 알람으로 재촬영을 요구합니다.
- **가드밴드는 측정 시스템에서 나옵니다**: Gage R&R의 GRR σ×2를 특성별로 적용(외경 0.008, 포켓 0.012). 경계 부품의 유출을 과검으로 바꾸는 의도적 선택이며 결과표에 그대로 드러납니다.
- **하이브리드 판정 규칙**: 이상탐지는 *설명되지 않는* 이상에만 NG를 추가합니다. 전통 비전이 찾은 경결함에 높은 점수가 나오면 등급 판정이 우선하고, 모델 오류는 판정을 바꾸지 않고 `anomaly_error`로 남습니다.
- **감사 체인은 원자적입니다**: `BEGIN IMMEDIATE` 안에서 머리 해시를 읽고 삽입하며 `UNIQUE(prev_hash)`로 분기가 불가능합니다. NaN·-0.0·numpy 타입은 SQLite 저장 표현으로 정규화해 해시하므로 검증이 어긋나지 않습니다. 최신 행 삭제는 체인만으로 못 보므로 **앵커**(`anchors`, Lot 마감·배치·교대 종료 시)로 잡습니다.
- **Lot은 규격 개정에 묶입니다**: 같은 Lot에 다른 `spec_hash`로 검사하면 거부됩니다. 대시보드·API는 Lot에 저장된 규격으로 한계·Cpk를 표시합니다.
- **문서 = 코드**: 검사 기준서는 `PartSpec`·Gage 결과·동결 관리한계·사유 코드 카탈로그에서 생성됩니다.
- **의존성 최소화**: 비전·이상탐지가 numpy/scipy만으로 동작해 폐쇄망 설치가 쉽습니다. CNN 백본은 `anomaly.patch_features` 한 함수만 바꾸면 됩니다.

## 한계와 다음 단계

- 모든 성능 수치는 **합성 데이터** 기준입니다. 실제 부품에서는 `detect_defects`의 임계값, 이상탐지 `threshold_sigma`, 등급 길이 기준 재조정과 MSA 재실시가 필요합니다. 결함 길이는 측정 extent 기준으로 정의 길이 대비 약 ±15 % 편차가 있습니다.
- ISO 2859-1 표는 표의 구조(Ac 0,↑,1,2,3,5,7,10,14,21 대각선)로 인코딩했고 전 셀에서 생산자 위험 ≤ 20 %를 테스트로 확인했지만, 계약상 판정 전에는 원문 표준과 대조하세요. MIL-STD-1916 고유 표는 미구현이고 c=0 계산기로 대체했습니다.
- API 인증은 단일 API 키입니다. 역할별 권한, mTLS, 앵커의 외부 타임스탬프 보관은 `docs/ROADMAP.md` 참고.
- SQLite + 단일 노드 구성입니다. 다중 스테이션이면 PostgreSQL + 메시지 큐로 바꾸는 것이 맞습니다(인터페이스는 `InspectionDB` 하나).

자세한 구조는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), 배포는 [docs/DEPLOY_OFFLINE.md](docs/DEPLOY_OFFLINE.md), 생성된 문서 예시는 [docs/inspection_standard_example.md](docs/inspection_standard_example.md).

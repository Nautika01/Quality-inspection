# 실제 부품 촬영 가이드 (파일럿)

합성 데이터에서 검증한 파이프라인을 실제 부품으로 옮길 때 가장 많이 실패하는 곳은 알고리즘이 아니라 **촬영 조건**입니다. 아래 순서대로 하면 `qinspect`가 기대하는 입력(균일한 배경, 부품 전체가 프레임 안, 안정된 조명)이 나옵니다.

## 1. 하드웨어 (저가 세팅 기준)

| 항목 | 권장 | 이유 |
|---|---|---|
| 카메라 | 1280×720 이상 모노/컬러 UVC 웹캠 또는 산업용 카메라(글로벌 셔터면 더 좋음) | 16 px/mm 정도가 나오도록 거리 조절 |
| 렌즈 | 고정 초점, 왜곡 적은 것 | 렌즈 왜곡은 외경 측정에 바로 오차로 들어감 |
| 조명 | **링 라이트**(확산판 부착) 또는 돔 조명 | 점광원은 광택면 하이라이트·그림자를 만듦. 기울기 조명은 알고리즘이 보정하지만 하이라이트는 `bright_spot`으로만 기록됨 |
| 배경 | 무광 검정(또는 무광 흰색) 플레이트 | 부품과 배경 대비 ≥ 25 gray 필요. 극성은 상관없음 |
| 지그 | 부품 중심이 프레임 중앙 ±10 % 안에 오도록 V블록/포켓 지그 | 경계에 걸리면 `ERROR: part_touches_border` |
| 기준 게이지 | 외경을 아는 디스크/핀(예: 12.000 mm 링 게이지) | px/mm 교정용 (`qinspect calibrate`) |

## 2. 카메라 설정

- 자동 노출·자동 화이트밸런스 **끄기**. 포화 픽셀(≥250) 비율 1 % 미만, 평균 밝기 80~160.
- 프레임 채움률(부품 면적/프레임) 8 % 이상. 너무 작으면 서브픽셀 정밀도가 떨어집니다.
- 초점은 부품 면(가공 흔적)이 보이게. `qinspect capture --preview`가 출력하는 `focus` 값을 비교하며 맞춥니다(상대값).

```bash
qinspect capture --device 0 --n 1 --out data/pilot --preview        # 품질 리포트만 확인
```
출력 예: `{"mean": 128.3, "saturated_pct": 0.0, "focus": 412.5, "part_found": true, "part_diameter_px": 193.2, "frame_fill_pct": 11.4, "warnings": []}`

## 3. 교정

```bash
qinspect capture --device 0 --n 1 --out data/gauge --prefix GAUGE
qinspect calibrate --image data/gauge/images/GAUGE-00001.png --diameter-mm 12.000 \
                   --spec spec_pilot.json --out spec_pilot.json --db pilot.db --gauge-id RG-12 --operator kim
```
- 교대 시작마다 반복하고, 두 번 연속 교정값이 0.2 % 이상 차이 나면 카메라/지그가 움직인 것입니다.
- `spec_pilot.json`의 `px_per_mm`가 갱신됩니다. `image_size`는 합성 렌더 전용이라 무시해도 됩니다.

## 4. 양품 촬영 (이상탐지 학습용)

```bash
qinspect capture --device 0 --n 40 --out data/pilot_good --prefix GOOD --interval 2
qinspect train-anomaly --images data/pilot_good/images --spec spec_pilot.json --out pilot_model.npz
```
- 서로 다른 부품 40개 이상, 같은 부품을 돌려 놓고 찍는 것은 중복으로 간주되지 않지만 변동성이 적어 임계값이 낮아집니다.
- 학습 전에 각 사진을 `qinspect inspect`로 돌려 `ERROR`/`NG`가 없는지 확인하세요. 결함 있는 부품이 섞이면 임계값이 올라갑니다.

## 5. 검사 운영

```bash
qinspect capture --device 0 --n 100 --out data/pilot_lot1 --prefix L1 --station cam1 --operator kim
qinspect inspect --images data/pilot_lot1/images --spec spec_pilot.json --model pilot_model.npz --db pilot.db --lot PILOT-1
qinspect gage --spec spec_pilot.json --out-json gage_pilot.json     # 합성 재고정 시뮬레이션이 아니라 실제 재촬영으로 할 때는 CSV를 직접 만들어 gage_rr_anova 사용
```

## 6. 임계값 조정 포인트

| 증상 | 조정 |
|---|---|
| 양품이 `surface_dent` 로 NG | `detect_defects(k_sigma, min_abs)` 상향, 또는 `min_defect_area_px` 상향 |
| 결함을 못 잡음 | `k_sigma` 하향, 조명 각도 변경(저각 조명이 스크래치에 유리) |
| 양품에 `anomaly_score` NG | 학습 이미지 추가, `train-anomaly --threshold-sigma 6` |
| 포켓 미검출 | 포켓이 면과 15 gray 이상 차이 나도록 조명 조정; 포켓 없는 부품이면 `pocket_diameter: null` |
| 외경이 일정하게 치우침 | 렌즈 왜곡 — 부품을 항상 프레임 중앙에, 또는 게이지로 재교정 |

## 7. 기록

촬영 세션은 `import_manifest.csv`(시리얼↔SHA-256)와 `capture_log.jsonl`(프레임별 품질 리포트)을 남깁니다. 검사 기록의 `image_sha256`과 대조하면 사진이 바뀌지 않았음을 증명할 수 있습니다.

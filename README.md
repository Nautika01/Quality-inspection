# Quality-inspection

**스마트 팩토리 방산부품 품질 검사** 1인 프로젝트.
정밀 가공 부품(예시: 탄피 바닥면)의 **비전 검사(치수 + 외관 결함) → 합/불 판정 → SPC 공정 관리**를 하나의 파이프라인으로 구현합니다.

> 실제 방산 도면·영상은 보안 대상이므로, MVP는 **정답(ground truth)이 정확히 알려진 합성 이미지**로 시작합니다.
> `PartSpec`의 치수는 모두 가상의 예시 값입니다.

## 현재 할 수 있는 것 (v0.1)

| 단계 | 내용 | 모듈 |
|---|---|---|
| 데이터 | 부품 상면 이미지 합성 — 가공 흔적, 조명 불균일, 노이즈, 스크래치/찍힘 결함, 치수 불량, **공구 마모로 인한 외경 드리프트** | `synth.py` |
| 치수 검사 | 외경·프라이머 포켓 직경·동심도를 **서브픽셀**로 측정 (오차 σ ≈ 0.003 mm) | `vision.py` |
| 외관 검사 | 무결함 모델(반경 프로파일 + 조명 평면) 잔차 → 히스테리시스 이진화 → 형상으로 스크래치/찍힘 분류 | `vision.py` |
| 판정 | 시리얼별 PASS/NG + NG 사유 (추적성) | `inspector.py` |
| SPC | X̄-R 관리도, Western Electric 규칙 1~4, Phase I/II 관리한계, Cp/Cpk/Pp/Ppk | `spc.py` |
| 리포트 | `results.csv`, `summary.json`, 관리도 PNG, NG 부품 오버레이 이미지, 정답 대비 유출/과검률 | `report.py` |

데모(200개, seed 0) 결과: 정확도 99.5%, 과검 0, 결함 유형 분류 100%, 공구 마모 드리프트를 **규격 이탈 전에** 관리도에서 탐지.

![](docs/img/spc_example.png)

## 빠른 시작

```bash
pip install -e ".[dev]"

# 생성 + 검사 한 번에
qinspect demo --out runs/demo

# 따로 실행
qinspect generate --out data/synth --n 300 --seed 1
qinspect inspect --images data/synth/images --truth data/synth/ground_truth.csv --out runs/lot1

pytest
```

## 구조

```
src/qinspect/
  spec.py       부품 규격(공차) 정의
  synth.py      합성 데이터 + 정답 생성
  vision.py     치수 측정 / 결함 검출
  inspector.py  판정 로직
  spc.py        관리도 · 공정능력
  report.py     배치 실행 · 산출물
  cli.py        명령줄 인터페이스
tests/          단위·통합 테스트
docs/ROADMAP.md 단계별 확장 계획
```

다음 단계는 [docs/ROADMAP.md](docs/ROADMAP.md)를 참고하세요.

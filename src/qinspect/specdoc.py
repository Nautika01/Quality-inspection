"""Generate a 검사 기준서 (inspection standard) in Markdown from the machine-readable spec.

The idea: the same `PartSpec` that drives the inspector also produces the human
document, so the paperwork can never drift from what the machine actually checks.
"""

from __future__ import annotations

from datetime import date

from .db import spec_hash
from .sampling import SamplingPlan, iso2859_single_normal
from .spec import PartSpec


def render_inspection_standard(spec: PartSpec, *, lot_size: int = 1000, aql: float = 1.0,
                               gage: dict | None = None, model_hash: str | None = None,
                               program_version: str = "") -> str:
    plan: SamplingPlan = iso2859_single_normal(lot_size, aql)
    gb = spec.guard_band_mm
    od, pk, gr = spec.outer_diameter, spec.pocket_diameter, spec.grading

    def dim_row(name, d):
        return (f"| {name} | {d.nominal:.3f} | {d.lsl:.3f} ~ {d.usl:.3f} | "
                f"{d.lsl + gb:.3f} ~ {d.usl - gb:.3f} | 비전 서브픽셀 측정 |")

    lines = [
        f"# 검사 기준서 — {spec.name}",
        "",
        f"| 항목 | 내용 |\n|---|---|\n| 품번 / 개정 | {spec.part_number} / Rev. {spec.revision} |",
        f"| 규격 해시 | `{spec_hash(spec)}` |",
        f"| 검사 프로그램 버전 | {program_version or '-'} |",
        f"| 이상탐지 모델 해시 | `{model_hash}` |" if model_hash else "| 이상탐지 모델 | 미사용 |",
        f"| 작성일 | {date.today().isoformat()} |",
        "",
        "> 본 문서는 `PartSpec`에서 자동 생성됩니다. 수치는 예시이며 실제 도면과 무관합니다.",
        "",
        "## 1. 적용 범위",
        "자동 비전 검사 셀에서 수행하는 전수 검사(치수·외관)와 Lot 판정(샘플링)에 적용한다.",
        "",
        "## 2. 치수 검사",
        "",
        "| 특성 | 공칭 (mm) | 도면 공차 (mm) | 판정 한계 (가드밴드 적용) | 측정 방법 |",
        "|---|---|---|---|---|",
        dim_row("외경", od),
        dim_row("프라이머 포켓 직경", pk),
        f"| 포켓 동심도 | 0 | ≤ {spec.max_concentricity_mm:.3f} | ≤ {spec.max_concentricity_mm - gb:.3f} | 외경·포켓 중심 거리 |",
        "",
        f"* 가드밴드 {gb:.3f} mm 는 측정 불확도를 반영해 공차 안쪽으로 판정 한계를 좁힌 값이다 (ISO 14253-1 판정 규칙).",
        f"* 픽셀 스케일 {spec.px_per_mm} px/mm, 영상 크기 {spec.image_size} px. 교정은 기준 게이지로 교대 시작 시 실시한다.",
        "",
        "## 3. 외관 검사",
        "",
        "| 등급 | 기준 | 처리 |",
        "|---|---|---|",
        f"| 치명 (critical) | 길이 ≥ {gr.critical_len_mm} mm 또는 유형 {', '.join(gr.critical_types)} | 불합격, 즉시 알람 |",
        f"| 중 (major) | 길이 ≥ {gr.major_len_mm} mm | 불합격 |",
        f"| 경 (minor) | 길이 < {gr.major_len_mm} mm | 합격, 기록 유지 |",
        "",
        f"* 검출 대상: 스크래치, 찍힘(dent). 최소 검출 면적 {spec.min_defect_area_px} px.",
        "* 이상탐지 모델 사용 시 점수가 학습 임계값을 넘으면 사유 `anomaly_score`로 불합격 처리한다.",
        "",
        "## 4. 측정 시스템",
    ]
    if gage:
        lines += ["", "| 특성 | %GRR (TV) | %GRR (공차) | ndc | 판정 |", "|---|---|---|---|---|"]
        for k, g in gage.items():
            lines.append(f"| {g['characteristic']} | {g['pct_grr_tv']:.1f} % | {g['pct_grr_tol']:.1f} % | {g['ndc']} | {g['verdict']} |")
        lines += ["", "* AIAG MSA 기준: 10 % 미만 양호, 10~30 % 조건부, 30 % 초과 부적합."]
    else:
        lines.append("Gage R&R 결과 없음 — `qinspect gage` 실행 후 재생성.")
    lines += [
        "",
        "## 5. Lot 판정 (샘플링)",
        "",
        f"* 전수 비전 검사 결과와 별도로, 출하 Lot 은 {plan.standard}, 검사수준 II, AQL {aql} % 로 판정한다.",
        f"* Lot 크기 {lot_size} → 코드문자 {plan.code_letter}, 샘플 n = {plan.sample_size}, Ac = {plan.accept}, Re = {plan.reject}"
        + (f" ({plan.note})" if plan.note else ""),
        "* 샘플에서 치명 결함 1건 이상 발견 시 AQL 과 무관하게 Lot 불합격 (c = 0 규칙).",
        "",
        "## 6. 기록 및 추적성",
        "* 시리얼별 판정, 측정값, 결함 좌표, 프로그램 버전, 규격 해시, 모델 해시를 DB 에 저장한다.",
        "* 검사 기록은 해시 체인으로 연결되어 사후 위·변조 시 `audit/verify` 에서 탐지된다.",
        "* 보존 기간: 계약 요구에 따름 (방산 품질보증 요구사항 참조).",
        "",
        "## 7. 공정 관리",
        "* 외경에 대해 X̄-R 관리도(n = 5)를 운용하고 Western Electric 규칙 1~4 위반 시 알람을 발행한다.",
        "* Cpk < 1.33 이면 공정 개선 조치, Cpk < 1.0 이면 전수 선별 및 원인 분석.",
        "",
    ]
    return "\n".join(lines)

"""Generate a 검사 기준서 (inspection standard) in Markdown from the machine-readable spec.

The idea: the same `PartSpec` that drives the inspector also produces the human
document, so the paperwork can never drift from what the machine actually checks.
"""

from __future__ import annotations

from datetime import date

from .db import spec_hash
from .reasons import GRADES, REASONS, VERDICTS
from .sampling import SamplingPlan, iso2859_single_normal
from .spec import PartSpec


def render_inspection_standard(spec: PartSpec, *, lot_size: int = 1000, aql: float = 1.0,
                               gage: dict | None = None, model_hash: str | None = None,
                               program_version: str = "", control_limits: dict | None = None) -> str:
    plan: SamplingPlan = iso2859_single_normal(lot_size, aql)
    od, pk, gr = spec.outer_diameter, spec.pocket_diameter, spec.grading

    def dim_row(name, d):
        gb = spec.guard_band_for(d)
        return (f"| {name} | {d.nominal:.3f} | {d.lsl:.3f} ~ {d.usl:.3f} | {gb:.3f} | "
                f"{d.lsl + gb:.3f} ~ {d.usl - gb:.3f} | 비전 서브픽셀 측정 |")

    def disposition(grade: str) -> str:
        return "불합격" if grade in gr.reject_grades else "합격, 기록 유지"

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
        "| 특성 | 공칭 (mm) | 도면 공차 (mm) | 가드밴드 (mm) | 판정 한계 | 측정 방법 |",
        "|---|---|---|---|---|---|",
        dim_row("외경", od),
    ]
    if pk is not None:
        lines.append(dim_row("프라이머 포켓 직경", pk))
        lines.append(f"| 포켓 동심도 | 0 | ≤ {spec.max_concentricity_mm:.3f} | {spec.guard_band_mm:.3f} | "
                     f"≤ {spec.max_concentricity_mm - spec.guard_band_mm:.3f} | 외경·포켓 중심 거리 |")
    lines += [
        "",
        "* 가드밴드는 측정 불확도(Gage R&R의 GRR σ × 2)를 반영해 공차 안쪽으로 판정 한계를 좁힌 값이다 (ISO 14253-1 판정 규칙).",
        f"* 픽셀 스케일 {spec.px_per_mm} px/mm. 교정은 기준 게이지로 교대 시작 시 실시하고 교정 기록(`calibrations`)에 남긴다.",
        "",
        "## 3. 외관 검사",
        "",
        "| 등급 | 기준 | 처리 |",
        "|---|---|---|",
        f"| {GRADES['critical'][0]} (critical) | 길이 ≥ {gr.critical_len_mm} mm 또는 유형 {', '.join(gr.critical_types)} | {disposition('critical')}, 즉시 알람 |",
        f"| {GRADES['major'][0]} (major) | 길이 ≥ {gr.major_len_mm} mm | {disposition('major')} |",
        f"| {GRADES['minor'][0]} (minor) | 길이 < {gr.major_len_mm} mm | {disposition('minor')} |",
        "",
        f"* 검출 대상: 스크래치, 찍힘(dent) 등 어두운(재료 제거) 결함. 길이는 검출 영역의 주축 방향 폭(측정값 기준, 정의 길이 대비 약 ±15 %).",
        f"* 밝은 점({', '.join(gr.non_rejecting_types) or '-'}: 광택·이물)은 기록만 하고 단독으로는 불합격 사유가 아니다.",
        f"* 최소 검출 면적 {spec.min_defect_area_px} px (@{spec.px_per_mm} px/mm).",
        "* 이상탐지 모델 사용 시: 전통 비전이 아무 결함도 찾지 못했는데 점수가 임계값을 넘으면 사유 `anomaly_score`로 불합격. 전통 비전이 결함을 찾은 경우 등급 판정이 우선한다.",
        "",
        "## 4. 측정 시스템",
    ]
    if gage:
        lines += ["", "| 특성 | %GRR (TV) | %GRR (공차) | ndc | 권장 가드밴드 | 판정 |", "|---|---|---|---|---|---|"]
        for g in gage.values():
            gb = g.get("suggested_guard_band_mm")
            lines.append(f"| {g['characteristic']} | {g['pct_grr_tv']:.1f} % | {g['pct_grr_tol']:.1f} % | {g['ndc']} | "
                         f"{gb:.4f} mm | {g['verdict']} |" if gb is not None else
                         f"| {g['characteristic']} | {g['pct_grr_tv']:.1f} % | {g['pct_grr_tol']:.1f} % | {g['ndc']} | - | {g['verdict']} |")
        lines += ["", "* AIAG MSA 기준: 10 % 미만 양호, 10~30 % 조건부, 30 % 초과 부적합."]
    else:
        lines.append("Gage R&R 결과 없음 — `qinspect gage --out-json` 실행 후 재생성.")
    lines += [
        "",
        "## 5. Lot 판정 (샘플링)",
        "",
        f"* 전수 비전 검사 결과와 별도로, 출하 Lot 은 {plan.standard}, 검사수준 II, AQL {aql} % 로 판정한다.",
        f"* Lot 크기 {lot_size} → 코드문자 {plan.code_letter}, 샘플 n = {plan.sample_size}, Ac = {plan.accept}, Re = {plan.reject}"
        + (f" ({plan.note})" if plan.note else ""),
        "* 샘플에서 치명 결함 1건 이상 발견 시 AQL 과 무관하게 Lot 불합격 (`sampling.lot_disposition`, 치명 결함 c = 0).",
        "* Lot 판정 결과는 `lot_dispositions` 에 판정자·시각과 함께 기록한다.",
        "",
        "## 6. 판정 코드",
        "",
        "| 판정 | 의미 |", "|---|---|",
    ] + [f"| {k} | {v[0]} |" for k, v in VERDICTS.items()] + [
        "",
        "| 사유 코드 | 설명 |", "|---|---|",
    ] + [f"| `{k}` | {v[0]} |" for k, v in REASONS.items()] + [
        "",
        "## 7. 기록 및 추적성",
        "* 시리얼별 판정, 측정값, 결함 좌표, 프로그램 버전, 규격 해시, 모델 해시, 스테이션, 작업자, 영상 해시를 DB 에 저장한다.",
        "* 검사 기록은 해시 체인으로 연결되고, 체인 머리(head)는 주기적으로 `anchors` 에 고정한다. 과거 기록의 수정·삭제는 `audit/verify` 에서 탐지된다.",
        "* NG 는 처분(`dispositions`: rework / scrap / use_as_is / mrb / reinspect)으로 종결하며 재검사는 원 기록에 연결된다.",
        "* 보존 기간: 계약 요구에 따름 (방산 품질보증 요구사항 참조).",
        "",
        "## 8. 공정 관리",
        "* 외경·포켓 직경·동심도에 대해 X̄-R 관리도(n = 5)를 운용하고 Western Electric 규칙 1~4 위반 시 알람을 발행한다.",
        "* 관리한계는 품질 담당자가 동결(`control_limits`)하며, 동결 전에는 최초 25개 부분군을 기준선으로 쓴다.",
        "* Cpk < 1.33 이면 공정 개선 조치, Cpk < 1.0 이면 전수 선별 및 원인 분석.",
    ]
    if control_limits:
        lines += ["", "| 특성 | n | CL | UCL | LCL | R̄ |", "|---|---|---|---|---|---|"]
        for ch, cl in control_limits.items():
            lines.append(f"| {ch} | {cl['n']} | {cl['center']:.4f} | {cl['ucl']:.4f} | {cl['lcl']:.4f} | {cl['r_bar']:.4f} |")
    lines.append("")
    return "\n".join(lines)

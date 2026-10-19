"""First Article Inspection report in the structure of AS9102 (Forms 1-3), from the DB.

Form 1  Part Number Accountability   - part, revision, spec hash, lot, program/model versions
Form 2  Product Accountability       - measurement system (calibration, Gage R&R), anomaly model
Form 3  Characteristic Accountability - each toleranced characteristic: designed / measured / result

This is a generated *draft*: the organisation's own AS9102 template, signatures and
material/process certifications still have to be attached by the quality engineer.
"""

from __future__ import annotations

from datetime import date

from . import __version__
from .db import InspectionDB, spec_hash
from .reasons import describe
from .spec import PartSpec


def render_fai(db: InspectionDB, serial: str, *, spec: PartSpec | None = None, gage: dict | None = None,
               prepared_by: str = "", customer: str = "", drawing_no: str = "") -> str:
    hist = db.serial_history(serial)
    if not hist:
        raise ValueError(f"serial {serial} has no inspection record")
    rec = hist[-1]
    spec = spec or db.lot_spec(rec["lot_id"]) or PartSpec()
    cal = db.calibrations(limit=1)
    cal = cal[-1] if cal else None
    gb = spec.guard_band_for

    def res_row(no, name, nominal, lo, hi, measured, unit="mm"):
        ok = "" if measured is None else ("합격" if lo <= measured <= hi else "불합격")
        meas = "-" if measured is None else f"{measured:.4f}"
        return f"| {no} | {name} | {nominal} | {lo:.3f} ~ {hi:.3f} | {meas} | {unit} | 비전(서브픽셀) | {ok} |"

    rows = [res_row(1, "외경", f"{spec.outer_diameter.nominal:.3f}", spec.outer_diameter.lsl, spec.outer_diameter.usl, rec["od_mm"])]
    if spec.has_pocket:
        pk = spec.pocket_diameter
        rows.append(res_row(2, "포켓 직경", f"{pk.nominal:.3f}", pk.lsl, pk.usl, rec["pocket_mm"]))
        rows.append(res_row(3, "포켓 동심도", "0", 0.0, spec.max_concentricity_mm, rec["concentricity_mm"]))
    n_surface = len(rows) + 1
    defects = rec.get("defects", [])
    surface = "결함 없음" if not defects else "; ".join(f"{d['kind']}/{d['grade']} {d['length_mm']} mm" for d in defects)
    rows.append(f"| {n_surface} | 표면 결함 | 없음 | 치명/중 결함 없음 | {surface} | - | 비전(잔차) | "
                f"{'합격' if not any(d['grade'] in spec.grading.reject_grades for d in defects) else '불합격'} |")

    verdict = {"PASS": "합격", "NG": "불합격", "ERROR": "검사 불가"}.get(rec["verdict"], rec["verdict"])
    reasons = ", ".join(describe(r) for r in rec["reasons"].split(";") if r) or "-"
    dispositions = "; ".join(f"{d['decision']} by {d['decided_by']} ({d['decided_at']})" for h in hist for d in h["dispositions"]) or "-"

    lines = [
        f"# 초도품 검사 보고서 (AS9102 양식 구조) — {spec.part_number}",
        "",
        f"작성일 {date.today().isoformat()} · 작성 {prepared_by or '-'} · 생성 프로그램 qinspect {__version__}",
        "",
        "> 자동 생성 초안입니다. 조직의 AS9102 양식, 서명, 자재/공정 인증서는 품질 담당자가 첨부합니다.",
        "",
        "## Form 1 — Part Number Accountability",
        "",
        "| 항목 | 내용 |", "|---|---|",
        f"| 1. 품번 | {spec.part_number} |",
        f"| 2. 품명 | {spec.name} |",
        f"| 3. 시리얼 | {serial} |",
        f"| 4. FAI 보고서 번호 | FAI-{serial} |",
        f"| 5. 도면 번호 / 개정 | {drawing_no or '-'} / Rev. {spec.revision} |",
        f"| 6. 규격 해시 | `{rec['spec_hash'] or spec_hash(spec)}` |",
        f"| 7. Lot | {rec['lot_id']} |",
        f"| 8. 고객 | {customer or '-'} |",
        f"| 9. 검사 일시 (UTC) | {rec['inspected_at']} |",
        f"| 10. 스테이션 / 작업자 | {rec['station'] or '-'} / {rec['operator'] or '-'} |",
        f"| 11. 검사 프로그램 버전 | {rec['program_version']} |",
        f"| 12. 이상탐지 모델 해시 | {rec['model_hash'] or '미사용'} |",
        f"| 13. 영상 SHA-256 | {rec['image_sha256'] or '-'} |",
        f"| 14. 기록 해시 (감사 체인) | `{rec['record_hash']}` |",
        f"| 15. 재검사 여부 | {'재검사 (원기록 #' + str(rec['reinspection_of']) + ')' if rec['reinspection_of'] else '초회 검사'} |",
        "",
        "## Form 2 — Product Accountability (측정 시스템)",
        "",
        "| 항목 | 내용 |", "|---|---|",
        f"| 측정 장비 | 비전 검사 셀 ({spec.px_per_mm} px/mm) |",
        "| 최근 교정 | " + (f"{cal['ts']} · 게이지 {cal['gauge_id'] or '-'} · {cal['px_per_mm']:.4f} px/mm" if cal else "교정 기록 없음") + " |",
    ]
    if gage:
        for g in gage.values():
            lines.append(f"| Gage R&R ({g['characteristic']}) | %GRR(공차) {g['pct_grr_tol']:.1f} %, ndc {g['ndc']}, {g['verdict']} |")
    else:
        lines.append("| Gage R&R | 결과 미첨부 (`qinspect gage --out-json`) |")
    lines += [
        f"| 가드밴드 | 외경 {gb(spec.outer_diameter):.3f} mm" + (f", 포켓 {gb(spec.pocket_diameter):.3f} mm" if spec.has_pocket else "") + " |",
        "",
        "## Form 3 — Characteristic Accountability, Verification and Compatibility Evaluation",
        "",
        "| No. | 특성 | 설계값 | 허용 한계 (도면) | 측정값 | 단위 | 측정 방법 | 결과 |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"**종합 판정: {verdict}**" + (f" — 사유: {reasons}" if reasons != "-" else ""),
        "",
        f"처분 이력: {dispositions}",
        "",
        "| 검토 | 성명 | 서명 | 일자 |", "|---|---|---|---|",
        "| 작성 (품질) |  |  |  |", "| 검토 (생산기술) |  |  |  |", "| 승인 (품질보증) |  |  |  |",
        "",
    ]
    return "\n".join(lines)

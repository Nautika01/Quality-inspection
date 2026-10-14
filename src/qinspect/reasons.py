"""Catalogue of reason / error codes used in verdicts, with Korean and English labels.

Codes are stable identifiers (stored in the DB and hashed into the audit chain);
labels are for documents and dashboards. Surface defects use the pattern
``surface_<kind>:<grade>``; vision errors use ``vision:<slug>``.
"""

from __future__ import annotations

REASONS: dict[str, tuple[str, str]] = {
    "od_out_of_tolerance": ("외경 공차 이탈", "outer diameter out of tolerance"),
    "pocket_out_of_tolerance": ("포켓 직경 공차 이탈", "pocket diameter out of tolerance"),
    "concentricity": ("동심도 초과", "concentricity exceeded"),
    "anomaly_score": ("이상탐지 점수 초과(미설명 이상)", "anomaly score above threshold (unexplained)"),
    "surface_scratch": ("표면 스크래치", "surface scratch"),
    "surface_dent": ("표면 찍힘", "surface dent"),
    "surface_bright_spot": ("표면 밝은 점(광택/이물)", "bright spot (glare / debris)"),
    "surface_crack": ("균열", "crack"),
    # vision / acquisition errors -> verdict ERROR, not NG
    "vision:part_not_found": ("부품 미검출(빈 지그/조명 불량)", "part not found (empty nest / lighting)"),
    "vision:low_contrast": ("부품-배경 대비 부족", "insufficient part/background contrast"),
    "vision:part_touches_border": ("부품이 화면 경계에 걸림", "part clipped by image border"),
    "vision:pocket_not_found": ("포켓 미검출", "pocket not found"),
    "vision:pocket_degenerate": ("포켓 형상 비정상", "pocket geometry degenerate"),
    "vision:degenerate_geometry": ("기하 측정값 비정상(NaN)", "non-finite geometry"),
    "vision:bad_image": ("이미지 형식 오류", "unsupported or corrupt image"),
    "vision:internal_error": ("비전 내부 오류", "vision internal error"),
}

GRADES: dict[str, tuple[str, str]] = {
    "critical": ("치명", "critical"),
    "major": ("중", "major"),
    "minor": ("경", "minor"),
}

VERDICTS: dict[str, tuple[str, str]] = {
    "PASS": ("합격", "pass"),
    "NG": ("불합격", "reject"),
    "ERROR": ("검사 불가(재촬영)", "inspection error (re-trigger)"),
}


def describe(code: str, lang: str = "ko") -> str:
    """Human label for a reason code such as ``surface_dent:major`` or ``vision:part_not_found``."""
    i = 0 if lang == "ko" else 1
    base, _, grade = code.partition(":")
    if code in REASONS:
        return REASONS[code][i]
    if base in REASONS:
        label = REASONS[base][i]
        if grade in GRADES:
            label += f" ({GRADES[grade][i]})"
        return label
    return code

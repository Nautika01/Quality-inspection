"""Turn vision measurements into a traceable PASS/NG decision per serial number."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .spec import PartSpec
from .vision import Defect, VisionResult, analyze


@dataclass
class GradedDefect:
    defect: Defect
    grade: str  # critical / major / minor


@dataclass
class InspectionResult:
    serial: str
    inspected_at: str
    verdict: str
    reasons: list[str]
    od_mm: float | None = None
    pocket_mm: float | None = None
    concentricity_mm: float | None = None
    defects: list[GradedDefect] = field(default_factory=list)
    vision: VisionResult | None = None
    program_version: str = __version__
    anomaly_score: float | None = None  # filled by the deep-learning style detector when enabled

    @property
    def worst_grade(self) -> str:
        order = {"critical": 0, "major": 1, "minor": 2}
        return min((g.grade for g in self.defects), key=order.get, default="")

    def to_row(self) -> dict:
        return {
            "serial": self.serial,
            "inspected_at": self.inspected_at,
            "verdict": self.verdict,
            "reasons": ";".join(self.reasons),
            "od_mm": self.od_mm,
            "pocket_mm": self.pocket_mm,
            "concentricity_mm": self.concentricity_mm,
            "n_defects": len(self.defects),
            "defect_types": ";".join(sorted({g.defect.kind for g in self.defects})),
            "worst_grade": self.worst_grade,
            "max_defect_len_mm": max((g.defect.length_mm for g in self.defects), default=0.0),
            "anomaly_score": self.anomaly_score,
            "program_version": self.program_version,
        }


def decide(vr: VisionResult, spec: PartSpec, serial: str, now: str | None = None) -> InspectionResult:
    """Apply tolerances (with guard band) and defect grading to a vision result."""
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    if vr.geometry is None:
        return InspectionResult(serial, now, "NG", [f"vision:{vr.error}"], vision=vr)

    g, ppm, gb = vr.geometry, spec.px_per_mm, spec.guard_band_mm
    od = round(2 * g.outer.r / ppm, 4)
    pk = round(2 * g.pocket.r / ppm, 4)
    conc = round(float(np.hypot(g.outer.cx - g.pocket.cx, g.outer.cy - g.pocket.cy)) / ppm, 4)

    reasons = []
    if not spec.outer_diameter.ok(od, gb):
        reasons.append("od_out_of_tolerance")
    if not spec.pocket_diameter.ok(pk, gb):
        reasons.append("pocket_out_of_tolerance")
    if conc > spec.max_concentricity_mm - gb:
        reasons.append("concentricity")

    graded = [GradedDefect(d, spec.grading.grade(d.kind, d.length_mm)) for d in vr.defects]
    for gd in sorted(graded, key=lambda x: (x.grade, x.defect.kind)):
        if gd.grade in spec.grading.reject_grades:
            tag = f"surface_{gd.defect.kind}:{gd.grade}"
            if tag not in reasons:
                reasons.append(tag)

    return InspectionResult(
        serial=serial,
        inspected_at=now,
        verdict="NG" if reasons else "PASS",
        reasons=reasons,
        od_mm=od,
        pocket_mm=pk,
        concentricity_mm=conc,
        defects=graded,
        vision=vr,
    )


def apply_anomaly(res: InspectionResult, img: np.ndarray, model) -> InspectionResult:
    """Hybrid rule: the one-class model may only *add* an NG for an unexplained anomaly.

    If the classical detector already found a defect (even an acceptable minor one) the
    high score is explained by it and the grading decision stands. An anomaly on a part
    where nothing was found is unknown territory -> NG with reason `anomaly_score`.
    """
    flagged, score = model.predict(img)
    res.anomaly_score = round(float(score), 3)
    if flagged and res.verdict == "PASS" and not res.defects:
        res.verdict = "NG"
        res.reasons = res.reasons + ["anomaly_score"]
    return res


def inspect_array(img: np.ndarray, serial: str, spec: PartSpec, model=None) -> InspectionResult:
    res = decide(analyze(img, spec), spec, serial)
    return apply_anomaly(res, img, model) if model is not None else res


def inspect_file(path: str | Path, spec: PartSpec, model=None) -> InspectionResult:
    path = Path(path)
    img = np.asarray(Image.open(path).convert("L"))
    return inspect_array(img, path.stem, spec, model)

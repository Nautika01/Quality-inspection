"""Turn vision measurements into a traceable PASS / NG / ERROR decision per serial number.

Verdicts
- PASS  : all toleranced characteristics inside their guard-banded limits, no rejecting defect
- NG    : at least one rejecting reason (reason codes in reasons.py)
- ERROR : no valid measurement (empty nest, clipped part, bad image). Not a part defect:
          excluded from yield and SPC, the station should re-trigger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .spec import PartSpec
from .util import utc_now
from .vision import Defect, VisionResult, analyze, to_gray8

VERDICT_PASS, VERDICT_NG, VERDICT_ERROR = "PASS", "NG", "ERROR"


@dataclass
class GradedDefect:
    defect: Defect
    grade: str  # critical / major / minor
    rejecting: bool = False


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
    anomaly_score: float | None = None  # filled by the one-class model when enabled
    anomaly_error: str | None = None
    error_detail: str = ""

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
            "error_detail": self.error_detail,
        }


def _finite(*xs: float) -> bool:
    return all(x is not None and math.isfinite(x) for x in xs)


def decide(vr: VisionResult, spec: PartSpec, serial: str, now: str | None = None) -> InspectionResult:
    """Apply tolerances (with guard bands) and defect grading to a vision result."""
    now = now or utc_now()
    if vr.geometry is None:
        return InspectionResult(serial, now, VERDICT_ERROR, [f"vision:{vr.error or 'internal_error'}"],
                                vision=vr, error_detail=vr.error_detail)

    g, ppm = vr.geometry, spec.px_per_mm
    od = 2 * g.outer.r / ppm
    pk = conc = None
    if g.pocket is not None:
        pk = 2 * g.pocket.r / ppm
        conc = float(np.hypot(g.outer.cx - g.pocket.cx, g.outer.cy - g.pocket.cy)) / ppm
    if not _finite(od) or (g.pocket is not None and not _finite(pk, conc)):
        return InspectionResult(serial, now, VERDICT_ERROR, ["vision:degenerate_geometry"], vision=vr)

    reasons: list[str] = []
    # compare the unrounded values; round only for storage / display
    if not spec.outer_diameter.ok(od, spec.guard_band_for(spec.outer_diameter)):
        reasons.append("od_out_of_tolerance")
    if spec.has_pocket and g.pocket is not None:
        if not spec.pocket_diameter.ok(pk, spec.guard_band_for(spec.pocket_diameter)):
            reasons.append("pocket_out_of_tolerance")
        if conc > spec.max_concentricity_mm - spec.guard_band_mm:
            reasons.append("concentricity")

    graded = []
    for d in vr.defects:
        grade = spec.grading.grade(d.kind, d.length_mm)
        graded.append(GradedDefect(d, grade, spec.grading.rejects(d.kind, grade)))
    for gd in sorted(graded, key=lambda x: (x.grade, x.defect.kind)):
        if gd.rejecting:
            tag = f"surface_{gd.defect.kind}:{gd.grade}"
            if tag not in reasons:
                reasons.append(tag)

    return InspectionResult(
        serial=serial,
        inspected_at=now,
        verdict=VERDICT_NG if reasons else VERDICT_PASS,
        reasons=reasons,
        od_mm=round(od, 4),
        pocket_mm=round(pk, 4) if pk is not None else None,
        concentricity_mm=round(conc, 4) if conc is not None else None,
        defects=graded,
        vision=vr,
    )


def apply_anomaly(res: InspectionResult, img: np.ndarray, model) -> InspectionResult:
    """Hybrid rule: the one-class model may only *add* an NG for an unexplained anomaly.

    If the classical detector already found a defect (even an acceptable minor one) the
    high score is explained by it and the grading decision stands. An anomaly on a part
    where nothing was found is unknown territory -> NG with reason `anomaly_score`.
    A frame without valid geometry (verdict ERROR) is not scored. A model failure never
    changes the verdict; it is recorded in `anomaly_error`.
    """
    if res.verdict == VERDICT_ERROR or res.vision is None or res.vision.geometry is None:
        return res
    try:
        flagged, score = model.predict(img)
        # contract: predict(img) -> (bool, float); accept numpy scalars / 1-element arrays, reject anything else
        score = float(np.asarray(score, dtype=float).reshape(()))
        flagged = bool(np.asarray(flagged).reshape(()))
    except Exception as exc:  # noqa: BLE001 - a model failure must not take the line down
        res.anomaly_error = f"{type(exc).__name__}: {exc}"[:200]
        return res
    if not math.isfinite(score):
        res.anomaly_error = "non-finite score"
        return res
    res.anomaly_score = round(score, 3)
    if flagged and res.verdict == VERDICT_PASS and not res.defects:
        res.verdict = VERDICT_NG
        res.reasons = res.reasons + ["anomaly_score"]
    return res


def inspect_array(img: np.ndarray, serial: str, spec: PartSpec, model=None) -> InspectionResult:
    try:
        img = to_gray8(img)
    except Exception as exc:  # noqa: BLE001
        return InspectionResult(serial, utc_now(), VERDICT_ERROR, ["vision:bad_image"], error_detail=str(exc)[:200])
    res = decide(analyze(img, spec), spec, serial)
    return apply_anomaly(res, img, model) if model is not None else res


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as uint8 grayscale, handling 16-bit / RGB / EXIF-rotated inputs."""
    from .datasets import load_gray

    return load_gray(path)


def inspect_file(path: str | Path, spec: PartSpec, model=None) -> InspectionResult:
    path = Path(path)
    try:
        img = load_image(path)
    except Exception as exc:  # noqa: BLE001
        return InspectionResult(path.stem, utc_now(), VERDICT_ERROR, ["vision:bad_image"], error_detail=str(exc)[:200])
    return inspect_array(img, path.stem, spec, model)

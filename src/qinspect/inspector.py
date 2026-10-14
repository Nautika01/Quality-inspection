"""Turn vision measurements into a traceable PASS/NG decision per serial number."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .spec import PartSpec
from .vision import Defect, VisionResult, analyze


@dataclass
class InspectionResult:
    serial: str
    inspected_at: str
    verdict: str
    reasons: list[str]
    od_mm: float | None = None
    pocket_mm: float | None = None
    concentricity_mm: float | None = None
    defects: list[Defect] = field(default_factory=list)
    vision: VisionResult | None = None

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
            "defect_types": ";".join(sorted({d.kind for d in self.defects})),
            "max_defect_len_mm": max((d.length_mm for d in self.defects), default=0.0),
        }


def inspect_array(img: np.ndarray, serial: str, spec: PartSpec) -> InspectionResult:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    vr = analyze(img, spec)
    if vr.geometry is None:
        return InspectionResult(serial, now, "NG", [f"vision:{vr.error}"], vision=vr)

    g, ppm = vr.geometry, spec.px_per_mm
    od = round(2 * g.outer.r / ppm, 4)
    pk = round(2 * g.pocket.r / ppm, 4)
    conc = round(float(np.hypot(g.outer.cx - g.pocket.cx, g.outer.cy - g.pocket.cy)) / ppm, 4)

    reasons = []
    if not spec.outer_diameter.ok(od):
        reasons.append("od_out_of_tolerance")
    if not spec.pocket_diameter.ok(pk):
        reasons.append("pocket_out_of_tolerance")
    if conc > spec.max_concentricity_mm:
        reasons.append("concentricity")
    for kind in sorted({d.kind for d in vr.defects}):
        reasons.append(f"surface_{kind}")

    return InspectionResult(
        serial=serial,
        inspected_at=now,
        verdict="NG" if reasons else "PASS",
        reasons=reasons,
        od_mm=od,
        pocket_mm=pk,
        concentricity_mm=conc,
        defects=vr.defects,
        vision=vr,
    )


def inspect_file(path: str | Path, spec: PartSpec) -> InspectionResult:
    path = Path(path)
    img = np.asarray(Image.open(path).convert("L"))
    return inspect_array(img, path.stem, spec)

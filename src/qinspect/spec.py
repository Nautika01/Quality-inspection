"""Part specification (drawing tolerances) used by both the generator and the inspector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DimensionSpec:
    """One toleranced dimension in millimetres."""

    nominal: float
    lsl: float
    usl: float

    def ok(self, value: float, guard_band: float = 0.0) -> bool:
        """True when `value` is inside the tolerance shrunk by `guard_band` on each side.

        A guard band equal to the measurement uncertainty turns borderline
        measurements into NG instead of letting them escape (ISO 14253-1 idea).
        """
        return self.lsl + guard_band <= value <= self.usl - guard_band


@dataclass(frozen=True)
class DefectGrading:
    """Severity classes for surface defects, by defect length in mm.

    critical: always reject (safety-relevant), major: reject, minor: accept but record.
    Values are illustrative; a real inspection standard defines these per drawing.
    """

    critical_len_mm: float = 1.0
    major_len_mm: float = 0.5
    critical_types: tuple[str, ...] = ("crack",)
    reject_grades: tuple[str, ...] = ("critical", "major")

    def grade(self, kind: str, length_mm: float) -> str:
        if kind in self.critical_types or length_mm >= self.critical_len_mm:
            return "critical"
        if length_mm >= self.major_len_mm:
            return "major"
        return "minor"


@dataclass(frozen=True)
class PartSpec:
    """Example part: the head (bottom face) of a cartridge case, seen from above.

    Values are illustrative only and do not come from any real drawing.
    """

    name: str = "cartridge-case-head (example)"
    part_number: str = "EX-CASE-HEAD-001"
    revision: str = "A"
    px_per_mm: float = 16.0
    image_size: int = 256
    outer_diameter: DimensionSpec = field(default_factory=lambda: DimensionSpec(12.0, 11.90, 12.10))
    pocket_diameter: DimensionSpec = field(default_factory=lambda: DimensionSpec(5.0, 4.92, 5.08))
    max_concentricity_mm: float = 0.10
    min_defect_area_px: int = 12
    # guard band applied to every dimensional decision (~2 sigma of the measurement error)
    guard_band_mm: float = 0.006
    grading: DefectGrading = field(default_factory=DefectGrading)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: dict) -> "PartSpec":
        raw = dict(raw)
        for key in ("outer_diameter", "pocket_diameter"):
            if isinstance(raw.get(key), dict):
                raw[key] = DimensionSpec(**raw[key])
        if isinstance(raw.get("grading"), dict):
            g = dict(raw["grading"])
            for k in ("critical_types", "reject_grades"):
                if k in g:
                    g[k] = tuple(g[k])
            raw["grading"] = DefectGrading(**g)
        return cls(**raw)

    @classmethod
    def from_json(cls, path: str | Path) -> "PartSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

"""Part specification (drawing tolerances) used by both the generator and the inspector."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DimensionSpec:
    """One toleranced dimension in millimetres.

    `guard_band_mm` overrides the part-level default; derive it from the measurement
    system (about 2 x the Gage R&R sigma of this characteristic).
    """

    nominal: float
    lsl: float
    usl: float
    guard_band_mm: float | None = None

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
    `non_rejecting_types` are recorded but never cause an NG by themselves (e.g. glare).
    Values are illustrative; a real inspection standard defines these per drawing.
    """

    critical_len_mm: float = 1.0
    major_len_mm: float = 0.5
    critical_types: tuple[str, ...] = ("crack",)
    reject_grades: tuple[str, ...] = ("critical", "major")
    non_rejecting_types: tuple[str, ...] = ("bright_spot",)

    def grade(self, kind: str, length_mm: float) -> str:
        if kind in self.critical_types:
            return "critical"
        if kind in self.non_rejecting_types:
            return "minor"
        if length_mm >= self.critical_len_mm:
            return "critical"
        if length_mm >= self.major_len_mm:
            return "major"
        return "minor"

    def rejects(self, kind: str, grade: str) -> bool:
        return grade in self.reject_grades and kind not in self.non_rejecting_types


@dataclass(frozen=True)
class PartSpec:
    """Example part: the head (bottom face) of a cartridge case, seen from above.

    Values are illustrative only and do not come from any real drawing.
    `image_size` is only the render size of the synthetic generator; the inspector
    accepts any image size and relies on `px_per_mm` alone.
    """

    name: str = "cartridge-case-head (example)"
    part_number: str = "EX-CASE-HEAD-001"
    revision: str = "A"
    px_per_mm: float = 16.0
    image_size: int = 256
    # guard bands = 2 x Gage R&R sigma of each characteristic (see `qinspect gage`), rounded up
    outer_diameter: DimensionSpec = field(default_factory=lambda: DimensionSpec(12.0, 11.90, 12.10, guard_band_mm=0.008))
    # None = part without a recess/pocket (plain disk, washer with through-hole handled as pocket)
    pocket_diameter: DimensionSpec | None = field(
        default_factory=lambda: DimensionSpec(5.0, 4.92, 5.08, guard_band_mm=0.012)
    )
    max_concentricity_mm: float = 0.10
    min_defect_area_px: int = 12
    # default guard band for characteristics without their own (also used for concentricity)
    guard_band_mm: float = 0.006
    grading: DefectGrading = field(default_factory=DefectGrading)
    schema_version: int = SCHEMA_VERSION

    @property
    def has_pocket(self) -> bool:
        return self.pocket_diameter is not None

    def guard_band_for(self, dim: DimensionSpec | None) -> float:
        if dim is not None and dim.guard_band_mm is not None:
            return dim.guard_band_mm
        return self.guard_band_mm

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: dict) -> "PartSpec":
        raw = _known_fields(cls, dict(raw), "PartSpec")
        for key in ("outer_diameter", "pocket_diameter"):
            v = raw.get(key)
            if isinstance(v, dict):
                raw[key] = DimensionSpec(**_known_fields(DimensionSpec, v, key))
            elif v is not None and not isinstance(v, DimensionSpec):
                raise ValueError(f"{key} must be an object or null")
        if isinstance(raw.get("grading"), dict):
            g = _known_fields(DefectGrading, raw["grading"], "grading")
            for k in ("critical_types", "reject_grades", "non_rejecting_types"):
                if k in g:
                    g[k] = _as_str_tuple(g[k], k)
            raw["grading"] = DefectGrading(**g)
        spec = cls(**raw)
        if spec.px_per_mm <= 0:
            raise ValueError("px_per_mm must be positive")
        for name, d in (("outer_diameter", spec.outer_diameter), ("pocket_diameter", spec.pocket_diameter)):
            if d is not None and not (d.lsl <= d.nominal <= d.usl):
                raise ValueError(f"{name}: need lsl <= nominal <= usl")
        return spec

    @classmethod
    def from_json(cls, path: str | Path) -> "PartSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _known_fields(cls, raw: dict, where: str) -> dict:
    names = {f.name for f in fields(cls)}
    unknown = sorted(k for k in raw if k not in names)
    if unknown:
        warnings.warn(f"{where}: ignoring unknown keys {unknown}", stacklevel=3)
    return {k: v for k, v in raw.items() if k in names}


def _as_str_tuple(v, name: str) -> tuple[str, ...]:
    if isinstance(v, str):
        return (v,)
    if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
        return tuple(v)
    raise ValueError(f"grading.{name} must be a string or a list of strings")

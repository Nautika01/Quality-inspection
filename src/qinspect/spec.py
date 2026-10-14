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

    def ok(self, value: float) -> bool:
        return self.lsl <= value <= self.usl


@dataclass(frozen=True)
class PartSpec:
    """Example part: the head (bottom face) of a cartridge case, seen from above.

    Values are illustrative only and do not come from any real drawing.
    """

    name: str = "cartridge-case-head (example)"
    px_per_mm: float = 16.0
    image_size: int = 256
    outer_diameter: DimensionSpec = field(default_factory=lambda: DimensionSpec(12.0, 11.90, 12.10))
    pocket_diameter: DimensionSpec = field(default_factory=lambda: DimensionSpec(5.0, 4.92, 5.08))
    max_concentricity_mm: float = 0.10
    min_defect_area_px: int = 12

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "PartSpec":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("outer_diameter", "pocket_diameter"):
            if key in raw:
                raw[key] = DimensionSpec(**raw[key])
        return cls(**raw)

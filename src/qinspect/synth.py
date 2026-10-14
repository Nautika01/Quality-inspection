"""Synthetic image generator for the example part.

Real defense-part images are usually restricted, so the MVP starts with rendered
images whose ground truth (true dimensions, injected defects) is known exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .spec import PartSpec

BACKGROUND_LEVEL = 30.0
RING_LEVEL = 180.0
POCKET_LEVEL = 60.0
DEFECT_TYPES = ("scratch", "dent")


@dataclass
class PartTruth:
    serial: str
    od_mm: float
    pocket_mm: float
    concentricity_mm: float
    defects: list[str] = field(default_factory=list)

    def expected_verdict(self, spec: PartSpec) -> str:
        ok = (
            not self.defects
            and spec.outer_diameter.ok(self.od_mm)
            and spec.pocket_diameter.ok(self.pocket_mm)
            and self.concentricity_mm <= spec.max_concentricity_mm
        )
        return "PASS" if ok else "NG"


def _scratch_map(rng, xx, yy, cx, cy, r_lo, r_hi):
    """Thin dark line fully inside the annulus [r_lo, r_hi]."""
    for _ in range(200):
        rho = rng.uniform(r_lo, r_hi)
        phi = rng.uniform(0, 2 * np.pi)
        x0, y0 = cx + rho * np.cos(phi), cy + rho * np.sin(phi)
        theta = rng.uniform(0, np.pi)
        half = rng.uniform(8, 20)
        dx, dy = np.cos(theta) * half, np.sin(theta) * half
        ts = np.linspace(-1, 1, 21)
        rr = np.hypot(x0 + ts * dx - cx, y0 + ts * dy - cy)
        if rr.min() >= r_lo and rr.max() <= r_hi:
            break
    else:  # pragma: no cover - practically unreachable
        raise RuntimeError("could not place scratch")
    ax, ay, bx, by = x0 - dx, y0 - dy, x0 + dx, y0 + dy
    vx, vy = bx - ax, by - ay
    t = np.clip(((xx - ax) * vx + (yy - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    d = np.hypot(xx - (ax + t * vx), yy - (ay + t * vy))
    width = rng.uniform(0.9, 1.4)
    depth = rng.uniform(28, 42)
    return depth * np.exp(-((d / width) ** 2))


def _dent_map(rng, xx, yy, cx, cy, r_lo, r_hi):
    """Round dark pit (gaussian) inside the annulus."""
    sigma = rng.uniform(2.0, 3.0)
    rho = rng.uniform(r_lo + 3 * sigma, r_hi - 3 * sigma)
    phi = rng.uniform(0, 2 * np.pi)
    x0, y0 = cx + rho * np.cos(phi), cy + rho * np.sin(phi)
    depth = rng.uniform(28, 45)
    return depth * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))


def render_part(
    spec: PartSpec,
    rng: np.random.Generator,
    *,
    od_mm: float,
    pocket_mm: float,
    pocket_offset_mm: tuple[float, float] = (0.0, 0.0),
    defects: tuple[str, ...] = (),
    noise_sigma: float = 3.0,
) -> np.ndarray:
    """Render one top-view grayscale image (uint8) of the part."""
    size = spec.image_size
    ppm = spec.px_per_mm
    yy, xx = np.mgrid[0:size, 0:size].astype(float)

    cx = (size - 1) / 2 + rng.uniform(-3, 3)
    cy = (size - 1) / 2 + rng.uniform(-3, 3)
    px = cx + pocket_offset_mm[0] * ppm
    py = cy + pocket_offset_mm[1] * ppm
    r_out = od_mm * ppm / 2
    r_in = pocket_mm * ppm / 2

    r = np.hypot(xx - cx, yy - cy)
    rp = np.hypot(xx - px, yy - py)
    a_out = np.clip(r_out - r + 0.5, 0, 1)  # anti-aliased coverage
    a_in = np.clip(r_in - rp + 0.5, 0, 1)

    # metal face: radial shading + concentric machining marks
    ring = RING_LEVEL + 10 * (r / r_out) + 4 * np.sin(2 * np.pi * r / 6.0 + rng.uniform(0, 2 * np.pi))
    r_lo, r_hi = r_in + 6 + np.hypot(px - cx, py - cy), r_out - 6
    for kind in defects:
        if kind == "scratch":
            ring = ring - _scratch_map(rng, xx, yy, cx, cy, r_lo, r_hi)
        elif kind == "dent":
            ring = ring - _dent_map(rng, xx, yy, cx, cy, r_lo, r_hi)
        else:
            raise ValueError(f"unknown defect type: {kind}")

    img = BACKGROUND_LEVEL * (1 - a_out) + a_out * (ring * (1 - a_in) + POCKET_LEVEL * a_in)

    # uneven illumination + sensor noise
    gx, gy = rng.uniform(-10, 10, size=2)
    img = img + gx * (xx - cx) / size + gy * (yy - cy) / size
    img = img + rng.normal(0, noise_sigma, size=img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def generate_lot(
    spec: PartSpec,
    n: int,
    out_dir: str | Path,
    *,
    seed: int = 0,
    lot_id: str = "LOT-0001",
    defect_rate: float = 0.12,
    dim_ng_rate: float = 0.04,
    drift_start: float | None = 0.6,
    drift_mm_per_part: float = 0.0012,
) -> pd.DataFrame:
    """Generate `n` part images plus `ground_truth.csv` in `out_dir`.

    `drift_*` simulates tool wear: after `drift_start * n` parts the outer
    diameter mean creeps upward, which the SPC charts should catch.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    od, pk = spec.outer_diameter, spec.pocket_diameter
    rows = []
    for i in range(n):
        serial = f"{lot_id}-{i + 1:05d}"
        od_mm = rng.normal(od.nominal, 0.02)
        pocket_mm = rng.normal(pk.nominal, 0.012)
        offset = rng.normal(0, 0.015, size=2)

        if drift_start is not None and i >= drift_start * n:
            od_mm += drift_mm_per_part * (i - drift_start * n)

        if rng.random() < dim_ng_rate:
            which = rng.choice(["od", "pocket", "concentricity"])
            sign = rng.choice([-1, 1])
            if which == "od":
                od_mm = od.nominal + sign * rng.uniform(0.15, 0.25)
            elif which == "pocket":
                pocket_mm = pk.nominal + sign * rng.uniform(0.13, 0.2)
            else:
                ang = rng.uniform(0, 2 * np.pi)
                mag = rng.uniform(0.16, 0.3)
                offset = np.array([np.cos(ang), np.sin(ang)]) * mag

        defects: tuple[str, ...] = ()
        if rng.random() < defect_rate:
            defects = (str(rng.choice(DEFECT_TYPES)),)

        img = render_part(
            spec, rng, od_mm=od_mm, pocket_mm=pocket_mm, pocket_offset_mm=tuple(offset), defects=defects
        )
        Image.fromarray(img).save(img_dir / f"{serial}.png")

        truth = PartTruth(serial, od_mm, pocket_mm, float(np.hypot(*offset)), list(defects))
        rows.append(
            {
                "serial": serial,
                "lot_id": lot_id,
                "seq": i + 1,
                "od_mm": round(truth.od_mm, 5),
                "pocket_mm": round(truth.pocket_mm, 5),
                "concentricity_mm": round(truth.concentricity_mm, 5),
                "defect_type": defects[0] if defects else "none",
                "expected_verdict": truth.expected_verdict(spec),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "ground_truth.csv", index=False)
    spec.to_json(out / "part_spec.json")
    return df

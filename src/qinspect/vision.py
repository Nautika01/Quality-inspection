"""Classical machine-vision inspection: sub-pixel dimensions and surface defects.

Pipeline per image
1. Segment the part from the background and the primer pocket from the face.
2. Measure both circles with sub-pixel area estimation (edge pixels count by
   their gray-level fraction), giving diameters and concentricity.
3. Model the defect-free face (radial profile + illumination plane), take the
   residual, and extract anomalies with hysteresis thresholding.
4. Classify each anomaly by shape: elongated -> scratch, compact -> dent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .spec import PartSpec

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class Circle:
    cx: float
    cy: float
    r: float


@dataclass
class Defect:
    kind: str
    area_px: int
    length_mm: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)
    peak: float


@dataclass
class Geometry:
    outer: Circle
    pocket: Circle


@dataclass
class VisionResult:
    geometry: Geometry | None
    defects: list[Defect] = field(default_factory=list)
    error: str | None = None


def _subpixel_disk(img: np.ndarray, region: np.ndarray, inside_level: float, outside_level: float) -> Circle:
    """Sub-pixel circle from a binary region using gray-level coverage at its edge."""
    interior = ndi.binary_erosion(region, iterations=2)
    band = ndi.binary_dilation(region, iterations=2) & ~interior
    frac = np.clip((img - outside_level) / (inside_level - outside_level), 0, 1)
    w = interior.astype(float)
    w[band] = frac[band]
    area = w.sum()
    yy, xx = np.indices(img.shape)
    return Circle(cx=float((w * xx).sum() / area), cy=float((w * yy).sum() / area), r=float(np.sqrt(area / np.pi)))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(mask, structure=_EIGHT)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def measure_geometry(img: np.ndarray) -> Geometry:
    img = img.astype(float)
    # Otsu-like split between dark background and bright face
    thr_part = 0.5 * (np.percentile(img, 5) + np.percentile(img, 95))
    part = ndi.binary_fill_holes(_largest_component(img > thr_part))
    if part.sum() < 100:
        raise ValueError("part not found")

    bg = ~ndi.binary_dilation(part, iterations=4)
    edge_in = part & ~ndi.binary_erosion(part, iterations=6)
    edge_in &= ndi.binary_erosion(part, iterations=2)
    bg_level = float(np.median(img[bg]))
    rim_level = float(np.median(img[edge_in & (img > thr_part)]))
    outer = _subpixel_disk(img, part, rim_level, bg_level)

    # pocket = dark component inside the face that contains the part centre
    face = ndi.binary_erosion(part, iterations=3)
    face_level = float(np.median(img[face & (img > thr_part)]))
    dark = face & (img < thr_part)
    labels, n = ndi.label(dark, structure=_EIGHT)
    ci, cj = int(round(outer.cy)), int(round(outer.cx))
    lab = labels[ci, cj] if n else 0
    if lab == 0:
        raise ValueError("pocket not found")
    pocket = ndi.binary_fill_holes(labels == lab)
    pocket_level = float(np.median(img[ndi.binary_erosion(pocket, iterations=3)]))
    around = ndi.binary_dilation(pocket, iterations=5) & ~ndi.binary_dilation(pocket, iterations=2)
    inner_ring_level = float(np.median(img[around])) if around.any() else face_level
    # pocket is the dark side, so swap the roles of inside/outside levels
    pk = _subpixel_disk(img, pocket, pocket_level, inner_ring_level)
    return Geometry(outer=outer, pocket=pk)


def _radial_model(values: np.ndarray, radius: np.ndarray, region: np.ndarray, bin_px: float = 0.25) -> np.ndarray:
    """Median intensity per radius bin, interpolated back onto every pixel."""
    rv, vv = radius[region], values[region]
    bins = np.floor(rv / bin_px).astype(int)
    order = np.argsort(bins, kind="stable")
    bins, vv, rv = bins[order], vv[order], rv[order]
    uniq, start = np.unique(bins, return_index=True)
    groups_v = np.split(vv, start[1:])
    groups_r = np.split(rv, start[1:])
    centers = np.array([g.mean() for g in groups_r])
    medians = np.array([np.median(g) for g in groups_v])
    return np.interp(radius, centers, medians)


def _fit_plane(values: np.ndarray, region: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(values.shape)
    A = np.column_stack([np.ones(region.sum()), xx[region], yy[region]])
    coef, *_ = np.linalg.lstsq(A, values[region], rcond=None)
    return coef[0] + coef[1] * xx + coef[2] * yy


def detect_defects(
    img: np.ndarray,
    geom: Geometry,
    spec: PartSpec,
    *,
    k_sigma: float = 5.0,
    min_abs: float = 6.0,
    margin_px: float = 3.0,
) -> list[Defect]:
    img = img.astype(float)
    yy, xx = np.indices(img.shape)
    o, p = geom.outer, geom.pocket
    r = np.hypot(xx - o.cx, yy - o.cy)
    rp = np.hypot(xx - p.cx, yy - p.cy)
    region = (r < o.r - margin_px) & (rp > p.r + margin_px)
    if region.sum() < 50:
        return []

    # defect-free model: radial profile, then illumination plane, then profile again
    res = img - _radial_model(img, r, region)
    res = res - _fit_plane(res, region)
    res = res - _radial_model(res, r, region)

    # normalised gaussian smoothing restricted to the face
    m = region.astype(float)
    num = ndi.gaussian_filter(np.where(region, res, 0.0), 1.0)
    den = ndi.gaussian_filter(m, 1.0)
    sm = np.where(region, num / np.maximum(den, 1e-6), 0.0)

    vals = sm[region]
    sigma = 1.4826 * float(np.median(np.abs(vals - np.median(vals))))
    hi = max(k_sigma * sigma, min_abs)
    lo = 0.5 * hi
    strong = region & (np.abs(sm) > hi)
    weak = region & (np.abs(sm) > lo)

    labels, n = ndi.label(weak, structure=_EIGHT)
    if n == 0:
        return []
    keep = np.unique(labels[strong])
    keep = keep[keep > 0]

    defects: list[Defect] = []
    for lab in keep:
        ys, xs = np.nonzero(labels == lab)
        area = len(xs)
        if area < spec.min_defect_area_px:
            continue
        cov = np.cov(np.vstack([xs, ys])) if area > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(ev[0] / max(ev[1], 1e-6)))
        kind = "scratch" if elong >= 3.0 else "dent"
        length_px = 4.0 * np.sqrt(max(ev[0], 0.0))  # ~full length of a uniform segment is sqrt(12*var)
        defects.append(
            Defect(
                kind=kind,
                area_px=int(area),
                length_mm=round(float(length_px / spec.px_per_mm), 3),
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                peak=round(float(np.abs(sm[ys, xs]).max()), 1),
            )
        )
    return defects


def analyze(img: np.ndarray, spec: PartSpec) -> VisionResult:
    try:
        geom = measure_geometry(img)
    except ValueError as exc:
        return VisionResult(geometry=None, error=str(exc))
    return VisionResult(geometry=geom, defects=detect_defects(img, geom, spec))

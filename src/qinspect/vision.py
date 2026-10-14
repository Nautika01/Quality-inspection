"""Classical machine-vision inspection: sub-pixel dimensions and surface defects.

Pipeline per image
1. Normalise the frame to 8-bit grayscale (RGB, 16-bit and float inputs accepted).
2. Segment the part from the background (Otsu, polarity-agnostic) and validate it
   (contrast, not clipped by the frame border).
3. Measure the outer circle, and the pocket if the part has one, with sub-pixel area
   estimation: edge pixels count by their gray-level fraction, using *local* (per
   angular sector) inside/outside levels so illumination gradients cancel.
4. Model the defect-free face (radial profile + illumination plane), take the
   residual, and extract anomalies with hysteresis thresholding.
5. Classify each anomaly: dark + elongated -> scratch, dark + compact -> dent,
   bright -> bright_spot (glare / debris; never rejected by default).

Every failure that means "no valid measurement" raises VisionError with a stable
slug (see reasons.py); `analyze()` turns it into VisionResult.error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .spec import PartSpec

_EIGHT = np.ones((3, 3), dtype=bool)
REFERENCE_PX_PER_MM = 16.0  # scale at which the pixel constants below were tuned
N_SECTORS = 16


class VisionError(ValueError):
    """Measurement impossible; `slug` is a code from reasons.py (without the 'vision:' prefix)."""

    def __init__(self, slug: str, detail: str = ""):
        super().__init__(slug if not detail else f"{slug}: {detail}")
        self.slug = slug
        self.detail = detail


@dataclass
class Circle:
    cx: float
    cy: float
    r: float

    def finite(self) -> bool:
        return bool(np.isfinite([self.cx, self.cy, self.r]).all())


@dataclass
class Defect:
    kind: str
    area_px: int
    length_mm: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)
    peak: float
    polarity: str = "dark"  # dark = material removed, bright = glare / debris


@dataclass
class Geometry:
    outer: Circle
    pocket: Circle | None = None


@dataclass
class VisionResult:
    geometry: Geometry | None
    defects: list[Defect] = field(default_factory=list)
    error: str | None = None  # slug from reasons.py (without prefix)
    error_detail: str = ""


# --------------------------------------------------------------------------- input


def to_gray8(img: np.ndarray) -> np.ndarray:
    """Normalise any reasonable camera frame to uint8 grayscale (H, W)."""
    a = np.asarray(img)
    if a.ndim == 3:
        if a.shape[2] in (3, 4):
            a = a[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        elif a.shape[2] == 1:
            a = a[..., 0]
        else:
            raise VisionError("bad_image", f"unsupported channel count {a.shape[2]}")
    if a.ndim != 2:
        raise VisionError("bad_image", f"expected 2-D image, got shape {a.shape}")
    if a.size == 0:
        raise VisionError("bad_image", "empty image")
    if a.dtype == np.uint8:
        return a
    src = a.dtype
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise VisionError("bad_image", "non-finite pixel values")
    if src.kind in "fc":
        peak = a.max()
        if peak <= 1.0:  # floats in [0, 1]
            a = a * 255.0
        elif peak > 255.0:
            a = a * 255.0 / peak
    elif src.kind in "iu" and src.itemsize > 1:  # 10/12/14/16-bit data in wider integer containers
        peak = a.max()
        depth_max = next((m for m in (255.0, 1023.0, 4095.0, 16383.0, 65535.0) if peak <= m), peak)
        a = a * 255.0 / depth_max
    elif src.kind == "b":
        a = a * 255.0
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- helpers


def otsu_threshold(values: np.ndarray) -> float:
    """Classic Otsu split on an 8-bit intensity sample."""
    v = np.asarray(values).ravel()
    if v.size == 0:
        raise VisionError("bad_image", "empty region")
    hist = np.bincount(np.clip(v, 0, 255).astype(np.intp), minlength=256).astype(float)
    total = hist.sum()
    levels = np.arange(256, dtype=float)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels) / np.maximum(w0, 1e-12)
    m1 = (hist * levels).sum() - np.cumsum(hist * levels)
    m1 = m1 / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    between[(w0 == 0) | (w1 == 0)] = -1
    return float(np.argmax(between)) + 0.5


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(mask, structure=_EIGHT)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _touches_border(mask: np.ndarray, margin: int = 2) -> bool:
    return bool(mask[:margin].any() or mask[-margin:].any() or mask[:, :margin].any() or mask[:, -margin:].any())


def _sector_levels(img: np.ndarray, region: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Median intensity of `region` in each of N_SECTORS angular sectors around (cx, cy).

    Empty sectors are filled from their neighbours.
    """
    yy, xx = np.nonzero(region)
    ang = np.arctan2(yy - cy, xx - cx)
    sector = ((ang + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    vals = img[yy, xx]
    out = np.full(N_SECTORS, np.nan)
    for s in range(N_SECTORS):
        sel = sector == s
        if sel.sum() >= 3:
            out[s] = np.median(vals[sel])
    if np.isnan(out).all():
        raise VisionError("degenerate_geometry", "no edge pixels")
    if np.isnan(out).any():  # circular nearest fill
        idx = np.arange(N_SECTORS)
        good = ~np.isnan(out)
        for s in idx[~good]:
            d = np.minimum(np.abs(idx[good] - s), N_SECTORS - np.abs(idx[good] - s))
            out[s] = out[good][np.argmin(d)]
    return out


def _subpixel_disk(img: np.ndarray, region: np.ndarray, min_contrast: float = 12.0) -> Circle:
    """Sub-pixel circle from a binary region using gray-level coverage at its edge.

    Inside/outside reference levels are estimated per angular sector right next to
    the edge band, so a smooth illumination gradient does not bias the centroid.
    """
    if region.sum() < 20:
        raise VisionError("degenerate_geometry", "region too small")
    interior = ndi.binary_erosion(region, iterations=2)
    band = ndi.binary_dilation(region, iterations=2) & ~interior
    inner_ref = interior & ~ndi.binary_erosion(region, iterations=5)
    outer_ref = ndi.binary_dilation(region, iterations=5) & ~ndi.binary_dilation(region, iterations=2)
    if not inner_ref.any() or not outer_ref.any():
        raise VisionError("degenerate_geometry", "no reference ring")

    yy0, xx0 = np.nonzero(region)
    cx0, cy0 = float(xx0.mean()), float(yy0.mean())
    imgf = img.astype(np.float64)
    lin = _sector_levels(imgf, inner_ref, cx0, cy0)
    lout = _sector_levels(imgf, outer_ref, cx0, cy0)
    if np.abs(lin - lout).min() < min_contrast:
        raise VisionError("low_contrast", f"edge contrast {np.abs(lin - lout).min():.1f} gray levels")

    by, bx = np.nonzero(band)
    ang = np.arctan2(by - cy0, bx - cx0)
    sec = ((ang + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    frac = np.clip((imgf[by, bx] - lout[sec]) / (lin[sec] - lout[sec]), 0, 1)

    w = interior.astype(np.float64)
    w[by, bx] = frac
    area = w.sum()
    if not np.isfinite(area) or area <= 0:
        raise VisionError("degenerate_geometry", "zero area")
    yy, xx = np.indices(img.shape)
    c = Circle(cx=float((w * xx).sum() / area), cy=float((w * yy).sum() / area), r=float(np.sqrt(area / np.pi)))
    if not c.finite():
        raise VisionError("degenerate_geometry", "non-finite circle")
    return c


# --------------------------------------------------------------------------- geometry


def segment_part(img8: np.ndarray) -> tuple[np.ndarray, bool]:
    """Binary mask of the part and whether the part is brighter than the background."""
    thr = otsu_threshold(img8)
    border = np.concatenate([img8[0], img8[-1], img8[:, 0], img8[:, -1]]).astype(float)
    bright_part = np.median(border) < thr
    mask = img8 > thr if bright_part else img8 < thr
    part = ndi.binary_fill_holes(_largest_component(mask))
    if part.sum() < 100:
        raise VisionError("part_not_found", f"largest blob {int(part.sum())} px")
    frac = part.mean()
    if frac > 0.95:
        raise VisionError("part_not_found", "blob covers the whole frame")
    if _touches_border(part):
        raise VisionError("part_touches_border")
    bg = ~ndi.binary_dilation(part, iterations=4)
    if bg.sum() < 50:
        raise VisionError("part_touches_border", "no background around the part")
    contrast = abs(float(np.median(img8[part])) - float(np.median(img8[bg])))
    if contrast < 25:
        raise VisionError("low_contrast", f"part/background contrast {contrast:.0f}")
    return part, bool(bright_part)


def measure_outer(img: np.ndarray) -> tuple[Circle, np.ndarray, bool]:
    """Outer circle only (used by calibration and pocket-less parts)."""
    img8 = to_gray8(img)
    part, bright = segment_part(img8)
    return _subpixel_disk(img8, part), part, bright


def _find_pocket(img8: np.ndarray, part: np.ndarray, outer: Circle, bright_part: bool) -> Circle:
    face = ndi.binary_erosion(part, iterations=4)
    fv = img8[face]
    if fv.size < 50:
        raise VisionError("pocket_not_found", "face too small")
    thr = otsu_threshold(fv)
    # a pocket is a recess: darker than the face on a bright part, brighter on a dark part
    cand = face & ((img8 < thr) if bright_part else (img8 > thr))
    labels, n = ndi.label(cand, structure=_EIGHT)
    if n == 0:
        raise VisionError("pocket_not_found")
    ci, cj = int(round(outer.cy)), int(round(outer.cx))
    lab = labels[ci, cj] if (0 <= ci < labels.shape[0] and 0 <= cj < labels.shape[1]) else 0
    if lab == 0:  # pocket may be off-centre: take the component whose centroid is nearest the centre
        cents = np.array(ndi.center_of_mass(cand, labels, index=np.arange(1, n + 1)))
        sizes = ndi.sum(cand, labels, index=np.arange(1, n + 1))
        ok = sizes >= 30
        if not ok.any():
            raise VisionError("pocket_not_found", "no component >= 30 px")
        d = np.hypot(cents[:, 1] - outer.cx, cents[:, 0] - outer.cy)
        d[~ok] = np.inf
        lab = int(np.argmin(d)) + 1
    pocket = ndi.binary_fill_holes(labels == lab)
    if pocket.sum() < 30:
        raise VisionError("pocket_not_found", f"component {int(pocket.sum())} px")
    face_level = float(np.median(img8[face & ~ndi.binary_dilation(pocket, iterations=6)]))
    pocket_level = float(np.median(img8[pocket]))
    if abs(face_level - pocket_level) < 15:
        raise VisionError("pocket_not_found", f"face/pocket contrast {abs(face_level - pocket_level):.0f}")
    c = _subpixel_disk(img8, pocket)
    ratio = c.r / outer.r
    if not (0.05 < ratio < 0.9):
        raise VisionError("pocket_degenerate", f"radius ratio {ratio:.2f}")
    if np.hypot(c.cx - outer.cx, c.cy - outer.cy) > 0.5 * outer.r:
        raise VisionError("pocket_degenerate", "pocket centre far from part centre")
    return c


def measure_geometry(img: np.ndarray, require_pocket: bool = True) -> Geometry:
    img8 = to_gray8(img)
    part, bright = segment_part(img8)
    outer = _subpixel_disk(img8, part)
    pocket = _find_pocket(img8, part, outer, bright) if require_pocket else None
    return Geometry(outer=outer, pocket=pocket)


# --------------------------------------------------------------------------- defects


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


def _extent_px(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """(length, elongation) of a blob: length = extent along the principal axis of the
    *weighted* pixel set, elongation = sqrt(ratio of principal variances)."""
    if len(xs) < 3:
        return float(max(xs.ptp(), ys.ptp()) + 1), 1.0
    w = weights / weights.sum()
    mx, my = (w * xs).sum(), (w * ys).sum()
    dx, dy = xs - mx, ys - my
    cov = np.array([[(w * dx * dx).sum(), (w * dx * dy).sum()], [(w * dx * dy).sum(), (w * dy * dy).sum()]])
    ev, evec = np.linalg.eigh(cov)
    major = evec[:, 1]
    proj = dx * major[0] + dy * major[1]
    elong = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
    return float(proj.max() - proj.min() + 1.0), elong


def detect_defects(
    img: np.ndarray,
    geom: Geometry,
    spec: PartSpec,
    *,
    k_sigma: float = 5.0,
    min_abs: float = 6.0,
    margin_px: float = 3.0,
) -> list[Defect]:
    img = to_gray8(img).astype(np.float64)
    scale = spec.px_per_mm / REFERENCE_PX_PER_MM  # pixel constants were tuned at 16 px/mm
    yy, xx = np.indices(img.shape)
    o, p = geom.outer, geom.pocket
    r = np.hypot(xx - o.cx, yy - o.cy)
    region = r < o.r - margin_px * scale
    if p is not None:
        rp = np.hypot(xx - p.cx, yy - p.cy)
        region &= rp > p.r + margin_px * scale
    if region.sum() < 50:
        return []

    # defect-free model: radial profile, then illumination plane, then profile again
    res = img - _radial_model(img, r, region)
    res = res - _fit_plane(res, region)
    res = res - _radial_model(res, r, region)

    # normalised gaussian smoothing restricted to the face
    sig = 1.0 * scale
    m = region.astype(float)
    num = ndi.gaussian_filter(np.where(region, res, 0.0), sig)
    den = ndi.gaussian_filter(m, sig)
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
        resp = sm[ys, xs]
        dark = float(np.sum(resp[resp < 0])) <= -float(np.sum(resp[resp > 0]))  # dominant polarity
        # length from the pixels above the strong threshold, weighted by response, so the
        # blurred halo of the weak set does not inflate the size (calibrated against ground truth)
        sel = np.abs(resp) > hi
        if sel.sum() >= 3:
            length_px, elong = _extent_px(xs[sel].astype(float), ys[sel].astype(float), np.abs(resp[sel]))
        else:
            length_px, elong = _extent_px(xs.astype(float), ys.astype(float), np.abs(resp))
        if not dark:
            kind = "bright_spot"
        else:
            kind = "scratch" if elong >= 3.0 else "dent"
        defects.append(
            Defect(
                kind=kind,
                area_px=int(area),
                length_mm=round(float(length_px / spec.px_per_mm), 3),
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                peak=round(float(np.abs(resp).max()), 1),
                polarity="dark" if dark else "bright",
            )
        )
    return defects


def analyze(img: np.ndarray, spec: PartSpec) -> VisionResult:
    try:
        geom = measure_geometry(img, require_pocket=spec.has_pocket)
        return VisionResult(geometry=geom, defects=detect_defects(img, geom, spec))
    except VisionError as exc:
        return VisionResult(geometry=None, error=exc.slug, error_detail=exc.detail)
    except Exception as exc:  # noqa: BLE001 - a frame must never crash a line
        return VisionResult(geometry=None, error="internal_error", error_detail=f"{type(exc).__name__}: {exc}")

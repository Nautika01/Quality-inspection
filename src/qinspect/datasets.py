"""Bring real images into the pipeline: import, calibrate, MVTec-style dataset loading."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .vision import measure_outer, to_gray8

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MANIFEST = "import_manifest.csv"


def _pad_square(a: np.ndarray, size: int) -> np.ndarray:
    """Letterbox to a square of `size` (never crops), padding with the border median."""
    h, w = a.shape
    s = max(h, w)
    fill = int(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))
    canvas = np.full((s, s), fill, dtype=np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = a
    if s != size:
        canvas = np.asarray(Image.fromarray(canvas).resize((size, size), Image.Resampling.LANCZOS))
    return canvas


def load_gray(path: str | Path, size: int | None = None) -> np.ndarray:
    """Read any camera file as uint8 grayscale.

    16-bit TIFF/PNG and float images are rescaled instead of clipped; EXIF rotation is
    honoured; with `size`, the frame is letterboxed (not cropped) to a square.
    """
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = to_gray8(np.asarray(im))
    else:
        a = np.asarray(im.convert("L"))
    if size:
        a = _pad_square(a, size)
    return a


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXT and not p.name.startswith("."))


def import_images(src: str | Path, dst: str | Path, *, size: int | None = None, prefix: str = "IMP") -> list[Path]:
    """Copy a folder of photos into the canonical grayscale PNG layout (`dst/images/<serial>.png`).

    Numbering continues after existing files, and `dst/import_manifest.csv` keeps
    serial -> source path + SHA-256 so a serial can always be traced to its photo.
    Already-imported sources (same hash) are skipped.
    """
    dst = Path(dst)
    out = dst / "images"
    out.mkdir(parents=True, exist_ok=True)
    manifest = dst / MANIFEST
    seen: dict[str, str] = {}
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seen[row["sha256"]] = row["serial"]
    existing = [p for p in out.glob(f"{prefix}-*.png")]
    next_idx = 1 + max((int(p.stem.rsplit("-", 1)[1]) for p in existing if p.stem.rsplit("-", 1)[1].isdigit()), default=0)

    import warnings

    written = []
    new = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["serial", "source", "sha256"])
        if new:
            w.writeheader()
            fh.flush()
        for p in list_images(src):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen:
                continue
            try:
                arr = load_gray(p, size)
            except Exception as exc:  # noqa: BLE001 - one bad photo must not abort the batch
                warnings.warn(f"import: skipping unreadable {p.name}: {exc}", stacklevel=2)
                continue
            serial = f"{prefix}-{next_idx:05d}"
            next_idx += 1
            Image.fromarray(arr).save(out / f"{serial}.png")
            # PNG first, then its manifest row, flushed at once: the two can never be separated
            w.writerow({"serial": serial, "source": str(p), "sha256": digest})
            fh.flush()
            written.append(out / f"{serial}.png")
            seen[digest] = serial
    return written


def calibrate_px_per_mm(image: np.ndarray, known_outer_diameter_mm: float) -> float:
    """Pixel scale from a reference disk/pin gauge with a known outer diameter (no pocket needed)."""
    if known_outer_diameter_mm <= 0:
        raise ValueError("known diameter must be positive")
    outer, _, _ = measure_outer(image)
    return float(2 * outer.r / known_outer_diameter_mm)


@dataclass
class AnomalyDataset:
    """MVTec-AD style layout: root/train/good, root/test/good, root/test/<defect_type>/."""

    root: Path
    train_good: list[Path]
    test_good: list[Path]
    test_bad: dict[str, list[Path]]

    @classmethod
    def from_root(cls, root: str | Path) -> "AnomalyDataset":
        root = Path(root)
        if not (root / "train" / "good").is_dir():
            raise FileNotFoundError(f"{root} has no train/good folder (MVTec-style layout expected)")
        test_bad = {}
        if (root / "test").is_dir():
            for d in sorted((root / "test").iterdir()):
                if d.is_dir() and d.name != "good" and not d.name.startswith("."):
                    imgs = list_images(d)
                    if imgs:
                        test_bad[d.name] = imgs
        test_good = list_images(root / "test" / "good") if (root / "test" / "good").is_dir() else []
        train = list_images(root / "train" / "good")
        if not train:
            raise FileNotFoundError(f"{root}/train/good contains no images")
        return cls(root, train, test_good, test_bad)


def export_synthetic_as_mvtec(truth_csv: str | Path, images_dir: str | Path, out_root: str | Path,
                              train_fraction: float = 0.6) -> AnomalyDataset:
    """Re-arrange a synthetic lot into the MVTec layout so the same evaluation code runs on both.

    'good' means defect-free *and* in tolerance; parts with an acceptable minor defect go to
    `test/<defect_type>` so the one-class model is trained on clean surfaces only.
    """
    import pandas as pd

    df = pd.read_csv(truth_csv)
    images_dir, out_root = Path(images_dir), Path(out_root)
    good = df[(df.expected_verdict == "PASS") & (df.defect_type == "none")]
    n_train = int(len(good) * train_fraction)
    plan = {"train/good": good.serial.iloc[:n_train].tolist(), "test/good": good.serial.iloc[n_train:].tolist()}
    rest = df[~df.serial.isin(good.serial)]
    for kind, grp in rest.groupby("defect_type"):
        plan[f"test/{kind if kind != 'none' else 'dimension'}"] = grp.serial.tolist()
    for rel, serials in plan.items():
        if not serials:
            continue
        d = out_root / rel
        d.mkdir(parents=True, exist_ok=True)
        for s in serials:
            Image.open(images_dir / f"{s}.png").save(d / f"{s}.png")
    return AnomalyDataset.from_root(out_root)

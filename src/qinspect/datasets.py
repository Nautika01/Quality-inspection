"""Bring real images into the pipeline: import, calibrate, MVTec-style dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .vision import measure_geometry

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_gray(path: str | Path, size: int | None = None) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    if size:
        im = ImageOps.fit(im, (size, size), Image.Resampling.LANCZOS)
    return np.asarray(im)


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def import_images(src: str | Path, dst: str | Path, *, size: int = 256, prefix: str = "IMP") -> list[Path]:
    """Copy a folder of photos into the canonical grayscale PNG layout (`dst/images/<serial>.png`)."""
    out = Path(dst) / "images"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for i, p in enumerate(list_images(src)):
        serial = f"{prefix}-{i + 1:05d}"
        Image.fromarray(load_gray(p, size)).save(out / f"{serial}.png")
        written.append(out / f"{serial}.png")
    return written


def calibrate_px_per_mm(image: np.ndarray, known_outer_diameter_mm: float) -> float:
    """Pixel scale from a reference part (gauge) with a known outer diameter."""
    g = measure_geometry(image)
    return float(2 * g.outer.r / known_outer_diameter_mm)


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
        test_bad = {
            d.name: list_images(d)
            for d in sorted((root / "test").iterdir())
            if d.is_dir() and d.name != "good"
        } if (root / "test").is_dir() else {}
        test_good = list_images(root / "test" / "good") if (root / "test" / "good").is_dir() else []
        return cls(root, list_images(root / "train" / "good"), test_good, test_bad)


def export_synthetic_as_mvtec(truth_csv: str | Path, images_dir: str | Path, out_root: str | Path,
                              train_fraction: float = 0.6) -> AnomalyDataset:
    """Re-arrange a synthetic lot into the MVTec layout so the same evaluation code runs on both."""
    import pandas as pd

    df = pd.read_csv(truth_csv)
    images_dir, out_root = Path(images_dir), Path(out_root)
    good = df[df.expected_verdict == "PASS"]
    n_train = int(len(good) * train_fraction)
    plan = {"train/good": good.serial.iloc[:n_train].tolist(), "test/good": good.serial.iloc[n_train:].tolist()}
    for kind, grp in df[df.expected_verdict == "NG"].groupby("defect_type"):
        plan[f"test/{kind if kind != 'none' else 'dimension'}"] = grp.serial.tolist()
    for rel, serials in plan.items():
        d = out_root / rel
        d.mkdir(parents=True, exist_ok=True)
        for s in serials:
            Image.open(images_dir / f"{s}.png").save(d / f"{s}.png")
    return AnomalyDataset.from_root(out_root)

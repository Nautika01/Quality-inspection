"""Image acquisition for the real-part pilot: a frame grabber abstraction and a capture
session that writes canonical PNGs plus a manifest (same layout `qinspect import` makes).

`OpenCvGrabber` needs `opencv-python-headless` (qinspect[capture]); tests use a fake grabber.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image

from .util import safe_name, utc_now
from .vision import VisionError, measure_outer, to_gray8


class OpenCvGrabber:
    def __init__(self, device: int | str = 0, width: int | None = None, height: int | None = None,
                 exposure: float | None = None, warmup: int = 5):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("capture needs `pip install opencv-python-headless` (qinspect[capture])") from exc
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {device!r} could not be opened")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual on most UVC drivers
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(warmup):  # let auto-gain settle
            self.cap.read()

    def grab(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera returned no frame")
        return to_gray8(frame[..., ::-1] if frame.ndim == 3 else frame)

    def close(self) -> None:
        self.cap.release()


def frame_quality(img: np.ndarray) -> dict:
    """Quick acquisition checks an operator can act on: exposure, focus, part found."""
    g = to_gray8(img).astype(float)
    report = {"mean": round(float(g.mean()), 1), "saturated_pct": round(float((g >= 250).mean() * 100), 2),
              "dark_pct": round(float((g <= 5).mean() * 100), 2)}
    # focus: variance of a Laplacian (higher = sharper); relative, compare between shots
    lap = (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1) - 4 * g)
    report["focus"] = round(float(lap.var()), 1)
    try:
        outer, part, bright = measure_outer(g.astype(np.uint8))
        report.update(part_found=True, part_diameter_px=round(2 * outer.r, 1), part_bright=bool(bright),
                      part_center=(round(outer.cx, 1), round(outer.cy, 1)),
                      frame_fill_pct=round(float(part.mean() * 100), 1))
    except VisionError as exc:
        report.update(part_found=False, problem=exc.slug)
    warnings = []
    if report["saturated_pct"] > 1:
        warnings.append("과노출: 노출/조명을 줄이세요")
    if report["mean"] < 40:
        warnings.append("저노출: 노출/조명을 올리세요")
    if report.get("frame_fill_pct", 100) < 8:
        warnings.append("부품이 너무 작게 찍힘: 카메라를 가까이 / 렌즈 교체")
    if report.get("problem") == "part_touches_border":
        warnings.append("부품이 화면 경계에 걸림: 지그 위치 조정")
    report["warnings"] = warnings
    return report


def capture_session(grab: Callable[[], np.ndarray], out_dir: str | Path, n: int, *, prefix: str = "CAP",
                    serials: Iterator[str] | None = None, interval_s: float = 0.0, station: str | None = None,
                    operator: str | None = None, on_frame: Callable[[str, dict], None] | None = None) -> list[dict]:
    """Grab `n` frames, save `<out>/images/<serial>.png` and append `<out>/import_manifest.csv`
    (serial, source, sha256) plus a `capture_log.jsonl` with the quality report per frame."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = out / "import_manifest.csv"
    new = not manifest.exists()
    existing = len(list((out / "images").glob(f"{prefix}-*.png")))
    rows = []
    with manifest.open("a", newline="", encoding="utf-8") as mf, (out / "capture_log.jsonl").open("a", encoding="utf-8") as lf:
        w = csv.DictWriter(mf, fieldnames=["serial", "source", "sha256"])
        if new:
            w.writeheader()
        for i in range(n):
            img = to_gray8(grab())
            serial = safe_name(next(serials)) if serials is not None else f"{prefix}-{existing + i + 1:05d}"
            path = out / "images" / f"{serial}.png"
            Image.fromarray(img).save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            q = frame_quality(img)
            w.writerow({"serial": serial, "source": f"camera:{station or 'local'}", "sha256": digest})
            mf.flush()
            entry = {"ts": utc_now(), "serial": serial, "sha256": digest, "station": station, "operator": operator, **q}
            lf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            lf.flush()
            rows.append(entry)
            if on_frame:
                on_frame(serial, q)
            if interval_s and i < n - 1:
                time.sleep(interval_s)
    return rows

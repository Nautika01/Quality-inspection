"""Run a batch inspection and write CSV / JSON / charts / overlay images."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .anomaly import AnomalyModel, model_hash  # noqa: E402
from .datasets import load_gray  # noqa: E402
from .db import image_sha256  # noqa: E402
from .inspector import VERDICT_ERROR, VERDICT_NG, VERDICT_PASS, InspectionResult, inspect_array  # noqa: E402
from .spc import XbarRChart, capability, xbar_r  # noqa: E402
from .spec import PartSpec  # noqa: E402
from .synth import DETECTED_KIND  # noqa: E402

_COLORS = {"scratch": (255, 40, 40), "dent": (255, 170, 0), "bright_spot": (120, 220, 120)}


def json_safe(obj):
    """Recursively convert numpy scalars and non-finite floats so json.dumps(allow_nan=False) works."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_overlay(img_path: Path, result: InspectionResult, out_path: Path) -> None:
    img = Image.fromarray(load_gray(img_path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    g = result.vision.geometry if result.vision else None
    if g is not None:
        for c in (g.outer, g.pocket):
            if c is not None:
                draw.ellipse([c.cx - c.r, c.cy - c.r, c.cx + c.r, c.cy + c.r], outline=(0, 200, 255))
    for gd in result.defects:
        d = gd.defect
        x0, y0, x1, y1 = d.bbox
        col = _COLORS.get(d.kind, (255, 0, 255))
        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=col, width=2)
        draw.text((x0, max(0, y0 - 12)), f"{d.kind}/{gd.grade}", fill=col)
    draw.text((4, 4), f"{result.serial} {result.verdict}", fill=(255, 255, 255))
    img.save(out_path)


def plot_spc(values: np.ndarray, chart: XbarRChart, spec: PartSpec, out_path: Path, baseline: int | None,
             title: str = "outer diameter", dim=None) -> None:
    dim = dim or spec.outer_diameter
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    idx = np.arange(1, len(chart.xbar) + 1)
    bad = np.array(sorted(chart.violations), dtype=int)

    ax = axes[0]
    ax.plot(idx, chart.xbar, "o-", ms=4, color="tab:blue")
    if len(bad):
        ax.plot(bad + 1, chart.xbar[bad], "o", ms=7, color="tab:red", label="rule violation")
    for y, ls in ((chart.center, "-"), (chart.ucl, "--"), (chart.lcl, "--")):
        ax.axhline(y, color="gray", ls=ls, lw=1)
    if baseline:
        ax.axvspan(0.5, baseline + 0.5, color="tab:green", alpha=0.08, label="baseline (Phase I)")
    ax.set_title(f"X-bar chart: {title} (n={chart.n})")
    ax.set_ylabel("mm")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(idx, chart.ranges, "o-", ms=4, color="tab:purple")
    for y, ls in ((chart.r_bar, "-"), (chart.r_ucl, "--"), (chart.r_lcl, "--")):
        ax.axhline(y, color="gray", ls=ls, lw=1)
    ax.set_title("R chart")
    ax.set_xlabel("subgroup")
    ax.set_ylabel("mm")

    ax = axes[2]
    ax.hist(values, bins=30, color="tab:blue", alpha=0.7)
    for x, lbl in ((dim.lsl, "LSL"), (dim.usl, "USL"), (dim.nominal, "nominal")):
        ax.axvline(x, color="tab:red" if lbl != "nominal" else "gray", ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.95, lbl, ha="center", fontsize=8)
    ax.set_title(f"{title} distribution")
    ax.set_xlabel("mm")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _metrics(df: pd.DataFrame, truth: pd.DataFrame) -> dict:
    from .anomaly import auroc

    m = df.merge(truth, on="serial", suffixes=("", "_true"))
    err = m[m.verdict == VERDICT_ERROR]
    m = m[m.verdict != VERDICT_ERROR]
    tp = int(((m.verdict == VERDICT_NG) & (m.expected_verdict == "NG")).sum())
    tn = int(((m.verdict == VERDICT_PASS) & (m.expected_verdict == "PASS")).sum())
    fp = int(((m.verdict == VERDICT_NG) & (m.expected_verdict == "PASS")).sum())
    fn = int(((m.verdict == VERDICT_PASS) & (m.expected_verdict == "NG")).sum())
    defect_rows = m[m.defect_type != "none"]
    expected_kind = defect_rows.defect_type.map(DETECTED_KIND)
    type_ok = (defect_rows.defect_types.fillna("") == expected_kind).mean() if len(defect_rows) else None
    grade_ok = (defect_rows.worst_grade.fillna("") == defect_rows.expected_grade.fillna("")).mean() if len(defect_rows) else None
    od_err = (m.od_mm - m.od_mm_true).dropna()
    out = {
        "n": int(len(m)),
        "n_error_frames": int(len(err)),
        "confusion": {"tp_ng": tp, "tn_pass": tn, "fp_overkill": fp, "fn_escape": fn},
        "accuracy": (tp + tn) / max(len(m), 1),
        "escape_rate": fn / max(tp + fn, 1),
        "overkill_rate": fp / max(tn + fp, 1),
        "defect_type_accuracy": type_ok,
        "defect_grade_accuracy": grade_ok,
        "od_measurement_error_mm": {"bias": float(od_err.mean()) if len(od_err) else None,
                                    "std": float(od_err.std()) if len(od_err) > 1 else None,
                                    "max_abs": float(od_err.abs().max()) if len(od_err) else None},
        "escaped_serials": m[(m.verdict == VERDICT_PASS) & (m.expected_verdict == "NG")].serial.tolist(),
        "overkill_serials": m[(m.verdict == VERDICT_NG) & (m.expected_verdict == "PASS")].serial.tolist(),
    }
    if "pocket_mm_true" in m and m.pocket_mm_true.notna().any():
        pe = (m.pocket_mm - m.pocket_mm_true).dropna()
        out["pocket_measurement_error_mm"] = {"bias": float(pe.mean()), "std": float(pe.std()) if len(pe) > 1 else None,
                                              "max_abs": float(pe.abs().max())}
    if "anomaly_score" in m and m.anomaly_score.notna().any():
        good = m[m.defect_type == "none"].anomaly_score.dropna().to_numpy()
        out["anomaly_auroc_by_defect_type"] = {
            kind: auroc(good, m[m.defect_type == kind].anomaly_score.dropna().to_numpy())
            for kind in sorted(m.defect_type.unique()) if kind != "none"
        }
    return out


def run_batch(
    image_dir: str | Path,
    out_dir: str | Path,
    spec: PartSpec,
    *,
    subgroup: int = 5,
    baseline: int | None = 25,
    truth_csv: str | Path | None = None,
    model_path: str | Path | None = None,
    db_path: str | Path | None = None,
    lot_id: str | None = None,
    station: str = "batch",
    operator: str | None = None,
) -> dict:
    image_dir, out = Path(image_dir), Path(out_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise FileNotFoundError(f"no images in {image_dir}")

    model = AnomalyModel.load(model_path) if model_path else None
    mh = model_hash(model_path) if model_path else None
    db = None
    if db_path:
        from .db import InspectionDB

        db = InspectionDB(db_path)
    lot_id = lot_id or paths[0].stem.rsplit("-", 1)[0]

    results = []
    for p in paths:
        try:
            img = load_gray(p)
        except Exception as exc:  # noqa: BLE001
            from .inspector import utc_now

            r = InspectionResult(p.stem, utc_now(), VERDICT_ERROR, ["vision:bad_image"], error_detail=str(exc)[:200])
            img = None
        else:
            r = inspect_array(img, p.stem, spec, model)
        results.append(r)
        if db is not None:
            db.add_inspection(r, lot_id, spec, mh, station, operator=operator,
                              image_hash=image_sha256(img) if img is not None else None, image_path=str(p))
        if r.verdict == VERDICT_NG and img is not None:
            save_overlay(p, r, out / "overlays" / f"{r.serial}.png")

    df = pd.DataFrame([r.to_row() for r in results])
    df.to_csv(out / "results.csv", index=False)
    measured = df[df.verdict != VERDICT_ERROR]

    summary: dict = {
        "part": spec.name,
        "lot_id": lot_id,
        "n_images": len(df),
        "n_error_frames": int((df.verdict == VERDICT_ERROR).sum()),
        "n_inspected": int(len(measured)),
        "n_pass": int((measured.verdict == VERDICT_PASS).sum()),
        "n_ng": int((measured.verdict == VERDICT_NG).sum()),
        "first_pass_yield": float((measured.verdict == VERDICT_PASS).mean()) if len(measured) else None,
        "ng_reasons": measured.reasons[measured.reasons != ""].str.split(";").explode().value_counts().to_dict(),
        "error_reasons": df[df.verdict == VERDICT_ERROR].reasons.value_counts().to_dict(),
        "minor_defects_recorded": int((measured.worst_grade == "minor").sum()),
        "anomaly_model": mh,
    }

    summary["spc"] = {}
    for ch, dim, title in (("od_mm", spec.outer_diameter, "outer diameter"), ("pocket_mm", spec.pocket_diameter, "pocket diameter")):
        if dim is None:
            continue
        vals = measured[ch].dropna().to_numpy(dtype=float)
        if len(vals) >= 2 * subgroup:
            chart = xbar_r(vals, n=subgroup, baseline=baseline)
            summary["spc"][ch] = {
                "subgroup_size": subgroup,
                "baseline_subgroups": chart.baseline,
                "center": chart.center, "ucl": chart.ucl, "lcl": chart.lcl,
                "out_of_control": chart.out_of_control,
                "first_xbar_violation_after_baseline": chart.first_violation_after_baseline(),
                "violations": {str(k + 1): v for k, v in sorted(chart.violations.items())},
                "capability": capability(vals, dim.lsl, dim.usl, chart.sigma_within),
            }
            plot_spc(vals, chart, spec, out / f"spc_{ch}.png", chart.baseline, title, dim)
    if "od_mm" in summary["spc"]:
        summary["spc_outer_diameter"] = summary["spc"]["od_mm"]  # backwards-compatible alias

    if truth_csv is not None and Path(truth_csv).exists():
        summary["vs_ground_truth"] = _metrics(df, pd.read_csv(truth_csv))
    if db is not None:
        summary["anchor"] = db.anchor_head(f"batch {lot_id}")
        summary["audit"] = db.verify_chain()
        db.close()

    write_json(out / "summary.json", summary)
    return summary

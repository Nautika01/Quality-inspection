"""Run a batch inspection and write CSV / JSON / charts / overlay images."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .anomaly import AnomalyModel, model_hash  # noqa: E402
from .inspector import InspectionResult, inspect_array  # noqa: E402
from .spc import XbarRChart, capability, xbar_r  # noqa: E402
from .spec import PartSpec  # noqa: E402
from .synth import DETECTED_KIND  # noqa: E402

_COLORS = {"scratch": (255, 40, 40), "dent": (255, 170, 0)}


def save_overlay(img_path: Path, result: InspectionResult, out_path: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    g = result.vision.geometry if result.vision else None
    if g is not None:
        for c in (g.outer, g.pocket):
            draw.ellipse([c.cx - c.r, c.cy - c.r, c.cx + c.r, c.cy + c.r], outline=(0, 200, 255))
    for gd in result.defects:
        d = gd.defect
        x0, y0, x1, y1 = d.bbox
        col = _COLORS.get(d.kind, (255, 0, 255))
        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=col, width=2)
        draw.text((x0, max(0, y0 - 12)), f"{d.kind}/{gd.grade}", fill=col)
    draw.text((4, 4), f"{result.serial} {result.verdict}", fill=(255, 255, 255))
    img.save(out_path)


def plot_spc(values: np.ndarray, chart: XbarRChart, spec: PartSpec, out_path: Path, baseline: int | None) -> None:
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
    ax.set_title(f"X-bar chart: outer diameter (n={chart.n})")
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
    od = spec.outer_diameter
    for x, lbl in ((od.lsl, "LSL"), (od.usl, "USL"), (od.nominal, "nominal")):
        ax.axvline(x, color="tab:red" if lbl != "nominal" else "gray", ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.95, lbl, ha="center", fontsize=8)
    ax.set_title("Outer diameter distribution")
    ax.set_xlabel("mm")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _metrics(df: pd.DataFrame, truth: pd.DataFrame) -> dict:
    m = df.merge(truth, on="serial", suffixes=("", "_true"))
    tp = int(((m.verdict == "NG") & (m.expected_verdict == "NG")).sum())
    tn = int(((m.verdict == "PASS") & (m.expected_verdict == "PASS")).sum())
    fp = int(((m.verdict == "NG") & (m.expected_verdict == "PASS")).sum())
    fn = int(((m.verdict == "PASS") & (m.expected_verdict == "NG")).sum())
    defect_rows = m[m.defect_type != "none"]
    expected_kind = defect_rows.defect_type.map(DETECTED_KIND)
    type_ok = (defect_rows.defect_types.fillna("") == expected_kind).mean() if len(defect_rows) else float("nan")
    grade_ok = (defect_rows.worst_grade.fillna("") == defect_rows.expected_grade.fillna("")).mean() if len(defect_rows) else float("nan")
    od_err = (m.od_mm - m.od_mm_true).dropna()
    out = {
        "n": len(m),
        "confusion": {"tp_ng": tp, "tn_pass": tn, "fp_overkill": fp, "fn_escape": fn},
        "accuracy": (tp + tn) / max(len(m), 1),
        "escape_rate": fn / max(tp + fn, 1),
        "overkill_rate": fp / max(tn + fp, 1),
        "defect_type_accuracy": float(type_ok),
        "defect_grade_accuracy": float(grade_ok),
        "od_measurement_error_mm": {"bias": float(od_err.mean()), "std": float(od_err.std()), "max_abs": float(od_err.abs().max())},
        "escaped_serials": m[(m.verdict == "PASS") & (m.expected_verdict == "NG")].serial.tolist(),
        "overkill_serials": m[(m.verdict == "NG") & (m.expected_verdict == "PASS")].serial.tolist(),
    }
    if "anomaly_score" in m and m.anomaly_score.notna().any():
        from .anomaly import auroc

        good = m[m.defect_type == "none"].anomaly_score.dropna()
        bad = m[m.defect_type != "none"].anomaly_score.dropna()
        out["anomaly_auroc_surface_defects"] = auroc(good.to_numpy(), bad.to_numpy())
    return out


def run_batch(
    image_dir: str | Path,
    out_dir: str | Path,
    spec: PartSpec,
    *,
    subgroup: int = 5,
    baseline: int | None = 10,
    truth_csv: str | Path | None = None,
    model_path: str | Path | None = None,
    db_path: str | Path | None = None,
    lot_id: str | None = None,
) -> dict:
    image_dir, out = Path(image_dir), Path(out_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no .png images in {image_dir}")

    model = AnomalyModel.load(model_path) if model_path else None
    mh = model_hash(model_path) if model_path else None
    db = None
    if db_path:
        from .db import InspectionDB

        db = InspectionDB(db_path)
    lot_id = lot_id or paths[0].stem.rsplit("-", 1)[0]

    results = []
    for p in paths:
        img = np.asarray(Image.open(p).convert("L"))
        r = inspect_array(img, p.stem, spec, model)
        results.append(r)
        if db is not None:
            db.add_inspection(r, lot_id, spec, mh, "batch")
        if r.verdict == "NG":
            save_overlay(p, r, out / "overlays" / f"{r.serial}.png")

    df = pd.DataFrame([r.to_row() for r in results])
    df.to_csv(out / "results.csv", index=False)

    summary: dict = {
        "part": spec.name,
        "lot_id": lot_id,
        "n_inspected": len(df),
        "n_pass": int((df.verdict == "PASS").sum()),
        "n_ng": int((df.verdict == "NG").sum()),
        "first_pass_yield": float((df.verdict == "PASS").mean()),
        "ng_reasons": df.reasons[df.reasons != ""].str.split(";").explode().value_counts().to_dict(),
        "minor_defects_recorded": int((df.worst_grade == "minor").sum()),
        "anomaly_model": mh,
    }

    od = df.od_mm.dropna().to_numpy()
    if len(od) >= 2 * subgroup:
        chart = xbar_r(od, n=subgroup, baseline=baseline)
        summary["spc_outer_diameter"] = {
            "subgroup_size": subgroup,
            "baseline_subgroups": baseline,
            "center": chart.center,
            "ucl": chart.ucl,
            "lcl": chart.lcl,
            "out_of_control": chart.out_of_control,
            "first_violation_subgroup": (min(chart.violations) + 1) if chart.violations else None,
            "violations": {str(k + 1): v for k, v in sorted(chart.violations.items())},
            "capability": capability(od, spec.outer_diameter.lsl, spec.outer_diameter.usl, chart.sigma_within),
        }
        plot_spc(od, chart, spec, out / "spc_outer_diameter.png", baseline)

    if truth_csv is not None and Path(truth_csv).exists():
        summary["vs_ground_truth"] = _metrics(df, pd.read_csv(truth_csv))
    if db is not None:
        summary["audit"] = db.verify_chain()
        db.close()

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

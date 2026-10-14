"""Command line entry point.

    qinspect generate | inspect | demo | gage | train-anomaly | anomaly-eval | import |
             calibrate | run-line | serve | dashboard | sampling | specdoc | audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .spec import PartSpec


def _load_spec(path: str | None) -> PartSpec:
    return PartSpec.from_json(path) if path else PartSpec()


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _cmd_generate(a):
    from .synth import generate_lot

    df = generate_lot(_load_spec(a.spec), a.n, a.out, seed=a.seed, lot_id=a.lot,
                      drift_start=None if a.no_drift else 0.6)
    print(f"wrote {len(df)} images to {Path(a.out) / 'images'}")
    print(df.expected_verdict.value_counts().to_string())


def _cmd_inspect(a):
    from .report import run_batch

    _print(run_batch(a.images, a.out, _load_spec(a.spec), subgroup=a.subgroup, baseline=a.baseline or None,
                     truth_csv=a.truth, model_path=a.model, db_path=a.db, lot_id=a.lot))


def _cmd_demo(a):
    """Full pipeline: generate -> train anomaly model on good parts -> inspect (hybrid) -> gage -> docs."""
    import numpy as np
    from PIL import Image

    from .anomaly import train
    from .msa import simulate_gage_study
    from .report import run_batch
    from .specdoc import render_inspection_standard
    from .synth import generate_lot

    spec, out = PartSpec(), Path(a.out)
    data = out / "data"
    df = generate_lot(spec, a.n, data, seed=a.seed)
    print(f"[1/4] generated {len(df)} parts -> {data}")

    model_path = None
    if not a.no_anomaly:
        train_dir = out / "train_good"
        train_dir.mkdir(parents=True, exist_ok=True)
        rng_df = df[df.expected_verdict == "PASS"].head(a.train_good)
        goods = [np.asarray(Image.open(data / "images" / f"{s}.png")) for s in rng_df.serial]
        model = train(goods, coreset=a.coreset, seed=a.seed)
        model_path = out / "anomaly_model.npz"
        mh = model.save(model_path, {"trained_on": rng_df.serial.tolist()})
        print(f"[2/4] anomaly model: {len(goods)} good parts, threshold {model.threshold:.2f}, hash {mh}")
    else:
        print("[2/4] anomaly model skipped")

    summary = run_batch(data / "images", out, spec, truth_csv=data / "ground_truth.csv",
                        model_path=model_path, db_path=out / "qinspect.db")
    print("[3/4] inspection done")

    _, studies = simulate_gage_study(spec, n_parts=a.gage_parts, n_operators=3, n_trials=2, seed=a.seed)
    gage = {k: v.to_dict() for k, v in studies.items()}
    (out / "gage_rr.json").write_text(json.dumps(gage, indent=2), encoding="utf-8")
    doc = render_inspection_standard(spec, gage=gage, program_version=__version__,
                                     model_hash=summary.get("anomaly_model"))
    (out / "inspection_standard.md").write_text(doc, encoding="utf-8")
    summary["gage_rr"] = {k: {"pct_grr_tol": v["pct_grr_tol"], "ndc": v["ndc"], "verdict": v["verdict"]} for k, v in gage.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[4/4] gage R&R + inspection standard written")
    _print(summary)
    print(f"\noutputs in {out}: results.csv summary.json spc_outer_diameter.png overlays/ qinspect.db "
          f"gage_rr.json inspection_standard.md" + (" anomaly_model.npz" if model_path else ""))


def _cmd_gage(a):
    from .msa import simulate_gage_study

    df, studies = simulate_gage_study(_load_spec(a.spec), n_parts=a.parts, n_operators=a.operators,
                                      n_trials=a.trials, seed=a.seed)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
    _print({k: v.to_dict() for k, v in studies.items()})


def _cmd_train_anomaly(a):
    from .anomaly import train
    from .datasets import list_images, load_gray

    paths = list_images(a.images)[: a.limit] if a.limit else list_images(a.images)
    model = train([load_gray(p) for p in paths], coreset=a.coreset, k=a.k, seed=a.seed,
                  threshold_sigma=a.threshold_sigma)
    mh = model.save(a.out, {"trained_on": [p.name for p in paths]})
    _print({"n_train": len(paths), "threshold": model.threshold, "bank": model.bank.shape, "hash": mh, "path": a.out})


def _cmd_anomaly_eval(a):
    import numpy as np

    from .anomaly import AnomalyModel, auroc, train
    from .datasets import AnomalyDataset, load_gray

    ds = AnomalyDataset.from_root(a.dataset)
    if a.model:
        model = AnomalyModel.load(a.model)
    else:
        model = train([load_gray(p, a.size) for p in ds.train_good[: a.limit or None]], coreset=a.coreset, seed=a.seed)
    good = np.array([model.score(load_gray(p, a.size)) for p in ds.test_good])
    out = {"n_train": len(ds.train_good), "n_test_good": len(good), "threshold": model.threshold,
           "good_fp_rate": float((good > model.threshold).mean()) if len(good) else None, "per_class": {}}
    all_bad = []
    for cls, paths in ds.test_bad.items():
        sc = np.array([model.score(load_gray(p, a.size)) for p in paths])
        all_bad.append(sc)
        out["per_class"][cls] = {"n": len(sc), "auroc": auroc(good, sc), "detection_rate": float((sc > model.threshold).mean())}
    if all_bad:
        out["auroc_all"] = auroc(good, np.concatenate(all_bad))
    if a.save_model and not a.model:
        out["model_hash"] = model.save(a.save_model)
    _print(out)


def _cmd_import(a):
    from .datasets import import_images

    files = import_images(a.src, a.dst, size=a.size, prefix=a.prefix)
    print(f"imported {len(files)} images -> {Path(a.dst) / 'images'}")


def _cmd_calibrate(a):
    import dataclasses

    from .datasets import calibrate_px_per_mm, load_gray

    ppm = calibrate_px_per_mm(load_gray(a.image), a.diameter_mm)
    spec = dataclasses.replace(_load_spec(a.spec), px_per_mm=round(ppm, 4))
    if a.out:
        spec.to_json(a.out)
    _print({"px_per_mm": ppm, "spec_written": a.out})


def _cmd_run_line(a):
    from .line import run_line

    def tick(res):
        if a.verbose:
            print(f"{res.serial} {res.verdict:4s} od={res.od_mm} {';'.join(res.reasons)}")

    summary = run_line(a.n, a.db, spec=_load_spec(a.spec), bus_url=a.bus, rate_hz=a.rate, seed=a.seed,
                       lot_id=a.lot, model_path=a.model, on_result=tick)
    _print(summary)


def _cmd_serve(a):
    import os

    import uvicorn

    os.environ["QINSPECT_DB"] = a.db
    if a.spec:
        os.environ["QINSPECT_SPEC"] = a.spec
    if a.model:
        os.environ["QINSPECT_MODEL"] = a.model
    uvicorn.run("qinspect.api:create_app", factory=True, host=a.host, port=a.port, reload=False)


def _cmd_dashboard(a):
    import os
    import subprocess

    env = os.environ | {"QINSPECT_DB": a.db}
    script = Path(__file__).with_name("dashboard.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(a.port),
                              "--", "--db", a.db], env=env))


def _cmd_sampling(a):
    from .sampling import iso2859_single_normal, zero_acceptance_plan

    out = {"iso2859_1": iso2859_single_normal(a.lot_size, a.aql, a.level).to_dict(),
           "c0": zero_acceptance_plan(a.lot_size, a.lq, a.confidence).to_dict()}
    if a.defectives is not None:
        out["iso2859_1"]["decision"] = iso2859_single_normal(a.lot_size, a.aql, a.level).decide(a.defectives)
        out["c0"]["decision"] = zero_acceptance_plan(a.lot_size, a.lq, a.confidence).decide(a.defectives)
    _print(out)


def _cmd_specdoc(a):
    from .specdoc import render_inspection_standard

    gage = json.loads(Path(a.gage).read_text()) if a.gage else None
    doc = render_inspection_standard(_load_spec(a.spec), lot_size=a.lot_size, aql=a.aql, gage=gage,
                                     program_version=__version__)
    if a.out:
        Path(a.out).write_text(doc, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(doc)


def _cmd_audit(a):
    from .db import InspectionDB

    _print(InspectionDB(a.db).verify_chain())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qinspect", description="Defense-part quality inspection")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="render a synthetic lot with ground truth")
    p.add_argument("--out", default="data/synth"); p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lot", default="LOT-0001")
    p.add_argument("--no-drift", action="store_true"); p.add_argument("--spec")
    p.set_defaults(fn=_cmd_generate)

    p = sub.add_parser("inspect", help="inspect a folder of images")
    p.add_argument("--images", required=True); p.add_argument("--out", default="runs/latest")
    p.add_argument("--truth"); p.add_argument("--model", help="anomaly model .npz (hybrid mode)")
    p.add_argument("--db", help="also store results + audit chain in this SQLite file"); p.add_argument("--lot")
    p.add_argument("--subgroup", type=int, default=5); p.add_argument("--baseline", type=int, default=10)
    p.add_argument("--spec"); p.set_defaults(fn=_cmd_inspect)

    p = sub.add_parser("demo", help="generate + train + inspect + gage + docs in one go")
    p.add_argument("--out", default="runs/demo"); p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--no-anomaly", action="store_true")
    p.add_argument("--train-good", type=int, default=40); p.add_argument("--coreset", type=int, default=3000)
    p.add_argument("--gage-parts", type=int, default=6); p.set_defaults(fn=_cmd_demo)

    p = sub.add_parser("gage", help="simulated Gage R&R study (ANOVA)")
    p.add_argument("--parts", type=int, default=10); p.add_argument("--operators", type=int, default=3)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="csv of raw measurements"); p.add_argument("--spec"); p.set_defaults(fn=_cmd_gage)

    p = sub.add_parser("train-anomaly", help="fit a memory-bank anomaly model on good images")
    p.add_argument("--images", required=True); p.add_argument("--out", default="anomaly_model.npz")
    p.add_argument("--limit", type=int); p.add_argument("--coreset", type=int, default=4000)
    p.add_argument("--k", type=int, default=1); p.add_argument("--threshold-sigma", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=_cmd_train_anomaly)

    p = sub.add_parser("anomaly-eval", help="evaluate on an MVTec-style dataset folder")
    p.add_argument("--dataset", required=True); p.add_argument("--model"); p.add_argument("--save-model")
    p.add_argument("--size", type=int, default=256); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--coreset", type=int, default=4000); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_cmd_anomaly_eval)

    p = sub.add_parser("import", help="convert a folder of photos into the canonical layout")
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--size", type=int, default=256); p.add_argument("--prefix", default="IMP"); p.set_defaults(fn=_cmd_import)

    p = sub.add_parser("calibrate", help="px/mm from a reference part with known outer diameter")
    p.add_argument("--image", required=True); p.add_argument("--diameter-mm", type=float, required=True)
    p.add_argument("--spec"); p.add_argument("--out", help="write updated spec json"); p.set_defaults(fn=_cmd_calibrate)

    p = sub.add_parser("run-line", help="simulated production line: camera -> inspect -> DB -> bus -> alarms")
    p.add_argument("--n", type=int, default=300); p.add_argument("--db", default="qinspect.db")
    p.add_argument("--bus", default="memory", help="memory | mqtt://host:port"); p.add_argument("--rate", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lot", default="LOT-SIM")
    p.add_argument("--model"); p.add_argument("--spec"); p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=_cmd_run_line)

    p = sub.add_parser("serve", help="REST API (FastAPI/uvicorn)")
    p.add_argument("--db", default="qinspect.db"); p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000); p.add_argument("--spec"); p.add_argument("--model")
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("dashboard", help="Streamlit dashboard")
    p.add_argument("--db", default="qinspect.db"); p.add_argument("--port", type=int, default=8501); p.set_defaults(fn=_cmd_dashboard)

    p = sub.add_parser("sampling", help="acceptance sampling plan (ISO 2859-1, c=0)")
    p.add_argument("--lot-size", type=int, required=True); p.add_argument("--aql", type=float, default=1.0)
    p.add_argument("--level", default="II"); p.add_argument("--lq", type=float, default=0.05)
    p.add_argument("--confidence", type=float, default=0.9); p.add_argument("--defectives", type=int)
    p.set_defaults(fn=_cmd_sampling)

    p = sub.add_parser("specdoc", help="generate the inspection standard (Markdown) from the spec")
    p.add_argument("--spec"); p.add_argument("--gage", help="gage_rr.json from `demo`/`gage`")
    p.add_argument("--lot-size", type=int, default=1000); p.add_argument("--aql", type=float, default=1.0)
    p.add_argument("--out"); p.set_defaults(fn=_cmd_specdoc)

    p = sub.add_parser("audit", help="verify the inspection record hash chain")
    p.add_argument("--db", default="qinspect.db"); p.set_defaults(fn=_cmd_audit)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

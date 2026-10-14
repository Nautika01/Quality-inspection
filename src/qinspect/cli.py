"""Command line entry point.

    qinspect generate | inspect | demo | gage | train-anomaly | anomaly-eval | import |
             calibrate | run-line | serve | dashboard | sampling | specdoc | audit |
             export | lot-close | disposition
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .spec import PartSpec

DEFAULT_DB = "qinspect.db"


def _load_spec(path: str | None) -> PartSpec:
    path = path or os.environ.get("QINSPECT_SPEC")
    return PartSpec.from_json(path) if path else PartSpec()


def _db_path(arg: str | None) -> str:
    return arg or os.environ.get("QINSPECT_DB", DEFAULT_DB)


def _print(obj) -> None:
    from .report import json_safe

    print(json.dumps(json_safe(obj), indent=2, ensure_ascii=False, default=str))


def _open_db(path: str):
    from .db import InspectionDB

    try:
        return InspectionDB(path, create=False)
    except FileNotFoundError as exc:
        sys.exit(f"error: {exc}")


def _cmd_generate(a):
    from .synth import generate_lot

    df = generate_lot(_load_spec(a.spec), a.n, a.out, seed=a.seed, lot_id=a.lot,
                      drift_start=None if a.no_drift else 0.6)
    print(f"wrote {len(df)} images to {Path(a.out) / 'images'}")
    print(df.expected_verdict.value_counts().to_string())


def _cmd_inspect(a):
    from .report import run_batch

    _print(run_batch(a.images, a.out, _load_spec(a.spec), subgroup=a.subgroup, baseline=a.baseline or None,
                     truth_csv=a.truth, model_path=a.model, db_path=a.db, lot_id=a.lot, operator=a.operator))


def _good_images(df, data_dir: Path, limit: int):
    """Clean training set for the one-class model: in tolerance AND no injected defect at all."""
    from .datasets import load_gray

    good = df[(df.expected_verdict == "PASS") & (df.defect_type == "none")].head(limit)
    return [load_gray(data_dir / "images" / f"{s}.png") for s in good.serial], good.serial.tolist()


def _cmd_demo(a):
    """Full pipeline: generate -> train anomaly model on clean parts -> inspect (hybrid) -> gage -> docs."""
    from .anomaly import train
    from .msa import simulate_gage_study
    from .report import run_batch, write_json
    from .specdoc import render_inspection_standard
    from .synth import generate_lot

    spec, out = _load_spec(a.spec), Path(a.out)
    data = out / "data"
    df = generate_lot(spec, a.n, data, seed=a.seed)
    print(f"[1/4] generated {len(df)} parts -> {data}")

    model_path = None
    if not a.no_anomaly:
        goods, names = _good_images(df, data, a.train_good)
        model = train(goods, coreset=a.coreset, seed=a.seed, has_pocket=spec.has_pocket, names=names)
        model_path = out / "anomaly_model.npz"
        mh = model.save(model_path, {"trained_on": names})
        print(f"[2/4] anomaly model: {len(goods)} clean parts, threshold {model.threshold:.2f}, hash {mh}")
    else:
        print("[2/4] anomaly model skipped")

    summary = run_batch(data / "images", out, spec, truth_csv=data / "ground_truth.csv",
                        model_path=model_path, db_path=out / "qinspect.db", operator="demo")
    print("[3/4] inspection done")

    _, studies = simulate_gage_study(spec, n_parts=a.gage_parts, n_operators=3, n_trials=2, seed=a.seed)
    gage = {k: v.to_dict() for k, v in studies.items()}
    write_json(out / "gage_rr.json", gage)
    doc = render_inspection_standard(spec, gage=gage, program_version=__version__, model_hash=summary.get("anomaly_model"))
    (out / "inspection_standard.md").write_text(doc, encoding="utf-8")
    summary["gage_rr"] = {k: {"pct_grr_tol": v["pct_grr_tol"], "ndc": v["ndc"], "verdict": v["verdict"],
                              "suggested_guard_band_mm": v["suggested_guard_band_mm"]} for k, v in gage.items()}
    write_json(out / "summary.json", summary)
    print("[4/4] gage R&R + inspection standard written")
    _print(summary)
    print(f"\noutputs in {out}: results.csv summary.json spc_od_mm.png spc_pocket_mm.png overlays/ qinspect.db "
          f"gage_rr.json inspection_standard.md" + (" anomaly_model.npz" if model_path else ""))


def _cmd_gage(a):
    from .msa import simulate_gage_study
    from .report import write_json

    df, studies = simulate_gage_study(_load_spec(a.spec), n_parts=a.parts, n_operators=a.operators,
                                      n_trials=a.trials, seed=a.seed)
    out = {k: v.to_dict() for k, v in studies.items()}
    if a.out_csv:
        Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out_csv, index=False)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json(Path(a.out_json), out)
    _print(out)


def _cmd_train_anomaly(a):
    from .anomaly import train
    from .datasets import list_images, load_gray

    spec = _load_spec(a.spec)
    paths = list_images(a.images)
    if a.truth:
        import pandas as pd

        t = pd.read_csv(a.truth)
        ok = set(t[(t.expected_verdict == "PASS") & (t.get("defect_type", "none") == "none")].serial)
        paths = [p for p in paths if p.stem in ok]
    if a.limit:
        paths = paths[: a.limit]
    model = train([load_gray(p) for p in paths], coreset=a.coreset, k=a.k, seed=a.seed,
                  threshold_sigma=a.threshold_sigma, has_pocket=spec.has_pocket, names=[p.name for p in paths])
    mh = model.save(a.out, {"trained_on": [p.name for p in paths]})
    _print({"n_train": len(paths), "threshold": model.threshold, "bank": list(model.bank.shape), "hash": mh, "path": a.out})


def _cmd_anomaly_eval(a):
    import numpy as np

    from .anomaly import AnomalyModel, auroc, train
    from .datasets import AnomalyDataset, load_gray

    spec = _load_spec(a.spec)
    ds = AnomalyDataset.from_root(a.dataset)
    if a.model:
        model = AnomalyModel.load(a.model)
    else:
        tr = ds.train_good[: a.limit or None]
        model = train([load_gray(p, a.size) for p in tr], coreset=a.coreset, seed=a.seed, has_pocket=spec.has_pocket,
                      names=[p.name for p in tr])

    def scores(paths):
        ok, failed = [], 0
        for p in paths:
            try:
                ok.append(model.score(load_gray(p, a.size)))
            except Exception:  # noqa: BLE001 - count unsegmentable images instead of aborting
                failed += 1
        return np.array(ok), failed

    good, good_failed = scores(ds.test_good)
    out = {"n_train": len(ds.train_good), "n_test_good": int(len(good)), "n_test_good_unsegmentable": good_failed,
           "threshold": model.threshold,
           "good_fp_rate": float((good > model.threshold).mean()) if len(good) else None, "per_class": {}}
    all_bad = []
    for cls, paths in ds.test_bad.items():
        sc, failed = scores(paths)
        all_bad.append(sc)
        out["per_class"][cls] = {"n": int(len(sc)), "unsegmentable": failed, "auroc": auroc(good, sc),
                                 "detection_rate": float((sc > model.threshold).mean()) if len(sc) else None}
    if all_bad and len(good):
        out["auroc_all"] = auroc(good, np.concatenate(all_bad))
    if a.save_model and not a.model:
        out["model_hash"] = model.save(a.save_model)
    _print(out)


def _cmd_import(a):
    from .datasets import import_images

    files = import_images(a.src, a.dst, size=a.size, prefix=a.prefix)
    print(f"imported {len(files)} new images -> {Path(a.dst) / 'images'} (manifest: {Path(a.dst) / 'import_manifest.csv'})")


def _cmd_calibrate(a):
    import dataclasses

    from .datasets import calibrate_px_per_mm, load_gray
    from .db import image_sha256, spec_hash

    img = load_gray(a.image)
    ppm = calibrate_px_per_mm(img, a.diameter_mm)
    spec = dataclasses.replace(_load_spec(a.spec), px_per_mm=round(ppm, 4))
    if a.out:
        spec.to_json(a.out)
    rec = None
    if a.db:
        from .db import InspectionDB

        with InspectionDB(_db_path(a.db)) as db:
            rec = db.add_calibration(ppm, a.diameter_mm, station=a.station, operator=a.operator, gauge_id=a.gauge_id,
                                     image_hash=image_sha256(img), spec_h=spec_hash(spec), note=a.note)
    _print({"px_per_mm": ppm, "spec_written": a.out, "calibration_record_id": rec})


def _cmd_run_line(a):
    from .line import run_line

    def tick(res):
        if a.verbose:
            print(f"{res.serial} {res.verdict:5s} od={res.od_mm} {';'.join(res.reasons)}")

    summary = run_line(a.n, _db_path(a.db), spec=_load_spec(a.spec), bus_url=a.bus, rate_hz=a.rate, seed=a.seed,
                       lot_id=a.lot, model_path=a.model, on_result=tick, baseline_subgroups=a.baseline,
                       operator=a.operator, image_dir=a.image_dir, freeze_by=a.freeze_by)
    if not a.verbose:
        summary["alarms"] = f"{summary['n_alarms']} alarms (use -v or the dashboard / API to list them)"
    _print(summary)


def _cmd_serve(a):
    import uvicorn

    os.environ["QINSPECT_DB"] = _db_path(a.db)
    if a.spec:
        os.environ["QINSPECT_SPEC"] = a.spec
    if a.model:
        os.environ["QINSPECT_MODEL"] = a.model
    if a.api_key:
        os.environ["QINSPECT_API_KEY"] = a.api_key
    if a.image_dir:
        os.environ["QINSPECT_IMAGE_DIR"] = a.image_dir
    uvicorn.run("qinspect.api:create_app", factory=True, host=a.host, port=a.port, reload=False)


def _cmd_dashboard(a):
    import subprocess

    env = os.environ | {"QINSPECT_DB": _db_path(a.db)}
    if a.spec:
        env["QINSPECT_SPEC"] = a.spec
    if a.tz:
        env["QINSPECT_TZ"] = a.tz
    script = Path(__file__).with_name("dashboard.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(a.port),
                              "--", "--db", _db_path(a.db)], env=env))


def _cmd_sampling(a):
    from .sampling import iso2859_single_normal, lot_disposition, zero_acceptance_plan

    iso = iso2859_single_normal(a.lot_size, a.aql, a.level)
    c0 = zero_acceptance_plan(a.lot_size, a.lq, a.confidence)
    out = {"iso2859_1": iso.to_dict(), "c0": c0.to_dict()}
    if a.defectives is not None:
        out["iso2859_1"]["decision"] = lot_disposition(iso, a.defectives, a.criticals)["decision"]
        out["c0"]["decision"] = lot_disposition(c0, a.defectives, a.criticals)["decision"]
    _print(out)


def _cmd_specdoc(a):
    from .specdoc import render_inspection_standard

    gage = json.loads(Path(a.gage).read_text(encoding="utf-8")) if a.gage else None
    limits = None
    if a.db and Path(_db_path(a.db)).exists():
        with _open_db(_db_path(a.db)) as db:
            spec = _load_spec(a.spec)
            limits = {k: v for k, v in db.active_limits(spec.part_number).items()} or None
    mh = a.model_hash
    if a.model and not mh:
        from .anomaly import model_hash

        mh = model_hash(a.model)
    doc = render_inspection_standard(_load_spec(a.spec), lot_size=a.lot_size, aql=a.aql, gage=gage,
                                     program_version=__version__, model_hash=mh, control_limits=limits)
    if a.out:
        Path(a.out).write_text(doc, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(doc)


def _cmd_audit(a):
    with _open_db(_db_path(a.db)) as db:
        out = db.verify_chain()
        if a.anchor:
            out["anchor_written"] = db.anchor_head(a.note or "manual")
    _print(out)
    if not out["ok"]:
        sys.exit(2)


def _cmd_export(a):
    import pandas as pd

    with _open_db(_db_path(a.db)) as db:
        rows = db.inspections(a.lot, since=a.since, until=a.until, station=a.station)
        summary = db.lot_summary(a.lot, since=a.since, until=a.until, station=a.station) if a.lot else None
    df = pd.DataFrame(rows)
    if a.tz and len(df):
        df["inspected_at_local"] = pd.to_datetime(df.inspected_at, utc=True).dt.tz_convert(a.tz).dt.strftime("%Y-%m-%d %H:%M:%S")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.out.endswith(".json"):
        from .report import write_json

        write_json(Path(a.out), {"summary": summary, "rows": rows})
    else:
        df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}")
    if summary:
        _print(summary)


def _cmd_lot_close(a):
    from .sampling import draw_sample, iso2859_single_normal, lot_disposition

    with _open_db(_db_path(a.db)) as db:
        rows = [r for r in db.inspections(a.lot) if r["verdict"] != "ERROR"]
        if not rows:
            sys.exit(f"error: lot {a.lot} has no measured inspections")
        latest = {}
        for r in rows:
            latest[r["serial"]] = r
        serials = sorted(latest)
        plan = iso2859_single_normal(len(serials), a.aql, a.level)
        sample = draw_sample(serials, plan, a.seed)
        n_def = sum(latest[s]["verdict"] == "NG" for s in sample)
        n_crit = sum(latest[s]["worst_grade"] == "critical" for s in sample)
        disp = lot_disposition(plan, n_def, n_crit)
        decision = a.decision or disp["decision"]
        out = db.close_lot(a.lot, decision, plan.to_dict(), sample, n_def, n_crit, a.by, a.note)
    _print(out | {"sampling": disp, "n_serials": len(serials)})


def _cmd_disposition(a):
    with _open_db(_db_path(a.db)) as db:
        if a.list:
            _print(db.open_ng(a.lot))
            return
        if a.inspection_id is None:
            sys.exit("error: --inspection-id is required (or --list)")
        did = db.add_disposition(a.inspection_id, a.decision, a.by, a.note, a.mrb_ref)
    _print({"id": did, "inspection_id": a.inspection_id, "decision": a.decision})


def build_parser() -> argparse.ArgumentParser:
    from .sampling import AQLS, INSPECTION_LEVELS
    from .spc import SUBGROUP_SIZES

    ap = argparse.ArgumentParser(prog="qinspect", description="Defense-part quality inspection")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_db(p, help_="SQLite file (default: $QINSPECT_DB or qinspect.db)"):
        p.add_argument("--db", default=None, help=help_)

    p = sub.add_parser("generate", help="render a synthetic lot with ground truth")
    p.add_argument("--out", default="data/synth"); p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lot", default="LOT-0001")
    p.add_argument("--no-drift", action="store_true"); p.add_argument("--spec")
    p.set_defaults(fn=_cmd_generate)

    p = sub.add_parser("inspect", help="inspect a folder of images")
    p.add_argument("--images", required=True); p.add_argument("--out", default="runs/latest")
    p.add_argument("--truth"); p.add_argument("--model", help="anomaly model .npz (hybrid mode)")
    add_db(p, "also store results + audit chain in this SQLite file (off by default)"); p.add_argument("--lot")
    p.add_argument("--operator"); p.add_argument("--subgroup", type=int, default=5, choices=SUBGROUP_SIZES)
    p.add_argument("--baseline", type=int, default=25, help="Phase I subgroups (0 = use all)")
    p.add_argument("--spec"); p.set_defaults(fn=_cmd_inspect)

    p = sub.add_parser("demo", help="generate + train + inspect + gage + docs in one go")
    p.add_argument("--out", default="runs/demo"); p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--no-anomaly", action="store_true")
    p.add_argument("--train-good", type=int, default=40); p.add_argument("--coreset", type=int, default=3000)
    p.add_argument("--gage-parts", type=int, default=6); p.add_argument("--spec"); p.set_defaults(fn=_cmd_demo)

    p = sub.add_parser("gage", help="simulated Gage R&R study (ANOVA)")
    p.add_argument("--parts", type=int, default=10); p.add_argument("--operators", type=int, default=3)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", help="raw measurements"); p.add_argument("--out-json", help="study results (for specdoc --gage)")
    p.add_argument("--spec"); p.set_defaults(fn=_cmd_gage)

    p = sub.add_parser("train-anomaly", help="fit a memory-bank anomaly model on known-good images")
    p.add_argument("--images", required=True); p.add_argument("--out", default="anomaly_model.npz")
    p.add_argument("--truth", help="ground_truth.csv: keep only PASS parts with no defect")
    p.add_argument("--limit", type=int); p.add_argument("--coreset", type=int, default=4000)
    p.add_argument("--k", type=int, default=1); p.add_argument("--threshold-sigma", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--spec"); p.set_defaults(fn=_cmd_train_anomaly)

    p = sub.add_parser("anomaly-eval", help="evaluate on an MVTec-style dataset folder")
    p.add_argument("--dataset", required=True); p.add_argument("--model"); p.add_argument("--save-model")
    p.add_argument("--size", type=int, default=256); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--coreset", type=int, default=4000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spec"); p.set_defaults(fn=_cmd_anomaly_eval)

    p = sub.add_parser("import", help="convert a folder of photos into the canonical layout (keeps a manifest)")
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--size", type=int, default=None, help="letterbox to this square size (default: keep)")
    p.add_argument("--prefix", default="IMP"); p.set_defaults(fn=_cmd_import)

    p = sub.add_parser("calibrate", help="px/mm from a reference disk/pin gauge with known outer diameter")
    p.add_argument("--image", required=True); p.add_argument("--diameter-mm", type=float, required=True)
    p.add_argument("--spec"); p.add_argument("--out", help="write updated spec json")
    add_db(p, "log the calibration in this DB"); p.add_argument("--station"); p.add_argument("--operator")
    p.add_argument("--gauge-id"); p.add_argument("--note"); p.set_defaults(fn=_cmd_calibrate)

    p = sub.add_parser("run-line", help="simulated production line: camera -> inspect -> DB -> bus -> alarms")
    p.add_argument("--n", type=int, default=300); add_db(p)
    p.add_argument("--bus", default="memory", help="memory | mqtt://host:port"); p.add_argument("--rate", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lot", default="LOT-SIM")
    p.add_argument("--model"); p.add_argument("--spec"); p.add_argument("--operator")
    p.add_argument("--baseline", type=int, default=25, help="Phase I subgroups when no frozen limits exist")
    p.add_argument("--image-dir", help="keep NG/ERROR frames here as evidence")
    p.add_argument("--freeze-by", help="freeze the resulting control limits in the DB under this name")
    p.add_argument("-v", "--verbose", action="store_true"); p.set_defaults(fn=_cmd_run_line)

    p = sub.add_parser("serve", help="REST API (FastAPI/uvicorn)")
    add_db(p); p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--spec"); p.add_argument("--model"); p.add_argument("--api-key", help="require X-API-Key on writes")
    p.add_argument("--image-dir"); p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("dashboard", help="Streamlit dashboard")
    add_db(p); p.add_argument("--port", type=int, default=8501); p.add_argument("--spec")
    p.add_argument("--tz", help="display time zone, e.g. Asia/Seoul (default $QINSPECT_TZ or Asia/Seoul)")
    p.set_defaults(fn=_cmd_dashboard)

    p = sub.add_parser("sampling", help="acceptance sampling plan (ISO 2859-1, c=0)")
    p.add_argument("--lot-size", type=int, required=True); p.add_argument("--aql", type=float, default=1.0, choices=AQLS)
    p.add_argument("--level", default="II", choices=INSPECTION_LEVELS); p.add_argument("--lq", type=float, default=0.05)
    p.add_argument("--confidence", type=float, default=0.9); p.add_argument("--defectives", type=int)
    p.add_argument("--criticals", type=int, default=0); p.set_defaults(fn=_cmd_sampling)

    p = sub.add_parser("specdoc", help="generate the inspection standard (Markdown) from the spec")
    p.add_argument("--spec"); p.add_argument("--gage", help="gage_rr.json from `demo` or `gage --out-json`")
    p.add_argument("--model", help="anomaly model file (its hash goes into the document)"); p.add_argument("--model-hash")
    add_db(p, "include frozen control limits from this DB"); p.add_argument("--lot-size", type=int, default=1000)
    p.add_argument("--aql", type=float, default=1.0, choices=AQLS); p.add_argument("--out"); p.set_defaults(fn=_cmd_specdoc)

    p = sub.add_parser("audit", help="verify the inspection record hash chain (exit 2 if broken)")
    add_db(p); p.add_argument("--anchor", action="store_true", help="also record the current head in `anchors`")
    p.add_argument("--note"); p.set_defaults(fn=_cmd_audit)

    p = sub.add_parser("export", help="export inspections for a lot / station / time window (shift report)")
    add_db(p); p.add_argument("--lot"); p.add_argument("--station")
    p.add_argument("--since", help="ISO-8601 UTC, e.g. 2026-10-14T00:00:00Z"); p.add_argument("--until")
    p.add_argument("--tz", default="Asia/Seoul"); p.add_argument("--out", required=True, help=".csv or .json")
    p.set_defaults(fn=_cmd_export)

    p = sub.add_parser("lot-close", help="acceptance sampling on a lot's results + record the lot disposition")
    add_db(p); p.add_argument("--lot", required=True); p.add_argument("--by", required=True)
    p.add_argument("--aql", type=float, default=1.0, choices=AQLS); p.add_argument("--level", default="II", choices=INSPECTION_LEVELS)
    p.add_argument("--seed", type=int); p.add_argument("--decision", choices=["ACCEPT", "REJECT", "HOLD"], help="override")
    p.add_argument("--note"); p.set_defaults(fn=_cmd_lot_close)

    p = sub.add_parser("disposition", help="record an NG disposition (rework/scrap/use_as_is/mrb/reinspect) or list open NG")
    add_db(p); p.add_argument("--inspection-id", type=int); p.add_argument("--decision", choices=["rework", "scrap", "use_as_is", "mrb", "reinspect"])
    p.add_argument("--by"); p.add_argument("--note"); p.add_argument("--mrb-ref")
    p.add_argument("--list", action="store_true"); p.add_argument("--lot"); p.set_defaults(fn=_cmd_disposition)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "disposition" and not args.list and (not args.decision or not args.by):
        sys.exit("error: --decision and --by are required")
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

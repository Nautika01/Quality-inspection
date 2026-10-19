"""Regressions for the second audit round (v0.3.1)."""
import dataclasses
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from qinspect.anomaly import AnomalyModel, model_file, train
from qinspect.api import create_app
from qinspect.datasets import import_images
from qinspect.db import GENESIS, InspectionDB, canonical_record_v1
from qinspect.inspector import inspect_array
from qinspect.line import InMemoryBus, SpcMonitor, Station
from qinspect.msa import gage_rr_anova
from qinspect.sampling import oc_curve, zero_acceptance_plan
from qinspect.spc import western_electric
from qinspect.spec import PartSpec
from qinspect.synth import render_part
from qinspect.util import safe_name, utc_param
from qinspect.vision import measure_geometry

SPEC = PartSpec()


def _img(seed=0, **kw):
    kw.setdefault("od_mm", 12.0)
    kw.setdefault("pocket_mm", 5.0)
    return render_part(SPEC, np.random.default_rng(seed), **kw)


# -- util ------------------------------------------------------------------------------


def test_utc_param_and_safe_name():
    assert utc_param("2026-10-14T09:00:00+09:00") == "2026-10-14T00:00:00Z"
    assert utc_param("2026-10-14") == "2026-10-14T00:00:00Z"
    assert utc_param(None) is None
    with pytest.raises(ValueError):
        utc_param("14/10/2026", "--since")
    assert safe_name("2026/10/14-001") == "14-001"
    assert safe_name("../../escape") == "escape"
    assert safe_name(None) == "unknown"


# -- vision / inspector ----------------------------------------------------------------


def test_dark_defect_touching_pocket_does_not_bias_pocket():
    img = _img(5).astype(float)
    p = measure_geometry(img.astype(np.uint8)).pocket
    yy, xx = np.mgrid[0:256, 0:256]
    for sig in (6, 10):
        dent = 150 * np.exp(-((xx - (p.cx + p.r + 4)) ** 2 + (yy - p.cy) ** 2) / (2 * sig**2))
        res = inspect_array(np.clip(img - dent, 0, 255).astype(np.uint8), "S", SPEC)
        assert res.verdict == "NG" and res.reasons == ["surface_dent:critical"]
        assert abs(res.pocket_mm - 5.0) < 0.012 and res.concentricity_mm < 0.02
    # genuine eccentricity is still measured
    res = inspect_array(_img(6, pocket_offset_mm=(0.2, 0)), "S", SPEC)
    assert abs(res.concentricity_mm - 0.2) < 0.012


def test_bright_recess_and_float_overshoot(monkeypatch):
    from qinspect import synth

    monkeypatch.setattr(synth, "POCKET_LEVEL", 240.0)
    res = inspect_array(_img(7), "S", SPEC)
    assert res.verdict == "PASS" and abs(res.pocket_mm - 5.0) < 0.012
    monkeypatch.setattr(synth, "POCKET_LEVEL", 60.0)
    f = _img(8) / 255.0
    f[0, 0] = 1.15  # white-balance overshoot must not collapse the frame
    assert inspect_array(f, "S", SPEC).verdict == "PASS"


def test_tiny_defects_at_low_resolution_do_not_error():
    sp = dataclasses.replace(SPEC, px_per_mm=8, image_size=128, min_defect_area_px=2)
    for i in range(4):
        res = inspect_array(render_part(sp, np.random.default_rng(i), od_mm=12, pocket_mm=5, defects=("pit",)), "S", sp)
        assert res.verdict != "ERROR", res.error_detail


def test_apply_anomaly_tolerates_array_outputs():
    class Arr:
        def __init__(self, out):
            self.out = out

        def predict(self, img):
            return self.out

    img = _img(3)
    res = inspect_array(img, "S", SPEC, Arr((np.array([True]), np.array([2.0]))))
    assert res.verdict == "NG" and res.anomaly_score == 2.0
    res = inspect_array(img, "S", SPEC, Arr((np.array([True, False]), 1.0)))
    assert res.verdict == "PASS" and res.anomaly_score is None and res.anomaly_error


# -- statistics ------------------------------------------------------------------------


def test_zero_repeatability_keeps_interaction_and_no_variation_raises():
    rows = []
    for part in range(3):
        for op in range(2):
            v = 10 + 0.1 * part + (0.02 if (part + op) % 2 else -0.02)  # interaction, identical repeats
            rows += [{"part": part, "operator": op, "trial": t, "value": v} for t in range(2)]
    with pytest.warns(UserWarning, match="identical"):
        g = gage_rr_anova(pd.DataFrame(rows), tolerance=1.0)
    assert g.repeatability == 0 and g.reproducibility > 0
    flat = [{"part": p, "operator": o, "trial": t, "value": 1.0} for p in range(2) for o in range(2) for t in range(2)]
    with pytest.raises(ValueError, match="no variation"):
        gage_rr_anova(pd.DataFrame(flat), tolerance=1.0)


def test_c0_plan_oc_curve_consistent_and_fast():
    plan = zero_acceptance_plan(20, 0.07, 0.9)
    _, pa = oc_curve(plan, np.array([0.07]))
    assert pa[0] <= 0.1 + 1e-9
    big = zero_acceptance_plan(1_000_000, 0.0005, 0.99)  # must return in well under a second
    assert big.sample_size > 0
    assert "100 %" in zero_acceptance_plan(5, 0.2, 0.999).note


def test_spc_monitor_degenerate_limits_and_sigma_guard():
    mon = SpcMonitor(subgroup=2, baseline_subgroups=1)
    assert mon.baseline_subgroups == 2
    for _ in range(8):
        mon.add(10.0)  # constant readings: limits must not be adopted
    assert mon.limits is None
    assert western_electric([0.0, 0.0], 0.0, 1.0) == {}


# -- traceability ----------------------------------------------------------------------


def test_close_lot_twice_and_anchor_laundering_refused(tmp_path):
    with InspectionDB(tmp_path / "c.db") as db:
        for i in range(3):
            db.add_inspection(inspect_array(_img(i), f"S{i}", SPEC), "L", SPEC)
        db.close_lot("L", "ACCEPT", {}, ["S0"], 0, 0, "qa")
        with pytest.raises(ValueError, match="already closed"):
            db.close_lot("L", "REJECT", {}, ["S0"], 0, 0, "qa")
        db.conn.execute("DELETE FROM inspections WHERE id=(SELECT MAX(id) FROM inspections)")
        assert db.verify_chain()["ok"] is False
        with pytest.raises(ValueError, match="refusing to anchor"):
            db.anchor_head("laundering attempt")
        assert db.verify_chain()["ok"] is False  # still detected


def test_v02_database_is_migrated_and_legacy_rows_verify(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE lots (lot_id TEXT PRIMARY KEY, part_number TEXT, spec_hash TEXT,
                       created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')));
    CREATE TABLE inspections (id INTEGER PRIMARY KEY AUTOINCREMENT, serial TEXT NOT NULL, lot_id TEXT NOT NULL,
        inspected_at TEXT NOT NULL, verdict TEXT NOT NULL, reasons TEXT NOT NULL, od_mm REAL, pocket_mm REAL,
        concentricity_mm REAL, worst_grade TEXT, anomaly_score REAL, program_version TEXT, spec_hash TEXT,
        model_hash TEXT, station TEXT, prev_hash TEXT NOT NULL, record_hash TEXT NOT NULL UNIQUE);
    CREATE TABLE defects (id INTEGER PRIMARY KEY AUTOINCREMENT, inspection_id INTEGER NOT NULL, kind TEXT, grade TEXT,
        length_mm REAL, area_px INTEGER, x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER);
    CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, lot_id TEXT, kind TEXT NOT NULL, payload TEXT NOT NULL);
    INSERT INTO lots(lot_id, part_number, spec_hash) VALUES ('OLD', 'EX', 'abc');
    """)
    prev = GENESIS
    for i in range(3):
        rec = {"serial": f"O{i}", "lot_id": "OLD", "inspected_at": "2026-01-01T00:00:00+00:00", "verdict": "PASS",
               "reasons": "", "od_mm": 12.0, "pocket_mm": 5.0, "concentricity_mm": 0.01, "worst_grade": "",
               "anomaly_score": None, "program_version": "0.2.0", "spec_hash": "abc", "model_hash": None,
               "station": "batch", "defects": []}
        rh = hashlib.sha256((prev + canonical_record_v1(rec)).encode()).hexdigest()
        con.execute("""INSERT INTO inspections(serial, lot_id, inspected_at, verdict, reasons, od_mm, pocket_mm,
            concentricity_mm, worst_grade, anomaly_score, program_version, spec_hash, model_hash, station, prev_hash, record_hash)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (rec["serial"], "OLD", rec["inspected_at"], "PASS", "", 12.0, 5.0, 0.01, "", None, "0.2.0", "abc", None,
                     "batch", prev, rh))
        prev = rh
    con.commit()
    con.close()

    with InspectionDB(path) as db:
        assert db.migrated_from == 0
        v = db.verify_chain()
        assert v["ok"] and v["checked"] == 3
        assert db.lots()[0]["status"] == "open"
        db.add_inspection(inspect_array(_img(1), "NEW", SPEC), "L-NEW", SPEC)  # new rows use the new canonical form
        v = db.verify_chain()
        assert v["ok"] and v["checked"] == 4
        assert db.conn.execute("PRAGMA user_version").fetchone()[0] == 2
    with InspectionDB(path) as db:  # reopening is idempotent
        assert db.migrated_from is None and db.verify_chain()["ok"]


def test_since_until_validation_and_evidence_names(tmp_path):
    with InspectionDB(tmp_path / "t.db") as db:
        with pytest.raises(ValueError, match="ISO-8601"):
            db.inspections(since="14/10/2026")
        with pytest.raises(ValueError, match="earlier"):
            db.inspections(since="2026-10-15", until="2026-10-14")
        st = Station(SPEC, db, InMemoryBus(), lot_id="L", image_dir=tmp_path / "ev")
        st.process("2026/10/14-001", np.full((256, 256), 30, np.uint8))  # ERROR frame with a barcode-style serial
        rows = db.inspections("L")
        assert rows[0]["serial"] == "2026/10/14-001"
        assert (tmp_path / "ev").is_dir() and list((tmp_path / "ev").glob("14-001_*.png"))


def test_api_freeze_under_lot_spec_and_window_validation(tmp_path):
    other = dataclasses.replace(SPEC, part_number="OTHER-1")
    with InspectionDB(tmp_path / "a.db") as db:
        for i in range(12):
            db.add_inspection(inspect_array(render_part(other, np.random.default_rng(i), od_mm=12, pocket_mm=5), f"O{i}", other), "L-O", other)
    app = create_app(db_path=str(tmp_path / "a.db"), spec=SPEC)
    c = TestClient(app)
    r = c.post("/spc/limits/freeze?lot_id=L-O", json={"characteristic": "od_mm", "frozen_by": "qa", "subgroup": 5})
    assert r.status_code == 200 and r.json()["part_number"] == "OTHER-1"
    assert c.get("/spc/limits").json() == {}  # app spec part has none
    assert "od_mm" in c.get("/spc/limits?lot_id=L-O").json()
    s = c.get("/lots/L-O/summary?subgroup=2").json()
    assert s["spc"]["od_mm"]["limits_source"] == "frozen:n=5"
    assert c.get("/lots/L-O/summary?since=yesterday").status_code == 422
    assert c.get("/reports/shift?lot_id=L-O&since=2026-10-15T00:00:00Z&until=2026-10-14T00:00:00Z").status_code == 422
    assert c.post("/spc/limits/freeze?lot_id=NOPE", json={"characteristic": "od_mm", "frozen_by": "qa"}).status_code == 404
    app.state.db.close()


# -- anomaly / datasets / cli ----------------------------------------------------------


def test_train_argument_validation_and_model_path(tmp_path):
    imgs = [_img(i) for i in range(4)]
    with pytest.raises(ValueError, match="names"):
        train(imgs, names=["a", "b"], coreset=200)
    with pytest.raises(ValueError, match="cannot exceed coreset"):
        train(imgs, coreset=2, k=3)
    with pytest.warns(UserWarning):
        m = train(imgs, coreset=200)
    m.save(tmp_path / "m")
    assert m.path == tmp_path / "m.npz" == model_file(tmp_path / "m")
    assert AnomalyModel.load(m.path).threshold == m.threshold


def test_import_survives_bad_file_and_keeps_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.fromarray(_img(0)).save(src / "a.png")
    (src / "b.png").write_bytes(b"not an image")
    Image.fromarray(_img(1)).save(src / "c.png")
    (src / "sub").mkdir()
    with pytest.warns(UserWarning, match="unreadable"):
        out = import_images(src, tmp_path / "dst")
    assert [p.name for p in out] == ["IMP-00001.png", "IMP-00002.png"]
    assert (tmp_path / "dst" / "import_manifest.csv").read_text().count("\n") == 3
    (src / "b.png").unlink()
    assert import_images(src, tmp_path / "dst") == []  # nothing re-imported, no duplicate serials


def test_minimal_truth_csv_and_bad_export_window(tmp_path, capsys):
    from qinspect.cli import main
    from qinspect.report import run_batch

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(3):
        Image.fromarray(_img(i)).save(imgs / f"P-{i}.png")
    pd.DataFrame({"serial": [f"P-{i}" for i in range(3)], "expected_verdict": ["PASS"] * 3}).to_csv(tmp_path / "t.csv", index=False)
    s = run_batch(imgs, tmp_path / "run", SPEC, truth_csv=tmp_path / "t.csv", db_path=tmp_path / "r.db")
    assert s["vs_ground_truth"]["accuracy"] == 1.0 and "defect_type_accuracy" not in s["vs_ground_truth"]
    assert s["audit"]["ok"]
    with pytest.raises(ValueError, match="lacks columns"):
        pd.DataFrame({"serial": ["x"]}).to_csv(tmp_path / "bad.csv", index=False)
        run_batch(imgs, tmp_path / "run2", SPEC, truth_csv=tmp_path / "bad.csv")
    with pytest.raises(SystemExit):
        main(["export", "--db", str(tmp_path / "r.db"), "--since", "14/10/2026", "--out", str(tmp_path / "x.csv")])

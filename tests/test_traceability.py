"""DB + audit chain, simulated line, REST API."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from qinspect.api import create_app
from qinspect.db import InspectionDB
from qinspect.inspector import inspect_array
from qinspect.line import InMemoryBus, Station, run_line
from qinspect.spec import PartSpec
from qinspect.synth import render_part

SPEC = PartSpec()


def _img(seed, **kw):
    kw.setdefault("od_mm", 12.0)
    kw.setdefault("pocket_mm", 5.0)
    return render_part(SPEC, np.random.default_rng(seed), **kw)


def test_audit_chain_detects_tampering(tmp_path):
    db = InspectionDB(tmp_path / "t.db")
    for i in range(5):
        db.add_inspection(inspect_array(_img(i, defects=("dent",) if i == 2 else ()), f"S{i}", SPEC), "L1", SPEC)
    assert db.verify_chain() == {"ok": True, "checked": 5, "head": db.verify_chain()["head"]}
    assert db.lot_summary("L1")["n_ng"] == 1

    db.conn.execute("UPDATE inspections SET verdict='PASS', reasons='' WHERE serial='S2'")
    db.conn.commit()
    v = db.verify_chain()
    assert v["ok"] is False and v["serial"] == "S2"


def test_station_raises_alarms_and_publishes(tmp_path):
    db, bus = InspectionDB(tmp_path / "s.db"), InMemoryBus()
    st = Station(SPEC, db, bus, lot_id="L", subgroup=2, baseline_subgroups=2)
    for i in range(6):
        st.process(f"P{i}", _img(i))
    st.process("CRIT", _img(99, defects=("scratch",)))
    for i in range(6):  # sudden +0.3 mm shift -> SPC alarm
        st.process(f"D{i}", _img(200 + i, od_mm=12.05))
    kinds = {e["kind"] for e in db.events()}
    assert "alarm:critical_defect" in kinds
    assert "alarm:spc_violation" in kinds
    topics = {t for t, _ in bus.messages}
    assert topics == {"qinspect/line1/result", "qinspect/line1/alarm"}


def test_run_line_summary(tmp_path):
    s = run_line(30, tmp_path / "l.db", seed=3)
    assert s["n"] == 30 and s["audit"]["ok"]


def test_rest_api(tmp_path):
    app = create_app(db_path=str(tmp_path / "api.db"), spec=SPEC)
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"

    for i, kw in enumerate([{}, {"defects": ("scratch",)}, {}]):
        p = tmp_path / f"PART-{i}.png"
        Image.fromarray(_img(50 + i, **kw)).save(p)
        with open(p, "rb") as fh:
            r = c.post("/inspect", files={"file": (p.name, fh, "image/png")}, data={"lot_id": "L9"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["serial"] == f"PART-{i}"
        assert body["verdict"] == ("NG" if kw else "PASS")
    assert c.get("/lots/L9/summary").json()["n_ng"] == 1
    assert c.get("/serials/PART-1").json()["defects"][0]["kind"] == "scratch"
    assert c.get("/audit/verify").json()["ok"] is True
    assert c.get("/lots/NOPE/summary").status_code == 404
    assert c.post("/inspect", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 400

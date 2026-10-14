"""DB + audit chain, dispositions, lot close, simulated line, REST API, concurrency."""
import dataclasses
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from qinspect.api import create_app
from qinspect.db import GENESIS, InspectionDB, canonical_record
from qinspect.inspector import InspectionResult, inspect_array
from qinspect.line import InMemoryBus, SpcMonitor, Station, make_bus, run_line
from qinspect.spec import PartSpec
from qinspect.synth import render_part

SPEC = PartSpec()


def _img(seed, **kw):
    kw.setdefault("od_mm", 12.0)
    kw.setdefault("pocket_mm", 5.0)
    return render_part(SPEC, np.random.default_rng(seed), **kw)


def _res(serial, seed=0, **kw):
    return inspect_array(_img(seed, **kw), serial, SPEC)


# -- chain ------------------------------------------------------------------------


def test_empty_db_verifies(tmp_path):
    with InspectionDB(tmp_path / "e.db") as db:
        assert db.verify_chain() == {"ok": True, "checked": 0, "head": GENESIS}


def test_audit_chain_detects_modification_and_deletions(tmp_path):
    with InspectionDB(tmp_path / "t.db") as db:
        for i in range(5):
            db.add_inspection(_res(f"S{i}", i, defects=("dent",) if i == 2 else ()), "L1", SPEC)
        v = db.verify_chain()
        assert v["ok"] and v["checked"] == 5 and len(v["head"]) == 64
        assert db.lot_summary("L1")["n_ng"] == 1

        db.conn.execute("UPDATE inspections SET verdict='PASS', reasons='' WHERE serial='S2'")
        v = db.verify_chain()
        assert v["ok"] is False and v["serial"] == "S2"
        db.conn.execute("UPDATE inspections SET verdict='NG', reasons='surface_dent:major' WHERE serial='S2'")
        assert db.verify_chain()["ok"]

        # deleting the newest row is invisible to the chain alone but caught by the anchor
        db.anchor_head("shift end")
        db.conn.execute("DELETE FROM defects WHERE inspection_id=(SELECT MAX(id) FROM inspections)")
        db.conn.execute("DELETE FROM inspections WHERE id=(SELECT MAX(id) FROM inspections)")
        v = db.verify_chain()
        assert v["ok"] is False and v["anchor_mismatch"]["anchored_n"] == 5 and v["checked"] == 4
        # deleting a middle row breaks the chain itself
        db.conn.execute("DELETE FROM inspections WHERE serial='S1'")
        assert db.verify_chain()["ok"] is False


def test_non_finite_and_numpy_values_do_not_break_the_chain(tmp_path):
    with InspectionDB(tmp_path / "n.db") as db:
        r = InspectionResult("S", "2026-01-01T00:00:00Z", "PASS", [], od_mm=np.float64(12.0), pocket_mm=float("nan"),
                             concentricity_mm=-0.0, anomaly_score=float("inf"))
        db.add_inspection(r, "L", SPEC)
        row = db.inspections("L")[0]
        assert row["pocket_mm"] is None and row["anomaly_score"] is None and row["concentricity_mm"] == 0.0
        assert db.verify_chain()["ok"]
    assert '"pocket_mm":null' in canonical_record({"pocket_mm": float("nan"), "defects": []})


def test_concurrent_writers_never_fork_the_chain(tmp_path):
    path = tmp_path / "c.db"
    dbs = [InspectionDB(path) for _ in range(3)]
    base = _res("X", 1)
    errors = []

    def writer(db, tag):
        for i in range(20):
            r = dataclasses.replace(base, serial=f"{tag}-{i}")
            try:
                db.add_inspection(r, "L", SPEC)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

    threads = [threading.Thread(target=writer, args=(dbs[i % 3], f"T{i}")) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    v = dbs[0].verify_chain()
    assert v["ok"] and v["checked"] == 120
    for db in dbs:
        db.close()


def test_lot_is_bound_to_one_spec_revision(tmp_path):
    with InspectionDB(tmp_path / "s.db") as db:
        db.add_inspection(_res("A"), "L", SPEC)
        assert db.lot_spec("L") == SPEC
        with pytest.raises(ValueError, match="bound to spec"):
            db.add_inspection(_res("B"), "L", dataclasses.replace(SPEC, revision="B"))
        db.add_inspection(_res("B"), "L2", dataclasses.replace(SPEC, revision="B"))
        assert {l["lot_id"] for l in db.lots()} == {"L", "L2"}


# -- dispositions / yields -------------------------------------------------------------


def test_reinspection_dispositions_and_yields(tmp_path):
    with InspectionDB(tmp_path / "d.db") as db:
        ng = _res("SN-1", 3, defects=("scratch",))
        assert ng.verdict == "NG"
        first_id = db.conn.execute("SELECT COALESCE(MAX(id),0) FROM inspections").fetchone()[0] + 1
        db.add_inspection(ng, "L", SPEC)
        db.add_inspection(_res("SN-2", 4), "L", SPEC)
        s = db.lot_summary("L")
        assert s["n_ng"] == 1 and s["open_ng"] == 1 and s["first_pass_yield"] == 0.5
        with pytest.raises(ValueError):
            db.add_disposition(first_id, "melt", "qa")
        db.add_disposition(first_id, "rework", "qa", note="polish")
        assert db.lot_summary("L")["open_ng"] == 0
        with pytest.raises(ValueError, match="same serial"):
            db.add_inspection(_res("SN-9", 5), "L", SPEC, reinspection_of=first_id)
        db.add_inspection(_res("SN-1", 5), "L", SPEC, reinspection_of=first_id)
        s = db.lot_summary("L")
        assert s["n_serials"] == 2 and s["first_pass_yield"] == 0.5 and s["final_yield"] == 1.0
        hist = db.serial_history("SN-1")
        assert [h["verdict"] for h in hist] == ["NG", "PASS"] and hist[0]["dispositions"][0]["decision"] == "rework"

        out = db.close_lot("L", "ACCEPT", {"n": 2}, ["SN-1", "SN-2"], 0, 0, "qa")
        assert out["chain_head"] == db.verify_chain()["head"]
        assert db.lots()[0]["status"] == "closed"
        with pytest.raises(ValueError, match="closed"):
            db.add_inspection(_res("SN-3", 6), "L", SPEC)
        assert db.verify_chain()["anchor"]["n_records"] == 3


def test_error_frames_excluded_from_yield(tmp_path):
    with InspectionDB(tmp_path / "e.db") as db:
        db.add_inspection(inspect_array(np.full((256, 256), 30, np.uint8), "E1", SPEC), "L", SPEC)
        db.add_inspection(_res("P1"), "L", SPEC)
        s = db.lot_summary("L")
        assert s["n_error"] == 1 and s["n"] == 1 and s["yield"] == 1.0 and s["ng_pareto"] == {}


# -- line -----------------------------------------------------------------------------


def test_spc_monitor_phase1_then_alarms_once_per_subgroup():
    mon = SpcMonitor(subgroup=2, baseline_subgroups=3)
    rng = np.random.default_rng(0)
    alarms = []
    for v in rng.normal(10, 0.01, 6):
        alarms += mon.add(v)
    assert alarms == [] and mon.limits is not None
    for v in [10.2, 10.2]:
        alarms += mon.add(v)
    assert len(alarms) == 1 and "R1:beyond_3sigma" in alarms[0]["rules"] and alarms[0]["subgroup"] == 4
    assert mon.add(float("nan")) == []


def test_station_alarms_frozen_limits_and_evidence(tmp_path):
    db, bus = InspectionDB(tmp_path / "s.db"), InMemoryBus()
    st = Station(SPEC, db, bus, lot_id="L", subgroup=2, baseline_subgroups=2, operator="kim", image_dir=tmp_path / "ev")
    for i in range(6):
        st.process(f"P{i}", _img(i))
    st.process("CRIT", _img(99, defects=("scratch",)))
    st.process("EMPTY", np.full((256, 256), 30, np.uint8))
    for i in range(6):  # +0.05 mm shift -> SPC alarm
        st.process(f"D{i}", _img(200 + i, od_mm=12.05))
    kinds = {e["kind"] for e in db.events()}
    assert {"alarm:critical_defect", "alarm:spc_violation", "alarm:vision_error"} <= kinds
    topics = {t for t, _ in bus.messages}
    assert topics == {"qinspect/line1/result", "qinspect/line1/alarm"}
    assert sorted(p.name.split("_")[0] for p in (tmp_path / "ev").glob("*.png")) == ["CRIT", "EMPTY"]
    rows = db.inspections("L")
    assert all(r["operator"] == "kim" and r["image_sha256"] for r in rows)
    assert db.lot_summary("L")["n_error"] == 1

    frozen = st.freeze_current_limits("qa")
    assert set(frozen) == {"od_mm", "pocket_mm", "concentricity_mm"}
    st2 = Station(SPEC, db, InMemoryBus(), lot_id="L2")
    assert all(m.limits is not None for m in st2.spc.values())
    ev = db.events(prefix="alarm:", unacked_only=True)
    assert db.ack_event(ev[0]["id"], "lee", "checked") and not db.ack_event(ev[0]["id"], "lee")
    db.close()


def test_run_line_summary(tmp_path):
    s = run_line(30, tmp_path / "l.db", seed=3, baseline_subgroups=2)
    assert s["n_inspections"] == 30 and s["audit"]["ok"] and s["anchor"]["n_records"] == 30
    assert isinstance(make_bus(None), InMemoryBus)
    with pytest.raises(ValueError):
        make_bus("amqp://x")
    with pytest.raises(ConnectionError):
        make_bus("mqtt://127.0.0.1:1")


# -- API -------------------------------------------------------------------------------


def _upload(c, path, **data):
    with open(path, "rb") as fh:
        return c.post("/inspect", files={"file": (path.name, fh, "image/png")}, data=data, headers={"X-API-Key": "k"})


def test_rest_api(tmp_path):
    app = create_app(db_path=str(tmp_path / "api.db"), spec=SPEC, api_key="k", image_dir=str(tmp_path / "ev"))
    c = TestClient(app)
    assert c.get("/health").json()["auth"] is True

    ids = []
    for i, kw in enumerate([{}, {"defects": ("scratch",)}, {}]):
        p = tmp_path / f"PART-{i}.png"
        Image.fromarray(_img(50 + i, **kw)).save(p)
        r = _upload(c, p, lot_id="L9", operator="park")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["serial"] == f"PART-{i}" and body["verdict"] == ("NG" if kw else "PASS")
        ids.append(body)
    assert c.post("/inspect", files={"file": ("x.png", b"123", "image/png")}).status_code == 401
    assert c.post("/inspect", files={"file": ("x.png", b"not an image", "image/png")},
                  headers={"X-API-Key": "k"}).status_code == 400
    assert c.post("/inspect", files={"file": ("../../etc/passwd.png", open(tmp_path / "PART-0.png", "rb"), "image/png")},
                  data={"lot_id": "L9"}, headers={"X-API-Key": "k"}).json()["serial"] == "passwd"

    s = c.get("/lots/L9/summary").json()
    assert s["n_ng"] == 1 and s["open_ng"] == 1 and s["spc"] == {}  # 3 parts: not enough for two subgroups
    assert c.get("/lots/L9/summary?subgroup=1").status_code == 422
    assert c.get("/lots/NOPE/summary").status_code == 404
    ser = c.get("/serials/PART-1").json()
    assert ser["defects"][0]["kind"] == "scratch" and len(ser["history"]) == 1
    assert c.get("/serials/NOPE").status_code == 404
    assert len(c.get("/lots/L9/open-ng").json()) == 1

    iid = ser["id"]
    assert c.post(f"/inspections/{iid}/disposition", json={"decision": "scrap", "decided_by": "qa"}).status_code == 401
    r = c.post(f"/inspections/{iid}/disposition", json={"decision": "scrap", "decided_by": "qa"}, headers={"X-API-Key": "k"})
    assert r.status_code == 200 and c.get("/lots/L9/summary").json()["open_ng"] == 0
    assert c.post("/inspections/9999/disposition", json={"decision": "scrap", "decided_by": "qa"},
                  headers={"X-API-Key": "k"}).status_code == 404

    # evidence kept for the NG frame only
    assert len(list((tmp_path / "ev").glob("PART-1_*.png"))) == 1 and not list((tmp_path / "ev").glob("PART-0_*.png"))

    # a reinspection links to the original
    p = tmp_path / "PART-1.png"
    Image.fromarray(_img(77)).save(p)
    r = _upload(c, p, lot_id="L9", reinspection_of=str(iid))
    assert r.status_code == 200 and r.json()["verdict"] == "PASS"
    s = c.get("/lots/L9/summary?subgroup=2").json()
    assert s["final_yield"] == 1.0 and "od_mm" in s["spc"] and "pocket_mm" in s["spc"]

    # lot close with sampling + anchors, audit, shift report
    r = c.post("/lots/L9/close", json={"decided_by": "qa", "aql": 1.0}, headers={"X-API-Key": "k"})
    assert r.status_code == 200 and r.json()["decision"] in ("ACCEPT", "REJECT")
    assert c.get("/lots/L9/summary").json()["lot_dispositions"][0]["decision"] == r.json()["decision"]
    assert c.post("/lots/L9/close", json={"decided_by": "qa", "aql": 0.3}, headers={"X-API-Key": "k"}).status_code == 422
    assert c.get("/audit/verify").json()["ok"] is True
    rep = c.get("/reports/shift", params={"lot_id": "L9", "since": "2000-01-01T00:00:00Z", "until": "2100-01-01T00:00:00Z"}).json()
    assert rep["n_inspections"] == 5 and rep["chain"]["ok"]  # 3 parts + 'passwd' + reinspection
    app.state.db.close()


def test_api_mixed_concurrent_requests(tmp_path):
    app = create_app(db_path=str(tmp_path / "cc.db"), spec=SPEC)
    c = TestClient(app, raise_server_exceptions=False)
    p = tmp_path / "P.png"
    Image.fromarray(_img(1)).save(p)
    data = p.read_bytes()
    statuses = []

    def post(i):
        for j in range(5):
            r = c.post("/inspect", files={"file": (f"S{i}-{j}.png", data, "image/png")}, data={"lot_id": "L"})
            statuses.append(r.status_code)

    def get():
        for _ in range(10):
            statuses.append(c.get("/lots").status_code)
            r = c.get("/lots/L/summary")
            statuses.append(r.status_code if r.status_code != 404 else 200)

    threads = [threading.Thread(target=post, args=(i,)) for i in range(4)] + [threading.Thread(target=get) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(statuses) == {200}, statuses
    v = c.get("/audit/verify").json()
    assert v["ok"] and v["checked"] == 20
    app.state.db.close()

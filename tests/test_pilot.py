"""Pilot-readiness features: OPC UA bus, role-based API keys, chain rotation, spec approval,
FAI report, capture session, anomaly backends."""
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from qinspect.api import ROLES, create_app, parse_api_keys
from qinspect.capture import capture_session, frame_quality
from qinspect.cli import main
from qinspect.db import InspectionDB
from qinspect.fai import render_fai
from qinspect.inspector import inspect_array
from qinspect.line import Station, make_bus
from qinspect.spec import PartSpec
from qinspect.synth import render_part

SPEC = PartSpec()


def _img(seed=0, **kw):
    kw.setdefault("od_mm", 12.0)
    kw.setdefault("pocket_mm", 5.0)
    return render_part(SPEC, np.random.default_rng(seed), **kw)


# -- OPC UA ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def opcua_server():
    asyncua = pytest.importorskip("asyncua")
    from asyncua.sync import Server

    srv = Server()
    srv.set_endpoint("opc.tcp://127.0.0.1:48400/qinspect-test/")
    idx = srv.register_namespace("urn:qinspect")
    # like a PLC: the tag tree exists before the station connects (anonymous clients may not add nodes)
    station = srv.nodes.objects.add_object(idx, "plant").add_object(idx, "cam1")
    for name, init in (("result", ""), ("alarm", ""), ("heartbeat", ""), ("Serial", ""), ("Verdict", ""), ("Reasons", ""),
                       ("OdMm", 0.0), ("PocketMm", 0.0), ("ConcentricityMm", 0.0)):
        station.add_variable(idx, name, init).set_writable()
    srv.start()
    time.sleep(0.5)
    yield srv
    srv.stop()


def test_opcua_bus_writes_result_and_typed_tags(opcua_server, tmp_path):
    from asyncua.sync import Client

    bus = make_bus("opc.tcp://127.0.0.1:48400/qinspect-test/?root=plant")
    db = InspectionDB(tmp_path / "o.db")
    try:
        st = Station(SPEC, db, bus, lot_id="L", station_id="cam1")
        res = st.process("SN-7", _img(3, defects=("scratch",)))
        assert res.verdict == "NG" and bus.errors == 0
        assert bus.publish("qinspect/cam2/result", {"serial": "x"}) is False and bus.errors == 1  # unknown tag: counted, not fatal
    finally:
        bus.close()
    c = Client("opc.tcp://127.0.0.1:48400/qinspect-test/")
    c.connect()
    try:
        idx = c.get_namespace_index("urn:qinspect")
        node = c.nodes.objects.get_child([f"{idx}:plant", f"{idx}:cam1", f"{idx}:Verdict"])
        assert node.read_value() == "NG"
        od = c.nodes.objects.get_child([f"{idx}:plant", f"{idx}:cam1", f"{idx}:OdMm"]).read_value()
        assert abs(od - 12.0) < 0.02
        raw = json.loads(c.nodes.objects.get_child([f"{idx}:plant", f"{idx}:cam1", f"{idx}:result"]).read_value())
        assert raw["serial"] == "SN-7"
        alarm = json.loads(c.nodes.objects.get_child([f"{idx}:plant", f"{idx}:cam1", f"{idx}:alarm"]).read_value())
        assert alarm["kind"] == "critical_defect"
    finally:
        c.disconnect()
        db.close()
    with pytest.raises(ConnectionError):
        make_bus("opc.tcp://127.0.0.1:1/x?root=plant")


# -- roles -----------------------------------------------------------------------------


def test_role_keys(tmp_path):
    assert parse_api_keys("adm", "ins:inspector, q:quality,aud:auditor") == {"adm": "admin", "ins": "inspector", "q": "quality", "aud": "auditor"}
    with pytest.raises(ValueError):
        parse_api_keys(None, "x:boss")
    app = create_app(db_path=str(tmp_path / "r.db"), spec=SPEC, api_keys=parse_api_keys(None, "ins:inspector,q:quality,aud:auditor"))
    c = TestClient(app)
    assert c.get("/health").json()["roles"] == ["auditor", "inspector", "quality"]
    p = tmp_path / "P.png"
    Image.fromarray(_img(1, defects=("dent",))).save(p)

    def post(key):
        with open(p, "rb") as fh:
            return c.post("/inspect", files={"file": ("P.png", fh, "image/png")}, data={"lot_id": "L"}, headers={"X-API-Key": key})

    assert post("ins").status_code == 200
    assert post("aud").status_code == 403
    assert post("nope").status_code == 401
    iid = c.get("/serials/P").json()["id"]
    body = {"decision": "rework", "decided_by": "qa"}
    assert c.post(f"/inspections/{iid}/disposition", json=body, headers={"X-API-Key": "ins"}).status_code == 403
    assert c.post(f"/inspections/{iid}/disposition", json=body, headers={"X-API-Key": "q"}).status_code == 200
    assert c.post("/audit/anchor", headers={"X-API-Key": "q"}).status_code == 403
    assert c.post("/audit/anchor", headers={"X-API-Key": "aud"}).status_code == 200
    assert ROLES["admin"] >= ROLES["quality"] >= ROLES["inspector"]
    app.state.db.close()


# -- rotation / approvals --------------------------------------------------------------


def test_rotate_links_chains_and_requires_closed_lots(tmp_path):
    db = InspectionDB(tmp_path / "live.db")
    for i in range(4):
        db.add_inspection(inspect_array(_img(i), f"S{i}", SPEC), "L", SPEC)
    with pytest.raises(ValueError, match="close these lots"):
        db.rotate(tmp_path / "arch", "qa")
    db.close_lot("L", "ACCEPT", {}, ["S0"], 0, 0, "qa")
    old_head = db.verify_chain()["head"]
    out = db.rotate(tmp_path / "arch", "qa")
    assert out["new_genesis"] == old_head and (tmp_path / "arch").joinpath(out["archived_db"].split("/")[-1]).exists()
    v = db.verify_chain()
    assert v == {"ok": True, "checked": 0, "head": old_head, "genesis": old_head}
    db.add_inspection(inspect_array(_img(9), "N1", SPEC), "L2", SPEC)
    assert db.verify_chain()["checked"] == 1 and db.get_meta("previous_db_sha256")
    with InspectionDB(out["archived_db"]) as old:
        assert old.verify_chain()["ok"] and old.verify_chain()["head"] == old_head
    exp = json.loads((tmp_path / "arch" / sorted(p.name for p in (tmp_path / "arch").iterdir() if p.is_dir())[0] / "chain.json").read_text())
    assert exp["chain"]["head"] == old_head and exp["counts"]["inspections"] == 4
    db.close()


def test_spec_approval_gate(tmp_path):
    db = InspectionDB(tmp_path / "a.db", require_approved_spec=True)
    with pytest.raises(ValueError, match="not approved"):
        db.add_inspection(inspect_array(_img(1), "S", SPEC), "L", SPEC)
    db.approve_spec(SPEC, "qa-lead", "initial release")
    db.add_inspection(inspect_array(_img(1), "S", SPEC), "L", SPEC)
    assert db.spec_approvals()[0]["approved_by"] == "qa-lead"
    db.close()


# -- FAI / capture / CLI ---------------------------------------------------------------


def test_fai_report_and_cli(tmp_path, capsys):
    db_path = tmp_path / "f.db"
    with InspectionDB(db_path) as db:
        db.add_calibration(16.01, 12.0, gauge_id="RG-12", operator="kim")
        db.add_inspection(inspect_array(_img(2), "FA-1", SPEC), "L", SPEC, operator="kim")
        doc = render_fai(db, "FA-1", prepared_by="qa", customer="ACME")
    assert "Form 1" in doc and "FA-1" in doc and "RG-12" in doc and "합격" in doc
    with pytest.raises(ValueError):
        with InspectionDB(db_path) as db:
            render_fai(db, "NOPE")
    assert main(["fai", "--db", str(db_path), "--serial", "FA-1", "--out", str(tmp_path / "fai.md")]) == 0
    assert (tmp_path / "fai.md").read_text().startswith("# 초도품 검사 보고서")
    assert main(["spec-approve", "--db", str(db_path), "--by", "qa", "--note", "rev A"]) == 0
    capsys.readouterr()
    assert main(["spec-approve", "--db", str(db_path), "--list"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["approved_by"] == "qa"
    assert main(["rotate", "--db", str(db_path), "--archive-dir", str(tmp_path / "arch"), "--export-only"]) == 0
    assert (tmp_path / "arch" / "chain.json").exists()


def test_capture_session_with_fake_grabber(tmp_path):
    frames = iter([_img(i) for i in range(3)] + [np.full((256, 256), 30, np.uint8)])
    rows = capture_session(lambda: next(frames), tmp_path / "cap", 4, prefix="T", station="cam1")
    assert [r["serial"] for r in rows] == ["T-00001", "T-00002", "T-00003", "T-00004"]
    assert rows[0]["part_found"] and rows[0]["warnings"] == []
    assert not rows[3]["part_found"]
    assert (tmp_path / "cap" / "import_manifest.csv").read_text().count("\n") == 5
    q = frame_quality(np.clip(_img(0).astype(int) + 120, 0, 255).astype(np.uint8))
    assert any("과노출" in w for w in q["warnings"])


def test_anomaly_backend_option_and_compare(tmp_path):
    from qinspect.anomaly import AnomalyModel, cnn_available, compare_backends, train

    good = [_img(i) for i in range(6)]
    with pytest.warns(UserWarning):
        m = train(good, coreset=300, backend="handcrafted")
    m.save(tmp_path / "h.npz")
    assert AnomalyModel.load(tmp_path / "h.npz").backend == "handcrafted"
    with pytest.raises(ValueError):
        train(good, coreset=300, backend="hog")
    if not cnn_available():
        with pytest.raises(RuntimeError, match="torch"):
            train(good, coreset=300, backend="cnn")
    with pytest.warns(UserWarning):
        r = compare_backends(good[:4], good[4:], {"scratch": [_img(9, defects=("scratch",))]}, coreset=200)
    assert r["handcrafted"]["per_class"]["scratch"]["auroc"] == 1.0
    assert "skipped" in r["cnn"] or "per_class" in r["cnn"]

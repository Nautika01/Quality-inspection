"""CLI subcommands beyond `demo`, and the Streamlit dashboard driven headlessly."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from qinspect.cli import main
from qinspect.spec import PartSpec
from qinspect.synth import render_part

SPEC = PartSpec()


@pytest.fixture(scope="module")
def lot(tmp_path_factory):
    d = tmp_path_factory.mktemp("lot")
    assert main(["generate", "--out", str(d), "--n", "40", "--seed", "2", "--lot", "LOT-T"]) == 0
    return d


def test_generate_inspect_db_audit_export(lot, tmp_path, capsys):
    db = tmp_path / "q.db"
    assert main(["inspect", "--images", str(lot / "images"), "--truth", str(lot / "ground_truth.csv"),
                 "--out", str(tmp_path / "run"), "--db", str(db), "--baseline", "4", "--operator", "me"]) == 0
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["vs_ground_truth"]["n"] + summary["n_error_frames"] == 40 and summary["audit"]["ok"]
    capsys.readouterr()

    assert main(["audit", "--db", str(db), "--anchor"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["anchor_written"]["n_records"] == 40
    with pytest.raises(SystemExit):
        main(["audit", "--db", str(tmp_path / "missing.db")])  # never silently creates an empty DB
    assert not (tmp_path / "missing.db").exists()

    assert main(["export", "--db", str(db), "--lot", "LOT-T", "--out", str(tmp_path / "shift.csv")]) == 0
    text = (tmp_path / "shift.csv").read_text()
    assert text.count("\n") == 41 and "inspected_at_local" in text
    assert main(["export", "--db", str(db), "--lot", "LOT-T", "--out", str(tmp_path / "shift.json")]) == 0
    assert json.loads((tmp_path / "shift.json").read_text())["summary"]["lot_id"] == "LOT-T"
    capsys.readouterr()

    assert main(["disposition", "--db", str(db), "--list", "--lot", "LOT-T"]) == 0
    open_ng = json.loads(capsys.readouterr().out)
    if open_ng:
        assert main(["disposition", "--db", str(db), "--inspection-id", str(open_ng[0]["id"]), "--decision", "mrb",
                     "--by", "qa", "--mrb-ref", "MRB-1"]) == 0
        capsys.readouterr()
    assert main(["lot-close", "--db", str(db), "--lot", "LOT-T", "--by", "qa", "--seed", "1"]) == 0
    closed = json.loads(capsys.readouterr().out)
    assert closed["decision"] in ("ACCEPT", "REJECT") and closed["sampling"]["plan"]["code_letter"] == "D"


def test_train_eval_and_specdoc(lot, tmp_path, capsys):
    model = tmp_path / "m"  # no suffix on purpose
    assert main(["train-anomaly", "--images", str(lot / "images"), "--truth", str(lot / "ground_truth.csv"),
                 "--limit", "10", "--coreset", "500", "--out", str(model)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["path"] == str(model) and (tmp_path / "m.npz").exists() and len(info["hash"]) == 16

    assert main(["gage", "--parts", "3", "--operators", "2", "--trials", "2", "--seed", "1",
                 "--out-json", str(tmp_path / "g.json"), "--out-csv", str(tmp_path / "g.csv")]) == 0
    capsys.readouterr()
    assert main(["specdoc", "--gage", str(tmp_path / "g.json"), "--model", str(tmp_path / "m.npz"),
                 "--out", str(tmp_path / "std.md")]) == 0
    doc = (tmp_path / "std.md").read_text()
    assert "%GRR" in doc and "이상탐지 모델 해시" in doc and info["hash"] in doc


def test_sampling_and_calibrate(tmp_path, capsys):
    assert main(["sampling", "--lot-size", "1000", "--aql", "1.0", "--defectives", "1", "--criticals", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["iso2859_1"]["sample_size"] == 80 and out["iso2859_1"]["decision"] == "REJECT"
    with pytest.raises(SystemExit):
        main(["sampling", "--lot-size", "1000", "--aql", "0.3"])  # not a table column
    with pytest.raises(SystemExit):
        main(["sampling", "--lot-size", "1000", "--level", "IV"])

    # a plain gauge disk (no pocket) calibrates and is logged
    import dataclasses

    gauge = dataclasses.replace(SPEC, pocket_diameter=None)
    p = tmp_path / "gauge.png"
    Image.fromarray(render_part(gauge, np.random.default_rng(0), od_mm=12.0)).save(p)
    db = tmp_path / "cal.db"
    assert main(["calibrate", "--image", str(p), "--diameter-mm", "12", "--out", str(tmp_path / "spec.json"),
                 "--db", str(db), "--gauge-id", "G-1", "--operator", "kim"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert abs(out["px_per_mm"] - 16) < 0.05 and out["calibration_record_id"] == 1
    assert PartSpec.from_json(tmp_path / "spec.json").px_per_mm == pytest.approx(out["px_per_mm"], abs=1e-3)


def test_run_line_cli_uses_env_db(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("QINSPECT_DB", str(tmp_path / "env.db"))
    assert main(["run-line", "--n", "12", "--baseline", "2", "--freeze-by", "qa"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (tmp_path / "env.db").exists() and out["n_inspections"] == 12 and out["control_limits"]["od_mm"]


def test_dashboard_renders_headless(tmp_path):
    from streamlit.testing.v1 import AppTest

    from qinspect.line import run_line

    run_line(30, tmp_path / "d.db", seed=4, baseline_subgroups=2)
    os.environ["QINSPECT_DB"] = str(tmp_path / "d.db")
    try:
        at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "src", "qinspect", "dashboard.py"),
                               default_timeout=90)
        at.run()
    finally:
        os.environ.pop("QINSPECT_DB", None)
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "검사 수" in labels and "감사 체인" in labels
    assert any("관리도" in h.value for h in at.subheader)

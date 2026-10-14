import json

import pandas as pd

from qinspect.cli import main


def test_demo_end_to_end(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--n", "80", "--seed", "3", "--train-good", "12", "--coreset", "800",
                 "--gage-parts", "3"]) == 0

    summary = json.loads((out / "summary.json").read_text())  # strict JSON (no NaN) by construction
    df = pd.read_csv(out / "results.csv")
    assert len(df) == summary["n_images"] == 80
    assert summary["n_inspected"] + summary["n_error_frames"] == 80
    assert (out / "spc_od_mm.png").exists() and (out / "spc_pocket_mm.png").exists()
    assert (out / "anomaly_model.npz").exists() and summary["anomaly_model"]
    assert (out / "inspection_standard.md").exists() and (out / "gage_rr.json").exists()

    v = summary["vs_ground_truth"]
    assert v["accuracy"] >= 0.95
    assert v["confusion"]["fn_escape"] == 0
    assert v["defect_type_accuracy"] == 1.0
    assert df.anomaly_score.notna().all()  # hybrid mode really ran
    assert summary["gage_rr"]["od_mm"]["ndc"] >= 5
    assert summary["audit"]["ok"] and summary["audit"]["checked"] == 80
    assert summary["spc"]["od_mm"]["out_of_control"]  # tool-wear drift is injected after 60 % of the lot
    doc = (out / "inspection_standard.md").read_text()
    assert summary["anomaly_model"] in doc and "%GRR" in doc

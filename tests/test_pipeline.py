import json

import pandas as pd

from qinspect.cli import main


def test_demo_end_to_end(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--n", "60", "--seed", "3"]) == 0

    summary = json.loads((out / "summary.json").read_text())
    df = pd.read_csv(out / "results.csv")
    assert len(df) == summary["n_inspected"] == 60
    assert (out / "spc_outer_diameter.png").exists()
    assert summary["vs_ground_truth"]["accuracy"] >= 0.95
    assert summary["vs_ground_truth"]["confusion"]["fp_overkill"] <= 1

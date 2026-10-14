"""Guard band, defect grading, Gage R&R."""
import numpy as np
import pandas as pd
import pytest

from qinspect.inspector import inspect_array
from qinspect.msa import gage_rr_anova, simulate_gage_study
from qinspect.spec import DimensionSpec, PartSpec
from qinspect.synth import render_part

SPEC = PartSpec()


def test_guard_band_shrinks_acceptance():
    d = DimensionSpec(12.0, 11.9, 12.1)
    assert d.ok(12.097)
    assert not d.ok(12.097, guard_band=0.006)
    assert d.ok(12.093, guard_band=0.006)


def test_grading_thresholds():
    g = SPEC.grading
    assert g.grade("dent", 0.2) == "minor"
    assert g.grade("dent", 0.6) == "major"
    assert g.grade("scratch", 1.5) == "critical"
    assert g.grade("crack", 0.1) == "critical"


@pytest.mark.parametrize("seed", range(4))
def test_pit_is_minor_and_accepted_but_recorded(seed):
    img = render_part(SPEC, np.random.default_rng(300 + seed), od_mm=12.0, pocket_mm=5.0, defects=("pit",))
    res = inspect_array(img, "S", SPEC)
    assert res.verdict == "PASS"
    assert res.worst_grade == "minor"
    assert res.to_row()["n_defects"] == 1


def test_borderline_part_rejected_by_guard_band():
    # true OD 12.098 is inside the drawing tolerance but inside the guard band -> NG
    img = render_part(SPEC, np.random.default_rng(9), od_mm=12.098, pocket_mm=5.0)
    res = inspect_array(img, "S", SPEC)
    assert "od_out_of_tolerance" in res.reasons


def test_gage_rr_anova_recovers_known_variances():
    rng = np.random.default_rng(0)
    parts, ops, trials = 10, 3, 3
    part_eff = rng.normal(0, 0.05, parts)
    op_eff = np.array([-0.01, 0.0, 0.01])
    rows = [{"part": p, "operator": o, "trial": t, "value": 10 + part_eff[p] + op_eff[o] + rng.normal(0, 0.004)}
            for p in range(parts) for o in range(ops) for t in range(trials)]
    g = gage_rr_anova(pd.DataFrame(rows), tolerance=0.3)
    assert g.repeatability == pytest.approx(0.004, rel=0.5)
    assert 0.004 < g.reproducibility < 0.02
    assert g.part_variation == pytest.approx(part_eff.std(ddof=1), rel=0.4)
    assert g.ndc >= 5


def test_simulated_gage_study_runs():
    df, studies = simulate_gage_study(SPEC, n_parts=3, n_operators=2, n_trials=2, seed=2)
    assert len(df) == 12
    assert studies["od_mm"].pct_grr_tol < 30

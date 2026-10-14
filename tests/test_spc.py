import numpy as np
import pytest

from qinspect.spc import ControlLimits, capability, evaluate, ewma, western_electric, xbar_r


def test_xbar_r_limits_match_textbook_constants():
    data = np.array([[1, 2, 3, 4, 5]] * 4, dtype=float).ravel()  # xbar=3, R=4 everywhere
    ch = xbar_r(data, n=5)
    assert ch.center == pytest.approx(3.0)
    assert ch.ucl == pytest.approx(3 + 0.577 * 4)
    assert ch.r_ucl == pytest.approx(2.114 * 4)
    assert ch.sigma_within == pytest.approx(4 / 2.326)
    assert not ch.out_of_control
    with pytest.raises(ValueError):
        xbar_r(data, n=11)


def test_western_electric_rules():
    assert western_electric([0, 0, 3.5], 0, 1) == {2: ["R1:beyond_3sigma"]}
    assert "R2:2of3_beyond_2sigma" in western_electric([2.5, 0, 2.5], 0, 1)[2]
    assert "R3:4of5_beyond_1sigma" in western_electric([1.5, 1.5, 0, 1.5, 1.5], 0, 1)[4]
    v = western_electric([0.5] * 8, 0, 1)
    assert list(v) == [7] and v[7] == ["R4:8_same_side"]
    # patterns completed at the very start of the series are flagged too
    assert western_electric([2.5, 2.5, 0, 0], 0, 1) == {1: ["R2:2of3_beyond_2sigma"]}
    assert western_electric([1.5, 1.5, 1.5, 1.5, 0], 0, 1) == {3: ["R3:4of5_beyond_1sigma"]}
    # opposite sides never combine
    assert western_electric([2.5, -2.5, 0], 0, 1) == {}


def test_baseline_limits_catch_drift_and_report_first_violation():
    rng = np.random.default_rng(0)
    stable = rng.normal(10, 0.01, 100)
    drift = rng.normal(10, 0.01, 100) + np.linspace(0, 0.05, 100)
    ch = xbar_r(np.concatenate([stable, drift]), n=5, baseline=20)
    assert ch.out_of_control
    assert min(ch.violations) >= 20  # nothing flagged inside the stable baseline
    assert ch.first_violation_after_baseline() >= 21


def test_frozen_limits_evaluation_roundtrip():
    rng = np.random.default_rng(1)
    ch = xbar_r(rng.normal(10, 0.01, 125), n=5)
    lim = ControlLimits.from_dict(ch.limits.to_dict())
    assert lim == ch.limits
    assert not evaluate(rng.normal(10, 0.01, 50), lim).out_of_control
    assert evaluate(rng.normal(10.05, 0.01, 10), lim).out_of_control


def test_capability():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 20000)
    c = capability(x, -3, 3, sigma_within=1.2)
    assert c["cp"] == pytest.approx(6 / (6 * 1.2))
    assert c["cpk"] == pytest.approx(min(3 - x.mean(), x.mean() + 3) / 3.6)
    assert c["ppk"] == pytest.approx(1.0, abs=0.03)
    assert "cp" not in capability(x, -3, 3)
    flat = capability([1.0, 1.0, 1.0], 0, 2, sigma_within=0.0)
    assert flat["pp"] is None and flat["cp"] is None and flat["cpk"] is None
    assert capability([1.0], 0, 2)["pp"] is None
    with pytest.raises(ValueError):
        capability(x, 3, -3)


def test_ewma_flags_small_shift_that_shewhart_misses():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(0, 1, 50), rng.normal(0.8, 1, 50)])
    e = ewma(x, 0, 1)
    assert e["first_signal"] is not None and e["first_signal"] >= 50

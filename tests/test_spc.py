import numpy as np
import pytest

from qinspect.spc import capability, western_electric, xbar_r


def test_xbar_r_limits_match_textbook_constants():
    data = np.array([[1, 2, 3, 4, 5]] * 4, dtype=float).ravel()  # xbar=3, R=4 everywhere
    ch = xbar_r(data, n=5)
    assert ch.center == pytest.approx(3.0)
    assert ch.ucl == pytest.approx(3 + 0.577 * 4)
    assert ch.r_ucl == pytest.approx(2.114 * 4)
    assert ch.sigma_within == pytest.approx(4 / 2.326)
    assert not ch.out_of_control


def test_western_electric_rules():
    assert western_electric([0, 0, 3.5], 0, 1) == {2: ["R1:beyond_3sigma"]}
    assert "R2:2of3_beyond_2sigma" in western_electric([2.5, 0, 2.5], 0, 1)[2]
    assert "R3:4of5_beyond_1sigma" in western_electric([1.5, 1.5, 0, 1.5, 1.5], 0, 1)[4]
    v = western_electric([0.5] * 8, 0, 1)
    assert list(v) == [7] and v[7] == ["R4:8_same_side"]


def test_baseline_limits_catch_drift():
    rng = np.random.default_rng(0)
    stable = rng.normal(10, 0.01, 100)
    drift = rng.normal(10, 0.01, 100) + np.linspace(0, 0.05, 100)
    ch = xbar_r(np.concatenate([stable, drift]), n=5, baseline=20)
    assert ch.out_of_control
    assert min(ch.violations) >= 20  # nothing flagged inside the stable baseline


def test_capability():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 20000)
    c = capability(x, -3, 3, sigma_within=1.0)
    assert c["cp"] == pytest.approx(1.0)
    assert c["ppk"] == pytest.approx(1.0, abs=0.03)

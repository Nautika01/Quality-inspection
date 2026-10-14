import numpy as np
import pytest
from scipy import stats

from qinspect.sampling import (AQLS, SAMPLE_SIZE, _LETTER_ORDER, code_letter, iso2859_single_normal, lot_disposition,
                               oc_curve, table_cell, zero_acceptance_plan)
from qinspect.spec import PartSpec
from qinspect.specdoc import render_inspection_standard


def test_code_letters():
    assert code_letter(1000) == "J"
    assert code_letter(500) == "H"
    assert code_letter(50, "I") == "C"
    assert code_letter(50, "III") == "E"
    assert code_letter(9, "I") == "A" and code_letter(16, "I") == "B"
    with pytest.raises(ValueError):
        code_letter(10, "IV")


@pytest.mark.parametrize("lot,aql,expected", [
    (1000, 1.0, ("J", 80, 2)),
    (100, 1.0, ("F", 20, 0)),
    (30, 1.0, ("D", 20, 0)),        # down arrow D -> F
    (200, 1.0, ("G", 20, 0)),       # up arrow G -> F
    (600000, 6.5, ("Q", 200, 21)),  # up arrow beyond Ac=21
    (100000, 2.5, ("N", 500, 21)),
    (600000, 2.5, ("Q", 500, 21)),
    (1000, 0.15, ("J", 125, 0)),
    (3000, 0.15, ("K", 125, 0)),
    (1000, 0.10, ("J", 200, 0)),
    (1000, 0.065, ("J", 315, 0)),
    (10000, 0.065, ("L", 315, 0)),
    (20000, 4.0, ("M", 315, 21)),
])
def test_iso2859_known_plans(lot, aql, expected):
    p = iso2859_single_normal(lot, aql)
    assert (p.code_letter, p.sample_size, p.accept) == expected
    assert p.reject == p.accept + 1


def test_iso2859_sample_capped_at_lot():
    p = iso2859_single_normal(10, 0.065)
    assert p.sample_size == 10 and "100 %" in p.note
    assert p.decide(0) == "ACCEPT" and p.decide(1) == "REJECT"


def test_iso2859_table_structure_producer_risk():
    """Every cell of Table II-A accepts an AQL-quality lot with probability >= 0.8 (ISO's design)."""
    for aql in AQLS:
        for letter in _LETTER_ORDER:
            use, ac = table_cell(letter, aql)
            pa = stats.binom.cdf(ac, SAMPLE_SIZE[use], aql / 100)
            assert pa >= 0.80, (aql, letter, use, ac, pa)
            if ac == 0:
                assert 0.18 <= SAMPLE_SIZE[use] * aql / 100 <= 0.21


def test_zero_acceptance_and_oc():
    p = zero_acceptance_plan(1000, 0.05, 0.90)
    assert p.accept == 0 and 40 <= p.sample_size <= 46
    assert zero_acceptance_plan(100, 0.07, 0.90).sample_size == 28  # exact D=7, not float round-up to 8
    x, pa = oc_curve(p)
    assert pa[0] == 1.0 and pa[-1] < 0.01
    small = zero_acceptance_plan(20, 0.1, 0.9)
    _, pa_small = oc_curve(small, np.array([0.0, 0.1]))
    assert pa_small[1] <= 0.1 + 1e-9  # hypergeometric: plan meets its own confidence on the finite lot


def test_lot_disposition_critical_overrides_plan():
    p = iso2859_single_normal(1000, 1.0)
    assert lot_disposition(p, 1, 0)["decision"] == "ACCEPT"
    assert lot_disposition(p, 1, 1)["decision"] == "REJECT"
    assert lot_disposition(p, 3, 0)["decision"] == "REJECT"


def test_specdoc_mentions_guard_band_plan_and_codes():
    spec = PartSpec()
    gage = {"od_mm": {"characteristic": "outer diameter", "pct_grr_tv": 20.0, "pct_grr_tol": 8.6, "ndc": 31,
                      "verdict": "acceptable", "suggested_guard_band_mm": 0.0067}}
    doc = render_inspection_standard(spec, lot_size=1000, aql=1.0, program_version="t", gage=gage, model_hash="abc123",
                                     control_limits={"od_mm": {"n": 5, "center": 12.0, "ucl": 12.03, "lcl": 11.97, "r_bar": 0.05}})
    assert "11.908 ~ 12.092" in doc and "4.932 ~ 5.068" in doc
    assert "n = 80, Ac = 2, Re = 3" in doc
    assert "abc123" in doc and "0.0067 mm" in doc and "vision:part_not_found" in doc
    assert "| od_mm | 5 | 12.0000 |" in doc
    no_pocket = render_inspection_standard(PartSpec(pocket_diameter=None))
    assert "포켓" not in no_pocket.split("## 3.")[0].split("## 2.")[1]

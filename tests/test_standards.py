import numpy as np
import pytest

from qinspect.sampling import code_letter, iso2859_single_normal, oc_curve, zero_acceptance_plan
from qinspect.spec import PartSpec
from qinspect.specdoc import render_inspection_standard


def test_code_letters():
    assert code_letter(1000) == "J"
    assert code_letter(500) == "H"
    assert code_letter(50, "I") == "C"
    assert code_letter(50, "III") == "E"


def test_iso2859_known_plans():
    p = iso2859_single_normal(1000, 1.0)
    assert (p.code_letter, p.sample_size, p.accept, p.reject) == ("J", 80, 2, 3)
    p = iso2859_single_normal(100, 1.0)  # F -> 20, Ac 0
    assert (p.sample_size, p.accept) == (20, 0)
    p = iso2859_single_normal(30, 1.0)  # D has an arrow -> F (n=20)
    assert (p.code_letter, p.sample_size, p.accept) == ("D", 20, 0) and "arrow" in p.note
    p = iso2859_single_normal(10, 0.065)  # arrow lands on L (200) > lot -> 100 %
    assert p.sample_size == 10 and "100 %" in p.note
    assert p.decide(0) == "ACCEPT" and p.decide(1) == "REJECT"


def test_zero_acceptance_and_oc():
    p = zero_acceptance_plan(1000, 0.05, 0.90)
    assert p.accept == 0 and 40 <= p.sample_size <= 46  # ~45 for infinite lot
    x, pa = oc_curve(p)
    assert pa[0] == 1.0 and pa[-1] < 0.01


def test_specdoc_mentions_guard_band_and_plan():
    doc = render_inspection_standard(PartSpec(), lot_size=1000, aql=1.0, program_version="t")
    assert "11.906 ~ 12.094" in doc
    assert "n = 80, Ac = 2, Re = 3" in doc
    assert "해시" in doc

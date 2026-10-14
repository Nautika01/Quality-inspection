import numpy as np
import pytest

from qinspect.inspector import inspect_array
from qinspect.spec import PartSpec
from qinspect.synth import render_part

SPEC = PartSpec()


def _render(seed, **kw):
    rng = np.random.default_rng(seed)
    kw.setdefault("od_mm", 12.0)
    kw.setdefault("pocket_mm", 5.0)
    return render_part(SPEC, rng, **kw)


@pytest.mark.parametrize("seed", range(5))
def test_good_part_passes_with_accurate_dimensions(seed):
    od, pk = 12.03, 4.98
    res = inspect_array(_render(seed, od_mm=od, pocket_mm=pk), "S", SPEC)
    assert res.verdict == "PASS", res.reasons
    assert abs(res.od_mm - od) < 0.015
    assert abs(res.pocket_mm - pk) < 0.015
    assert res.concentricity_mm < 0.02


@pytest.mark.parametrize("kind", ["scratch", "dent"])
@pytest.mark.parametrize("seed", range(5))
def test_surface_defect_detected_and_classified(kind, seed):
    res = inspect_array(_render(100 + seed, defects=(kind,)), "S", SPEC)
    assert res.verdict == "NG"
    assert {g.defect.kind for g in res.defects} == {kind}


def test_out_of_tolerance_dimensions_flagged():
    res = inspect_array(_render(7, od_mm=12.2), "S", SPEC)
    assert "od_out_of_tolerance" in res.reasons
    res = inspect_array(_render(8, pocket_offset_mm=(0.2, 0.0)), "S", SPEC)
    assert "concentricity" in res.reasons
    assert abs(res.concentricity_mm - 0.2) < 0.015


def test_blank_image_is_ng_not_crash():
    res = inspect_array(np.full((256, 256), 30, dtype=np.uint8), "S", SPEC)
    assert res.verdict == "NG"
    assert res.reasons[0].startswith("vision:")

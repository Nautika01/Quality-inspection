"""Real-image robustness: input formats, framing, lighting, polarity, pocket-less parts."""
import dataclasses

import numpy as np
import pytest
from PIL import Image

from qinspect import synth
from qinspect.datasets import import_images, load_gray
from qinspect.inspector import inspect_array
from qinspect.spec import DimensionSpec, PartSpec
from qinspect.synth import render_part
from qinspect.vision import to_gray8

SPEC = PartSpec()


def _img(seed=0, **kw):
    kw.setdefault("od_mm", 12.0)
    kw.setdefault("pocket_mm", 5.0)
    return render_part(SPEC, np.random.default_rng(seed), **kw)


@pytest.mark.parametrize("convert", [
    lambda a: np.stack([a] * 3, -1),                       # RGB
    lambda a: np.dstack([a, a, a, np.full_like(a, 255)]),  # RGBA
    lambda a: a.astype(np.uint16) * 257,                   # 16-bit container
    lambda a: a.astype(np.uint16) * 16,                    # 12-bit data in 16-bit container
    lambda a: a / 255.0,                                   # float [0, 1]
    lambda a: (255 - a).astype(np.uint8),                  # bright background, dark part
])
def test_input_variants_measure_the_same_part(convert):
    base = _img(5)
    ref = inspect_array(base, "S", SPEC)
    res = inspect_array(convert(base), "S", SPEC)
    assert res.verdict == "PASS", res.reasons
    assert abs(res.od_mm - ref.od_mm) < 0.01
    assert abs(res.pocket_mm - ref.pocket_mm) < 0.012


def test_large_frame_small_part_is_measured_not_hallucinated():
    big = render_part(dataclasses.replace(SPEC, image_size=768), np.random.default_rng(1), od_mm=12.0, pocket_mm=5.0)
    res = inspect_array(big, "S", SPEC)
    assert res.verdict == "PASS" and abs(res.od_mm - 12.0) < 0.01


def test_clipped_part_and_empty_nest_are_errors():
    clipped = np.roll(_img(2), -40, axis=1)
    res = inspect_array(clipped, "S", SPEC)
    assert res.verdict == "ERROR" and res.reasons == ["vision:part_touches_border"]
    assert res.od_mm is None
    res = inspect_array(np.full((256, 256), 30, np.uint8), "S", SPEC)
    assert res.verdict == "ERROR" and res.reasons == ["vision:part_not_found"]
    res = inspect_array(np.zeros((10, 10, 7), np.uint8), "S", SPEC)
    assert res.verdict == "ERROR" and res.reasons == ["vision:bad_image"]


@pytest.mark.parametrize("gradient", [(-70, 0), (70, 0), (0, 60)])
def test_illumination_gradient_does_not_bias_concentricity(gradient):
    rng = np.random.default_rng(11)
    vals = [inspect_array(render_part(SPEC, rng, od_mm=12, pocket_mm=5, pocket_offset_mm=(0.085, 0), gradient=gradient),
                          "S", SPEC).concentricity_mm for _ in range(4)]
    assert all(abs(v - 0.085) < 0.012 for v in vals), vals


def test_bright_spot_is_recorded_but_not_rejected():
    img = _img(7).astype(float)
    yy, xx = np.mgrid[0:256, 0:256]
    img += 60 * np.exp(-((xx - 60) ** 2 + (yy - 128) ** 2) / (2 * 3**2))
    res = inspect_array(np.clip(img, 0, 255).astype(np.uint8), "S", SPEC)
    assert res.verdict == "PASS"
    assert [(g.defect.kind, g.rejecting) for g in res.defects] == [("bright_spot", False)]


def test_shallow_pocket_is_found(monkeypatch):
    monkeypatch.setattr(synth, "POCKET_LEVEL", 120.0)
    res = inspect_array(_img(3), "S", SPEC)
    assert res.verdict == "PASS" and abs(res.pocket_mm - 5.0) < 0.012


def test_part_without_pocket():
    sp = dataclasses.replace(SPEC, pocket_diameter=None)
    res = inspect_array(render_part(sp, np.random.default_rng(4), od_mm=12.0), "S", sp)
    assert res.verdict == "PASS" and res.pocket_mm is None and res.concentricity_mm is None
    res = inspect_array(render_part(sp, np.random.default_rng(4), od_mm=12.0, defects=("scratch",)), "S", sp)
    assert res.verdict == "NG" and res.reasons == ["surface_scratch:critical"]


def test_narrow_face_spec_renders_without_crashing():
    sp = dataclasses.replace(SPEC, pocket_diameter=DimensionSpec(9.0, 8.9, 9.1))
    for kind in ("dent", "pit", "scratch"):
        render_part(sp, np.random.default_rng(1), od_mm=12.0, pocket_mm=9.0, defects=(kind,))


def test_load_gray_handles_16bit_and_letterboxes(tmp_path):
    base = _img(8)
    p16 = tmp_path / "cam.tiff"
    Image.fromarray((base.astype(np.uint16) * 16)).save(p16)  # 12-bit data in 16-bit tiff
    a = load_gray(p16)
    assert a.dtype == np.uint8 and np.abs(a.astype(int) - base.astype(int)).max() <= 1  # 12-bit depth inferred
    wide = np.full((200, 400), 30, np.uint8)
    wide[:, 50:250] = base[28:228, 28:228]  # off-centre part in a non-square frame
    pw = tmp_path / "wide.png"
    Image.fromarray(wide).save(pw)
    sq = load_gray(pw, 256)
    assert sq.shape == (256, 256)
    assert inspect_array(sq, "S", dataclasses.replace(SPEC, px_per_mm=16 * 256 / 400)).verdict == "PASS"


def test_import_manifest_and_rerun(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    for i in range(3):
        Image.fromarray(_img(i)).save(src / f"p{i}.png")
    first = import_images(src, tmp_path / "imp")
    assert [p.name for p in first] == ["IMP-00001.png", "IMP-00002.png", "IMP-00003.png"]
    assert import_images(src, tmp_path / "imp") == []  # same photos: nothing new, serials unchanged
    Image.fromarray(_img(9)).save(src / "p9.png")
    again = import_images(src, tmp_path / "imp")
    assert [p.name for p in again] == ["IMP-00004.png"]
    assert (tmp_path / "imp" / "import_manifest.csv").read_text().count("\n") == 5  # header + 4


def test_spec_from_dict_validation():
    raw = SPEC.to_dict()
    raw["grading"]["critical_types"] = "crack"
    raw["comment"] = "hand edited"
    with pytest.warns(UserWarning, match="unknown keys"):
        sp = PartSpec.from_dict(raw)
    assert sp.grading.critical_types == ("crack",)
    assert sp.grading.grade("crack", 0.1) == "critical"
    raw["grading"]["critical_types"] = 5
    with pytest.raises(ValueError):
        PartSpec.from_dict(raw)
    bad = SPEC.to_dict()
    bad["outer_diameter"]["lsl"] = 13.0
    with pytest.raises(ValueError):
        PartSpec.from_dict(bad)


def test_to_gray8_rejects_garbage():
    from qinspect.vision import VisionError

    with pytest.raises(VisionError):
        to_gray8(np.zeros((0, 0), np.uint8))
    with pytest.raises(VisionError):
        to_gray8(np.array([[np.nan, 1.0]]))

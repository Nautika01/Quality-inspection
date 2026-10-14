import numpy as np
from PIL import Image

from qinspect.cli import main
from qinspect.datasets import AnomalyDataset, calibrate_px_per_mm, export_synthetic_as_mvtec, import_images
from qinspect.spec import PartSpec
from qinspect.synth import generate_lot, render_part


def test_calibrate_from_reference_part():
    spec = PartSpec()
    img = render_part(spec, np.random.default_rng(0), od_mm=12.0, pocket_mm=5.0)
    assert abs(calibrate_px_per_mm(img, 12.0) - spec.px_per_mm) < 0.02


def test_import_and_mvtec_export(tmp_path):
    spec = PartSpec()
    src = tmp_path / "photos"
    src.mkdir()
    for i in range(3):
        Image.fromarray(render_part(spec, np.random.default_rng(i), od_mm=12, pocket_mm=5)).convert("RGB").save(src / f"p{i}.jpg")
    (src / ".hidden.jpg").write_bytes(b"junk")
    files = import_images(src, tmp_path / "imp", size=256)
    assert len(files) == 3 and files[0].name == "IMP-00001.png"

    lot = tmp_path / "lot"
    df = generate_lot(spec, 40, lot, seed=1, drift_start=None)
    ds = export_synthetic_as_mvtec(lot / "ground_truth.csv", lot / "images", tmp_path / "mvtec")
    (tmp_path / "mvtec" / "test" / "empty").mkdir()
    (tmp_path / "mvtec" / "test" / ".DS_Store").mkdir()
    ds = AnomalyDataset.from_root(tmp_path / "mvtec")
    assert len(ds.train_good) > 0 and ds.test_bad and "empty" not in ds.test_bad
    clean = set(df[(df.expected_verdict == "PASS") & (df.defect_type == "none")].serial)
    assert all(p.stem in clean for p in ds.train_good + ds.test_good)  # minor-defect parts never count as 'good'
    assert main(["anomaly-eval", "--dataset", str(tmp_path / "mvtec"), "--coreset", "500", "--size", "256"]) == 0

import numpy as np
import pytest

from qinspect.anomaly import AnomalyModel, auroc, train
from qinspect.inspector import inspect_array
from qinspect.spec import PartSpec
from qinspect.synth import render_part

SPEC = PartSpec()


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(42)
    good = [render_part(SPEC, rng, od_mm=rng.normal(12, 0.02), pocket_mm=rng.normal(5, 0.012)) for _ in range(12)]
    return train(good, coreset=800, seed=0)


def test_auroc():
    assert auroc([0, 1, 2], [3, 4]) == 1.0
    assert auroc([0, 1, 2], [0, 1, 2]) == pytest.approx(0.5)
    assert np.isnan(auroc([], [1]))


def test_model_separates_defects_and_roundtrips(model, tmp_path):
    rng = np.random.default_rng(7)
    good = [model.score(render_part(SPEC, rng, od_mm=12, pocket_mm=5)) for _ in range(6)]
    bad = [model.score(render_part(SPEC, rng, od_mm=12, pocket_mm=5, defects=(k,))) for k in ("scratch", "dent") * 3]
    assert auroc(good, bad) >= 0.95
    assert max(good) < model.threshold < min(bad)

    h = model.save(tmp_path / "m")  # suffix added automatically
    assert (tmp_path / "m.npz").exists() and len(h) == 16
    loaded = AnomalyModel.load(tmp_path / "m.npz")
    assert loaded.threshold == model.threshold and loaded.has_pocket
    probe = render_part(SPEC, np.random.default_rng(1), od_mm=12, pocket_mm=5)
    assert loaded.score(probe) == pytest.approx(model.score(probe), rel=1e-5)


def test_hybrid_rule_only_flags_unexplained_anomaly(model):
    rng = np.random.default_rng(11)
    # explained: classical finds a minor pit -> stays PASS even though the model flags it
    pit = render_part(SPEC, rng, od_mm=12, pocket_mm=5, defects=("pit",))
    strict = AnomalyModel(model.bank, model.mu, model.sd, 0.0, model.train_scores)
    res = inspect_array(pit, "S", SPEC, strict)
    assert res.anomaly_score is not None and res.anomaly_score > strict.threshold
    assert res.verdict == "PASS" and res.defects and "anomaly_score" not in res.reasons
    # unexplained: clean part, forced flag -> NG with the anomaly reason only
    res = inspect_array(render_part(SPEC, rng, od_mm=12, pocket_mm=5), "S", SPEC, strict)
    assert res.verdict == "NG" and res.reasons == ["anomaly_score"]


def test_hybrid_mode_survives_unsegmentable_frames(model):
    res = inspect_array(np.full((256, 256), 30, np.uint8), "S", SPEC, model)
    assert res.verdict == "ERROR" and res.anomaly_score is None and res.anomaly_error is None
    clipped = np.roll(render_part(SPEC, np.random.default_rng(2), od_mm=12, pocket_mm=5), -40, axis=1)
    assert inspect_array(clipped, "S", SPEC, model).verdict == "ERROR"


def test_model_failure_is_recorded_not_fatal(model):
    class Broken:
        def predict(self, img):
            raise RuntimeError("gpu on fire")

    res = inspect_array(render_part(SPEC, np.random.default_rng(3), od_mm=12, pocket_mm=5), "S", SPEC, Broken())
    assert res.verdict == "PASS" and "gpu on fire" in res.anomaly_error


def test_train_validation():
    rng = np.random.default_rng(5)
    img = render_part(SPEC, rng, od_mm=12, pocket_mm=5)
    with pytest.raises(ValueError, match="distinct"):
        with pytest.warns(UserWarning, match="duplicate"):
            train([img] * 6, coreset=200)
    imgs = [render_part(SPEC, rng, od_mm=12, pocket_mm=5) for _ in range(4)]
    with pytest.raises(ValueError, match="image\\[2\\]"):
        train(imgs[:2] + [np.full((256, 256), 30, np.uint8)] + imgs[2:], coreset=200)
    with pytest.warns(UserWarning, match="rough"):
        m = train(imgs, coreset=200)
    assert m.threshold > np.median(m.train_scores) * 1.4  # spread floor keeps the threshold away from the median
    with pytest.raises(ValueError, match="k must"):
        AnomalyModel(m.bank, m.mu, m.sd, m.threshold, m.train_scores, k=len(m.bank) + 1)

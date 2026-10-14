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


def test_model_separates_defects(model, tmp_path):
    rng = np.random.default_rng(7)
    good = [model.score(render_part(SPEC, rng, od_mm=12, pocket_mm=5)) for _ in range(6)]
    bad = [model.score(render_part(SPEC, rng, od_mm=12, pocket_mm=5, defects=(k,))) for k in ("scratch", "dent") * 3]
    assert auroc(good, bad) >= 0.95
    assert max(good) < min(bad)

    h = model.save(tmp_path / "m.npz")
    loaded = AnomalyModel.load(tmp_path / "m.npz")
    assert len(h) == 16
    assert loaded.threshold == model.threshold
    assert loaded.score(render_part(SPEC, np.random.default_rng(1), od_mm=12, pocket_mm=5)) == pytest.approx(
        model.score(render_part(SPEC, np.random.default_rng(1), od_mm=12, pocket_mm=5)))


def test_hybrid_rule_only_flags_unexplained_anomaly(model):
    rng = np.random.default_rng(11)
    # explained: classical finds a minor pit -> stays PASS even if anomaly score is high
    pit = render_part(SPEC, rng, od_mm=12, pocket_mm=5, defects=("pit",))
    res = inspect_array(pit, "S", SPEC, model)
    assert res.anomaly_score is not None
    assert "anomaly_score" not in res.reasons
    # unexplained: force a flag on a clean part by lowering the threshold
    strict = AnomalyModel(model.bank, model.mu, model.sd, 0.0, model.train_scores)
    res = inspect_array(render_part(SPEC, rng, od_mm=12, pocket_mm=5), "S", SPEC, strict)
    assert res.verdict == "NG" and res.reasons == ["anomaly_score"]

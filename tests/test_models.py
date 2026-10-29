import pytest
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.tfidf_lr import TfidfBaseline
from civinlp.labels.taxonomy import RoadDefectClass

def test_rule_baseline():
    model = RuleBaseline()
    texts = ["pothole on main st", "broken sign", "random text"]
    preds = model.predict(texts)
    
    assert preds[0] == RoadDefectClass.POTHOLE
    assert preds[1] == RoadDefectClass.SIGNAGE
    assert preds[2] == RoadDefectClass.OTHER

def test_tfidf_baseline(tmp_path):
    model = TfidfBaseline(max_features=10)
    texts = ["pothole danger", "sign broken", "pothole big", "sign missing"]
    labels = ["pothole", "signage", "pothole", "signage"]
    
    model.train(texts, labels)
    preds = model.predict(["pothole here", "sign gone"])
    
    assert len(preds) == 2
    # Simple model on small data might not be perfect, but should run
    
    model_path = tmp_path / "model.joblib"
    model.save(model_path)
    assert model_path.exists()
    
    loaded = TfidfBaseline.load(model_path)
    assert loaded


def test_calibration_report_overconfident_model():
    import numpy as np
    from civinlp.models.calibrate import calibration_report

    rng = np.random.RandomState(0)
    classes = ["a", "b"]
    n = 400
    y = rng.randint(0, 2, n)
    # Predicts the right class 70% of the time but always with 0.99 confidence
    correct = rng.rand(n) < 0.7
    pred = np.where(correct, y, 1 - y)
    probs = np.where(pred[:, None] == np.arange(2), 0.99, 0.01)
    labels = [classes[i] for i in y]

    rep = calibration_report(probs[:200], labels[:200], probs[200:], labels[200:], classes)
    assert rep["ece_before"] > 0.2
    assert rep["ece_after"] < rep["ece_before"] / 2
    assert rep["temperature"] > 1  # overconfidence -> soften

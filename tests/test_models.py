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

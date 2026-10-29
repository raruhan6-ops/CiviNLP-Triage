import numpy as np
import pytest

from civinlp.models.calibrate import TemperatureScaling
from civinlp.models.transformer import TransformerClassifier


def test_transformer_init():
    # Smoke test for initialization
    # Use a tiny model for speed if possible, or just check class structure
    # Here we mock to avoid downloading full bert in CI test if we can,
    # but for real test we just ensure it runs.
    try:
        clf = TransformerClassifier(model_name="distilbert-base-uncased", num_labels=2)
        assert clf.model
    except OSError:
        pytest.skip("Model not found or no internet")


def test_calibration_logic():
    # Synthetic logits: 10 samples, 3 classes
    np.random.seed(42)
    logits = np.random.randn(10, 3)
    labels = np.random.randint(0, 3, 10)

    calibrator = TemperatureScaling()
    calibrator.fit(logits, labels)

    assert calibrator.temperature > 0

    probs = calibrator.predict(logits)
    assert probs.shape == (10, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

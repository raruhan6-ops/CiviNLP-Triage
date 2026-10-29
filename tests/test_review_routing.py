import math

import numpy as np
import pytest

from civinlp.eval.review_routing import route_by_confidence, routing_curve


def test_route_by_confidence():
    assert route_by_confidence([0.2, 0.7, 0.9], 0.7).tolist() == [True, False, False]


def test_routing_curve_tradeoff():
    y_true = ["a", "a", "b", "high", "high"]
    y_pred = ["a", "b", "b", "high", "a"]  # errors at index 1 and 4
    conf = [0.95, 0.4, 0.9, 0.99, 0.5]
    curve = routing_curve(
        y_true, y_pred, conf, thresholds=[0.0, 0.6], critical_class="high"
    ).set_index("threshold")

    # Threshold 0: everything automated
    assert curve.loc[0.0, "coverage"] == 1.0
    assert curve.loc[0.0, "auto_accuracy"] == pytest.approx(0.6)
    assert curve.loc[0.0, "critical_missed"] == pytest.approx(0.5)
    assert math.isnan(curve.loc[0.0, "review_accuracy"])

    # Threshold 0.6: both low-confidence errors routed to review
    assert curve.loc[0.6, "coverage"] == pytest.approx(0.6)
    assert curve.loc[0.6, "auto_accuracy"] == 1.0
    assert curve.loc[0.6, "errors_caught"] == 1.0
    assert curve.loc[0.6, "critical_missed"] == 0.0
    assert curve.loc[0.6, "n_review"] == 2


def test_routing_curve_no_errors():
    curve = routing_curve(["a"], ["a"], np.array([0.8]), thresholds=[0.5])
    assert math.isnan(curve.iloc[0]["errors_caught"])

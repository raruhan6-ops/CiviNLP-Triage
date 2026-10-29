"""
Uncertainty-based routing to manual review (RQ4).

Predictions with confidence below a threshold are sent to a human reviewer;
the rest are handled automatically. The trade-off is reported as a
selective-prediction curve: for each threshold, how much is automated
(coverage) and how accurate the automated part is.
"""

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def route_by_confidence(confidences: Sequence[float], threshold: float) -> np.ndarray:
    """Boolean mask: True where the prediction should go to manual review."""
    return np.asarray(confidences, dtype=float) < threshold


def routing_curve(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    thresholds: Optional[List[float]] = None,
    critical_class: Optional[str] = None,
) -> pd.DataFrame:
    """
    Selective-prediction table, one row per threshold.

    Columns:
        coverage:        fraction handled automatically.
        auto_accuracy:   accuracy on the automated part (NaN if none).
        review_accuracy: model accuracy on the routed part, i.e. how often the
                         reviewer would just confirm the model (NaN if none).
        errors_caught:   fraction of all model errors that were routed to review.
        critical_missed: (if critical_class) critical cases handled automatically
                         but predicted as something else, as a fraction of all
                         critical cases.
    """
    y_true = np.asarray([str(y) for y in y_true])
    y_pred = np.asarray([str(y) for y in y_pred])
    conf = np.asarray(confidences, dtype=float)
    correct = y_true == y_pred
    n_errors = int((~correct).sum())
    is_critical = y_true == critical_class if critical_class else None

    rows = []
    for t in thresholds or DEFAULT_THRESHOLDS:
        review = route_by_confidence(conf, t)
        auto = ~review
        row = {
            "threshold": t,
            "coverage": float(auto.mean()) if len(conf) else float("nan"),
            "n_review": int(review.sum()),
            "auto_accuracy": (
                float(correct[auto].mean()) if auto.any() else float("nan")
            ),
            "review_accuracy": (
                float(correct[review].mean()) if review.any() else float("nan")
            ),
            "errors_caught": (
                float((review & ~correct).sum() / n_errors)
                if n_errors
                else float("nan")
            ),
        }
        if is_critical is not None:
            n_crit = int(is_critical.sum())
            missed = (auto & is_critical & ~correct).sum()
            row["critical_missed"] = float(missed / n_crit) if n_crit else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)

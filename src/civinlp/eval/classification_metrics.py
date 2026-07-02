"""
Classification metrics for CiviNLP Triage.

Provides macro-F1, per-class F1, confusion matrix, critical recall,
and Expected Calibration Error (ECE) for model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
import logging

logger = logging.getLogger(__name__)


def compute_classification_report(
    y_true: List[str],
    y_pred: List[str],
    critical_class: Optional[str] = "high",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        critical_class: Label for the critical class (for critical recall). 
                        Set to None to skip.

    Returns:
        Dictionary with:
            - macro_f1: Macro-averaged F1 score.
            - per_class_f1: Dict mapping each class to its F1 score.
            - confusion_matrix: 2D array confusion matrix.
            - labels: Ordered list of class labels.
            - critical_recall: Recall for the critical class (if applicable).
            - classification_report_str: Full sklearn classification report string.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    labels = sorted(list(set(y_true) | set(y_pred)))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = {label: float(score) for label, score in zip(labels, per_class)}

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    result = {
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "classification_report_str": report_str,
    }

    # Critical recall
    if critical_class and critical_class in set(y_true):
        critical_true = [1 if y == critical_class else 0 for y in y_true]
        critical_pred = [1 if y == critical_class else 0 for y in y_pred]
        result["critical_recall"] = float(
            recall_score(critical_true, critical_pred, zero_division=0)
        )
    else:
        result["critical_recall"] = None

    logger.info(f"Classification: macro_f1={macro_f1:.4f}, critical_recall={result['critical_recall']}")
    return result


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well the predicted probabilities match the actual
    frequency of correct predictions, binned by confidence.

    Args:
        probs: Array of shape (n_samples, n_classes) with predicted probabilities.
        labels: Array of shape (n_samples,) with true class indices.
        n_bins: Number of confidence bins.

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum()

        if bin_size > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += (bin_size / n) * abs(avg_accuracy - avg_confidence)

    return float(ece)

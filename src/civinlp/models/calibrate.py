from typing import Any, Dict, List

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TemperatureScaling(BaseEstimator, RegressorMixin):
    """
    Scales logits by a temperature T to calibrate probabilities.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """
        Ideally, finds optimal T to minimize NLL on validation set.
        Simplistic implementation: just stores a scalar or optimizes via grid search.
        For rigorous scaling, one would use L-BFGS-B to optimize T.
        Here we implement a simple search.
        """
        best_nll = float("inf")
        best_t = 1.0

        # Grid search for T
        search_space = np.linspace(0.1, 5.0, 50)

        # One-hot labels

        for t in search_space:
            scaled_logits = logits / t
            # Softmax
            exp_logits = np.exp(
                scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
            )
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # NLL
            # Avoiding log(0)
            dataset_nll = -np.mean(
                np.log(probs[np.arange(len(labels)), labels] + 1e-12)
            )

            if dataset_nll < best_nll:
                best_nll = dataset_nll
                best_t = t

        self.temperature = best_t
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """
        Returns calibrated probabilities.
        """
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(
            scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
        )
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs


def calibration_report(
    val_probs: np.ndarray,
    val_labels: List[str],
    test_probs: np.ndarray,
    test_labels: List[str],
    classes: List[str],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Fit temperature scaling on validation probabilities and report test ECE
    before and after. Log-probabilities stand in for logits (softmax of
    log p / T equals p^(1/T) renormalised). Rows whose label the model never
    saw are dropped.
    """
    from civinlp.eval.classification_metrics import compute_ece

    index = {c: i for i, c in enumerate(classes)}

    def encode(probs, labels):
        keep = np.array([lab in index for lab in labels], dtype=bool)
        y = np.array([index[lab] for lab in labels if lab in index], dtype=int)
        return np.log(np.clip(probs[keep], 1e-12, 1.0)), y

    val_logits, y_val = encode(val_probs, val_labels)
    test_logits, y_test = encode(test_probs, test_labels)
    if len(y_val) == 0 or len(y_test) == 0:
        return {
            "ece_before": None,
            "ece_after": None,
            "temperature": None,
            "n_test": len(y_test),
        }

    scaler = TemperatureScaling().fit(val_logits, y_val)
    return {
        "ece_before": compute_ece(np.exp(test_logits), y_test, n_bins),
        "ece_after": compute_ece(scaler.predict(test_logits), y_test, n_bins),
        "temperature": float(scaler.temperature),
        "n_test": int(len(y_test)),
    }

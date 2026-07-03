import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from typing import Union

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
        best_nll = float('inf')
        best_t = 1.0
        
        # Grid search for T
        search_space = np.linspace(0.1, 5.0, 50)
        
        # One-hot labels
        n_classes = logits.shape[1]
        
        for t in search_space:
            scaled_logits = logits / t
            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # NLL
            # Avoiding log(0)
            dataset_nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12))
            
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
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

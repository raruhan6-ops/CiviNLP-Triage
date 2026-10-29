from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfBaseline:
    """
    TF-IDF + Logistic Regression baseline.

    Supports separate tasks for defect classification and severity prediction.
    """

    def __init__(self, max_features: int = 5000, task: str = "defect"):
        """
        Args:
            max_features: Maximum number of TF-IDF features.
            task: Either "defect" or "severity" to indicate the classification target.
        """
        self.task = task
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=max_features, stop_words="english"),
                ),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ]
        )
        self.classes_: Optional[List[str]] = None

    def train(self, texts: List[str], labels: List[str]):
        """Train the model."""
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.classes_)

    def predict(self, texts: List[str]) -> List[str]:
        """Predict labels for texts."""
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities."""
        return self.pipeline.predict_proba(texts)

    def predict_proba_with_labels(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict probabilities with class names.

        Returns:
            List of dicts mapping class_label -> probability for each text.
        """
        probs = self.predict_proba(texts)
        classes = self.pipeline.classes_
        return [{cls: float(prob) for cls, prob in zip(classes, row)} for row in probs]

    def get_top_features(self, n: int = 10) -> Dict[str, List[str]]:
        """
        Get top TF-IDF features per class for explainability.

        Returns:
            Dict mapping class_label -> list of top feature names.
        """
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        feature_names = tfidf.get_feature_names_out()

        top_features = {}
        for i, cls in enumerate(clf.classes_):
            coefs = clf.coef_[i] if len(clf.coef_.shape) > 1 else clf.coef_[0]
            top_idx = np.argsort(coefs)[-n:][::-1]
            top_features[cls] = [feature_names[j] for j in top_idx]

        return top_features

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "task": self.task}, path)

    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load model from disk."""
        data = joblib.load(path)
        # Handle both old format (bare pipeline) and new format (dict)
        if isinstance(data, dict):
            instance = cls(task=data.get("task", "defect"))
            instance.pipeline = data["pipeline"]
        else:
            instance = cls()
            instance.pipeline = data
        instance.classes_ = list(instance.pipeline.classes_)
        return instance

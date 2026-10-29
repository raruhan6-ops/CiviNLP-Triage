from typing import Any, Dict, List

from civinlp.labels.taxonomy import RoadSeverity
from civinlp.labels.weak_labeling import get_severity_label, get_weak_label


class RuleBaseline:
    """
    Simple keyword-based baseline using the weak labeling logic.
    """

    def __init__(self):
        pass

    def train(self, X, y):
        # Rules don't need training
        pass

    def predict(
        self, texts: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Predict defect class for a list of texts.
        """
        predictions = []
        for i, text in enumerate(texts):
            # If metadata available, use both text and mapped fields
            # Here we assume text contains all info or we just use text
            descriptor = ""
            if metadatas and i < len(metadatas):
                descriptor = metadatas[i].get("descriptor", "")

            # Note: get_weak_label expects (complaint_type, descriptor)
            # but here we might just have raw text.
            # If text is the only input, we pass it as complaint_type to search keywords.
            label = get_weak_label(complaint_type=text, descriptor=descriptor)
            predictions.append(label)

        return predictions

    def predict_severity(self, texts: List[str]) -> List[RoadSeverity]:
        """
        Predict severity for a list of texts using keyword heuristics.

        Args:
            texts: List of complaint text strings.

        Returns:
            List of RoadSeverity enum values.
        """
        return [get_severity_label(text) for text in texts]

    def predict_full(
        self, texts: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict both defect class and severity.

        Args:
            texts: List of complaint text strings.
            metadatas: Optional metadata dicts per text.

        Returns:
            List of dicts with keys: defect_class, severity, confidence.
            Confidence is always 1.0 for rule-based predictions.
        """
        defects = self.predict(texts, metadatas)
        severities = self.predict_severity(texts)

        return [
            {
                "defect_class": str(d),
                "severity": str(s),
                "confidence": 1.0,  # Rules are deterministic
            }
            for d, s in zip(defects, severities)
        ]

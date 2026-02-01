from typing import List, Dict, Any
import pandas as pd
from civinlp.labels.taxonomy import RoadSeverity

class RoadPriorityScorer:
    """
    Computes priority scores for road segments or areas.
    """
    def __init__(self, alpha=1.0, beta=2.0, gamma=0.5):
        self.alpha = alpha  # weight for count
        self.beta = beta    # weight for severity
        self.gamma = gamma  # weight for repeat rate # simplified
        
    def score_area(self, area_df: pd.DataFrame) -> float:
        """
        Score an area based on a DataFrame of complaints in that area.
        """
        if area_df.empty:
            return 0.0
            
        count = len(area_df)
        
        # Simple severity mapping
        severity_map = {
            RoadSeverity.HIGH: 3,
            RoadSeverity.MEDIUM: 2,
            RoadSeverity.LOW: 1,
            RoadSeverity.UNKNOWN: 1
        }
        
        # If 'predicted_severity' col exists, use it. Else default.
        if 'predicted_severity' in area_df.columns:
            avg_severity = area_df['predicted_severity'].map(severity_map).mean()
        else:
            avg_severity = 1.0
            
        # Uncertainty Penalty
        # If 'confidence' is low, we might want to deprioritize or flag for review.
        # Here: simple penalty if confidence is available.
        # penalty logic: priority -= delta * (1 - avg_confidence)
        if 'confidence' in area_df.columns:
            avg_conf = area_df['confidence'].mean()
            uncertainty_penalty = (1.0 - avg_conf) * 2.0  # Example factor
        else:
            uncertainty_penalty = 0.0

        # Simplified score with penalty
        score = (self.alpha * count) + (self.beta * avg_severity) - (self.gamma * uncertainty_penalty)
        return max(0.0, score)

    def rank_areas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group by area_id and return ranked areas.
        """
        if 'area_id' not in df.columns:
            raise ValueError("Dataframe must have 'area_id'")
            
        scores = []
        for area_id, group in df.groupby('area_id'):
            s = self.score_area(group)
            scores.append({"area_id": area_id, "priority_score": s, "count": len(group)})
            
        results = pd.DataFrame(scores).sort_values("priority_score", ascending=False)
        return results

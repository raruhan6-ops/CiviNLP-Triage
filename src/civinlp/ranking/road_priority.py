from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from civinlp.labels.taxonomy import RoadSeverity
import logging

logger = logging.getLogger(__name__)


class RoadPriorityScorer:
    """
    Computes priority scores for road segments or areas.

    Formula: priority = α·recent_count + β·avg_severity + γ·repeat_rate - δ·uncertainty_penalty
    """
    def __init__(self, alpha=1.0, beta=2.0, gamma=0.5, delta=1.0):
        self.alpha = alpha  # weight for count
        self.beta = beta    # weight for severity
        self.gamma = gamma  # weight for repeat rate
        self.delta = delta  # weight for uncertainty penalty

    def _severity_to_numeric(self, severity) -> float:
        """Map severity enum/string to numeric value."""
        severity_map = {
            RoadSeverity.HIGH: 3,
            RoadSeverity.MEDIUM: 2,
            RoadSeverity.LOW: 1,
            RoadSeverity.UNKNOWN: 1,
            "high": 3,
            "medium": 2,
            "low": 1,
            "unknown": 1,
        }
        return severity_map.get(severity, 1)

    def score_area(self, area_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Score an area based on a DataFrame of complaints in that area.

        Returns:
            Dict with priority_score and contributing factor details.
        """
        if area_df.empty:
            return {"priority_score": 0.0, "count": 0, "rationale": "No data"}

        count = len(area_df)

        # Average severity
        if 'predicted_severity' in area_df.columns:
            avg_severity = area_df['predicted_severity'].map(self._severity_to_numeric).mean()
        else:
            avg_severity = 1.0

        # Repeat rate: fraction of complaints from same area within 7 days of another
        repeat_rate = 0.0
        if 'created_at' in area_df.columns and len(area_df) > 1:
            try:
                dates = pd.to_datetime(area_df['created_at'], errors='coerce').dropna().sort_values()
                if len(dates) > 1:
                    diffs = dates.diff().dt.total_seconds() / 86400  # days
                    repeat_count = (diffs <= 7).sum()
                    repeat_rate = repeat_count / len(dates)
            except Exception:
                repeat_rate = 0.0

        # Uncertainty penalty
        if 'confidence' in area_df.columns:
            avg_conf = area_df['confidence'].mean()
            uncertainty_penalty = 1.0 - avg_conf
        else:
            uncertainty_penalty = 0.0

        # Priority formula
        score = (
            self.alpha * count
            + self.beta * avg_severity
            + self.gamma * repeat_rate
            - self.delta * uncertainty_penalty
        )
        score = max(0.0, score)

        rationale = (
            f"count={count} (×{self.alpha}), "
            f"avg_sev={avg_severity:.2f} (×{self.beta}), "
            f"repeat={repeat_rate:.2f} (×{self.gamma}), "
            f"unc_pen={uncertainty_penalty:.2f} (×{self.delta})"
        )

        return {
            "priority_score": score,
            "count": count,
            "avg_severity": avg_severity,
            "repeat_rate": repeat_rate,
            "uncertainty_penalty": uncertainty_penalty,
            "rationale": rationale,
        }

    def rank_areas(
        self,
        df: pd.DataFrame,
        time_window: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Group by area_id (and optionally time window) and return ranked areas.

        Args:
            df: DataFrame with at least 'area_id' column.
            time_window: Optional pandas frequency string (e.g., 'D' for daily, 'W' for weekly).
                         If provided, also groups by time window.

        Returns:
            DataFrame sorted by priority_score descending.
        """
        if 'area_id' not in df.columns:
            raise ValueError("Dataframe must have 'area_id'")

        # Optional time window grouping
        group_cols = ['area_id']
        if time_window and 'created_at' in df.columns:
            try:
                df = df.copy()
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                df['time_window'] = df['created_at'].dt.to_period(time_window).astype(str)
                group_cols.append('time_window')
            except Exception as e:
                logger.warning(f"Time window grouping failed: {e}")

        scores = []
        # Grouping by a list always yields tuple keys (pandas >= 2), even for one column
        for group_key, group in df.groupby(group_cols):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            result = self.score_area(group)
            result.update(dict(zip(group_cols, group_key)))
            scores.append(result)

        results = pd.DataFrame(scores).sort_values("priority_score", ascending=False)
        results = results.reset_index(drop=True)
        results['priority_rank'] = range(1, len(results) + 1)
        return results

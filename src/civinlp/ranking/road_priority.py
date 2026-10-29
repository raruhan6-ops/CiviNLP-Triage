import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from civinlp.labels.taxonomy import RoadSeverity

logger = logging.getLogger(__name__)


class RoadPriorityScorer:
    """
    Computes priority scores for road segments or areas.

    Formula: priority = α·count_norm + β·severity_norm + γ·repeat_rate - δ·uncertainty_penalty

    Every term is scaled to [0, 1] so the weights are comparable:
      - count_norm    = log(1 + count) / log(1 + max_count over ranked areas)
      - severity_norm = (avg_severity - 1) / 2   (low=1 -> 0, high=3 -> 1)
      - repeat_rate and uncertainty_penalty are already fractions.
    Without this, the raw count dominated (50 minor complaints beat 5 severe
    ones whatever beta was).
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5):
        self.alpha = alpha  # weight for count
        self.beta = beta  # weight for severity
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

    def score_area(
        self, area_df: pd.DataFrame, max_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Score an area based on a DataFrame of complaints in that area.

        Args:
            area_df: Complaints in the area.
            max_count: Largest area count in the set being ranked (for count
                normalisation). Defaults to this area's own count.

        Returns:
            Dict with priority_score and contributing factor details.
        """
        if area_df.empty:
            return {"priority_score": 0.0, "count": 0, "rationale": "No data"}

        count = len(area_df)

        # Average severity
        if "predicted_severity" in area_df.columns:
            avg_severity = (
                area_df["predicted_severity"].map(self._severity_to_numeric).mean()
            )
        else:
            avg_severity = 1.0

        # Repeat rate: fraction of complaints from same area within 7 days of another
        repeat_rate = 0.0
        if "created_at" in area_df.columns and len(area_df) > 1:
            try:
                dates = (
                    pd.to_datetime(area_df["created_at"], errors="coerce")
                    .dropna()
                    .sort_values()
                )
                if len(dates) > 1:
                    diffs = dates.diff().dt.total_seconds() / 86400  # days
                    repeat_count = (diffs <= 7).sum()
                    repeat_rate = repeat_count / len(dates)
            except Exception:
                repeat_rate = 0.0

        # Uncertainty penalty
        if "confidence" in area_df.columns:
            avg_conf = area_df["confidence"].mean()
            uncertainty_penalty = 1.0 - avg_conf
        else:
            uncertainty_penalty = 0.0

        max_count = max(max_count or count, count)
        count_norm = float(np.log1p(count) / np.log1p(max_count))
        severity_norm = (avg_severity - 1.0) / 2.0

        # Priority formula
        score = (
            self.alpha * count_norm
            + self.beta * severity_norm
            + self.gamma * repeat_rate
            - self.delta * uncertainty_penalty
        )
        score = max(0.0, score)

        rationale = (
            f"count={count} [{count_norm:.2f}] (×{self.alpha}), "
            f"avg_sev={avg_severity:.2f} [{severity_norm:.2f}] (×{self.beta}), "
            f"repeat={repeat_rate:.2f} (×{self.gamma}), "
            f"unc_pen={uncertainty_penalty:.2f} (×{self.delta})"
        )

        return {
            "priority_score": score,
            "count": count,
            "count_norm": count_norm,
            "avg_severity": avg_severity,
            "severity_norm": severity_norm,
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
        if "area_id" not in df.columns:
            raise ValueError("Dataframe must have 'area_id'")

        # Optional time window grouping
        # Parse timestamps once for the whole frame, not once per area
        if "created_at" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["created_at"]
        ):
            df = df.assign(created_at=pd.to_datetime(df["created_at"], errors="coerce"))

        group_cols = ["area_id"]
        if time_window and "created_at" in df.columns:
            try:
                df = df.copy()
                df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
                df["time_window"] = (
                    df["created_at"].dt.to_period(time_window).astype(str)
                )
                group_cols.append("time_window")
            except Exception as e:
                logger.warning(f"Time window grouping failed: {e}")

        scores = []
        groups = df.groupby(group_cols)
        max_count = int(groups.size().max()) if len(df) else 0
        # Grouping by a list always yields tuple keys (pandas >= 2), even for one column
        for group_key, group in groups:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            result = self.score_area(group, max_count=max_count)
            result.update(dict(zip(group_cols, group_key)))
            scores.append(result)

        results = pd.DataFrame(scores).sort_values("priority_score", ascending=False)
        results = results.reset_index(drop=True)
        results["priority_rank"] = range(1, len(results) + 1)
        return results

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BridgePriorityScorer:
    """
    Computes priority scores for bridges based on element conditions (NBI).

    Supports both single-bridge scoring and batch ranking across a DataFrame.
    """
    def __init__(self, severity_weight: float = 1.0, traffic_weight: float = 0.5):
        self.severity_weight = severity_weight
        self.traffic_weight = traffic_weight

    def calculate_severity(self, element_states: Dict[str, Any]) -> float:
        """
        Calculate severity from element states.

        Handles both single-rating and multi-element formats:
          - If 'primary_rating' present: uses 9 - rating.
          - If multiple element ratings: averages (9 - rating) across elements.

        NBI rating scale: 9=Excellent, 0=Failed. Inverted so higher = worse.
        """
        # Single primary rating
        if "primary_rating" in element_states:
            rating = float(element_states["primary_rating"])
            return max(0.0, 9.0 - rating)

        # Multiple element conditions (e.g., deck, superstructure, substructure)
        rating_keys = [k for k in element_states if "rating" in k.lower() or "cond" in k.lower()]
        if rating_keys:
            ratings = []
            for k in rating_keys:
                try:
                    ratings.append(float(element_states[k]))
                except (ValueError, TypeError):
                    continue
            if ratings:
                # Use worst (minimum) rating for severity
                worst_rating = min(ratings)
                return max(0.0, 9.0 - worst_rating)

        return 0.0  # No usable ratings

    def score_bridge(self, bridge_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single bridge.

        Args:
            bridge_record: Dict with bridge fields (deck_rating, etc.).

        Returns:
            Dict with priority_score and details.
        """
        # Element-based severity
        element_states = bridge_record.get("element_states", {})
        if element_states:
            severity = self.calculate_severity(element_states)
        else:
            # Fallback to individual rating columns
            rating_cols = {
                k: v for k, v in bridge_record.items()
                if "rating" in k.lower() and v is not None
            }
            if rating_cols:
                severity = self.calculate_severity(rating_cols)
            else:
                severity = 0.0

        # Traffic/importance factor
        traffic = bridge_record.get("traffic", bridge_record.get("adt", 0))
        try:
            traffic_factor = np.log1p(float(traffic))
        except (ValueError, TypeError):
            traffic_factor = 0.0

        score = self.severity_weight * severity + self.traffic_weight * traffic_factor

        return {
            "priority_score": float(score),
            "severity": float(severity),
            "traffic_factor": float(traffic_factor),
        }

    def rank_bridges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank all bridges in a DataFrame.

        Expects at minimum a bridge identifier column. Will look for
        rating columns (deck_rating, superstructure_rating, etc.).

        Args:
            df: DataFrame with bridge records.

        Returns:
            DataFrame sorted by priority_score descending, with priority_rank.
        """
        # Identify bridge ID column
        id_col = None
        for candidate in ["bridge_id", "STRUCTURE_NUMBER_008", "structure_number"]:
            if candidate in df.columns:
                id_col = candidate
                break
        if id_col is None:
            id_col = df.columns[0]
            logger.warning(f"No standard bridge ID column found; using '{id_col}'")

        scores = []
        for _, row in df.iterrows():
            record = row.to_dict()
            result = self.score_bridge(record)
            result["bridge_id"] = row[id_col]
            scores.append(result)

        results = pd.DataFrame(scores).sort_values("priority_score", ascending=False)
        results = results.reset_index(drop=True)
        results["priority_rank"] = range(1, len(results) + 1)
        return results

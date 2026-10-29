from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Condition rating fields: internal names (see preprocess_bridges) and raw NBI items
RATING_KEYS = {
    "deck_rating", "superstructure_rating", "substructure_rating", "culvert_rating",
    "DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060", "CULVERT_COND_062",
}


def _to_rating(value) -> Optional[float]:
    """Parse an NBI condition code (0-9); None for 'N', blanks, NaN or out-of-range."""
    try:
        rating = float(value)
    except (ValueError, TypeError):
        return None
    if np.isnan(rating) or not 0 <= rating <= 9:
        return None
    return rating


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
          - If multiple element ratings: uses 9 - worst (minimum) rating;
            'N', blank and NaN ratings are skipped.

        NBI rating scale: 9=Excellent, 0=Failed. Inverted so higher = worse.
        """
        # Single primary rating
        if "primary_rating" in element_states:
            rating = float(element_states["primary_rating"])
            return max(0.0, 9.0 - rating)

        # Multiple element conditions (e.g., deck, superstructure, substructure)
        rating_keys = [k for k in element_states if "rating" in k.lower() or "cond" in k.lower()]
        ratings = [r for r in (_to_rating(element_states[k]) for k in rating_keys) if r is not None]
        if ratings:
            # Use worst (minimum) rating for severity
            return max(0.0, 9.0 - min(ratings))

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
            # Fallback to individual rating columns (internal or raw NBI names)
            rating_cols = {
                k: v for k, v in bridge_record.items()
                if k in RATING_KEYS and _to_rating(v) is not None
            }
            severity = self.calculate_severity(rating_cols) if rating_cols else 0.0

        # Traffic/importance factor
        traffic = bridge_record.get("traffic", bridge_record.get("adt", bridge_record.get("ADT_029", 0)))
        try:
            traffic_factor = float(np.log1p(max(0.0, float(traffic))))
        except (ValueError, TypeError):
            traffic_factor = 0.0
        if np.isnan(traffic_factor):
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

        rating_cols = [c for c in df.columns if c in RATING_KEYS]
        if rating_cols:
            ratings = df[rating_cols].apply(pd.to_numeric, errors="coerce")
            ratings = ratings.where((ratings >= 0) & (ratings <= 9))
            worst = ratings.min(axis=1, skipna=True)
            severity = (9.0 - worst).clip(lower=0).fillna(0.0)
        else:
            logger.warning("No rating columns found; all severities are 0")
            severity = pd.Series(0.0, index=df.index)

        traffic_col = next((c for c in ["traffic", "adt", "ADT_029"] if c in df.columns), None)
        if traffic_col:
            traffic = pd.to_numeric(df[traffic_col], errors="coerce").clip(lower=0)
            traffic_factor = np.log1p(traffic).fillna(0.0)
        else:
            traffic_factor = pd.Series(0.0, index=df.index)

        results = pd.DataFrame({
            "bridge_id": df[id_col].values,
            "priority_score": (self.severity_weight * severity + self.traffic_weight * traffic_factor).values,
            "severity": severity.values,
            "traffic_factor": traffic_factor.values,
        })
        if "all_ratings_missing" in df.columns:
            results["all_ratings_missing"] = df["all_ratings_missing"].values

        results = results.sort_values("priority_score", ascending=False, kind="stable")
        results = results.reset_index(drop=True)
        results["priority_rank"] = range(1, len(results) + 1)
        return results

"""
Outcome-based ranking evaluation for CiviNLP Triage.

Rule-derived labels cannot be used to judge a ranking that is itself built on
those rules. Instead we use a forward-looking outcome that the rules never see:

    1. Split complaints at a time cutoff into a HISTORY and a FUTURE window.
    2. Rank areas using HISTORY only (priority score and simple baselines).
    3. Grade each ranking against what happened in FUTURE: how many new road
       complaints each area received (a proxy for persistent, unresolved need).

Only areas that appear in HISTORY can be ranked, so relevance is computed over
those areas; areas with no future complaints get relevance 0.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from civinlp.eval.ranking_metrics import ndcg_at_k, spearman_correlation, top_k_recall
from civinlp.ranking.road_priority import RoadPriorityScorer

logger = logging.getLogger(__name__)

EXCLUDED_AREAS = {"UNKNOWN", "", "nan", "None"}


def temporal_outcome_split(
    df: pd.DataFrame,
    time_column: str = "created_at",
    history_frac: float = 0.5,
    cutoff: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split rows into history (< cutoff) and future (>= cutoff).

    If `cutoff` is not given it is the `history_frac` quantile of `time_column`.
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame")

    times = pd.to_datetime(df[time_column], errors="coerce")
    valid = times.notna()
    if valid.sum() < 2:
        raise ValueError("Need at least two rows with valid timestamps")

    if cutoff is None:
        cutoff = times[valid].quantile(history_frac)
    cutoff = pd.Timestamp(cutoff)

    history = df[valid & (times < cutoff)].copy()
    future = df[valid & (times >= cutoff)].copy()
    logger.info(
        f"Outcome split at {cutoff}: history={len(history)}, future={len(future)}"
    )
    return history, future, cutoff


OUTCOMES = {
    "volume": "future road-complaint count per area (linear-gain NDCG)",
    "resolution_days": (
        "median days to close future complaints per area (still-open ones count "
        "as open until the end of the data; areas with too few future complaints "
        "excluded)"
    ),
}


def resolution_days(
    df: pd.DataFrame,
    time_column: str = "created_at",
    closed_column: str = "closed_date",
    snapshot: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Days from creation to closure. Unclosed requests are right-censored at
    `snapshot` (default: latest creation time in `df`), i.e. counted as open
    until then. Negative durations (data errors) are clipped to 0.
    """
    created = pd.to_datetime(df[time_column], errors="coerce")
    closed = pd.to_datetime(df[closed_column], errors="coerce")
    snapshot = pd.Timestamp(snapshot) if snapshot is not None else created.max()
    closed = closed.where(closed.notna(), snapshot)
    return ((closed - created).dt.total_seconds() / 86400).clip(lower=0)


def future_outcomes(
    history: pd.DataFrame,
    future: pd.DataFrame,
    area_column: str = "area_id",
    critical_frac: float = 0.2,
    outcome: str = "volume",
    min_future: int = 3,
) -> Tuple[Dict[str, float], set]:
    """
    Build independent ground truth from the future window.

    Args:
        outcome: "volume" (number of future complaints) or "resolution_days"
            (median days to close them; areas with fewer than `min_future`
            future complaints are left out, since their median is noise).

    Returns:
        relevance: area -> outcome value. Used directly as NDCG relevance with
                   linear gain (coarse grades saturate NDCG, and 2^rel - 1 gain
                   overflows on raw values).
        critical: areas in the top `critical_frac` by that value.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome '{outcome}'; choose from {list(OUTCOMES)}")

    areas = [
        a for a in history[area_column].astype(str).unique() if a not in EXCLUDED_AREAS
    ]
    future_areas = future[area_column].astype(str)
    future_counts = future_areas.value_counts()

    if outcome == "volume":
        counts = {a: float(future_counts.get(a, 0)) for a in areas}
    else:
        if "closed_date" not in future.columns:
            raise ValueError("resolution_days outcome needs a 'closed_date' column")
        medians = resolution_days(future).groupby(future_areas).median()
        counts = {
            a: float(medians[a])
            for a in areas
            if future_counts.get(a, 0) >= min_future and pd.notna(medians.get(a))
        }
        areas = list(counts)

    n_critical = max(1, int(round(len(areas) * critical_frac))) if areas else 0
    ordered = sorted(counts, key=lambda a: (-counts[a], a))
    critical = {a for a in ordered[:n_critical] if counts[a] > 0}
    return counts, critical


def baseline_rankings(
    history: pd.DataFrame,
    area_column: str = "area_id",
    time_column: str = "created_at",
    scorer: Optional[RoadPriorityScorer] = None,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Rank history areas with the priority score and simple heuristics.

      - priority:      RoadPriorityScorer (needs 'predicted_severity').
      - count_only:    number of complaints (H1).
      - recency:       most recent complaint first (H2).
      - severity_only: mean predicted severity (H3 / severity-only ablation).
      - random:        shuffled order (sanity floor).
    Ties are broken by area id so results are deterministic.
    """
    df = history[~history[area_column].astype(str).isin(EXCLUDED_AREAS)].copy()
    df[area_column] = df[area_column].astype(str)
    grouped = df.groupby(area_column)

    def order(series: pd.Series) -> List[str]:
        frame = series.rename("v").reset_index()
        frame = frame.sort_values(["v", area_column], ascending=[False, True])
        return frame[area_column].tolist()

    rankings = {"count_only": order(grouped.size())}

    if time_column in df.columns:
        times = pd.to_datetime(df[time_column], errors="coerce")
        rankings["recency"] = order(times.groupby(df[area_column]).max())

    if "predicted_severity" in df.columns:
        scorer = scorer or RoadPriorityScorer()
        sev = df["predicted_severity"].map(scorer._severity_to_numeric)
        rankings["severity_only"] = order(sev.groupby(df[area_column]).mean())
        ranked = scorer.rank_areas(df)
        rankings["priority"] = order(ranked.set_index(area_column)["priority_score"])

    rng = np.random.RandomState(seed)
    rankings["random"] = list(rng.permutation(sorted(grouped.groups.keys())))
    return rankings


def evaluate_rankings(
    rankings: Dict[str, List[str]],
    counts: Dict[str, float],
    critical: set,
    k_values: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Score each ranking against the future-window ground truth."""
    if k_values is None:
        k_values = [5, 10, 20]
    k_values = [k for k in k_values if k <= len(counts)] or [len(counts)]

    truth_order = sorted(counts, key=lambda a: (-counts[a], a))
    rows = []
    for name, ranked in rankings.items():
        ranked = [a for a in ranked if a in counts]
        row = {"ranking": name, "n_areas": len(ranked)}
        for k in k_values:
            row[f"recall@{k}"] = top_k_recall(ranked, critical, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, counts, k)
        row["spearman_vs_future"] = spearman_correlation(ranked, truth_order)
        rows.append(row)
    return pd.DataFrame(rows)


def run_outcome_evaluation(
    df: pd.DataFrame,
    area_column: str = "area_id",
    time_column: str = "created_at",
    history_frac: float = 0.5,
    critical_frac: float = 0.2,
    k_values: Optional[List[int]] = None,
    scorer: Optional[RoadPriorityScorer] = None,
    seed: int = 42,
    outcome: str = "volume",
) -> Tuple[pd.DataFrame, Dict]:
    """Full pipeline: split by time, build outcomes, rank, and evaluate."""
    history, future, cutoff = temporal_outcome_split(df, time_column, history_frac)
    counts, critical = future_outcomes(
        history, future, area_column, critical_frac, outcome=outcome
    )
    rankings = baseline_rankings(history, area_column, time_column, scorer, seed)
    results = evaluate_rankings(rankings, counts, critical, k_values)
    info = {
        "cutoff": str(cutoff),
        "history_rows": len(history),
        "future_rows": len(future),
        "rankable_areas": len(counts),
        "areas_with_positive_outcome": int(sum(c > 0 for c in counts.values())),
        "critical_areas": len(critical),
        "outcome": outcome,
        "relevance": OUTCOMES[outcome],
    }
    return results, info


def weight_sensitivity(
    df: pd.DataFrame,
    grid: Optional[List[Dict[str, float]]] = None,
    area_column: str = "area_id",
    time_column: str = "created_at",
    history_frac: float = 0.5,
    critical_frac: float = 0.2,
    k_values: Optional[List[int]] = None,
    outcome: str = "volume",
) -> pd.DataFrame:
    """
    Evaluate the priority score over a grid of weights on one fixed split.

    Each grid entry is a dict of RoadPriorityScorer kwargs (alpha, beta, gamma, delta).
    The default grid varies the severity weight beta with alpha fixed at 1.
    """
    if grid is None:
        grid = [{"alpha": 1.0, "beta": b} for b in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]]
        grid.append({"alpha": 0.0, "beta": 1.0})

    history, future, _ = temporal_outcome_split(df, time_column, history_frac)
    counts, critical = future_outcomes(
        history, future, area_column, critical_frac, outcome=outcome
    )

    rows = []
    for weights in grid:
        ranked = baseline_rankings(
            history, area_column, time_column, RoadPriorityScorer(**weights)
        )
        result = evaluate_rankings(
            {"priority": ranked["priority"]}, counts, critical, k_values
        )
        rows.append({**weights, **result.drop(columns=["ranking"]).iloc[0].to_dict()})
    return pd.DataFrame(rows)

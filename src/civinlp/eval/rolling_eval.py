"""
Rolling-origin ranking evaluation for CiviNLP Triage.

`outcome_eval` grades rankings on one history/future split, which gives a single
number with no error bar. Here the cutoff rolls forward through the data:

    fold i:  [cutoff_i - history_days, cutoff_i) -> rank areas
             [cutoff_i, cutoff_i + horizon_days) -> outcome per area

Folds are split chronologically into TUNING folds (early) and TEST folds (late).
Priority weights and the learned rankers are fit on tuning folds only and then
frozen; every number reported for test folds is out-of-time. With
step_days >= horizon_days the future windows do not overlap.

The question this answers: does text-derived severity add anything beyond
complaint volume? Compare `learned_volume` (volume features only) with
`learned_volume_severity` (same plus severity features), and every ranking with
`count_only`, using paired per-fold differences. The rule severity labels are
largely fixed by complaint type, and complaint types differ in how long the
agency takes to close them, so `learned_volume_type` vs
`learned_volume_type_severity` checks whether severity adds anything once the
complaint-type mix is known.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from civinlp.data.schema import ROAD_COMPLAINT_TYPES
from civinlp.eval.outcome_eval import (
    EXCLUDED_AREAS,
    evaluate_rankings,
    future_outcomes,
)
from civinlp.ranking.road_priority import RoadPriorityScorer

logger = logging.getLogger(__name__)

METERS_PER_DEG_LAT = 111_320.0

VOLUME_FEATURES = [
    "log_count",
    "log_recent_7d",
    "log_recent_28d",
    "days_since_last",
    "repeat_rate",
]
SEVERITY_FEATURES = ["share_high", "share_medium", "log_n_high"]
TYPE_FEATURES = [f"share_type_{i}" for i in range(len(ROAD_COMPLAINT_TYPES))]

DEFAULT_WEIGHT_GRID = [
    {"alpha": 1.0, "beta": b, "gamma": g}
    for b in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for g in [0.0, 0.5]
]


def grid_area_ids(
    df: pd.DataFrame,
    cell_m: float = 500.0,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
) -> pd.Series:
    """
    Assign each row to a square lat/lon grid cell of about `cell_m` metres.

    Rows with missing or (0, 0) coordinates get "UNKNOWN", which the evaluation
    excludes like an unknown zip code.
    """
    lat = pd.to_numeric(df[lat_column], errors="coerce")
    lon = pd.to_numeric(df[lon_column], errors="coerce")
    valid = lat.notna() & lon.notna() & ~((lat == 0) & (lon == 0))
    if not valid.any():
        return pd.Series("UNKNOWN", index=df.index)

    dlat = cell_m / METERS_PER_DEG_LAT
    dlon = cell_m / (METERS_PER_DEG_LAT * math.cos(math.radians(lat[valid].mean())))
    i = np.floor(lat[valid] / dlat).astype(int).astype(str)
    j = np.floor(lon[valid] / dlon).astype(int).astype(str)
    ids = pd.Series("UNKNOWN", index=df.index)
    ids[valid] = "g" + i + "_" + j
    return ids


def rolling_origins(
    times: pd.Series,
    history_days: int = 28,
    horizon_days: int = 28,
    step_days: int = 28,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Return (history_start, cutoff, end) for every fold that fits in the data.

    The first cutoff is `history_days` after the earliest timestamp; folds stop
    when a full horizon no longer fits before the latest timestamp.
    """
    times = pd.to_datetime(times, errors="coerce").dropna()
    if times.empty:
        return []
    start, last = times.min(), times.max()
    history, horizon = pd.Timedelta(days=history_days), pd.Timedelta(days=horizon_days)
    step = pd.Timedelta(days=step_days)

    folds = []
    cutoff = start + history
    while cutoff + horizon <= last:
        folds.append((cutoff - history, cutoff, cutoff + horizon))
        cutoff += step
    return folds


def area_features(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    area_column: str = "area_id",
    time_column: str = "created_at",
    shrinkage: float = 10.0,
) -> pd.DataFrame:
    """
    Per-area features from the history window only, indexed by area id.

    `repeat_rate` and `severity_norm` match RoadPriorityScorer.score_area, so
    `priority_from_features` reproduces its score without the per-area loop.
    `severity_shrunk` pulls each area's mean severity toward the window mean by
    `shrinkage` pseudo-complaints, so an area with one high complaint no longer
    tops the severity ranking.
    """
    df = history[~history[area_column].astype(str).isin(EXCLUDED_AREAS)].copy()
    df["_area"] = df[area_column].astype(str)
    df["_t"] = pd.to_datetime(df[time_column], errors="coerce")
    scorer = RoadPriorityScorer()
    if "predicted_severity" in df.columns:
        df["_sev"] = df["predicted_severity"].map(scorer._severity_to_numeric)
    else:
        df["_sev"] = 1.0

    df = df.sort_values(["_area", "_t"])
    gap_days = df.groupby("_area")["_t"].diff().dt.total_seconds() / 86400
    df["_repeat"] = (gap_days <= 7).astype(float)
    cutoff = pd.Timestamp(cutoff)
    df["_recent_7d"] = (df["_t"] >= cutoff - pd.Timedelta(days=7)).astype(float)
    df["_recent_28d"] = (df["_t"] >= cutoff - pd.Timedelta(days=28)).astype(float)
    df["_high"] = (df["_sev"] == 3).astype(float)
    df["_medium"] = (df["_sev"] == 2).astype(float)
    types = (
        df["complaint_type"].astype(str).str.strip().str.lower()
        if "complaint_type" in df.columns
        else pd.Series("", index=df.index)
    )
    for i, name in enumerate(ROAD_COMPLAINT_TYPES):
        df[f"_type_{i}"] = (types == name.lower()).astype(float)

    g = df.groupby("_area")
    feats = pd.DataFrame(
        {
            "count": g.size(),
            "recent_7d": g["_recent_7d"].sum(),
            "recent_28d": g["_recent_28d"].sum(),
            "last_seen": g["_t"].max(),
            # score_area divides repeats by the number of valid timestamps
            "repeat_rate": g["_repeat"].sum() / g["_t"].count().clip(lower=1),
            "mean_severity": g["_sev"].mean(),
            "n_high": g["_high"].sum(),
            "n_medium": g["_medium"].sum(),
        }
    )
    feats.loc[g["_t"].count() < 2, "repeat_rate"] = 0.0
    feats["log_count"] = np.log1p(feats["count"])
    feats["log_recent_7d"] = np.log1p(feats["recent_7d"])
    feats["log_recent_28d"] = np.log1p(feats["recent_28d"])
    feats["days_since_last"] = (
        (cutoff - feats.pop("last_seen")).dt.total_seconds() / 86400
    ).fillna(float("inf"))
    feats["share_high"] = feats["n_high"] / feats["count"]
    feats["share_medium"] = feats["n_medium"] / feats["count"]
    feats["log_n_high"] = np.log1p(feats["n_high"])
    for i in range(len(ROAD_COMPLAINT_TYPES)):
        feats[f"share_type_{i}"] = g[f"_type_{i}"].mean()
    feats["severity_norm"] = (feats["mean_severity"] - 1.0) / 2.0

    prior = df["_sev"].mean() if len(df) else 1.0
    feats["severity_shrunk"] = (
        feats["mean_severity"] * feats["count"] + prior * shrinkage
    ) / (feats["count"] + shrinkage)
    feats.index.name = area_column
    return feats


def priority_from_features(
    feats: pd.DataFrame, alpha=1.0, beta=1.0, gamma=0.5
) -> pd.Series:
    """RoadPriorityScorer's score (no confidence column) from `area_features`."""
    max_count = feats["count"].max() if len(feats) else 1
    count_norm = feats["log_count"] / np.log1p(max_count)
    score = (
        alpha * count_norm
        + beta * feats["severity_norm"]
        + gamma * feats["repeat_rate"]
    )
    return score.clip(lower=0.0)


def _order(values: pd.Series) -> List[str]:
    """Highest value first; ties broken by area id so results are deterministic."""
    frame = values.rename("v").reset_index()
    key = frame.columns[0]
    frame[key] = frame[key].astype(str)
    return frame.sort_values(["v", key], ascending=[False, True])[key].tolist()


def fold_rankings(
    feats: pd.DataFrame,
    tuned_weights: Optional[Dict[str, float]] = None,
    models: Optional[Dict[str, object]] = None,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """All rankings for one fold, built from history features only."""
    rankings = {
        "count_only": _order(feats["count"]),
        "recency": _order(-feats["days_since_last"]),
        "severity_only": _order(feats["mean_severity"]),
        "severity_shrunk": _order(feats["severity_shrunk"]),
        "priority": _order(priority_from_features(feats)),
    }
    if tuned_weights is not None:
        rankings["priority_tuned"] = _order(
            priority_from_features(feats, **tuned_weights)
        )
    for name, (model, columns) in (models or {}).items():
        rankings[name] = _order(
            pd.Series(model.predict(feats[columns].to_numpy()), index=feats.index)
        )
    rng = np.random.RandomState(seed)
    rankings["random"] = list(rng.permutation(sorted(feats.index.astype(str))))
    return rankings


def _prepare_folds(
    df: pd.DataFrame,
    folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    area_column: str,
    time_column: str,
    outcome: str,
    critical_frac: float,
) -> List[Dict]:
    """Features and outcomes for each fold; folds with no rankable areas are dropped."""
    times = pd.to_datetime(df[time_column], errors="coerce")
    prepared = []
    for start, cutoff, end in folds:
        history = df[(times >= start) & (times < cutoff)]
        future = df[(times >= cutoff) & (times < end)]
        counts, critical = future_outcomes(
            history, future, area_column, critical_frac, outcome=outcome
        )
        if not counts:
            continue
        feats = area_features(history, cutoff, area_column, time_column)
        prepared.append(
            {
                "cutoff": cutoff,
                "end": end,
                "feats": feats,
                "counts": counts,
                "critical": critical,
                "history_rows": len(history),
                "future_rows": len(future),
            }
        )
    return prepared


def tune_weights(
    tuning: List[Dict],
    grid: Optional[List[Dict[str, float]]] = None,
    metric: str = "ndcg@10",
    k_values: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Pick the priority weights with the best mean `metric` over tuning folds."""
    grid = grid or DEFAULT_WEIGHT_GRID
    rows = []
    for weights in grid:
        scores = []
        for fold in tuning:
            ranked = _order(priority_from_features(fold["feats"], **weights))
            result = evaluate_rankings(
                {"p": ranked}, fold["counts"], fold["critical"], k_values
            )
            scores.append(result[metric].iloc[0] if metric in result else np.nan)
        rows.append({**weights, metric: float(np.nanmean(scores))})
    table = pd.DataFrame(rows)
    best = table.loc[table[metric].idxmax(), list(grid[0])].to_dict()
    return {k: float(v) for k, v in best.items()}, table


def fit_learned_rankers(
    tuning: List[Dict], ridge_alpha: float = 1.0
) -> Dict[str, Tuple[object, List[str]]]:
    """
    Fit ridge regressions of log1p(outcome) on history features, pooled over
    tuning folds. Only the order of predictions matters, so a log target keeps
    a few very busy areas from dominating the fit.
    """
    frames = []
    for fold in tuning:
        feats = fold["feats"]
        target = pd.Series(fold["counts"], dtype=float)
        frames.append(feats.join(target.rename("_y"), how="inner"))
    data = pd.concat(frames)
    all_features = VOLUME_FEATURES + SEVERITY_FEATURES + TYPE_FEATURES
    finite = np.isfinite(data[all_features]).all(axis=1)
    data = data[finite]
    y = np.log1p(data["_y"].clip(lower=0).to_numpy())

    models = {}
    for name, columns in [
        ("learned_volume", VOLUME_FEATURES),
        ("learned_volume_severity", VOLUME_FEATURES + SEVERITY_FEATURES),
        ("learned_volume_type", VOLUME_FEATURES + TYPE_FEATURES),
        ("learned_volume_type_severity", all_features),
    ]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        model.fit(data[columns].to_numpy(), y)
        models[name] = (model, columns)
    return models


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean over folds."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def summarize_folds(
    per_fold: pd.DataFrame,
    metrics: List[str],
    reference: str = "count_only",
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Mean and 95% bootstrap CI per ranking, plus the paired per-fold difference
    from `reference` (its CI and the share of folds where the ranking wins).
    """
    rows = []
    ref = per_fold[per_fold["ranking"] == reference].set_index("fold")
    for name, group in per_fold.groupby("ranking", sort=False):
        group = group.set_index("fold")
        for metric in metrics:
            values = group[metric].to_numpy()
            lo, hi = bootstrap_ci(values, n_boot, seed=seed)
            diff = (group[metric] - ref[metric].reindex(group.index)).to_numpy()
            dlo, dhi = bootstrap_ci(diff, n_boot, seed=seed)
            rows.append(
                {
                    "ranking": name,
                    "metric": metric,
                    "n_folds": int(np.sum(~np.isnan(values))),
                    "mean": float(np.nanmean(values)),
                    "ci_low": lo,
                    "ci_high": hi,
                    f"diff_vs_{reference}": float(np.nanmean(diff)),
                    "diff_ci_low": dlo,
                    "diff_ci_high": dhi,
                    "win_rate": float(np.nanmean(diff > 0)),
                }
            )
    return pd.DataFrame(rows)


def run_rolling_evaluation(
    df: pd.DataFrame,
    area_column: str = "area_id",
    time_column: str = "created_at",
    outcome: str = "volume",
    history_days: int = 28,
    horizon_days: int = 28,
    step_days: int = 28,
    tune_frac: float = 0.3,
    critical_frac: float = 0.2,
    k_values: Optional[List[int]] = None,
    weight_grid: Optional[List[Dict[str, float]]] = None,
    tune_metric: Optional[str] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run the full rolling-origin evaluation.

    Returns:
        per_fold: one row per (test fold, ranking) with all metrics.
        summary:  `summarize_folds` over test folds.
        info:     fold layout, tuned weights and the tuning grid.
    """
    k_values = k_values or [10, 50]
    tune_metric = tune_metric or f"ndcg@{k_values[0]}"
    folds = rolling_origins(df[time_column], history_days, horizon_days, step_days)
    prepared = _prepare_folds(
        df, folds, area_column, time_column, outcome, critical_frac
    )
    if len(prepared) < 3:
        raise ValueError(
            f"Only {len(prepared)} usable folds; need a longer time range or "
            "shorter history/horizon windows"
        )

    n_tune = max(1, int(math.ceil(len(prepared) * tune_frac)))
    tuning = prepared[:n_tune]
    # A test fold's cutoff must not precede any tuning outcome
    tuning_end = tuning[-1]["end"]
    testing = [f for f in prepared[n_tune:] if f["cutoff"] >= tuning_end]
    if len(testing) < 2:
        raise ValueError("Fewer than 2 test folds after tuning; lower tune_frac")

    tuned, grid_table = tune_weights(tuning, weight_grid, tune_metric, k_values)
    models = fit_learned_rankers(tuning)
    logger.info(
        f"Rolling eval ({outcome}): {len(tuning)} tuning folds, "
        f"{len(testing)} test folds; tuned weights {tuned}"
    )

    rows = []
    for i, fold in enumerate(testing):
        rankings = fold_rankings(fold["feats"], tuned, models, seed=seed + i)
        result = evaluate_rankings(rankings, fold["counts"], fold["critical"], k_values)
        result.insert(0, "fold", i)
        result.insert(1, "cutoff", str(fold["cutoff"]))
        rows.append(result)
    per_fold = pd.concat(rows, ignore_index=True)

    metrics = [
        c for c in per_fold.columns if c.startswith(("ndcg@", "recall@", "spearman"))
    ]
    summary = summarize_folds(per_fold, metrics, seed=seed)
    info = {
        "outcome": outcome,
        "history_days": history_days,
        "horizon_days": horizon_days,
        "step_days": step_days,
        "tuning_folds": [str(f["cutoff"]) for f in tuning],
        "test_folds": [str(f["cutoff"]) for f in testing],
        "mean_rankable_areas": float(np.mean([len(f["counts"]) for f in testing])),
        "tuned_weights": tuned,
        "tune_metric": tune_metric,
        "weight_grid": grid_table.to_dict(orient="records"),
    }
    return per_fold, summary, info

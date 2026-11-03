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
complaint-type mix is known. Rule severity is a fixed function of complaint
type and descriptor, so `learned_volume_descriptor` vs
`learned_volume_descriptor_severity` asks whether the severity mapping adds
anything beyond the raw descriptor mix.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

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
from civinlp.eval.ranking_metrics import spearman_from_scores
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
DESCRIPTOR_PREFIX = "share_desc_"
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]


def descriptor_keys(df: pd.DataFrame) -> pd.Series:
    """'complaint type|descriptor' per row, lowercased."""
    ctype = df.get("complaint_type", pd.Series("", index=df.index))
    desc = df.get("descriptor", pd.Series("", index=df.index))
    return (
        ctype.astype(str).str.strip().str.lower()
        + "|"
        + desc.astype(str).str.strip().str.lower()
    )


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
    ref_lat: Optional[float] = None,
) -> pd.Series:
    """
    Assign each row to a square lat/lon grid cell of about `cell_m` metres.

    The east-west cell width depends on a reference latitude, by default the
    mean latitude of `df`. Pass the same `ref_lat` (see `grid_reference_lat`)
    to place a second dataset, such as pavement ratings, on the same grid.

    Rows with missing or (0, 0) coordinates get "UNKNOWN", which the evaluation
    excludes like an unknown zip code.
    """
    lat = pd.to_numeric(df[lat_column], errors="coerce")
    lon = pd.to_numeric(df[lon_column], errors="coerce")
    valid = lat.notna() & lon.notna() & ~((lat == 0) & (lon == 0))
    if not valid.any():
        return pd.Series("UNKNOWN", index=df.index)

    if ref_lat is None:
        ref_lat = lat[valid].mean()
    dlat = cell_m / METERS_PER_DEG_LAT
    dlon = cell_m / (METERS_PER_DEG_LAT * math.cos(math.radians(ref_lat)))
    i = np.floor(lat[valid] / dlat).astype(int).astype(str)
    j = np.floor(lon[valid] / dlon).astype(int).astype(str)
    ids = pd.Series("UNKNOWN", index=df.index)
    ids[valid] = "g" + i + "_" + j
    return ids


def grid_reference_lat(
    df: pd.DataFrame, lat_column: str = "latitude", lon_column: str = "longitude"
) -> float:
    """The reference latitude `grid_area_ids` uses for `df` by default."""
    lat = pd.to_numeric(df[lat_column], errors="coerce")
    lon = pd.to_numeric(df[lon_column], errors="coerce")
    return float(lat[lat.notna() & lon.notna() & ~((lat == 0) & (lon == 0))].mean())


def nearest_area_ids(
    points: pd.DataFrame,
    reference: pd.DataFrame,
    area_column: str = "area_id",
    max_m: float = 300.0,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
) -> pd.Series:
    """
    Give each row of `points` the area of the nearest located row of
    `reference`, e.g. pavement ratings the zip code of the nearest complaint.

    Points with no reference row within `max_m` metres, or with missing
    coordinates, get "UNKNOWN".
    """
    from scipy.spatial import cKDTree

    def located(df):
        lat = pd.to_numeric(df[lat_column], errors="coerce")
        lon = pd.to_numeric(df[lon_column], errors="coerce")
        return lat, lon, lat.notna() & lon.notna() & ~((lat == 0) & (lon == 0))

    rlat, rlon, rvalid = located(reference)
    rvalid &= ~reference[area_column].astype(str).isin(EXCLUDED_AREAS)
    ids = pd.Series("UNKNOWN", index=points.index)
    plat, plon, pvalid = located(points)
    if not rvalid.any() or not pvalid.any():
        return ids

    scale = METERS_PER_DEG_LAT * math.cos(math.radians(rlat[rvalid].mean()))
    tree = cKDTree(np.c_[rlat[rvalid] * METERS_PER_DEG_LAT, rlon[rvalid] * scale])
    dist, nearest = tree.query(
        np.c_[plat[pvalid] * METERS_PER_DEG_LAT, plon[pvalid] * scale], k=1
    )
    areas = reference.loc[rvalid, area_column].astype(str).to_numpy()[nearest]
    ids[pvalid] = np.where(dist <= max_m, areas, "UNKNOWN")
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
    descriptors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Per-area features from the history window only, indexed by area id.

    `repeat_rate` and `severity_norm` match RoadPriorityScorer.score_area, so
    `priority_from_features` reproduces its score without the per-area loop.
    `severity_shrunk` pulls each area's mean severity toward the window mean by
    `shrinkage` pseudo-complaints, so an area with one high complaint no longer
    tops the severity ranking. With `descriptors` (a vocabulary of
    `descriptor_keys`), adds the share of each as `share_desc_<i>`.
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

    descriptors = descriptors or []
    if descriptors:
        keys = descriptor_keys(df)
        for i, key in enumerate(descriptors):
            df[f"_desc_{i}"] = (keys == key).astype(float)

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
    for i in range(len(descriptors)):
        feats[f"{DESCRIPTOR_PREFIX}{i}"] = g[f"_desc_{i}"].mean()
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


def fold_rankings(
    feats: pd.DataFrame,
    tuned_weights: Optional[Dict[str, float]] = None,
    models: Optional[Dict[str, object]] = None,
    seed: int = 42,
) -> Dict[str, Union[List[str], pd.Series]]:
    """
    All rankings for one fold, built from history features only. Rankings are
    score Series so Spearman can treat tied scores as ties; random is a list.
    """
    rankings = {
        "count_only": feats["count"],
        "recency": -feats["days_since_last"],
        "severity_only": feats["mean_severity"],
        "severity_shrunk": feats["severity_shrunk"],
        "priority": priority_from_features(feats),
    }
    if tuned_weights is not None:
        rankings["priority_tuned"] = priority_from_features(feats, **tuned_weights)
    for name, (model, columns) in (models or {}).items():
        rankings[name] = pd.Series(
            model.predict(feats[columns].to_numpy()), index=feats.index
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
    descriptors: Optional[List[str]] = None,
    outcome_df: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Features and outcomes for each fold; folds with no rankable areas are
    dropped. The future window is taken from `outcome_df` when given (e.g.
    pavement ratings), else from the complaints themselves.
    """
    times = pd.to_datetime(df[time_column], errors="coerce")
    source = df if outcome_df is None else outcome_df
    source_times = pd.to_datetime(source[time_column], errors="coerce")
    prepared = []
    for start, cutoff, end in folds:
        history = df[(times >= start) & (times < cutoff)]
        future = source[(source_times >= cutoff) & (source_times < end)]
        counts, critical = future_outcomes(
            history, future, area_column, critical_frac, outcome=outcome
        )
        if not counts:
            continue
        feats = area_features(
            history, cutoff, area_column, time_column, descriptors=descriptors
        )
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
            ranked = priority_from_features(fold["feats"], **weights)
            result = evaluate_rankings(
                {"p": ranked}, fold["counts"], fold["critical"], k_values
            )
            scores.append(result[metric].iloc[0] if metric in result else np.nan)
        rows.append({**weights, metric: float(np.nanmean(scores))})
    table = pd.DataFrame(rows)
    best = table.loc[table[metric].idxmax(), list(grid[0])].to_dict()
    return {k: float(v) for k, v in best.items()}, table


def _ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _choose_alpha(data: pd.DataFrame, columns: List[str], y: np.ndarray, alphas):
    """
    Leave-one-tuning-fold-out choice of the ridge penalty: fit on the other
    tuning folds and score Spearman against the held-out fold's raw outcome.
    Ties go to the stronger penalty.
    """
    folds = data["_fold"].to_numpy()
    if len(np.unique(folds)) < 2:
        return alphas[0], {}
    X = data[columns].to_numpy()
    scores = {}
    for alpha in alphas:
        per_fold = []
        for f in np.unique(folds):
            train, test = folds != f, folds == f
            pred = _ridge(alpha).fit(X[train], y[train]).predict(X[test])
            idx = data.index[test]
            per_fold.append(
                spearman_from_scores(
                    dict(zip(idx, pred)), dict(zip(idx, data["_y"].to_numpy()[test]))
                )
            )
        scores[alpha] = float(np.mean(per_fold))
    best = max(alphas, key=lambda a: (round(scores[a], 6), a))
    return best, scores


def fit_learned_rankers(
    tuning: List[Dict], alphas: Optional[List[float]] = None
) -> Tuple[Dict[str, Tuple[object, List[str]]], Dict[str, float]]:
    """
    Fit ridge regressions of log1p(outcome) on history features, pooled over
    tuning folds. Only the order of predictions matters, so a log target keeps
    a few very busy areas from dominating the fit. Each model gets its own
    penalty from `alphas`, chosen on the tuning folds (`_choose_alpha`), so a
    model with ~100 descriptor features is not held to the penalty that suits
    a 5-feature one.

    Returns the fitted models and the penalty chosen for each.
    """
    alphas = alphas or RIDGE_ALPHAS
    frames = []
    for i, fold in enumerate(tuning):
        feats = fold["feats"]
        target = pd.Series(fold["counts"], dtype=float)
        frame = feats.join(target.rename("_y"), how="inner")
        frame.index = [f"{i}:{a}" for a in frame.index]
        frames.append(frame.assign(_fold=i))
    data = pd.concat(frames)
    all_features = VOLUME_FEATURES + SEVERITY_FEATURES + TYPE_FEATURES
    desc_features = [c for c in data.columns if c.startswith(DESCRIPTOR_PREFIX)]
    finite = np.isfinite(data[all_features + desc_features]).all(axis=1)
    data = data[finite]
    y = np.log1p(data["_y"].clip(lower=0).to_numpy())

    specs = [
        ("learned_volume", VOLUME_FEATURES),
        ("learned_volume_severity", VOLUME_FEATURES + SEVERITY_FEATURES),
        ("learned_volume_type", VOLUME_FEATURES + TYPE_FEATURES),
        ("learned_volume_type_severity", all_features),
    ]
    if desc_features:
        specs += [
            ("learned_volume_descriptor", VOLUME_FEATURES + desc_features),
            (
                "learned_volume_descriptor_severity",
                VOLUME_FEATURES + desc_features + SEVERITY_FEATURES,
            ),
        ]
    models, chosen = {}, {}
    for name, columns in specs:
        alpha, _ = _choose_alpha(data, columns, y, alphas)
        models[name] = (_ridge(alpha).fit(data[columns].to_numpy(), y), columns)
        chosen[name] = alpha
    return models, chosen


BLOCK_SIZE = 3
N_BOOT = 5000

# (label, ranking, reference): the paired differences behind the paper's Table 4.
# Each asks what the first ranker adds over the second on the same folds.
PAIRED_COMPARISONS = [
    ("severity_shrunk_minus_count", "severity_shrunk", "count_only"),
    ("severity_on_volume", "learned_volume_severity", "learned_volume"),
    ("type_on_volume", "learned_volume_type", "learned_volume"),
    (
        "type_on_volume_severity",
        "learned_volume_type_severity",
        "learned_volume_severity",
    ),
    ("severity_on_volume_type", "learned_volume_type_severity", "learned_volume_type"),
    (
        "severity_on_volume_descriptor",
        "learned_volume_descriptor_severity",
        "learned_volume_descriptor",
    ),
    (
        "descriptor_vs_type_severity",
        "learned_volume_descriptor",
        "learned_volume_type_severity",
    ),
]


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = N_BOOT,
    level: float = 0.95,
    seed: int = 42,
    block: int = BLOCK_SIZE,
) -> Tuple[float, float]:
    """
    Percentile moving-block bootstrap CI of the mean over folds.

    Adjacent folds share data (each fold's future window is the next fold's
    history), so folds are resampled in runs of `block` consecutive folds;
    resampling single folds would give intervals that are too narrow.
    `block=1` is the ordinary bootstrap.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    k = len(values)
    if k < 2:
        return (float("nan"), float("nan"))
    means = values[_block_indices(k, n_boot, block, seed)].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def _block_indices(k: int, n_boot: int, block: int, seed: int) -> np.ndarray:
    """(n_boot, k) fold indices, each row `k` folds drawn in runs of `block`."""
    block = max(1, min(block, k))
    n_blocks = int(math.ceil(k / block))
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, k - block + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :k]


def simultaneous_intervals(
    diffs: pd.DataFrame,
    n_boot: int = N_BOOT,
    level: float = 0.95,
    seed: int = 42,
    block: int = BLOCK_SIZE,
) -> Tuple[pd.DataFrame, float]:
    """
    Simultaneous (sup-t) moving-block bootstrap intervals for a family of
    paired differences measured on the same folds.

    `diffs` has one row per fold, in time order, and one column per comparison.
    Every bootstrap draw resamples the same blocks of folds for all columns, so
    the correlation between comparisons is kept. Each interval is the mean
    plus or minus `crit` bootstrap standard errors, where `crit` is the `level`
    quantile of the largest standardised deviation across the family: all
    intervals cover their true values together with probability about `level`.

    Returns one row per comparison (mean, standard error, interval) and `crit`.
    """
    values = diffs.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("simultaneous_intervals needs every comparison on every fold")
    k = len(values)
    means = values[_block_indices(k, n_boot, block, seed)].mean(axis=1)
    est = values.mean(axis=0)
    se = means.std(axis=0, ddof=1)
    # A comparison that never varies (identical rankers) has no spread to scale
    scaled = np.divide(np.abs(means - est), se, out=np.zeros_like(means), where=se > 0)
    crit = float(np.percentile(scaled.max(axis=1), level * 100))
    table = pd.DataFrame(
        {
            "comparison": diffs.columns,
            "diff": est,
            "se": se,
            "sim_low": est - crit * se,
            "sim_high": est + crit * se,
        }
    )
    return table, crit


def paired_comparisons(
    per_fold: pd.DataFrame,
    comparisons: Optional[List[Tuple[str, str, str]]] = None,
    metric: str = "spearman_vs_future",
    n_boot: int = N_BOOT,
    block: int = BLOCK_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Mean paired per-fold difference `ranking - reference` with its block
    bootstrap CI, for each comparison whose rankers are both in `per_fold`.
    """
    comparisons = comparisons or PAIRED_COMPARISONS
    wide = per_fold.pivot(index="fold", columns="ranking", values=metric)
    rows = []
    for label, name, reference in comparisons:
        if name not in wide or reference not in wide:
            continue
        diff = (wide[name] - wide[reference]).to_numpy()
        lo, hi = bootstrap_ci(diff, n_boot, seed=seed, block=block)
        rows.append(
            {
                "comparison": label,
                "ranking": name,
                "reference": reference,
                "metric": metric,
                "n_folds": int(np.sum(~np.isnan(diff))),
                "diff": float(np.nanmean(diff)),
                "ci_low": lo,
                "ci_high": hi,
                "win_rate": float(np.nanmean(diff > 0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_folds(
    per_fold: pd.DataFrame,
    metrics: List[str],
    reference: str = "count_only",
    n_boot: int = N_BOOT,
    seed: int = 42,
    block: int = BLOCK_SIZE,
) -> pd.DataFrame:
    """
    Mean and 95% block bootstrap CI per ranking, plus the paired per-fold
    difference from `reference` (its CI and the share of folds where the
    ranking wins).
    """
    rows = []
    ref = per_fold[per_fold["ranking"] == reference].set_index("fold")
    for name, group in per_fold.groupby("ranking", sort=False):
        group = group.set_index("fold")
        for metric in metrics:
            values = group[metric].to_numpy()
            lo, hi = bootstrap_ci(values, n_boot, seed=seed, block=block)
            diff = (group[metric] - ref[metric].reindex(group.index)).to_numpy()
            dlo, dhi = bootstrap_ci(diff, n_boot, seed=seed, block=block)
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
    outcome_df: Optional[pd.DataFrame] = None,
    block: int = BLOCK_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run the full rolling-origin evaluation.

    `outcome_df` supplies the future window instead of the complaints (pavement
    ratings, with the same area and time columns). `block` is the bootstrap
    block length in folds: with a horizon longer than the step, future windows
    overlap, and a block should span about horizon_days / step_days folds.

    Returns:
        per_fold: one row per (test fold, ranking) with all metrics.
        summary:  `summarize_folds` over test folds.
        info:     fold layout, tuned weights and the tuning grid.
    """
    k_values = k_values or [10, 50]
    tune_metric = tune_metric or f"ndcg@{k_values[0]}"
    folds = rolling_origins(df[time_column], history_days, horizon_days, step_days)
    descriptors = None
    if "descriptor" in df.columns and folds:
        # Vocabulary from the tuning period only, so test folds add no new columns
        tuning_end = folds[min(len(folds), math.ceil(len(folds) * tune_frac)) - 1][2]
        times = pd.to_datetime(df[time_column], errors="coerce")
        descriptors = sorted(descriptor_keys(df[times < tuning_end]).unique())
    prepared = _prepare_folds(
        df,
        folds,
        area_column,
        time_column,
        outcome,
        critical_frac,
        descriptors,
        outcome_df,
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
    models, ridge_alphas = fit_learned_rankers(tuning)
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
    summary = summarize_folds(per_fold, metrics, seed=seed, block=block)
    info = {
        "outcome": outcome,
        "history_days": history_days,
        "horizon_days": horizon_days,
        "step_days": step_days,
        "bootstrap_block": block,
        "tuning_folds": [str(f["cutoff"]) for f in tuning],
        "test_folds": [str(f["cutoff"]) for f in testing],
        "mean_rankable_areas": float(np.mean([len(f["counts"]) for f in testing])),
        "descriptor_vocabulary": len(descriptors or []),
        "tuned_weights": tuned,
        "ridge_alphas": ridge_alphas,
        "tune_metric": tune_metric,
        "weight_grid": grid_table.to_dict(orient="records"),
    }
    return per_fold, summary, info

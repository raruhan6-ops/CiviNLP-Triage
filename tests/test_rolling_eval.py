import numpy as np
import pandas as pd
import pytest

from civinlp.eval.rolling_eval import (
    area_features,
    bootstrap_ci,
    grid_area_ids,
    priority_from_features,
    rolling_origins,
    run_rolling_evaluation,
    summarize_folds,
)
from civinlp.ranking.road_priority import RoadPriorityScorer


def _synthetic(days=200, n_areas=30, seed=0):
    """Areas with a stable complaint rate; busier areas get more high severity."""
    rng = np.random.RandomState(seed)
    rows = []
    start = pd.Timestamp("2024-01-01")
    for a in range(n_areas):
        rate = 0.1 + 0.1 * a
        for d in range(days):
            for _ in range(rng.poisson(rate)):
                sev = "high" if rng.rand() < a / (2 * n_areas) else "low"
                t = start + pd.Timedelta(days=d, hours=int(rng.randint(24)))
                rows.append((f"A{a:02d}", t, sev))
    return pd.DataFrame(rows, columns=["area_id", "created_at", "predicted_severity"])


def test_rolling_origins_tile_without_overlap():
    times = pd.Series(pd.date_range("2024-01-01", "2024-04-30", freq="D"))
    folds = rolling_origins(times, history_days=28, horizon_days=14, step_days=14)
    assert folds[0][1] == pd.Timestamp("2024-01-29")
    for (_, c1, e1), (_, c2, _) in zip(folds, folds[1:]):
        assert e1 <= c2  # future windows don't overlap
    assert folds[-1][2] <= times.max()


def test_grid_area_ids():
    df = pd.DataFrame(
        {
            "latitude": [40.7000, 40.7001, 40.7200, None, 0.0],
            "longitude": [-74.0000, -74.0001, -74.0000, -74.0, 0.0],
        }
    )
    ids = grid_area_ids(df, cell_m=500)
    assert ids[0] == ids[1]  # ~15 m apart
    assert ids[0] != ids[2]  # ~2 km apart
    assert ids[3] == "UNKNOWN" and ids[4] == "UNKNOWN"


def test_priority_from_features_matches_scorer():
    df = _synthetic(days=40, n_areas=8)
    cutoff = df["created_at"].max()
    feats = area_features(df, cutoff)
    for weights in [{}, {"alpha": 1.0, "beta": 2.0, "gamma": 0.0}]:
        fast = priority_from_features(feats, **weights)
        slow = RoadPriorityScorer(delta=0.0, **weights).rank_areas(df)
        slow = slow.set_index("area_id")["priority_score"]
        pd.testing.assert_series_equal(
            fast.sort_index(), slow.sort_index(), check_names=False, atol=1e-9
        )


def test_shrunk_severity_discounts_single_complaints():
    rows = [("small", "2024-01-05", "high")]
    rows += [
        ("big", f"2024-01-{d:02d}", "high" if d % 4 else "low") for d in range(1, 21)
    ]
    # Most complaints are low severity, as in the real data
    rows += [(f"rest{i}", "2024-01-10", "low") for i in range(40)]
    df = pd.DataFrame(rows, columns=["area_id", "created_at", "predicted_severity"])
    feats = area_features(df, pd.Timestamp("2024-01-25"))
    assert feats.loc["small", "mean_severity"] > feats.loc["big", "mean_severity"]
    assert feats.loc["small", "severity_shrunk"] < feats.loc["big", "severity_shrunk"]


def test_descriptor_shares():
    df = pd.DataFrame(
        {
            "area_id": ["A", "A", "A", "B"],
            "created_at": pd.to_datetime(["2024-01-01"] * 4),
            "complaint_type": ["Street Condition"] * 3 + ["Curb Condition"],
            "descriptor": ["Pothole", "Pothole", "Cave-in", "Broken Curb"],
        }
    )
    vocab = ["curb condition|broken curb", "street condition|pothole"]
    feats = area_features(df, pd.Timestamp("2024-01-02"), descriptors=vocab)
    assert feats.loc["A", "share_desc_1"] == pytest.approx(2 / 3)
    assert feats.loc["A", "share_desc_0"] == 0.0
    assert feats.loc["B", "share_desc_0"] == 1.0


def test_bootstrap_ci_and_paired_summary():
    lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert lo <= 2.5 <= hi
    per_fold = pd.DataFrame(
        {
            "fold": [0, 1, 2, 0, 1, 2],
            "ranking": ["count_only"] * 3 + ["other"] * 3,
            "ndcg@10": [0.5, 0.6, 0.7, 0.6, 0.7, 0.6],
        }
    )
    summary = summarize_folds(per_fold, ["ndcg@10"]).set_index("ranking")
    assert summary.loc["count_only", "diff_vs_count_only"] == 0.0
    assert summary.loc["other", "diff_vs_count_only"] == pytest.approx(0.1 / 3)
    assert summary.loc["other", "win_rate"] == pytest.approx(2 / 3)


def test_run_rolling_evaluation_end_to_end():
    df = _synthetic()
    per_fold, summary, info = run_rolling_evaluation(
        df, history_days=28, horizon_days=14, step_days=14, k_values=[5]
    )
    expected = {
        "count_only",
        "recency",
        "severity_only",
        "severity_shrunk",
        "priority",
        "priority_tuned",
        "learned_volume",
        "learned_volume_severity",
        "learned_volume_type",
        "learned_volume_type_severity",
        "random",
    }
    assert expected == set(per_fold["ranking"])
    # Tuning outcomes end before the first test cutoff
    assert max(info["tuning_folds"]) < min(info["test_folds"])
    assert len(info["test_folds"]) >= 2
    # Volume is stable per area here, so count beats random on average
    ndcg = summary[summary["metric"] == "ndcg@5"].set_index("ranking")["mean"]
    assert ndcg["count_only"] > ndcg["random"]
    assert ndcg["learned_volume"] > ndcg["random"]


def test_ridge_penalty_is_chosen_per_model_on_tuning_folds():
    df = _synthetic()
    _, _, info = run_rolling_evaluation(
        df, history_days=28, horizon_days=14, step_days=14, k_values=[5]
    )
    from civinlp.eval.rolling_eval import RIDGE_ALPHAS

    assert set(info["ridge_alphas"]) == {
        "learned_volume",
        "learned_volume_severity",
        "learned_volume_type",
        "learned_volume_type_severity",
    }
    assert all(a in RIDGE_ALPHAS for a in info["ridge_alphas"].values())

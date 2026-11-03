import numpy as np
import pandas as pd
import pytest

from civinlp.data.pavement import line_midpoint, preprocess_pavement
from civinlp.eval.outcome_eval import future_outcomes
from civinlp.eval.rolling_eval import (
    grid_area_ids,
    grid_reference_lat,
    run_rolling_evaluation,
)


def test_line_midpoint():
    lat, lon = line_midpoint("MULTILINESTRING ((-73.95 40.77, -73.94 40.78))")
    assert lat == pytest.approx(40.775)
    assert lon == pytest.approx(-73.945)
    assert np.isnan(line_midpoint(None)[0])


def test_preprocess_pavement_keeps_located_ratings():
    raw = pd.DataFrame(
        {
            "oftcode": ["1", "1", "2", "3"],
            "boroughname": ["Manhattan"] * 4,
            "road_type": [None] * 4,
            "systemrating": [8.6, 0.0, 3.2, 7.0],
            "nonratingreason": [None, "Duplicate", None, None],
            "inspection": ["2026-07-10", "2026-07-10", "2025-01-02", None],
            "the_geom": ["MULTILINESTRING ((-73.95 40.77, -73.94 40.78))"] * 4,
        }
    )
    out = preprocess_pavement(raw)
    # The unrated duplicate and the undated row are dropped
    assert out["rating"].tolist() == [8.6, 3.2]
    assert out["segment_id"].tolist() == ["1", "2"]
    assert out["latitude"].iloc[0] == pytest.approx(40.775)


def test_pavement_outcomes():
    history = pd.DataFrame({"area_id": ["A", "B", "C"]})
    future = pd.DataFrame(
        {
            "area_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 2 + ["D"] * 3,
            "rating": [9.0, 8.0, 10.0, 3.0, 6.0, 7.5, 2.0, 2.0, 1.0, 1.0, 1.0],
        }
    )
    deficit, _ = future_outcomes(history, future, outcome="pavement_deficit")
    # C has too few ratings; D was not in the history, so it cannot be ranked
    assert deficit == pytest.approx({"A": 1.0, "B": 10 - 16.5 / 3})
    poor, critical = future_outcomes(
        history, future, outcome="pavement_poor_share", critical_frac=0.5
    )
    assert poor == pytest.approx({"A": 0.0, "B": 2 / 3})
    assert critical == {"B"}


def test_second_dataset_shares_the_grid():
    complaints = pd.DataFrame({"latitude": [40.6, 40.9], "longitude": [-74.0, -73.8]})
    ratings = pd.DataFrame({"latitude": [40.6], "longitude": [-74.0]})
    ref = grid_reference_lat(complaints)
    assert ref == pytest.approx(40.75)
    same = grid_area_ids(ratings, ref_lat=ref)
    assert same.iloc[0] == grid_area_ids(complaints).iloc[0]


def test_rolling_evaluation_grades_on_ratings():
    rng = np.random.RandomState(0)
    start = pd.Timestamp("2024-01-01")
    complaints, ratings = [], []
    for a in range(20):
        # Busier areas have worse roads, so count-only should track the deficit
        for d in range(300):
            for _ in range(rng.poisson(0.05 + 0.05 * a)):
                complaints.append((f"A{a:02d}", start + pd.Timedelta(days=d), "low"))
            if d % 5 == 0:
                rating = 9.5 - 0.3 * a + rng.normal(0, 0.3)
                ratings.append((f"A{a:02d}", start + pd.Timedelta(days=d), rating))
    df = pd.DataFrame(
        complaints, columns=["area_id", "created_at", "predicted_severity"]
    )
    rt = pd.DataFrame(ratings, columns=["area_id", "created_at", "rating"])
    per_fold, summary, info = run_rolling_evaluation(
        df,
        outcome="pavement_deficit",
        history_days=28,
        horizon_days=56,
        step_days=28,
        k_values=[5],
        outcome_df=rt,
        block=2,
    )
    assert info["bootstrap_block"] == 2
    rho = summary[summary["metric"] == "spearman_vs_future"].set_index("ranking")
    assert rho.loc["count_only", "mean"] > 0.5

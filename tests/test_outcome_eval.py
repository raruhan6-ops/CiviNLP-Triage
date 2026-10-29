import pandas as pd
import pytest

from civinlp.eval.outcome_eval import (
    baseline_rankings,
    evaluate_rankings,
    future_outcomes,
    run_outcome_evaluation,
    temporal_outcome_split,
)
from civinlp.labels.taxonomy import RoadSeverity
from civinlp.ranking.road_priority import RoadPriorityScorer


def _make_df():
    # History (Jan): A has 3 low complaints, B has 1 high, C has 1 low.
    # Future (Feb): B gets 4 new complaints, A gets 1, C gets none.
    rows = [
        ("A", "2023-01-01", "low"), ("A", "2023-01-02", "low"), ("A", "2023-01-03", "low"),
        ("B", "2023-01-04", "high"),
        ("C", "2023-01-10", "low"),
        ("UNKNOWN", "2023-01-05", "high"),
        ("B", "2023-02-01", "low"), ("B", "2023-02-02", "low"),
        ("B", "2023-02-03", "low"), ("B", "2023-02-04", "low"),
        ("A", "2023-02-05", "low"),
        ("D", "2023-02-06", "low"),  # new area, not rankable
    ]
    return pd.DataFrame(rows, columns=["area_id", "created_at", "predicted_severity"])


def test_temporal_split_respects_cutoff():
    df = _make_df()
    history, future, cutoff = temporal_outcome_split(df, cutoff="2023-02-01")
    assert len(history) == 6 and len(future) == 6
    assert pd.to_datetime(history["created_at"]).max() < cutoff
    assert pd.to_datetime(future["created_at"]).min() >= cutoff


def test_future_outcomes_only_uses_future_window():
    df = _make_df()
    history, future, _ = temporal_outcome_split(df, cutoff="2023-02-01")
    counts, critical = future_outcomes(history, future, critical_frac=0.34)
    assert counts == {"A": 1.0, "B": 4.0, "C": 0.0}  # UNKNOWN and D excluded
    assert critical == {"B"}


def test_rankings_and_metrics():
    df = _make_df()
    history, future, _ = temporal_outcome_split(df, cutoff="2023-02-01")
    counts, critical = future_outcomes(history, future, critical_frac=0.34)
    rankings = baseline_rankings(history, scorer=RoadPriorityScorer(alpha=0.1, beta=2.0))

    assert rankings["count_only"][0] == "A"
    assert rankings["recency"][0] == "C"
    assert rankings["severity_only"][0] == "B"
    assert set(rankings["random"]) == {"A", "B", "C"}
    assert "UNKNOWN" not in rankings["count_only"]

    results = evaluate_rankings(rankings, counts, critical, k_values=[1, 3]).set_index("ranking")
    assert results.loc["severity_only", "recall@1"] == 1.0
    assert results.loc["count_only", "recall@1"] == 0.0
    assert results.loc["severity_only", "ndcg@1"] == pytest.approx(1.0)
    assert results.loc["count_only", "ndcg@1"] < 1.0


def test_run_outcome_evaluation_end_to_end():
    results, info = run_outcome_evaluation(_make_df(), history_frac=0.5)
    assert {"priority", "count_only", "recency", "severity_only", "random"} <= set(results["ranking"])
    assert info["rankable_areas"] >= 1


def test_severity_enum_str_matches_scorer():
    # str() of the enum must be the plain value so string lookups work
    assert str(RoadSeverity.HIGH) == "high"
    assert RoadPriorityScorer()._severity_to_numeric(str(RoadSeverity.HIGH)) == 3

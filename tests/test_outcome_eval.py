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
        ("A", "2023-01-01", "low"),
        ("A", "2023-01-02", "low"),
        ("A", "2023-01-03", "low"),
        ("B", "2023-01-04", "high"),
        ("C", "2023-01-10", "low"),
        ("UNKNOWN", "2023-01-05", "high"),
        ("B", "2023-02-01", "low"),
        ("B", "2023-02-02", "low"),
        ("B", "2023-02-03", "low"),
        ("B", "2023-02-04", "low"),
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
    rankings = baseline_rankings(
        history, scorer=RoadPriorityScorer(alpha=0.1, beta=2.0)
    )

    assert rankings["count_only"][0] == "A"
    assert rankings["recency"][0] == "C"
    assert rankings["severity_only"][0] == "B"
    assert set(rankings["random"]) == {"A", "B", "C"}
    assert "UNKNOWN" not in rankings["count_only"]

    results = evaluate_rankings(rankings, counts, critical, k_values=[1, 3]).set_index(
        "ranking"
    )
    assert results.loc["severity_only", "recall@1"] == 1.0
    assert results.loc["count_only", "recall@1"] == 0.0
    assert results.loc["severity_only", "ndcg@1"] == pytest.approx(1.0)
    assert results.loc["count_only", "ndcg@1"] < 1.0


def test_run_outcome_evaluation_end_to_end():
    results, info = run_outcome_evaluation(_make_df(), history_frac=0.5)
    assert {"priority", "count_only", "recency", "severity_only", "random"} <= set(
        results["ranking"]
    )
    assert info["rankable_areas"] >= 1


def test_severity_enum_str_matches_scorer():
    # str() of the enum must be the plain value so string lookups work
    assert str(RoadSeverity.HIGH) == "high"
    assert RoadPriorityScorer()._severity_to_numeric(str(RoadSeverity.HIGH)) == 3


def test_weight_sensitivity_grid():
    from civinlp.eval.outcome_eval import weight_sensitivity

    grid = [{"alpha": 1.0, "beta": 0.0}, {"alpha": 0.0, "beta": 1.0}]
    out = weight_sensitivity(_make_df(), grid=grid, k_values=[1])
    assert len(out) == 2 and {"alpha", "beta", "ndcg@1", "spearman_vs_future"} <= set(
        out.columns
    )


def test_resolution_days_outcome():
    from civinlp.eval.outcome_eval import resolution_days

    df = pd.DataFrame(
        {
            "area_id": ["A"] * 2 + ["B"] * 2 + ["A"] * 3 + ["B"] * 3 + ["C"],
            "created_at": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            + ["2023-02-01"] * 3
            + ["2023-02-01"] * 3
            + ["2023-02-05"],
            # Future: A closes in 1 day; B takes 10 days, one still open
            "closed_date": [None] * 4
            + ["2023-02-02"] * 3
            + ["2023-02-11", "2023-02-11", None]
            + ["2023-02-06"],
        }
    )
    days = resolution_days(df.iloc[4:])
    assert days.iloc[0] == 1.0
    assert days.iloc[5] == 4.0  # open: censored at latest creation (Feb 5)

    history, future, _ = temporal_outcome_split(df, cutoff="2023-02-01")
    rel, critical = future_outcomes(
        history, future, outcome="resolution_days", min_future=3, critical_frac=0.5
    )
    assert set(rel) == {"A", "B"}  # C not in history
    assert rel["B"] == 10.0 and rel["A"] == 1.0
    assert critical == {"B"}

    with pytest.raises(ValueError):
        future_outcomes(history, future, outcome="nope")


def test_area_split_holds_out_zones(tmp_path):
    from civinlp.eval.ablation import AblationRunner

    df = pd.DataFrame(
        {
            "text": ["Street Condition - Pothole"] * 100,
            "area_id": [f"Z{i % 10}" for i in range(100)],
            "created_at": pd.date_range("2023-01-01", periods=100, freq="h"),
        }
    )
    path = tmp_path / "d.parquet"
    df.to_parquet(path)
    runner = AblationRunner(str(path), split_method="area")
    zones = [
        set(x["area_id"]) for x in (runner.train_df, runner.val_df, runner.test_df)
    ]
    assert [len(z) for z in zones] == [7, 1, 2]
    assert not (zones[0] & zones[2] or zones[1] & zones[2] or zones[0] & zones[1])


def test_confirmed_share_outcome():
    from civinlp.eval.outcome_eval import resolution_verdict

    repaired = "The Department of Transportation inspected this complaint and repaired the problem."
    not_found = "The Department of Transportation inspected this complaint and did not find the reported problem."
    duplicate = "The Department of Transportation determined that this complaint is a duplicate."
    verdict = resolution_verdict(pd.Series([repaired, not_found, duplicate, None]))
    assert verdict.tolist()[:2] == [1.0, 0.0]
    assert verdict[2:].isna().all()

    history = pd.DataFrame({"area_id": ["A", "B", "C"]})
    future = pd.DataFrame(
        {
            "area_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 2,
            # A: 3/3 confirmed (+1 duplicate), B: 1/4, C: too few verdicts
            "resolution_description": [repaired] * 3
            + [duplicate]
            + [repaired]
            + [not_found] * 3
            + [repaired] * 2,
        }
    )
    counts, critical = future_outcomes(
        history, future, outcome="confirmed_share", min_future=3, critical_frac=0.5
    )
    assert counts == {"A": 1.0, "B": 0.25}
    assert critical == {"A"}

    # A precomputed verdict column overrides the phrase rules
    overridden = future.assign(verdict=[0.0] * 3 + [None] + [1.0] * 4 + [1.0] * 2)
    counts, _ = future_outcomes(history, overridden, outcome="confirmed_share")
    assert counts == {"A": 0.0, "B": 1.0}


def test_spearman_treats_tied_scores_as_ties():
    # Three areas tie on score; their outcomes happen to follow id order.
    # Ordering ties by id would make this ranking look perfect.
    counts = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
    scores = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.0})
    tied = evaluate_rankings({"s": scores}, counts, {"A"}, k_values=[2])
    as_list = evaluate_rankings(
        {"s": ["A", "B", "C", "D"]}, counts, {"A"}, k_values=[2]
    )
    assert as_list["spearman_vs_future"].iloc[0] == pytest.approx(1.0)
    assert tied["spearman_vs_future"].iloc[0] == pytest.approx(0.7746, abs=1e-3)
    # Order-based metrics are unchanged by passing scores
    assert tied["ndcg@2"].iloc[0] == pytest.approx(as_list["ndcg@2"].iloc[0])


def test_resolution_verdict_reviewed_templates():
    from civinlp.eval.outcome_eval import resolution_verdict

    texts = pd.Series(
        [
            "The Department of Transportation inspected the condition and made the "
            "determination that the street meets resurfacing standards and will be "
            "planned for either the next or future resurfacing seasons (2 - 3 years).",
            "Upon inspection, the Department of Transportation could not find the "
            "reported condition, therefore no action was taken.",
            "The Department of Transportation's inspectors could not find the address "
            "you reported.",
            "The Department of Transportation (DOT) inspected but did not see a "
            "violation of their rules.",
        ]
    )
    verdict = resolution_verdict(texts)
    assert verdict[0] == 1.0  # qualifies for resurfacing: need confirmed
    assert verdict[1] == 0.0
    assert pd.isna(verdict[2])  # wrong address is not an inspection verdict
    assert verdict[3] == 0.0

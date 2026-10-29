import pandas as pd

from civinlp.labels.taxonomy import RoadSeverity
from civinlp.ranking.bridge_priority import BridgePriorityScorer
from civinlp.ranking.road_priority import RoadPriorityScorer


def test_road_ranking():
    scorer = RoadPriorityScorer(alpha=1, beta=1)

    df = pd.DataFrame(
        {
            "area_id": ["A1", "A1", "A2"],
            "predicted_severity": [
                RoadSeverity.HIGH,
                RoadSeverity.LOW,
                RoadSeverity.LOW,
            ],
            "text": ["foo", "bar", "baz"],
        }
    )

    # Needs logic to map severity strings to priority input or scorer handles it?
    # The Scorer implementation I wrote maps enum/strings to numbers.

    results = scorer.rank_areas(df)

    assert len(results) == 2
    top = results.iloc[0]
    assert top["area_id"] == "A1"
    # A1: count=2, avg_sev = (3+1)/2 = 2. Score = 2 + 2 = 4 (approx)
    # A2: count=1, avg_sev = 1. Score = 1 + 1 = 2
    assert top["priority_score"] > results.iloc[1]["priority_score"]


def test_bridge_ranking():
    scorer = BridgePriorityScorer()
    # Mock bridge record
    b1 = {"bridge_id": "B1", "deck_rating": 4}  # Bad
    b2 = {"bridge_id": "B2", "deck_rating": 8}  # Good

    s1 = scorer.score_bridge(b1)
    s2 = scorer.score_bridge(b2)

    assert s1["priority_score"] > s2["priority_score"]  # B1 should be higher priority


def test_severity_can_outweigh_volume():
    # 50 minor complaints vs 5 severe ones: with normalised terms, severity matters
    df = pd.DataFrame(
        {
            "area_id": ["MINOR"] * 50 + ["SEVERE"] * 5,
            "predicted_severity": ["low"] * 50 + ["high"] * 5,
        }
    )
    ranked = RoadPriorityScorer(alpha=1, beta=1).rank_areas(df)
    assert ranked.iloc[0]["area_id"] == "SEVERE"
    assert ranked["count_norm"].max() == 1.0
    assert ranked["severity_norm"].between(0, 1).all()

    # With no severity weight, volume wins
    ranked = RoadPriorityScorer(alpha=1, beta=0).rank_areas(df)
    assert ranked.iloc[0]["area_id"] == "MINOR"

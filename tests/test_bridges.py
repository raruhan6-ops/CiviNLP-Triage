import numpy as np
import pandas as pd
import pytest

from civinlp.data.preprocess_bridges import clean_bridges, nbi_coord_to_decimal, preprocess_bridges
from civinlp.data.split import bridge_split
from civinlp.ranking.bridge_priority import BridgePriorityScorer
from civinlp.utils.io import load_parquet


def _raw_nbi():
    # Shaped like the FHWA delimited NBI file: codes are strings, "N" = not applicable
    return pd.DataFrame({
        "STRUCTURE_NUMBER_008": ["'000000000001001'", "  B2  ", "B3", "B4"],
        "DECK_COND_058": ["7", "4", "N", "N"],
        "SUPERSTRUCTURE_COND_059": ["6", "5", "N", "N"],
        "SUBSTRUCTURE_COND_060": ["7", "3", "N", "N"],
        "CULVERT_COND_062": ["N", "N", "2", "N"],
        "ADT_029": ["12000", "800", "50", ""],
        "YEAR_BUILT_027": ["1965", "1930", "1990", "2001"],
        "LAT_016": ["40451234", "40300000", "", "41000000"],
        "LONG_017": ["073585678", "074000000", "", "075300000"],
    })


def test_clean_bridges_maps_and_cleans():
    df = clean_bridges(_raw_nbi())
    assert {"bridge_id", "deck_rating", "culvert_rating", "traffic", "latitude"} <= set(df.columns)
    assert df["bridge_id"].tolist() == ["000000000001001", "B2", "B3", "B4"]

    # "N" -> missing -> 9, with a count of missing ratings per bridge
    b3 = df.iloc[2]
    assert b3["deck_rating"] == 9 and b3["culvert_rating"] == 2
    assert b3["n_missing_ratings"] == 3
    assert not b3["all_ratings_missing"]
    assert df.iloc[3]["all_ratings_missing"]
    assert np.isnan(df.iloc[3]["traffic"])


def test_nbi_coordinates():
    lat = nbi_coord_to_decimal(pd.Series(["40451234"]), 2).iloc[0]
    lon = nbi_coord_to_decimal(pd.Series(["073585678"]), 3).iloc[0]
    assert lat == pytest.approx(40 + 45 / 60 + 12.34 / 3600)
    assert lon == pytest.approx(73 + 58 / 60 + 56.78 / 3600)

    df = clean_bridges(_raw_nbi())
    assert df.iloc[0]["longitude"] < 0  # NBI stores degrees West as positive
    assert np.isnan(df.iloc[2]["latitude"])


def test_rank_bridges_on_cleaned_nbi():
    ranked = BridgePriorityScorer().rank_bridges(clean_bridges(_raw_nbi()))
    sev = ranked.set_index("bridge_id")["severity"]
    assert sev["B3"] == 7  # culvert 2 is the worst rating
    assert sev["B2"] == 6  # substructure 3
    assert sev["000000000001001"] == 3
    assert sev["B4"] == 0  # nothing usable
    # Default traffic weight: B2 (sev 6, ADT 800) outranks B3 (sev 7, ADT 50)
    assert ranked.iloc[0]["bridge_id"] == "B2"
    assert ranked["priority_rank"].tolist() == [1, 2, 3, 4]

    by_severity = BridgePriorityScorer(traffic_weight=0).rank_bridges(clean_bridges(_raw_nbi()))
    assert by_severity["bridge_id"].tolist()[:2] == ["B3", "B2"]


def test_rank_bridges_on_raw_nbi_columns():
    # Raw NBI columns must work without preprocessing too
    ranked = BridgePriorityScorer().rank_bridges(_raw_nbi())
    assert ranked.set_index("bridge_id")["severity"]["  B2  "] == 6


def test_score_bridge_handles_nan_and_n():
    scorer = BridgePriorityScorer(traffic_weight=0)
    # NaN first must not hide the real worst rating (min() with NaN is order-dependent)
    a = scorer.score_bridge({"deck_rating": float("nan"), "substructure_rating": 3})
    b = scorer.score_bridge({"substructure_rating": 3, "deck_rating": float("nan")})
    assert a["severity"] == b["severity"] == 6
    assert scorer.score_bridge({"DECK_COND_058": "N", "SUBSTRUCTURE_COND_060": "4"})["severity"] == 5
    assert scorer.score_bridge({"deck_rating": None, "traffic": float("nan")})["priority_score"] == 0


def test_bridge_split_keeps_bridges_together():
    df = pd.DataFrame({"bridge_id": [f"B{i % 10}" for i in range(40)], "year": range(40)})
    train, val, test = bridge_split(df, test_frac=0.2, val_frac=0.1)
    ids = [set(x["bridge_id"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(ids[2]) == 2 and len(ids[1]) == 1
    assert len(train) + len(val) + len(test) == len(df)


def test_preprocess_bridges_file(tmp_path):
    raw, out = tmp_path / "raw.parquet", tmp_path / "out.parquet"
    _raw_nbi().to_parquet(raw)
    preprocess_bridges(str(raw), str(out))
    assert len(load_parquet(out)) == 4

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "verdict_agreement",
    Path(__file__).resolve().parents[1] / "scripts" / "verdict_agreement.py",
)
verdict_agreement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verdict_agreement)

MAPPING = pd.DataFrame(
    {
        "template_id": [0, 1, 2, 3],
        "resolution_description": ["repaired", "did not find", "duplicate", None],
        "n": [90, 6, 3, 1],
        "verdict": ["confirmed", "not_confirmed", "no_verdict", "no_verdict"],
    }
)


def test_sheet_hides_verdicts_and_drops_empty_text():
    sheet = verdict_agreement.make_sheet(MAPPING)
    assert set(sheet["template_id"]) == {0, 1, 2}
    assert (sheet["verdict"] == "").all()


def test_score_weights_by_complaints():
    sheet = pd.DataFrame(
        {"template_id": [0, 1, 2], "verdict": ["confirmed", "no_verdict", "No_Verdict"]}
    )
    result = verdict_agreement.score(MAPPING, sheet)
    assert result["agreement"] == pytest.approx(2 / 3)
    # The one disagreement covers 6 of 99 complaints
    assert result["agreement_by_complaints"] == pytest.approx(93 / 99)
    assert [d["template_id"] for d in result["disagreements"]] == [1]


def test_score_rejects_unlabelled_templates():
    sheet = pd.DataFrame({"template_id": [0, 1], "verdict": ["confirmed", ""]})
    with pytest.raises(ValueError, match="lack a verdict"):
        verdict_agreement.score(MAPPING, sheet)

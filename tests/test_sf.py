import numpy as np
import pandas as pd

from civinlp.data.sf import preprocess_sf311, preprocess_sf_pavement, sf_complaint_types
from civinlp.models.baseline_rules import RuleBaseline


def _raw311():
    return pd.DataFrame(
        {
            "service_request_id": ["1", "2", "3", "4", "5", "5"],
            "requested_datetime": [
                "2024-01-02T10:00:00",
                "2024-01-03T10:00:00",
                "2024-01-04T10:00:00",
                "2024-01-05T10:00:00",
                "2024-01-06T10:00:00",
                "2024-01-06T10:00:00",
            ],
            "closed_date": ["2024-01-05T00:00:00", None, None, None, None, None],
            "status_description": ["Closed", "Open", "Open", "Open", "Open", "Open"],
            "status_notes": ["Case Resolved", None, None, None, None, None],
            "service_name": [
                "Street Defects",
                "Sidewalk or Curb",
                "Sidewalk and Curb",
                "Street Defect",
                "Sidewalk or Curb",
                "Sidewalk or Curb",
            ],
            "service_subtype": [
                "Pavement_Defect",
                "Sidewalk_Defect",
                "curb_or_curb_ramp_defect",
                "other",
                "Two holes in the ground.",
                "Two holes in the ground.",
            ],
            "service_details": [
                "Pavement_Defect",
                "Collapsed_sidewalk",
                None,
                "other",
                "Curb or Sidewalk Issues - Collapsed/Cracked",
                "Curb or Sidewalk Issues - Collapsed/Cracked",
            ],
            "lat": ["37.77", "37.78", "0", "37.76", "37.75", "37.75"],
            "long": ["-122.42", "-122.41", "0", "-122.43", "-122.44", "-122.44"],
            "analysis_neighborhood": [
                "Mission",
                "Mission",
                None,
                "Excelsior",
                "Excelsior",
                "Excelsior",
            ],
        }
    )


def test_sf_types_map_to_nyc_types():
    types = sf_complaint_types(_raw311())
    assert list(types["complaint_type"]) == [
        "Street Condition",
        "Sidewalk Condition",
        "Curb Condition",
        "Street Condition",
        "Sidewalk Condition",
        "Sidewalk Condition",
    ]
    assert list(types["descriptor"])[:4] == [
        "Pavement Defect",
        "Sidewalk Defect: Collapsed Sidewalk",
        "Curb Or Curb Ramp Defect",
        "Other",
    ]


def test_preprocess_sf311_schema_and_severity():
    out = preprocess_sf311(_raw311())
    assert len(out) == 5  # duplicate request id dropped
    assert out.loc[2, "area_id"] == "UNKNOWN" and np.isnan(out.loc[2, "latitude"])
    severity = [
        str(s.value) for s in RuleBaseline().predict_severity(out["text"].tolist())
    ]
    assert severity[0] == "low" and severity[1] == "high"


def test_pavement_rescaled_and_repairs_drop_previous_rating():
    raw = pd.DataFrame(
        {
            "cnn": ["A", "A", "A", "B", "B"],
            "pci_score": ["80", None, "90", "70", "60"],
            "pci_change_date": [
                "2022-08-01",
                "2023-03-01",
                "2023-08-01",
                "2022-08-01",
                "2023-08-01",
            ],
            "treatment_or_survey": [
                "Survey",
                "Treatment",
                "Survey",
                "Survey",
                "Survey",
            ],
            "latitude": ["37.7", "37.7", "37.7", "37.8", "37.8"],
            "longitude": ["-122.4", "-122.4", "-122.4", "-122.5", "-122.5"],
        }
    )
    out = preprocess_sf_pavement(raw).set_index(["segment_id", "created_at"])
    assert out["rating"].tolist() == [8.0, 9.0, 7.0, 6.0]
    # A was repaired between its surveys, so its 2023 survey has no previous rating
    assert np.isnan(out.loc[("A", pd.Timestamp("2023-08-01")), "prev_rating"])
    assert out.loc[("B", pd.Timestamp("2023-08-01")), "prev_rating"] == 7.0

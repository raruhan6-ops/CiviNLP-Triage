import pytest
from civinlp.labels.taxonomy import RoadDefectClass, RoadSeverity
from civinlp.labels.weak_labeling import get_weak_label, get_severity_label

def test_weak_labeling():
    assert get_weak_label("Pothole") == RoadDefectClass.POTHOLE
    assert get_weak_label("Street Condition", "Potholes") == RoadDefectClass.POTHOLE
    assert get_weak_label("My street has a crack") == RoadDefectClass.CRACKING
    assert get_weak_label("Rough, Pitted or Cracked Roads") == RoadDefectClass.CRACKING
    assert get_weak_label("Street Light Condition") == RoadDefectClass.STREET_CONDITION
    assert get_weak_label("Street Sign - Damaged") == RoadDefectClass.SIGNAGE
    assert get_weak_label("Blocked Driveway") == RoadDefectClass.OBSTRUCTION
    assert get_weak_label("Random thing") == RoadDefectClass.OTHER


@pytest.mark.parametrize("text", [
    "Traffic Signal Condition",  # "sign" inside "signal"
    "New street design",          # "sign" inside "design"
])
def test_weak_labeling_no_substring_sign(text):
    assert get_weak_label(text) != RoadDefectClass.SIGNAGE


@pytest.mark.parametrize("text, expected", [
    ("Road caved in after storm", RoadSeverity.HIGH),
    ("Dangerous condition", RoadSeverity.HIGH),
    ("Cracked pavement", RoadSeverity.MEDIUM),
    ("Deteriorated surface", RoadSeverity.MEDIUM),
    ("Pothole", RoadSeverity.LOW),
    # substrings that used to trigger false matches
    ("Pothole reported through app", RoadSeverity.LOW),        # rough in through
    ("Pothole near the cavern club", RoadSeverity.LOW),        # cave in cavern
    ("Pothole by enlarged sign post", RoadSeverity.LOW),       # large in enlarged
    ("Pothole near majority hall", RoadSeverity.LOW),          # major in majority
    ("", RoadSeverity.UNKNOWN),
])
def test_severity_word_boundaries(text, expected):
    assert get_severity_label(text) == expected

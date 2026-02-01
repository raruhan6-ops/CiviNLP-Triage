import pytest
from civinlp.labels.taxonomy import RoadDefectClass
from civinlp.labels.weak_labeling import get_weak_label

def test_weak_labeling():
    assert get_weak_label("Pothole") == RoadDefectClass.POTHOLE
    assert get_weak_label("My street has a crack") == RoadDefectClass.CRACKING
    assert get_weak_label("Street Light Condition") == RoadDefectClass.STREET_CONDITION
    assert get_weak_label("Random thing") == RoadDefectClass.OTHER

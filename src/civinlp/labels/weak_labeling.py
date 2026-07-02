from civinlp.labels.taxonomy import RoadDefectClass, RoadSeverity

# Mapping from NYC 311 Complaint Type (or Descriptor) to RoadDefectClass
KEYWORD_MAPPING = {
    "Pothole": RoadDefectClass.POTHOLE,
    "Street Condition": RoadDefectClass.STREET_CONDITION,
    "Rough, Pitted or Cracked Roads": RoadDefectClass.CRACKING,
    "Blocked": RoadDefectClass.OBSTRUCTION,
    "Sign": RoadDefectClass.SIGNAGE,
}

# Keywords indicating severity levels
HIGH_SEVERITY_KEYWORDS = [
    "collapse", "sinkhole", "cave", "hazard", "danger", "emergency",
    "flood", "major", "severe", "urgent", "impassable", "accident",
]
MEDIUM_SEVERITY_KEYWORDS = [
    "crack", "rough", "damage", "broken", "large", "deep", "multiple",
    "deteriorat", "erosion", "uneven", "bump",
]


def get_weak_label(complaint_type: str, descriptor: str = "") -> RoadDefectClass:
    """
    Derive a weak label from complaint metadata.
    """
    text = (str(complaint_type) + " " + str(descriptor)).lower()
    
    if "pothole" in text:
        return RoadDefectClass.POTHOLE
    if "crack" in text:
        return RoadDefectClass.CRACKING
    if "blocked" in text or "obstruction" in text:
        return RoadDefectClass.OBSTRUCTION
    if "sign" in text:
        return RoadDefectClass.SIGNAGE
    if "condition" in text:
        return RoadDefectClass.STREET_CONDITION
        
    return RoadDefectClass.OTHER


def get_severity_label(text: str, complaint_type: str = "") -> RoadSeverity:
    """
    Derive a severity label from text using danger/condition keywords.

    Strategy: keyword-based heuristic (spec option c).
      - HIGH: words indicating danger, collapse, or emergency.
      - MEDIUM: words indicating damage, cracks, or deterioration.
      - LOW: everything else with identifiable content.
      - UNKNOWN: empty or unclassifiable text.

    Args:
        text: The complaint text (cleaned).
        complaint_type: Original complaint type for additional signal.

    Returns:
        RoadSeverity enum value.
    """
    combined = (str(text) + " " + str(complaint_type)).lower()

    if not combined.strip():
        return RoadSeverity.UNKNOWN

    for keyword in HIGH_SEVERITY_KEYWORDS:
        if keyword in combined:
            return RoadSeverity.HIGH

    for keyword in MEDIUM_SEVERITY_KEYWORDS:
        if keyword in combined:
            return RoadSeverity.MEDIUM

    # If we can identify a defect type but no severity cues, default to LOW
    defect = get_weak_label(complaint_type or text)
    if defect != RoadDefectClass.OTHER:
        return RoadSeverity.LOW

    return RoadSeverity.UNKNOWN

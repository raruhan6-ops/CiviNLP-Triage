import re

from civinlp.labels.taxonomy import RoadDefectClass, RoadSeverity

# Keywords indicating severity levels.
# Plain words match as whole words with common inflections (crack -> cracks, cracked);
# a trailing "*" marks a stem that matches any word starting with it (danger* -> dangerous).
HIGH_SEVERITY_KEYWORDS = [
    "collapse", "sinkhole", "cave", "hazard*", "danger*", "emergency",
    "flood", "major", "severe", "urgent", "impassable", "accident",
]
MEDIUM_SEVERITY_KEYWORDS = [
    "crack", "rough", "damage", "broken", "large", "deep", "multiple",
    "deteriorat*", "erosion", "uneven", "bump",
]


def _keyword_pattern(keyword: str) -> str:
    if keyword.endswith("*"):
        return rf"\b{re.escape(keyword[:-1])}\w*"
    return rf"\b{re.escape(keyword)}(?:s|es|d|ed|ing)?\b"


def _compile_keywords(keywords) -> re.Pattern:
    return re.compile("|".join(_keyword_pattern(k) for k in keywords))


_HIGH_RE = _compile_keywords(HIGH_SEVERITY_KEYWORDS)
_MEDIUM_RE = _compile_keywords(MEDIUM_SEVERITY_KEYWORDS)

# Ordered (pattern, class) rules; the first match wins.
DEFECT_RULES = [
    (re.compile(r"\bpot\s?holes?\b"), RoadDefectClass.POTHOLE),
    (re.compile(r"\bcrack\w*"), RoadDefectClass.CRACKING),
    (re.compile(r"\bblock(?:ed|age)\b|\bobstruct\w*"), RoadDefectClass.OBSTRUCTION),
    (re.compile(r"\bsigns?\b|\bsignage\b"), RoadDefectClass.SIGNAGE),
    (re.compile(r"\bconditions?\b"), RoadDefectClass.STREET_CONDITION),
]


def get_weak_label(complaint_type: str, descriptor: str = "") -> RoadDefectClass:
    """
    Derive a weak label from complaint metadata.
    """
    text = (str(complaint_type) + " " + str(descriptor)).lower()

    for pattern, defect in DEFECT_RULES:
        if pattern.search(text):
            return defect

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

    if _HIGH_RE.search(combined):
        return RoadSeverity.HIGH

    if _MEDIUM_RE.search(combined):
        return RoadSeverity.MEDIUM

    # If we can identify a defect type but no severity cues, default to LOW
    defect = get_weak_label(complaint_type or text)
    if defect != RoadDefectClass.OTHER:
        return RoadSeverity.LOW

    return RoadSeverity.UNKNOWN

from enum import StrEnum

# StrEnum so str(RoadSeverity.HIGH) == "high" (a plain (str, Enum) gives
# "RoadSeverity.HIGH", which silently broke string-based lookups downstream).

class RoadDefectClass(StrEnum):
    POTHOLE = "pothole"
    CRACKING = "cracking"
    STREET_CONDITION = "street_condition"
    OBSTRUCTION = "obstruction"
    SIGNAGE = "signage"
    OTHER = "other"

class RoadSeverity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

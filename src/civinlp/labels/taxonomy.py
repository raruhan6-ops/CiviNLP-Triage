from enum import Enum

class RoadDefectClass(str, Enum):
    POTHOLE = "pothole"
    CRACKING = "cracking"
    STREET_CONDITION = "street_condition"
    OBSTRUCTION = "obstruction"
    SIGNAGE = "signage"
    OTHER = "other"

class RoadSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

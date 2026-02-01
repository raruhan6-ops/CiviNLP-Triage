from civinlp.labels.taxonomy import RoadDefectClass

# Mapping from NYC 311 Complaint Type (or Descriptor) to RoadDefectClass
KEYWORD_MAPPING = {
    "Pothole": RoadDefectClass.POTHOLE,
    "Street Condition": RoadDefectClass.STREET_CONDITION,
    "Rough, Pitted or Cracked Roads": RoadDefectClass.CRACKING,
    "Blocked": RoadDefectClass.OBSTRUCTION,
    "Sign": RoadDefectClass.SIGNAGE,
}

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

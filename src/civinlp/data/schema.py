from typing import List, Dict, Optional
import pandas as pd
from pydantic import BaseModel, ValidationError

class RoadRecord(BaseModel):
    """Internal schema for a road defect record."""
    record_id: str
    created_at: pd.Timestamp
    text: str
    complaint_type: Optional[str] = None
    descriptor: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # Flexible schema to allow extra fields during ingestion
    class Config:
        arbitrary_types_allowed = True

class RawRoadSchema:
    """Expected columns in the raw NYC 311 dataset."""
    REQUIRED_COLUMNS = [
        "unique_key",
        "created_date",
        "complaint_type"
    ]
    
    # Common mappings from NYC 311 to internal schema
    COLUMN_MAPPING = {
        "unique_key": "record_id",
        "created_date": "created_at",
        "complaint_type": "complaint_type",
        "descriptor": "descriptor",
        "latitude": "latitude",
        "longitude": "longitude",
        # Text often needs concatenation of multiple fields
    }

    @classmethod
    def validate_columns(cls, df: pd.DataFrame) -> List[str]:
        """Returns a list of missing required columns."""
        return [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]

class BridgeSchema:
    """Expected columns in NBI data."""
    REQUIRED_COLUMNS = [
        "STRUCTURE_NUMBER_008",  # bridge_id
        # Add other essential NBI columns as needed
    ]

    @classmethod
    def validate_columns(cls, df: pd.DataFrame) -> List[str]:
        return [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]

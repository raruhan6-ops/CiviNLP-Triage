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


class BridgeRecord(BaseModel):
    """Internal schema for a bridge inspection record."""
    bridge_id: str
    inspection_year: Optional[int] = None
    deck_rating: Optional[float] = None
    superstructure_rating: Optional[float] = None
    substructure_rating: Optional[float] = None
    location: Optional[str] = None
    traffic: Optional[float] = None
    derived_severity: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class PredictionOutput(BaseModel):
    """Schema for prediction outputs (roads and bridges)."""
    record_id: str
    asset_type: str  # "road" or "bridge"
    predicted_defect: Optional[str] = None
    predicted_severity: Optional[str] = None  # "low", "medium", "high"
    confidence: Optional[float] = None
    explanation: Optional[List[str]] = None
    priority_score: Optional[float] = None
    priority_rank: Optional[int] = None


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
    ]

    OPTIONAL_COLUMNS = [
        "DECK_COND_058",            # deck condition rating
        "SUPERSTRUCTURE_COND_059",  # superstructure condition
        "SUBSTRUCTURE_COND_060",    # substructure condition
        "ADT_029",                  # average daily traffic
        "YEAR_BUILT_027",           # year built
        "STATE_CODE_001",           # state
        "COUNTY_CODE_003",          # county
        "LAT_016",                  # latitude
        "LONG_017",                 # longitude
    ]

    @classmethod
    def validate_columns(cls, df: pd.DataFrame) -> List[str]:
        return [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]

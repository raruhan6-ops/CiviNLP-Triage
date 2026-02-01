import pytest
import pandas as pd
from pathlib import Path
from civinlp.data.schema import RawRoadSchema, BridgeSchema
from civinlp.data.download_roads_311 import download_roads_311
from civinlp.data.download_nbi import load_nbi_data
from civinlp.utils.io import load_parquet

def test_road_schema_validation():
    df = pd.DataFrame({
        "unique_key": ["123"],
        "created_date": ["2023-01-01"],
        "complaint_type": ["Pothole"]
    })
    assert RawRoadSchema.validate_columns(df) == []
    
    df_bad = pd.DataFrame({"Wrong": [1]})
    missing = RawRoadSchema.validate_columns(df_bad)
    assert "unique_key" in missing

def test_download_roads_mock(tmp_path, monkeypatch):
    """Test the download logic by mocking pandas read_csv."""
    output_file = tmp_path / "roads.parquet"
    
    # Mock data
    df_mock = pd.DataFrame({
        "unique_key": ["1"],
        "created_date": ["2023-01-01"],
        "complaint_type": ["Pothole"]
    })
    
    def mock_read_csv(*args, **kwargs):
        return df_mock
        
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    
    download_roads_311(str(output_file), source_url_or_path="dummy_url")
    
    assert output_file.exists()
    df_loaded = load_parquet(output_file)
    assert len(df_loaded) == 1
    assert df_loaded.iloc[0]["complaint_type"] == "Pothole"

def test_load_nbi_mock(tmp_path, monkeypatch):
    input_csv = tmp_path / "nbi.csv"
    output_parquet = tmp_path / "nbi.parquet"
    
    df_mock = pd.DataFrame({
        "STRUCTURE_NUMBER_008": ["B123"],
        "OTHER_COL": [1]
    })
    df_mock.to_csv(input_csv, index=False)
    
    load_nbi_data(str(input_csv), str(output_parquet))
    
    assert output_parquet.exists()
    df_loaded = load_parquet(output_parquet)
    assert len(df_loaded) == 1

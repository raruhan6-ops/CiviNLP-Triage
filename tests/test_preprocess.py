import pytest
import pandas as pd
from civinlp.utils.text_clean import clean_text
from civinlp.utils.pii_redaction import redact_pii
from civinlp.data.preprocess_roads import preprocess_roads
from civinlp.utils.io import load_parquet

def test_clean_text():
    raw = "  Hello   World! \u00a0 "
    expected = "Hello World!"
    assert clean_text(raw) == expected
    assert clean_text(None) == ""

def test_redact_pii():
    text = "Contact me at user@example.com or 123-456-7890."
    redacted = redact_pii(text)
    assert "[EMAIL_REDACTED]" in redacted
    assert "[PHONE_REDACTED]" in redacted
    assert "user@example.com" not in redacted
    assert "123-456-7890" not in redacted

def test_preprocess_flow(tmp_path, monkeypatch):
    input_file = tmp_path / "raw.parquet"
    output_file = tmp_path / "processed.parquet"
    
    # Create mock raw data
    df = pd.DataFrame({
        "Unique Key": ["1", "2"],
        "Created Date": ["2023-01-01", "2023-01-02"],
        "Complaint Type": ["Pothole", "Noise"],
        "Descriptor": ["Large", None],
        "Incident Zip": ["10001", None]  # Testing area_id creation
    })
    df.to_parquet(input_file)
    
    preprocess_roads(str(input_file), str(output_file))
    
    assert output_file.exists()
    df_proc = load_parquet(output_file)
    
    assert "text" in df_proc.columns
    assert "area_id" in df_proc.columns
    assert "record_id" in df_proc.columns  # Renamed from Unique Key
    
    # Check content
    row1 = df_proc.iloc[0]
    assert "Pothole" in row1['text']
    assert "Large" in row1['text']
    assert row1['area_id'] == "10001"
    
    row2 = df_proc.iloc[1]
    assert row2['area_id'] == "UNKNOWN"

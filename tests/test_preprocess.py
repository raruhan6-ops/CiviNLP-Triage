import pandas as pd
import pytest

from civinlp.data.preprocess_roads import normalize_zip, preprocess_roads
from civinlp.utils.io import load_parquet
from civinlp.utils.pii_redaction import redact_pii
from civinlp.utils.text_clean import clean_text


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
    # incident_zip as floats, as pandas reads it from a CSV with missing values
    df = pd.DataFrame(
        {
            "unique_key": ["1", "2", "3"],
            "created_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "complaint_type": ["Pothole", "Street Condition", "Noise - Residential"],
            "descriptor": ["Large", None, "Loud Music"],
            "incident_zip": [10001.0, None, 10002.0],
        }
    )
    df.to_parquet(input_file)

    preprocess_roads(str(input_file), str(output_file))

    assert output_file.exists()
    df_proc = load_parquet(output_file)
    assert len(df_proc) == 2  # non-road "Noise" row dropped

    assert "text" in df_proc.columns
    assert "area_id" in df_proc.columns
    assert "record_id" in df_proc.columns  # Renamed from Unique Key

    # Check content
    row1 = df_proc.iloc[0]
    assert "Pothole" in row1["text"]
    assert "Large" in row1["text"]
    assert row1["area_id"] == "10001"

    row2 = df_proc.iloc[1]
    assert row2["area_id"] == "UNKNOWN"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("10001", "10001"),
        (10001.0, "10001"),
        ("10001.0", "10001"),
        (7001.0, "07001"),
        ("10001-1234", "10001"),
        (None, "UNKNOWN"),
        (float("nan"), "UNKNOWN"),
        ("N/A", "UNKNOWN"),
    ],
)
def test_normalize_zip(raw, expected):
    assert normalize_zip(raw) == expected


@pytest.mark.parametrize(
    "text, redacted, kept",
    [
        ("Mail a@b.co now", "[EMAIL_REDACTED]", None),
        ("Call (212) 555-1234", "[PHONE_REDACTED]", None),
        ("Call +1 212 555 1234", "[PHONE_REDACTED]", None),
        ("Call 212.555.1234", "[PHONE_REDACTED]", None),
        ("Pothole at 123 W 45th St near corner", "[ADDRESS_REDACTED]", "near corner"),
        ("Outside 55-10 Queens Blvd.", "[ADDRESS_REDACTED]", None),
        ("Lot BBL 1012340056 flagged", None, "1012340056"),  # bare 10-digit id kept
        ("Street Condition - Pothole", None, "Street Condition - Pothole"),
        ("Cave-in over 3 feet wide", None, "3 feet wide"),
    ],
)
def test_redact_pii_cases(text, redacted, kept):
    out = redact_pii(text)
    if redacted:
        assert redacted in out
    else:
        assert "REDACTED" not in out
    if kept:
        assert kept in out


def test_setup_logging_attaches_file_handler(tmp_path):
    import logging

    from civinlp.utils.logging_config import setup_logging

    root = logging.getLogger()
    root.addHandler(logging.StreamHandler())  # something already configured logging
    try:
        setup_logging(log_dir=str(tmp_path), name="t")
        setup_logging(log_dir=str(tmp_path), name="t")  # repeat call must not duplicate
        ours = [h for h in root.handlers if getattr(h, "_civinlp", False)]
        assert [type(h) for h in ours] == [logging.FileHandler]
        logging.getLogger("x").warning("hello")
        assert any("hello" in f.read_text() for f in tmp_path.glob("t_*.log"))
    finally:
        for h in list(root.handlers):
            if getattr(h, "_civinlp", False) or type(h) is logging.StreamHandler:
                root.removeHandler(h)
                h.close()

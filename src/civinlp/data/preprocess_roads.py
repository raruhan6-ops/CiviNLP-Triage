import argparse
import re
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path

from civinlp.utils.io import load_parquet, save_parquet
from civinlp.utils.text_clean import clean_text
from civinlp.utils.pii_redaction import redact_pii
from civinlp.data.schema import RawRoadSchema, is_road_complaint

logger = logging.getLogger(__name__)

tqdm.pandas()


def normalize_zip(value) -> str:
    """Normalise a ZIP code to 5 digits ("10001.0" -> "10001", 7001.0 -> "07001")."""
    if value is None or pd.isna(value):
        return "UNKNOWN"
    s = str(value).strip()
    match = re.fullmatch(r"(\d{1,5})(?:\.0+)?", s) or re.match(r"(\d{5})(?:-\d{4})?$", s)
    if not match:
        return "UNKNOWN"
    return match.group(1).zfill(5)


def preprocess_roads(input_path: str, output_path: str, road_only: bool = True):
    """
    Load raw road data, clean text, redact PII, and save processed data.

    If `road_only` is True, rows whose complaint_type is not road-related are dropped.
    """
    logger.info(f"Loading raw data from {input_path}")
    df = load_parquet(input_path)

    if road_only and "complaint_type" in df.columns:
        before = len(df)
        df = df[df["complaint_type"].map(is_road_complaint)].reset_index(drop=True)
        logger.info(f"Kept {len(df)}/{before} road-related rows")

    # Ensure text column exists (concatenating Complaint Type, Descriptor)
    # Note: 'text' field is primary input. 
    # We might want to construct it if it doesn't exist, but spec says "text: string (concatenated relevant text fields)"
    
    # Mapping raw columns to internal names if needed
    # For now, let's assume raw data. We create 'text' column.
    
    # Helper to combine text fields
    def combine_text(row):
        parts = [
            str(row.get("complaint_type", "")),
            str(row.get("descriptor", "")),
            # Description is often not in the public 311 1GB slices, but "Resolution Description" might be.
            # We'll stick to what we have.
        ]
        return " - ".join([p for p in parts if p and p != "nan"])

    logger.info("Constructing text field...")
    df['text'] = df.apply(combine_text, axis=1)
    
    logger.info("Cleaning text and redacting PII...")
    df['text'] = df['text'].progress_apply(clean_text)
    df['text'] = df['text'].progress_apply(redact_pii)
    
    # Rename columns to internal schema (lowercase, snake_case)
    initial_cols = len(df.columns)
    
    rename_map = {k: v for k, v in RawRoadSchema.COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    # Ensure area_id (use Incident Zip if available)
    if 'incident_zip' in df.columns:
        df['area_id'] = df['incident_zip'].map(normalize_zip)
    else:
        df['area_id'] = 'UNKNOWN'
        
    # Select only columns of interest? Or keep all?
    # Spec "2.4.1 Road Record Schema" lists required.
    # We'll accept extra columns but ensure basics.
    
    save_parquet(df, output_path)
    logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Preprocess Road Data")
    parser.add_argument("--input", type=str, default="data_raw/roads_311.parquet")
    parser.add_argument("--output", type=str, default="data_processed/roads_processed.parquet")
    parser.add_argument("--keep-all", action="store_true", help="Keep non-road complaint types")

    args = parser.parse_args()
    preprocess_roads(args.input, args.output, road_only=not args.keep_all)

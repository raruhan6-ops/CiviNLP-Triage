import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from civinlp.utils.io import save_parquet, load_csv
from civinlp.data.schema import RawRoadSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default URL for NYC 311 data (last 1000 records example)
NYC_311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit=2000&$order=created_date%20DESC"

def download_roads_311(output_path: str, source_url_or_path: Optional[str] = None, limit: Optional[int] = None):
    """
    Download or load NYC 311 data, simple validation, and save to Parquet.
    
    Args:
        output_path: Destination for the raw parquet file.
        source_url_or_path: URL or local path to CSV. If None, uses default NYC Open Data URL.
        limit: Max rows to load (mostly for testing).
    """
    source = source_url_or_path or NYC_311_URL
    
    if limit and "http" in source and "$limit" not in source:
        source += f"&$limit={limit}"
    
    logger.info(f"Fetching data from {source}...")
    
    # Read CSV (pandas handles URLs seamlessly)
    try:
        if limit:
            df = pd.read_csv(source, nrows=limit)
        else:
            df = pd.read_csv(source)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(f"Loaded {len(df)} rows.")

    # Basic Validation
    missing_cols = RawRoadSchema.validate_columns(df)
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}. Proceeding, but downstream steps may fail.")
    else:
        logger.info("Schema validation passed.")

    # Save prompt
    save_parquet(df, output_path)
    logger.info(f"Saved raw data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NYC 311 Road Data")
    parser.add_argument("--output", type=str, default="data_raw/roads_311.parquet", help="Output path")
    parser.add_argument("--source", type=str, help="Local CSV path or URL")
    parser.add_argument("--limit", type=int, help="Limit number of rows")
    
    args = parser.parse_args()
    download_roads_311(args.output, args.source, args.limit)

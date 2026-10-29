import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
import logging

from civinlp.utils.io import save_parquet, load_csv
from civinlp.data.schema import RawRoadSchema, ROAD_COMPLAINT_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NYC_311_RESOURCE = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
DEFAULT_LIMIT = 2000


def build_311_url(
    base_url: str = NYC_311_RESOURCE, limit: Optional[int] = DEFAULT_LIMIT
) -> str:
    """
    Build a Socrata query URL restricted to road-related complaint types.

    Existing query parameters in `base_url` are kept, except `$limit`, which is
    overridden when `limit` is given.
    """
    parts = urlsplit(base_url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    if "$where" not in params:
        types = ",".join(f"'{t}'" for t in ROAD_COMPLAINT_TYPES)
        params["$where"] = f"complaint_type in({types})"
    params.setdefault("$order", "created_date DESC")
    if limit:
        params["$limit"] = str(limit)
    query = urlencode(params, quote_via=quote, safe="$,'()")
    return urlunsplit(parts._replace(query=query))


# Default URL: latest road-related NYC 311 requests
NYC_311_URL = build_311_url()


def download_roads_311(output_path: str, source_url_or_path: Optional[str] = None, limit: Optional[int] = None):
    """
    Download or load NYC 311 data, simple validation, and save to Parquet.

    Args:
        output_path: Destination for the raw parquet file.
        source_url_or_path: URL or local path to CSV. If None, uses default NYC Open Data URL.
        limit: Max rows to load (mostly for testing).
    """
    source = source_url_or_path or NYC_311_RESOURCE
    is_url = source.startswith(("http://", "https://"))

    if is_url and "data.cityofnewyork.us/resource/" in source:
        source = build_311_url(source, limit or DEFAULT_LIMIT)

    logger.info(f"Fetching data from {source}...")

    # Read CSV (pandas handles URLs seamlessly). Keep ZIP codes as strings so
    # missing values don't turn them into floats like "10001.0".
    read_kwargs = {"dtype": {"incident_zip": str}}
    if limit and not is_url:
        read_kwargs["nrows"] = limit
    try:
        df = pd.read_csv(source, **read_kwargs)
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

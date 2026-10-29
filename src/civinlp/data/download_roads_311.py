import argparse
import io
import logging
import time
from typing import Optional
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

import pandas as pd

from civinlp.data.schema import ROAD_COMPLAINT_TYPES, RawRoadSchema
from civinlp.utils.io import save_parquet

logger = logging.getLogger(__name__)

NYC_311_RESOURCE = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
DEFAULT_LIMIT = 2000
SELECT_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "status",
    "complaint_type",
    "descriptor",
    "incident_zip",
    "borough",
    "latitude",
    "longitude",
]


def build_311_url(
    base_url: str = NYC_311_RESOURCE,
    limit: Optional[int] = DEFAULT_LIMIT,
    offset: Optional[int] = None,
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
    # Full rows have ~40 columns; large pulls get cut off without a narrower select
    params.setdefault("$select", ",".join(SELECT_COLUMNS))
    # unique_key breaks ties so paging with $offset is stable
    params.setdefault("$order", "created_date DESC,unique_key DESC")
    if limit:
        params["$limit"] = str(limit)
    if offset:
        params["$offset"] = str(offset)
    query = urlencode(params, quote_via=quote, safe="$,'()")
    return urlunsplit(parts._replace(query=query))


# Default URL: latest road-related NYC 311 requests
NYC_311_URL = build_311_url()
PAGE_SIZE = 5000
REQUEST_TIMEOUT = 60  # seconds without data before a page is retried


def fetch_csv(
    url: str, timeout: float = REQUEST_TIMEOUT, **read_kwargs
) -> pd.DataFrame:
    """GET a CSV with a socket timeout (pd.read_csv on a URL can hang forever)."""
    with urlopen(url, timeout=timeout) as response:
        return pd.read_csv(io.BytesIO(response.read()), **read_kwargs)


def read_socrata_paged(
    base_url: str,
    limit: int,
    page_size: Optional[int] = None,
    retries: int = 3,
    **read_kwargs,
) -> pd.DataFrame:
    """Fetch up to `limit` rows in pages; large single responses tend to get cut off."""
    page_size = page_size or PAGE_SIZE
    pages, offset = [], 0
    while offset < limit:
        n = min(page_size, limit - offset)
        url = build_311_url(base_url, n, offset)
        for attempt in range(1, retries + 1):
            try:
                page = fetch_csv(url, **read_kwargs)
                break
            except Exception as e:
                if attempt == retries:
                    raise
                logger.warning(
                    f"Page at offset {offset} failed ({e}); retry {attempt}/{retries - 1}"
                )
                time.sleep(2 * attempt)
        pages.append(page)
        logger.info(f"Fetched {offset + len(page)} rows")
        if len(page) < n:
            break
        offset += n
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()


def download_roads_311(
    output_path: str,
    source_url_or_path: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Download or load NYC 311 data, simple validation, and save to Parquet.

    Args:
        output_path: Destination for the raw parquet file.
        source_url_or_path: URL or local path to CSV. If None, uses default NYC Open Data URL.
        limit: Max rows to load (mostly for testing).
    """
    source = source_url_or_path or NYC_311_RESOURCE
    is_url = source.startswith(("http://", "https://"))

    is_socrata = is_url and "data.cityofnewyork.us/resource/" in source

    logger.info(f"Fetching data from {source}...")

    # Read CSV (pandas handles URLs seamlessly). Keep ZIP codes as strings so
    # missing values don't turn them into floats like "10001.0".
    read_kwargs = {"dtype": {"incident_zip": str}}
    if limit and not is_url:
        read_kwargs["nrows"] = limit
    try:
        if is_socrata:
            df = read_socrata_paged(source, limit or DEFAULT_LIMIT, **read_kwargs)
        else:
            df = pd.read_csv(source, **read_kwargs)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(f"Loaded {len(df)} rows.")

    # Basic Validation
    missing_cols = RawRoadSchema.validate_columns(df)
    if missing_cols:
        logger.warning(
            f"Missing expected columns: {missing_cols}. Proceeding, but downstream steps may fail."
        )
    else:
        logger.info("Schema validation passed.")

    # Save prompt
    save_parquet(df, output_path)
    logger.info(f"Saved raw data to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download NYC 311 Road Data")
    parser.add_argument(
        "--output", type=str, default="data_raw/roads_311.parquet", help="Output path"
    )
    parser.add_argument("--source", type=str, help="Local CSV path or URL")
    parser.add_argument("--limit", type=int, help="Limit number of rows")

    args = parser.parse_args()
    download_roads_311(args.output, args.source, args.limit)

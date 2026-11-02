import argparse
import gzip
import io
import logging
import os
import time
from typing import Optional
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

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
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> str:
    """
    Build a Socrata query URL restricted to road-related complaint types.

    Existing query parameters in `base_url` are kept, except `$limit`, which is
    overridden when `limit` is given. `since` (e.g. "2025-11-01") keeps only
    requests created on or after that date; `until` keeps only those created
    before it.
    """
    parts = urlsplit(base_url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    if "$where" not in params:
        types = ",".join(f"'{t}'" for t in ROAD_COMPLAINT_TYPES)
        params["$where"] = f"complaint_type in({types})"
        if since:
            params["$where"] += f" AND created_date>='{since}'"
        if until:
            params["$where"] += f" AND created_date<'{until}'"
    # Full rows have ~40 columns; large pulls get cut off without a narrower select
    params.setdefault("$select", ",".join(SELECT_COLUMNS))
    # unique_key breaks ties so paging with $offset is stable
    params.setdefault("$order", "created_date DESC,unique_key DESC")
    if limit:
        params["$limit"] = str(limit)
    if offset:
        params["$offset"] = str(offset)
    query = urlencode(params, quote_via=quote, safe="$,'()<>=")
    return urlunsplit(parts._replace(query=query))


# Default URL: latest road-related NYC 311 requests
NYC_311_URL = build_311_url()
PAGE_SIZE = 5000
REQUEST_TIMEOUT = 60  # seconds without data before a page is retried


def fetch_csv(
    url: str, timeout: float = REQUEST_TIMEOUT, **read_kwargs
) -> pd.DataFrame:
    """
    GET a CSV with a socket timeout (pd.read_csv on a URL can hang forever).
    Asks for gzip, which cuts the transfer several-fold on slow links.
    """
    request = Request(url, headers={"Accept-Encoding": "gzip"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    return pd.read_csv(io.BytesIO(body), **read_kwargs)


def read_socrata_paged(
    base_url: str,
    limit: int,
    page_size: Optional[int] = None,
    retries: int = 3,
    since: Optional[str] = None,
    until: Optional[str] = None,
    **read_kwargs,
) -> pd.DataFrame:
    """Fetch up to `limit` rows in pages; large single responses tend to get cut off."""
    page_size = page_size or PAGE_SIZE
    pages, offset = [], 0
    while offset < limit:
        n = min(page_size, limit - offset)
        url = build_311_url(base_url, n, offset, since, until)
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


def download_roads_311_monthly(
    output_path: str,
    since: str,
    until: Optional[str] = None,
    chunk_dir: Optional[str] = None,
    page_size: Optional[int] = None,
    retries: int = 3,
    max_rows_per_month: int = 100_000,
) -> pd.DataFrame:
    """
    Download road 311 requests one calendar month at a time.

    Each month is saved to `chunk_dir` as soon as it is complete, and months
    already on disk are skipped, so an interrupted run resumes where it
    stopped. The month still in progress is always re-fetched. Months that
    fail are logged and skipped; run again to fill them in. The combined
    months are written to `output_path`.
    """
    chunk_dir = chunk_dir or os.path.splitext(output_path)[0] + "_months"
    os.makedirs(chunk_dir, exist_ok=True)
    now = pd.Timestamp.now()
    end = pd.Timestamp(until) if until else now
    starts = pd.date_range(pd.Timestamp(since).normalize(), end, freq="MS")
    starts = starts[starts < end]
    if len(starts) == 0 or starts[0] > pd.Timestamp(since):
        starts = starts.insert(0, pd.Timestamp(since))

    read_kwargs = {"dtype": {"incident_zip": str}}
    failed = []
    for start in starts:
        stop = min(start + pd.offsets.MonthBegin(1), end)
        path = os.path.join(chunk_dir, f"roads_311_{start:%Y-%m}.parquet")
        if os.path.exists(path) and stop < now:
            logger.info(f"{start:%Y-%m}: already downloaded")
            continue
        try:
            month = read_socrata_paged(
                NYC_311_RESOURCE,
                max_rows_per_month,
                page_size=page_size,
                retries=retries,
                since=f"{start:%Y-%m-%dT%H:%M:%S}",
                until=f"{stop:%Y-%m-%dT%H:%M:%S}",
                **read_kwargs,
            )
        except Exception as e:
            logger.error(f"{start:%Y-%m}: failed ({e}); rerun to retry")
            failed.append(f"{start:%Y-%m}")
            continue
        save_parquet(month, path)
        logger.info(f"{start:%Y-%m}: {len(month)} rows")

    months = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".parquet"))
    frames = [pd.read_parquet(os.path.join(chunk_dir, f)) for f in months]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "unique_key" in df.columns:
        df = df.drop_duplicates("unique_key")
    save_parquet(df, output_path)
    logger.info(f"Saved {len(df)} rows from {len(months)} months to {output_path}")
    if failed:
        logger.warning(f"Missing months: {', '.join(failed)}")
    return df


def download_roads_311(
    output_path: str,
    source_url_or_path: Optional[str] = None,
    limit: Optional[int] = None,
    since: Optional[str] = None,
    page_size: Optional[int] = None,
    retries: int = 3,
):
    """
    Download or load NYC 311 data, simple validation, and save to Parquet.

    Args:
        output_path: Destination for the raw parquet file.
        source_url_or_path: URL or local path to CSV. If None, uses default NYC Open Data URL.
        limit: Max rows to load (mostly for testing).
        since: Socrata only; keep requests created on or after this date
            (YYYY-MM-DD). Set `limit` high enough to cover the whole window.
        page_size, retries: Socrata paging; smaller pages fail less on slow links.
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
            df = read_socrata_paged(
                source,
                limit or DEFAULT_LIMIT,
                page_size=page_size,
                retries=retries,
                since=since,
                **read_kwargs,
            )
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
    parser.add_argument(
        "--since", type=str, help="Only requests created on/after this date"
    )

    parser.add_argument("--page_size", type=int, help="Rows per Socrata request")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per page")
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="With --since: fetch month by month, saving and resuming per month",
    )
    parser.add_argument("--until", type=str, help="With --monthly: end date")

    args = parser.parse_args()
    if args.monthly:
        if not args.since:
            parser.error("--monthly needs --since")
        download_roads_311_monthly(
            args.output,
            args.since,
            args.until,
            page_size=args.page_size,
            retries=args.retries,
        )
        raise SystemExit(0)
    download_roads_311(
        args.output, args.source, args.limit, args.since, args.page_size, args.retries
    )

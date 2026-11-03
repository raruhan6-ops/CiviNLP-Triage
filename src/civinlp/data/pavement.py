"""
NYC DOT street pavement ratings: an independent measure of road condition.

DOT inspectors rate street segments from 1 (poor) to 10 (good); each segment
is re-rated roughly every 14 months. The ratings never touch 311 data, so they
can grade a complaint-based ranking against measured condition rather than
against more complaints.

    python -m civinlp.data.pavement --raw data_raw/pavement_ratings.csv \
        --output data_processed/pavement_ratings.parquet
"""

import argparse
import logging
import os
import re
import time
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd

from civinlp.data.download_roads_311 import fetch_csv
from civinlp.utils.io import save_parquet

logger = logging.getLogger(__name__)

PAVEMENT_RESOURCE = "https://data.cityofnewyork.us/resource/6yyb-pb25.csv"
PAVEMENT_COLUMNS = [
    "oftcode",
    "boroughname",
    "road_type",
    "systemrating",
    "nonratingreason",
    "inspection",
    "the_geom",
]
PAGE_SIZE = 50_000

_POINT_RE = re.compile(r"(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)")


def download_pavement_ratings(
    output_path: str, page_size: int = PAGE_SIZE, retries: int = 3
) -> pd.DataFrame:
    """Download every rating row (all years, rated or not) to a CSV."""
    pages, offset = [], 0
    while True:
        params = {
            "$select": ",".join(PAVEMENT_COLUMNS),
            "$order": ":id",  # stable order, so paging with $offset is exact
            "$limit": str(page_size),
            "$offset": str(offset),
        }
        url = f"{PAVEMENT_RESOURCE}?{urlencode(params, quote_via=quote, safe='$,:')}"
        for attempt in range(1, retries + 1):
            try:
                page = fetch_csv(url, timeout=300, dtype={"oftcode": str})
                break
            except Exception as e:
                if attempt == retries:
                    raise
                logger.warning(f"Offset {offset} failed ({e}); retry {attempt}")
                time.sleep(2 * attempt)
        pages.append(page)
        logger.info(f"Fetched {offset + len(page)} rows")
        if len(page) < page_size:
            break
        offset += page_size
    df = pd.concat(pages, ignore_index=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def line_midpoint(wkt: str) -> tuple:
    """
    (latitude, longitude) of a WKT (MULTI)LINESTRING: the mean of its
    vertices, which for a single street block lies on or next to the block.
    """
    points = _POINT_RE.findall(str(wkt))
    if not points:
        return (np.nan, np.nan)
    lon, lat = np.array(points, dtype=float).T
    return (float(lat.mean()), float(lon.mean()))


def preprocess_pavement(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Keep actual ratings and give each a point location.

    Rows with a `nonratingreason` (duplicate, construction, multi-pass, ...)
    carry a rating of 0 and are dropped, as are rows without a date or
    geometry. Output columns: segment_id, created_at (inspection date),
    rating, latitude, longitude, borough, road_type.
    """
    rating = pd.to_numeric(raw["systemrating"], errors="coerce")
    inspected = pd.to_datetime(raw["inspection"], errors="coerce")
    keep = raw["nonratingreason"].isna() & (rating > 0) & inspected.notna()
    df = raw[keep]
    lat, lon = zip(*df["the_geom"].map(line_midpoint)) if len(df) else ((), ())
    out = pd.DataFrame(
        {
            "segment_id": df["oftcode"].astype(str).to_numpy(),
            "created_at": inspected[keep].to_numpy(),
            "rating": rating[keep].to_numpy(),
            "latitude": np.asarray(lat, dtype=float),
            "longitude": np.asarray(lon, dtype=float),
            "borough": df["boroughname"].to_numpy(),
            "road_type": df["road_type"].to_numpy(),
        }
    )
    out = out[out["latitude"].notna()].reset_index(drop=True)
    logger.info(f"Kept {len(out)} of {len(raw)} rows as located ratings")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="NYC DOT street pavement ratings")
    parser.add_argument("--raw", type=str, default="data_raw/pavement_ratings.csv")
    parser.add_argument(
        "--output", type=str, default="data_processed/pavement_ratings.parquet"
    )
    parser.add_argument(
        "--skip_download", action="store_true", help="Use the CSV already at --raw"
    )
    args = parser.parse_args()
    if args.skip_download:
        raw = pd.read_csv(args.raw, dtype={"oftcode": str})
    else:
        raw = download_pavement_ratings(args.raw)
    save_parquet(preprocess_pavement(raw), args.output)

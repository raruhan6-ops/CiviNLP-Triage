"""
San Francisco data for replicating the NYC evaluation in a second city.

Two public DataSF datasets:

- SF311 cases (vw6y-z8j6): street, sidewalk and curb defect requests.
- Public Works pavement condition history (78va-8dhi): Pavement Condition
  Index (0-100) surveys of nearly every block once a year, in a July-October
  campaign, plus records of repair treatments.

Both are converted to the NYC schemas, so the rolling evaluation runs on them
unchanged:

    python -m civinlp.data.sf --since 2016-07-01 --until 2026-01-01 \
        --output data_processed/sf311_roads.parquet \
        --pavement_output data_processed/sf_pavement_ratings.parquet
"""

import argparse
import logging
import os
import time
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd

from civinlp.data.download_roads_311 import fetch_csv
from civinlp.data.pavement import add_previous_rating
from civinlp.utils.io import save_parquet

logger = logging.getLogger(__name__)

SF311_RESOURCE = "https://data.sf.gov/resource/vw6y-z8j6.csv"
SF_PAVEMENT_RESOURCE = "https://data.sf.gov/resource/78va-8dhi.csv"
PAGE_SIZE = 50_000

SF311_COLUMNS = [
    "service_request_id",
    "requested_datetime",
    "closed_date",
    "status_description",
    "status_notes",
    "service_name",
    "service_subtype",
    "service_details",
    "lat",
    "long",
    "analysis_neighborhood",
]
# Service names of road defect requests; the names changed during the period
SF311_SERVICES = [
    "Street Defects",
    "Street Defect",
    "Sidewalk or Curb",
    "Sidewalk and Curb",
]

SF_PAVEMENT_COLUMNS = [
    "cnn",
    "pci_score",
    "pci_change_date",
    "treatment_or_survey",
    "latitude",
    "longitude",
]

# SF subtypes mapped onto the NYC complaint types; the subtype (with its
# details) becomes the descriptor
CURB_SUBTYPES = {"curb_or_curb_ramp_defect"}
SIDEWALK_SUBTYPES = {
    "sidewalk_defect",
    "missing_side_sewer_vent_cover",
    "damaged_side_sewer_vent_cover",
    "public_stairway_defect",
}
STREET_SUBTYPES = {
    "pavement_defect",
    "manhole_cover_off",
    "construction_plate_shifted",
    "utility_excavation",
}


def _download(resource, select, where, output_path, retries=3):
    pages, offset = [], 0
    while True:
        params = {
            "$select": ",".join(select),
            "$where": where,
            "$order": ":id",  # stable order, so paging with $offset is exact
            "$limit": str(PAGE_SIZE),
            "$offset": str(offset),
        }
        url = f"{resource}?{urlencode(params, quote_via=quote, safe='$,:')}"
        for attempt in range(1, retries + 1):
            try:
                page = fetch_csv(url, timeout=300, dtype=str)
                break
            except Exception as e:
                if attempt == retries:
                    raise
                logger.warning(f"Offset {offset} failed ({e}); retry {attempt}")
                time.sleep(2 * attempt)
        pages.append(page)
        logger.info(f"Fetched {offset + len(page)} rows")
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    df = pd.concat(pages, ignore_index=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def download_sf311(output_path: str, since: str, until: str) -> pd.DataFrame:
    """Road defect requests created in [since, until) to a CSV."""
    services = ", ".join(f"'{s}'" for s in SF311_SERVICES)
    where = (
        f"service_name in ({services}) AND requested_datetime >= '{since}' "
        f"AND requested_datetime < '{until}'"
    )
    return _download(SF311_RESOURCE, SF311_COLUMNS, where, output_path)


def download_sf_pavement(output_path: str) -> pd.DataFrame:
    """Every survey and treatment record to a CSV."""
    return _download(
        SF_PAVEMENT_RESOURCE,
        SF_PAVEMENT_COLUMNS,
        "pci_change_date IS NOT NULL",
        output_path,
    )


def _label(value) -> str:
    """'Lifted_sidewalk_tree_roots' -> 'Lifted Sidewalk Tree Roots'."""
    text = str(value).replace("_", " ").strip() if pd.notna(value) else ""
    return " ".join(w.capitalize() for w in text.split())


def sf_complaint_types(raw: pd.DataFrame) -> pd.DataFrame:
    """
    complaint_type and descriptor in the NYC style.

    Each subtype maps to Street, Sidewalk or Curb Condition; subtypes outside
    the known sets (e.g. "other" or free text) take the type of their service
    name and the descriptor "Other". The descriptor is the subtype, with the
    details appended when they add something (the kind of sidewalk defect).
    """
    subtype = raw["service_subtype"].astype(str).str.strip().str.lower()
    street_service = (
        raw["service_name"].astype(str).str.lower().str.startswith("street")
    )
    ctype = np.select(
        [
            subtype.isin(CURB_SUBTYPES),
            subtype.isin(SIDEWALK_SUBTYPES),
            subtype.isin(STREET_SUBTYPES),
            street_service,
        ],
        [
            "Curb Condition",
            "Sidewalk Condition",
            "Street Condition",
            "Street Condition",
        ],
        default="Sidewalk Condition",
    )
    known = subtype.isin(CURB_SUBTYPES | SIDEWALK_SUBTYPES | STREET_SUBTYPES)
    sub_label = raw["service_subtype"].map(_label).where(known, "Other")
    details = raw["service_details"].map(_label)
    adds = known & (details != "") & (details.str.lower() != sub_label.str.lower())
    descriptor = sub_label.where(~adds, sub_label + ": " + details)
    return pd.DataFrame(
        {"complaint_type": ctype, "descriptor": descriptor}, index=raw.index
    )


def preprocess_sf311(raw: pd.DataFrame) -> pd.DataFrame:
    """
    SF311 road defect requests in the processed NYC schema. The area is the
    analysis neighborhood (SF311 has no zip code); requests with no location
    (coordinates of 0) keep the neighborhood but no coordinates.
    """
    types = sf_complaint_types(raw)
    lat = pd.to_numeric(raw["lat"], errors="coerce")
    lon = pd.to_numeric(raw["long"], errors="coerce")
    located = lat.notna() & lon.notna() & (lat != 0) & (lon != 0)
    area = raw["analysis_neighborhood"].fillna("UNKNOWN").astype(str).str.strip()
    out = pd.DataFrame(
        {
            "record_id": raw["service_request_id"].astype(str),
            "created_at": pd.to_datetime(raw["requested_datetime"], errors="coerce"),
            "closed_date": pd.to_datetime(raw["closed_date"], errors="coerce"),
            "status": raw["status_description"],
            "complaint_type": types["complaint_type"],
            "descriptor": types["descriptor"],
            "latitude": lat.where(located),
            "longitude": lon.where(located),
            "resolution_description": raw["status_notes"],
            "area_id": area.replace("", "UNKNOWN"),
        }
    )
    out["text"] = out["complaint_type"] + " - " + out["descriptor"]
    out = out[out["created_at"].notna()].sort_values("created_at")
    out = out.drop_duplicates("record_id").reset_index(drop=True)
    logger.info(f"Kept {len(out)} of {len(raw)} SF311 rows")
    return out


def preprocess_sf_pavement(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Surveys in the NYC pavement schema, with PCI divided by 10 so the NYC
    outcomes apply unchanged (10 minus rating; "poor" below 7, i.e. PCI below
    70, where SF's "good" band starts).

    A segment repaired (a treatment record) after its previous survey gets no
    previous rating, so rating-change outcomes skip it: a repair, not wear,
    explains that change.
    """
    rating = pd.to_numeric(raw["pci_score"], errors="coerce")
    when = pd.to_datetime(raw["pci_change_date"], errors="coerce")
    kind = raw["treatment_or_survey"].astype(str).str.strip().str.lower()
    lat = pd.to_numeric(raw["latitude"], errors="coerce")
    lon = pd.to_numeric(raw["longitude"], errors="coerce")

    survey = (kind == "survey") & rating.between(0, 100) & when.notna() & lat.notna()
    out = pd.DataFrame(
        {
            "segment_id": raw.loc[survey, "cnn"].astype(str).to_numpy(),
            "created_at": when[survey].to_numpy(),
            "rating": (rating[survey] / 10.0).to_numpy(),
            "latitude": lat[survey].to_numpy(),
            "longitude": lon[survey].to_numpy(),
        }
    )
    out = add_previous_rating(out)

    treated = pd.DataFrame(
        {
            "segment_id": raw.loc[kind == "treatment", "cnn"].astype(str),
            "treated_at": when[kind == "treatment"],
        }
    ).dropna()
    out["repaired_since_prev"] = repaired_between(out, treated)
    out.loc[out["repaired_since_prev"], ["prev_rating", "prev_date"]] = np.nan
    logger.info(
        f"Kept {len(out)} surveys; {out['repaired_since_prev'].mean():.1%} follow "
        "a repair and lose their previous rating"
    )
    return out


def repaired_between(ratings: pd.DataFrame, treated: pd.DataFrame) -> pd.Series:
    """True where the segment has a treatment in (prev_date, created_at]."""
    rows = ratings[["segment_id", "created_at", "prev_date"]].reset_index()
    pairs = rows.merge(treated, on="segment_id", how="inner")
    hit = (pairs["treated_at"] > pairs["prev_date"]) & (
        pairs["treated_at"] <= pairs["created_at"]
    )
    flagged = set(pairs.loc[hit, "index"])
    return pd.Series(ratings.index.isin(flagged), index=ratings.index)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="San Francisco 311 and pavement data")
    parser.add_argument("--since", type=str, default="2016-07-01")
    parser.add_argument("--until", type=str, default="2026-01-01")
    parser.add_argument("--raw", type=str, default="data_raw/sf311_roads.csv")
    parser.add_argument(
        "--output", type=str, default="data_processed/sf311_roads.parquet"
    )
    parser.add_argument("--pavement_raw", type=str, default="data_raw/sf_pavement.csv")
    parser.add_argument(
        "--pavement_output",
        type=str,
        default="data_processed/sf_pavement_ratings.parquet",
    )
    parser.add_argument(
        "--skip_download", action="store_true", help="Use the CSVs already on disk"
    )
    args = parser.parse_args()
    if args.skip_download:
        raw311 = pd.read_csv(args.raw, dtype=str)
        raw_pav = pd.read_csv(args.pavement_raw, dtype=str)
    else:
        raw311 = download_sf311(args.raw, args.since, args.until)
        raw_pav = download_sf_pavement(args.pavement_raw)
    save_parquet(preprocess_sf311(raw311), args.output)
    save_parquet(preprocess_sf_pavement(raw_pav), args.pavement_output)

"""
Preprocess raw NBI (National Bridge Inventory) records into the internal bridge schema.

  - Maps NBI item columns (DECK_COND_058, ...) to internal names (deck_rating, ...).
  - Cleans condition ratings: 0-9 kept; "N" (not applicable) and blanks become
    missing, then filled with 9 so they never drive the worst-rating severity.
    The number of missing ratings is kept in `n_missing_ratings`, and bridges
    with no usable rating at all are flagged with `all_ratings_missing`.
  - Converts NBI DDMMSSss / DDDMMSSss coordinates to decimal degrees.
"""

import argparse
import logging

import pandas as pd

from civinlp.utils.io import load_parquet, save_parquet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NBI_COLUMN_MAPPING = {
    "STRUCTURE_NUMBER_008": "bridge_id",
    "DECK_COND_058": "deck_rating",
    "SUPERSTRUCTURE_COND_059": "superstructure_rating",
    "SUBSTRUCTURE_COND_060": "substructure_rating",
    "CULVERT_COND_062": "culvert_rating",
    "ADT_029": "traffic",
    "YEAR_ADT_030": "traffic_year",
    "YEAR_BUILT_027": "year_built",
    "STATE_CODE_001": "state_code",
    "COUNTY_CODE_003": "county_code",
    "LAT_016": "latitude",
    "LONG_017": "longitude",
}

RATING_COLUMNS = ["deck_rating", "superstructure_rating", "substructure_rating", "culvert_rating"]
MISSING_RATING_FILL = 9


def clean_rating(series: pd.Series) -> pd.Series:
    """Coerce NBI condition codes to floats in [0, 9]; anything else becomes NaN."""
    values = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    return values.where(values.between(0, 9))


def nbi_coord_to_decimal(series: pd.Series, degree_digits: int) -> pd.Series:
    """
    Convert NBI packed coordinates to decimal degrees.

    LAT_016 is DDMMSSss (degree_digits=2); LONG_017 is DDDMMSSss (degree_digits=3).
    Values that are missing or out of range become NaN.
    """
    s = series.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    width = degree_digits + 6
    s = s.where(s.str.fullmatch(r"\d+"), None).str.zfill(width)
    deg = pd.to_numeric(s.str[:degree_digits], errors="coerce")
    minutes = pd.to_numeric(s.str[degree_digits:degree_digits + 2], errors="coerce")
    seconds = pd.to_numeric(s.str[degree_digits + 2:], errors="coerce") / 100
    decimal = deg + minutes / 60 + seconds / 3600
    valid = (minutes < 60) & (seconds < 60) & (deg > 0)
    return decimal.where(valid)


def preprocess_bridges(input_path: str, output_path: str) -> pd.DataFrame:
    """Load raw NBI parquet, clean it, and save the processed bridge table."""
    logger.info(f"Loading raw NBI data from {input_path}")
    df = load_parquet(input_path)
    df = clean_bridges(df)
    save_parquet(df, output_path)
    logger.info(f"Saved {len(df)} processed bridges to {output_path}")
    return df


def clean_bridges(df: pd.DataFrame) -> pd.DataFrame:
    """In-memory version of preprocess_bridges (used by tests and the UI)."""
    df = df.rename(columns={k: v for k, v in NBI_COLUMN_MAPPING.items() if k in df.columns})

    if "bridge_id" not in df.columns:
        raise ValueError("NBI data needs STRUCTURE_NUMBER_008 (or bridge_id)")
    df["bridge_id"] = df["bridge_id"].astype(str).str.strip().str.strip("'")

    present = [c for c in RATING_COLUMNS if c in df.columns]
    for col in present:
        df[col] = clean_rating(df[col])
    if present:
        missing = df[present].isna()
        df["n_missing_ratings"] = missing.sum(axis=1).astype(int)
        df["all_ratings_missing"] = missing.all(axis=1)
        df[present] = df[present].fillna(MISSING_RATING_FILL)
    else:
        logger.warning("No condition rating columns found")
        df["n_missing_ratings"] = 0
        df["all_ratings_missing"] = True

    if "traffic" in df.columns:
        df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce").clip(lower=0)
    if "year_built" in df.columns:
        df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")

    # NBI longitudes are stored as positive degrees West
    if "latitude" in df.columns and not _already_decimal(df["latitude"]):
        df["latitude"] = nbi_coord_to_decimal(df["latitude"], 2)
    if "longitude" in df.columns and not _already_decimal(df["longitude"]):
        df["longitude"] = -nbi_coord_to_decimal(df["longitude"], 3)

    n_all_missing = int(df["all_ratings_missing"].sum())
    if n_all_missing:
        logger.warning(f"{n_all_missing} bridges have no usable condition rating")
    return df


def _already_decimal(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return len(values) > 0 and bool((values.abs() <= 180).all())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NBI Bridge Data")
    parser.add_argument("--input", type=str, default="data_raw/bridges_nbi.parquet")
    parser.add_argument("--output", type=str, default="data_processed/bridges_processed.parquet")
    args = parser.parse_args()
    preprocess_bridges(args.input, args.output)

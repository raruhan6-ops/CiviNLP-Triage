import argparse
import logging

from civinlp.data.schema import BridgeSchema
from civinlp.utils.io import load_csv, save_parquet

logger = logging.getLogger(__name__)


def load_nbi_data(input_csv: str, output_path: str):
    """
    Load NBI Element data from local CSV, validate, and save to Parquet.
    """
    logger.info(f"Loading NBI data from {input_csv}...")

    df = load_csv(
        input_csv, encoding="latin1", low_memory=False
    )  # NBI often uses latin1
    logger.info(f"Loaded {len(df)} rows.")

    missing_cols = BridgeSchema.validate_columns(df)
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")

    save_parquet(df, output_path)
    logger.info(f"Saved raw NBI data to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Load NBI Bridge Data")
    parser.add_argument("input_csv", type=str, help="Path to input NBI CSV file")
    parser.add_argument(
        "--output", type=str, default="data_raw/bridges_nbi.parquet", help="Output path"
    )

    args = parser.parse_args()
    load_nbi_data(args.input_csv, args.output)

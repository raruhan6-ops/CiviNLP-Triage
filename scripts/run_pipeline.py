"""
CiviNLP Triage — End-to-end pipeline runner.

Executes: data download → preprocessing → training → evaluation.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path


def run_step(command, description=""):
    print(f"\n{'='*60}")
    print(f">>> {description or command}")
    print(f"{'='*60}")
    subprocess.check_call(command, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Run the CiviNLP pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--transformer", action="store_true", help="Also train transformer model")
    parser.add_argument("--limit", type=int, default=1000, help="Limit rows for download")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    os.chdir(root)

    # Ensure directories
    for d in ["data_raw", "data_processed", "results", "models", "logs"]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  CiviNLP Triage — Full Pipeline")
    print("=" * 60)

    # ── Step 1: Data Ingestion ──
    if not args.skip_download:
        run_step(
            f"python -m civinlp.data.download_roads_311 --limit {args.limit} --output data_raw/roads_311.parquet",
            "Step 1: Downloading NYC 311 road data",
        )
    else:
        print("\n>>> Skipping data download (--skip-download)")

    # ── Step 2: Preprocessing ──
    run_step(
        "python -m civinlp.data.preprocess_roads --input data_raw/roads_311.parquet --output data_processed/roads_processed.parquet",
        "Step 2: Preprocessing road data",
    )

    # ── Step 3: Model Training ──
    if not args.skip_training:
        transformer_flag = "--transformer" if args.transformer else ""
        run_step(
            f"python scripts/train_models.py --input data_processed/roads_processed.parquet {transformer_flag}",
            "Step 3: Training models",
        )
    else:
        print("\n>>> Skipping training (--skip-training)")

    # ── Step 4: Evaluation ──
    run_step(
        "python scripts/evaluate_all.py --data data_processed/roads_processed.parquet --output_dir results/",
        "Step 4: Running evaluation & ablations",
    )

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {root / 'results'}")
    print(f"Models saved to:  {root / 'models'}")
    print(f"\nTo launch the dashboard:")
    print(f"  streamlit run src/civinlp/ui/streamlit_app.py")


if __name__ == "__main__":
    main()

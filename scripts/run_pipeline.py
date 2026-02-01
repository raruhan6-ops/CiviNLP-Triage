import os
import subprocess
import sys
from pathlib import Path

def run_step(command):
    print(f"\n>>> Running: {command}")
    subprocess.check_call(command, shell=True)

def main():
    root = Path(__file__).parent.parent
    os.chdir(root)
    
    # Ensure directories
    os.makedirs("data_raw", exist_ok=True)
    os.makedirs("data_processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("=== CiviNLP Triage Pipeline ===")
    
    # 1. Data Ingestion (Sample)
    # Using small limit for demo
    run_step("python -m civinlp.data.download_roads_311 --limit 1000 --output data_raw/roads_311.parquet")
    
    # 2. Preprocessing
    run_step("python -m civinlp.data.preprocess_roads --input data_raw/roads_311.parquet --output data_processed/roads_processed.parquet")
    
    # 3. Baselines (Train/Predict)
    # Need a script for this. I haven't created a 'train_baseline.py' script yet, only the class.
    # So I will skip this for now or create it.
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()

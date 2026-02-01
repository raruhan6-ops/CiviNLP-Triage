import argparse
import pandas as pd
from civinlp.eval.ablation import AblationRunner
from civinlp.eval.error_analysis import get_top_errors, generate_error_report
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_processed/roads_processed.parquet")
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Running Ablations ===")
    runner = AblationRunner(args.data)
    results = runner.run_baselines()
    print("\nResults:")
    print(results)
    results.to_csv(os.path.join(args.output_dir, "ablation_results.csv"), index=False)
    
    print("\n=== Generating Error Report (Sample) ===")
    # Simulate errors for demo
    dummy_texts = ["test1", "test2"]
    dummy_true = ["pothole", "pothole"]
    dummy_pred = ["pothole", "signage"] # 1 error
    dummy_conf = [0.9, 0.8]
    
    errors = get_top_errors(dummy_texts, dummy_true, dummy_pred, dummy_conf)
    print("Top Errors:")
    print(errors)
    generate_error_report(errors, os.path.join(args.output_dir, "error_analysis.csv"))
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

"""
Rolling-origin ranking evaluation (see civinlp.eval.rolling_eval).

    python scripts/rolling_eval.py --data data_processed/roads_processed_2y.parquet \
        --area grid --cell_m 500 --output_dir results/rolling/
"""

import argparse
import json
import logging
import os

import pandas as pd

from civinlp.eval.rolling_eval import grid_area_ids, run_rolling_evaluation
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.utils.logging_config import setup_logging
from civinlp.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Rolling-origin ranking evaluation")
    parser.add_argument(
        "--data", type=str, default="data_processed/roads_processed.parquet"
    )
    parser.add_argument("--output_dir", type=str, default="results/rolling/")
    parser.add_argument(
        "--area",
        choices=["zip", "grid"],
        default="grid",
        help="Ranking unit: zip code (area_id) or a lat/lon grid cell",
    )
    parser.add_argument("--cell_m", type=float, default=500.0, help="Grid cell size")
    parser.add_argument("--history_days", type=int, default=28)
    parser.add_argument("--horizon_days", type=int, default=28)
    parser.add_argument("--step_days", type=int, default=28)
    parser.add_argument(
        "--tune_frac",
        type=float,
        default=0.3,
        help="Share of the earliest folds used to tune weights and fit rankers",
    )
    parser.add_argument("--outcomes", nargs="+", default=["volume", "resolution_days"])
    parser.add_argument("--k", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    setup_logging(name="rolling_eval")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.data)
    df["predicted_severity"] = [
        str(s.value)
        for s in RuleBaseline().predict_severity(
            df["text"].fillna("").astype(str).tolist()
        )
    ]
    if args.area == "grid":
        df["area_id"] = grid_area_ids(df, args.cell_m)
    unit = "zip" if args.area == "zip" else f"grid{int(args.cell_m)}m"

    report = {"data": args.data, "rows": len(df), "unit": unit, "outcomes": {}}
    for outcome in args.outcomes:
        if outcome == "resolution_days" and "closed_date" not in df.columns:
            print("No closed_date column; skipping resolution_days.")
            continue
        per_fold, summary, info = run_rolling_evaluation(
            df,
            outcome=outcome,
            history_days=args.history_days,
            horizon_days=args.horizon_days,
            step_days=args.step_days,
            tune_frac=args.tune_frac,
            k_values=args.k,
            seed=args.seed,
        )
        tag = f"{outcome}_{unit}"
        per_fold.to_csv(
            os.path.join(args.output_dir, f"rolling_folds_{tag}.csv"), index=False
        )
        summary.to_csv(
            os.path.join(args.output_dir, f"rolling_summary_{tag}.csv"), index=False
        )
        report["outcomes"][outcome] = info

        print(f"\n=== {outcome} | {unit} | {len(info['test_folds'])} test folds ===")
        print(f"Tuned weights (on {len(info['tuning_folds'])} earlier folds): ", end="")
        print(info["tuned_weights"])
        view = summary[
            summary["metric"].isin([f"ndcg@{args.k[0]}", "spearman_vs_future"])
        ]
        print(view.round(3).to_string(index=False))

    with open(os.path.join(args.output_dir, f"rolling_info_{unit}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()

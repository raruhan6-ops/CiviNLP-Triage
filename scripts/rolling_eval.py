"""
Rolling-origin ranking evaluation (see civinlp.eval.rolling_eval).

    python scripts/rolling_eval.py --data data_processed/roads_processed_2y_res.parquet \
        --area grid --cell_m 500 --output_dir results/rolling/

With --from_folds, the summaries and paired comparisons are rebuilt from the
rolling_folds_*.csv files already in --output_dir, without rerunning the folds.
"""

import argparse
import json
import logging
import os

import pandas as pd

from civinlp.eval.outcome_eval import PAVEMENT_OUTCOMES
from civinlp.eval.rolling_eval import (
    BLOCK_SIZE,
    grid_area_ids,
    grid_reference_lat,
    nearest_area_ids,
    paired_comparisons,
    run_rolling_evaluation,
    summarize_folds,
)
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
    parser.add_argument(
        "--unit_name",
        type=str,
        help="Name of the area_id unit in output files (default 'zip'), e.g. "
        "'nbhd' when area_id holds neighborhoods",
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
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=["volume", "resolution_days", "confirmed_share"],
    )
    parser.add_argument("--k", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratings",
        type=str,
        help="Processed pavement ratings (needed for the pavement_* outcomes)",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=BLOCK_SIZE,
        help="Bootstrap block length in folds (about horizon_days / step_days)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for the run's info and comparisons files, e.g. _pavement",
    )
    parser.add_argument(
        "--from_folds",
        action="store_true",
        help="Rebuild summaries and comparisons from saved per-fold files",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    setup_logging(name="rolling_eval")
    os.makedirs(args.output_dir, exist_ok=True)
    unit = (
        (args.unit_name or "zip") if args.area == "zip" else f"grid{int(args.cell_m)}m"
    )

    if args.from_folds:
        rebuild_from_folds(
            args.output_dir, unit, args.outcomes, args.seed, args.block, args.tag
        )
        return

    df = pd.read_parquet(args.data)
    df["predicted_severity"] = [
        str(s.value)
        for s in RuleBaseline().predict_severity(
            df["text"].fillna("").astype(str).tolist()
        )
    ]
    if args.area == "grid":
        df["area_id"] = grid_area_ids(df, args.cell_m)

    ratings = None
    if PAVEMENT_OUTCOMES & set(args.outcomes):
        if not args.ratings:
            parser.error("pavement outcomes need --ratings")
        ratings = pd.read_parquet(args.ratings)
        if args.area == "grid":
            # Place ratings on the complaints' grid
            ratings["area_id"] = grid_area_ids(
                ratings, args.cell_m, ref_lat=grid_reference_lat(df)
            )
        else:
            # Ratings carry no area; take the nearest complaint's
            ratings["area_id"] = nearest_area_ids(ratings, df)
            logger.info(
                f"{(ratings['area_id'] == 'UNKNOWN').mean():.2%} of ratings "
                "have no complaint within 300 m"
            )

    report = {"data": args.data, "rows": len(df), "unit": unit, "outcomes": {}}
    comparisons = []
    for outcome in args.outcomes:
        needs = {
            "resolution_days": "closed_date",
            "confirmed_share": "resolution_description",
        }.get(outcome)
        if needs and needs not in df.columns:
            print(f"No {needs} column; skipping {outcome}.")
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
            outcome_df=ratings if outcome in PAVEMENT_OUTCOMES else None,
            block=args.block,
        )
        tag = f"{outcome}_{unit}"
        per_fold.to_csv(
            os.path.join(args.output_dir, f"rolling_folds_{tag}.csv"), index=False
        )
        summary.to_csv(
            os.path.join(args.output_dir, f"rolling_summary_{tag}.csv"), index=False
        )
        report["outcomes"][outcome] = info
        comparisons.append(
            paired_comparisons(per_fold, seed=args.seed, block=args.block).assign(
                outcome=outcome
            )
        )

        print(f"\n=== {outcome} | {unit} | {len(info['test_folds'])} test folds ===")
        print(f"Tuned weights (on {len(info['tuning_folds'])} earlier folds): ", end="")
        print(info["tuned_weights"])
        view = summary[
            summary["metric"].isin([f"ndcg@{args.k[0]}", "spearman_vs_future"])
        ]
        print(view.round(3).to_string(index=False))

    info_path = os.path.join(args.output_dir, f"rolling_info_{unit}{args.tag}.json")
    with open(info_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    save_comparisons(comparisons, args.output_dir, unit + args.tag)
    print(f"\nResults saved to {args.output_dir}")


def save_comparisons(comparisons, output_dir, unit):
    """Write the paired comparisons for all outcomes of one unit to one file."""
    if not comparisons:
        return
    table = pd.concat(comparisons, ignore_index=True)
    table.insert(0, "outcome", table.pop("outcome"))
    path = os.path.join(output_dir, f"rolling_comparisons_{unit}.csv")
    table.to_csv(path, index=False)
    print(f"\n=== paired comparisons | {unit} ===")
    print(table.drop(columns=["metric"]).round(3).to_string(index=False))


def rebuild_from_folds(output_dir, unit, outcomes, seed, block=BLOCK_SIZE, tag=""):
    """Recompute summaries and comparisons from saved rolling_folds_*.csv files."""
    comparisons = []
    for outcome in outcomes:
        name = f"{outcome}_{unit}"
        path = os.path.join(output_dir, f"rolling_folds_{name}.csv")
        if not os.path.exists(path):
            print(f"No {path}; skipping {outcome}.")
            continue
        per_fold = pd.read_csv(path)
        metrics = [
            c
            for c in per_fold.columns
            if c.startswith(("ndcg@", "recall@", "spearman"))
        ]
        summarize_folds(per_fold, metrics, seed=seed, block=block).to_csv(
            os.path.join(output_dir, f"rolling_summary_{name}.csv"), index=False
        )
        comparisons.append(
            paired_comparisons(per_fold, seed=seed, block=block).assign(outcome=outcome)
        )
    save_comparisons(comparisons, output_dir, unit + tag)


if __name__ == "__main__":
    main()

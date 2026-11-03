"""
Simultaneous intervals for every paired comparison in the rolling evaluation.

    python scripts/simultaneous_intervals.py --output_dir results/rolling/

Reads the rolling_folds_<outcome>_<unit>.csv files written by
scripts/rolling_eval.py, forms every paired difference in PAIRED_COMPARISONS
for every outcome and unit, and gives them sup-t moving-block bootstrap
intervals that hold for the whole family at once. The per-comparison
(unadjusted) intervals are kept beside them for reference.
"""

import argparse
import os

import pandas as pd

from civinlp.eval.rolling_eval import (
    PAIRED_COMPARISONS,
    bootstrap_ci,
    simultaneous_intervals,
)

OUTCOMES = ["volume", "resolution_days", "confirmed_share"]
UNITS = ["zip", "grid500m"]


def main():
    parser = argparse.ArgumentParser(description="Simultaneous comparison intervals")
    parser.add_argument("--output_dir", type=str, default="results/rolling/")
    parser.add_argument("--metric", type=str, default="spearman_vs_future")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    columns, keys = {}, []
    for outcome in OUTCOMES:
        for unit in UNITS:
            path = os.path.join(args.output_dir, f"rolling_folds_{outcome}_{unit}.csv")
            if not os.path.exists(path):
                print(f"No {path}; skipping.")
                continue
            wide = pd.read_csv(path).pivot(
                index="cutoff", columns="ranking", values=args.metric
            )
            for label, name, reference in PAIRED_COMPARISONS:
                if name in wide and reference in wide:
                    key = f"{outcome}|{unit}|{label}"
                    columns[key] = wide[name] - wide[reference]
                    keys.append((outcome, unit, label))

    # Cutoffs sort in time order; every file must cover the same folds
    diffs = pd.DataFrame(columns).sort_index()
    table, crit = simultaneous_intervals(diffs, seed=args.seed)
    table[["outcome", "unit", "comparison"]] = pd.DataFrame(keys)
    lows, highs = zip(
        *(bootstrap_ci(diffs[c].to_numpy(), seed=args.seed) for c in diffs.columns)
    )
    table["ci_low"], table["ci_high"] = lows, highs
    table["excludes_zero"] = (table["ci_low"] > 0) | (table["ci_high"] < 0)
    table["sim_excludes_zero"] = (table["sim_low"] > 0) | (table["sim_high"] < 0)
    table["family_size"] = len(table)
    table["crit"] = crit
    table = table[
        [
            "outcome",
            "unit",
            "comparison",
            "diff",
            "ci_low",
            "ci_high",
            "sim_low",
            "sim_high",
            "excludes_zero",
            "sim_excludes_zero",
            "se",
            "crit",
            "family_size",
        ]
    ]

    path = os.path.join(args.output_dir, "rolling_comparisons_simultaneous.csv")
    table.to_csv(path, index=False)
    print(
        f"{len(table)} comparisons over {len(diffs)} folds; critical value {crit:.2f}"
    )
    changed = table[table["excludes_zero"] != table["sim_excludes_zero"]]
    print("\nNo longer clear of zero after adjustment:")
    print(
        changed.drop(columns=["se", "crit", "family_size"])
        .round(3)
        .to_string(index=False)
    )
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()

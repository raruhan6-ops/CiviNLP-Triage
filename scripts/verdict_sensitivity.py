"""
Sensitivity of the confirmed-share results to the two ambiguous DOT templates.

    python scripts/verdict_sensitivity.py --data data_processed/roads_processed_2y_res.parquet \
        --output results/rolling/verdict_sensitivity.csv

Template 16 ("referred this complaint to the appropriate Maintenance Unit for
repair") counts as confirmed and template 22 ("found that the problem was
fixed") as no verdict (docs/verdict_mapping.csv). The rolling confirmed-share
evaluation is rerun with 16 as no verdict, and with 22 as confirmed or as not
confirmed, by zip code and by grid cell.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from civinlp.eval.outcome_eval import resolution_verdict
from civinlp.eval.rolling_eval import grid_area_ids, run_rolling_evaluation
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.utils.logging_config import setup_logging
from civinlp.utils.seed import set_global_seed

logger = logging.getLogger(__name__)

TEMPLATE_16 = "maintenance unit for repair"
TEMPLATE_22 = "found that the problem was fixed"
RANKINGS = [
    "count_only",
    "severity_shrunk",
    "learned_volume_type",
    "learned_volume_type_severity",
    "learned_volume_descriptor",
    "learned_volume_descriptor_severity",
]


def verdict_variants(descriptions: pd.Series) -> dict:
    """Main verdicts and the three reassignments of the ambiguous templates."""
    text = descriptions.fillna("").astype(str).str.lower()
    is_16 = text.str.contains(TEMPLATE_16, regex=False)
    is_22 = text.str.contains(TEMPLATE_22, regex=False)
    main = resolution_verdict(descriptions)
    return {
        "main": main,
        "16_no_verdict": main.mask(is_16, np.nan),
        "22_confirmed": main.mask(is_22, 1.0),
        "22_not_confirmed": main.mask(is_22, 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Verdict-rule sensitivity")
    parser.add_argument(
        "--data", type=str, default="data_processed/roads_processed_2y_res.parquet"
    )
    parser.add_argument(
        "--output", type=str, default="results/rolling/verdict_sensitivity.csv"
    )
    parser.add_argument("--cell_m", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    setup_logging(name="verdict_sensitivity")

    df = pd.read_parquet(args.data)
    df["predicted_severity"] = [
        str(s.value)
        for s in RuleBaseline().predict_severity(
            df["text"].fillna("").astype(str).tolist()
        )
    ]
    variants = verdict_variants(df["resolution_description"])
    units = {"zip": df["area_id"], "grid": grid_area_ids(df, args.cell_m)}

    rows = []
    for name, verdict in variants.items():
        for unit, areas in units.items():
            per_fold, _, _ = run_rolling_evaluation(
                df.assign(verdict=verdict, area_id=areas),
                outcome="confirmed_share",
                seed=args.seed,
            )
            wide = per_fold.pivot(
                index="fold", columns="ranking", values="spearman_vs_future"
            )
            row = {"variant": name, "unit": unit, "coverage": verdict.notna().mean()}
            row.update({r: wide[r].mean() for r in RANKINGS})
            row["sev_on_type"] = (
                wide["learned_volume_type_severity"] - wide["learned_volume_type"]
            ).mean()
            row["sev_on_desc"] = (
                wide["learned_volume_descriptor_severity"]
                - wide["learned_volume_descriptor"]
            ).mean()
            rows.append(row)
            logger.info(f"{name} | {unit} done")

    table = pd.DataFrame(rows).round(3)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

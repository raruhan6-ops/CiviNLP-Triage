"""
Inter-rater agreement for the DOT resolution-template verdicts.

The confirmed-share outcome rests on a template-to-verdict table
(docs/verdict_mapping.csv). A second person labels every template blind, and
this script reports how well the two agree.

    # 1. Write a blank, shuffled sheet with no verdicts on it
    python scripts/verdict_agreement.py make-sheet --sheet paper/verdict_second_rater.csv

    # 2. The second rater fills the `verdict` column (see docs/LABEL_GUIDE.md)

    # 3. Score the filled sheet against the published table
    python scripts/verdict_agreement.py score --sheet paper/verdict_second_rater.csv

Agreement is reported per template and weighted by complaints, since a few
templates cover most complaints.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

VERDICTS = ["confirmed", "not_confirmed", "no_verdict"]
MAPPING = "docs/verdict_mapping.csv"


def make_sheet(mapping: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Templates with text, shuffled, verdicts removed."""
    sheet = mapping[mapping["resolution_description"].notna()]
    sheet = sheet.sample(frac=1.0, random_state=seed)
    return sheet[["template_id", "resolution_description", "n"]].assign(verdict="")


def score(mapping: pd.DataFrame, sheet: pd.DataFrame) -> dict:
    """Cohen's kappa and raw agreement, per template and per complaint."""
    labels = sheet[["template_id", "verdict"]].rename(columns={"verdict": "second"})
    labels["second"] = labels["second"].astype(str).str.strip().str.lower()
    missing = labels[~labels["second"].isin(VERDICTS)]
    if len(missing):
        raise ValueError(
            f"{len(missing)} templates lack a verdict in {VERDICTS}, e.g. "
            f"template {missing['template_id'].iloc[0]}"
        )
    both = mapping.merge(labels, on="template_id", how="inner")
    first, second, weight = both["verdict"], both["second"], both["n"]
    agree = first == second
    return {
        "templates": int(len(both)),
        "complaints": int(weight.sum()),
        "agreement": float(agree.mean()),
        "kappa": float(cohen_kappa_score(first, second, labels=VERDICTS)),
        "agreement_by_complaints": float(np.average(agree, weights=weight)),
        "kappa_by_complaints": float(
            cohen_kappa_score(first, second, labels=VERDICTS, sample_weight=weight)
        ),
        "confusion": pd.DataFrame(
            confusion_matrix(first, second, labels=VERDICTS),
            index=[f"published:{v}" for v in VERDICTS],
            columns=[f"second:{v}" for v in VERDICTS],
        ).to_dict(),
        "disagreements": both.loc[
            ~agree, ["template_id", "n", "verdict", "second", "resolution_description"]
        ]
        .sort_values("n", ascending=False)
        .to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Verdict inter-rater agreement")
    parser.add_argument("action", choices=["make-sheet", "score"])
    parser.add_argument("--mapping", type=str, default=MAPPING)
    parser.add_argument("--sheet", type=str, default="paper/verdict_second_rater.csv")
    parser.add_argument("--output", type=str, default="results/verdict_agreement.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mapping = pd.read_csv(args.mapping)
    if args.action == "make-sheet":
        make_sheet(mapping, args.seed).to_csv(args.sheet, index=False)
        print(f"Wrote blank sheet to {args.sheet}")
        return

    result = score(mapping, pd.read_csv(args.sheet, keep_default_na=False))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"{result['templates']} templates ({result['complaints']:,} complaints): "
        f"agreement {result['agreement']:.1%}, kappa {result['kappa']:.3f}; "
        f"by complaints {result['agreement_by_complaints']:.1%}, "
        f"kappa {result['kappa_by_complaints']:.3f}"
    )
    print(f"{len(result['disagreements'])} disagreements; details in {args.output}")


if __name__ == "__main__":
    main()

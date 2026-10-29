import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from civinlp.eval.ablation import AblationRunner
from civinlp.eval.classification_metrics import compute_classification_report
from civinlp.eval.error_analysis import generate_error_report, get_top_errors
from civinlp.eval.outcome_eval import run_outcome_evaluation, weight_sensitivity
from civinlp.eval.review_routing import routing_curve
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.calibrate import calibration_report
from civinlp.models.tfidf_lr import TfidfBaseline
from civinlp.utils.logging_config import setup_logging
from civinlp.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def evaluate_against_gold(runner, tfidf, gold_path):
    """
    Compare rule and TF-IDF predictions with hand labels.

    The gold CSV needs a `record_id` column plus `defect` and/or `severity` columns.
    Only gold rows NOT in the TF-IDF training split are scored.

    Returns (results, routing): metrics per model, and the TF-IDF review-routing
    curve against the hand labels (None if there are no defect labels).
    """
    gold = pd.read_csv(gold_path, dtype={"record_id": str})
    if "record_id" not in gold.columns or "record_id" not in runner.df.columns:
        print("Gold file and data both need a 'record_id' column; skipping.")
        return None, None

    train_ids = set(runner.train_df["record_id"].astype(str))
    df = runner.df.assign(record_id=runner.df["record_id"].astype(str))
    merged = gold.merge(df[["record_id", "text"]], on="record_id", how="inner")
    merged = merged[~merged["record_id"].isin(train_ids)]
    if merged.empty:
        print("No gold rows outside the training split; skipping.")
        return None, None

    texts = merged["text"].fillna("").astype(str).tolist()
    rules = RuleBaseline()
    rows = []
    routing = None
    if "defect" in merged.columns:
        y_true = merged["defect"].astype(str).str.lower().tolist()
        conf = tfidf.predict_proba(texts).max(axis=1)
        routing = routing_curve(y_true, tfidf.predict(texts), conf)
        for name, preds in [
            ("RuleBaseline", [p.value for p in rules.predict(texts)]),
            ("TF-IDF (defect)", tfidf.predict(texts)),
        ]:
            rep = compute_classification_report(
                y_true, [str(p) for p in preds], critical_class=None
            )
            rows.append(
                {
                    "Task": "defect",
                    "Model": name,
                    "n": len(y_true),
                    "MacroF1": rep["macro_f1"],
                }
            )
    if "severity" in merged.columns:
        y_true = merged["severity"].astype(str).str.lower().tolist()
        preds = [s.value for s in rules.predict_severity(texts)]
        rep = compute_classification_report(y_true, preds, critical_class="high")
        rows.append(
            {
                "Task": "severity",
                "Model": "RuleBaseline",
                "n": len(y_true),
                "MacroF1": rep["macro_f1"],
                "CriticalRecall": rep["critical_recall"],
            }
        )
    return pd.DataFrame(rows), routing


def main():
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--data", default="data_processed/roads_processed.parquet")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--transformer_path", default="models/distilbert_finetuned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gold",
        default=None,
        help="Optional hand-labelled CSV (record_id, defect, severity)",
    )
    parser.add_argument(
        "--history_frac",
        type=float,
        default=0.5,
        help="Fraction of the time range used as ranking history",
    )
    parser.add_argument(
        "--split",
        choices=["time", "area"],
        default="time",
        help="Classification split: chronological, or held-out zones (20%% of areas)",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    setup_logging(name="evaluate")
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # 1. Classification Ablations
    # ============================================================
    print("=== Running Classification Ablations ===")
    runner = AblationRunner(args.data, split_method=args.split)

    # Run all models (including transformer if available)
    transformer_path = (
        args.transformer_path if Path(args.transformer_path).exists() else None
    )
    results = runner.run_all(transformer_path=transformer_path)
    print("\nAblation Results:")
    print(results.to_string(index=False))
    results.to_csv(os.path.join(args.output_dir, "ablation_results.csv"), index=False)

    # ============================================================
    # 2. Error Analysis (Real Predictions)
    # ============================================================
    print("\n=== Generating Error Analysis ===")
    test_texts = runner.test_df["text"].fillna("").astype(str).tolist()
    test_labels = [str(lab) for lab in runner.test_df["label"].tolist()]

    # Use TF-IDF for error analysis (most interesting errors)
    tfidf = TfidfBaseline(task="defect")
    train_texts = runner.train_df["text"].fillna("").astype(str).tolist()
    train_labels = [str(lab) for lab in runner.train_df["label"].tolist()]
    tfidf.train(train_texts, train_labels)

    y_pred_tfidf = tfidf.predict(test_texts)
    proba = tfidf.predict_proba(test_texts)
    confidences = [float(max(row)) for row in proba]

    errors = get_top_errors(test_texts, test_labels, y_pred_tfidf, confidences, k=20)
    print(f"Found {len(errors)} errors in top-20:")
    if len(errors) > 0:
        print(
            errors[["text", "y_true", "y_pred", "confidence"]]
            .head()
            .to_string(index=False)
        )
    generate_error_report(errors, os.path.join(args.output_dir, "error_analysis.csv"))

    # Calibration: temperature fitted on validation, ECE on test (vs weak labels)
    print("\n=== Calibration (TF-IDF defect) ===")
    calibration = calibration_report(
        tfidf.predict_proba(runner.val_df["text"].fillna("").astype(str).tolist()),
        [str(lab) for lab in runner.val_df["label"].tolist()],
        proba,
        test_labels,
        tfidf.classes_,
    )
    print(calibration)

    # Manual-review routing (RQ4): low-confidence predictions go to a reviewer
    print("\n=== Review routing (TF-IDF defect, vs weak labels) ===")
    routing = routing_curve(test_labels, y_pred_tfidf, confidences)
    print(routing.round(3).to_string(index=False))
    routing.to_csv(os.path.join(args.output_dir, "routing_weak.csv"), index=False)

    # ============================================================
    # 3. Gold-label Classification (independent of the rules)
    # ============================================================
    gold_results, gold_routing = None, None
    if args.gold:
        print("\n=== Evaluating against hand labels ===")
        gold_results, gold_routing = evaluate_against_gold(runner, tfidf, args.gold)
        if gold_results is not None:
            print(gold_results.to_string(index=False))
            gold_results.to_csv(
                os.path.join(args.output_dir, "gold_results.csv"), index=False
            )
        if gold_routing is not None:
            print("\nReview routing vs hand labels:")
            print(gold_routing.round(3).to_string(index=False))
            gold_routing.to_csv(
                os.path.join(args.output_dir, "routing_gold.csv"), index=False
            )

    # ============================================================
    # 4. Ranking Evaluation (future outcomes, not rule labels)
    # ============================================================
    print("\n=== Running Outcome-based Ranking Evaluation ===")
    df = runner.df.copy()
    rule_model = RuleBaseline()
    df["predicted_severity"] = [
        str(s.value)
        for s in rule_model.predict_severity(df["text"].fillna("").astype(str).tolist())
    ]

    ranking_evals = {}
    outcomes = ["volume"] + (["resolution_days"] if "closed_date" in df.columns else [])
    if "area_id" in df.columns and "created_at" in df.columns:
        for outcome in outcomes:
            suffix = "" if outcome == "volume" else f"_{outcome}"
            try:
                ranking_results, ranking_info = run_outcome_evaluation(
                    df, history_frac=args.history_frac, seed=args.seed, outcome=outcome
                )
            except ValueError as e:
                print(f"Skipping '{outcome}' ranking evaluation: {e}")
                continue
            print(f"\n--- Outcome: {outcome} ---")
            print(f"Ground truth: {ranking_info['relevance']}")
            print(
                f"Cutoff {ranking_info['cutoff']}, "
                f"{ranking_info['rankable_areas']} rankable areas, "
                f"{ranking_info['critical_areas']} critical"
            )
            print(ranking_results.round(3).to_string(index=False))
            ranking_results.to_csv(
                os.path.join(args.output_dir, f"ranking_results{suffix}.csv"),
                index=False,
            )

            # Severity weight trade-off (alpha fixed at 1). Report it; don't tune
            # beta to one outcome metric.
            sensitivity = weight_sensitivity(
                df, history_frac=args.history_frac, outcome=outcome
            )
            print("Priority-score sensitivity to weights:")
            print(sensitivity.round(3).to_string(index=False))
            sensitivity.to_csv(
                os.path.join(args.output_dir, f"ranking_sensitivity{suffix}.csv"),
                index=False,
            )
            ranking_evals[outcome] = {
                **ranking_info,
                "results": ranking_results.to_dict(orient="records"),
                "sensitivity": sensitivity.to_dict(orient="records"),
            }
    else:
        print("Need 'area_id' and 'created_at' columns; skipping ranking evaluation.")

    # ============================================================
    # 5. Summary JSON
    # ============================================================
    summary = {
        "seed": args.seed,
        "data_path": args.data,
        "train_size": len(runner.train_df),
        "val_size": len(runner.val_df),
        "test_size": len(runner.test_df),
        # Measured against rule-generated weak labels: shows agreement with the
        # rules, not real-world accuracy. See gold_results for that.
        "weak_label_agreement": results.to_dict(orient="records"),
        "gold_results": (
            gold_results.to_dict(orient="records") if gold_results is not None else None
        ),
        "split": args.split,
        "ranking_outcome_eval": ranking_evals,
        "calibration_tfidf_defect": calibration,
        "review_routing_weak": routing.to_dict(orient="records"),
        "review_routing_gold": (
            gold_routing.to_dict(orient="records") if gold_routing is not None else None
        ),
        "error_count": len(errors),
    }

    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== Evaluation complete. Results saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()

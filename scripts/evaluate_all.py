import argparse
import pandas as pd
import json
import os
import logging
from pathlib import Path

from civinlp.eval.ablation import AblationRunner
from civinlp.eval.error_analysis import get_top_errors, generate_error_report
from civinlp.eval.ranking_metrics import top_k_recall, ndcg_at_k, top_k_recall_curve
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.tfidf_lr import TfidfBaseline
from civinlp.ranking.road_priority import RoadPriorityScorer
from civinlp.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--data", default="data_processed/roads_processed.parquet")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--transformer_path", default="models/distilbert_finetuned")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # 1. Classification Ablations
    # ============================================================
    print("=== Running Classification Ablations ===")
    runner = AblationRunner(args.data)

    # Run all models (including transformer if available)
    transformer_path = args.transformer_path if Path(args.transformer_path).exists() else None
    results = runner.run_all(transformer_path=transformer_path)
    print("\nAblation Results:")
    print(results.to_string(index=False))
    results.to_csv(os.path.join(args.output_dir, "ablation_results.csv"), index=False)

    # ============================================================
    # 2. Error Analysis (Real Predictions)
    # ============================================================
    print("\n=== Generating Error Analysis ===")
    test_texts = runner.test_df['text'].fillna("").astype(str).tolist()
    test_labels = [str(l) for l in runner.test_df['label'].tolist()]

    # Use TF-IDF for error analysis (most interesting errors)
    tfidf = TfidfBaseline(task="defect")
    train_texts = runner.train_df['text'].fillna("").astype(str).tolist()
    train_labels = [str(l) for l in runner.train_df['label'].tolist()]
    tfidf.train(train_texts, train_labels)

    y_pred_tfidf = tfidf.predict(test_texts)
    proba = tfidf.predict_proba(test_texts)
    confidences = [float(max(row)) for row in proba]

    errors = get_top_errors(test_texts, test_labels, y_pred_tfidf, confidences, k=20)
    print(f"Found {len(errors)} errors in top-20:")
    if len(errors) > 0:
        print(errors[['text', 'y_true', 'y_pred', 'confidence']].head().to_string(index=False))
    generate_error_report(errors, os.path.join(args.output_dir, "error_analysis.csv"))

    # ============================================================
    # 3. Ranking Evaluation
    # ============================================================
    print("\n=== Running Ranking Evaluation ===")
    df = runner.test_df.copy()

    # Add predictions to test data for ranking
    df['predicted_defect'] = y_pred_tfidf
    rule_model = RuleBaseline()
    df['predicted_severity'] = [str(s) for s in rule_model.predict_severity(test_texts)]

    # Score and rank areas
    scorer = RoadPriorityScorer()
    if 'area_id' in df.columns:
        ranked = scorer.rank_areas(df)
        ranked_ids = ranked['area_id'].tolist()

        # Define "critical" areas (those with any HIGH severity complaint)
        critical_areas = set(
            df[df['predicted_severity'] == 'high']['area_id'].unique()
        ) if 'predicted_severity' in df.columns else set()

        if critical_areas and len(ranked_ids) > 0:
            k_values = [k for k in [3, 5, 10] if k <= len(ranked_ids)]
            recall_curve = top_k_recall_curve(ranked_ids, critical_areas, k_values=k_values)
            print(f"Top-K Recall: {recall_curve}")

            # NDCG (using priority scores as relevance)
            relevance = dict(zip(ranked['area_id'], ranked['priority_score']))
            for k in k_values:
                ndcg = ndcg_at_k(ranked_ids, relevance, k)
                print(f"NDCG@{k}: {ndcg:.4f}")
        else:
            recall_curve = {}
            print("No critical areas found for ranking evaluation.")
    else:
        recall_curve = {}
        print("No area_id column; skipping ranking evaluation.")

    # ============================================================
    # 4. Summary JSON
    # ============================================================
    summary = {
        "seed": args.seed,
        "data_path": args.data,
        "train_size": len(runner.train_df),
        "val_size": len(runner.val_df),
        "test_size": len(runner.test_df),
        "ablation_results": results.to_dict(orient="records"),
        "top_k_recall": {str(k): v for k, v in recall_curve.items()} if recall_curve else {},
        "error_count": len(errors),
    }

    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== Evaluation complete. Results saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()

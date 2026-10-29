import argparse

from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.tfidf_lr import TfidfBaseline
from civinlp.utils.io import load_parquet


def train_eval_baselines(data_path):
    print(f"Loading data from {data_path}")
    df = load_parquet(data_path)

    # Prepare data
    # Assume 'text' column and we need labels.
    # For baseline, we might check if 'weak_label' exists or generate it?
    # Spec says "Derive weak labels...". Rule baseline GENERATES labels.
    # TF-IDF needs labels to train. We use RuleBaseline to generate weak labels for TF-IDF training?
    # Or strict mapping.
    # Let's use RuleBaseline to generate 'y_train' for TF-IDF as a warm start or just demonstrate.

    texts = df["text"].astype(str).tolist()

    # 1. Rule Baseline
    print("Running Rule Baseline...")
    rule_model = RuleBaseline()
    # Pass metadata if we had it, for now just text
    preds_rule = rule_model.predict(texts)

    # 2. TF-IDF Baseline
    # We need labels to train. Let's use the rule predictions as "weak labels" to train TF-IDF
    # (Distillation from rules)
    print("Training TF-IDF Baseline on Rule Labels...")
    tfidf_model = TfidfBaseline()
    tfidf_model.train(texts, preds_rule)

    preds_tfidf = tfidf_model.predict(texts[:5])
    print("Top 5 TF-IDF predictions:", preds_tfidf)

    # Save model
    tfidf_model.save("models/tfidf_baseline.joblib")
    print("Saved TF-IDF model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_processed/roads_processed.parquet")
    args = parser.parse_args()

    train_eval_baselines(args.input)

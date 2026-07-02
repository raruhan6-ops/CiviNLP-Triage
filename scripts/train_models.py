"""
Train all CiviNLP models with proper splits and label mapping.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.tfidf_lr import TfidfBaseline
from civinlp.utils.io import load_parquet
from civinlp.utils.seed import set_global_seed
from civinlp.data.split import time_based_split
from civinlp.labels.taxonomy import RoadDefectClass


def train_eval_all(data_path: str, train_transformer: bool = False, seed: int = 42):
    set_global_seed(seed)
    print(f"Loading data from {data_path}")
    df = load_parquet(data_path)
    texts = df['text'].astype(str).tolist()

    # ── Generate weak labels ──
    print("Generating weak labels via Rule Baseline...")
    rule_model = RuleBaseline()
    preds_rule = [str(p) for p in rule_model.predict(texts)]
    severity_preds = [str(s) for s in rule_model.predict_severity(texts)]

    df['label'] = preds_rule
    df['severity_label'] = severity_preds

    # ── Split data ──
    print("Splitting data (70/10/20 time-based)...")
    try:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        train_df, val_df, test_df = time_based_split(df, time_column='created_at')
    except Exception as e:
        print(f"Time-based split failed ({e}), using sequential split.")
        n = len(df)
        t1, t2 = int(n * 0.7), int(n * 0.8)
        train_df, val_df, test_df = df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_texts = train_df['text'].astype(str).tolist()
    train_labels = train_df['label'].tolist()
    train_severity = train_df['severity_label'].tolist()

    # ── Build label map ──
    unique_labels = sorted(list(set(preds_rule)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    print(f"Label map: {label_map}")

    # ── TF-IDF (defect) ──
    print("\nTraining TF-IDF Baseline (defect)...")
    tfidf_defect = TfidfBaseline(task="defect")
    tfidf_defect.train(train_texts, train_labels)
    tfidf_defect.save("models/tfidf_baseline.joblib")
    print("Saved TF-IDF defect model.")

    # ── TF-IDF (severity) ──
    print("Training TF-IDF Baseline (severity)...")
    tfidf_severity = TfidfBaseline(task="severity")
    tfidf_severity.train(train_texts, train_severity)
    tfidf_severity.save("models/tfidf_severity.joblib")
    print("Saved TF-IDF severity model.")

    # Quick check
    sample_preds = tfidf_defect.predict(train_texts[:5])
    print(f"Top 5 TF-IDF defect predictions: {sample_preds}")

    # ── Transformer ──
    if train_transformer:
        print("\nTraining Transformer (this may take a while)...")
        from civinlp.models.transformer import TransformerClassifier

        train_indices = [label_map[l] for l in train_labels]
        val_texts = val_df['text'].astype(str).tolist()
        val_labels = val_df['label'].tolist()
        val_indices = [label_map.get(l, 0) for l in val_labels]

        clf = TransformerClassifier(num_labels=len(unique_labels))
        clf.set_label_map(label_map)

        clf.train(
            texts=train_texts,
            labels=train_indices,
            val_texts=val_texts if val_texts else None,
            val_labels=val_indices if val_texts else None,
            epochs=2,
        )
        clf.save("models/distilbert_finetuned")
        print("Saved Transformer model with label map.")

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_processed/roads_processed.parquet")
    parser.add_argument("--transformer", action="store_true", help="Train transformer model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_eval_all(args.input, args.transformer, args.seed)

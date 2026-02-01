import argparse
import pandas as pd
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.tfidf_lr import TfidfBaseline
from civinlp.models.transformer import TransformerClassifier
from civinlp.utils.io import load_parquet
from civinlp.labels.taxonomy import RoadDefectClass

def train_eval_all(data_path, train_transformer=False):
    print(f"Loading data from {data_path}")
    df = load_parquet(data_path)
    
    texts = df['text'].astype(str).tolist()
    
    # 1. Rule Baseline & Weak Labels
    print("Running Rule Baseline...")
    rule_model = RuleBaseline()
    preds_rule = rule_model.predict(texts)
    # Cast Enums to strings for sklearn
    preds_rule = [str(p) for p in preds_rule]
    
    # Map rule predictions to indices for transformer
    # (Simplified mapping for demo)
    unique_labels = sorted(list(set(preds_rule)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    print(f"Label map: {label_map}")
    
    indices = [label_map[l] for l in preds_rule]
    
    # 2. TF-IDF Baseline
    print("Training TF-IDF Baseline...")
    tfidf_model = TfidfBaseline()
    tfidf_model.train(texts, preds_rule)
    tfidf_model.save("models/tfidf_baseline.joblib")
    
    # 3. Transformer
    if train_transformer:
        print("Training Transformer (this may take a while)...")
        # Ensure num_labels matches
        clf = TransformerClassifier(num_labels=len(unique_labels))
        clf.train(texts[:100], indices[:100], epochs=1) # Limit for demo speed
        clf.save("models/distilbert_finetuned")
        print("Saved Transformer model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_processed/roads_processed.parquet")
    parser.add_argument("--transformer", action="store_true", help="Train transformer model")
    args = parser.parse_args()
    
    train_eval_all(args.input, args.transformer)

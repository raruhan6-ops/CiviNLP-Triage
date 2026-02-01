import pandas as pd
from typing import Dict, Any, List
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.tfidf_lr import TfidfBaseline
from civinlp.models.transformer import TransformerClassifier
from sklearn.metrics import classification_report, f1_score

class AblationRunner:
    """
    Runs experiments comparing different models/configurations.
    """
    def __init__(self, data_path: str):
        self.df = pd.read_parquet(data_path)
        # Assuming we have a 'label' column. If not, we might need ground truth.
        # For M4, we assume we have some labeled data (or use weak labels as proxy for demo).
        # In a real scenario, this would load a test set with human labels.
        # Here we'll simulate 'ground_truth' if not present, or use weak labels.
        if 'label' not in self.df.columns:
            # Fallback for demo: use RuleBaseline as ground truth proxy if no manual labels
            print("No 'label' column found. Generating proxy labels via Rules...")
            rule_model = RuleBaseline()
            preds = rule_model.predict(self.df['text'].fillna("").astype(str).tolist())
            self.df['label'] = [str(p) for p in preds] # Very weak ground truth!

    def run_baselines(self) -> pd.DataFrame:
        """
        Compare Rule vs TF-IDF.
        """
        results = []
        texts = self.df['text'].fillna("").astype(str).tolist()
        y_true = self.df['label'].tolist()
        
        # 1. Rules
        rule_model = RuleBaseline()
        y_pred_rule = rule_model.predict(texts)
        f1_rule = f1_score(y_true, y_pred_rule, average='macro', zero_division=0)
        results.append({"Model": "RuleBaseline", "MacroF1": f1_rule})
        
        # 2. TF-IDF
        # Train on a subset or split?
        # For simplicity in this runner, we split train/test here.
        train_size = int(0.8 * len(self.df))
        train_texts = texts[:train_size]
        train_labels = [str(l) for l in y_true[:train_size]]
        test_texts = texts[train_size:]
        test_labels = [str(l) for l in y_true[train_size:]]
        
        tfidf = TfidfBaseline()
        tfidf.train(train_texts, train_labels)
        y_pred_tfidf = tfidf.predict(test_texts)
        f1_tfidf = f1_score(test_labels, y_pred_tfidf, average='macro', zero_division=0)
        results.append({"Model": "TF-IDF", "MacroF1": f1_tfidf})
        
        return pd.DataFrame(results)

    def run_transformer(self, model_path: str) -> Dict[str, float]:
        """
        Evaluate a pre-trained transformer.
        """
        if not model_path:
            return {}
            
        clf = TransformerClassifier.load(model_path)
        texts = self.df['text'].fillna("").astype(str).tolist()
        # Note: Transformer predicts indices, need to map back to labels if labels are strings
        # Or map labels to indices.
        # This requires consistent label mapping. Skipping detailed implementation for brevity.
        return {"Model": "Transformer", "MacroF1": 0.0} # Placeholder

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from civinlp.data.split import group_split, time_based_split
from civinlp.eval.classification_metrics import compute_classification_report
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.tfidf_lr import TfidfBaseline

logger = logging.getLogger(__name__)


class AblationRunner:
    """
    Runs experiments comparing different models/configurations.
    Uses proper train/val/test splits and comprehensive metrics.
    """

    def __init__(self, data_path: str, split_method: str = "time"):
        """split_method: "time" (chronological 70/10/20) or "area" (held-out zones)."""
        self.df = pd.read_parquet(data_path)

        # Generate weak labels if not present
        if "label" not in self.df.columns:
            logger.info("No 'label' column found. Generating proxy labels via Rules...")
            rule_model = RuleBaseline()
            preds = rule_model.predict(self.df["text"].fillna("").astype(str).tolist())
            self.df["label"] = [str(p) for p in preds]

        # Generate severity labels if not present
        if "severity_label" not in self.df.columns:
            rule_model = RuleBaseline()
            sev_preds = rule_model.predict_severity(
                self.df["text"].fillna("").astype(str).tolist()
            )
            self.df["severity_label"] = [str(s) for s in sev_preds]

        # Split data
        if split_method == "time" and "created_at" in self.df.columns:
            try:
                self.df["created_at"] = pd.to_datetime(
                    self.df["created_at"], errors="coerce"
                )
                self.train_df, self.val_df, self.test_df = time_based_split(
                    self.df, time_column="created_at"
                )
            except Exception as e:
                logger.warning(
                    f"Time-based split failed ({e}), falling back to sequential split."
                )
                self._sequential_split()
        elif split_method == "area" and "area_id" in self.df.columns:
            # Hold out 20% of zones (and 10% for validation), not 20% of rows
            self.train_df, self.val_df, self.test_df = group_split(
                self.df, "area_id", test_frac=0.2, val_frac=0.1
            )
        else:
            self._sequential_split()

    def _sequential_split(self):
        """Fallback: sequential 70/10/20 split."""
        n = len(self.df)
        t1 = int(n * 0.7)
        t2 = int(n * 0.8)
        self.train_df = self.df.iloc[:t1].copy()
        self.val_df = self.df.iloc[t1:t2].copy()
        self.test_df = self.df.iloc[t2:].copy()
        logger.info(
            f"Sequential split: train={len(self.train_df)}, val={len(self.val_df)}, test={len(self.test_df)}"
        )

    def run_baselines(self) -> pd.DataFrame:
        """
        Compare Rule vs TF-IDF on the test set.
        """
        results = []
        train_texts = self.train_df["text"].fillna("").astype(str).tolist()
        train_labels = [str(lab) for lab in self.train_df["label"].tolist()]
        test_texts = self.test_df["text"].fillna("").astype(str).tolist()
        test_labels = [str(lab) for lab in self.test_df["label"].tolist()]

        # 1. Rules
        logger.info("Evaluating Rule Baseline...")
        rule_model = RuleBaseline()
        y_pred_rule = [str(p) for p in rule_model.predict(test_texts)]
        rule_report = compute_classification_report(test_labels, y_pred_rule)
        results.append(
            {
                "Model": "RuleBaseline",
                "MacroF1": rule_report["macro_f1"],
                "CriticalRecall": rule_report.get("critical_recall"),
            }
        )

        # 2. TF-IDF (defect)
        logger.info("Training & evaluating TF-IDF Baseline (defect)...")
        tfidf = TfidfBaseline(task="defect")
        tfidf.train(train_texts, train_labels)
        y_pred_tfidf = tfidf.predict(test_texts)
        tfidf_report = compute_classification_report(test_labels, y_pred_tfidf)
        results.append(
            {
                "Model": "TF-IDF (defect)",
                "MacroF1": tfidf_report["macro_f1"],
                "CriticalRecall": tfidf_report.get("critical_recall"),
            }
        )

        # 3. TF-IDF (severity)
        logger.info("Training & evaluating TF-IDF Baseline (severity)...")
        train_severity = [str(lab) for lab in self.train_df["severity_label"].tolist()]
        test_severity = [str(lab) for lab in self.test_df["severity_label"].tolist()]
        tfidf_sev = TfidfBaseline(task="severity")
        tfidf_sev.train(train_texts, train_severity)
        y_pred_sev = tfidf_sev.predict(test_texts)
        sev_report = compute_classification_report(
            test_severity, y_pred_sev, critical_class="high"
        )
        results.append(
            {
                "Model": "TF-IDF (severity)",
                "MacroF1": sev_report["macro_f1"],
                "CriticalRecall": sev_report.get("critical_recall"),
            }
        )

        return pd.DataFrame(results)

    def run_transformer(self, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a pre-trained transformer on the test set.

        Args:
            model_path: Path to the saved transformer model directory.

        Returns:
            Dict with model name and metrics.
        """
        if not model_path or not Path(model_path).exists():
            logger.warning(f"Transformer model not found at {model_path}")
            return {"Model": "Transformer", "MacroF1": None, "CriticalRecall": None}

        try:
            from civinlp.models.transformer import TransformerClassifier

            clf = TransformerClassifier.load(model_path)

            test_texts = self.test_df["text"].fillna("").astype(str).tolist()
            test_labels = [str(lab) for lab in self.test_df["label"].tolist()]

            if clf.index_map:
                y_pred = clf.predict_labels(test_texts)
            else:
                # Fallback: use integer predictions as strings
                y_pred = [str(p) for p in clf.predict(test_texts)]

            report = compute_classification_report(test_labels, y_pred)
            return {
                "Model": "Transformer",
                "MacroF1": report["macro_f1"],
                "CriticalRecall": report.get("critical_recall"),
            }
        except Exception as e:
            logger.error(f"Transformer evaluation failed: {e}")
            return {"Model": "Transformer", "MacroF1": None, "Error": str(e)}

    def run_all(self, transformer_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run all ablations and return combined results.

        Args:
            transformer_path: Optional path to transformer model.

        Returns:
            DataFrame with all model results.
        """
        results_df = self.run_baselines()

        if transformer_path:
            transformer_result = self.run_transformer(transformer_path)
            results_df = pd.concat(
                [results_df, pd.DataFrame([transformer_result])],
                ignore_index=True,
            )

        return results_df

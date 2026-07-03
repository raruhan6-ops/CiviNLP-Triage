import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class TransformerClassifier:
    """
    Wrapper around DistilBERT for text classification.

    Persists label_map alongside model for consistent label decoding.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 6):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.label_map: Optional[Dict[str, int]] = None  # label_str -> index
        self.index_map: Optional[Dict[int, str]] = None  # index -> label_str

    def set_label_map(self, label_map: Dict[str, int]):
        """
        Set the label mapping for encoding/decoding.

        Args:
            label_map: Dict mapping label strings to integer indices.
        """
        self.label_map = label_map
        self.index_map = {v: k for k, v in label_map.items()}

    def train(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = "results/checkpoints",
        epochs: int = 3,
    ):
        """
        Fine-tune the model.

        Args:
            texts: Training texts.
            labels: Training label indices.
            val_texts: Optional validation texts (for evaluation during training).
            val_labels: Optional validation label indices.
            output_dir: Directory for checkpoints.
            epochs: Number of training epochs.
        """
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128)
        
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = Dataset(encodings, labels)

        # Validation dataset
        eval_dataset = None
        eval_strategy = "no"
        if val_texts and val_labels:
            val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=128)
            eval_dataset = Dataset(val_encodings, val_labels)
            eval_strategy = "epoch"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            logging_dir=f'{output_dir}/logs',
            save_strategy="epoch",
            eval_strategy=eval_strategy,
            load_best_model_at_end=eval_dataset is not None,
            use_cpu=not torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        logger.info("Transformer training complete.")
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict class indices.
        """
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
        return preds.cpu().numpy().tolist()

    def predict_labels(self, texts: List[str]) -> List[str]:
        """
        Predict class labels as strings (using the stored label_map).

        Args:
            texts: Input texts.

        Returns:
            List of predicted label strings.

        Raises:
            ValueError: If label_map has not been set.
        """
        if self.index_map is None:
            raise ValueError("Label map not set. Call set_label_map() or load a model with saved label_map.")

        indices = self.predict(texts)
        return [self.index_map.get(i, f"UNKNOWN_{i}") for i in indices]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities.
        """
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            
        return probs.cpu().numpy()

    def predict_full(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict labels with confidence scores.

        Returns:
            List of dicts with keys: defect_class, confidence, probabilities.
        """
        probs = self.predict_proba(texts)
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        results = []
        for i, (pred_idx, conf) in enumerate(zip(predictions, confidences)):
            label = self.index_map.get(int(pred_idx), f"UNKNOWN_{pred_idx}") if self.index_map else str(pred_idx)
            results.append({
                "defect_class": label,
                "confidence": float(conf),
            })
        return results

    def save(self, path: Union[str, Path]):
        """Save model, tokenizer, and label_map to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save label map alongside model
        if self.label_map:
            label_map_path = path / "label_map.json"
            with open(label_map_path, "w") as f:
                json.dump(self.label_map, f, indent=2)
            logger.info(f"Saved label_map to {label_map_path}")

    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load model, tokenizer, and label_map from disk."""
        path = Path(path)
        instance = cls(model_name=str(path))

        # Load label map if it exists
        label_map_path = path / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r") as f:
                instance.set_label_map(json.load(f))
            logger.info(f"Loaded label_map from {label_map_path}")

        return instance

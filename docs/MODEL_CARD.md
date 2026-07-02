# Model Card — CiviNLP Triage

## Overview

CiviNLP Triage includes three model tiers for road defect classification, plus a bridge severity scorer.

---

## Model 1: Rule Baseline

| | |
|:---|:---|
| **Type** | Keyword-matching heuristic |
| **Training Data** | None (rule-based) |
| **Input** | Free text (complaint description) |
| **Output** | Defect class + severity |
| **Confidence** | Always 1.0 (deterministic) |

### Intended Use
- Baseline comparison for NLP models.
- Quick, interpretable predictions without training.

### Limitations
- No context understanding; relies on exact keyword presence.
- First-match-wins; multi-issue complaints may be mislabeled.
- Cannot generalize to new vocabulary.

---

## Model 2: TF-IDF + Logistic Regression

| | |
|:---|:---|
| **Type** | TF-IDF vectorizer + Logistic Regression classifier |
| **Training Data** | Weak labels from Rule Baseline |
| **Features** | Up to 5,000 TF-IDF features, English stop words removed |
| **Regularization** | Balanced class weights, max 1,000 iterations |
| **Input** | Free text |
| **Output** | Defect class + probabilities |

### Intended Use
- Strong baseline with probability calibration.
- Feature-based explainability (top TF-IDF features per class).
- Fast inference on CPU (~10k records in <60s).

### Limitations
- Trained on weak (rule-generated) labels, so ceiling is bounded by rule quality.
- Bag-of-words representation; loses word order and context.
- Performance may degrade on text significantly different from NYC 311 format.

### Metrics (Illustrative)
- Macro F1: ~1.0 on weak labels (expected, since TF-IDF learns the rule patterns).
- Real-world F1 would require human-annotated test data.

---

## Model 3: DistilBERT Transformer

| | |
|:---|:---|
| **Type** | Fine-tuned DistilBERT (66M parameters) |
| **Base Model** | `distilbert-base-uncased` |
| **Training Data** | Weak labels from Rule Baseline |
| **Max Length** | 128 tokens |
| **Batch Size** | 16 |
| **Input** | Free text |
| **Output** | Defect class + softmax probabilities |

### Intended Use
- Context-aware classification that can capture nuance beyond keywords.
- Best suited when fine-tuned on human-annotated data.

### Limitations
- Currently fine-tuned on weak labels only; context advantages not fully realized.
- Requires more compute than TF-IDF (GPU recommended for training).
- 268 MB model size.

### Known Biases
- Trained on NYC 311 data; may not generalize to other cities' complaint formats.
- Inherits biases from DistilBERT pre-training corpus.

---

## Bridge Severity Scorer

| | |
|:---|:---|
| **Type** | Deterministic formula |
| **Input** | NBI element condition ratings (0–9 scale) |
| **Output** | Severity score + priority score |

### Method
`severity = 9 - min(element_ratings)`
`priority = severity_weight × severity + traffic_weight × log(1 + ADT)`

### Limitations
- Uses worst-element rating; doesn't capture element-level nuance.
- Traffic weighting is log-scaled; very high traffic bridges may be underweighted.

---

## Responsible Use

> **⚠️ Decision-Support Only**: This tool supports maintenance triage and **does not replace engineering judgment**. All prioritized lists should be reviewed by qualified engineers before field action.

- Predictions are based on text analysis and structural ratings, not physical inspection.
- Model confidence should be used to flag uncertain cases for manual review.
- See [ETHICS_AND_LIMITATIONS.md](ETHICS_AND_LIMITATIONS.md) for full details.

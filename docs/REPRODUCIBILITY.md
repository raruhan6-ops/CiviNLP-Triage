# Reproducibility Guide

This document provides exact commands to reproduce all CiviNLP Triage results from a fresh clone.

---

## Prerequisites

- **Python**: 3.11+
- **OS**: Tested on Windows 11, Ubuntu 22.04, macOS 14
- **Hardware**: CPU sufficient for TF-IDF; GPU optional for transformer training
- **Internet**: Required for initial data download from NYC Open Data

---

## Setup

```bash
# Clone the repository
git clone https://github.com/raruhan6-ops/CiviNLP-Triage.git
cd CiviNLP-Triage

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e .[dev]
```

---

## Full Pipeline (Single Command)

```bash
python scripts/run_pipeline.py
```

This runs all steps: download → preprocess → train baselines → evaluate.

To also train the transformer:
```bash
python scripts/run_pipeline.py --transformer
```

---

## Step-by-Step Reproduction

### 1. Data Download

```bash
python -m civinlp.data.download_roads_311 --limit 2000 --output data_raw/roads_311.parquet
```

### 2. Preprocessing

```bash
python -m civinlp.data.preprocess_roads --input data_raw/roads_311.parquet --output data_processed/roads_processed.parquet
```

### 3. Model Training

```bash
# Baselines only
python scripts/train_models.py --input data_processed/roads_processed.parquet

# With transformer
python scripts/train_models.py --input data_processed/roads_processed.parquet --transformer
```

### 4. Evaluation

```bash
python scripts/evaluate_all.py --data data_processed/roads_processed.parquet --output_dir results/
```

### 5. Launch Dashboard

```bash
streamlit run src/civinlp/ui/streamlit_app.py
```

### 6. Launch API

```bash
uvicorn civinlp.api.main:app --reload
```

---

## Seeds and Determinism

| Setting | Value |
|:---|:---|
| Global seed | `42` |
| Python random | Set via `random.seed(42)` |
| NumPy | Set via `np.random.seed(42)` |
| PyTorch | Set via `torch.manual_seed(42)` |
| CUDA deterministic | `torch.backends.cudnn.deterministic = True` |
| Data split | Time-based: 70% train / 10% val / 20% test |

---

## Expected Outputs

After running the full pipeline, the following files should exist:

```
models/
  tfidf_baseline.joblib        # TF-IDF defect model
  tfidf_severity.joblib         # TF-IDF severity model
  distilbert_finetuned/         # Transformer model (if --transformer used)
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    label_map.json

results/
  ablation_results.csv          # Model comparison table
  error_analysis.csv            # Top misclassifications
  evaluation_summary.json       # Comprehensive results JSON
```

---

## Dependency Versions

Key pinned versions (see `pyproject.toml` for full list):

| Package | Version |
|:---|:---|
| Python | ≥ 3.11 |
| pandas | ≥ 2.0.0 |
| scikit-learn | ≥ 1.3.0 |
| transformers | ≥ 4.30.0 |
| torch | ≥ 2.0.0 |
| fastapi | ≥ 0.100.0 |
| streamlit | ≥ 1.24.0 |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Skip transformer tests (no GPU / no model download)
pytest tests/ -v --ignore=tests/test_advanced_models.py

# With coverage
pytest tests/ -v --cov=civinlp
```

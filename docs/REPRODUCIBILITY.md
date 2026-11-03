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

Two-year data used in the paper (298,193 complaints, 1 Nov 2024 to 1 Nov 2026):

```bash
python -m civinlp.data.download_roads_311 --monthly --since 2024-11-01 --until 2026-11-02 \
    --output data_raw/roads_311_2y_res.parquet
```

Only the seven road complaint types are requested, one calendar month at a time, with gzip transfers and retries. Each month is saved to `data_raw/roads_311_2y_res_months/` as soon as it completes, so an interrupted run resumes where it stopped. Request status, closing dates and resolution text change over time, so a later download can differ from ours.

The 20,000-row pilot sample instead takes the most recent requests, so it depends on the download date:

```bash
python -m civinlp.data.download_roads_311 --limit 20000 --output data_raw/roads_311.parquet
```

Bridges (optional; download a state's delimited NBI file from FHWA first):

```bash
python -m civinlp.data.download_nbi path/to/nbi.csv --output data_raw/bridges_nbi.parquet
python -m civinlp.data.preprocess_bridges --input data_raw/bridges_nbi.parquet --output data_processed/bridges_processed.parquet
```

### 2. Preprocessing

```bash
python -m civinlp.data.preprocess_roads --input data_raw/roads_311.parquet --output data_processed/roads_processed.parquet

# Two-year data for the paper
python -m civinlp.data.preprocess_roads --input data_raw/roads_311_2y_res.parquet --output data_processed/roads_processed_2y_res.parquet
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

# Options
#   --split area         hold out 20% of zones instead of the chronological split
#   --gold labels.csv    score against hand labels (record_id, defect, severity)
#   --history_frac 0.5   share of the time range used as ranking history
```

What is evaluated against what:

| Part | Ground truth |
|:---|:---|
| Classification F1 | Rule-generated weak labels: measures agreement with the rules only |
| Classification with `--gold` | Hand labels |
| Ranking | Future outcomes after a time cutoff: complaint volume per area, and median days to close |
| Review routing | Accuracy of auto-handled vs. routed predictions at each confidence threshold |

### 5. Rolling Out-of-Time Evaluation (Paper Tables 3 and 4)

```bash
python scripts/rolling_eval.py --data data_processed/roads_processed_2y_res.parquet --area zip --output_dir results/rolling/
python scripts/rolling_eval.py --data data_processed/roads_processed_2y_res.parquet --area grid --cell_m 500 --output_dir results/rolling/
```

Each run evaluates all three outcomes (future volume, resolution days, confirmed-defect share) with 28-day history and future windows advancing 28 days at a time. The first 30% of folds (8 of 25) tune the priority weights and fit the ridge rankers; the remaining 17 are test folds. Rule severity is recomputed from `text` on every run.

| File | Contents | Paper |
|:---|:---|:---|
| `rolling_folds_<outcome>_<unit>.csv` | Every metric for every ranker on every test fold | Source of all numbers |
| `rolling_summary_<outcome>_<unit>.csv` | Mean per ranker and metric, with the difference from count-only | Table 3 (`spearman_vs_future` rows, `mean` column) |
| `rolling_comparisons_<unit>.csv` | Paired per-fold differences between rankers (severity added to volume, type added to volume, and so on) | Table 4 and the differences quoted in Section 6 |
| `rolling_info_<unit>.json` | Fold cutoffs, tuned priority weights, chosen ridge penalties, areas ranked per fold | Section 4 penalties, "Tuned weights" in Section 6 |

`<unit>` is `zip` or `grid500m`. All intervals are 95% moving-block bootstrap intervals (blocks of 3 consecutive folds, 5,000 resamples, seed 42), because adjacent folds share data.

To recompute the summaries and comparisons from saved per-fold files without rerunning the folds (for example after changing the bootstrap):

```bash
python scripts/rolling_eval.py --from_folds --area zip --output_dir results/rolling/
python scripts/rolling_eval.py --from_folds --area grid --cell_m 500 --output_dir results/rolling/
```

The 42 paired comparisons (7 comparisons × 3 outcomes × 2 units) are also given simultaneous intervals, which hold for the whole family at once rather than one comparison at a time. They use the same moving-block bootstrap, resampling the same blocks of folds for every comparison:

```bash
python scripts/simultaneous_intervals.py --output_dir results/rolling/
```

This writes `rolling_comparisons_simultaneous.csv`, with the unadjusted and the simultaneous interval for each comparison side by side.

#### Pavement condition (independent outcome)

NYC DOT street pavement ratings (dataset `6yyb-pb25`, 1 to 10 per street segment) grade the same complaint-based rankings against measured road condition. Ratings have no zip code, so this runs by grid cell only, on the complaints' grid. Each segment is re-rated about every 14 months, so a 28-day window holds too few ratings; the future window is 91 days, the cutoff still moves 28 days at a time, and the bootstrap blocks are 4 folds long to cover the overlap.

```bash
python -m civinlp.data.pavement --raw data_raw/pavement_ratings.csv --output data_processed/pavement_ratings.parquet
python scripts/rolling_eval.py --data data_processed/roads_processed_2y_res.parquet --area grid --ratings data_processed/pavement_ratings.parquet --outcomes pavement_deficit pavement_poor_share --horizon_days 91 --block 4 --tag _pavement --output_dir results/rolling/
python scripts/simultaneous_intervals.py --outcomes pavement_deficit pavement_poor_share --units grid500m --block 4 --tag _pavement
```

Rating change, a check against standing differences between areas: `pavement_change` is rating points lost per year since each segment's previous inspection (gaps of 180 days or more), over all re-rated segments; `pavement_decline` drops segments whose rating rose (repair or rating noise). Outputs carry the `_pavement_change` tag.

```bash
python scripts/rolling_eval.py --data data_processed/roads_processed_2y_res.parquet --area grid --ratings data_processed/pavement_ratings.parquet --outcomes pavement_change pavement_decline --horizon_days 91 --block 4 --tag _pavement_change --output_dir results/rolling/
python scripts/simultaneous_intervals.py --outcomes pavement_change pavement_decline --units grid500m --block 4 --tag _pavement_change
```

`pavement_deficit` is the mean of 10 minus the rating over segments rated in the future window (higher is worse); `pavement_poor_share` is the share rated below 7. Outputs carry the `_pavement` tag: `rolling_info_grid500m_pavement.json`, `rolling_comparisons_grid500m_pavement.csv` and `rolling_comparisons_simultaneous_pavement.csv`.

By zip code, as a check that the grid does not drive the pavement results: with `--area zip`, each rating takes the zip code of the nearest geocoded complaint within 300 m (on the complaints themselves, the nearest other complaint has the same zip 99.3% of the time; 0.02% of ratings have none within 300 m). Give the zip runs their own simultaneous-interval tag, so they do not overwrite the grid files:

```bash
python scripts/rolling_eval.py --data data_processed/roads_processed_2y_res.parquet --area zip --ratings data_processed/pavement_ratings.parquet --outcomes pavement_deficit pavement_poor_share --horizon_days 91 --block 4 --tag _pavement --output_dir results/rolling/
python scripts/rolling_eval.py --data data_processed/roads_processed_2y_res.parquet --area zip --ratings data_processed/pavement_ratings.parquet --outcomes pavement_change pavement_decline --horizon_days 91 --block 4 --tag _pavement_change --output_dir results/rolling/
python scripts/simultaneous_intervals.py --outcomes pavement_deficit pavement_poor_share --units zip --block 4 --tag _pavement_zip
python scripts/simultaneous_intervals.py --outcomes pavement_change pavement_decline --units zip --block 4 --tag _pavement_change_zip
```

Agreement with a second, blind rater on the verdict table (see `docs/LABEL_GUIDE.md`):

```bash
python scripts/verdict_agreement.py make-sheet --sheet paper/verdict_second_rater.csv
python scripts/verdict_agreement.py score --sheet paper/verdict_second_rater.csv
```

The template-to-verdict table behind the confirmed-share outcome is `docs/verdict_mapping.csv`; the phrase rules that produce it are in `civinlp/eval/outcome_eval.py`.

Sensitivity to the two ambiguous templates ("Ambiguous templates" in Appendix C) reruns the confirmed-share evaluation with each reassigned, by zip code and by grid cell, and writes `results/rolling/verdict_sensitivity.csv`:

```bash
python scripts/verdict_sensitivity.py --data data_processed/roads_processed_2y_res.parquet --output results/rolling/verdict_sensitivity.csv
```

#### Second city: San Francisco

The same evaluation on DataSF data: SF311 street, sidewalk and curb defect requests (dataset `vw6y-z8j6`, 1 Jul 2016 to 31 Dec 2025) and Public Works pavement condition history (`78va-8dhi`). `civinlp.data.sf` maps SF subtypes onto the NYC complaint types (Street, Sidewalk or Curb Condition), keeps the subtype as the descriptor, and divides PCI (0 to 100) by 10, so the NYC outcomes and severity rules apply unchanged. A survey that follows a repair loses its previous rating, so rating-change outcomes skip it.

Public Works surveys nearly every block once a year, almost all between July and October, so folds are annual: a 365-day history, a 184-day future window starting about 1 July, one fold per year (9 folds: 3 tuning, 6 test, 2020 to 2025). SF311 has no zip code; areas are the 41 analysis neighborhoods (`--unit_name nbhd`) or 500 m grid cells. The confirmed-defect outcome is not run, because SF status notes are mostly generic ("Case Resolved").

```bash
python -m civinlp.data.sf --since 2016-07-01 --until 2026-01-01
python scripts/rolling_eval.py --data data_processed/sf311_roads.parquet --ratings data_processed/sf_pavement_ratings.parquet --history_days 365 --horizon_days 184 --step_days 365 --block 2 --output_dir results/rolling_sf/ --outcomes volume resolution_days pavement_deficit pavement_poor_share pavement_change pavement_decline --area zip --unit_name nbhd
python scripts/rolling_eval.py --data data_processed/sf311_roads.parquet --ratings data_processed/sf_pavement_ratings.parquet --history_days 365 --horizon_days 184 --step_days 365 --block 2 --output_dir results/rolling_sf/ --outcomes volume resolution_days pavement_deficit pavement_poor_share pavement_change pavement_decline --area grid --cell_m 500
python scripts/simultaneous_intervals.py --output_dir results/rolling_sf/ --outcomes volume resolution_days --units nbhd grid500m --block 2
python scripts/simultaneous_intervals.py --output_dir results/rolling_sf/ --outcomes pavement_deficit pavement_poor_share --units nbhd grid500m --block 2 --tag _pavement
python scripts/simultaneous_intervals.py --output_dir results/rolling_sf/ --outcomes pavement_change pavement_decline --units nbhd grid500m --block 2 --tag _pavement_change
```

### 6. Launch Dashboard

```bash
streamlit run src/civinlp/ui/streamlit_app.py
```

### 7. Launch API

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
| Data split | Time-based 70/10/20 (default) or `--split area`: 20% of zones test, 10% val |

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
  ablation_results.csv                      # Model agreement with weak labels
  error_analysis.csv                        # Top misclassifications
  routing_weak.csv                          # Review routing vs weak labels
  routing_gold.csv                          # Review routing vs hand labels (--gold)
  gold_results.csv                          # Hand-label metrics (--gold)
  ranking_results.csv                       # Ranking vs future complaint volume
  ranking_results_resolution_days.csv       # Ranking vs future resolution time
  ranking_sensitivity*.csv                  # Priority score across weight settings
  evaluation_summary.json                   # Everything above in one JSON
  rolling/                                  # Rolling evaluation (step 5), per unit and outcome
    rolling_folds_*.csv
    rolling_summary_*.csv
    rolling_comparisons_*.csv
    rolling_info_*.json
```

---

## Dependency Versions

Key pinned versions (see `pyproject.toml` for full list):

| Package | Version |
|:---|:---|
| Python | ≥ 3.11 |
| pandas | ≥ 2.0.0, < 3.0 |
| scikit-learn | ≥ 1.3.0 |
| transformers | ≥ 4.30.0 |
| torch | ≥ 2.0.0 |
| fastapi | ≥ 0.100.0 |
| streamlit | ≥ 1.46.0 |

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

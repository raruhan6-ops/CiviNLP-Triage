# CiviNLP Triage: Road + Bridge Maintenance Prioritization Tool

## Overview
CiviNLP Triage is an end-to-end NLP decision-support tool designed to convert public maintenance records (like NYC 311 complaints and NBI bridge data) into structured defect predictions and prioritized maintenance lists.

## Features
- **Data Ingestion**: Tools to download and process 311 service requests and NBI bridge data.
- **NLP Modeling**: Classifies free-text descriptions into defect types and severity levels using baselines and Transformer models.
- **Prioritization**: Generates ranked lists of maintenance hotspots based on severity, frequency, and optional uncertainty calibration.
- **Decision Support UI**: Interactive Streamlit dashboard for exploring predictions and rankings.

## Quickstart

### Installation
Requires Python 3.11+.

```bash
pip install .
```

### Running Tests
```bash
pytest
```

## Documentation
See `docs/` for detailed guides on data, models, and usage.

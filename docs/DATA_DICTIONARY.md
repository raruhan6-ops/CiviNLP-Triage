# Data Dictionary

This document describes all fields used in CiviNLP Triage, covering raw inputs, derived fields, and prediction outputs.

---

## Road Records (NYC 311)

### Raw Fields (from NYC Open Data)

| Field | Type | Source | Description |
|:---|:---|:---|:---|
| `unique_key` | string | NYC 311 | Unique complaint identifier |
| `created_date` | datetime | NYC 311 | When the complaint was filed |
| `closed_date` | datetime | NYC 311 | When the complaint was resolved (nullable) |
| `complaint_type` | string | NYC 311 | Category of complaint (e.g., "Pothole", "Street Condition") |
| `descriptor` | string | NYC 311 | Sub-category or additional detail (nullable) |
| `incident_zip` | string | NYC 311 | ZIP code of the incident location |
| `latitude` | float | NYC 311 | Latitude of incident (nullable) |
| `longitude` | float | NYC 311 | Longitude of incident (nullable) |
| `status` | string | NYC 311 | Complaint status (Open, Closed, etc.) |

### Internal / Derived Fields

| Field | Type | Derived From | Description |
|:---|:---|:---|:---|
| `record_id` | string | `unique_key` | Renamed primary key |
| `created_at` | datetime | `created_date` | Renamed timestamp |
| `text` | string | `complaint_type` + `descriptor` | Concatenated, cleaned, PII-redacted text input for models |
| `area_id` | string | `incident_zip` | Geographic area identifier (ZIP code or "UNKNOWN") |
| `label` | string | Rule Baseline | Weak defect label (e.g., "pothole", "cracking") |
| `severity_label` | string | Severity heuristic | Derived severity ("high", "medium", "low", "unknown") |
| `predicted_defect` | string | Model output | Model-predicted defect class |
| `predicted_severity` | string | Model/heuristic | Predicted severity level |
| `confidence` | float | Model output | Prediction confidence (0.0–1.0) |

---

## Bridge Records (NBI)

### Raw Fields (from FHWA National Bridge Inventory)

| Field | Type | Source | Description |
|:---|:---|:---|:---|
| `STRUCTURE_NUMBER_008` | string | NBI | Unique bridge identifier |
| `DECK_COND_058` | int (0–9) | NBI | Deck condition rating (9=Excellent, 0=Failed) |
| `SUPERSTRUCTURE_COND_059` | int (0–9) | NBI | Superstructure condition rating |
| `SUBSTRUCTURE_COND_060` | int (0–9) | NBI | Substructure condition rating |
| `ADT_029` | int | NBI | Average Daily Traffic count |
| `YEAR_BUILT_027` | int | NBI | Year the bridge was built |
| `STATE_CODE_001` | string | NBI | FIPS state code |
| `COUNTY_CODE_003` | string | NBI | FIPS county code |
| `LAT_016` | float | NBI | Latitude |
| `LONG_017` | float | NBI | Longitude |

### Derived Fields

| Field | Type | Description |
|:---|:---|:---|
| `severity` | float | `9.0 - min(rating columns)` — higher = worse condition |
| `traffic_factor` | float | `log(1 + ADT)` — importance weighting |
| `priority_score` | float | `severity_weight × severity + traffic_weight × traffic_factor` |

---

## Prediction Output Schema

| Field | Type | Description |
|:---|:---|:---|
| `record_id` | string | Source record identifier |
| `asset_type` | string | `"road"` or `"bridge"` |
| `predicted_defect` | string | Predicted defect class (roads only) |
| `predicted_severity` | string | `"low"`, `"medium"`, `"high"` |
| `confidence` | float | Model confidence (0.0–1.0) |
| `explanation` | list[string] | Top contributing features (optional) |
| `priority_score` | float | Computed priority score |
| `priority_rank` | int | Rank position (1 = highest priority) |

---

## Null Handling

- **Text fields**: `NaN` or missing values are replaced with empty string `""`.
- **ZIP codes**: Missing ZIPs are mapped to `"UNKNOWN"` area_id.
- **Dates**: Invalid or missing dates are coerced to `NaT` and excluded from time-based splits.
- **Ratings (NBI)**: Missing ratings are treated as 9 (Excellent) by default, meaning they don't inflate severity.

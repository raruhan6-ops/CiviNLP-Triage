# Data Dictionary

Fields used in CiviNLP Triage: raw inputs, processed tables and derived fields. All processed tables are Parquet files in `data_processed/` (gitignored).

---

## NYC 311 road complaints

Downloaded by `civinlp.data.download_roads_311` (dataset `erm2-nwe9`), restricted to the seven road complaint types: Street Condition, Highway Condition, Curb Condition, Sidewalk Condition, and Street Sign - Damaged, Missing or Dangling.

### Raw fields

| Field | Type | Description |
|:---|:---|:---|
| `unique_key` | string | Complaint identifier |
| `created_date` | datetime | When the complaint was filed |
| `closed_date` | datetime | When it was closed (nullable) |
| `status` | string | Open, Closed, etc. |
| `complaint_type` | string | One of the seven road types |
| `descriptor` | string | Sub-category, e.g. "Pothole" (nullable) |
| `resolution_description` | string | The agency's resolution text, mostly fixed templates |
| `incident_zip` | string | Zip code of the incident |
| `borough` | string | Borough |
| `latitude`, `longitude` | float | Incident location (nullable) |

### Processed table (`civinlp.data.preprocess_roads`)

| Field | Type | Description |
|:---|:---|:---|
| `record_id` | string | From `unique_key` |
| `created_at` | datetime | From `created_date` |
| `closed_date`, `status`, `complaint_type`, `descriptor`, `resolution_description`, `incident_zip`, `borough`, `latitude`, `longitude` | | As above |
| `text` | string | `complaint_type - descriptor`, cleaned and PII-redacted; the model input |
| `area_id` | string | Zip code normalised to 5 digits, or `UNKNOWN` |

### Added during training and evaluation

| Field | Type | Description |
|:---|:---|:---|
| `label` | string | Weak defect label from the keyword rules |
| `severity_label` / `predicted_severity` | string | Rule severity: `high`, `medium`, `low` or `unknown` |
| `predicted_defect`, `confidence` | string, float | Model prediction and confidence (0 to 1) |
| `area_id` (grid runs) | string | 500 m grid cell, `g<i>_<j>`, replacing the zip code when `--area grid` is used |

---

## Pavement ratings

### NYC DOT Street Pavement Ratings (`civinlp.data.pavement`, dataset `6yyb-pb25`)

Inspectors rate street segments from 1 (poor) to 10 (good). Rows with a non-rating reason (duplicate, construction and others) are dropped.

| Field | Type | Description |
|:---|:---|:---|
| `segment_id` | string | Street segment (`oftcode`) |
| `created_at` | datetime | Inspection date |
| `rating` | float | Rating, 1 to 10 |
| `latitude`, `longitude` | float | Mean of the segment's vertices |
| `borough`, `road_type` | string | As published |
| `prev_rating`, `prev_date` | float, datetime | The segment's rating on its last earlier inspection date; same-day ratings are averaged first |

Ratings carry no zip code. Grid runs place each at its midpoint on the complaints' grid; zip-code runs give it the zip code of the nearest geocoded complaint within 300 m.

### San Francisco Public Works pavement condition (`civinlp.data.sf`, dataset `78va-8dhi`)

The same fields as NYC (no `borough` or `road_type`), from survey records only:

| Field | Type | Description |
|:---|:---|:---|
| `rating` | float | Pavement Condition Index (0 to 100) divided by 10, so the NYC outcomes apply unchanged |
| `repaired_since_prev` | bool | A treatment (repair) record falls between the previous survey and this one; such rows have no `prev_rating`, so rating-change outcomes skip them |

---

## San Francisco 311 road defects (`civinlp.data.sf`, dataset `vw6y-z8j6`)

Requests under the service names Street Defect(s), Sidewalk or Curb, and Sidewalk and Curb, in the NYC processed schema:

| Field | Type | Description |
|:---|:---|:---|
| `record_id`, `created_at`, `closed_date`, `status` | | From the SF311 case fields |
| `complaint_type` | string | The subtype mapped onto a NYC type: Street, Sidewalk or Curb Condition |
| `descriptor` | string | The subtype, with its details when they add something (e.g. `Sidewalk Defect: Collapsed Sidewalk`) |
| `resolution_description` | string | SF311 status notes, mostly generic ("Case Resolved") |
| `latitude`, `longitude` | float | Request location; 0 becomes missing |
| `area_id` | string | Analysis neighbourhood (SF311 has no zip code), or `UNKNOWN` |
| `text` | string | `complaint_type - descriptor` |

---

## Evaluation outcomes and tables

| Outcome | Definition |
|:---|:---|
| `volume` | Road complaints per area in the future window |
| `resolution_days` | Median days to close an area's future complaints; open requests are censored at the window's last complaint; areas with fewer than three are excluded |
| `confirmed_share` | Share of an area's future complaints whose resolution template has the verdict `confirmed`, over those with a verdict |
| `pavement_deficit` | Mean of 10 minus the rating over segments rated in the future window |
| `pavement_poor_share` | Share of those segments rated below 7 |
| `pavement_change` | Rating points lost per year since each segment's previous inspection (gaps of 180 days or more) |
| `pavement_decline` | As `pavement_change`, excluding segments whose rating rose |

`docs/verdict_mapping.csv` holds the template-to-verdict table behind `confirmed_share`:

| Field | Description |
|:---|:---|
| `template_id` | Template number |
| `resolution_description` | The template text |
| `n` | Complaints with this template |
| `verdict` | `confirmed`, `not_confirmed` or `no_verdict` (definitions in `LABEL_GUIDE.md`) |

The files written by the rolling evaluation are listed in `REPRODUCIBILITY.md`.

---

## Bridges (FHWA National Bridge Inventory)

Used by the bridge scorer only, not by the evaluation.

### Raw fields

| Field | Type | Description |
|:---|:---|:---|
| `STRUCTURE_NUMBER_008` | string | Bridge identifier |
| `DECK_COND_058` | int (0 to 9) | Deck condition (9 excellent, 0 failed) |
| `SUPERSTRUCTURE_COND_059` | int (0 to 9) | Superstructure condition |
| `SUBSTRUCTURE_COND_060` | int (0 to 9) | Substructure condition |
| `ADT_029` | int | Average daily traffic |
| `YEAR_BUILT_027` | int | Year built |
| `STATE_CODE_001`, `COUNTY_CODE_003` | string | FIPS codes |
| `LAT_016`, `LONG_017` | float | Location |

### Derived fields

| Field | Type | Description |
|:---|:---|:---|
| `severity` | float | `9 - min(condition ratings)`; higher is worse |
| `traffic_factor` | float | `log(1 + ADT)` |
| `priority_score` | float | `severity_weight × severity + traffic_weight × traffic_factor` |
| `n_missing_ratings`, `all_ratings_missing` | int, bool | Missing or out-of-range ratings, which are filled with 9 so they never drive severity |

---

## Prediction output schema (API)

| Field | Type | Description |
|:---|:---|:---|
| `record_id` | string | Source record |
| `asset_type` | string | `road` or `bridge` |
| `predicted_defect` | string | Defect class (roads only) |
| `predicted_severity` | string | `low`, `medium` or `high` |
| `confidence` | float | Model confidence (0 to 1) |
| `explanation` | list[string] | Top contributing features (optional) |
| `priority_score`, `priority_rank` | float, int | Score and rank (1 = highest priority) |

---

## Missing values

- **Text:** missing values become an empty string.
- **Zip codes:** missing or invalid zip codes become `UNKNOWN`, which the evaluation excludes; so are complaints without coordinates in grid runs.
- **Dates:** invalid dates become `NaT` and are excluded from time-based splits and folds.

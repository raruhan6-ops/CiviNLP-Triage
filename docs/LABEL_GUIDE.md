# Label Guide

This document defines the defect taxonomy and severity levels used in CiviNLP Triage, along with the weak labeling methodology.

---

## Road Defect Taxonomy

Six mutually exclusive defect classes:

| Class | Enum Value | Description | Example Complaints |
|:---|:---|:---|:---|
| **Pothole** | `pothole` | Road surface hole or depression | "Large pothole on Main St", "Pothole causing flat tires" |
| **Cracking** | `cracking` | Surface cracks, rough or pitted roads | "Rough, pitted or cracked roads", "Road surface cracking" |
| **Street Condition** | `street_condition` | General road deterioration not fitting other categories | "Street condition complaint", "Poor road condition" |
| **Obstruction** | `obstruction` | Road blockage or debris | "Blocked driveway", "Road obstruction from construction" |
| **Signage** | `signage` | Missing, damaged, or obscured road signs | "Broken sign on 5th Ave", "Missing stop sign" |
| **Other** | `other` | Catch-all for unrecognized complaint types | "General complaint", text not matching any keywords |

---

## Severity Levels

Three severity levels plus an unknown category:

| Level | Enum Value | Description | Trigger Keywords |
|:---|:---|:---|:---|
| **High** | `high` | Immediate safety hazard or emergency | collapse, sinkhole, cave, hazard, danger, emergency, flood, major, severe, urgent, impassable, accident |
| **Medium** | `medium` | Significant damage requiring attention | crack, rough, damage, broken, large, deep, multiple, deteriorat, erosion, uneven, bump |
| **Low** | `low` | Minor issue with identifiable defect type | Any recognized defect class without severity keywords |
| **Unknown** | `unknown` | Cannot determine severity | Empty text or text not matching any defect or severity patterns |

---

## Weak Labeling Methodology

Since manually labeled training data is not available, CiviNLP Triage uses **weak supervision** to derive labels:

### Defect Labels

1. **Source**: NYC 311 `complaint_type` and `descriptor` fields.
2. **Method**: Keyword matching against the text (case-insensitive).
3. **Priority**: Keywords are checked in order: pothole → cracking → obstruction → signage → street condition → other.
4. **Limitation**: Labels are noisy — a complaint about a "pothole near a broken sign" will be labeled as "pothole" (first match wins).

### Severity Labels

1. **Source**: Combined text from `complaint_type`, `descriptor`, and any free-text fields.
2. **Method**: Keyword-based heuristic (spec option c: danger keywords).
3. **Priority**: HIGH keywords checked first, then MEDIUM, then LOW if a defect type is recognized.
4. **Limitation**: Keyword-based severity is a proxy; true severity requires field inspection data.

### Bridge Severity (NBI)

1. **Source**: NBI element condition ratings (0–9 scale).
2. **Method**: `severity = 9 - min(element_ratings)` — uses the worst condition rating.
3. **Rationale**: A bridge is as strong as its weakest component.

---

## Notes for Annotators

If manual labels are added in the future:

- Labels should follow the taxonomy above exactly.
- Multi-label scenarios (e.g., "pothole and broken sign"): assign the **primary** issue.
- Severity should reflect the **described** severity, not inferred urgency.
- Document any edge cases in a separate annotation log.

# Ethics and Limitations

## Purpose and Scope

CiviNLP Triage is a **decision-support prototype** designed for research and operational planning. It converts public text records into prioritized maintenance lists using NLP.

> **⚠️ This tool does NOT replace engineering judgment.** All outputs should be reviewed by qualified professionals before any field action is taken.

---

## Known Biases

### Reporting Bias
- **311 complaint data reflects who reports, not what exists.** Neighborhoods with higher civic engagement or internet access will generate more complaints, regardless of actual infrastructure condition.
- Areas with underreporting may appear as low-priority even if conditions are poor.
- This tool may **reinforce existing disparities** if used without awareness of reporting patterns.

### Geographic Bias
- The tool is developed and tested on **NYC 311 data**. Complaint formats, categories, and vocabulary differ across cities.
- Applying the tool to other cities' data without retraining may produce unreliable results.

### Language Bias
- Models are trained on English text only.
- Complaints written in other languages or non-standard English may be misclassified.
- DistilBERT pre-training corpus has known biases in representation.

### Temporal Bias
- Models trained on historical data may not reflect current conditions (e.g., after extreme weather events or policy changes).

---

## Privacy and Data Governance

### PII Handling
- The tool includes automatic PII redaction (emails, phone numbers, street addresses) before saving processed data. Personal names are not detected; the NYC 311 fields used (complaint type, descriptor) contain no free text, but any free-text source added later needs NER-based redaction.
- Raw text containing potential personal information is never stored without redaction.
- A configuration flag can disable text storage entirely, keeping only embeddings and predictions.

### Data Sources
- Only **publicly available** datasets are used (NYC Open Data, FHWA NBI).
- No private or proprietary data is required.

---

## Limitations

### Model Limitations
1. **Weak supervision**: Models are trained on keyword-derived labels, not human annotations. This creates a ceiling on achievable accuracy.
2. **Single-label classification**: Each complaint is assigned one defect class. Multi-issue complaints are labeled by the first matching keyword.
3. **Severity is approximate**: Keyword-based severity is a proxy; true severity requires physical inspection.
4. **Confidence is not calibrated by default**: Model confidence scores may not reflect true prediction reliability without temperature scaling.

### System Limitations
1. **No real-time data**: The tool processes batch data, not live feeds.
2. **No geospatial analysis**: Priority is based on area_id (ZIP code) aggregation, not precise location.
3. **Bridge data is structural**: NBI data reflects inspection ratings, not text-based analysis.

---

## Fairness Considerations

- **Equal treatment**: The tool applies the same algorithms to all areas. However, equal treatment does not guarantee equitable outcomes when reporting rates vary.
- **Recommendation**: Pair tool outputs with independent condition surveys to ensure underreported areas receive appropriate attention.
- **Transparency**: All model decisions can be explained via top features (TF-IDF) or attention patterns.

---

## Responsible Use Guidelines

1. **Always review outputs** before making resource allocation decisions.
2. **Cross-reference** with field inspection data when available.
3. **Monitor for drift**: Retrain models periodically as complaint patterns evolve.
4. **Document decisions**: Record when and how tool outputs influenced maintenance scheduling.
5. **Seek feedback**: Incorporate field crew feedback on prediction accuracy.

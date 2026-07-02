from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CiviNLP Triage API",
    description="Road & Bridge maintenance prioritization API",
    version="0.1.0",
)

# ─── Request / Response Schemas ───────────────────────────────────────────────

class RoadPredictionRequest(BaseModel):
    record_id: Optional[str] = None
    text: str
    metadata: Optional[dict] = None

class RoadPredictionResponse(BaseModel):
    record_id: Optional[str] = None
    defect_class: str
    severity: str
    confidence: float

class BridgePredictionRequest(BaseModel):
    bridge_id: str
    deck_rating: Optional[float] = None
    superstructure_rating: Optional[float] = None
    substructure_rating: Optional[float] = None
    traffic: Optional[float] = None

class BridgePredictionResponse(BaseModel):
    bridge_id: str
    severity_score: float
    priority_score: float

class RankingRequest(BaseModel):
    road_predictions: Optional[List[dict]] = None
    bridge_predictions: Optional[List[dict]] = None
    k: int = 10

class RankingResponse(BaseModel):
    road_rankings: Optional[List[dict]] = None
    bridge_rankings: Optional[List[dict]] = None


# ─── Model Loading ────────────────────────────────────────────────────────────

_models: Dict[str, Any] = {}

def _load_models():
    """Load models at startup (lazy, once)."""
    if _models:
        return

    # Try to load TF-IDF model
    tfidf_path = Path("models/tfidf_baseline.joblib")
    if tfidf_path.exists():
        from civinlp.models.tfidf_lr import TfidfBaseline
        _models["tfidf"] = TfidfBaseline.load(tfidf_path)
        logger.info(f"Loaded TF-IDF model from {tfidf_path}")

    # Always have rule baseline available
    from civinlp.models.baseline_rules import RuleBaseline
    _models["rules"] = RuleBaseline()
    logger.info("Rule baseline loaded")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "online", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.post("/predict/roads", response_model=List[RoadPredictionResponse])
def predict_roads(requests: List[RoadPredictionRequest]):
    """
    Batch prediction for road complaints.
    Uses TF-IDF model if available, falls back to rule baseline.
    """
    _load_models()

    texts = [r.text for r in requests]

    # Defect prediction
    model = _models.get("tfidf", _models.get("rules"))
    if model is None:
        raise HTTPException(status_code=503, detail="No model available")

    defect_preds = model.predict(texts)

    # Severity prediction (always via rules for now)
    rule_model = _models["rules"]
    severity_preds = rule_model.predict_severity(texts)

    # Confidence
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(texts)
            confidences = [float(max(row)) for row in proba]
        except Exception:
            confidences = [1.0] * len(texts)
    else:
        confidences = [1.0] * len(texts)

    responses = []
    for i, req in enumerate(requests):
        responses.append(RoadPredictionResponse(
            record_id=req.record_id,
            defect_class=str(defect_preds[i]),
            severity=str(severity_preds[i]),
            confidence=confidences[i],
        ))
    return responses


@app.post("/predict/bridges", response_model=List[BridgePredictionResponse])
def predict_bridges(requests: List[BridgePredictionRequest]):
    """
    Batch prediction for bridge severity.
    """
    from civinlp.ranking.bridge_priority import BridgePriorityScorer
    scorer = BridgePriorityScorer()

    responses = []
    for req in requests:
        record = req.model_dump()
        result = scorer.score_bridge(record)
        responses.append(BridgePredictionResponse(
            bridge_id=req.bridge_id,
            severity_score=result["severity"],
            priority_score=result["priority_score"],
        ))
    return responses


@app.post("/rank", response_model=RankingResponse)
def rank_assets(request: RankingRequest):
    """
    Rank road areas and/or bridges by priority.
    """
    import pandas as pd

    result = RankingResponse()

    if request.road_predictions:
        from civinlp.ranking.road_priority import RoadPriorityScorer
        scorer = RoadPriorityScorer()
        df = pd.DataFrame(request.road_predictions)
        if "area_id" in df.columns:
            ranked = scorer.rank_areas(df)
            result.road_rankings = ranked.head(request.k).to_dict(orient="records")

    if request.bridge_predictions:
        from civinlp.ranking.bridge_priority import BridgePriorityScorer
        scorer = BridgePriorityScorer()
        df = pd.DataFrame(request.bridge_predictions)
        ranked = scorer.rank_bridges(df)
        result.bridge_rankings = ranked.head(request.k).to_dict(orient="records")

    return result

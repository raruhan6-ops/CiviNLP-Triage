from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="CiviNLP Triage API")

class PredictionRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None

class PredictionResponse(BaseModel):
    defect_class: str
    severity: str
    confidence: float

@app.get("/")
def read_root():
    return {"status": "online", "version": "0.1.0"}

@app.post("/predict/roads", response_model=List[PredictionResponse])
def predict_roads(request: List[PredictionRequest]):
    """
    Batch prediction endpoint (placeholder).
    """
    # In real implementation: load model, run inference
    return [
        {"defect_class": "pothole", "severity": "medium", "confidence": 0.85}
        for _ in request
    ]

@app.get("/health")
def health_check():
    return {"status": "ok"}

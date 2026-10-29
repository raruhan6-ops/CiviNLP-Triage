import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import civinlp.api.main as api  # noqa: E402

client = TestClient(api.app)


def test_predict_roads_open_by_default(monkeypatch):
    monkeypatch.delenv("CIVINLP_API_KEY", raising=False)
    r = client.post("/predict/roads", json=[{"text": "Street Condition - Cave-in"}])
    assert r.status_code == 200
    assert r.json()[0]["severity"] == "high"


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("CIVINLP_API_KEY", "secret")
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health", headers={"X-API-Key": "secret"}).status_code == 200


def test_batch_cap(monkeypatch):
    monkeypatch.delenv("CIVINLP_API_KEY", raising=False)
    monkeypatch.setattr(api, "MAX_BATCH", 2)
    r = client.post("/predict/bridges", json=[{"bridge_id": str(i), "deck_rating": 5} for i in range(3)])
    assert r.status_code == 413
    r = client.post("/rank", json={"road_predictions": [{"area_id": "A"}] * 3})
    assert r.status_code == 413

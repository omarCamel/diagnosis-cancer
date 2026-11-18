import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app

SAMPLE = (
    pd.read_csv("data.csv")
    .dropna(axis=1, how="all")
    .drop(columns=["id", "diagnosis"])
    .iloc[0]
    .to_dict()
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTIONS_DB_PATH", str(tmp_path / "predictions.db"))
    monkeypatch.setenv("PREDICTIONS_HISTORY_API_KEY", "test-history-key")
    with TestClient(app) as test_client:
        yield test_client


def test_predict(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["diagnosis_probability"] <= 1
    assert body["risk_level"] in ("Bajo", "Medio", "Alto")
    assert any(keyword in body["message"].lower() for keyword in ("benigno", "maligno", "diagnóstico"))


def test_invalid_input(client):
    invalid = SAMPLE.copy()
    invalid["radius_mean"] = -1
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_dataset_summary(client):
    response = client.get("/dataset")
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] > 0
    assert body["columns"] > 0
    assert "diagnosis" in body["column_names"]


def test_prediction_is_persisted_and_history_requires_api_key(client, monkeypatch):
    class FixedModel:
        def predict(self, frame):
            return [1]

        def predict_proba(self, frame):
            return np.array([[0.1, 0.9]])

    monkeypatch.setattr(main_module, "model", FixedModel())
    prediction_response = client.post("/predict", json=SAMPLE)
    assert prediction_response.status_code == 200

    unauthorized_response = client.get("/predictions")
    assert unauthorized_response.status_code == 401

    history_response = client.get(
        "/predictions",
        headers={"X-History-API-Key": "test-history-key"},
    )
    assert history_response.status_code == 200
    history = history_response.json()
    assert len(history) == 1
    assert history[0]["input_features"] == {
        key: float(value) for key, value in SAMPLE.items()
    }
    assert history[0]["prediction"] == prediction_response.json()["prediction"]


def test_history_requires_configured_api_key(client, monkeypatch):
    monkeypatch.delenv("PREDICTIONS_HISTORY_API_KEY")

    response = client.get("/predictions")

    assert response.status_code == 503

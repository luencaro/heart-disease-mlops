"""Tests automaticos para la API de prediccion cardiaca."""

import sys
import os

# Añadir raiz del proyecto al path antes de importar app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402
from app.api import app  # noqa: E402

client = TestClient(app)

SAMPLE_FEATURES = [40, 140, 289, 0, 172, 0.0, 1, 1, 0, 0, 1, 0, 0, 0, 1]


def test_root_online():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_features_endpoint():
    response = client.get("/features")
    assert response.status_code == 200
    assert response.json()["n_features"] == 15
    assert len(response.json()["features"]) == 15


def test_predict_returns_200():
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert response.status_code == 200


def test_predict_output_schema():
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    data = response.json()
    assert "prediction" in data
    assert "heart_disease_probability" in data
    assert "risk_level" in data
    assert "message" in data


def test_predict_binary_output():
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert response.json()["prediction"] in [0, 1]


def test_predict_probability_range():
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    proba = response.json()["heart_disease_probability"]
    assert 0.0 <= proba <= 1.0


def test_predict_risk_level_valid():
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert response.json()["risk_level"] in ["Bajo", "Moderado", "Alto"]


def test_predict_wrong_feature_count():
    response = client.post("/predict", json={"features": [1, 2, 3]})
    assert response.status_code == 422


def test_predict_invalid_input():
    response = client.post("/predict", json={"wrong_field": [1, 2, 3]})
    assert response.status_code == 422

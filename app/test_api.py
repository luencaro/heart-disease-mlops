import pytest
import numpy as np
from fastapi.testclient import TestClient
import sys, os

# Asegurarse de que el path apunta a la raíz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.api import app

client = TestClient(app)

# ── Features de ejemplo (mismo orden que el entrenamiento) ───────────────────
SAMPLE_FEATURES = [40, 140, 289, 172, 0, 0, 1, 0, 0, 1, 0, 1]


# ── Tests de endpoints básicos ───────────────────────────────────────────────
def test_root_online():
    """El endpoint raíz debe responder con status online."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_check():
    """El health check debe retornar healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ── Tests del endpoint /predict ──────────────────────────────────────────────
def test_predict_returns_200():
    """Una predicción válida debe retornar HTTP 200."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert response.status_code == 200


def test_predict_output_schema():
    """La respuesta debe contener todos los campos del esquema."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    data = response.json()
    assert "prediction" in data
    assert "heart_disease_probability" in data
    assert "risk_level" in data
    assert "message" in data


def test_predict_binary_output():
    """La predicción debe ser 0 o 1."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    pred = response.json()["prediction"]
    assert pred in [0, 1]


def test_predict_probability_range():
    """La probabilidad debe estar entre 0 y 1."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    proba = response.json()["heart_disease_probability"]
    assert 0.0 <= proba <= 1.0


def test_predict_risk_level_valid():
    """El nivel de riesgo debe ser Bajo, Moderado o Alto."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    risk = response.json()["risk_level"]
    assert risk in ["Bajo", "Moderado", "Alto"]


def test_predict_invalid_input():
    """Un payload sin 'features' debe retornar error 422."""
    response = client.post("/predict", json={"wrong_field": [1, 2, 3]})
    assert response.status_code == 422

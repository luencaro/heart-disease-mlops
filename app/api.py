"""
Etapa 3: API REST con FastAPI para prediccion de falla cardiaca.
Archivo: app/api.py
Proyecto: Heart Disease MLOps - Dr. Lihki Rubio
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# Cargar modelo
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado desde: {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"No se encontro el modelo en: {MODEL_PATH}")


app = FastAPI(
    title="Heart Disease Prediction API",
    description="API REST para predecir el riesgo de falla cardiaca. MLOps - Dr. Lihki Rubio.",
    version="1.0.0",
)


class PatientInput(BaseModel):
    """15 features en el orden del entrenamiento tras One-Hot Encoding."""
    features: list = Field(
        ...,
        description="15 features en orden: Age, RestingBP, Cholesterol, FastingBS, MaxHR, "
                    "Oldpeak, Sex_M, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, "
                    "RestingECG_Normal, RestingECG_ST, ExerciseAngina_Y, ST_Slope_Flat, ST_Slope_Up",
        example=[40, 140, 289, 0, 172, 0.0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
    )


class PredictionOutput(BaseModel):
    prediction: int
    heart_disease_probability: float
    risk_level: str
    message: str


@app.get("/", summary="Estado de la API")
def root():
    """Verifica que la API esta en linea."""
    return {"status": "online", "model": MODEL_PATH, "features": 15, "docs": "/docs"}


@app.get("/health", summary="Health check")
def health():
    """Endpoint de salud para Kubernetes y Docker."""
    return {"status": "healthy"}


@app.get("/features", summary="Descripcion de las features")
def features():
    """Retorna el nombre y orden de las 15 features requeridas."""
    feature_list = [
        {"index": 0, "name": "Age", "type": "int", "description": "Edad del paciente"},
        {"index": 1, "name": "RestingBP", "type": "int", "description": "Presion arterial en reposo (mm Hg)"},
        {"index": 2, "name": "Cholesterol", "type": "int", "description": "Colesterol serico (mm/dl)"},
        {"index": 3, "name": "FastingBS", "type": "int", "description": "Glucosa en ayunas > 120 mg/dl (1=Si)"},
        {"index": 4, "name": "MaxHR", "type": "int", "description": "Frecuencia cardiaca maxima"},
        {"index": 5, "name": "Oldpeak", "type": "float", "description": "Depresion del segmento ST"},
        {"index": 6, "name": "Sex_M", "type": "int", "description": "Sexo masculino (1=M, 0=F)"},
        {"index": 7, "name": "ChestPainType_ATA", "type": "int", "description": "Dolor atipico (1=Si)"},
        {"index": 8, "name": "ChestPainType_NAP", "type": "int", "description": "Dolor no anginoso (1=Si)"},
        {"index": 9, "name": "ChestPainType_TA", "type": "int", "description": "Angina tipica (1=Si)"},
        {"index": 10, "name": "RestingECG_Normal", "type": "int", "description": "ECG normal (1=Si)"},
        {"index": 11, "name": "RestingECG_ST", "type": "int", "description": "ECG con anomalia ST (1=Si)"},
        {"index": 12, "name": "ExerciseAngina_Y", "type": "int", "description": "Angina por ejercicio (1=Si)"},
        {"index": 13, "name": "ST_Slope_Flat", "type": "int", "description": "Pendiente ST plana (1=Si)"},
        {"index": 14, "name": "ST_Slope_Up", "type": "int", "description": "Pendiente ST ascendente (1=Si)"},
    ]
    return {"n_features": 15, "features": feature_list}


@app.post("/predict", response_model=PredictionOutput, summary="Prediccion de riesgo cardiaco")
def predict(data: PatientInput):
    """Recibe 15 features y retorna prediccion, probabilidad y nivel de riesgo."""
    if len(data.features) != 15:
        raise HTTPException(
            status_code=422,
            detail=f"Se esperan 15 features, se recibieron {len(data.features)}. "
                   "Consulta GET /features para ver el orden correcto."
        )
    try:
        X = np.array(data.features, dtype=float).reshape(1, -1)
        proba = float(model.predict_proba(X)[0][1])
        pred = int(proba > 0.5)

        if proba < 0.3:
            risk = "Bajo"
            msg = "El paciente presenta bajo riesgo de enfermedad cardiaca."
        elif proba < 0.6:
            risk = "Moderado"
            msg = "El paciente presenta riesgo moderado. Se recomienda seguimiento medico."
        else:
            risk = "Alto"
            msg = "El paciente presenta alto riesgo. Se recomienda evaluacion medica urgente."

        return PredictionOutput(
            prediction=pred,
            heart_disease_probability=round(proba, 4),
            risk_level=risk,
            message=msg,
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error al procesar la prediccion: {str(e)}")

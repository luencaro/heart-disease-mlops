# Heart Disease MLOps 🫀

Proyecto integrador de Machine Learning con prácticas de MLOps para predecir falla cardíaca.

## Descripción

Modelo de clasificación binaria que predice si un paciente está en riesgo de sufrir falla cardíaca (`HeartDisease = 1 o 0`), usando el dataset *Heart Failure Prediction* de Kaggle.

## Estructura del Proyecto

```
heart-disease-mlops/
├── app/
│   └── api.py                       # API REST con FastAPI
├── docker/
│   ├── Dockerfile                   # Imagen del contenedor
│   └── requirements.txt             # Dependencias Python
├── k8s/
│   ├── deployment.yaml              # Despliegue en Kubernetes
│   └── service.yaml                 # Servicio LoadBalancer
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb   # EDA + detección de data leakage
│   └── 2_model_pipeline_cv.ipynb    # Modelado con Pipeline + GridSearchCV
├── tests/
│   └── __init__.py                  # Tests automáticos
├── .github/
│   └── workflows/
│       └── ci.yml                   # CI/CD con GitHub Actions
├── model.joblib                     # Modelo entrenado serializado
├── drift_report.html                # Reporte de data drift (Evidently)
└── README.md
```
## Dataset

- **Fuente:** [Heart Failure Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Target:** `HeartDisease` (0 = sin riesgo, 1 = en riesgo)

## Autor

Luis Cabarcas, Luis Cantillo y Natalia Frias.
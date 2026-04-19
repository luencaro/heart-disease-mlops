# ── Etapa 3: Dockerfile ───────────────────────────────────────────────────────
# scikit-learn 1.8.0 requiere Python >= 3.11

FROM python:3.11-slim

LABEL maintainer="Heart Disease MLOps"
LABEL version="1.0.0"

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY model.joblib ./model.joblib

ENV MODEL_PATH=/app/model.joblib

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

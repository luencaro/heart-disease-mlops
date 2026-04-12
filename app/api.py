from fastapi import FastAPI

app = FastAPI(title="Heart Disease MLOps API")


@app.get("/health")
def health_check():
    return {"status": "ok"}

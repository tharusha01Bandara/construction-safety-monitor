from fastapi import FastAPI
from .routes.predict import router as predict_router
import os
from .config import settings

app = FastAPI(title="Construction Safety Monitor", version="1.0.0")

app.include_router(predict_router)

@app.on_event("startup")
def init_directories():
    if not os.path.exists(settings.OUTPUT_DIR):
        os.makedirs(settings.OUTPUT_DIR)
    if not os.path.exists("models"):
        os.makedirs("models")

@app.get("/health")
def health_check():
    return {"status": "ok"}

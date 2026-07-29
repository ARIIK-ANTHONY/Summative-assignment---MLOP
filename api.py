"""
api.py

FastAPI service for the African Wildlife classifier: single-image
prediction, bulk data upload for retraining, and a retraining trigger.
Reuses the exact same preprocessing/model code as the training notebook
so training and serving stay in sync. Interactive docs at /docs.
"""
import gc
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.model import FINAL_MODEL_PATH, train
from src.prediction import predict_image, reload_model
from src.preprocessing import CLASS_NAMES, save_uploaded_images_for_retraining

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "retrain_uploads")

START_TIME = time.time()

retrain_state = {
    "status": "idle",  # idle | running | done | failed
    "message": None,
    "started_at": None,
    "finished_at": None,
}

app = FastAPI(title="African Wildlife Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "server_time": datetime.now(timezone.utc).isoformat(),
        "classes": CLASS_NAMES,
    }


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        return predict_image(file.file)
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")


@app.post("/upload")
async def upload(class_label: str = Form(...), files: list[UploadFile] = File(...)):
    if class_label not in CLASS_NAMES:
        raise HTTPException(400, f"class_label must be one of {CLASS_NAMES}")
    saved = save_uploaded_images_for_retraining(
        [(f.filename, f.file) for f in files], class_label, UPLOAD_DIR
    )
    return {"saved": len(saved), "class_label": class_label, "paths": saved}


def _run_retraining(epochs: int):
    retrain_state.update(
        status="running", message=None,
        started_at=datetime.now(timezone.utc).isoformat(), finished_at=None,
    )
    try:
        # Free the cached inference model before loading a fresh copy for
        # training - on memory-constrained deployments, holding both the
        # serving model and the training model at once risks an OOM crash.
        reload_model()
        gc.collect()

        if os.path.isdir(UPLOAD_DIR):
            for class_name in os.listdir(UPLOAD_DIR):
                src_dir = os.path.join(UPLOAD_DIR, class_name)
                if not os.path.isdir(src_dir):
                    continue
                dest_dir = os.path.join(TRAIN_DIR, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                for fname in os.listdir(src_dir):
                    shutil.copy2(os.path.join(src_dir, fname), dest_dir)

        train(
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR,
            epochs=epochs,
            initial_model_path=FINAL_MODEL_PATH,
        )
        reload_model()
        retrain_state.update(status="done", finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        retrain_state.update(
            status="failed", message=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


@app.post("/retrain")
def retrain(epochs: int = 5):
    if retrain_state["status"] == "running":
        raise HTTPException(409, "Retraining already in progress")
    thread = threading.Thread(target=_run_retraining, args=(epochs,), daemon=True)
    thread.start()
    return {"message": "Retraining started", "epochs": epochs}


@app.get("/retrain/status")
def retrain_status():
    return retrain_state

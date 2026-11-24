"""
PhytoGuard AI - FastAPI service.

Run locally:
    uvicorn src.api:app --reload

Interactive docs:
    http://127.0.0.1:8000/docs
"""

import io
import json
import os
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    METRICS_PATH, RETRAIN_DIR, RETRAIN_LOG_PATH, RETRAIN_THRESHOLD, TEST_DIR,
    TRAIN_DIR, UPLOAD_DIR, VALID_EXTENSIONS,
)
from src.prediction import Predictor
from src.preprocessing import decode_image_bytes
from src.retrain import count_images, run_retraining, should_retrain

AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "true").lower() == "true"
MAX_UPLOAD_MB = 10

app = FastAPI(
    title="PhytoGuard AI API",
    description="Plant disease detection from leaf images, with bulk upload and retraining.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

START_TIME = time.time()
predictor = Predictor()

request_stats = {"predictions": 0, "total_latency_ms": 0.0}
stats_lock = threading.Lock()

retrain_state = {
    "status": "idle",          # idle | running | completed | failed
    "trigger": None,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}
retrain_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _class_counts(folder):
    counts = {}
    if os.path.isdir(folder):
        for class_name in sorted(os.listdir(folder)):
            class_dir = os.path.join(folder, class_name)
            if os.path.isdir(class_dir):
                counts[class_name] = sum(
                    1 for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)
                )
    return counts


def _pending_uploads():
    return count_images(UPLOAD_DIR, predictor.class_names)


# ---------------------------------------------------------------------------
# Retraining in the background
# ---------------------------------------------------------------------------

def _retrain_worker(manual):
    try:
        result = run_retraining(new_data_dir=UPLOAD_DIR, manual_trigger=manual)
        if result.get("promoted"):
            predictor.reload()
        retrain_state.update(status="completed", result=result, finished_at=_now())
    except Exception as error:  # keep the API alive whatever happens in training
        traceback.print_exc()
        retrain_state.update(status="failed", error=str(error), finished_at=_now())


def start_retraining(manual, trigger_label):
    """Start retraining in a background thread. Returns False if one is already running."""
    with retrain_lock:
        if retrain_state["status"] == "running":
            return False
        retrain_state.update(
            status="running", trigger=trigger_label, started_at=_now(),
            finished_at=None, result=None, error=None,
        )
    threading.Thread(target=_retrain_worker, args=(manual,), daemon=True).start()
    return True


def _maybe_auto_retrain():
    """Automatic trigger: start retraining once pending uploads reach the threshold."""
    pending = _pending_uploads()
    start, reason = should_retrain(pending)
    if AUTO_RETRAIN and start:
        started = start_retraining(manual=False, trigger_label=f"automatic: {reason}")
        return {"auto_retrain_started": started, "reason": reason}
    return {"auto_retrain_started": False, "reason": reason}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "PhytoGuard AI API",
        "docs": "/docs",
        "endpoints": [
            "GET /health", "GET /metrics", "GET /classes", "GET /dataset/stats",
            "POST /predict", "POST /upload", "POST /upload/zip",
            "POST /retrain", "GET /retrain/status",
        ],
    }


@app.get("/health")
def health():
    """Model up-time and readiness."""
    uptime = time.time() - START_TIME
    metrics = _read_json(METRICS_PATH, {})
    return {
        "status": "ok",
        "model_loaded": predictor.model is not None,
        "model_version": metrics.get("model_version", "unknown"),
        "started_at": datetime.fromtimestamp(START_TIME, timezone.utc).isoformat(timespec="seconds"),
        "uptime_seconds": round(uptime, 1),
        "uptime_readable": time.strftime("%H:%M:%S", time.gmtime(uptime)) if uptime < 86400
        else f"{int(uptime // 86400)}d " + time.strftime("%H:%M:%S", time.gmtime(uptime % 86400)),
        "retraining": retrain_state["status"],
        "container_id": os.environ.get("HOSTNAME", "local"),
    }


@app.get("/metrics")
def metrics():
    """Model evaluation metrics, retraining history and live request statistics."""
    with stats_lock:
        count = request_stats["predictions"]
        avg_latency = request_stats["total_latency_ms"] / count if count else 0.0
    return {
        "model": _read_json(METRICS_PATH, {}),
        "retrain_history": _read_json(RETRAIN_LOG_PATH, []),
        "requests": {
            "predictions_served": count,
            "average_latency_ms": round(avg_latency, 1),
        },
    }


@app.get("/classes")
def classes():
    return {"count": len(predictor.class_names), "classes": predictor.class_names}


@app.get("/dataset/stats")
def dataset_stats():
    """Image counts per class in each data folder (used by the UI visualisations)."""
    return {
        "train": _class_counts(TRAIN_DIR),
        "test": _class_counts(TEST_DIR),
        "retrain_sample": _class_counts(RETRAIN_DIR),
        "uploads_pending": _class_counts(UPLOAD_DIR),
        "retrain_threshold": RETRAIN_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the disease class of one leaf image (JPEG or PNG)."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_UPLOAD_MB} MB.")

    start = time.perf_counter()
    try:
        result = predictor.predict_bytes(image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    latency_ms = (time.perf_counter() - start) * 1000

    with stats_lock:
        request_stats["predictions"] += 1
        request_stats["total_latency_ms"] += latency_ms

    return {"filename": file.filename, **result, "latency_ms": round(latency_ms, 1)}


def _save_image(image_bytes, class_name, original_name):
    """Validate an image and store it in data/uploads/<class_name>/."""
    decode_image_bytes(image_bytes)  # raises ValueError if not a real image
    ext = os.path.splitext(original_name or "")[1].lower()
    if ext not in VALID_EXTENSIONS:
        ext = ".jpg"
    class_dir = os.path.join(UPLOAD_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    with open(os.path.join(class_dir, f"upload_{uuid.uuid4().hex[:12]}{ext}"), "wb") as f:
        f.write(image_bytes)


@app.post("/upload")
async def upload_images(class_name: str = Form(...), files: List[UploadFile] = File(...)):
    """Bulk upload labelled images for one class. They are stored for the next retraining."""
    if class_name not in predictor.class_names:
        raise HTTPException(status_code=400, detail=f"Unknown class '{class_name}'. See GET /classes.")

    saved, rejected = 0, []
    for upload in files:
        try:
            _save_image(await upload.read(), class_name, upload.filename)
            saved += 1
        except ValueError:
            rejected.append(upload.filename)

    return {
        "class_name": class_name,
        "saved": saved,
        "rejected": rejected,
        "pending_uploads": _pending_uploads(),
        "retrain_threshold": RETRAIN_THRESHOLD,
        **_maybe_auto_retrain(),
    }


@app.post("/upload/zip")
async def upload_zip(file: UploadFile = File(...)):
    """
    Bulk upload a .zip of labelled images, organised as one folder per class:
        Tomato___Early_blight/leaf1.jpg
        Corn___Common_rust/leaf2.jpg
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(await file.read()))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The file is not a valid .zip archive.")

    saved, rejected, per_class = 0, [], {}
    for entry in archive.infolist():
        if entry.is_dir() or "__MACOSX" in entry.filename:
            continue
        parts = entry.filename.replace("\\", "/").split("/")
        if len(parts) < 2 or not parts[-1].lower().endswith(VALID_EXTENSIONS):
            continue
        class_name = parts[-2]
        if class_name not in predictor.class_names:
            rejected.append(entry.filename)
            continue
        try:
            _save_image(archive.read(entry), class_name, parts[-1])
            saved += 1
            per_class[class_name] = per_class.get(class_name, 0) + 1
        except ValueError:
            rejected.append(entry.filename)

    return {
        "saved": saved,
        "saved_per_class": per_class,
        "rejected": rejected[:50],
        "rejected_count": len(rejected),
        "pending_uploads": _pending_uploads(),
        "retrain_threshold": RETRAIN_THRESHOLD,
        **_maybe_auto_retrain(),
    }


@app.post("/retrain")
def retrain():
    """Manual trigger: start retraining on the uploaded images now."""
    pending = _pending_uploads()
    if pending == 0:
        raise HTTPException(status_code=400, detail="No uploaded images to retrain on. Upload data first.")
    if not start_retraining(manual=True, trigger_label="manual"):
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    return {"message": "Retraining started.", "new_images": pending, "status_url": "/retrain/status"}


@app.get("/retrain/status")
def retrain_status():
    return {**retrain_state, "pending_uploads": _pending_uploads(), "retrain_threshold": RETRAIN_THRESHOLD}

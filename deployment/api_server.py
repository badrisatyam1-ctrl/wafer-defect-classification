"""
FastAPI REST API — Wafer Defect Classification
================================================
Production-grade API for integrating with Manufacturing Execution Systems (MES).

Endpoints:
    POST /predict        — Classify a single wafer image
    POST /batch          — Classify multiple wafer images
    GET  /health         — Health check + model status
    GET  /classes        — List all defect classes
    GET  /docs           — Auto-generated Swagger UI (built-in)

Run:
    uvicorn deployment.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# FastAPI
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.inference import WaferInferenceEngine
from models.resnet18_classifier import DEFECT_CLASSES_V2

# ═══════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Wafer Defect Classification API",
    description=(
        "Industry-grade macro-level wafer defect detection powered by ResNet18 + Focal Loss + Grad-CAM. "
        "Designed for integration with semiconductor manufacturing execution systems (MES)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Engine ─────────────────────────────────────────────────────
CHECKPOINT_REAL = PROJECT_ROOT / "models" / "best.pt"
CHECKPOINT_SYNTH = PROJECT_ROOT / "models" / "synthetic_model.pt"

_engine_real: Optional[WaferInferenceEngine] = None
_engine_synth: Optional[WaferInferenceEngine] = None


def get_engine(mode: str = "production") -> WaferInferenceEngine:
    global _engine_real, _engine_synth
    if mode == "synthetic":
        if _engine_synth is None:
            _engine_synth = WaferInferenceEngine(checkpoint_path=CHECKPOINT_SYNTH)
        return _engine_synth
    else:
        if _engine_real is None:
            _engine_real = WaferInferenceEngine(checkpoint_path=CHECKPOINT_REAL)
        return _engine_real


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════

class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    severity: str
    class_probabilities: Dict[str, float]
    wafer_detected: bool
    grad_cam_reliable: bool
    inference_time_ms: float
    timestamp: str


class BatchResult(BaseModel):
    total_wafers: int
    normal_count: int
    defect_count: int
    yield_rate: float
    average_confidence: float
    results: List[PredictionResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_path: str
    device: str
    num_classes: int
    class_names: List[str]
    uptime_seconds: float


# ═══════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════

_start_time = time.time()


def _classify_severity(pred: str, conf: float) -> str:
    if pred == "normal":
        return "none"
    if conf > 0.85:
        return "high"
    if conf > 0.60:
        return "medium"
    return "low"


def _image_from_upload(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded file bytes to numpy RGB array."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(image)


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    try:
        engine = get_engine("production")
        return HealthResponse(
            status="healthy",
            model_loaded=engine.model is not None,
            checkpoint_path=str(engine.checkpoint_path),
            device=engine.device,
            num_classes=len(engine.CLASS_NAMES),
            class_names=engine.CLASS_NAMES,
            uptime_seconds=round(time.time() - _start_time, 1),
        )
    except Exception as e:
        return HealthResponse(
            status=f"degraded: {e}",
            model_loaded=False,
            checkpoint_path="N/A",
            device="cpu",
            num_classes=0,
            class_names=[],
            uptime_seconds=round(time.time() - _start_time, 1),
        )


@app.get("/classes", tags=["System"])
async def list_classes():
    """List all supported defect classes."""
    return {
        "classes": DEFECT_CLASSES_V2,
        "num_classes": len(DEFECT_CLASSES_V2),
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict_single(
    file: UploadFile = File(..., description="Wafer image (PNG/JPG)"),
    mode: str = "production",
):
    """
    Classify a single wafer image.

    **Mode:**
    - `production` — Uses the WM-811K trained model (default)
    - `synthetic` — Uses the synthetic sandbox model
    """
    if mode not in ("production", "synthetic"):
        raise HTTPException(400, "Mode must be 'production' or 'synthetic'")

    try:
        file_bytes = await file.read()
        image = _image_from_upload(file_bytes)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    engine = get_engine(mode)

    t0 = time.time()
    result = engine.predict_from_array(image, api_safe=True)
    inference_ms = (time.time() - t0) * 1000

    pred = result.get("predicted_class", "unknown")
    conf = result.get("confidence", 0.0)

    return PredictionResult(
        predicted_class=pred,
        confidence=round(conf, 4),
        severity=_classify_severity(pred, conf),
        class_probabilities={k: round(v, 4) for k, v in result.get("class_probabilities", {}).items()},
        wafer_detected=result.get("wafer_detected", False),
        grad_cam_reliable=result.get("grad_cam_reliable", False),
        inference_time_ms=round(inference_ms, 1),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/batch", response_model=BatchResult, tags=["Inference"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple wafer images"),
    mode: str = "production",
):
    """
    Classify a batch of wafer images.

    Returns individual results for each wafer plus aggregate statistics
    including yield rate, defect distribution, and average confidence.
    """
    if mode not in ("production", "synthetic"):
        raise HTTPException(400, "Mode must be 'production' or 'synthetic'")

    if len(files) > 100:
        raise HTTPException(400, "Maximum 100 images per batch")

    engine = get_engine(mode)
    results: List[PredictionResult] = []

    t0 = time.time()

    for upload in files:
        try:
            file_bytes = await upload.read()
            image = _image_from_upload(file_bytes)

            t_single = time.time()
            result = engine.predict_from_array(image, api_safe=True)
            single_ms = (time.time() - t_single) * 1000

            pred = result.get("predicted_class", "unknown")
            conf = result.get("confidence", 0.0)

            results.append(PredictionResult(
                predicted_class=pred,
                confidence=round(conf, 4),
                severity=_classify_severity(pred, conf),
                class_probabilities={k: round(v, 4) for k, v in result.get("class_probabilities", {}).items()},
                wafer_detected=result.get("wafer_detected", False),
                grad_cam_reliable=result.get("grad_cam_reliable", False),
                inference_time_ms=round(single_ms, 1),
                timestamp=datetime.now().isoformat(),
            ))
        except Exception as e:
            results.append(PredictionResult(
                predicted_class="error",
                confidence=0.0,
                severity="unknown",
                class_probabilities={},
                wafer_detected=False,
                grad_cam_reliable=False,
                inference_time_ms=0.0,
                timestamp=datetime.now().isoformat(),
            ))

    total_ms = (time.time() - t0) * 1000

    normal_count = sum(1 for r in results if r.predicted_class == "normal")
    defect_count = len(results) - normal_count
    avg_conf = np.mean([r.confidence for r in results]) if results else 0.0
    yield_rate = normal_count / len(results) if results else 0.0

    return BatchResult(
        total_wafers=len(results),
        normal_count=normal_count,
        defect_count=defect_count,
        yield_rate=round(yield_rate, 4),
        average_confidence=round(float(avg_conf), 4),
        results=results,
        processing_time_ms=round(total_ms, 1),
    )


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  Wafer Defect Classification API")
    print("  Docs:   http://localhost:8000/docs")
    print("  Health: http://localhost:8000/health")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Stage 6 — FastAPI inference service.

Provides:
  GET  /health          — readiness check
  POST /predict         — single-image detection + optional LLaVA reasoning
  POST /predict/batch   — batch detection (list of images)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from visionllm_interactionanalysis.api.schemas import (
    HealthResponse,
    PredictionResponse,
)
from visionllm_interactionanalysis.pipeline.prediction_pipeline import predict_single
from visionllm_interactionanalysis.utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="VisionLLM InteractionAnalysis API",
    version="0.2.0",
    description="Object detection + multimodal scene reasoning",
)

# ── Globals loaded at startup ────────────────────────────────────────
_CHECKPOINT: str = os.environ.get("VISIONLLM_CHECKPOINT", "")
_NUM_CLASSES: int = int(os.environ.get("VISIONLLM_NUM_CLASSES", "81"))
_SCORE_THRESHOLD: float = float(os.environ.get("VISIONLLM_SCORE_THRESHOLD", "0.5"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=bool(_CHECKPOINT and Path(_CHECKPOINT).exists()),
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File(description="Image file (JPEG/PNG)")],
    use_llava: Annotated[bool, Query(description="Enable LLaVA reasoning")] = False,
    score_threshold: Annotated[float, Query(ge=0, le=1)] = _SCORE_THRESHOLD,
) -> PredictionResponse:
    """Run detection (+ optional reasoning) on a single uploaded image."""

    if not _CHECKPOINT or not Path(_CHECKPOINT).exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not configured")

    suffix = Path(file.filename or "img.jpg").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_single(
            image_path=tmp_path,
            checkpoint=_CHECKPOINT,
            num_classes=_NUM_CLASSES,
            score_threshold=score_threshold,
            use_llava=use_llava,
        )
        scene = result.copy()
        reasoning = scene.pop("reasoning", None)
        return PredictionResponse(scene=scene, reasoning=reasoning)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(description="Multiple image files"),
    score_threshold: float = Query(default=_SCORE_THRESHOLD, ge=0, le=1),
) -> JSONResponse:
    """Batch detection on multiple images (no LLaVA for batch)."""

    if not _CHECKPOINT or not Path(_CHECKPOINT).exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not configured")

    results = []
    for f in files:
        suffix = Path(f.filename or "img.jpg").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name
        try:
            r = predict_single(
                image_path=tmp_path,
                checkpoint=_CHECKPOINT,
                num_classes=_NUM_CLASSES,
                score_threshold=score_threshold,
                use_llava=False,
            )
            results.append({"filename": f.filename, "result": r})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
        finally:
            os.unlink(tmp_path)

    return JSONResponse(content=results)

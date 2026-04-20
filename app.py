"""
Swin2-MoSE FastAPI sidecar.

GPL-v2 isolated: this image runs the upstream IMPLabUniPr/swin2-mose model
(once vendored) and exposes /predict, /health, /warmup over HTTP. The
Taqneen Flask backend calls this via services/swin2_client.py.

Phase 1 status:
    The real Swin2-MoSE model is TODO until the GPL-v2 written-offer landing
    page is approved by ministry legal (docs/phase0/gpl-written-offer.md).
    Until then, the sidecar runs a deterministic bicubic upscale that
    exercises the full pipeline contract (inputs, outputs, provenance fields)
    end-to-end. Callers can flip MODEL_MODE=real once the upstream repo is
    vendored + weights present at /app/weights/.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger("swin2_mose")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_MODE = os.environ.get("MODEL_MODE", "placeholder").lower()   # placeholder | real
MODEL_VERSION = os.environ.get("MODEL_VERSION", "swin2_mose_placeholder_v0.1")
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/app/weights/sen2venus_x4.ckpt")
SIDECAR_IMAGE_DIGEST = os.environ.get("SIDECAR_IMAGE_DIGEST", "unknown")

# Preempt channel support (Redis pub/sub, optional)
REDIS_URL = os.environ.get("REDIS_URL", "redis://taqneen_redis:6379/0")


app = FastAPI(title="Swin2-MoSE sidecar", version=MODEL_VERSION)


# ── Models ──────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    input_path: str
    output_path: str
    scale: int = 4
    request_id: str | None = None


class PredictResponse(BaseModel):
    output_path: str
    scale: int
    model_version: str
    model_sha: str
    sidecar_image_digest: str
    inference_ms: int
    output_bytes: int
    mode: str


# ── Lazy model singleton ────────────────────────────────────────────────
_model_lock = threading.Lock()
_model = None  # placeholder mode keeps this None


def _load_model():
    """Load Swin2-MoSE weights lazily. No-op in placeholder mode."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        if MODEL_MODE != "real":
            logger.info("Swin2 sidecar running in PLACEHOLDER mode (bicubic).")
            _model = "placeholder"
            return _model

        # TODO(phase1): real model load once upstream is vendored.
        # Example shape (commented until vendored):
        # import torch
        # from swin2_mose.model import load_checkpoint
        # model = load_checkpoint(WEIGHTS_PATH)
        # model.eval().to("cuda")
        # _model = model
        raise RuntimeError("MODEL_MODE=real requested but upstream not vendored yet")


def _weights_sha() -> str:
    """SHA-256 of the weights file (or placeholder for placeholder mode)."""
    if MODEL_MODE != "real" or not os.path.exists(WEIGHTS_PATH):
        return f"placeholder-v0.1-scale-agnostic"
    sha = hashlib.sha256()
    with open(WEIGHTS_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ── Inference ──────────────────────────────────────────────────────────
def _infer_placeholder(input_path: str, output_path: str, scale: int) -> None:
    """Bicubic upscale via rasterio. Preserves CRS + transform correctly."""
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        new_w = src.width * scale
        new_h = src.height * scale
        new_transform = src.transform * src.transform.scale(
            (src.width / new_w), (src.height / new_h)
        )
        profile.update(
            width=new_w,
            height=new_h,
            transform=new_transform,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        data = src.read(
            out_shape=(src.count, new_h, new_w),
            resampling=Resampling.cubic,
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)


def _infer_real(input_path: str, output_path: str, scale: int) -> None:
    """TODO(phase1): real Swin2-MoSE inference."""
    raise NotImplementedError(
        "Swin2-MoSE real inference is not wired yet. "
        "Vendor IMPLabUniPr/swin2-mose, implement model load + forward pass, "
        "then flip MODEL_MODE=real."
    )


# ── Endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "mode": MODEL_MODE,
        "version": MODEL_VERSION,
        "sidecar_digest": SIDECAR_IMAGE_DIGEST,
    }


@app.post("/warmup")
def warmup():
    _load_model()
    return {"ok": True, "mode": MODEL_MODE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.scale not in (2, 4):
        raise HTTPException(status_code=400, detail="scale must be 2 or 4")
    if not os.path.exists(req.input_path):
        raise HTTPException(status_code=400, detail="input_path does not exist")
    os.makedirs(os.path.dirname(req.output_path), exist_ok=True)

    _load_model()
    t0 = time.monotonic()
    try:
        if MODEL_MODE == "real":
            _infer_real(req.input_path, req.output_path, req.scale)
        else:
            _infer_placeholder(req.input_path, req.output_path, req.scale)
    except Exception as exc:
        logger.exception("swin2 predict failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}")

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    size_bytes = os.path.getsize(req.output_path) if os.path.exists(req.output_path) else 0

    return PredictResponse(
        output_path=req.output_path,
        scale=req.scale,
        model_version=MODEL_VERSION,
        model_sha=_weights_sha(),
        sidecar_image_digest=SIDECAR_IMAGE_DIGEST,
        inference_ms=elapsed_ms,
        output_bytes=size_bytes,
        mode=MODEL_MODE,
    )

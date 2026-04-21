"""
Swin2-MoSE FastAPI sidecar.

GPL-v2 isolated: this image runs the upstream IMPLabUniPr/swin2-mose model
(vendored at /app/swin2_mose_upstream via Dockerfile pin to the v1.0 commit
SHA) and exposes /predict, /health, /warmup over HTTP. Proprietary LCM
applications call this sidecar only over HTTP.

Modes:
    MODEL_MODE=placeholder (default)
        Deterministic bicubic upscale via rasterio. Exercises the full
        pipeline contract (inputs, outputs, provenance fields) end-to-end
        without loading any GPL-v2 code.
    MODEL_MODE=real
        Real Swin2-MoSE inference. Requires the pretrained weights to be
        extracted into /app/weights/ (see sidecars/swin2_mose/README.md and
        docs/phase0/flip-to-real-procedure.md).
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
# WEIGHTS_DIR holds an extracted upstream release (cfg.yml + checkpoints/*.pt).
# For backwards compat, WEIGHTS_PATH is preserved and used for the SHA field
# when it points at a file. The loader walks WEIGHTS_DIR to find cfg + ckpt.
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "")
SIDECAR_IMAGE_DIGEST = os.environ.get("SIDECAR_IMAGE_DIGEST", "unknown")
TILE_SIZE = int(os.environ.get("SWIN2_TILE_SIZE", "128"))
TILE_OVERLAP = int(os.environ.get("SWIN2_TILE_OVERLAP", "16"))

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
_ckpt_path_cached: str | None = None  # populated on first real-mode load


def _load_model():
    """Load Swin2-MoSE weights lazily. No-op in placeholder mode."""
    global _model, _ckpt_path_cached
    with _model_lock:
        if _model is not None:
            return _model
        if MODEL_MODE != "real":
            logger.info("Swin2 sidecar running in PLACEHOLDER mode (bicubic).")
            _model = "placeholder"
            return _model

        import torch
        from inference import build_and_load

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            logger.warning(
                "CUDA not available; running Swin2-MoSE on CPU. Inference will be slow."
            )
        model, cfg, ckpt_path = build_and_load(WEIGHTS_DIR, device)
        _ckpt_path_cached = ckpt_path
        _model = model
        logger.info(
            "Swin2-MoSE model loaded (device=%s, ckpt=%s, upstream_commit=%s)",
            device, ckpt_path, os.environ.get("SWIN2_COMMIT", "unknown"),
        )
        return _model


def _weights_sha() -> str:
    """SHA-256 of the weights file (or placeholder marker for placeholder mode)."""
    if MODEL_MODE != "real":
        return "placeholder-v0.1-scale-agnostic"
    path = _ckpt_path_cached or WEIGHTS_PATH
    if not path or not os.path.exists(path):
        return "real-weights-unknown"
    sha = hashlib.sha256()
    with open(path, "rb") as f:
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
    """Real Swin2-MoSE inference via the vendored upstream model."""
    import torch
    from inference import infer_to_geotiff

    model = _model  # guaranteed loaded by _load_model()
    if model is None or model == "placeholder":
        raise RuntimeError(
            "_infer_real called but model is not loaded in real mode"
        )
    device = next(model.parameters()).device
    infer_to_geotiff(
        model=model,
        input_path=input_path,
        output_path=output_path,
        scale=scale,
        device=device,
        tile_size=TILE_SIZE,
        overlap=TILE_OVERLAP,
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

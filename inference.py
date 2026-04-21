"""
Swin2-MoSE real inference: load upstream model + weights, tile input, stitch.

This module is imported by app.py when MODEL_MODE=real. It depends on the
upstream IMPLabUniPr/swin2-mose repo being vendored at /app/swin2_mose_upstream/
(see Dockerfile) and pretrained weights being mounted at WEIGHTS_DIR.

Expected weights layout (after extracting the release zip, e.g.
sen2venus_exp4_4x_v5.zip from the v1.0 release, into the weights volume):

    /app/weights/
      sen2venus_exp4_4x_v5/
        cfg.yml                       (model config - constructor kwargs)
        checkpoints/
          model-70.pt                 (torch checkpoint with model_state_dict)
        eval/                         (optional; not used at inference time)

The sidecar walks the volume to find the cfg + ckpt pair, so the exact
folder name is not hardcoded.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger("swin2_mose.inference")

UPSTREAM_ROOT = os.environ.get("SWIN2_UPSTREAM", "/app/swin2_mose_upstream")
for _p in (UPSTREAM_ROOT, os.path.join(UPSTREAM_ROOT, "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


# Swin2MoSE.__init__ accepts these kwargs (see upstream swin2_mose_model/model.py).
KNOWN_MODEL_KWARGS = frozenset({
    "img_size", "patch_size", "in_chans", "embed_dim", "depths", "num_heads",
    "window_size", "mlp_ratio", "qkv_bias", "drop_rate", "attn_drop_rate",
    "drop_path_rate", "ape", "patch_norm", "use_checkpoint", "upscale",
    "img_range", "upsampler", "resi_connection", "use_lepe", "use_cpb_bias",
    "MoE_config", "use_rpe_bias",
})


def discover_weights(weights_dir: str) -> Tuple[str, str]:
    """Find cfg.yml + model checkpoint in the weights volume.

    Walks the tree and returns (cfg_path, ckpt_path). Raises FileNotFoundError
    if either is missing.
    """
    cfg_path: Optional[str] = None
    ckpt_path: Optional[str] = None

    for root, _, files in os.walk(weights_dir):
        if cfg_path is None and "cfg.yml" in files:
            cfg_path = os.path.join(root, "cfg.yml")
        if ckpt_path is None:
            # Prefer model-XX.pt with the highest epoch number
            pts = sorted(f for f in files if f.startswith("model-") and f.endswith(".pt"))
            if pts:
                ckpt_path = os.path.join(root, pts[-1])

    if not cfg_path or not ckpt_path:
        raise FileNotFoundError(
            f"Weights not discovered under {weights_dir!r}. "
            f"Expected subdirectory with cfg.yml and checkpoints/model-*.pt "
            f"(e.g. extract sen2venus_exp4_4x_v5.zip from upstream v1.0 release). "
            f"Found cfg={cfg_path!r}, ckpt={ckpt_path!r}."
        )
    return cfg_path, ckpt_path


def _extract_model_kwargs(cfg: dict) -> dict:
    """Pull Swin2MoSE constructor kwargs out of a cfg dict.

    Upstream puts model params under cfg['generator'] for SRGAN-style configs.
    We check a few canonical keys, and filter to only the kwargs Swin2MoSE
    actually accepts (to avoid accidental upstream-private keys).
    """
    candidates = [
        cfg.get("generator"),
        cfg.get("model"),
        cfg.get("swin2_mose"),
        cfg,
    ]
    for c in candidates:
        if not isinstance(c, dict):
            continue
        kwargs = {k: v for k, v in c.items() if k in KNOWN_MODEL_KWARGS}
        # heuristic: a real Swin2MoSE config will always carry at least upscale
        # and embed_dim, so we use them to decide we found the right section
        if "upscale" in kwargs and "embed_dim" in kwargs:
            return kwargs
    raise ValueError(
        "Could not locate Swin2MoSE model kwargs in cfg.yml. Inspect the file "
        "and, if the model section has a non-standard name, update "
        "inference._extract_model_kwargs to check it."
    )


def build_model(cfg_path: str):
    """Load cfg.yml, import Swin2MoSE from vendored upstream, instantiate."""
    # Upstream serializes cfg.yml with Python tags (!!python/object/new:easydict.EasyDict).
    # SafeLoader rejects these; unsafe_load handles them by calling the
    # easydict constructor. Acceptable because the cfg.yml comes directly
    # from the upstream GPL-v2 release zip fetched by download_weights.sh
    # (no arbitrary third-party YAML is passed through here).
    with open(cfg_path, "r") as f:
        cfg = yaml.unsafe_load(f)

    try:
        from swin2_mose_model.model import Swin2MoSE  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import swin2_mose_model.model.Swin2MoSE. "
            f"Verify the upstream repo is vendored at {UPSTREAM_ROOT!r} "
            f"(Dockerfile should git clone + checkout the pinned SWIN2_COMMIT). "
            f"Error: {exc}"
        ) from exc

    kwargs = _extract_model_kwargs(cfg)
    logger.info("Instantiating Swin2MoSE with kwargs: %s", kwargs)
    model = Swin2MoSE(**kwargs)
    return model, cfg


def load_weights(model, ckpt_path: str, device):
    """Load a PyTorch checkpoint into the model. Tolerant of extra/missing keys."""
    import torch

    logger.info("Loading checkpoint from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Checkpoint missing %d keys: %s...", len(missing), list(missing)[:5])
    if unexpected:
        logger.warning("Checkpoint has %d unexpected keys: %s...", len(unexpected), list(unexpected)[:5])
    return model


def build_and_load(weights_dir: str, device):
    """Discover weights, build model, load state dict, move to device, eval mode."""
    cfg_path, ckpt_path = discover_weights(weights_dir)
    model, cfg = build_model(cfg_path)
    model = load_weights(model, ckpt_path, device)
    model.eval().to(device)
    return model, cfg, ckpt_path


# --- tiled inference helpers ---------------------------------------------

def _half_cosine_weight(h: int, w: int, ramp: int) -> np.ndarray:
    """2D weight mask that is 1.0 in the center and cosine-tapered at edges.

    Used to blend overlapping tile outputs without visible seams.
    """
    def _1d(n: int, r: int) -> np.ndarray:
        x = np.ones(n, dtype=np.float32)
        r = max(0, min(r, n // 2))
        if r == 0:
            return x
        ramp_vals = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, r, dtype=np.float32))
        x[:r] *= ramp_vals
        x[-r:] *= ramp_vals[::-1]
        return x
    return np.outer(_1d(h, ramp), _1d(w, ramp)).astype(np.float32)


def tiled_inference(model, x_tensor, scale: int, tile_size: int = 128, overlap: int = 16):
    """Run model.forward on overlapping tiles; stitch with half-cosine blending.

    Args:
        model: Swin2MoSE instance (already .eval() and on the target device).
        x_tensor: torch.Tensor of shape (1, C, H, W), on model's device.
        scale: upscale factor (2 or 4).
        tile_size: input tile size in pixels.
        overlap: pixels of overlap between adjacent tiles.

    Returns:
        torch.Tensor of shape (1, C, H*scale, W*scale).
    """
    import torch

    if x_tensor.ndim != 4 or x_tensor.shape[0] != 1:
        raise ValueError(f"expected shape (1, C, H, W); got {tuple(x_tensor.shape)}")
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"invalid stride: tile_size={tile_size}, overlap={overlap}")

    _, c, h, w = x_tensor.shape
    out_h, out_w = h * scale, w * scale
    device = x_tensor.device
    # Accumulate in float32 to avoid precision loss from weighted sums.
    accum = torch.zeros((1, c, out_h, out_w), dtype=torch.float32, device=device)
    weight = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=device)

    # Pre-build the output-space blend window (one tile is tile_size*scale wide).
    win_np = _half_cosine_weight(tile_size * scale, tile_size * scale, overlap * scale)
    win_full = torch.from_numpy(win_np).to(device).view(1, 1, *win_np.shape)

    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))
    ys = sorted({y for y in ys if y >= 0})
    xs = sorted({x for x in xs if x >= 0})

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile_in = x_tensor[:, :, y:y + tile_size, x:x + tile_size]
                th_in, tw_in = tile_in.shape[2], tile_in.shape[3]
                pad_b = tile_size - th_in
                pad_r = tile_size - tw_in
                if pad_b > 0 or pad_r > 0:
                    tile_in = torch.nn.functional.pad(
                        tile_in, (0, pad_r, 0, pad_b), mode="reflect"
                    )

                out = model(tile_in)
                if isinstance(out, tuple):
                    # Swin2-MoSE returns (tensor, loss_moe)
                    out = out[0]

                # Crop padded region from the output (it scales with `scale`).
                crop_h = th_in * scale
                crop_w = tw_in * scale
                out = out[:, :, :crop_h, :crop_w].to(torch.float32)

                oy = y * scale
                ox = x * scale
                th_out, tw_out = out.shape[2], out.shape[3]

                if th_out == win_full.shape[2] and tw_out == win_full.shape[3]:
                    w_tile = win_full
                else:
                    w_np = _half_cosine_weight(th_out, tw_out, overlap * scale)
                    w_tile = torch.from_numpy(w_np).to(device).view(1, 1, th_out, tw_out)

                accum[:, :, oy:oy + th_out, ox:ox + tw_out] += out * w_tile
                weight[:, :, oy:oy + th_out, ox:ox + tw_out] += w_tile

    return accum / weight.clamp(min=1e-8)


# --- GeoTIFF I/O wrapper -------------------------------------------------

def infer_to_geotiff(
    model,
    input_path: str,
    output_path: str,
    scale: int,
    device,
    tile_size: int = 128,
    overlap: int = 16,
    band_mean: Optional[list] = None,
    band_std: Optional[list] = None,
):
    """Read a GeoTIFF, run tiled SR inference, write a scaled GeoTIFF.

    CRS preservation: the output affine transform is scaled by 1/scale so the
    same geographic bbox is covered at higher pixel density.

    Args:
        band_mean / band_std: optional per-band Z-normalization stats. When
            provided, input is z-normalized before forward() and de-normalized
            on the output. Typical Sen2Venus values are in [0, 1] already, so
            leaving these None + relying on img_range baked into the model is
            usually correct.
    """
    import rasterio
    import torch

    with rasterio.open(input_path) as src:
        arr = src.read()  # (C, H, W)
        profile = src.profile.copy()
        src_transform = src.transform
        src_width = src.width
        src_height = src.height
        src_dtype = arr.dtype

    x = arr.astype(np.float32)

    if band_mean is not None and band_std is not None:
        if len(band_mean) != x.shape[0] or len(band_std) != x.shape[0]:
            raise ValueError(
                f"band_mean/std length ({len(band_mean)}/{len(band_std)}) "
                f"does not match input bands ({x.shape[0]})"
            )
        for i in range(x.shape[0]):
            x[i] = (x[i] - band_mean[i]) / max(float(band_std[i]), 1e-6)
    elif x.max() > 1.0:
        # Default to per-image scale to [0, 1] if no normalization provided.
        x = x / max(x.max(), 1e-6)

    tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    logger.info(
        "Running tiled inference: input shape=%s, scale=%d, tile=%d, overlap=%d",
        tuple(tensor.shape), scale, tile_size, overlap,
    )
    out = tiled_inference(model, tensor, scale=scale, tile_size=tile_size, overlap=overlap)
    out_np = out.squeeze(0).cpu().numpy()

    if band_mean is not None and band_std is not None:
        for i in range(out_np.shape[0]):
            out_np[i] = out_np[i] * float(band_std[i]) + float(band_mean[i])

    # Cast back to the input's dtype range to preserve downstream expectations.
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        out_np = np.clip(out_np, info.min, info.max).astype(src_dtype)
    else:
        out_np = out_np.astype(np.float32)

    new_w = src_width * scale
    new_h = src_height * scale
    new_transform = src_transform * src_transform.scale(
        src_width / new_w, src_height / new_h
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

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_np)

    logger.info("Wrote SR GeoTIFF: %s shape=%s dtype=%s", output_path, out_np.shape, out_np.dtype)

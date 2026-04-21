# taqneen-swin2-mose-sidecar

GPL v2.0 super-resolution sidecar operated by
**Land Connectivity Matrix ("LCM")** for use inside LCM's commercial
software systems, for Sentinel-2 imagery enhancement.

This is LCM's maintained fork of
[IMPLabUniPr/swin2-mose](https://github.com/IMPLabUniPr/swin2-mose),
pinned at the upstream v1.0 commit
`5342a1c1d23791084dc7507df1faa209ce5b16ae`. LCM adds a FastAPI serving
wrapper, a Dockerfile, tiled GeoTIFF inference, and integration with LCM
applications over a strict HTTP-only boundary.

## License

All code in this repository is licensed under **GNU General Public
License v2.0** (GPL v2.0). See [`LICENSE`](LICENSE) for full terms.

For GPL-v2 Section 3 written-offer terms (source availability
commitment), see [`WRITTEN_OFFER.md`](WRITTEN_OFFER.md).

## Upstream

- Upstream repo: https://github.com/IMPLabUniPr/swin2-mose
- Upstream license: GPL v2.0
- Upstream model: Swin2-MoSE - Swin Transformer V2 with Mixture of
  Experts, trained for multi-spectral super-resolution on Sentinel-2.

## LCM modifications

- FastAPI serving wrapper (`app.py`): HTTP endpoints `/health`, `/warmup`,
  `/predict`.
- Tiled GeoTIFF inference (`inference.py`): model loading from a weights
  directory, half-cosine overlap blending for seamless stitching, and
  GeoTIFF I/O with CRS preservation.
- Dockerfile (CUDA 12.4 runtime base) that git-clones the upstream repo
  at the pinned commit and exposes it via `PYTHONPATH`.
- Integration-specific environment variables.
- Two modes:
  - **`MODEL_MODE=placeholder`** (default): deterministic bicubic upscale
    via rasterio, used for pipeline testing without loading GPL-v2 code
    at runtime.
  - **`MODEL_MODE=real`**: real Swin2-MoSE inference on GPU. Requires
    pretrained weights extracted under `WEIGHTS_DIR`.
- Helper script `download_weights.sh` for fetching the upstream release
  zip (`sen2venus_exp4_{2,4}x_v5.zip`).

## Build

```bash
docker build -t taqneen_swin2_mose:dev .
```

## Run

```bash
# Placeholder mode (no weights needed):
docker run --rm -p 8801:8801 \
  -e MODEL_MODE=placeholder \
  taqneen_swin2_mose:dev

# Real mode (weights mounted under /app/weights):
docker run --rm --gpus all -p 8801:8801 \
  -v /path/to/weights:/app/weights \
  -e MODEL_MODE=real \
  -e MODEL_VERSION=swin2_mose_sen2venus_x4_v1.0 \
  taqneen_swin2_mose:dev
```

## Endpoints

| Method | Path       | Purpose                                       |
|--------|------------|-----------------------------------------------|
| GET    | `/health`  | Liveness + mode + version + sidecar digest    |
| POST   | `/warmup`  | Load weights if not already loaded            |
| POST   | `/predict` | Super-resolve an input COG at scale 2 or 4    |

## Environment variables

| Variable                | Default                                     | Purpose |
|-------------------------|---------------------------------------------|---------|
| `MODEL_MODE`            | `placeholder`                               | `placeholder` or `real` |
| `MODEL_VERSION`         | `swin2_mose_placeholder_v0.1`               | Surfaced in provenance. MUST NOT contain "placeholder" in real mode. |
| `WEIGHTS_DIR`           | `/app/weights`                              | Loader walks this for `cfg.yml` + `checkpoints/model-*.pt` |
| `SWIN2_TILE_SIZE`       | `128`                                       | Input-space tile size in pixels |
| `SWIN2_TILE_OVERLAP`    | `16`                                        | Pixels of overlap between adjacent tiles (half-cosine blended) |
| `SIDECAR_IMAGE_DIGEST`  | `unknown`                                   | Set at deploy to `sha256:<digest>` |

## Relationship to LCM's proprietary software

The LCM platforms that call this sidecar (Flask backends, React
frontends, databases, and everything else) are **not** part of this
repository and are **not** licensed under GPL v2.0. They are proprietary,
closed-source, all rights reserved, owned by LCM.

Those LCM platforms invoke this sidecar only over HTTP. No GPL code
crosses the HTTP boundary into any LCM platform, and no proprietary LCM
code crosses into this sidecar. The process-level boundary is what keeps
the licensing separation clean.

## Contact

- GPL source inquiries: **`contact@lcm.com`**
- Bug reports / questions: open an issue on this repository.

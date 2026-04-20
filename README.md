# taqneen-swin2-mose-sidecar

GPL v2.0 super-resolution sidecar used by Taqneen (a land-survey
platform operated by **Land Connectivity Matrix ("LCM")**) for
Sentinel-2 imagery enhancement.

This is LCM's maintained fork of
[IMPLabUniPr/swin2-mose](https://github.com/IMPLabUniPr/swin2-mose).
LCM adds a FastAPI serving wrapper, a Dockerfile, and integration with
the Taqneen platform over a strict HTTP-only boundary.

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

- FastAPI serving wrapper (`app.py`): HTTP endpoints `/health`,
  `/warmup`, `/predict`.
- Dockerfile (CUDA 12.4 runtime base) for containerized deployment.
- Integration-specific environment variables.
- Placeholder mode (`MODEL_MODE=placeholder`): deterministic bicubic
  upscale via rasterio, used for pipeline testing without vendoring
  upstream weights. Real mode (`MODEL_MODE=real`) will be activated
  once LCM vendors the upstream repository at a pinned commit.

## Build

```bash
docker build -t taqneen_swin2_mose:dev .
```

## Run

```bash
docker run --rm -p 8801:8801 \
  -e MODEL_MODE=placeholder \
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
| `MODEL_VERSION`         | `swin2_mose_placeholder_v0.1`               | Surfaced in provenance |
| `WEIGHTS_PATH`          | `/app/weights/sen2venus_x4.ckpt`            | Only used when `MODEL_MODE=real` |
| `SIDECAR_IMAGE_DIGEST`  | `unknown`                                   | Set at deploy to `sha256:<digest>` |
| `REDIS_URL`             | `redis://taqneen_redis:6379/0`              | Preempt channel |

## Relationship to the Taqneen platform

The Taqneen platform itself (Flask backend, React frontend, database,
and everything else that *calls* this sidecar) is **not** part of this
repository and is **not** licensed under GPL v2.0. Taqneen is
proprietary, closed-source, all rights reserved, owned by LCM.

This sidecar is invoked by Taqneen only over HTTP. No GPL code crosses
the HTTP boundary into Taqneen, and no Taqneen code crosses into this
sidecar. The process-level boundary is what keeps the licensing
separation clean.

## Contact

- GPL source inquiries: **`contact@lcm.com`**
- Bug reports / questions: open an issue on this repository.

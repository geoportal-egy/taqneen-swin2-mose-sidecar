# Swin2-MoSE GPU inference sidecar.
#
# GPL-v2 ISOLATION BOUNDARY
# -------------------------
# This image contains GPL-v2-licensed code and pretrained weights from
# IMPLabUniPr/swin2-mose (https://github.com/IMPLabUniPr/swin2-mose).
# The Taqneen backend (proprietary, closed-source, all rights reserved, LCM-owned) calls this service ONLY over HTTP
# via services/swin2_client.py. No Taqneen source code runs inside this
# image, and no GPL-v2 code is imported into the Taqneen backend process.
#
# Corresponding source code is available at the public GPL-v2 fork:
#   https://github.com/geoportal-egy/taqneen-swin2-mose-sidecar
#
# Build:
#   docker build -t taqneen_swin2_mose:<version> sidecars/swin2_mose/
#
# Pin by digest in docker-compose.production.yml (never by tag), so result
# rows can be audited against the exact image that produced them.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgdal-dev gdal-bin \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

# TODO(phase1): clone and pin IMPLabUniPr/swin2-mose at a specific commit,
# then install the library. Until legal sign-off completes, the sidecar runs
# a bicubic-fallback placeholder that preserves the full pipeline contract.
# RUN git clone --depth 1 https://github.com/IMPLabUniPr/swin2-mose.git \
#     && cd swin2-mose && git checkout <pinned-sha> \
#     && pip install -e .

COPY app.py /app/app.py

# GPL-v2 source disclosure notice travels with every image so that LCM
# (as the GPL-compliance party) and any recipient can see the offer
# terms without needing external context. See the CI workflow
# .github/workflows/build-sidecar-swin2.yml which additionally publishes
# a source tarball to each GitHub Release on the public fork repo.
COPY WRITTEN_OFFER.txt /WRITTEN_OFFER.txt

# OCI image labels - GHCR exposes these on the package page, making the
# license posture discoverable by anyone who pulls the image.
LABEL org.opencontainers.image.source="https://github.com/geoportal-egy/taqneen-swin2-mose-sidecar" \
      org.opencontainers.image.licenses="GPL-2.0-only" \
      org.opencontainers.image.description="Swin2-MoSE super-resolution sidecar for Taqneen (GPL-v2 isolated)" \
      org.opencontainers.image.documentation="https://github.com/geoportal-egy/taqneen-swin2-mose-sidecar/blob/main/WRITTEN_OFFER.md" \
      org.opencontainers.image.vendor="Land Connectivity Matrix (LCM)" \
      taqneen.gpl.written-offer="/WRITTEN_OFFER.txt"

# Build-time arg for git traceability (set by CI).
ARG SIDECAR_GIT_SHA=unknown
ENV SIDECAR_GIT_SHA=${SIDECAR_GIT_SHA}

# Weights volume mount (bind from docker-compose):
#   -v taqneen_swin2_weights:/app/weights
VOLUME ["/app/weights"]

# Health + predict port
EXPOSE 8801

# non-root user
RUN groupadd -g 1100 swin2 && useradd -u 1100 -g 1100 -s /bin/false swin2
USER swin2

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8801", "--workers", "1", "--timeout-keep-alive", "30"]

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

# CUDA 12.8 is the first base with Blackwell (sm_120) support, required for
# RTX 50-series / H200 GPUs. Pairs with the cu128 torch wheels installed below.
ARG CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgdal-dev gdal-bin \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# torch is installed from the cu128 wheel index to get Blackwell-ready
# CUDA kernels. Other deps install from PyPI default. The --extra-index-url
# lets pip fall back to PyPI for packages not on the torch index.
RUN python -m pip install --upgrade pip && \
    pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r /app/requirements.txt

# Vendor IMPLabUniPr/swin2-mose at the v1.0 release commit. v1.0 is pinned
# because it ships pretrained weights for both 2x and 4x Sen2Venus (v1.1
# only re-trains the 2x task). The upstream repo has no setup.py, so we
# clone source and expose it via PYTHONPATH rather than pip-installing it.
#
# The commit SHA is pinned (never a branch name) so rebuilds are byte-stable
# and the source tarball attached to each GitHub Release on the public
# mirror matches the exact binary.
ARG SWIN2_COMMIT=5342a1c1d23791084dc7507df1faa209ce5b16ae
RUN git clone --no-checkout https://github.com/IMPLabUniPr/swin2-mose.git /app/swin2_mose_upstream \
    && cd /app/swin2_mose_upstream \
    && git checkout ${SWIN2_COMMIT} \
    && rm -rf .git

ENV PYTHONPATH="/app:/app/swin2_mose_upstream:/app/swin2_mose_upstream/src"
ENV SWIN2_COMMIT=${SWIN2_COMMIT}

COPY app.py /app/app.py
COPY inference.py /app/inference.py

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

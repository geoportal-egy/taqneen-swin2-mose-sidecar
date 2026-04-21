#!/usr/bin/env bash
#
# Fetch and extract pretrained Swin2-MoSE weights into the sidecar weights volume.
#
# Usage (from repo root):
#   bash sidecars/swin2_mose/download_weights.sh <target_dir> [scale]
#
# Arguments:
#   target_dir   Destination directory (e.g. ./data/swin2_weights/ locally,
#                or /app/weights inside the container). Created if missing.
#   scale        2 (default) or 4. Selects which release asset to fetch:
#                  2 -> sen2venus_exp4_2x_v5.zip
#                  4 -> sen2venus_exp4_4x_v5.zip
#
# After running, the target_dir looks like:
#   <target_dir>/
#     sen2venus_exp4_<scale>x_v5/
#       cfg.yml
#       checkpoints/model-70.pt
#       eval/...
#
# The sidecar walks the target_dir at load time to find cfg.yml + model-*.pt,
# so the exact subfolder name does not need to match anything else.

set -euo pipefail

TARGET_DIR="${1:-}"
SCALE="${2:-2}"

if [[ -z "${TARGET_DIR}" ]]; then
    echo "Usage: $0 <target_dir> [scale=2|4]" >&2
    exit 1
fi

case "${SCALE}" in
    2) ASSET="sen2venus_exp4_2x_v5.zip" ;;
    4) ASSET="sen2venus_exp4_4x_v5.zip" ;;
    *) echo "scale must be 2 or 4, got: ${SCALE}" >&2; exit 1 ;;
esac

URL="https://github.com/IMPLabUniPr/swin2-mose/releases/download/v1.0/${ASSET}"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

if [[ -f "${ASSET}" ]]; then
    echo "already present: ${TARGET_DIR}/${ASSET}; skipping download"
else
    echo "downloading ${URL}"
    curl -fSL --retry 3 -o "${ASSET}" "${URL}"
fi

echo "extracting ${ASSET} into ${TARGET_DIR}"
if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "${ASSET}"
else
    python3 -c "import zipfile; zipfile.ZipFile('${ASSET}').extractall('.')"
fi

echo "done. expected structure under ${TARGET_DIR}:"
find . -maxdepth 3 -type f \( -name "cfg.yml" -o -name "model-*.pt" \) -print

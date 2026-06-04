#!/usr/bin/env bash
# Copy everything required INTO E2E_CHILD_TSE_2 (no symlinks outside the bundle).
#
# Usage (one-time, from a machine that has the source trees):
#   SOURCE_DATA=/path/to/00_source_data \
#   SOURCE_TSE=/path/to/tse \
#   SOURCE_INFER=/path/to/latest-child-speech-inference \
#   bash scripts/install_self_contained.sh
#
# Defaults (host layout next to TSE-alpha):
#   SOURCE_DATA=$REPO_ROOT/../TSE-alpha/00_source_data  — only used as copy source, not at runtime
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_PARENT="$(cd "${ROOT}/../.." && pwd)"

SOURCE_DATA="${SOURCE_DATA:-${REPO_PARENT}/TSE-alpha/00_source_data}"
SOURCE_TSE="${SOURCE_TSE:-${REPO_PARENT}/tse}"
SOURCE_INFER="${SOURCE_INFER:-${REPO_PARENT}/latest-child-speech-inference}"

echo "=== Install self-contained bundle ==="
echo "  ROOT=$ROOT"
echo "  SOURCE_DATA=$SOURCE_DATA"
echo "  SOURCE_TSE=$SOURCE_TSE"
echo "  SOURCE_INFER=$SOURCE_INFER"

for label in SOURCE_DATA SOURCE_TSE SOURCE_INFER; do
  p="${!label}"
  [[ -e "$p" ]] || { echo "Missing $label=$p"; exit 1; }
done

# --- data/ (manifests + libri + myst) ---
if [[ -L "${ROOT}/data" ]]; then
  rm -f "${ROOT}/data"
fi
echo ">>> Copy data/ (~5 GB, may take several minutes) ..."
mkdir -p "${ROOT}/data"
rsync -a --info=progress2 "${SOURCE_DATA}/" "${ROOT}/data/"

# --- vendor/tse (inference only) ---
echo ">>> Copy vendor/tse ..."
mkdir -p "${ROOT}/vendor/tse"
for item in later libri2talker_libri2vox embedding_model.ckpt ecapa.py conformer_tse.py; do
  rsync -a "${SOURCE_TSE}/${item}" "${ROOT}/vendor/tse/"
done

# --- vendor/selection_inference (anon) ---
echo ">>> Copy vendor/selection_inference (~3 GB checkpoints) ..."
mkdir -p "${ROOT}/vendor/selection_inference"
for item in inference_selective.py inference_relative_dataset.py adapted_from_facebookresearch adapted_from_speechbrain all_checkpoints; do
  rsync -aL "${SOURCE_INFER}/${item}" "${ROOT}/vendor/selection_inference/"
done

# --- vendor/speechbrain (EER ECAPA) ---
echo ">>> Install vendor/speechbrain (EER ECAPA, downloads if missing) ..."
mkdir -p "${ROOT}/vendor/speechbrain"
ECAPA_DIR="${ROOT}/vendor/speechbrain/spkrec-ecapa-voxceleb"
if [[ ! -f "${ECAPA_DIR}/embedding_model.ckpt" ]]; then
  PY_DL="${PY:-python3}"
  "$PY_DL" -c "
from speechbrain.inference.speaker import EncoderClassifier
EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='${ECAPA_DIR}',
)
print('ECAPA saved to ${ECAPA_DIR}')
" || echo "WARN: ECAPA download failed — run EER once online or copy spkrec-ecapa-voxceleb into vendor/speechbrain/"
fi

[[ -f "${ROOT}/data/manifests/mixture_mfa_catalog_docker.csv" ]] || exit 1
[[ -f "${ROOT}/vendor/tse/later/inference_server.py" ]] || exit 1
[[ -f "${ROOT}/vendor/selection_inference/inference_selective.py" ]] || exit 1
[[ -f "${ROOT}/vendor/selection_inference/adapted_from_speechbrain/ecapa_tdnn_sb.py" ]] || exit 1

echo ""
echo "Done. Bundle is self-contained under:"
echo "  ${ROOT}/data/"
echo "  ${ROOT}/vendor/tse/"
echo "  ${ROOT}/vendor/selection_inference/"
echo "Run: bash scripts/preflight.sh && bash run.sh"

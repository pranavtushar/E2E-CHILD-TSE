#!/usr/bin/env bash
# Selection-based anonymization for TSA: child (FT) + adult (base). GPU 0.
# SELECTION anonymization (inference under bundle vendor/selection_inference).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Helpers live next to tsa_mix scripts, or in parent when this file is under scripts/
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
  PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  PIPE_DIR="$SCRIPT_DIR"
fi
BASE="${BASE_DIR:?Set BASE_DIR to E2E_CHILD_TSE_2 root (source scripts/env.sh)}"
TSA_RUN="${TSA_RUN:-tsa_mix}"
RUN_ROOT="$BASE/tse/$TSA_RUN"
MANIFEST="${MANIFEST:-$RUN_ROOT/manifests/tsa_items.jsonl}"
LISTS_DIR="${LISTS_DIR:-$RUN_ROOT/manifests}"
TSE_ROOT="${TSE_ROOT:-$RUN_ROOT/outputs/TSE_OUT}"
ANON_ROOT="${ANON_ROOT:-$RUN_ROOT/outputs/ANON_SELECTION}"
INFER_DIR="${INFER_DIR:?Set INFER_DIR — source scripts/env.sh (latest-child-speech-inference)}"
CHECKPOINT_FT="all_checkpoints/HIFI-GAN_FT_Soft_FT"
CHECKPOINT_BASE="all_checkpoints/HiFi-GAN_B_Soft_B"
SD="${DATA_ROOT:-${BASE}/data}"
BUNDLED_ADULT="${SD}/libri/test-clean-normalised"
BUNDLED_CHILD_DEV="${SD}/myst/development"
if [[ ! -d "${ADULT_REFERENCE_DIR:-}" ]]; then
  ADULT_REFERENCE_DIR="${BUNDLED_ADULT}"
fi
if [[ ! -d "${CHILD_REFERENCE_DIR:-}" ]] || [[ -z "$(ls -A "${CHILD_REFERENCE_DIR}" 2>/dev/null)" ]]; then
  if [[ -d "${BUNDLED_CHILD_DEV}" ]] && [[ -n "$(ls -A "${BUNDLED_CHILD_DEV}" 2>/dev/null)" ]]; then
    CHILD_REFERENCE_DIR="${BUNDLED_CHILD_DEV}"
  elif [[ -d "${SD}/myst/ai_kids_processed" ]]; then
    CHILD_REFERENCE_DIR="${SD}/myst/ai_kids_processed"
  else
    CHILD_REFERENCE_DIR="${SD}/myst/myST_test_filtered"
    echo "WARN: using myST_test_filtered as child reference (copy myst/development into 00_source_data for beta-matching anon)"
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PY:-${PYTHON:-python3}}"

MODE="${1:-full}"
[[ "${MODE}" == "ful" ]] && MODE="full"

echo "=== TSA Selection anonymization (GPU ${CUDA_VISIBLE_DEVICES}) ==="
echo "Mode: ${MODE}"
echo "Manifest: ${MANIFEST}"
echo "TSE root: ${TSE_ROOT}"
echo "Anon root: ${ANON_ROOT}"
echo "Inference dir: ${INFER_DIR}"
echo "Child reference: ${CHILD_REFERENCE_DIR}"
echo "Adult reference: ${ADULT_REFERENCE_DIR}"
echo ""

if [[ "${MODE}" == "smoke" ]]; then
  "${PYTHON}" "${PIPE_DIR}/build_anon_lists.py" --manifest "${MANIFEST}" --out-dir "${LISTS_DIR}" --smoke
  CHILD_LIST="${LISTS_DIR}/tsa_anon_child_smoke.txt"
  ADULT_LIST="${LISTS_DIR}/tsa_anon_adult_smoke.txt"
else
  "${PYTHON}" "${PIPE_DIR}/build_anon_lists.py" --manifest "${MANIFEST}" --out-dir "${LISTS_DIR}"
  CHILD_LIST="${LISTS_DIR}/tsa_anon_child_list.txt"
  ADULT_LIST="${LISTS_DIR}/tsa_anon_adult_list.txt"
fi
N_CHILD=$(wc -l < "${CHILD_LIST}" 2>/dev/null || echo 0)
N_ADULT=$(wc -l < "${ADULT_LIST}" 2>/dev/null || echo 0)
echo "Child list: ${CHILD_LIST} (${N_CHILD})"
echo "Adult list: ${ADULT_LIST} (${N_ADULT})"
echo ""

if [[ ! -d "${INFER_DIR}" ]]; then
  echo "Inference dir not found: ${INFER_DIR}"
  exit 1
fi

if [[ ! -d "${CHILD_REFERENCE_DIR}" ]]; then
  echo "Child reference dir not found: ${CHILD_REFERENCE_DIR}"
  exit 1
fi
if [[ ! -d "${ADULT_REFERENCE_DIR}" ]]; then
  echo "Adult reference dir not found: ${ADULT_REFERENCE_DIR}"
  exit 1
fi

if [[ -s "${CHILD_LIST}" ]]; then
  echo "=== Selection: CHILD (FT checkpoint) ==="
  echo "  reference_dir: ${CHILD_REFERENCE_DIR}"
  cd "${INFER_DIR}"
  "${PYTHON}" inference_selective.py \
    --input_test_file "${CHILD_LIST}" \
    --checkpoint_file "${CHECKPOINT_FT}" \
    --output_dir "${ANON_ROOT}" \
    --reference_dir "${CHILD_REFERENCE_DIR}" \
    --input_root "${TSE_ROOT}" \
    --output_list "${ANON_ROOT}/selection_child_outputs.txt" \
    --min_duration 3.0 \
    --skip_existing
  cd - >/dev/null
  echo ""
fi

if [[ -s "${ADULT_LIST}" ]]; then
  echo "=== Selection: ADULT (base checkpoint) ==="
  echo "  reference_dir: ${ADULT_REFERENCE_DIR}"
  cd "${INFER_DIR}"
  "${PYTHON}" inference_selective.py \
    --input_test_file "${ADULT_LIST}" \
    --checkpoint_file "${CHECKPOINT_BASE}" \
    --output_dir "${ANON_ROOT}" \
    --reference_dir "${ADULT_REFERENCE_DIR}" \
    --input_root "${TSE_ROOT}" \
    --output_list "${ANON_ROOT}/selection_adult_outputs.txt" \
    --min_duration 3.0 \
    --skip_existing
  cd - >/dev/null
  echo ""
fi

echo "Renaming outputs to *.target_anon.wav..."
find "${ANON_ROOT}" -type f -name "*.wav" 2>/dev/null | while read -r f; do
  dir=$(dirname "$f")
  base=$(basename "$f" .wav)
  if [[ "${base}" == *".tse_hat" ]]; then
    newbase="${base%.tse_hat}"
    mv -n "$f" "${dir}/${newbase}.target_anon.wav"
  fi
done
echo "Selection TSA done. Outputs under ${ANON_ROOT}"

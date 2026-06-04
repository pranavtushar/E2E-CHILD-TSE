# Source from scripts:  source "$(dirname "$0")/env.sh"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
PIPE="${ROOT}/pipeline"
EVAL_SCRIPTS="${ROOT}/evaluation/scripts"
DATA_ROOT="${ROOT}/data"

export ROOT PIPE EVAL_SCRIPTS DATA_ROOT
export BASE_DIR="${BASE_DIR:-$ROOT}"
export DATA_SUBDIR="${DATA_SUBDIR:-data}"
export PY="${PY:-/root/miniconda3/envs/va_toolkit_py39/bin/python3}"
export SEED="${SEED:-42}"

export MIX_ROOT="${ROOT}/mixtures"
export TSE_ROOT="${ROOT}/outputs/tse"
export ANON_ROOT="${ROOT}/outputs/anon_selection"
export RECOMB_ROOT="${ROOT}/outputs/recombined_selection"
export MANIFEST_DIR="${ROOT}/outputs/manifests"
export TSA_ITEMS="${MANIFEST_DIR}/tsa_items.jsonl"
export TSA_ITEMS_SMOKE="${MANIFEST_DIR}/tsa_items_smoke.jsonl"
export EVAL_ROOT="${ROOT}/results"
export LOG_DIR="${ROOT}/logs"

export MYST_MANIFEST="${DATA_ROOT}/manifests/myst_test_manifest.jsonl"
export LIBRI_MANIFEST="${DATA_ROOT}/manifests/librispeech_test_clean_manifest.jsonl"
export LIBRI_AUDIO="${DATA_ROOT}/libri/test-clean-normalised"
export MYST_AUDIO="${DATA_ROOT}/myst/myST_test_filtered"
export CATALOG="${CATALOG:-${DATA_ROOT}/manifests/mixture_mfa_catalog_docker.csv}"

if [[ -d "${CHILD_REFERENCE_DIR:-}" ]] && [[ -n "$(ls -A "${CHILD_REFERENCE_DIR}" 2>/dev/null)" ]]; then
  :
elif [[ -d "${DATA_ROOT}/myst/development" ]] && [[ -n "$(ls -A "${DATA_ROOT}/myst/development" 2>/dev/null)" ]]; then
  CHILD_REFERENCE_DIR="${DATA_ROOT}/myst/development"
elif [[ -d "${DATA_ROOT}/myst/ai_kids_processed" ]]; then
  CHILD_REFERENCE_DIR="${DATA_ROOT}/myst/ai_kids_processed"
else
  CHILD_REFERENCE_DIR="${MYST_AUDIO}"
  export CHILD_REFERENCE_FALLBACK=1
fi
export CHILD_REFERENCE_DIR

if [[ ! -d "${ADULT_REFERENCE_DIR:-}" ]]; then
  ADULT_REFERENCE_DIR="${LIBRI_AUDIO}"
fi
export ADULT_REFERENCE_DIR

# All runtime assets live under ROOT (no paths outside E2E_CHILD_TSE_2).
export VENDOR_DIR="${ROOT}/vendor"
export TSE_DIR="${TSE_DIR:-${VENDOR_DIR}/tse}"
export TSE_CKPT="${TSE_CKPT:-${TSE_DIR}/libri2talker_libri2vox}"
export ECAPA_CKPT="${ECAPA_CKPT:-${TSE_DIR}/embedding_model.ckpt}"
export INFER_DIR="${INFER_DIR:-${VENDOR_DIR}/selection_inference}"
export ECAPA_SAVEDIR="${ECAPA_SAVEDIR:-${VENDOR_DIR}/speechbrain/spkrec-ecapa-voxceleb}"

export SMOKE_PER_SUBSET="${SMOKE_PER_SUBSET:-10}"

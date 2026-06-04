#!/usr/bin/env bash
# End-to-end child TSE benchmark (900 items): mixtures → TSE → anon → EER/WER.
#
#   cd /path/to/E2E_CHILD_TSE_2 && bash scripts/download_from_hf.sh  # once (or install_self_contained.sh)
#   bash run.sh
#
# Options:
#   SMOKE=1        30 items only (or use run_smoke.sh)
#   SKIP_CLEAN=1   keep mixtures; only wipe outputs/
#   SKIP_MIX=1     skip mixture generation (use with SKIP_CLEAN=1)
#   EER_ONLY=1 / WER_ONLY=1   evaluation subset
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT}/scripts"
# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

export PY="${PY:-/root/miniconda3/envs/va_toolkit_py39/bin/python3}"
unset ADULT_REFERENCE_DIR PYTHON ROOT_APP

SKIP_CLEAN="${SKIP_CLEAN:-0}"
SKIP_MIX="${SKIP_MIX:-0}"
SMOKE="${SMOKE:-0}"
MODE="full"
[[ "$SMOKE" == "1" ]] && MODE="smoke"

echo "=============================================="
echo " E2E Child TSE — mode=$MODE (900 if full, ~30 if smoke)"
echo " ROOT=$ROOT"
echo "=============================================="

"$PY" -c "import openpyxl" 2>/dev/null || "$PY" -m pip install -q openpyxl

if [[ "$SKIP_CLEAN" != "1" ]]; then
  bash "${SCRIPT_DIR}/clean_all.sh" || {
    sudo rm -rf "${MIX_ROOT}" "${ROOT}/outputs" "${ROOT}/results" "${LOG_DIR}"
    mkdir -p "${ROOT}/outputs/manifests" "${EVAL_ROOT}" "${LOG_DIR}"
  }
else
  bash "${SCRIPT_DIR}/clean_outputs.sh"
fi

if [[ "$SKIP_MIX" != "1" ]]; then
  FORCE=1 bash "${SCRIPT_DIR}/create_mixtures_from_catalog.sh"
else
  "$PY" "${SCRIPT_DIR}/verify_catalog_mixtures.py" \
    --catalog "$CATALOG" --mix-root "$MIX_ROOT" --base-dir "$ROOT"
fi

bash "${SCRIPT_DIR}/run_tse_anon_recomb.sh" "$MODE"

if [[ "${EER_ONLY:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_eval.sh" "$MODE" eer
elif [[ "${WER_ONLY:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_eval.sh" "$MODE" wer
else
  bash "${SCRIPT_DIR}/run_eval.sh" "$MODE"
fi

EXPECT=900
[[ "$MODE" == "smoke" ]] && EXPECT=30
echo ""
echo ">>> Counts (expect ~${EXPECT} for mode=$MODE)"
echo "  mixtures:    $(find "$MIX_ROOT" -name '*.mix.wav' 2>/dev/null | wc -l)"
echo "  tse_hat:     $(find "$TSE_ROOT" -name '*.tse_hat.wav' 2>/dev/null | wc -l)"
echo "  target_anon: $(find "$ANON_ROOT" -name '*.target_anon.wav' 2>/dev/null | wc -l)"
echo "  mix_anon:    $(find "$RECOMB_ROOT" -name '*.mix_anon.wav' 2>/dev/null | wc -l)"
echo ""
echo "Results under ${ROOT}/results/"
echo "  EER: ${EVAL_ROOT}/eer/results/eer_summary.csv"
echo "  WER: ${EVAL_ROOT}/wer/s1_s2_whisper/s1_s2_wer_pack.xlsx"

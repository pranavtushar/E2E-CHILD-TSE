#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODE="${1:-full}"
WHAT="${2:-all}"

if [[ "$MODE" == "smoke" ]]; then
  ITEMS_MANIFEST="${TSA_ITEMS_SMOKE}"
else
  ITEMS_MANIFEST="$TSA_ITEMS"
fi

[[ -f "$ITEMS_MANIFEST" ]] || { echo "Missing $ITEMS_MANIFEST — run run_tse_anon_recomb.sh first"; exit 1; }

mkdir -p "$LOG_DIR" "$EVAL_ROOT"
EVAL_MANIFEST="${EVAL_ROOT}/eval_manifest.jsonl"
EER_DATA="${EVAL_ROOT}/eer/eer_data"
EER_RESULTS="${EVAL_ROOT}/eer/results"
WER_OUT="${EVAL_ROOT}/wer/s1_s2_whisper"

export PYTHONPATH="${PIPE}:${PYTHONPATH:-}"

if [[ "$WHAT" == "all" || "$WHAT" == "eer" || "$WHAT" == "wer" ]]; then
  "$PY" "${PIPE}/build_eval_manifest.py" \
    --manifest "$ITEMS_MANIFEST" \
    --base-dir "$ROOT" \
    --myst-manifest "$MYST_MANIFEST" \
    --libri-manifest "$LIBRI_MANIFEST" \
    --out-dir "$EVAL_ROOT" \
    --systems SELECTION \
    2>&1 | tee "${LOG_DIR}/eval_manifest_${MODE}.log"
fi

if [[ "$WHAT" == "all" || "$WHAT" == "eer" ]]; then
  rm -rf "$EER_DATA" "$EER_RESULTS"
  mkdir -p "$EER_DATA" "$EER_RESULTS"
  "$PY" "${EVAL_SCRIPTS}/eer/build_eer_protocol.py" \
    --eval-manifest "$EVAL_MANIFEST" \
    --mixed-root "$MIX_ROOT" \
    --myst-manifest "$MYST_MANIFEST" \
    --libri-manifest "$LIBRI_MANIFEST" \
    --base-dir "$ROOT" \
    --out-dir "$EER_DATA" \
    2>&1 | tee "${LOG_DIR}/eer_protocol_${MODE}.log"
  EER_DATA="$EER_DATA" OUT_RESULTS="$EER_RESULTS" BASE_DIR="$ROOT" \
    EER_DEVICE="${EER_DEVICE:-cuda:0}" \
    bash "${EVAL_SCRIPTS}/eer/run_eer_only.sh" \
    2>&1 | tee "${LOG_DIR}/eer_run_${MODE}.log"
fi

if [[ "$WHAT" == "all" || "$WHAT" == "wer" ]]; then
  MANIFEST="$ITEMS_MANIFEST" OUT_DIR="$WER_OUT" BASE_DIR="$ROOT" \
    PY="$PY" WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}" \
    WHISPER_MODEL="${WHISPER_MODEL:-large-v3}" \
    bash "${EVAL_SCRIPTS}/wer/run_wer_s1_s2_whisper.sh" "$MODE" \
    2>&1 | tee "${LOG_DIR}/wer_${MODE}.log"
fi

echo "EER: ${EER_RESULTS}/eer_summary.csv"
echo "WER: ${WER_OUT}/s1_s2_wer_pack.xlsx"

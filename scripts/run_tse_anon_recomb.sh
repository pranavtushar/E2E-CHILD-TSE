#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODE="${1:-full}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 smoke|full"
  exit 1
fi

mkdir -p "$LOG_DIR" "$TSE_ROOT" "$ANON_ROOT" "$RECOMB_ROOT" "$MANIFEST_DIR"

if [[ ! -d "$MIX_ROOT" ]] || [[ "$(find "$MIX_ROOT" -name '*.mix.wav' 2>/dev/null | wc -l)" -lt 1 ]]; then
  echo "Missing mixtures. Run: bash scripts/create_mixtures_from_catalog.sh"
  exit 1
fi

echo "=== TSE + SELECTION + recombine (mode=$MODE) ==="

TSE_EXTRA=()
ACTIVE_MANIFEST="$TSA_ITEMS"

if [[ "$MODE" == "smoke" ]]; then
  "$PY" "${PIPE}/build_smoke_plan.py" \
    --data-root "$MIX_ROOT" \
    --out-manifest "$TSA_ITEMS_SMOKE" \
    --per-subset "$SMOKE_PER_SUBSET"
  TSE_EXTRA=(--items-manifest "$TSA_ITEMS_SMOKE")
  ACTIVE_MANIFEST="$TSA_ITEMS_SMOKE"
fi

"$PY" -u "${PIPE}/run_tsa_tse.py" \
  --tsa-root "$MIX_ROOT" \
  --out-root "$TSE_ROOT" \
  --tse-dir "$TSE_DIR" \
  --ecapa "$ECAPA_CKPT" \
  --tse-checkpoint "$TSE_CKPT" \
  "${TSE_EXTRA[@]}" \
  2>&1 | tee "${LOG_DIR}/tse_${MODE}.log"

MANIFEST_ARGS=(
  --data-root "$MIX_ROOT"
  --tse-root "$TSE_ROOT"
  --anon-root-selection "$ANON_ROOT"
  --recombined-root-selection "$RECOMB_ROOT"
  --out-dir "$MANIFEST_DIR"
)
if [[ "$MODE" == "smoke" ]]; then
  MANIFEST_ARGS+=(--only-items-manifest "$TSA_ITEMS_SMOKE")
  "$PY" "${PIPE}/build_anon_manifest.py" "${MANIFEST_ARGS[@]}"
  mv -f "$TSA_ITEMS" "$TSA_ITEMS_SMOKE"
  ACTIVE_MANIFEST="$TSA_ITEMS_SMOKE"
else
  "$PY" "${PIPE}/build_anon_manifest.py" "${MANIFEST_ARGS[@]}"
fi

BASE_DIR="$ROOT" \
DATA_ROOT="$DATA_ROOT" \
MANIFEST="$ACTIVE_MANIFEST" \
LISTS_DIR="$MANIFEST_DIR" \
TSE_ROOT="$TSE_ROOT" \
ANON_ROOT="$ANON_ROOT" \
INFER_DIR="$INFER_DIR" \
PY="$PY" \
CHILD_REFERENCE_DIR="${CHILD_REFERENCE_DIR}" \
ADULT_REFERENCE_DIR="${LIBRI_AUDIO}" \
bash "${PIPE}/scripts/run_anon_selection_tsa.sh" full \
  2>&1 | tee "${LOG_DIR}/anon_${MODE}.log"

"$PY" "${PIPE}/recombine_tsa_mix.py" \
  --manifest "$ACTIVE_MANIFEST" \
  --mode selection \
  2>&1 | tee "${LOG_DIR}/recombine_${MODE}.log"

echo "=== Done ($MODE) ==="
echo "  tse_hat:     $(find "$TSE_ROOT" -name '*.tse_hat.wav' 2>/dev/null | wc -l)"
echo "  target_anon: $(find "$ANON_ROOT" -name '*.target_anon.wav' 2>/dev/null | wc -l)"
echo "  mix_anon:    $(find "$RECOMB_ROOT" -name '*.mix_anon.wav' 2>/dev/null | wc -l)"

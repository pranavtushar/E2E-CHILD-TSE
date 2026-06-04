#!/usr/bin/env bash
# Channel WER: s1 (MFA vs target_anon Whisper), s2 (MFA vs nontarget Whisper).
# Exports CSV + Excel (Detail + WER%% tables per subset x overlap).
#
# Usage:
#   ./run_wer_s1_s2_whisper.sh smoke    # all rows in tsa_items_smoke.jsonl (~30)
#   ./run_wer_s1_s2_whisper.sh full     # all 900
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="${BASE_DIR:?Set BASE_DIR — run via scripts/run_eval.sh or source scripts/env.sh}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/results/wer/s1_s2_whisper}"
CATALOG="${CATALOG:-${BASE_DIR}/data/manifests/mixture_mfa_catalog_docker.csv}"
# Use PY from scripts/env.sh — do not set PYTHON= to a model name (e.g. large-v3).
if [[ -n "${PYTHON:-}" && ! -x "${PYTHON}" ]]; then
  unset PYTHON
fi
PY="${PY:-${PYTHON:-python3}}"
[[ -x "$PY" ]] || PY=python3
DEVICE="${WHISPER_DEVICE:-cuda}"
BACKEND="${WHISPER_BACKEND:-auto}"
MODEL="${WHISPER_MODEL:-large-v3}"

mode="${1:-smoke}"
if [[ "$mode" == "smoke" ]]; then
  MANIFEST="${MANIFEST:-${BASE_DIR}/outputs/manifests/tsa_items_smoke.jsonl}"
  echo "=== s1/s2 WER smoke (all items in smoke manifest, WER per overlap) ==="
else
  MANIFEST="${MANIFEST:-${BASE_DIR}/outputs/manifests/tsa_items.jsonl}"
  echo "=== s1/s2 WER full (900) ==="
fi

"$PY" "${SCRIPT_DIR}/run_wer_s1_s2_whisper.py" \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --base-dir "$BASE_DIR" \
  --catalog "$CATALOG" \
  --backend "$BACKEND" \
  --model="${MODEL}" \
  --device="${DEVICE}" \
  --write-xlsx

echo ""
echo "Detail CSV:  $OUT_DIR/s1_s2_wer_detail.csv"
echo "Excel:       $OUT_DIR/s1_s2_wer_pack.xlsx  (Detail + WER_s1_pct + WER_s2_pct sheets)"
echo "Matrices:    $OUT_DIR/wer_s1_pct_by_subset_overlap.md"
echo "Summary:     $OUT_DIR/s1_s2_wer_summary.json"

#!/usr/bin/env bash
# EER calculation only: run run_eer.py on existing eer_data (no protocol build).
# Use when eer_data/ is already built and you only want to recompute EER (OO + OA).
# Inside Docker: cd /app/tse/tsa_mix/evaluations/eer && BASE_DIR=/app bash run_eer_only.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_DIR="${BASE_DIR:-$BUNDLE_ROOT}"
EER_DATA="${EER_DATA:-$SCRIPT_DIR/eer_data}"
OUT_RESULTS="${OUT_RESULTS:-$SCRIPT_DIR/results}"
DEVICE="${EER_DEVICE:-cuda:0}"
T_NORM="${T_NORM:-0}"
PY="${PY:-python3}"
ECAPA_SAVEDIR="${ECAPA_SAVEDIR:-${BUNDLE_ROOT}/vendor/speechbrain/spkrec-ecapa-voxceleb}"

if [[ ! -f "$EER_DATA/enroll_orig/wav.scp" ]]; then
  echo "EER data not found at $EER_DATA. Build protocol first: BASE_DIR=$BASE_DIR bash run_all_eer.sh" >&2
  exit 1
fi

echo "=== EER calculation only (existing protocol) ==="
echo "eer_data=$EER_DATA out=$OUT_RESULTS device=$DEVICE"
RUN_OPTS="--eer-data $EER_DATA --device $DEVICE --out-dir $OUT_RESULTS"
[[ "$T_NORM" == "1" ]] && RUN_OPTS="$RUN_OPTS --t-norm"
"$PY" run_eer.py $RUN_OPTS

echo "Done. Results: $OUT_RESULTS/eer_Step2_OO.json $OUT_RESULTS/eer_Step2_OA.json $OUT_RESULTS/eer_summary.csv"

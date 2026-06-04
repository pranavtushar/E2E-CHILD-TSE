#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

TIMING_JSON_ROOT="${TIMING_JSON_ROOT:-}"
mkdir -p "$LOG_DIR" "$MIX_ROOT"

echo "=== Create mixtures from catalog ==="
echo "  ROOT=$ROOT"
echo "  catalog=$CATALOG"
echo "  out=$MIX_ROOT"

for need in "$CATALOG" "$LIBRI_AUDIO" "$MYST_AUDIO"; do
  [[ -e "$need" ]] || { echo "Missing: $need"; exit 1; }
done

if [[ "${FORCE:-0}" != "1" ]] && [[ -d "$MIX_ROOT" ]] && [[ -n "$(find "$MIX_ROOT" -name '*.mix.wav' 2>/dev/null | head -1)" ]]; then
  echo "Mixtures exist — skip (FORCE=1 or clean_mixtures.sh to regenerate)."
  find "$MIX_ROOT" -name '*.mix.wav' | wc -l
  exit 0
fi

rm -rf "$MIX_ROOT"
mkdir -p "$MIX_ROOT"

EXTRA=()
if [[ -n "$TIMING_JSON_ROOT" && -d "$TIMING_JSON_ROOT" ]]; then
  EXTRA=(--timing-json-root "$TIMING_JSON_ROOT")
fi

"$PY" "${PIPE}/create_tsa_mixtures_from_catalog.py" \
  --catalog "$CATALOG" \
  --base-dir "$ROOT" \
  --out-dir "$MIX_ROOT" \
  "${EXTRA[@]}" \
  2>&1 | tee "${LOG_DIR}/create_mixtures.log"

N=$(find "$MIX_ROOT" -name '*.mix.wav' | wc -l)
echo "Done. mix.wav count: $N (expect 900)"

"$PY" "${SCRIPT_DIR}/verify_catalog_mixtures.py" \
  --catalog "$CATALOG" --mix-root "$MIX_ROOT" --base-dir "$ROOT"
"$PY" "${SCRIPT_DIR}/check_mixture_files.py" \
  --mix-root "$MIX_ROOT" --base-dir "$ROOT" --check-ref

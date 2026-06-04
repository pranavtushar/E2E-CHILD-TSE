#!/usr/bin/env bash
# Quick checks before run.sh / run_smoke.sh. Exit 0 = ready.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

OK=0
FAIL=0
check() {
  if eval "$2" >/dev/null 2>&1; then
    echo "  OK   $1"
    OK=$((OK + 1))
  else
    echo "  FAIL $1"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== E2E_CHILD_TSE_2 preflight ==="
echo "ROOT=$ROOT (all paths must stay inside this tree)"
echo ""

echo "Data + catalog"
check "data/ exists" "[[ -e '$DATA_ROOT' ]]"
check "catalog CSV" "[[ -f '$CATALOG' ]]"
check "Libri audio" "[[ -d '$LIBRI_AUDIO' ]]"
check "MyST audio" "[[ -d '$MYST_AUDIO' ]]"
check "myst manifest" "[[ -f '$MYST_MANIFEST' ]]"
check "libri manifest" "[[ -f '$LIBRI_MANIFEST' ]]"
echo ""
echo "Bundled models (vendor/)"
check "vendor/tse" "[[ -d '${TSE_DIR}/later' ]]"
check "TSE checkpoint" "[[ -d '$TSE_CKPT' || -f '$TSE_CKPT' ]]"
check "ECAPA ckpt" "[[ -f '$ECAPA_CKPT' ]]"
check "anon inference" "[[ -f '${INFER_DIR}/inference_selective.py' ]]"
check "anon ECAPA module" "[[ -f '${INFER_DIR}/adapted_from_speechbrain/ecapa_tdnn_sb.py' ]]"
check "EER ECAPA (vendor/speechbrain)" "[[ -d '${ECAPA_SAVEDIR}' ]]"
echo ""
echo "Python"
check "PY executable" "[[ -x '$PY' ]]"
check "soundfile" "'$PY' -c 'import soundfile'"
check "torch" "'$PY' -c 'import torch'"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
  echo "FAILED ($FAIL checks). Fix above before bash run.sh"
  exit 1
fi
echo "All $OK checks passed. Run: bash run.sh  or  bash run_smoke.sh"
exit 0

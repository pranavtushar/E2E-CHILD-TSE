#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
echo "Removing outputs/ and results/ ..."
rm -rf "${ROOT}/outputs" "${ROOT}/results"
mkdir -p "${ROOT}/outputs" "${ROOT}/outputs/manifests" "${EVAL_ROOT}" "${LOG_DIR}"
echo "Done. Mixtures unchanged at ${MIX_ROOT}"

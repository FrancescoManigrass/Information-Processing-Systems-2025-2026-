#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[GPU-TEST] Virtualenv not found: ${PYTHON_BIN}" >&2
  echo "[GPU-TEST] Run main_custom.py once so it can create the same .venv, then retry." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
echo "[GPU-TEST] Using Python: $(command -v python)"
echo "[GPU-TEST] VIRTUAL_ENV: ${VIRTUAL_ENV:-}"

exec python "${SCRIPT_DIR}/tools/gpu_diagnostics.py" "$@"

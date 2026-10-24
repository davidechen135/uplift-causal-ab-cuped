#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Locate python venv
if [ -f "${PROJECT_DIR}/../.venv/bin/python" ]; then
    VENV_PYTHON="${PROJECT_DIR}/../.venv/bin/python"
    VENV_PYTEST="${PROJECT_DIR}/../.venv/bin/pytest"
    SKLEARN_DYLIBS="${PROJECT_DIR}/../.venv/lib/python3.12/site-packages/sklearn/.dylibs"
elif [ -f "${PROJECT_DIR}/../../.venv/bin/python" ]; then
    VENV_PYTHON="${PROJECT_DIR}/../../.venv/bin/python"
    VENV_PYTEST="${PROJECT_DIR}/../../.venv/bin/pytest"
    SKLEARN_DYLIBS="${PROJECT_DIR}/../../.venv/lib/python3.12/site-packages/sklearn/.dylibs"
else
    VENV_PYTHON="$(which python3)"
    VENV_PYTEST="$(which pytest)"
    SKLEARN_DYLIBS=""
fi

echo "=== Running: Uplift Modeling & A/B Testing with CUPED ==="
cd "${PROJECT_DIR}"

export DYLD_FALLBACK_LIBRARY_PATH="${SKLEARN_DYLIBS}:${DYLD_FALLBACK_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "[1/2] Running Unit & Statistical Tests..."
"${VENV_PYTEST}" tests -v

echo "[2/2] Running Experiments & Benchmarks..."
"${VENV_PYTHON}" src/experiments.py

echo "=== Uplift Modeling & A/B Testing Completed Successfully! ==="

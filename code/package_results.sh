#!/usr/bin/env bash
# Package benchmark CSV tables, predictions, logs, and summary digests into archives.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "${SCRIPT_DIR}")" == "code" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    CODE_DIR="${SCRIPT_DIR}"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
    CODE_DIR="${SCRIPT_DIR}/code"
fi
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_PREFIX="scPathProto_results_bundle"

"${PYTHON_BIN}" "${CODE_DIR}/real_data/run_benchmark.py" --summarize_only --out_dir "${PROJECT_ROOT}/results/real_benchmark" || true

"${PYTHON_BIN}" "${CODE_DIR}/package_results.py" \
    --results_dir "results/real_benchmark" \
    --processed_dir "data/real/processed" \
    --out_prefix "${OUT_PREFIX}"

cp -f "${OUT_PREFIX}.zip" "${OUT_PREFIX}_${TIMESTAMP}.zip"

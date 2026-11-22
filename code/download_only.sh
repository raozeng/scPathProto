#!/usr/bin/env bash
# Download and build the four benchmark datasets (Zenodo 3357167, GSE109774, GSE145926)
# into data/real/processed/{pancreas,pbmc,tabula_muris,lung_covid}.h5ad.
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

PROC_DIR="${PROJECT_ROOT}/data/real/processed"
LOG_DIR="${PROJECT_ROOT}/results/real_benchmark/logs"
mkdir -p "${PROC_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/download_real_data.log"

echo "======================================================================"
echo "[download_only.sh] Checking/downloading 4 real scRNA-seq benchmarks..."
echo "Project root : ${PROJECT_ROOT}"
echo "Python       : ${PYTHON_BIN}"
echo "Processed dir: ${PROC_DIR}"
echo "======================================================================"

NEED_ZENODO=0
for ds in pancreas pbmc; do
    if [[ ! -f "${PROC_DIR}/${ds}.h5ad" ]]; then
        NEED_ZENODO=1
    fi
done

if [[ "${NEED_ZENODO}" -eq 1 ]]; then
    echo "[1/3] Downloading Zenodo 3357167 benchmarks (pancreas, pbmc)..."
    "${PYTHON_BIN}" "${CODE_DIR}/real_data/download_benchmarks.py" --datasets pancreas pbmc 2>&1 | tee -a "${LOG_FILE}"
    "${PYTHON_BIN}" "${CODE_DIR}/real_data/loaders.py" --datasets pancreas pbmc 2>&1 | tee -a "${LOG_FILE}"
else
    echo "[1/3] pancreas.h5ad and pbmc.h5ad already exist, skipping download."
fi

for ds in lung_covid tabula_muris; do
    if [[ ! -f "${PROC_DIR}/${ds}.h5ad" ]]; then
        echo "[2/3] Downloading & building ${ds} from NCBI GEO..."
        "${PYTHON_BIN}" "${CODE_DIR}/real_data/download_geo.py" --datasets "${ds}" --threads 4 2>&1 | tee -a "${LOG_FILE}"
        "${PYTHON_BIN}" "${CODE_DIR}/real_data/loaders.py" --datasets "${ds}" 2>&1 | tee -a "${LOG_FILE}"
    else
        echo "[2/3] ${ds}.h5ad already exists, skipping download."
    fi
done

echo "======================================================================"
echo "[download_only.sh] All 4 real datasets ready in ${PROC_DIR}:"
ls -lh "${PROC_DIR}"/*.h5ad
echo "======================================================================"

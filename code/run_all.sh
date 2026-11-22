#!/usr/bin/env bash
# Batch runner for scPathProto and baseline benchmarks on a Linux server.
# Supports background execution (`bash run_all.sh start`), resumable caching of
# finished (task, method) pairs, automatic GPU detection for scANVI/TOSICA, and
# result packaging on completion.
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

OUT_DIR="${PROJECT_ROOT}/results/real_benchmark"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/run_all_master.log"
PID_FILE="${LOG_DIR}/run_all.pid"
N_JOBS="${N_JOBS:-8}"
MAX_FOLDS="${MAX_FOLDS:-5}"

ACTION="${1:-start}"

run_pipeline_worker() {
    echo "======================================================================"
    echo "[scPathProto Real-Data Benchmark] Started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Hostname     : $(hostname)"
    echo "Project Root : ${PROJECT_ROOT}"
    echo "Python       : ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"
    echo "Output Dir   : ${OUT_DIR}"
    echo "======================================================================"

    echo "[Step 1/5] Ensuring all 4 real benchmark .h5ad datasets are ready..."
    PYTHON_BIN="${PYTHON_BIN}" bash "${CODE_DIR}/download_only.sh"

    echo "[Step 2/5] Stage A: Running scPathProto (+ 5 Ablations) & Fast Official Baselines (SVM, CellTypist, SingleR) on Cross-Platform & Open-Set tasks..."
    "${PYTHON_BIN}" "${CODE_DIR}/real_data/run_benchmark.py" \
        --datasets pancreas pbmc tabula_muris lung_covid \
        --settings cross openset \
        --methods scPathProto SVM CellTypist SingleR \
        --include_ablation \
        --n_jobs "${N_JOBS}" \
        --out_dir "${OUT_DIR}"

    echo "[Step 3/5] Stage B: Running Deep Official Baselines (TOSICA, scANVI) on Cross-Platform & Open-Set tasks..."
    "${PYTHON_BIN}" "${CODE_DIR}/real_data/run_benchmark.py" \
        --datasets pancreas pbmc tabula_muris lung_covid \
        --settings cross openset \
        --methods TOSICA scANVI \
        --accelerator auto \
        --n_jobs "${N_JOBS}" \
        --out_dir "${OUT_DIR}"

    echo "[Step 4/5] Stage C: Running Intra-Dataset ${MAX_FOLDS}-Fold Cross-Validation across all 4 datasets..."
    "${PYTHON_BIN}" "${CODE_DIR}/real_data/run_benchmark.py" \
        --datasets pancreas pbmc tabula_muris lung_covid \
        --settings intra \
        --methods scPathProto SVM CellTypist SingleR TOSICA scANVI \
        --include_ablation \
        --max_folds "${MAX_FOLDS}" \
        --accelerator auto \
        --n_jobs "${N_JOBS}" \
        --out_dir "${OUT_DIR}"

    echo "[Step 5/5] Packaging all real-benchmark CSV tables and digests for download..."
    PYTHON_BIN="${PYTHON_BIN}" bash "${CODE_DIR}/package_results.sh"

    echo "======================================================================"
    echo "[scPathProto Real-Data Benchmark] ALL STAGES COMPLETED at $(date '+%Y-%m-%d %H:%M:%S')!"
    echo "======================================================================"
}

case "${ACTION}" in
    --worker)
        run_pipeline_worker
        ;;

    --fg|fg|foreground)
        echo "[run_all.sh] Running full real-data benchmark in FOREGROUND mode..."
        run_pipeline_worker 2>&1 | tee -a "${MASTER_LOG}"
        ;;

    start|"")
        if [[ -f "${PID_FILE}" ]]; then
            OLD_PID="$(cat "${PID_FILE}")"
            if kill -0 "${OLD_PID}" 2>/dev/null; then
                echo "[run_all.sh] Background task is ALREADY running (PID: ${OLD_PID})."
                echo "Use 'bash ${BASH_SOURCE[0]} monitor' or 'tail -f ${MASTER_LOG}' to view progress."
                exit 0
            fi
        fi

        echo "======================================================================"
        echo "[run_all.sh] Launching real-data benchmark in background (nohup)..."
        nohup bash "${BASH_SOURCE[0]}" --worker > "${MASTER_LOG}" 2>&1 &
        BG_PID=$!
        disown "${BG_PID}" 2>/dev/null || true
        echo "${BG_PID}" > "${PID_FILE}"

        echo "  -> Background PID : ${BG_PID}"
        echo "  -> Master Log File: ${MASTER_LOG}"
        echo "  -> Safe to exit SSH session; execution will continue in background."
        echo ""
        echo "Useful commands:"
        echo "  1. Monitor live log : bash ${BASH_SOURCE[0]} monitor"
        echo "  2. Check job status : bash ${BASH_SOURCE[0]} status"
        echo "  3. Stop background  : bash ${BASH_SOURCE[0]} stop"
        echo "======================================================================"
        ;;

    status)
        if [[ -f "${PID_FILE}" ]]; then
            PID="$(cat "${PID_FILE}")"
            if kill -0 "${PID}" 2>/dev/null; then
                echo "[STATUS] Real-data benchmark is RUNNING (PID: ${PID})."
            else
                echo "[STATUS] Real-data benchmark process (PID: ${PID}) has FINISHED."
            fi
        else
            echo "[STATUS] No PID file found."
        fi
        if [[ -f "${MASTER_LOG}" ]]; then
            echo "--- Latest 25 lines of ${MASTER_LOG} ---"
            tail -n 25 "${MASTER_LOG}"
        fi
        ;;

    monitor|log|logs)
        echo "[MONITOR] Following live log ${MASTER_LOG} (Press Ctrl+C to exit monitor; job keeps running)..."
        touch "${MASTER_LOG}"
        tail -f "${MASTER_LOG}"
        ;;

    stop|kill)
        if [[ -f "${PID_FILE}" ]]; then
            PID="$(cat "${PID_FILE}")"
            if kill -0 "${PID}" 2>/dev/null; then
                echo "[STOP] Terminating background process PID ${PID}..."
                kill -TERM "${PID}" 2>/dev/null || true
            else
                echo "[STOP] Process ${PID} is not running."
            fi
            rm -f "${PID_FILE}"
        else
            echo "[STOP] No active PID file found."
        fi
        ;;

    *)
        echo "Usage: bash run_all.sh [start | --fg | status | monitor | stop]"
        exit 1
        ;;
esac

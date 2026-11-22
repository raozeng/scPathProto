# Server Execution Guide

This guide covers running the full 4-atlas, 126-task benchmark on a remote Linux server.

## 1. Environment Setup

Clone the repository (or unpack the source archive) on the server and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps --force-reinstall ./code/third_party/TOSICA
```

## 2. Background Benchmark Execution

Use `code/run_all.sh` to download/build any missing `.h5ad` datasets, run all cross-platform, open-set, and 5-fold intra-dataset tasks, and package the resulting CSV tables:

```bash
# Start full benchmark in background (survives SSH disconnect)
bash code/run_all.sh start

# Follow live progress log (Ctrl+C detaches without stopping the job)
bash code/run_all.sh monitor

# Check status or stop the background process
bash code/run_all.sh status
bash code/run_all.sh stop
```

To adjust parallelism or run in the foreground:

```bash
N_JOBS=16 bash code/run_all.sh --fg
```

Completed `(task, method)` predictions under `results/real_benchmark/predictions/` are cached automatically, so interrupted runs resume cleanly.

## 3. Syncing Results Back to a Local Machine

When the pipeline finishes, `code/package_results.sh` creates `scPathProto_results_bundle.tar.gz` (and `.zip`) in the project root. After downloading the archive locally, unpack and print the 4-dataset summaries with:

```bash
python3 code/apply_server_results.py scPathProto_results_bundle.tar.gz
```

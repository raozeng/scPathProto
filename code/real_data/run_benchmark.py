#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark runner for intra-dataset, cross-platform, and open-set evaluation.

Usage:
    python code/real_data/download_benchmarks.py
    python code/real_data/loaders.py
    python code/real_data/run_benchmark.py --datasets pancreas pbmc \
        --settings intra cross openset --methods SVM CellTypist SingleR scANVI TOSICA scPathProto

Outputs (results/real_benchmark/):
    tasks.csv                      task split definitions
    predictions/<task>__<method>.csv.gz   per-cell predictions (val + test cells)
    metrics_closed.csv / metrics_open.csv / per_class_f1.csv
    runtime.csv, errors.log
Completed (task, method) pairs are skipped on rerun unless --overwrite is set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

from baselines import ABLATION_METHODS, BASE_METHODS, FEATURE_POLICY, METHODS  # noqa: E402
from loaders import load_processed  # noqa: E402
from metrics import closed_set_metrics, open_set_metrics, per_class_f1, rare_classes_from_train  # noqa: E402
from preprocess import DATASET_GMT, log_normalize, select_hvgs  # noqa: E402
from tasks import all_tasks  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)


def run(args):
    out_dir = os.path.abspath(args.out_dir)
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    err_log = os.path.join(out_dir, "errors.log")

    for ds in args.datasets:
        adata = load_processed(ds)
        log_normalize(adata)
        tasks = all_tasks(adata, ds, settings=tuple(args.settings), seed=args.seed)
        if args.max_folds is not None:
            tasks = [t for t in tasks if t.fold is None or t.fold < args.max_folds]
        if args.task_filter:
            tasks = [t for t in tasks if re.search(args.task_filter, t.task_id)]
        tdf = pd.DataFrame([t.describe() for t in tasks])
        tpath = os.path.join(out_dir, "tasks.csv")
        if os.path.exists(tpath):
            tdf = pd.concat([pd.read_csv(tpath), tdf]).drop_duplicates("task_id", keep="last")
        tdf.to_csv(tpath, index=False)
        print(f"[{ds}] {len(tasks)} tasks", flush=True)

        for t in tasks:
            a_tr = adata[t.train_idx]
            a_va = adata[t.val_idx]
            a_te = adata[t.test_idx]
            eval_idx = np.concatenate([t.val_idx, t.test_idx])
            a_ev = adata[eval_idx]
            split = np.array(["val"] * len(t.val_idx) + ["test"] * len(t.test_idx))
            hvgs = None
            for mname in args.methods:
                fpath = os.path.join(pred_dir, f"{_safe(t.task_id)}__{mname}.csv.gz")
                if os.path.exists(fpath) and not args.overwrite:
                    continue
                if hvgs is None and any(FEATURE_POLICY[m] == "hvg" for m in args.methods):
                    hvgs = select_hvgs(a_tr, n_top=args.n_hvg, batch_key="batch")
                feats = hvgs if FEATURE_POLICY[mname] == "hvg" else None
                kw = dict(scvi_epochs=args.scvi_epochs, scanvi_epochs=args.scanvi_epochs,
                          query_epochs=args.query_epochs, tosica_epochs=args.tosica_epochs,
                          tosica_batch_size=args.tosica_batch_size, accelerator=args.accelerator,
                          gmt=DATASET_GMT[ds])
                print(f"  -> {t.task_id} | {mname} (train={len(t.train_idx)}, val={len(t.val_idx)}, test={len(t.test_idx)})", flush=True)
                t0 = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = METHODS[mname](seed=args.seed, n_jobs=args.n_jobs, **kw).set_features(feats)
                        model.fit(a_tr.copy(), t.label_key)
                        t_fit = time.time() - t0
                        # Predict on validation and test splits separately so reference validation
                        # cells and unseen query test cells never contaminate each other's batch statistics.
                        res_va = model.predict(a_va.copy())
                        res_te = model.predict(a_te.copy())
                    t_all = time.time() - t0
                except Exception:  # noqa: BLE001
                    with open(err_log, "a") as fh:
                        fh.write(f"\n=== {t.task_id} | {mname}\n{traceback.format_exc()}")
                    print(f"     FAILED (see {err_log})", flush=True)
                    continue
                pred_labels = np.concatenate([res_va["labels"], res_te["labels"]])
                pred_conf = np.concatenate([res_va["confidence"], res_te["confidence"]])
                df = pd.DataFrame({
                    "cell": a_ev.obs_names, "split": split,
                    "true": a_ev.obs[t.label_key].astype(str).to_numpy(),
                    "batch": a_ev.obs["batch"].astype(str).to_numpy(),
                    "pred": pred_labels, "confidence": pred_conf,
                })
                df.to_csv(fpath, index=False, compression="gzip")
                y_tr = a_tr.obs[t.label_key].astype(str)
                meta = dict(task_id=t.task_id, method=mname, fit_seconds=t_fit, total_seconds=t_all,
                            has_probability=res_te["proba"] is not None, n_features=len(feats) if feats else a_tr.n_vars,
                            train_classes=sorted(y_tr.unique().tolist()),
                            rare_classes=rare_classes_from_train(y_tr.to_numpy()))
                with open(fpath.replace(".csv.gz", ".json"), "w") as fh:
                    json.dump(meta, fh)
                print(f"     done in {t_all:.1f}s", flush=True)
    summarize(out_dir)


def summarize(out_dir: str):
    pred_dir = os.path.join(out_dir, "predictions")
    tpath = os.path.join(out_dir, "tasks.csv")
    if not os.path.exists(tpath) or not os.path.isdir(pred_dir):
        return
    tdf = pd.read_csv(tpath).set_index("task_id")
    rows_c, rows_o, rows_pc, rows_t = [], [], [], []
    for f in sorted(os.listdir(pred_dir)):
        if not f.endswith(".json"):
            continue
        meta = json.load(open(os.path.join(pred_dir, f)))
        if meta["task_id"] not in tdf.index:
            continue
        t = tdf.loc[meta["task_id"]]
        df = pd.read_csv(os.path.join(pred_dir, f.replace(".json", ".csv.gz")))
        rows_t.append(meta)
        te = df[df.split == "test"]
        va = df[df.split == "val"]
        base = dict(task_id=meta["task_id"], method=meta["method"], dataset=t.dataset, setting=t.setting,
                    train_batches=t.train_batches, test_batches=t.test_batches, fold=t.fold)
        heldout = [] if pd.isna(t.heldout_types) else str(t.heldout_types).split(";")
        train_classes = meta["train_classes"]
        rare = meta["rare_classes"]
        if t.setting in ("intra", "cross"):
            m = closed_set_metrics(te["true"], te["pred"], train_classes, rare)
            rows_c.append({**base, **m})
            pc = per_class_f1(te["true"], te["pred"], train_classes)
            pc = pc.assign(**base)
            rows_pc.append(pc)
        else:
            m = open_set_metrics(te["true"], te["pred"], te["confidence"], heldout, va["confidence"],
                                 has_probability=bool(meta.get("has_probability", True)))
            rows_o.append({**base, "heldout": ";".join(heldout), **m})
    if rows_c:
        dfc = pd.DataFrame(rows_c)
        dfc.to_csv(os.path.join(out_dir, "metrics_closed.csv"), index=False)
        num_cols = [c for c in ["accuracy", "macro_f1", "median_f1", "weighted_f1", "balanced_accuracy", "cohen_kappa", "rare_f1"] if c in dfc.columns]
        dfc.groupby(["dataset", "setting", "method"], as_index=False)[num_cols].mean().to_csv(
            os.path.join(out_dir, "summary_closed_by_dataset_method.csv"), index=False
        )
    if rows_o:
        dfo = pd.DataFrame(rows_o)
        dfo.to_csv(os.path.join(out_dir, "metrics_open.csv"), index=False)
        num_cols_o = [c for c in ["auroc", "auprc", "acc_known_closed", "acc_known", "novel_recall", "novel_precision", "novel_f1", "h_score", "ece_known"] if c in dfo.columns]
        dfo.groupby(["dataset", "setting", "method"], as_index=False)[num_cols_o].mean().to_csv(
            os.path.join(out_dir, "summary_open_by_dataset_method.csv"), index=False
        )
    if rows_pc:
        pd.concat(rows_pc).to_csv(os.path.join(out_dir, "per_class_f1.csv"), index=False)
    if rows_t:
        pd.DataFrame(rows_t).to_csv(os.path.join(out_dir, "runtime.csv"), index=False)
    print(f"[summary] written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pancreas", "pbmc", "tabula_muris", "lung_covid"])
    ap.add_argument("--settings", nargs="+", default=["intra", "cross", "openset"])
    ap.add_argument("--methods", nargs="+", default=BASE_METHODS, choices=list(METHODS))
    ap.add_argument("--include_ablation", action="store_true", help="also evaluate the 5 scPathProto ablation variants")
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "results", "real_benchmark"))
    ap.add_argument("--n_hvg", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--max_folds", type=int, default=None, help="limit CV folds (for quick runs)")
    ap.add_argument("--task_filter", default=None, help="regex on task_id")
    ap.add_argument("--scvi_epochs", type=int, default=None, help="None = scvi-tools default heuristic")
    ap.add_argument("--scanvi_epochs", type=int, default=20)
    ap.add_argument("--query_epochs", type=int, default=100)
    ap.add_argument("--tosica_epochs", type=int, default=10)
    ap.add_argument("--tosica_batch_size", type=int, default=64)
    ap.add_argument("--accelerator", default="auto", help="scvi-tools accelerator: auto, gpu, cpu, mps")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args()
    if args.include_ablation:
        for m in ABLATION_METHODS:
            if m not in args.methods:
                args.methods.append(m)
    if args.summarize_only:
        summarize(os.path.abspath(args.out_dir))
    else:
        run(args)

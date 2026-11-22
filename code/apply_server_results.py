#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_server_results.py
Unpack a result archive (`.tar.gz` or `.zip`) generated on a remote server into
the project root and print 4-dataset benchmark summaries.

Usage:
    python3 code/apply_server_results.py scPathProto_results_bundle.tar.gz
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def extract_bundle(archive_path: str) -> None:
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    with tempfile.TemporaryDirectory(prefix="scpathproto_sync_") as tmpdir:
        if archive_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(tmpdir)
        elif archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(tmpdir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

        src_root = os.path.join(tmpdir, "scPathProto_real_results")
        if not os.path.isdir(src_root):
            src_root = tmpdir

        copied = 0
        for dirpath, _, filenames in os.walk(src_root):
            for fn in filenames:
                src_f = os.path.join(dirpath, fn)
                rel = os.path.relpath(src_f, src_root)
                # Strip accidental absolute path prefix if tar stored absolute rels
                for marker in ("results/", "code/results/", "paper/figures/", "data/real/"):
                    if marker in rel and not rel.startswith(marker):
                        rel = rel[rel.index(marker):]
                        break
                dst_f = os.path.join(ROOT_DIR, rel)
                os.makedirs(os.path.dirname(dst_f), exist_ok=True)
                shutil.copy2(src_f, dst_f)
                copied += 1
        print(f"[Sync] Extracted and updated {copied} files into {ROOT_DIR}")

    # Copy any extracted PDF figures into figures/
    pfigs = os.path.join(ROOT_DIR, "paper", "figures")
    if os.path.isdir(pfigs):
        for fn in os.listdir(pfigs):
            if fn.endswith(".pdf"):
                for extra in ["figures", "code/results/figures"]:
                    os.makedirs(os.path.join(ROOT_DIR, extra), exist_ok=True)
                    shutil.copy2(os.path.join(pfigs, fn), os.path.join(ROOT_DIR, extra, fn))


def print_summary_report() -> None:
    tdir = os.path.join(ROOT_DIR, "code", "results", "tables")
    if not os.path.isdir(tdir):
        tdir = os.path.join(ROOT_DIR, "results", "tables")

    ab_p = os.path.join(tdir, "ablation_study.csv")
    if os.path.exists(ab_p):
        df_ab = pd.read_csv(ab_p)
        cols = [
            "Intra_Acc", "Intra_Macro_F1", "CrossBatch_Acc", "CrossBatch_Macro_F1",
            "CrossBatch_Rare_F1", "OpenSet_AUROC", "OpenSet_AUPRC", "OpenSet_H_Score",
        ]
        print("\n=== Updated 4-Dataset Mean Ablation Table (Table 5) ===")
        mean_ab = df_ab.groupby("Variant", sort=False)[cols].mean().round(4)
        print(mean_ab.to_string())

    rt_p = os.path.join(tdir, "runtime_summary_126tasks.csv")
    if os.path.exists(rt_p):
        print("\n=== Updated 126-Task Runtime Summary ===")
        print(pd.read_csv(rt_p).round(2).to_string(index=False))

    in_p = os.path.join(tdir, "perf_intra_dataset_summary.csv")
    if os.path.exists(in_p):
        df_in = pd.read_csv(in_p)
        print("\n=== Updated 4-Dataset Mean Intra-Dataset Summary (Table 2) ===")
        cols_in = ["Accuracy_Mean", "Macro_F1_Mean", "Median_F1_Mean", "Weighted_F1_Mean", "Kappa_Mean", "Rare_F1_Mean"]
        print(df_in.groupby("Method", sort=False)[cols_in].mean().round(4).to_string())

    cb_p = os.path.join(tdir, "perf_cross_batch_transfer.csv")
    if os.path.exists(cb_p):
        df_cb = pd.read_csv(cb_p)
        print("\n=== Updated 4-Dataset Mean Cross-Platform Summary (Table 3) ===")
        cols_cb = ["Accuracy", "Macro_F1", "Median_F1", "Weighted_F1", "Kappa", "Rare_F1"]
        print(df_cb.groupby("Method", sort=False)[cols_cb].mean().round(4).to_string())

    os_p = os.path.join(tdir, "perf_openset_discovery.csv")
    if os.path.exists(os_p):
        df_os = pd.read_csv(os_p)
        print("\n=== Updated 4-Dataset Mean Open-Set Summary (Table 4) ===")
        cols_os = ["ClosedSet_Acc_NoReject", "OpenSet_AUROC", "OpenSet_AUPRC", "Novel_F1", "H_Score"]
        print(df_os.groupby("Method", sort=False)[cols_os].mean().round(4).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_bundle(sys.argv[1])
    print_summary_report()

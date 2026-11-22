#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect and archive benchmark outputs (CSV tables, cell-type-by-batch counts,
logs, and Markdown summary digest) into `.zip` and `.tar.gz` bundles.
"""

import argparse
import glob
import os
import tarfile
import zipfile
from datetime import datetime
from typing import List, Tuple

import pandas as pd


def generate_paper_results_digest(results_dir: str, processed_dir: str) -> str:
    digest_path = os.path.join(results_dir, "PAPER_RESULTS_DIGEST.md")
    lines: List[str] = [
        "# Real-Benchmark Experimental Results Digest",
        "",
        f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
    ]

    for title, fname in [
        ("1. Closed-Set Summary by Dataset & Method (`summary_closed_by_dataset_method.csv`)", "summary_closed_by_dataset_method.csv"),
        ("2. Open-Set Summary by Dataset & Method (`summary_open_by_dataset_method.csv`)", "summary_open_by_dataset_method.csv"),
        ("3. Per-Task Closed-Set Metrics (`metrics_closed.csv`)", "metrics_closed.csv"),
        ("4. Per-Task Open-Set Metrics (`metrics_open.csv`)", "metrics_open.csv"),
    ]:
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            lines.extend([
                f"## {title}",
                "",
                "```",
                df.to_string(index=False),
                "```",
                "",
            ])

    with open(digest_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return digest_path


def create_results_archives(
    results_dir: str = "results/real_benchmark",
    processed_dir: str = "data/real/processed",
    out_prefix: str = "scPathProto_results_bundle",
) -> Tuple[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    generate_paper_results_digest(results_dir, processed_dir)

    files_to_pack: List[str] = []
    patterns = [
        os.path.join(results_dir, "*.csv"),
        os.path.join(results_dir, "*.md"),
        os.path.join(results_dir, "*.log"),
        os.path.join(results_dir, "logs", "*.log"),
        os.path.join(results_dir, "predictions", "*.csv.gz"),
        os.path.join(results_dir, "predictions", "*.json"),
        os.path.join(processed_dir, "*_celltype_by_batch.csv"),
        "results/tables/*.csv",
        "code/results/tables/*.csv",
        "paper/figures/*.pdf",
        "paper/figures/*.png",
    ]
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if os.path.isfile(p):
                files_to_pack.append(p)

    zip_path = f"{out_prefix}.zip"
    tar_path = f"{out_prefix}.tar.gz"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files_to_pack:
            zf.write(fpath, arcname=os.path.join("scPathProto_real_results", fpath))

    with tarfile.open(tar_path, "w:gz") as tf:
        for fpath in files_to_pack:
            tf.add(fpath, arcname=os.path.join("scPathProto_real_results", fpath))

    print(f"[Package Results] Packed {len(files_to_pack)} result files:")
    print(f"  -> ZIP Archive   : {os.path.abspath(zip_path)} ({os.path.getsize(zip_path)/1024:.1f} KB)")
    print(f"  -> TAR.GZ Archive: {os.path.abspath(tar_path)} ({os.path.getsize(tar_path)/1024:.1f} KB)")
    return zip_path, tar_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Package real-benchmark experimental results for download")
    ap.add_argument("--results_dir", type=str, default="results/real_benchmark")
    ap.add_argument("--processed_dir", type=str, default="data/real/processed")
    ap.add_argument("--out_prefix", type=str, default="scPathProto_results_bundle")
    args = ap.parse_args()
    create_results_archives(args.results_dir, args.processed_dir, args.out_prefix)

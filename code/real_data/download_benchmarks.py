#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the real, author-annotated benchmark datasets used in this study.

Source
------
Abdelaal T., Michielsen L., Cats D., et al. "A comparison of automatic cell
identification methods for single-cell RNA sequencing data." Genome Biology 20,
194 (2019). Zenodo record 10.5281/zenodo.3357167 (scRNAseq_Benchmark_datasets.zip).

This archive is the de-facto standard benchmark for supervised cell-type
annotation and redistributes the original studies together with the labels
provided by the original authors:

* Human pancreas (4 studies / 4 sequencing protocols)
    - Baron et al. 2016 (inDrop, GSE84133)
    - Muraro et al. 2016 (CEL-Seq2, GSE85241)
    - Segerstolpe et al. 2016 (Smart-Seq2, E-MTAB-5061)
    - Xin et al. 2016 (SMARTer, GSE81608)
* PbmcBench: Ding et al. 2020, Nature Biotechnology (SCP424 / GSE132044),
  PBMC sample "pbmc1" profiled with 7 protocols
  (10Xv2, 10Xv3, CEL-Seq2, Drop-Seq, inDrop, Seq-Well, Smart-Seq2).

Only the needed members are fetched (HTTP range requests), the full 3.7 GB
archive is never downloaded. Each member's CRC-32 is verified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from http_zip import HTTPRangeFile, parallel_extract_member  # noqa: E402

ZENODO_URL = "https://zenodo.org/api/records/3357167/files/scRNAseq_Benchmark_datasets.zip/content"

PANCREAS_MEMBERS = {
    "Baron": ("Intra-dataset/Pancreatic_data/Baron Human/Filtered_Baron_HumanPancreas_data.csv",
              "Intra-dataset/Pancreatic_data/Baron Human/Labels.csv"),
    "Muraro": ("Intra-dataset/Pancreatic_data/Muraro/Filtered_Muraro_HumanPancreas_data.csv",
               "Intra-dataset/Pancreatic_data/Muraro/Labels.csv"),
    "Segerstolpe": ("Intra-dataset/Pancreatic_data/Segerstolpe/Filtered_Segerstolpe_HumanPancreas_data.csv",
                    "Intra-dataset/Pancreatic_data/Segerstolpe/Labels.csv"),
    "Xin": ("Intra-dataset/Pancreatic_data/Xin/Filtered_Xin_HumanPancreas_data.csv",
            "Intra-dataset/Pancreatic_data/Xin/Labels.csv"),
}

# In the PbmcBench folders, each "<protocol>_pbmc1.csv" file is the concatenation of
# ALL protocols (training protocol + every test protocol); the folder name only
# indicates which protocol Abdelaal et al. used for training, and cell populations
# absent from that training protocol were removed. The 10Xv2 folder keeps the
# largest label set (9 populations) and is therefore sufficient for every
# train-protocol -> test-protocol combination used here. Cell-to-protocol indices
# are given in Statisitics.xlsx (sic, original file name).
PBMC_MEMBERS = {
    "10Xv2_folder": ("Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1.csv", "Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1Labels.csv"),
}

DOC_MEMBERS = [
    "Inter-dataset/PbmcBench/ReadME.txt",
    "Inter-dataset/PbmcBench/Statisitics.xlsx",
    "Inter-dataset/Pancreatic/ReadME.txt",
]

DATASETS = {"pancreas": PANCREAS_MEMBERS, "pbmc": PBMC_MEMBERS}


def _load_index(raw_dir: str) -> dict:
    """Read (and cache) the archive's central directory; listing it remotely is slow."""
    idx_path = os.path.join(raw_dir, "zip_index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    print("Reading remote ZIP central directory (one-time)...", file=sys.stderr)
    zf = zipfile.ZipFile(HTTPRangeFile(ZENODO_URL, block_size=4 << 20))
    index = {
        zi.filename: dict(
            header_offset=zi.header_offset,
            compress_size=zi.compress_size,
            file_size=zi.file_size,
            CRC=zi.CRC,
            compress_type=zi.compress_type,
        )
        for zi in zf.infolist()
    }
    os.makedirs(raw_dir, exist_ok=True)
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    return index


def _zinfo(name: str, meta: dict) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name)
    zi.header_offset = meta["header_offset"]
    zi.compress_size = meta["compress_size"]
    zi.file_size = meta["file_size"]
    zi.CRC = meta["CRC"]
    zi.compress_type = meta["compress_type"]
    return zi


def local_path(raw_dir: str, member: str) -> str:
    return os.path.join(raw_dir, *member.split("/"))


def download(datasets, raw_dir: str, n_threads: int = 24) -> None:
    index = _load_index(raw_dir)
    members = list(DOC_MEMBERS)
    for ds in datasets:
        for data_m, label_m in DATASETS[ds].values():
            members += [label_m, data_m]  # small label files first
    for m in members:
        if m not in index:
            raise KeyError(f"Member not found in archive: {m}")
        out = local_path(raw_dir, m)
        print(f"[download] {m}", file=sys.stderr)
        parallel_extract_member(ZENODO_URL, _zinfo(m, index[m]), out, n_threads=n_threads)
    print("[download] all requested members verified (CRC-32 OK).", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["pancreas", "pbmc"], choices=sorted(DATASETS))
    ap.add_argument("--raw_dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "real", "raw", "abdelaal2019"))
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    download(args.datasets, os.path.abspath(args.raw_dir), n_threads=args.threads)


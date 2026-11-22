#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download raw count matrices and author annotations for Tabula Muris and Lung COVID-19.

Tabula Muris (Tabula Muris Consortium, Nature 2018; GEO GSE109774)
    * FACS Smart-seq2: GSE109774_<Tissue>.tar.gz (12 tissues shared with Droplet)
    * Droplet 10x v2: GSE109774_RAW.tar
    * Annotations: github.com/czbiohub-sf/tabula-muris (00_data_ingest/18_global_annotation_csv/)

Lung COVID-19 BALF (Liao et al., Nature Medicine 2020; GEO GSE145926)
    * 12 BALF samples (10x 5'), filtered_feature_bc_matrix.h5 from GSE145926_RAW.tar
    * Healthy control HC4: GSM3660650 (Morse et al., Eur Respir J 2019; GSE128033)
    * Annotations: github.com/zhangzlab/covid_balf (all.cell.annotation.meta.txt)
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from http_zip import parallel_download  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "real", "raw"))
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"

TM_FACS_TISSUES = ["Bladder", "Heart", "Kidney", "Liver", "Lung", "Mammary", "Marrow",
                   "Muscle", "Spleen", "Thymus", "Tongue", "Trachea"]
TM_FILES = (
    [(f"{GEO}/series/GSE109nnn/GSE109774/suppl/GSE109774_{t}.tar.gz", f"facs/GSE109774_{t}.tar.gz")
     for t in TM_FACS_TISSUES]
    + [(f"{GEO}/series/GSE109nnn/GSE109774/suppl/GSE109774_RAW.tar", "droplet/GSE109774_RAW.tar")]
)
TM_GITHUB = "https://raw.githubusercontent.com/czbiohub-sf/tabula-muris/master/00_data_ingest/18_global_annotation_csv"
TM_ANNOT = [(f"{TM_GITHUB}/annotations_facs.csv", "annotations_facs.csv"),
            (f"{TM_GITHUB}/annotations_droplet.csv", "annotations_droplet.csv")]

LIAO_FILES = [
    (f"{GEO}/series/GSE145nnn/GSE145926/suppl/GSE145926_RAW.tar", "GSE145926_RAW.tar"),
    (f"{GEO}/samples/GSM3660nnn/GSM3660650/suppl/GSM3660650_SC249NORbal_fresh_barcodes.tsv.gz", "GSM3660650/barcodes.tsv.gz"),
    (f"{GEO}/samples/GSM3660nnn/GSM3660650/suppl/GSM3660650_SC249NORbal_fresh_genes.tsv.gz", "GSM3660650/genes.tsv.gz"),
    (f"{GEO}/samples/GSM3660nnn/GSM3660650/suppl/GSM3660650_SC249NORbal_fresh_matrix.mtx.gz", "GSM3660650/matrix.mtx.gz"),
]
LIAO_GITHUB = "https://raw.githubusercontent.com/zhangzlab/covid_balf/master"
LIAO_ANNOT = [(f"{LIAO_GITHUB}/all.cell.annotation.meta.txt", "all.cell.annotation.meta.txt"),
              (f"{LIAO_GITHUB}/meta.txt", "meta.txt")]


def _small(url: str, out: str):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    urllib.request.urlretrieve(url, out)


def download(datasets, threads: int):
    if "tabula_muris" in datasets:
        d = os.path.join(RAW, "tabula_muris")
        for url, rel in TM_ANNOT:
            _small(url, os.path.join(d, rel))
        for url, rel in TM_FILES:
            print(f"[tabula_muris] {rel}", file=sys.stderr)
            parallel_download(url, os.path.join(d, rel), n_threads=threads)
    if "lung_covid" in datasets:
        d = os.path.join(RAW, "liao2020")
        for url, rel in LIAO_ANNOT:
            _small(url, os.path.join(d, rel))
        for url, rel in LIAO_FILES:
            print(f"[lung_covid] {rel}", file=sys.stderr)
            parallel_download(url, os.path.join(d, rel), n_threads=threads)
    print("[download] done", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["lung_covid", "tabula_muris"])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    download(a.datasets, a.threads)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build standardized AnnData (.h5ad) files from raw benchmark datasets
(Abdelaal et al., Genome Biology 2019; Tabula Muris; Liao et al., 2020).

Processing summary:
* Preserves original count matrices in both ``X`` and ``layers['counts']``.
* Retains original author annotations in ``obs['cell_type_original']`` and
  harmonizes synonymous Pancreas labels in ``obs['cell_type']`` via
  ``PANCREAS_LABEL_MAP``.
* Records original study and sequencing protocol in ``obs['batch']``,
  ``obs['protocol']``, and ``obs['study']``.

Outputs ``data/real/processed/<dataset>.h5ad`` with:
    X                 raw counts (float32, CSR)
    layers['counts']  same raw counts
    obs['cell_type']  harmonized author label
    obs['cell_type_original'], obs['batch'], obs['protocol'], obs['study']
    uns['provenance'] source metadata
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "real", "raw", "abdelaal2019")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "real", "processed")

# -----------------------------------------------------------------------------
# Pancreas: four studies, four protocols
# -----------------------------------------------------------------------------
PANCREAS_STUDIES = {
    # study key: (data csv, labels csv, protocol, accession, citation)
    "Baron": ("Intra-dataset/Pancreatic_data/Baron Human/Filtered_Baron_HumanPancreas_data.csv",
              "Intra-dataset/Pancreatic_data/Baron Human/Labels.csv",
              "inDrop", "GSE84133", "Baron et al., Cell Systems 2016"),
    "Muraro": ("Intra-dataset/Pancreatic_data/Muraro/Filtered_Muraro_HumanPancreas_data.csv",
               "Intra-dataset/Pancreatic_data/Muraro/Labels.csv",
               "CEL-Seq2", "GSE85241", "Muraro et al., Cell Systems 2016"),
    "Segerstolpe": ("Intra-dataset/Pancreatic_data/Segerstolpe/Filtered_Segerstolpe_HumanPancreas_data.csv",
                    "Intra-dataset/Pancreatic_data/Segerstolpe/Labels.csv",
                    "Smart-Seq2", "E-MTAB-5061", "Segerstolpe et al., Cell Metabolism 2016"),
    "Xin": ("Intra-dataset/Pancreatic_data/Xin/Filtered_Xin_HumanPancreas_data.csv",
            "Intra-dataset/Pancreatic_data/Xin/Labels.csv",
            "SMARTer", "GSE81608", "Xin et al., Cell Metabolism 2016"),
}

# Harmonisation of author labels across studies (lower-cased original -> common name).
# Rationale: identical biology under different names (duct/ductal, pp/gamma,
# PSC/mesenchymal/activated+quiescent stellate are all pancreatic stellate cells).
PANCREAS_LABEL_MAP: Dict[str, Optional[str]] = {
    "alpha": "alpha",
    "beta": "beta",
    "delta": "delta",
    "epsilon": "epsilon",
    "gamma": "PP (gamma)",
    "pp": "PP (gamma)",
    "acinar": "acinar",
    "ductal": "ductal",
    "duct": "ductal",
    "activated_stellate": "stellate",
    "quiescent_stellate": "stellate",
    "mesenchymal": "stellate",
    "psc": "stellate",
    "endothelial": "endothelial",
    "macrophage": "macrophage",
    "mhc class ii": "MHC class II",
    "mast": "mast",
    "schwann": "schwann",
    "t_cell": "T cell",
    # Labels that are not a cell type (doublet-like / unresolved) are removed:
    "co-expression": None,
    "unclassified endocrine": None,
    "unclassified": None,
    "not applicable": None,
    "none/other": None,
}

# -----------------------------------------------------------------------------
# PBMC: Ding et al. 2020 (PbmcBench), 7 protocols on sample pbmc1 + 10Xv2 pbmc2
# -----------------------------------------------------------------------------
PBMC_DATA = "Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1.csv"
PBMC_LABELS = "Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1Labels.csv"
# 1-based inclusive row ranges, copied from Statisitics.xlsx (Sheet2, block "X10v2.pbmc1").
PBMC_BLOCKS: List[Tuple[str, str, int, int]] = [
    ("10Xv2", "pbmc1", 1, 6444),
    ("Smart-Seq2", "pbmc1", 6445, 6697),
    ("10Xv3", "pbmc1", 6698, 9919),
    ("CEL-Seq2", "pbmc1", 9920, 10172),
    ("Drop-Seq", "pbmc1", 10173, 13394),
    ("inDrop", "pbmc1", 13395, 16616),
    ("Seq-Well", "pbmc1", 16617, 19792),
    ("10Xv2", "pbmc2", 19793, 23154),
]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _clean_gene(g: str) -> str:
    g = str(g).strip().strip('"')
    g = g.split("__")[0]  # Muraro: "A1BG__chr19"
    return g


def read_cells_by_genes_csv(path: str, chunksize: int = 1000) -> Tuple[sp.csr_matrix, List[str], List[str]]:
    """Read an Abdelaal-format CSV (rows = cells, columns = genes) into a sparse matrix."""
    blocks, cells = [], []
    genes: Optional[List[str]] = None
    for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize):
        if genes is None:
            genes = [str(c) for c in chunk.columns]
        blocks.append(sp.csr_matrix(chunk.to_numpy(dtype=np.float32)))
        cells.extend([str(c) for c in chunk.index])
    X = sp.vstack(blocks).tocsr()
    return X, cells, genes or []


def read_labels(path: str) -> np.ndarray:
    lab = pd.read_csv(path)
    return lab.iloc[:, 0].astype(str).str.strip().to_numpy()


def collapse_duplicate_genes(X: sp.csr_matrix, genes: List[str]) -> Tuple[sp.csr_matrix, List[str]]:
    """Sum columns whose cleaned gene symbol is identical."""
    s = pd.Series(range(len(genes)), index=genes)
    if s.index.is_unique:
        return X, list(genes)
    uniq, inv = np.unique(np.asarray(genes), return_inverse=True)
    agg = sp.csr_matrix((np.ones(len(genes), dtype=np.float32), (np.arange(len(genes)), inv)),
                        shape=(len(genes), len(uniq)))
    return (X @ agg).tocsr(), uniq.tolist()


def _is_integer_matrix(X: sp.csr_matrix) -> bool:
    d = X.data
    return bool(np.all(np.abs(d - np.round(d)) < 1e-6))


# -----------------------------------------------------------------------------
# Dataset builders
# -----------------------------------------------------------------------------
def build_pancreas(raw_dir: str = RAW_DIR) -> ad.AnnData:
    adatas = []
    provenance = []
    for study, (dpath, lpath, protocol, acc, cite) in PANCREAS_STUDIES.items():
        X, cells, genes = read_cells_by_genes_csv(os.path.join(raw_dir, dpath))
        labels = read_labels(os.path.join(raw_dir, lpath))
        if len(labels) != X.shape[0]:
            raise ValueError(f"{study}: {len(labels)} labels vs {X.shape[0]} cells")
        genes = [_clean_gene(g) for g in genes]
        X, genes = collapse_duplicate_genes(X, genes)
        harmon = np.array([PANCREAS_LABEL_MAP.get(l.lower(), "__UNMAPPED__") for l in labels], dtype=object)
        unmapped = sorted(set(labels[harmon == "__UNMAPPED__"]))
        if unmapped:
            raise ValueError(f"{study}: unmapped labels {unmapped}; extend PANCREAS_LABEL_MAP explicitly")
        keep = np.array([h is not None for h in harmon])
        a = ad.AnnData(
            X=X[keep],
            obs=pd.DataFrame(
                {
                    "cell_type": harmon[keep].astype(str),
                    "cell_type_original": labels[keep],
                    "study": study,
                    "protocol": protocol,
                    "batch": f"{study} ({protocol})",
                },
                index=[f"{study}_{c}" for c in np.asarray(cells)[keep]],
            ),
            var=pd.DataFrame(index=genes),
        )
        a.obs["counts_are_integer"] = _is_integer_matrix(a.X)
        provenance.append(
            f"{study}: {cite}, {protocol}, {acc}; {X.shape[0]} cells in file, "
            f"{int((~keep).sum())} removed (non-cell-type labels: "
            f"{sorted(set(labels[~keep]))}); integer counts={_is_integer_matrix(a.X)}"
        )
        adatas.append(a)
    adata = ad.concat(adatas, join="inner", merge="same")
    adata.layers["counts"] = adata.X.copy()
    adata.uns["provenance"] = (
        "Abdelaal et al. 2019 benchmark (Zenodo 10.5281/zenodo.3357167), author labels; "
        "genes = intersection across studies. " + " | ".join(provenance)
    )
    adata.uns["label_map"] = pd.DataFrame(
        {"original_label_lower": list(PANCREAS_LABEL_MAP),
         "harmonised_label": [v if v is not None else "REMOVED" for v in PANCREAS_LABEL_MAP.values()]}
    )
    return adata


def build_pbmc(raw_dir: str = RAW_DIR) -> ad.AnnData:
    X, cells, genes = read_cells_by_genes_csv(os.path.join(raw_dir, PBMC_DATA))
    labels = read_labels(os.path.join(raw_dir, PBMC_LABELS))
    if len(labels) != X.shape[0] or X.shape[0] != PBMC_BLOCKS[-1][3]:
        raise ValueError(f"PBMC: {X.shape[0]} cells, {len(labels)} labels, expected {PBMC_BLOCKS[-1][3]}")
    genes = [_clean_gene(g) for g in genes]
    X, genes = collapse_duplicate_genes(X, genes)
    protocol = np.empty(X.shape[0], dtype=object)
    sample = np.empty(X.shape[0], dtype=object)
    for prot, smp, s, e in PBMC_BLOCKS:
        protocol[s - 1 : e] = prot
        sample[s - 1 : e] = smp
    batch = np.array([f"{p} ({s})" for p, s in zip(protocol, sample)], dtype=object)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "cell_type": labels,
                "cell_type_original": labels,
                "study": "Ding2020",
                "protocol": protocol.astype(str),
                "sample": sample.astype(str),
                "batch": batch.astype(str),
            },
            index=[f"cell{i}_{c}" for i, c in enumerate(cells)],
        ),
        var=pd.DataFrame(index=genes),
    )
    adata.obs["counts_are_integer"] = False
    for b in adata.obs["batch"].unique():
        m = (adata.obs["batch"] == b).to_numpy()
        adata.obs.loc[m, "counts_are_integer"] = _is_integer_matrix(adata.X[m])
    adata.layers["counts"] = adata.X.copy()
    adata.uns["provenance"] = (
        "PbmcBench (Ding et al., Nat Biotechnol 2020; SCP424/GSE132044) as redistributed by "
        "Abdelaal et al. 2019 (Zenodo 10.5281/zenodo.3357167), folder 10Xv2 (all protocols "
        "concatenated; populations absent from 10Xv2 pbmc1 removed by Abdelaal et al.). "
        "Protocol blocks from Statisitics.xlsx: "
        + "; ".join(f"{p}/{s}: rows {a}-{b}" for p, s, a, b in PBMC_BLOCKS)
    )
    return adata


BUILDERS = {"pancreas": build_pancreas, "pbmc": build_pbmc}


# -----------------------------------------------------------------------------
# Tabula Muris: FACS (Smart-seq2) vs droplet (10x), 12 shared tissues
# -----------------------------------------------------------------------------
TM_DIR = os.path.join(PROJECT_ROOT, "data", "real", "raw", "tabula_muris")
# droplet tissue names -> FACS archive names
TM_TISSUE_ALIAS = {"Limb_Muscle": "Muscle", "Mammary_Gland": "Mammary", "Heart_and_Aorta": "Heart"}


def _read_tm_facs(tissue_archive: str, keep_cells: set) -> Tuple[sp.csr_matrix, List[str], List[str]]:
    """Each member is one cell: 'Tissue/A1-B001717-3_38_F-1-1.csv' with lines 'gene,count'."""
    import tarfile

    rows, cells = [], []
    genes: Optional[List[str]] = None
    with tarfile.open(tissue_archive, "r:gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith(".csv"):
                continue
            cid = os.path.basename(m.name)[:-4].replace("-", ".")
            if cid not in keep_cells:
                continue
            txt = tar.extractfile(m).read().decode()
            g, c = zip(*(ln.rsplit(",", 1) for ln in txt.strip().splitlines()))
            if genes is None:
                genes = list(g)
            elif len(g) != len(genes) or g[0] != genes[0] or g[-1] != genes[-1]:
                raise ValueError(f"gene order differs in {m.name}")
            v = np.asarray(c, dtype=np.float32)
            rows.append(sp.csr_matrix(v))
            cells.append(cid)
    return sp.vstack(rows).tocsr(), cells, genes or []


def _read_tm_droplet(raw_tar: str, keep_cells: set):
    """GSE109774_RAW.tar -> one GSM*_<Tissue>-10X_Px_y.tar.gz per channel (10x v2 mtx)."""
    import io as _io
    import tarfile

    import scipy.io as spio

    mats, cells, genes_ref = [], [], None
    with tarfile.open(raw_tar, "r:") as outer:
        for m in outer:
            if not m.name.endswith(".tar.gz"):
                continue
            channel = re.search(r"(10X_P\d+_\d+)", m.name).group(1)
            inner_bytes = outer.extractfile(m).read()
            files = {}
            with tarfile.open(fileobj=_io.BytesIO(inner_bytes), mode="r:gz") as inner:
                for im in inner:
                    if im.isfile():
                        files[os.path.basename(im.name)] = inner.extractfile(im).read()
            mtx = spio.mmread(_io.BytesIO(files["matrix.mtx"])).T.tocsr().astype(np.float32)
            bcs = [b.split("-")[0] for b in files["barcodes.tsv"].decode().split()]
            genes = [ln.split("\t")[-1] if "\t" in ln else ln for ln in files["genes.tsv"].decode().strip().splitlines()]
            if genes_ref is None:
                genes_ref = genes
            elif genes != genes_ref:
                raise ValueError(f"gene list differs in {m.name}")
            ids = np.array([f"{channel}_{b}" for b in bcs])
            sel = np.isin(ids, list(keep_cells))
            mats.append(mtx[sel])
            cells.extend(ids[sel].tolist())
    return sp.vstack(mats).tocsr(), cells, genes_ref


def build_tabula_muris(raw_dir: str = TM_DIR) -> ad.AnnData:
    ann_d = pd.read_csv(os.path.join(raw_dir, "annotations_droplet.csv"), low_memory=False)
    ann_f = pd.read_csv(os.path.join(raw_dir, "annotations_facs.csv"), low_memory=False)
    ann_d["tissue_std"] = ann_d["tissue"].map(lambda t: TM_TISSUE_ALIAS.get(t, t))
    ann_f["tissue_std"] = ann_f["tissue"].replace({"Limb_Muscle": "Muscle", "Mammary_Gland": "Mammary", "Aorta": "Heart"})
    shared = sorted(set(ann_d.tissue_std) & set(ann_f.tissue_std))
    ann_d = ann_d[ann_d.tissue_std.isin(shared) & ann_d.cell_ontology_class.notna()]
    ann_f = ann_f[ann_f.tissue_std.isin(shared) & ann_f.cell_ontology_class.notna()]

    parts = []
    # FACS
    Xs, cs, gref = [], [], None
    for t in shared:
        arch = os.path.join(raw_dir, "facs", f"GSE109774_{t}.tar.gz")
        keep = set(ann_f.loc[ann_f.tissue_std == t, "cell"])
        X, cells, genes = _read_tm_facs(arch, keep)
        if gref is None:
            gref = genes
        elif genes != gref:
            raise ValueError(f"FACS gene list differs for {t}")
        Xs.append(X)
        cs.extend(cells)
        print(f"  FACS {t}: {X.shape[0]} annotated cells", flush=True)
    Xf = sp.vstack(Xs).tocsr()
    af = ann_f.set_index("cell").loc[cs]
    parts.append(ad.AnnData(X=Xf, var=pd.DataFrame(index=gref), obs=pd.DataFrame({
        "cell_type": af.cell_ontology_class.astype(str).values,
        "cell_type_original": af.cell_ontology_class.astype(str).values,
        "free_annotation": af.free_annotation.astype(str).values,
        "tissue": af.tissue_std.values, "mouse": af["mouse.id"].astype(str).values,
        "study": "TabulaMuris", "protocol": "FACS Smart-seq2", "batch": "FACS (Smart-seq2)",
    }, index=["facs_" + c for c in cs])))
    # droplet
    Xd, cd, gd = _read_tm_droplet(os.path.join(raw_dir, "droplet", "GSE109774_RAW.tar"), set(ann_d.cell))
    adx = ann_d.set_index("cell").loc[cd]
    print(f"  droplet: {Xd.shape[0]} annotated cells", flush=True)
    parts.append(ad.AnnData(X=Xd, var=pd.DataFrame(index=gd), obs=pd.DataFrame({
        "cell_type": adx.cell_ontology_class.astype(str).values,
        "cell_type_original": adx.cell_ontology_class.astype(str).values,
        "free_annotation": adx.free_annotation.astype(str).values,
        "tissue": adx.tissue_std.values, "mouse": adx["mouse.id"].astype(str).values,
        "study": "TabulaMuris", "protocol": "10x Chromium v2", "batch": "Droplet (10x)",
    }, index=["droplet_" + c for c in cd])))
    for p in parts:
        keep = ~p.var_names.str.startswith("ERCC-")
        p._inplace_subset_var(keep)
        p.var_names_make_unique()
    adata = ad.concat(parts, join="inner")
    adata.layers["counts"] = adata.X.copy()
    adata.uns["provenance"] = (
        "Tabula Muris Consortium, Nature 2018 (GEO GSE109774). FACS Smart-seq2 per-cell count files "
        "(GSE109774_<Tissue>.tar.gz) and 10x droplet channels (GSE109774_RAW.tar); consortium annotations "
        "(cell_ontology_class) from github.com/czbiohub-sf/tabula-muris (18_global_annotation_csv). "
        f"Restricted to the {len(shared)} tissues profiled by both technologies: {', '.join(shared)}. "
        "Heart (FACS) incl. Aorta is matched to droplet 'Heart_and_Aorta'. ERCC spike-ins removed; genes = intersection."
    )
    return adata


# -----------------------------------------------------------------------------
# Lung COVID-19 BALF: Liao et al., Nature Medicine 2020
# -----------------------------------------------------------------------------
LIAO_DIR = os.path.join(PROJECT_ROOT, "data", "real", "raw", "liao2020")


def build_lung_covid(raw_dir: str = LIAO_DIR) -> ad.AnnData:
    import io as _io
    import tarfile

    import scanpy as sc

    meta = pd.read_csv(os.path.join(raw_dir, "all.cell.annotation.meta.txt"), sep="\t")
    meta["suffix"] = meta.ID.str.rsplit("_", n=1).str[1]
    meta["barcode"] = meta.ID.str.rsplit("_", n=1).str[0]
    suffix_of = meta.groupby("sample").suffix.first().to_dict()  # e.g. C51 -> '1', GSM3660650 -> '4'
    parts = []
    with tarfile.open(os.path.join(raw_dir, "GSE145926_RAW.tar"), "r:") as tar:
        for m in tar:
            if not m.name.endswith("_filtered_feature_bc_matrix.h5"):
                continue
            sample = re.search(r"_(C\d+)_filtered", m.name).group(1)
            tmp = os.path.join(raw_dir, f"_{sample}.h5")
            with open(tmp, "wb") as fh:
                fh.write(tar.extractfile(m).read())
            a = sc.read_10x_h5(tmp)
            os.remove(tmp)
            a.var_names_make_unique()
            a.obs_names = [f"{b.split('-')[0]}_{suffix_of[sample]}" for b in a.obs_names]
            parts.append(a)
    # HC4 (GSM3660650, 10x v2 mtx)
    d = os.path.join(raw_dir, "GSM3660650")
    import scipy.io as spio
    X = spio.mmread(os.path.join(d, "matrix.mtx.gz")).T.tocsr().astype(np.float32)
    genes = pd.read_csv(os.path.join(d, "genes.tsv.gz"), sep="\t", header=None)
    bcs = pd.read_csv(os.path.join(d, "barcodes.tsv.gz"), sep="\t", header=None)[0]
    a = ad.AnnData(X=X, var=pd.DataFrame(index=genes.iloc[:, 1].astype(str).values))
    a.var_names_make_unique()
    a.obs_names = [f"{b.split('-')[0]}_{suffix_of['GSM3660650']}" for b in bcs]
    parts.append(a)

    adata = ad.concat(parts, join="inner")
    m = meta.set_index("ID")
    keep = adata.obs_names.isin(m.index)
    adata = adata[keep].copy()
    mm = m.loc[adata.obs_names]
    group_name = {"HC": "Healthy", "M": "Moderate", "S": "Severe"}
    adata.obs = pd.DataFrame({
        "cell_type": mm.celltype.astype(str).values,
        "cell_type_original": mm.celltype.astype(str).values,
        "sample": mm.sample_new.astype(str).values,
        "condition": mm.group.map(group_name).values,
        "study": "Liao2020", "protocol": "10x 5'",
        "batch": mm.sample_new.astype(str).values,   # each patient is a separate 10x run
    }, index=adata.obs_names)
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    adata.layers["counts"] = adata.X.copy()
    adata.uns["provenance"] = (
        "Liao et al., Nat Med 2020 (GEO GSE145926, 12 BALF samples) + healthy control HC4 GSM3660650 "
        "(Morse et al. 2019, GSE128033) as used by Liao et al.; author annotations (celltype) from "
        f"github.com/zhangzlab/covid_balf all.cell.annotation.meta.txt; {int(keep.sum())} of "
        f"{len(meta)} annotated cells matched."
    )
    return adata


BUILDERS.update({"tabula_muris": build_tabula_muris, "lung_covid": build_lung_covid})


def load_processed(name: str, processed_dir: str = PROCESSED_DIR) -> ad.AnnData:
    path = os.path.join(processed_dir, f"{name}.h5ad")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `python code/real_data/loaders.py --datasets {name}` first")
    return ad.read_h5ad(path)


def summarize(adata: ad.AnnData) -> pd.DataFrame:
    tab = pd.crosstab(adata.obs["cell_type"], adata.obs["batch"], margins=True, margins_name="Total")
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--raw_dir", default=None, help="override the builder default raw directory")
    ap.add_argument("--out_dir", default=PROCESSED_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.datasets:
        print(f"[loaders] building {name} ...", flush=True)
        a = BUILDERS[name](args.raw_dir) if args.raw_dir else BUILDERS[name]()
        a.obs = a.obs.astype({c: "category" for c in a.obs.columns if a.obs[c].dtype == object})
        out = os.path.join(args.out_dir, f"{name}.h5ad")
        a.write_h5ad(out, compression="gzip")
        tab = summarize(a)
        tab.to_csv(os.path.join(args.out_dir, f"{name}_celltype_by_batch.csv"))
        print(f"[loaders] {name}: {a.n_obs} cells x {a.n_vars} genes -> {out}")
        print(tab.to_string())
        print(a.uns["provenance"])


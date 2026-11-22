#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared preprocessing utilities (log-normalization, training-only HVG selection,
and GMT pathway mask construction).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
_TOSICA_RES = os.path.abspath(os.path.join(HERE, "..", "third_party", "TOSICA", "TOSICA", "resources"))
REACTOME_GMT = os.path.join(_TOSICA_RES, "reactome.gmt")
GMT_FILES = {
    "human_gobp": os.path.join(_TOSICA_RES, "GO_bp.gmt"),
    "human_immune": os.path.join(_TOSICA_RES, "immune.gmt"),
    "human_reactome": REACTOME_GMT,
    "mouse_gobp": os.path.join(_TOSICA_RES, "m_GO_bp.gmt"),
}
# Gene-set prior per dataset (shared by TOSICA and scPathProto). Pancreas follows the
# official TOSICA tutorial on the same Baron/Muraro -> Xin/Segerstolpe benchmark
# (human_gobp, 3000 HVGs); PBMC uses TOSICA's immune gene-set collection.
DATASET_GMT = {"pancreas": "human_gobp", "pbmc": "human_immune", "lung_covid": "human_immune", "tabula_muris": "mouse_gobp"}


def log_normalize(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """X <- log1p(counts / library_size * target_sum); raw counts stay in layers['counts']."""
    counts = adata.layers["counts"]
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    lib = np.asarray(counts.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    Xn = sp.diags((target_sum / lib).astype(np.float32)) @ counts
    Xn = Xn.tocsr()
    Xn.data = np.log1p(Xn.data).astype(np.float32)
    adata.X = Xn
    adata.uns["log1p"] = {"base": None}
    return adata


def select_hvgs(adata_train: ad.AnnData, n_top: int = 2000, batch_key: Optional[str] = "batch") -> List[str]:
    """Batch-aware HVG selection (scanpy 'seurat' flavour) on training cells only."""
    import scanpy as sc

    a = ad.AnnData(X=adata_train.X.copy(), obs=adata_train.obs[[batch_key]].copy() if batch_key else None,
                   var=pd.DataFrame(index=adata_train.var_names))
    a.uns["log1p"] = {"base": None}
    # genes must be expressed in the training data
    expressed = np.asarray((a.X > 0).sum(axis=0)).ravel() >= 3
    a = a[:, expressed].copy()
    use_batch = batch_key if (batch_key and a.obs[batch_key].nunique() > 1) else None
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_top, a.n_vars), flavor="seurat", batch_key=use_batch)
    hv = a.var_names[a.var["highly_variable"].to_numpy()]
    return list(hv)


def read_gmt(path: str = REACTOME_GMT, min_size: int = 0, max_size: int = 5000) -> "OrderedDict[str, List[str]]":
    out: "OrderedDict[str, List[str]]" = OrderedDict()
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genes = [g for g in parts[2:] if g]
            if min_size <= len(genes) <= max_size:
                out[parts[0]] = genes
    return out


def build_pathway_mask(
    genes: Sequence[str],
    gmt_path: str = REACTOME_GMT,
    min_genes_in_data: int = 5,
    max_gene_set_size: int = 300,
    max_pathways: int = 300,
) -> Tuple[np.ndarray, List[str]]:
    """
    Binary prior M in {0,1}^{P x G} from a GMT file, restricted to ``genes``.
    Mirrors TOSICA's defaults (max_g = 300 genes per set, max_gs = 300 sets,
    keep sets with > 4 member genes in the data, keep the largest ones).
    """
    gs = read_gmt(gmt_path, max_size=max_gene_set_size)
    gidx = {g: i for i, g in enumerate(genes)}
    names, rows = [], []
    for name, members in gs.items():
        cols = sorted({gidx[g] for g in members if g in gidx})
        if len(cols) >= min_genes_in_data:
            r = np.zeros(len(genes), dtype=np.float32)
            r[cols] = 1.0
            rows.append(r)
            names.append(name)
    if not rows:
        raise ValueError("No gene set overlaps the selected genes")
    M = np.vstack(rows)
    if M.shape[0] > max_pathways:
        keep = np.sort(np.argsort(M.sum(axis=1))[-max_pathways:])
        M = M[keep]
        names = [names[i] for i in keep]
    return M, names


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quickstart inference demo for scPathProto on the bundled PBMC slice
(`data/demo/pbmc_demo_500cells.h5ad`) using pretrained checkpoints
(`checkpoints/pbmc_scPathProto.pth` and `.pkl`).

Usage:
    python3 code/demo_inference.py
"""
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from real_data.preprocess import log_normalize


def main():
    pth_path = REPO_ROOT / "checkpoints" / "pbmc_scPathProto.pth"
    pkl_path = REPO_ROOT / "checkpoints" / "pbmc_scPathProto.pkl"
    demo_h5 = REPO_ROOT / "data" / "demo" / "pbmc_demo_500cells.h5ad"

    print(f"[1/3] Loading portable PyTorch checkpoint: {pth_path.name}")
    ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    gene_names = ckpt["gene_names"]
    class_names = ckpt["class_names"]
    print(
        f"      Atlas={ckpt['dataset_name']} | Genes={len(gene_names)} | "
        f"Classes={len(class_names)} | Pathways={len(ckpt['pathway_names'])} | "
        f"Prototypes={tuple(ckpt['state_tensors']['prototypes'].shape)}"
    )

    print(f"[2/3] Loading bundled real PBMC demo data: {demo_h5.name}")
    adata = log_normalize(ad.read_h5ad(str(demo_h5)))
    X = adata[:, gene_names].X
    X = X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)
    y_true = np.array([class_names.index(c) for c in adata.obs["cell_type"].astype(str)])
    b_map = {b: i for i, b in enumerate(sorted(set(adata.obs["batch"].astype(str))))}
    b_ids = np.array([b_map[b] for b in adata.obs["batch"].astype(str)], dtype=np.int64)

    print(f"[3/3] Running scPathProto inference & Dirichlet vacuity scoring on {len(y_true)} cells...")
    model = joblib.load(str(pkl_path))
    out = model.predict_detailed(X, b_ids)
    preds = out["pred_labels"]
    vacuity = out["vacuity"]
    novelty = out["novelty_score"]

    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    print(f"\n=== Demo Results ({len(y_true)} real PBMC cells across {len(b_map)} protocols) ===")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro-F1          : {macro_f1:.4f}")
    print(f"  Median Vacuity u  : {float(np.median(vacuity)):.4f}")
    print(f"  Median Novelty s  : {float(np.median(novelty)):.4f}")
    print("[PASS] End-to-end scPathProto inference verified.")


if __name__ == "__main__":
    main()

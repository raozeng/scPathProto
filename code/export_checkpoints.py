#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and export scPathProto checkpoints across the four benchmark atlases
(Pancreas, PBMC, Tabula Muris, Lung COVID-19):

  - `checkpoints/<atlas>_scPathProto.pth`: PyTorch state dictionary containing
    learned tensors, HVG/pathway/class names, config, validation metrics, and
    training history.
  - `checkpoints/<atlas>_scPathProto.pkl`: compressed joblib `ScPathProtoModel`
    object for direct inference via `predict_detailed`.
  - `checkpoints/checkpoint_manifest.json`: SHA-256 checksums and metadata.

Also saves per-atlas training histories to `results/tables/training_history_<atlas>.csv`.
"""
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from model import ScPathProtoConfig, ScPathProtoModel
from real_data.preprocess import build_pathway_mask, log_normalize, select_hvgs

GMT_ROOT = SCRIPT_DIR / "third_party" / "TOSICA" / "TOSICA" / "resources"
ATLAS_SPECS = [
    ("pancreas", "Pancreas", GMT_ROOT / "GO_bp.gmt", "batch"),
    ("pbmc", "PBMC", GMT_ROOT / "immune.gmt", "batch"),
    ("tabula_muris", "Tabula Muris", GMT_ROOT / "m_GO_bp.gmt", "batch"),
    ("lung_covid", "Lung COVID-19", GMT_ROOT / "immune.gmt", "batch"),
]


def to_tensor_dict(model: ScPathProtoModel) -> dict:
    """Convert all learned numerical parameters of ScPathProtoModel into torch.Tensors."""
    attr_names = [
        "prototypes",
        "proj_W",
        "proj_b",
        "W_pg",
        "pathway_mask",
        "pathway_self_attn",
        "zirg_W",
        "zirg_gene_reliability_",
        "mpca_gene_prior_",
        "rare_margins",
        "rare_skip_margins_",
        "hasgmp_feat_w",
        "class_priors",
        "class_centroids",
        "class_pathway_profiles",
        "class_gene_profiles",
        "gene_cond_mean",
        "gene_non_zero_rate",
        "ot_class_means",
        "global_centroid",
        "global_std",
    ]
    tensors = {}
    for name in attr_names:
        val = getattr(model, name, None)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.dtype.kind in ("f", "i", "u", "b"):
            dtype = torch.float32 if arr.dtype.kind == "f" else torch.int64
            tensors[name] = torch.as_tensor(arr.copy(), dtype=dtype)
    if getattr(model, "oor_ref_scale_", None) is not None:
        tensors["oor_ref_scale_"] = torch.tensor(float(model.oor_ref_scale_), dtype=torch.float32)
    return tensors


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_pth(path: str | Path) -> dict:
    """Helper demonstrating safe portable loading of a `.pth` checkpoint."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def main():
    out_ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    out_tables_dir = PROJECT_ROOT / "results" / "tables"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_tables_dir.mkdir(parents=True, exist_ok=True)

    # Remove any legacy stale checkpoints first
    for stale in out_ckpt_dir.glob("*"):
        if stale.is_file():
            stale.unlink()

    manifest = {
        "framework": "scPathProto",
        "format_version": "1.0-real-atlas",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "checkpoints": {},
    }

    for key, display_name, gmt_path, batch_col in ATLAS_SPECS:
        t0 = time.time()
        h5_path = PROJECT_ROOT / "data" / "real" / "processed" / f"{key}.h5ad"
        print(f"\n=== [{display_name}] Loading {h5_path.name} ===")
        adata = log_normalize(ad.read_h5ad(str(h5_path)))
        y_str = adata.obs["cell_type"].astype(str).to_numpy()
        b_str = adata.obs[batch_col].astype(str).to_numpy()

        classes = sorted(set(y_str))
        cmap = {c: i for i, c in enumerate(classes)}
        y_all = np.array([cmap[c] for c in y_str], dtype=np.int64)
        batches = sorted(set(b_str))
        bmap = {b: i for i, b in enumerate(batches)}
        b_all = np.array([bmap[b] for b in b_str], dtype=np.int64)

        # Stratified 85/15 train/validation split
        vc = pd.Series(y_all).value_counts()
        strat_y = y_all.copy()
        # Keep singleton classes in training split
        singletons = set(vc[vc < 2].index)
        if singletons:
            sing_mask = np.isin(y_all, list(singletons))
            non_sing_idx = np.where(~sing_mask)[0]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
            tr_rel, va_rel = next(sss.split(non_sing_idx, y_all[non_sing_idx]))
            tr_idx = np.concatenate([non_sing_idx[tr_rel], np.where(sing_mask)[0]])
            va_idx = non_sing_idx[va_rel]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
            tr_idx, va_idx = next(sss.split(np.arange(len(y_all)), y_all))

        # Select 3,000 HVGs on training split
        adata_tr = adata[tr_idx].copy()
        hvgs = select_hvgs(adata_tr, n_top=3000, batch_key=batch_col if len(batches) > 1 else None)
        M_mask, pnames = build_pathway_mask(hvgs, gmt_path=str(gmt_path), max_pathways=300)

        X_tr = adata[tr_idx, hvgs].X
        X_va = adata[va_idx, hvgs].X
        X_tr = X_tr.toarray().astype(np.float32) if hasattr(X_tr, "toarray") else np.asarray(X_tr, dtype=np.float32)
        X_va = X_va.toarray().astype(np.float32) if hasattr(X_va, "toarray") else np.asarray(X_va, dtype=np.float32)
        y_tr, b_tr = y_all[tr_idx], b_all[tr_idx]
        y_va, b_va = y_all[va_idx], b_all[va_idx]

        cfg = ScPathProtoConfig(
            n_genes=len(hvgs),
            n_classes=len(classes),
            n_batches=len(batches),
            n_known_pathways=int(M_mask.shape[0]),
            n_latent_pathways=8,
            embed_dim=128,
            knn_k=15,
            proto_temp=0.12,
            rare_margin_scale=0.18,
            epochs=45,
            seed=0,
        )
        model = ScPathProtoModel(
            config=cfg,
            pathway_mask=M_mask,
            pathway_names=pnames,
            gene_names=list(hvgs),
        )
        model.fit(X_tr, y_tr, b_tr, X_val=X_va, y_val=y_va, batch_val=b_va)

        # Evaluate on held-out 15% validation split
        out_va = model.predict_detailed(X_va, b_va)
        preds_va = out_va["pred_labels"]
        acc_va = float(accuracy_score(y_va, preds_va))
        present = sorted(set(y_va))
        per_c = f1_score(y_va, preds_va, labels=present, average=None, zero_division=0)
        macro_va = float(np.mean(per_c))
        vc_tr = pd.Series(y_tr).value_counts()
        rare_set = set(vc_tr[vc_tr / len(y_tr) < 0.025].index)
        rare_vals = [per_c[present.index(c)] for c in present if c in rare_set]
        rare_va = float(np.mean(rare_vals)) if rare_vals else float("nan")

        # Export 45-step training history CSV
        hist_df = pd.DataFrame(model.training_history)
        hist_csv = out_tables_dir / f"training_history_{key}.csv"
        hist_df.to_csv(hist_csv, index=False)

        # Save PyTorch checkpoint bundle
        pth_path = out_ckpt_dir / f"{key}_scPathProto.pth"
        pth_bundle = {
            "format": "scPathProto-v1.0-real-atlas",
            "dataset_key": key,
            "dataset_name": display_name,
            "gmt_file": gmt_path.name,
            "n_total_cells": int(len(y_all)),
            "n_train_cells": int(len(y_tr)),
            "n_val_cells": int(len(y_va)),
            "n_genes": int(len(hvgs)),
            "n_classes": int(len(classes)),
            "n_batches": int(len(batches)),
            "n_known_pathways": int(M_mask.shape[0]),
            "n_latent_pathways": 8,
            "config": asdict(cfg),
            "gene_names": list(hvgs),
            "pathway_names": list(model.pathway_names),
            "class_names": classes,
            "batch_names": batches,
            "val_metrics": {
                "val_accuracy": round(acc_va, 4),
                "val_macro_f1": round(macro_va, 4),
                "val_rare_f1": round(rare_va, 4) if not np.isnan(rare_va) else None,
            },
            "training_history": model.training_history,
            "state_tensors": to_tensor_dict(model),
        }
        torch.save(pth_bundle, str(pth_path))

        # Save compressed joblib `.pkl` checkpoint for direct `predict_detailed` inference
        pkl_path = out_ckpt_dir / f"{key}_scPathProto.pkl"
        joblib.dump(model, str(pkl_path), compress=3)

        # Round-trip load verification
        reloaded = load_checkpoint_pth(pth_path)
        assert reloaded["n_genes"] == 3000 and reloaded["n_classes"] == len(classes)
        assert "prototypes" in reloaded["state_tensors"] and "W_pg" in reloaded["state_tensors"]

        elapsed = time.time() - t0
        manifest["checkpoints"][key] = {
            "dataset_name": display_name,
            "gmt_file": gmt_path.name,
            "n_total_cells": int(len(y_all)),
            "n_train_cells": int(len(y_tr)),
            "n_val_cells": int(len(y_va)),
            "n_genes": int(len(hvgs)),
            "n_classes": int(len(classes)),
            "n_batches": int(len(batches)),
            "n_known_pathways": int(M_mask.shape[0]),
            "n_latent_pathways": 8,
            "val_accuracy": round(acc_va, 4),
            "val_macro_f1": round(macro_va, 4),
            "val_rare_f1": round(rare_va, 4) if not np.isnan(rare_va) else None,
            "pth_file": pth_path.name,
            "pth_size_bytes": pth_path.stat().st_size,
            "pth_sha256": sha256_file(pth_path),
            "pkl_file": pkl_path.name,
            "pkl_size_bytes": pkl_path.stat().st_size,
            "pkl_sha256": sha256_file(pkl_path),
            "fit_and_export_sec": round(elapsed, 2),
        }
        print(
            f"  [OK] {display_name}: Val Acc={acc_va:.4f}, Macro-F1={macro_va:.4f}, Rare-F1={rare_va:.4f} | "
            f"{pth_path.name} ({pth_path.stat().st_size/1e6:.2f} MB), {pkl_path.name} ({pkl_path.stat().st_size/1e6:.2f} MB) [{elapsed:.1f}s]"
        )

    man_path = out_ckpt_dir / "checkpoint_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved manifest to {man_path}")


if __name__ == "__main__":
    main()


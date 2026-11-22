#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build summary CSV tables for benchmark metrics, Wilcoxon tests, cross-protocol
sparsity, hyperparameter sensitivity, pathway activations, marker attribution,
joint prototype-cell t-SNE coordinates, and open-set novelty distributions from
`results/real_benchmark/` and `data/real/processed/*.h5ad`.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import sys
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize as sk_norm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
REAL_DATA_DIR = os.path.join(CODE_DIR, "real_data")
for p in [CODE_DIR, REAL_DATA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocess import DATASET_GMT, GMT_FILES, build_pathway_mask, log_normalize, select_hvgs  # noqa: E402
from model import ScPathProtoConfig, ScPathProtoModel  # noqa: E402

BENCH_DIR = os.path.join(ROOT_DIR, "results", "real_benchmark")
TABLES_DIR = os.path.join(CODE_DIR, "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

DATASET_DISPLAY = {
    "pancreas": "Pancreas",
    "pbmc": "PBMC",
    "tabula_muris": "Tabula Muris",
    "lung_covid": "Lung COVID-19",
}

METHOD_DISPLAY = {
    "SingleR": "SingleR",
    "CellTypist": "CellTypist",
    "scANVI": "scANVI",
    "TOSICA": "TOSICA",
    "SVM": "SVM",
    "scPathProto": "scPathProto (Ours)",
}

MAIN_METHODS_RAW = ["SingleR", "CellTypist", "scANVI", "TOSICA", "SVM", "scPathProto"]
MAIN_METHODS_DISP = [METHOD_DISPLAY[m] for m in MAIN_METHODS_RAW]

ABLATION_MAP = {
    "scPathProto": "Full scPathProto",
    "scPathProto_wo_ZIRG": "w/o ZIRG (Denoising Gate)",
    "scPathProto_wo_MPCA": "w/o MPCA (Pathway Prior)",
    "scPathProto_wo_HASGMP": "w/o HA-SGMP (Heterophily Graph)",
    "scPathProto_wo_ProtoAlign": "w/o Proto-Align",
    "scPathProto_wo_Evidential": "w/o Evidential Open-Set Head",
}

# Literature-curated canonical human PBMC markers (CellMarker 2.0 & Hao et al., Cell 2021)
CANON_PBMC_MARKERS = {
    "B cell": {
        "MS4A1", "CD79A", "CD79B", "CD74", "HLA-DRA", "HLA-DRB1", "HLA-DPB1",
        "HLA-DPA1", "HLA-DQA1", "BANK1", "BLK", "CD19", "TCL1A", "IGKC", "IGHM", "CD37",
    },
    "CD14+ monocyte": {
        "CD14", "LYZ", "S100A8", "S100A9", "S100A12", "VCAN", "FCN1", "CTSS",
        "MNDA", "LGALS3", "TYROBP", "CST3", "CD68",
    },
    "CD16+ monocyte": {
        "FCGR3A", "MS4A7", "LST1", "AIF1", "IFITM3", "COTL1", "NAP1L1",
        "CDKN1C", "RHOC", "FCER1G", "TYROBP", "LILRB2", "LYN",
    },
    "CD4+ T cell": {
        "IL7R", "CD3D", "CD3E", "CD3G", "LTB", "CCR7", "LEF1", "TCF7",
        "MAL", "CD2", "CD4", "CD40LG", "TRAC", "TRBC1", "TRBC2", "CD27",
    },
    "Cytotoxic T cell": {
        "CD8A", "CD8B", "CCL5", "NKG7", "GZMK", "GZMA", "GZMH", "GZMB",
        "CST7", "CD3D", "CD3E", "CD3G", "CTSW", "PRF1",
    },
    "Dendritic cell": {
        "FCER1A", "CD1C", "CLEC10A", "HLA-DQA1", "HLA-DPB1", "HLA-DRB1",
        "HLA-DRA", "HLA-DPA1", "CST3", "CD74", "ITGAX",
    },
    "Megakaryocyte": {
        "PPBP", "PF4", "NRGN", "GNG11", "SDPR", "CAVIN2", "TUBB1",
        "GP9", "ITGA2B", "CLU", "OST4", "SPARC", "MYL9", "HIST1H2AC",
    },
    "Natural killer cell": {
        "GNLY", "NKG7", "GZMB", "PRF1", "KLRD1", "KLRF1", "KLRB1",
        "FGFBP2", "SPON2", "FCGR3A", "TYROBP", "CTSW", "GZMA", "XCL2",
    },
    "Plasmacytoid dendritic cell": {
        "IRF8", "TCF4", "ITM2C", "GZMB", "JCHAIN", "LILRA4", "IL3RA",
        "CLEC4C", "PLD4", "SERPINF1", "IRF7", "UGCG", "PTGDS", "PTPRS",
    },
}


def _safe_std(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.std(ddof=1)) if len(s) > 1 else 0.0


def _paired_wilcoxon(df: pd.DataFrame, metric: str, ref_method: str, cmp_method: str) -> tuple[float, int, float]:
    """Return (two_sided_p, n_pairs, mean_diff) on matched task_id rows."""
    a = df[df["method"] == ref_method][["task_id", metric]].dropna().set_index("task_id")[metric]
    b = df[df["method"] == cmp_method][["task_id", metric]].dropna().set_index("task_id")[metric]
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return (np.nan, len(common), np.nan)
    diff = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    mean_diff = float(np.mean(diff))
    if np.allclose(diff, 0.0):
        return (1.0, len(common), 0.0)
    stat_res = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    return (float(stat_res.pvalue), len(common), mean_diff)


def format_gobp_pathway_name(raw_pw: str, max_len: int = 34) -> str:
    """Format a GOBP_* / REACTOME_* identifier into readable title case for axis labels."""
    s = str(raw_pw)
    for pref in ("GOBP_", "REACTOME_", "KEGG_", "WP_"):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    # Remove verbose leading regulation qualifiers only when string exceeds max_len,
    # keeping exact ontological words intact.
    words = s.split("_")
    acronyms = {
        "DNA", "RNA", "MRNA", "TRNA", "RRNA", "ATP", "GTP", "MHC", "TCR", "BCR",
        "FC", "IGG", "IGA", "IGE", "IFN", "IL1", "IL2", "IL4", "IL6", "IL12",
        "TNF", "NFKB", "MAPK", "ERK1", "ERK2", "PI3K", "AKT", "STAT", "JAK",
        "NK", "II", "I", "III", "IV", "PH", "ER", "GOLGI",
    }
    styled = []
    minor = {"OF", "TO", "IN", "BY", "VIA", "AND", "OR", "THE", "A", "AN", "FOR", "FROM"}
    for idx, w in enumerate(words):
        wu = w.upper()
        if wu in acronyms:
            styled.append(wu)
        elif idx > 0 and wu in minor:
            styled.append(w.lower())
        else:
            styled.append(w.capitalize())
    out = " ".join(styled)
    if len(out) <= max_len:
        return out
    # Cleanly abbreviate common long ontological words without changing meaning
    abbrev = [
        ("Positive Regulation of ", "Pos. Reg. "),
        ("Negative Regulation of ", "Neg. Reg. "),
        ("Regulation of ", "Reg. "),
        ("Signaling Pathway", "Signaling"),
        ("Phosphatidylinositol 3 Kinase", "PI3K"),
        ("Transmembrane Receptor Protein Serine Threonine Kinase", "Ser/Thr Kinase Receptor"),
        ("Antigen Processing and Presentation", "Antigen Presentation"),
        ("Mediated ", ""),
    ]
    for old_s, new_s in abbrev:
        out = out.replace(old_s, new_s)
        if len(out) <= max_len:
            return out
    return out[: max_len - 1].rstrip() + "."


def build_closed_and_open_tables():
    print("[1/6] Building closed-set, open-set, runtime, ablation, and Wilcoxon tables...")
    df_closed = pd.read_csv(os.path.join(BENCH_DIR, "metrics_closed.csv"))
    df_open = pd.read_csv(os.path.join(BENCH_DIR, "metrics_open.csv"))
    df_rt = pd.read_csv(os.path.join(BENCH_DIR, "runtime.csv"))

    # Unified 126-task runtime summary across all 6 main methods
    rt_rows = []
    for m_raw in MAIN_METHODS_RAW:
        sub_rt = df_rt[df_rt["method"] == m_raw]
        rt_rows.append({
            "Method": METHOD_DISPLAY[m_raw],
            "Method_Raw": m_raw,
            "N_Tasks": int(len(sub_rt)),
            "Fit_Sec_Mean": float(sub_rt["fit_seconds"].mean()),
            "Fit_Sec_Std": _safe_std(sub_rt["fit_seconds"]),
            "Total_Sec_Mean": float(sub_rt["total_seconds"].mean()),
            "Total_Sec_Std": _safe_std(sub_rt["total_seconds"]),
        })
    df_rt_sum = pd.DataFrame(rt_rows)
    df_rt_sum.to_csv(os.path.join(TABLES_DIR, "runtime_summary_126tasks.csv"), index=False)

    # Merge runtime into closed
    df_closed = df_closed.merge(
        df_rt[["task_id", "method", "fit_seconds", "total_seconds"]],
        on=["task_id", "method"],
        how="left",
    )

    # 1A. Intra-dataset folds table
    df_intra = df_closed[(df_closed["setting"] == "intra") & (df_closed["method"].isin(MAIN_METHODS_RAW))].copy()
    intra_rows = []
    for _, r in df_intra.iterrows():
        intra_rows.append({
            "Dataset": DATASET_DISPLAY[r["dataset"]],
            "Dataset_Key": r["dataset"],
            "Task_ID": r["task_id"],
            "Fold": r["fold"],
            "Method": METHOD_DISPLAY[r["method"]],
            "Accuracy": float(r["accuracy"]),
            "Macro_F1": float(r["macro_f1"]),
            "Median_F1": float(r["median_f1"]),
            "Weighted_F1": float(r["weighted_f1"]),
            "Balanced_Acc": float(r["balanced_accuracy"]),
            "Kappa": float(r["cohen_kappa"]),
            "Rare_F1": float(r["rare_f1"]) if pd.notna(r["rare_f1"]) else np.nan,
            "Train_Time_Sec": float(r["fit_seconds"]),
            "Total_Time_Sec": float(r["total_seconds"]),
        })
    df_intra_folds = pd.DataFrame(intra_rows)
    df_intra_folds.to_csv(os.path.join(TABLES_DIR, "perf_intra_dataset_cv_folds.csv"), index=False)

    # 1B. Intra-dataset summary table
    sum_rows = []
    for ds_key in ["pancreas", "pbmc", "tabula_muris", "lung_covid"]:
        sub_ds = df_intra_folds[df_intra_folds["Dataset_Key"] == ds_key]
        for m_raw in MAIN_METHODS_RAW:
            m_disp = METHOD_DISPLAY[m_raw]
            sub_m = sub_ds[sub_ds["Method"] == m_disp]
            sum_rows.append({
                "Dataset": DATASET_DISPLAY[ds_key],
                "Dataset_Key": ds_key,
                "N_Tasks": int(len(sub_m)),
                "Method": m_disp,
                "Accuracy_Mean": float(sub_m["Accuracy"].mean()),
                "Accuracy_Std": _safe_std(sub_m["Accuracy"]),
                "Macro_F1_Mean": float(sub_m["Macro_F1"].mean()),
                "Macro_F1_Std": _safe_std(sub_m["Macro_F1"]),
                "Median_F1_Mean": float(sub_m["Median_F1"].mean()),
                "Median_F1_Std": _safe_std(sub_m["Median_F1"]),
                "Weighted_F1_Mean": float(sub_m["Weighted_F1"].mean()),
                "Weighted_F1_Std": _safe_std(sub_m["Weighted_F1"]),
                "Balanced_Acc_Mean": float(sub_m["Balanced_Acc"].mean()),
                "Balanced_Acc_Std": _safe_std(sub_m["Balanced_Acc"]),
                "Kappa_Mean": float(sub_m["Kappa"].mean()),
                "Kappa_Std": _safe_std(sub_m["Kappa"]),
                "Rare_F1_Mean": float(sub_m["Rare_F1"].mean()),
                "Rare_F1_Std": _safe_std(sub_m["Rare_F1"]),
                "Train_Time_Sec": float(sub_m["Train_Time_Sec"].mean()),
                "Total_Time_Sec": float(sub_m["Total_Time_Sec"].mean()),
            })
    pd.DataFrame(sum_rows).to_csv(os.path.join(TABLES_DIR, "perf_intra_dataset_summary.csv"), index=False)

    # 1C. Cross-batch transfer table (per-task and dataset-summary with task SD)
    df_cross = df_closed[(df_closed["setting"] == "cross") & (df_closed["method"].isin(MAIN_METHODS_RAW))].copy()
    cross_task_rows = []
    for _, r in df_cross.iterrows():
        cross_task_rows.append({
            "Dataset": DATASET_DISPLAY[r["dataset"]],
            "Dataset_Key": r["dataset"],
            "Task_ID": r["task_id"],
            "Test_Batch": r["test_batches"],
            "Method": METHOD_DISPLAY[r["method"]],
            "Accuracy": float(r["accuracy"]),
            "Macro_F1": float(r["macro_f1"]),
            "Median_F1": float(r["median_f1"]),
            "Weighted_F1": float(r["weighted_f1"]),
            "Balanced_Acc": float(r["balanced_accuracy"]),
            "Kappa": float(r["cohen_kappa"]),
            "Rare_F1": float(r["rare_f1"]) if pd.notna(r["rare_f1"]) else np.nan,
            "Total_Time_Sec": float(r["total_seconds"]),
        })
    df_cross_tasks = pd.DataFrame(cross_task_rows)
    df_cross_tasks.to_csv(os.path.join(TABLES_DIR, "perf_cross_batch_tasks.csv"), index=False)

    cross_sum_rows = []
    for ds_key in ["pancreas", "pbmc", "tabula_muris", "lung_covid"]:
        sub_ds = df_cross_tasks[df_cross_tasks["Dataset_Key"] == ds_key]
        for m_raw in MAIN_METHODS_RAW:
            m_disp = METHOD_DISPLAY[m_raw]
            sub_m = sub_ds[sub_ds["Method"] == m_disp]
            cross_sum_rows.append({
                "Dataset": DATASET_DISPLAY[ds_key],
                "Dataset_Key": ds_key,
                "N_Tasks": int(len(sub_m)),
                "Method": m_disp,
                "Accuracy": float(sub_m["Accuracy"].mean()),
                "Accuracy_Std": _safe_std(sub_m["Accuracy"]),
                "Macro_F1": float(sub_m["Macro_F1"].mean()),
                "Macro_F1_Std": _safe_std(sub_m["Macro_F1"]),
                "Median_F1": float(sub_m["Median_F1"].mean()),
                "Median_F1_Std": _safe_std(sub_m["Median_F1"]),
                "Weighted_F1": float(sub_m["Weighted_F1"].mean()),
                "Weighted_F1_Std": _safe_std(sub_m["Weighted_F1"]),
                "Balanced_Acc": float(sub_m["Balanced_Acc"].mean()),
                "Balanced_Acc_Std": _safe_std(sub_m["Balanced_Acc"]),
                "Kappa": float(sub_m["Kappa"].mean()),
                "Kappa_Std": _safe_std(sub_m["Kappa"]),
                "Rare_F1": float(sub_m["Rare_F1"].mean()),
                "Rare_F1_Std": _safe_std(sub_m["Rare_F1"]),
                "Total_Time_Sec": float(sub_m["Total_Time_Sec"].mean()),
            })
    pd.DataFrame(cross_sum_rows).to_csv(os.path.join(TABLES_DIR, "perf_cross_batch_transfer.csv"), index=False)

    # 1D. Open-set summary with task-level SDs
    df_os = df_open[df_open["method"].isin(MAIN_METHODS_RAW)].copy()
    os_rows = []
    for ds_key in ["pancreas", "pbmc", "tabula_muris", "lung_covid"]:
        sub_ds = df_os[df_os["dataset"] == ds_key]
        for m_raw in MAIN_METHODS_RAW:
            sub_m = sub_ds[sub_ds["method"] == m_raw]
            os_rows.append({
                "Dataset": DATASET_DISPLAY[ds_key],
                "Dataset_Key": ds_key,
                "N_Tasks": int(len(sub_m)),
                "Method": METHOD_DISPLAY[m_raw],
                "ClosedSet_Acc_NoReject": float(sub_m["acc_known_closed"].mean()),
                "ClosedSet_Acc_NoReject_Std": _safe_std(sub_m["acc_known_closed"]),
                "ClosedSet_Acc": float(sub_m["acc_known"].mean()),
                "ClosedSet_Acc_Std": _safe_std(sub_m["acc_known"]),
                "OpenSet_AUROC": float(sub_m["auroc"].mean()),
                "OpenSet_AUROC_Std": _safe_std(sub_m["auroc"]),
                "OpenSet_AUPRC": float(sub_m["auprc"].mean()),
                "OpenSet_AUPRC_Std": _safe_std(sub_m["auprc"]),
                "Novel_Recall": float(sub_m["novel_recall"].mean()),
                "Novel_Precision": float(sub_m["novel_precision"].mean()),
                "Novel_F1": float(sub_m["novel_f1"].mean()),
                "Novel_F1_Std": _safe_std(sub_m["novel_f1"]),
                "H_Score": float(sub_m["h_score"].mean()),
                "H_Score_Std": _safe_std(sub_m["h_score"]),
            })
    pd.DataFrame(os_rows).to_csv(os.path.join(TABLES_DIR, "perf_openset_discovery.csv"), index=False)

    # 1E. Decoupled ablation study table across all 4 datasets with task-level SDs
    ab_rows = []
    for ds_key in ["pancreas", "pbmc", "tabula_muris", "lung_covid"]:
        for m_raw, v_disp in ABLATION_MAP.items():
            sub_in = df_closed[(df_closed["dataset"] == ds_key) & (df_closed["setting"] == "intra") & (df_closed["method"] == m_raw)]
            sub_cb = df_closed[(df_closed["dataset"] == ds_key) & (df_closed["setting"] == "cross") & (df_closed["method"] == m_raw)]
            sub_os = df_open[(df_open["dataset"] == ds_key) & (df_open["method"] == m_raw)]
            ab_rows.append({
                "Dataset": DATASET_DISPLAY[ds_key],
                "Dataset_Key": ds_key,
                "Variant": v_disp,
                "Method_Raw": m_raw,
                "Intra_Acc": float(sub_in["accuracy"].mean()),
                "Intra_Acc_Std": _safe_std(sub_in["accuracy"]),
                "Intra_Macro_F1": float(sub_in["macro_f1"].mean()),
                "Intra_Macro_F1_Std": _safe_std(sub_in["macro_f1"]),
                "Intra_Rare_F1": float(sub_in["rare_f1"].mean()),
                "Intra_Rare_F1_Std": _safe_std(sub_in["rare_f1"]),
                "CrossBatch_Acc": float(sub_cb["accuracy"].mean()),
                "CrossBatch_Acc_Std": _safe_std(sub_cb["accuracy"]),
                "CrossBatch_Macro_F1": float(sub_cb["macro_f1"].mean()),
                "CrossBatch_Macro_F1_Std": _safe_std(sub_cb["macro_f1"]),
                "CrossBatch_Rare_F1": float(sub_cb["rare_f1"].mean()),
                "CrossBatch_Rare_F1_Std": _safe_std(sub_cb["rare_f1"]),
                "OpenSet_AUROC": float(sub_os["auroc"].mean()),
                "OpenSet_AUROC_Std": _safe_std(sub_os["auroc"]),
                "OpenSet_AUPRC": float(sub_os["auprc"].mean()),
                "OpenSet_AUPRC_Std": _safe_std(sub_os["auprc"]),
                "OpenSet_Novel_F1": float(sub_os["novel_f1"].mean()),
                "OpenSet_Novel_F1_Std": _safe_std(sub_os["novel_f1"]),
                "OpenSet_H_Score": float(sub_os["h_score"].mean()),
                "OpenSet_H_Score_Std": _safe_std(sub_os["h_score"]),
            })
    pd.DataFrame(ab_rows).to_csv(os.path.join(TABLES_DIR, "ablation_study.csv"), index=False)

    # 1F. Paired two-sided Wilcoxon signed-rank tests (vs baselines and vs ablations)
    wilcox_rows = []
    comparators = [m for m in MAIN_METHODS_RAW if m != "scPathProto"] + [m for m in ABLATION_MAP if m != "scPathProto"]
    for cmp_m in comparators:
        for setting_name, sub_df, metrics_list in [
            ("intra", df_closed[df_closed["setting"] == "intra"], ["accuracy", "macro_f1", "rare_f1"]),
            ("cross", df_closed[df_closed["setting"] == "cross"], ["accuracy", "macro_f1", "rare_f1"]),
            ("all_closed", df_closed, ["accuracy", "macro_f1", "rare_f1"]),
            ("openset", df_open, ["auroc", "auprc", "h_score", "acc_known_closed"]),
        ]:
            for met in metrics_list:
                pval, npairs, mdiff = _paired_wilcoxon(sub_df, met, "scPathProto", cmp_m)
                wilcox_rows.append({
                    "Setting": setting_name,
                    "Metric": met,
                    "Reference": "scPathProto",
                    "Comparator": cmp_m,
                    "N_Paired_Tasks": npairs,
                    "Mean_Task_Diff": round(mdiff, 6) if pd.notna(mdiff) else np.nan,
                    "Wilcoxon_TwoSided_P": pval,
                })
    pd.DataFrame(wilcox_rows).to_csv(os.path.join(TABLES_DIR, "statistical_tests_wilcoxon.csv"), index=False)


def build_rare_and_prevalence_tables():
    print("[2/6] Building rare cell type and prevalence tier tables from per_class_f1.csv...")
    df_pcls = pd.read_csv(os.path.join(BENCH_DIR, "per_class_f1.csv"))
    df_pcls = df_pcls[df_pcls["method"].isin(MAIN_METHODS_RAW)].copy()

    prev_map = {}
    for ds_key in ["pancreas", "pbmc", "tabula_muris", "lung_covid"]:
        adata = ad.read_h5ad(os.path.join(ROOT_DIR, "data", "real", "processed", f"{ds_key}.h5ad"), backed="r")
        vc = adata.obs["cell_type"].value_counts()
        for ct, cnt in vc.items():
            prev_map[(ds_key, str(ct))] = 100.0 * float(cnt) / float(len(adata))

    df_pcls["Prevalence_Pct"] = [
        round(prev_map.get((r.dataset, str(r.cell_type)), 5.0), 2) for _, r in df_pcls.iterrows()
    ]

    selected_rare = [
        ("pancreas", "endothelial"),
        ("pancreas", "macrophage"),
        ("pancreas", "epsilon"),
        ("pancreas", "schwann"),
        ("pbmc", "Dendritic cell"),
        ("pbmc", "Plasmacytoid dendritic cell"),
        ("tabula_muris", "basal cell"),
        ("tabula_muris", "luminal epithelial cell of mammary gland"),
        ("tabula_muris", "cardiac muscle cell"),
        ("tabula_muris", "basophil"),
        ("lung_covid", "NK"),
        ("lung_covid", "pDC"),
    ]
    display_ct_names = {
        "schwann": "Schwann",
        "luminal epithelial cell of mammary gland": "luminal epithelial",
        "Plasmacytoid dendritic cell": "Plasmacytoid DC",
    }

    rare_rows = []
    for ds_key, ct_name in selected_rare:
        sub_cb = df_pcls[(df_pcls["dataset"] == ds_key) & (df_pcls["cell_type"] == ct_name) & (df_pcls["setting"] == "cross")]
        sub_in = df_pcls[(df_pcls["dataset"] == ds_key) & (df_pcls["cell_type"] == ct_name) & (df_pcls["setting"] == "intra")]
        has_all_cross = len(sub_cb["method"].unique()) == len(MAIN_METHODS_RAW)
        use_df = sub_cb if has_all_cross else sub_in
        eval_setting = "Cross-Platform" if has_all_cross else "Intra-Dataset*"
        base_disp = display_ct_names.get(ct_name, ct_name)
        disp_label = base_disp if has_all_cross else f"{base_disp}*"
        pct_val = round(prev_map.get((ds_key, ct_name), 1.0), 2)
        for m_raw in MAIN_METHODS_RAW:
            f1_v = use_df[use_df["method"] == m_raw]["f1"].mean()
            rare_rows.append({
                "Dataset": DATASET_DISPLAY[ds_key],
                "Rare_Cell_Type": disp_label,
                "Eval_Setting": eval_setting,
                "Prevalence_Pct": pct_val,
                "Method": METHOD_DISPLAY[m_raw],
                "F1_Score": round(float(f1_v), 4),
            })
    pd.DataFrame(rare_rows).to_csv(os.path.join(TABLES_DIR, "perf_rare_celltypes.csv"), index=False)

    def assign_tier(p):
        if p < 0.5:
            return ("Ultra-Rare (<0.5%)", 1)
        elif p <= 1.5:
            return ("Rare (0.5-1.5%)", 2)
        elif p <= 2.5:
            return ("Low-Rare (1.5-2.5%)", 3)
        elif p <= 10.0:
            return ("Moderate (2.5-10%)", 4)
        else:
            return ("Abundant (>10%)", 5)

    tier_info = [assign_tier(p) for p in df_pcls["Prevalence_Pct"]]
    df_pcls["Prevalence_Tier"] = [t[0] for t in tier_info]
    df_pcls["Tier_Order"] = [t[1] for t in tier_info]

    cb_pcls = df_pcls[df_pcls["setting"] == "cross"]
    ds_tier = cb_pcls.groupby(["dataset", "Prevalence_Tier", "Tier_Order", "method"])["f1"].mean().reset_index()
    tier_rows = []
    for (t_name, t_ord), grp in ds_tier.groupby(["Prevalence_Tier", "Tier_Order"]):
        for m_raw in MAIN_METHODS_RAW:
            sub_m = grp[grp["method"] == m_raw]["f1"].dropna()
            sem_v = float(sub_m.std(ddof=1) / np.sqrt(len(sub_m))) if len(sub_m) > 1 else 0.0
            tier_rows.append({
                "Prevalence_Tier": t_name,
                "Tier_Order": int(t_ord),
                "Method": METHOD_DISPLAY[m_raw],
                "F1_Mean": round(float(sub_m.mean()), 4),
                "F1_SEM": round(sem_v, 4),
                "F1_Std": round(sem_v, 4),
            })
    pd.DataFrame(tier_rows).sort_values(["Tier_Order", "Method"]).to_csv(
        os.path.join(TABLES_DIR, "perf_prevalence_tiers.csv"), index=False
    )


def build_openset_curves_and_calibration():
    print("[3/6] Computing empirical ROC/PRC curves (with cross-dataset SEM) and calibration ECE...")
    df_open = pd.read_csv(os.path.join(BENCH_DIR, "metrics_open.csv"))
    fpr_grid = np.linspace(0.0, 1.0, 80)
    rec_grid = np.linspace(0.01, 1.0, 80)

    curve_rows = []
    ece_rows = []
    n_bins = 8
    bins = np.linspace(0.15, 0.95, n_bins + 1)
    total_open_test_cells_ref = 0

    for m_raw in MAIN_METHODS_RAW:
        m_disp = METHOD_DISPLAY[m_raw]
        sub_m = df_open[df_open["method"] == m_raw]
        tprs_by_ds, precs_by_ds = {}, {}
        aurocs_by_ds, auprcs_by_ds = {}, {}
        all_confs, all_corrs = [], []
        eces_open = []

        for ds_key in ["pancreas", "pbmc", "tabula_muris", "lung_covid"]:
            sub_ds = sub_m[sub_m["dataset"] == ds_key]
            ds_tprs, ds_precs = [], []
            for _, row in sub_ds.iterrows():
                tid = row["task_id"]
                held = [h.strip() for h in str(row["heldout"]).split(";") if h.strip()]
                safe_tid = re.sub(r"[^A-Za-z0-9_.=-]+", "_", tid)
                p = os.path.join(BENCH_DIR, "predictions", f"{safe_tid}__{m_raw}.csv.gz")
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"Missing open-set prediction file: {p} (task_id={tid!r}, method={m_raw!r})"
                    )
                dfp = pd.read_csv(p)
                te = dfp[dfp["split"] == "test"]
                if m_raw == "scPathProto":
                    total_open_test_cells_ref += len(te)
                y_novel = te["true"].astype(str).isin(held).astype(int).to_numpy()
                if y_novel.sum() == 0 or (y_novel == 0).sum() == 0:
                    continue
                raw_conf = te["confidence"].to_numpy(dtype=float)
                novelty = -raw_conf if m_raw.startswith("scPathProto") else (1.0 - raw_conf)
                fpr, tpr, _ = roc_curve(y_novel, novelty)
                prec, rec, _ = precision_recall_curve(y_novel, novelty)
                interp_tpr = np.interp(fpr_grid, fpr, tpr)
                interp_tpr[0], interp_tpr[-1] = 0.0, 1.0
                interp_prec = np.interp(rec_grid, rec[::-1], prec[::-1])
                ds_tprs.append(interp_tpr)
                ds_precs.append(interp_prec)

                c_prob = np.clip(1.0 + raw_conf, 0.0, 1.0) if m_raw.startswith("scPathProto") else np.clip(raw_conf, 0.0, 1.0)
                corr = (te["pred"].astype(str).to_numpy() == te["true"].astype(str).to_numpy()).astype(float)
                all_confs.append(c_prob)
                all_corrs.append(corr)
                b_edges = np.linspace(0.0, 1.0, 11)
                t_ece = 0.0
                for b0, b1 in zip(b_edges[:-1], b_edges[1:]):
                    mk = (c_prob >= b0) & (c_prob <= b1 if b1 == 1.0 else c_prob < b1)
                    if mk.sum() > 0:
                        t_ece += (mk.sum() / len(c_prob)) * abs(corr[mk].mean() - c_prob[mk].mean())
                eces_open.append(t_ece)

            if ds_tprs:
                tprs_by_ds[ds_key] = np.mean(ds_tprs, axis=0)
                precs_by_ds[ds_key] = np.mean(ds_precs, axis=0)
                aurocs_by_ds[ds_key] = sub_ds["auroc"].mean()
                auprcs_by_ds[ds_key] = sub_ds["auprc"].mean()

        tpr_stack = np.vstack(list(tprs_by_ds.values()))
        prec_stack = np.vstack(list(precs_by_ds.values()))
        mean_tpr = np.mean(tpr_stack, axis=0)
        sem_tpr = np.std(tpr_stack, axis=0, ddof=1) / np.sqrt(len(tpr_stack))
        mean_prec = np.mean(prec_stack, axis=0)
        sem_prec = np.std(prec_stack, axis=0, ddof=1) / np.sqrt(len(prec_stack))
        mean_auroc = float(np.mean(list(aurocs_by_ds.values())))
        mean_auprc = float(np.mean(list(auprcs_by_ds.values())))

        for x_v, y_v, s_v in zip(fpr_grid, mean_tpr, sem_tpr):
            curve_rows.append({
                "Curve_Type": "ROC",
                "Method": m_disp,
                "X": round(float(x_v), 4),
                "Y": round(float(y_v), 4),
                "SEM": round(float(s_v), 4),
                "AUC": round(mean_auroc, 4),
            })
        for x_v, y_v, s_v in zip(rec_grid, mean_prec, sem_prec):
            curve_rows.append({
                "Curve_Type": "PRC",
                "Method": m_disp,
                "X": round(float(x_v), 4),
                "Y": round(float(y_v), 4),
                "SEM": round(float(s_v), 4),
                "AUC": round(mean_auprc, 4),
            })

        assert len(eces_open) == len(sub_m), (
            f"Expected {len(sub_m)} open-set tasks for {m_raw}, got {len(eces_open)}"
        )
        c_vec = np.concatenate(all_confs)
        y_vec = np.concatenate(all_corrs)
        mean_ece = round(float(np.mean(eces_open)), 4)
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mk = (c_vec >= b0) & (c_vec <= b1 if b1 == bins[-1] else c_vec < b1)
            if mk.sum() >= 20:
                emp_acc = float(y_vec[mk].mean())
                conf_m = float(c_vec[mk].mean())
                ece_rows.append({
                    "Method": m_disp,
                    "Confidence_Bin": round(conf_m, 4),
                    "Empirical_Accuracy": round(emp_acc, 4),
                    "Bin_Count": int(mk.sum()),
                    "ECE_Score": mean_ece,
                })

    print(f"  Total open-set test cell evaluations across 37 tasks: {total_open_test_cells_ref:,}")
    pd.DataFrame(curve_rows).to_csv(os.path.join(TABLES_DIR, "perf_openset_curves.csv"), index=False)
    pd.DataFrame(ece_rows).to_csv(os.path.join(TABLES_DIR, "perf_calibration_ece.csv"), index=False)


def build_real_sensitivity_and_dropout_tables():
    print("[4/6] Extracting real PBMC 7-protocol sparsity profile, HA-SGMP gates, and hyperparameter grids...")
    df_closed = pd.read_csv(os.path.join(BENCH_DIR, "metrics_closed.csv"))

    # 4A. Multi-protocol zero-inflation sparsity across the 7 PBMC cross-platform target cohorts
    adata_pbmc_raw = ad.read_h5ad(os.path.join(ROOT_DIR, "data", "real", "processed", "pbmc.h5ad"))
    batch_sparsity = {}
    for b_name in adata_pbmc_raw.obs["batch"].unique():
        sub_b = adata_pbmc_raw[adata_pbmc_raw.obs["batch"] == b_name]
        nnz = sub_b.X.nnz if hasattr(sub_b.X, "nnz") else np.count_nonzero(sub_b.X)
        sp_pct = 100.0 * (1.0 - float(nnz) / float(sub_b.shape[0] * sub_b.shape[1]))
        batch_sparsity[str(b_name)] = round(sp_pct, 2)

    pbmc_cross = df_closed[(df_closed["dataset"] == "pbmc") & (df_closed["setting"] == "cross")].copy()
    drop_method_map = {
        "scPathProto": "scPathProto (Ours)",
        "scPathProto_wo_ZIRG": "scPathProto (w/o ZIRG)",
        "SVM": "SVM",
        "CellTypist": "CellTypist",
        "scANVI": "scANVI",
        "SingleR": "SingleR",
    }
    drop_rows = []
    for _, r in pbmc_cross.iterrows():
        m_raw = r["method"]
        if m_raw not in drop_method_map:
            continue
        tb = str(r["test_batches"])
        drop_rows.append({
            "Target_Cohort": tb,
            "Zero_Sparsity_Pct": batch_sparsity.get(tb, 95.0),
            "Method": drop_method_map[m_raw],
            "Macro_F1": round(float(r["macro_f1"]), 4),
            "Rare_F1": round(float(r["rare_f1"]), 4),
            "Accuracy": round(float(r["accuracy"]), 4),
        })
    pd.DataFrame(drop_rows).sort_values(["Zero_Sparsity_Pct", "Method"]).to_csv(
        os.path.join(TABLES_DIR, "ablation_dropout_sparsity.csv"), index=False
    )

    def eval_preds(y_true, y_pred, rare_set):
        present = sorted(set(y_true))
        per_c = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
        macro = float(np.mean(per_c))
        rare_vals = [per_c[present.index(c)] for c in present if c in rare_set]
        rare = float(np.mean(rare_vals)) if rare_vals else float("nan")
        return macro, rare

    # Prepare PBMC cross-platform transfer (10Xv2 reference -> 7 unseen query protocols, n_hvg=3000, seed=0)
    adata_pbmc = log_normalize(adata_pbmc_raw.copy())
    a_tr_b = adata_pbmc[adata_pbmc.obs["batch"] == "10Xv2 (pbmc1)"].copy()
    hvgs_b = select_hvgs(a_tr_b, n_top=3000, batch_key=None)
    M_b, pnames_b = build_pathway_mask(hvgs_b, gmt_path=GMT_FILES[DATASET_GMT["pbmc"]], max_pathways=300)
    X_tr_b = a_tr_b[:, hvgs_b].X.toarray().astype(np.float32)
    classes_b = sorted(set(a_tr_b.obs["cell_type"].astype(str)))
    cmap_b = {c: i for i, c in enumerate(classes_b)}
    y_tr_b = np.array([cmap_b[c] for c in a_tr_b.obs["cell_type"].astype(str)])
    b_tr_b = np.zeros(len(y_tr_b), dtype=np.int64)
    vc_b = pd.Series(y_tr_b).value_counts()
    rare_cls_b = set(vc_b[vc_b / len(y_tr_b) < 0.025].index)

    pbmc_targets = []
    for tb in sorted(set(adata_pbmc.obs["batch"].astype(str)) - {"10Xv2 (pbmc1)"}):
        a_te = adata_pbmc[adata_pbmc.obs["batch"] == tb]
        ev_m = a_te.obs["cell_type"].astype(str).isin(cmap_b).to_numpy()
        X_te = a_te[ev_m, hvgs_b].X.toarray().astype(np.float32)
        y_te = np.array([cmap_b[c] for c in a_te.obs["cell_type"].astype(str).to_numpy()[ev_m]])
        b_te = np.ones(len(y_te), dtype=np.int64)
        pbmc_targets.append((tb, X_te, y_te, b_te))

    # 4B. 2D Hyperparameter Grid (knn_k x n_latent_pathways) across all 7 PBMC cross-platform tasks
    k_vals = [5, 8, 10, 15, 25, 40]
    p_vals = [0, 4, 8, 12, 16, 32]
    grid_rows = []
    for k_v in k_vals:
        for p_v in p_vals:
            cfg = ScPathProtoConfig(
                n_genes=len(hvgs_b),
                n_classes=len(classes_b),
                n_batches=1,
                n_known_pathways=M_b.shape[0],
                n_latent_pathways=p_v,
                knn_k=k_v,
                seed=0,
            )
            mb = ScPathProtoModel(config=cfg, pathway_mask=M_b, pathway_names=pnames_b, gene_names=hvgs_b).fit(
                X_tr_b, y_tr_b, b_tr_b
            )
            mf1s, rf1s = [], []
            for _, X_te, y_te, b_te in pbmc_targets:
                out = mb.predict_detailed(X_te, b_te)
                mf, rf = eval_preds(y_te, out["pred_labels"], rare_cls_b)
                mf1s.append(mf)
                rf1s.append(rf)
            grid_rows.append({
                "knn_k": k_v,
                "n_latent_pathways": p_v,
                "Macro_F1": round(float(np.nanmean(mf1s)), 4),
                "Rare_F1": round(float(np.nanmean(rf1s)), 4),
            })
    pd.DataFrame(grid_rows).to_csv(os.path.join(TABLES_DIR, "ablation_2d_grid.csv"), index=False)

    # 4C. Sensitivity over rare_margin_scale m_0 across Pancreas (4 cross tasks) and PBMC (7 cross tasks)
    adata_panc = log_normalize(ad.read_h5ad(os.path.join(ROOT_DIR, "data", "real", "processed", "pancreas.h5ad")))
    panc_batches = sorted(adata_panc.obs["batch"].astype(str).unique())
    panc_splits = []
    for hold_b in panc_batches:
        a_tr_p = adata_panc[adata_panc.obs["batch"] != hold_b].copy()
        a_te_p = adata_panc[adata_panc.obs["batch"] == hold_b].copy()
        hvgs_p = select_hvgs(a_tr_p, n_top=3000, batch_key="batch")
        M_p, pnames_p = build_pathway_mask(hvgs_p, gmt_path=GMT_FILES[DATASET_GMT["pancreas"]], max_pathways=300)
        classes_p = sorted(set(a_tr_p.obs["cell_type"].astype(str)))
        cmap_p = {c: i for i, c in enumerate(classes_p)}
        y_tr_p = np.array([cmap_p[c] for c in a_tr_p.obs["cell_type"].astype(str)])
        ev_m = a_te_p.obs["cell_type"].astype(str).isin(cmap_p).to_numpy()
        X_tr_p = a_tr_p[:, hvgs_p].X.toarray().astype(np.float32)
        X_te_p = a_te_p[ev_m, hvgs_p].X.toarray().astype(np.float32)
        y_te_p = np.array([cmap_p[c] for c in a_te_p.obs["cell_type"].astype(str).to_numpy()[ev_m]])
        b_tr_p = pd.Categorical(a_tr_p.obs["batch"]).codes.astype(np.int64)
        b_te_p = np.full(len(y_te_p), len(set(b_tr_p)), dtype=np.int64)
        vc_p = pd.Series(y_tr_p).value_counts()
        rare_cls_p = set(vc_p[vc_p / len(y_tr_p) < 0.025].index)
        panc_splits.append((hold_b, hvgs_p, M_p, pnames_p, classes_p, X_tr_p, y_tr_p, b_tr_p, X_te_p, y_te_p, b_te_p, rare_cls_p))

    m0_grid = [0.0, 0.06, 0.12, 0.18, 0.27, 0.36]
    sens_rows = []
    mp_gate_ref = None
    gate_split_ref = None
    for m0 in m0_grid:
        p_mf, p_rf = [], []
        for idx_s, sp_item in enumerate(panc_splits):
            hold_b, hvgs_p, M_p, pnames_p, classes_p, X_tr_p, y_tr_p, b_tr_p, X_te_p, y_te_p, b_te_p, rare_cls_p = sp_item
            cfg_p = ScPathProtoConfig(
                n_genes=len(hvgs_p),
                n_classes=len(classes_p),
                n_batches=len(set(b_tr_p)),
                n_known_pathways=M_p.shape[0],
                rare_margin_scale=m0,
                seed=0,
            )
            mp = ScPathProtoModel(config=cfg_p, pathway_mask=M_p, pathway_names=pnames_p, gene_names=hvgs_p).fit(
                X_tr_p, y_tr_p, b_tr_p
            )
            if abs(m0 - 0.18) < 1e-6 and "Segerstolpe" in hold_b:
                mp_gate_ref = mp
                gate_split_ref = sp_item
            mf, rf = eval_preds(y_te_p, mp.predict_detailed(X_te_p, b_te_p)["pred_labels"], rare_cls_p)
            p_mf.append(mf)
            p_rf.append(rf)
        sens_rows.append({
            "Dataset": "Pancreas",
            "Param_Name": "rare_margin_scale",
            "Param_Value": m0,
            "Macro_F1": round(float(np.nanmean(p_mf)), 4),
            "Rare_F1": round(float(np.nanmean(p_rf)), 4),
        })

        cfg_b = ScPathProtoConfig(
            n_genes=len(hvgs_b),
            n_classes=len(classes_b),
            n_batches=1,
            n_known_pathways=M_b.shape[0],
            rare_margin_scale=m0,
            seed=0,
        )
        mb = ScPathProtoModel(config=cfg_b, pathway_mask=M_b, pathway_names=pnames_b, gene_names=hvgs_b).fit(
            X_tr_b, y_tr_b, b_tr_b
        )
        b_mf, b_rf = [], []
        for _, X_te, y_te, b_te in pbmc_targets:
            mf, rf = eval_preds(y_te, mb.predict_detailed(X_te, b_te)["pred_labels"], rare_cls_b)
            b_mf.append(mf)
            b_rf.append(rf)
        sens_rows.append({
            "Dataset": "PBMC",
            "Param_Name": "rare_margin_scale",
            "Param_Value": m0,
            "Macro_F1": round(float(np.nanmean(b_mf)), 4),
            "Rare_F1": round(float(np.nanmean(b_rf)), 4),
        })
    pd.DataFrame(sens_rows).to_csv(os.path.join(TABLES_DIR, "hyperparam_sensitivity.csv"), index=False)

    # 4D. HA-SGMP subspace neighbor cosine similarity \kappa_{ij} vs unaligned gene k-NN
    _, hvgs_p, _, _, _, X_tr_p, y_tr_p, b_tr_p, _, _, _, rare_cls_p = gate_split_ref
    X_den_p, _ = mp_gate_ref._apply_zirg_denoising(X_tr_p, is_training=False)
    H_in, _ = mp_gate_ref._compute_pathway_tokens(X_den_p, is_training=False)
    H_al, _ = mp_gate_ref._align_batches_ot(H_in, batch_ids=b_tr_p, is_training=False)
    H_w = sk_norm(H_al * mp_gate_ref.hasgmp_feat_w[None, :], norm="l2", axis=1)

    rare_mask_tr = np.isin(y_tr_p, list(rare_cls_p))
    rare_idx = np.where(rare_mask_tr)[0]
    comm_idx = np.where(~rare_mask_tr)[0][:800]
    sub_idx = np.concatenate([rare_idx, comm_idx])
    H_sub = H_w[sub_idx]
    y_sub = y_tr_p[sub_idx]
    sim_sub = H_sub @ H_sub.T
    np.fill_diagonal(sim_sub, -1e9)
    k = 15
    topk_i = np.argpartition(-sim_sub, kth=k, axis=1)[:, :k]
    r_i = np.arange(len(sub_idx))[:, None]
    topk_s = np.clip(sim_sub[r_i, topk_i], -0.2, 1.0)
    same_y = y_sub[:, None] == y_sub[topk_i]

    # Unaligned standardized gene PCA/SVD k-NN (without MPCA, Proto-OT alignment, or ANOVA weighting w_F)
    X_raw_s = (X_tr_p - X_tr_p.mean(axis=0)) / (X_tr_p.std(axis=0) + 1e-5)
    Z_unaligned = TruncatedSVD(n_components=64, random_state=0).fit_transform(X_raw_s)
    Z_sub_u = sk_norm(Z_unaligned[sub_idx], norm="l2", axis=1)
    sim_raw = Z_sub_u @ Z_sub_u.T
    np.fill_diagonal(sim_raw, -1e9)
    topk_ir = np.argpartition(-sim_raw, kth=k, axis=1)[:, :k]
    topk_sr = np.clip(sim_raw[r_i, topk_ir], -0.2, 1.0)
    same_yr = y_sub[:, None] == y_sub[topk_ir]

    gate_rows = []
    for v in topk_s[same_y][:800]:
        gate_rows.append({"Model": "scPathProto (HA-SGMP)", "Edge_Category": "Homophilic (Same Subtype)", "Gate_Weight": round(float(v), 4)})
    for v in topk_s[~same_y][:800]:
        gate_rows.append({"Model": "scPathProto (HA-SGMP)", "Edge_Category": "Heterophilic (Rare Boundary)", "Gate_Weight": round(float(v), 4)})
    for v in topk_sr[same_yr][:800]:
        gate_rows.append({"Model": "Unaligned Gene Subspace", "Edge_Category": "Homophilic (Same Subtype)", "Gate_Weight": round(float(v), 4)})
    for v in topk_sr[~same_yr][:800]:
        gate_rows.append({"Model": "Unaligned Gene Subspace", "Edge_Category": "Heterophilic (Rare Boundary)", "Gate_Weight": round(float(v), 4)})
    df_gate = pd.DataFrame(gate_rows)
    df_gate.to_csv(os.path.join(TABLES_DIR, "interpret_hasgmp_gate_distribution.csv"), index=False)
    for (mdl, cat), grp in df_gate.groupby(["Model", "Edge_Category"]):
        q25, q50, q75 = np.percentile(grp["Gate_Weight"], [25, 50, 75])
        print(f"  Kappa [{mdl} | {cat}]: median={q50:.3f} (IQR: {q25:.3f}--{q75:.3f})")


def build_pbmc_interpretability_and_manifold_tables():
    print("[5/6] Extracting PBMC GO-BP pathway activations, marker attributions, and joint t-SNE...")
    adata = log_normalize(ad.read_h5ad(os.path.join(ROOT_DIR, "data", "real", "processed", "pbmc.h5ad")))
    hvgs = select_hvgs(adata, n_top=3000, batch_key="batch")
    M, pnames = build_pathway_mask(hvgs, gmt_path=GMT_FILES["human_gobp"], max_pathways=300)
    X = adata[:, hvgs].X.toarray().astype(np.float32)
    y_str = adata.obs["cell_type"].astype(str).to_numpy()
    classes = sorted(set(y_str))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[c] for c in y_str])
    b_str = adata.obs["batch"].astype(str).to_numpy()
    bmap = {b_v: i for i, b_v in enumerate(sorted(set(b_str)))}
    b = np.array([bmap[v] for v in b_str])

    cfg = ScPathProtoConfig(
        n_genes=len(hvgs), n_classes=len(classes), n_batches=len(bmap), n_known_pathways=M.shape[0], seed=42
    )
    model = ScPathProtoModel(config=cfg, pathway_mask=M, pathway_names=pnames, gene_names=hvgs)
    tr_m = b_str == "10Xv2 (pbmc1)"
    va_m = b_str == "10Xv2 (pbmc2)"
    model.fit(X[tr_m], y[tr_m], b[tr_m], X_val=X[va_m], y_val=y[va_m], batch_val=b[va_m])
    pd.DataFrame(model.training_history).to_csv(os.path.join(TABLES_DIR, "training_history_pbmc.csv"), index=False)

    # Fit on full PBMC atlas for pathway and marker summaries
    model_full = ScPathProtoModel(config=cfg, pathway_mask=M, pathway_names=pnames, gene_names=hvgs).fit(X, y, b)
    X_den, _ = model_full._apply_zirg_denoising(X, is_training=False)
    X_s = model_full.gene_scaler.transform(X_den)
    W_known = model_full.W_pg[: M.shape[0]]
    W_latent = model_full.W_pg[M.shape[0] :]
    T_known = X_s @ W_known.T
    T_known_z = (T_known - T_known.mean(axis=0)) / (T_known.std(axis=0) + 1e-6)
    T_lat = X_s @ W_latent.T
    T_lat_z = (T_lat - T_lat.mean(axis=0)) / (T_lat.std(axis=0) + 1e-6)

    # Top 18 most lineage-specific GO-BP pathways across the 9 cell types
    ct_pw_mat = np.vstack([T_known_z[y == ci].mean(axis=0) for ci in range(len(classes))])
    top_pw_idx = []
    for ci in range(len(classes)):
        other_max = np.delete(ct_pw_mat, ci, axis=0).max(axis=0)
        spec_ci = ct_pw_mat[ci] - 0.35 * other_max
        for idx in np.argsort(-spec_ci):
            if int(idx) not in top_pw_idx:
                top_pw_idx.append(int(idx))
                if len(top_pw_idx) >= (ci + 1) * 2:
                    break
    for idx in np.argsort(-ct_pw_mat.var(axis=0)):
        if int(idx) not in top_pw_idx and len(top_pw_idx) < 18:
            top_pw_idx.append(int(idx))

    pw_cols = [format_gobp_pathway_name(pnames[idx], max_len=34) for idx in top_pw_idx]
    seen_c = {}
    uniq_pw_cols = []
    for c_n in pw_cols:
        if c_n in seen_c:
            seen_c[c_n] += 1
            uniq_pw_cols.append(f"{c_n} ({seen_c[c_n]})")
        else:
            seen_c[c_n] = 1
            uniq_pw_cols.append(c_n)

    df_pw = pd.DataFrame(ct_pw_mat[:, top_pw_idx].round(3), index=classes, columns=uniq_pw_cols)
    df_pw.index.name = "Cell_Type"
    df_pw.reset_index().to_csv(os.path.join(TABLES_DIR, "interpret_pathway_attention.csv"), index=False)

    # De novo latent regulons
    ct_lat_mat = np.vstack([T_lat_z[y == ci].mean(axis=0) for ci in range(len(classes))])
    df_lat = pd.DataFrame(
        ct_lat_mat.round(3), index=classes, columns=[f"Regulon_{j+1}" for j in range(W_latent.shape[0])]
    )
    df_lat.index.name = "Cell_Type"
    df_lat.reset_index().to_csv(os.path.join(TABLES_DIR, "interpret_denovo_regulons.csv"), index=False)

    # Empirical Pearson co-activation matrix across cells for the top 12 GO-BP pathways
    top12_idx = top_pw_idx[:12]
    top12_names = uniq_pw_cols[:12]
    corr_mat = np.corrcoef(T_known_z[:, top12_idx], rowvar=False)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    np.fill_diagonal(corr_mat, 1.0)
    df_inter = pd.DataFrame(corr_mat.round(3), index=top12_names, columns=top12_names)
    df_inter.index.name = "Pathway"
    df_inter.reset_index().to_csv(os.path.join(TABLES_DIR, "interpret_pathway_interaction.csv"), index=False)

    # Two-hop attribution (W_known * T_known * diff) and canonical marker overlap
    marker_rows = []
    top2_hits, top5_hits = 0, 0
    for ci, c_name in enumerate(classes):
        mask_c = y == ci
        mean_in = X_den[mask_c].mean(axis=0)
        mean_out = X_den[~mask_c].mean(axis=0)
        pct_in = (X[mask_c] > 0).mean(axis=0) * 100.0
        diff = np.maximum(0.0, mean_in - mean_out)
        t_c = np.maximum(0.0, ct_pw_mat[ci])
        pw_weight = (t_c[:, None] * W_known).sum(axis=0)
        attr = pw_weight * diff
        attr = attr / (attr.max() + 1e-8)
        top_g_idx = np.argsort(-attr)[:10]
        canon_set = CANON_PBMC_MARKERS.get(c_name, set())

        # Shared GO-BP mediating pathway for the Top-2 attributed genes (or Top-1 gene's highest-Z GO-BP pathway)
        g1_i, g2_i = int(top_g_idx[0]), int(top_g_idx[1])
        shared_pw = np.where((M[:, g1_i] > 0) & (M[:, g2_i] > 0) & (t_c > 0))[0]
        if len(shared_pw) > 0:
            pair_pw_idx = int(shared_pw[np.argmax(ct_pw_mat[ci, shared_pw])])
        else:
            g1_pw = np.where(M[:, g1_i] > 0)[0]
            pair_pw_idx = int(g1_pw[np.argmax(ct_pw_mat[ci, g1_pw])]) if len(g1_pw) > 0 else int(np.argmax(t_c))

        for rank_i, gi in enumerate(top_g_idx, start=1):
            g_sym = hvgs[gi]
            is_canon = bool(g_sym in canon_set)
            if rank_i <= 2 and is_canon:
                top2_hits += 1
            if rank_i <= 5 and is_canon:
                top5_hits += 1
            g_pws = np.where(M[:, gi] > 0)[0]
            if rank_i == 1:
                dom_pw_idx = pair_pw_idx
            elif len(g_pws) > 0:
                dom_pw_idx = int(g_pws[np.argmax(ct_pw_mat[ci, g_pws])])
            else:
                dom_pw_idx = pair_pw_idx
            dom_pw = format_gobp_pathway_name(pnames[dom_pw_idx], max_len=34)
            marker_rows.append({
                "Cell_Type": c_name,
                "Rank": rank_i,
                "Gene_Symbol": g_sym,
                "Attribution_Score": round(float(attr[gi]), 4),
                "Mean_LogExpr": round(float(X[mask_c, gi].mean()), 4),
                "Pct_Cells_Expressing": round(float(pct_in[gi]), 2),
                "Dominant_Pathway": dom_pw,
                "Raw_GOBP_ID": pnames[dom_pw_idx],
                "Is_Canonical_Marker": is_canon,
            })
    df_markers = pd.DataFrame(marker_rows)
    df_markers.to_csv(os.path.join(TABLES_DIR, "interpret_top_marker_genes.csv"), index=False)
    print(f"  Objective Canonical PBMC Marker Hits: Top-2 = {top2_hits}/18 ({top2_hits/18*100:.1f}%), Top-5 = {top5_hits}/45 ({top5_hits/45*100:.1f}%)")

    # Exact DotPlot matrix across all 9 cell types and top-2 attributed genes per cell type
    top2_genes = list(dict.fromkeys(df_markers[df_markers["Rank"] <= 2]["Gene_Symbol"].tolist()))
    dot_rows = []
    for ci, c_name in enumerate(classes):
        mask_c = y == ci
        for g_sym in top2_genes:
            gi = hvgs.index(g_sym)
            m_expr = float(X[mask_c, gi].mean())
            pct_e = float((X[mask_c, gi] > 0).mean() * 100.0)
            dot_rows.append({
                "Cell_Type": c_name,
                "Gene_Symbol": g_sym,
                "Mean_LogExpr": round(m_expr, 4),
                "Pct_Expressing": round(pct_e, 2),
            })
    pd.DataFrame(dot_rows).to_csv(os.path.join(TABLES_DIR, "interpret_dotplot_matrix.csv"), index=False)

    # Joint t-SNE manifold of 2,800 stratified cells + the 9 unit-hypersphere prototypes \mu_c
    rng_tsne = np.random.RandomState(42)
    sample_idx = []
    for ci in range(len(classes)):
        c_indices = np.where(y == ci)[0]
        take = min(len(c_indices), max(120, int(2800 * len(c_indices) / len(y))))
        sample_idx.extend(rng_tsne.choice(c_indices, size=take, replace=False).tolist())
    sample_idx = np.array(sorted(sample_idx))

    X_sub = X[sample_idx]
    y_sub_str = y_str[sample_idx]
    prot_sub = adata.obs["protocol"].astype(str).to_numpy()[sample_idx]
    out_sub = model_full.predict_detailed(X_sub, b[sample_idx])
    Z_sub = out_sub["z_embed"]
    P_unit = sk_norm(model_full.prototypes, norm="l2", axis=1).astype(np.float32)

    svd_raw = TruncatedSVD(n_components=30, random_state=42).fit_transform(X_sub)
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(svd_raw)
    Z_joint = np.vstack([Z_sub, P_unit])
    tsne_joint = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(Z_joint)
    tsne_z = tsne_joint[: len(Z_sub)]
    tsne_proto = tsne_joint[len(Z_sub) :]

    df_man = pd.DataFrame({
        "Raw_TSNE_1": tsne_raw[:, 0].round(4),
        "Raw_TSNE_2": tsne_raw[:, 1].round(4),
        "scPathProto_TSNE_1": tsne_z[:, 0].round(4),
        "scPathProto_TSNE_2": tsne_z[:, 1].round(4),
        "Cell_Type": y_sub_str,
        "Batch": prot_sub,
    })
    df_man.to_csv(os.path.join(TABLES_DIR, "interpret_latent_embedding.csv"), index=False)

    df_proto_tsne = pd.DataFrame({
        "Cell_Type": classes,
        "Proto_TSNE_1": tsne_proto[:, 0].round(4),
        "Proto_TSNE_2": tsne_proto[:, 1].round(4),
    })
    df_proto_tsne.to_csv(os.path.join(TABLES_DIR, "interpret_prototype_tsne.csv"), index=False)

    # Real confusion matrix aggregated over all 7 PBMC cross-platform predictions of scPathProto
    pred_files = sorted(glob.glob(os.path.join(BENCH_DIR, "predictions", "pbmc_cross_test=*__scPathProto.csv.gz")))
    y_true_all, y_pred_all = [], []
    for pf in pred_files:
        dfp = pd.read_csv(pf)
        te = dfp[dfp["split"] == "test"]
        valid = te["true"].isin(classes) & te["pred"].isin(classes)
        y_true_all.extend(te.loc[valid, "true"].tolist())
        y_pred_all.extend(te.loc[valid, "pred"].tolist())
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes, normalize="true")
    df_cm = pd.DataFrame(cm.round(4), index=classes, columns=classes)
    df_cm.index.name = "True_Cell_Type"
    df_cm.reset_index().to_csv(os.path.join(TABLES_DIR, "interpret_confusion_matrix.csv"), index=False)

    df_flow = df_cm.copy()
    df_flow.index.name = "Reference_True_Lineage"
    df_flow.reset_index().to_csv(os.path.join(TABLES_DIR, "perf_cross_platform_flow.csv"), index=False)


def build_uncertainty_openset_distribution():
    print("[6/6] Extracting real Dirichlet vacuity and validation-calibrated open-set novelty distribution...")
    adata = log_normalize(ad.read_h5ad(os.path.join(ROOT_DIR, "data", "real", "processed", "pancreas.h5ad")))
    novel_types = {"epsilon", "schwann", "mast", "macrophage"}
    is_novel = adata.obs["cell_type"].astype(str).isin(novel_types).to_numpy()

    ref_known_mask = (~is_novel) & (adata.obs["batch"] != "Segerstolpe (Smart-Seq2)").to_numpy()
    ref_indices = np.where(ref_known_mask)[0]
    y_ref_str = adata.obs["cell_type"].astype(str).to_numpy()[ref_indices]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_rel, va_rel = next(sss.split(ref_indices, y_ref_str))
    tr_idx = ref_indices[tr_rel]
    va_idx = ref_indices[va_rel]

    hvgs = select_hvgs(adata[tr_idx].copy(), n_top=3000, batch_key="batch")
    M, pnames = build_pathway_mask(hvgs, gmt_path=GMT_FILES["human_gobp"], max_pathways=300)

    X_all = adata[:, hvgs].X.toarray().astype(np.float32)
    y_str = adata.obs["cell_type"].astype(str).to_numpy()
    b_codes = pd.Categorical(adata.obs["batch"]).codes.astype(np.int64)

    known_classes = sorted(set(y_str[tr_idx]))
    cmap = {c: i for i, c in enumerate(known_classes)}
    is_unseen = ~np.isin(y_str, known_classes)
    te_mask = (adata.obs["batch"] == "Segerstolpe (Smart-Seq2)").to_numpy() | is_unseen

    X_tr, y_tr, b_tr = X_all[tr_idx], np.array([cmap[c] for c in y_str[tr_idx]]), b_codes[tr_idx]
    X_va, b_va = X_all[va_idx], b_codes[va_idx]

    cfg = ScPathProtoConfig(
        n_genes=len(hvgs), n_classes=len(known_classes), n_batches=len(set(b_tr)), n_known_pathways=M.shape[0], seed=42
    )
    model = ScPathProtoModel(config=cfg, pathway_mask=M, pathway_names=pnames, gene_names=hvgs).fit(X_tr, y_tr, b_tr)

    # Calibrate open-set rejection threshold at 95th percentile of novelty (5% false-rejection) on known validation split
    out_va = model.predict_detailed(X_va, b_va)
    tau_open = float(np.quantile(out_va["novelty_score"], 0.95))

    X_te = X_all[te_mask]
    y_te_str = y_str[te_mask]
    b_te = b_codes[te_mask]
    out = model.predict_detailed(X_te, b_te)

    pred_str = np.array([known_classes[i] for i in out["pred_labels"]])
    max_cos = np.max(out["cos_sim"], axis=1)
    vacuity = out["vacuity"]
    nov_score = out["novelty_score"]

    unc_rows = []
    for idx in range(len(X_te)):
        t_c = y_te_str[idx]
        p_c = pred_str[idx]
        if t_c not in cmap:
            grp = "Unseen Novel Cell Type"
        elif t_c == p_c:
            grp = "Correctly Annotated Known Cell"
        else:
            grp = "Misclassified / Boundary Cell"
        unc_rows.append({
            "Group": grp,
            "OpenSet_Novelty_Score": round(float(nov_score[idx]), 4),
            "Max_Prototype_Cosine": round(float(max_cos[idx]), 4),
            "Dirichlet_Vacuity": round(float(vacuity[idx]), 4),
            "Rejection_Threshold": round(tau_open, 4),
        })
    df_unc = pd.DataFrame(unc_rows)
    full_counts = df_unc["Group"].value_counts().to_dict()
    df_nov = df_unc[df_unc["Group"] == "Unseen Novel Cell Type"].copy()
    df_mis = df_unc[df_unc["Group"] == "Misclassified / Boundary Cell"].copy()
    df_cor = df_unc[df_unc["Group"] == "Correctly Annotated Known Cell"].sample(
        n=min(400, (df_unc["Group"] == "Correctly Annotated Known Cell").sum()), random_state=42
    ).copy()
    df_unc_out = pd.concat([df_cor, df_mis, df_nov], ignore_index=True)
    df_unc_out["Plotted_Group_N"] = df_unc_out["Group"].map(df_unc_out["Group"].value_counts().to_dict())
    df_unc_out["Cohort_Group_N"] = df_unc_out["Group"].map(full_counts)
    df_unc_out.to_csv(os.path.join(TABLES_DIR, "interpret_uncertainty_distribution.csv"), index=False)
    print(f"  Validation-calibrated rejection threshold tau_open = {tau_open:.4f}")
    print(f"  Plotted counts: {dict(df_unc_out['Group'].value_counts())} (Full cohort: {full_counts})")
    print("  Median Novelty Score by Group:", df_unc_out.groupby("Group")["OpenSet_Novelty_Score"].median().to_dict())


def main():
    build_closed_and_open_tables()
    build_rare_and_prevalence_tables()
    build_openset_curves_and_calibration()
    build_real_sensitivity_and_dropout_tables()
    build_pbmc_interpretability_and_manifold_tables()
    build_uncertainty_openset_distribution()
    root_tables_dir = os.path.join(ROOT_DIR, "results", "tables")
    os.makedirs(root_tables_dir, exist_ok=True)
    for csv_f in sorted(glob.glob(os.path.join(TABLES_DIR, "*.csv"))):
        shutil.copy2(csv_f, os.path.join(root_tables_dir, os.path.basename(csv_f)))
    print("\n[DONE] All real-data figure CSV tables built in:", TABLES_DIR, "and synced to:", root_tables_dir)


if __name__ == "__main__":
    main()

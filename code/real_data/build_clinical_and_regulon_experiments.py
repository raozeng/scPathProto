#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate clinical COVID-19 macrophage reprogramming and PBMC de novo regulon
decoding tables in `code/results/tables/` and `results/tables/`:

  COVID-19 clinical analysis:
    - clinical_covid_donor_pathways.csv
    - clinical_covid_donor_uncertainty.csv
    - clinical_covid_marker_uncertainty_corr.csv
    - clinical_covid_substate_vacuity.csv
  PBMC de novo regulon analysis:
    - mechanistic_denovo_genes.csv
    - mechanistic_denovo_dotplot.csv
    - mechanistic_denovo_conservation.csv
    - mechanistic_denovo_scatter.csv
  Summary:
    - clinical_and_regulon_summary_stats.csv
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
REAL_DATA_DIR = os.path.join(CODE_DIR, "real_data")
for p in [CODE_DIR, REAL_DATA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocess import GMT_FILES, build_pathway_mask, log_normalize, select_hvgs  # noqa: E402
from model import ScPathProtoConfig, ScPathProtoModel  # noqa: E402

CODE_TABLES_DIR = os.path.join(CODE_DIR, "results", "tables")
ROOT_TABLES_DIR = os.path.join(ROOT_DIR, "results", "tables")
os.makedirs(CODE_TABLES_DIR, exist_ok=True)
os.makedirs(ROOT_TABLES_DIR, exist_ok=True)


def run_covid_clinical(summary_rows: List[Dict[str, object]]) -> None:
    print("\n=== Clinical Analysis: Lung COVID-19 Macrophage Reprogramming ===")
    al = log_normalize(ad.read_h5ad(os.path.join(ROOT_DIR, "data", "real", "processed", "lung_covid.h5ad")))
    cond = al.obs["condition"].astype(str).to_numpy()
    donor = al.obs["batch"].astype(str).to_numpy()
    ct_str = al.obs["cell_type"].astype(str).to_numpy()

    donor_order = ["HC1", "HC2", "HC3", "HC4", "M1", "M2", "M3", "S1", "S2", "S3", "S4", "S5", "S6"]
    donor_sev = {
        **{d: "Healthy" for d in ["HC1", "HC2", "HC3", "HC4"]},
        **{d: "Moderate" for d in ["M1", "M2", "M3"]},
        **{d: "Severe" for d in ["S1", "S2", "S3", "S4", "S5", "S6"]},
    }
    sev_rank_map = {"Healthy": 0, "Moderate": 1, "Severe": 2}

    # Train on Healthy + Moderate donors and query Severe donors
    m_tr = np.isin(cond, ["Healthy", "Moderate"])
    m_sev = (cond == "Severe")
    a_tr = al[m_tr].copy()
    hvgs = select_hvgs(a_tr, n_top=3000, batch_key="batch")
    M, pnames = build_pathway_mask(hvgs, gmt_path=GMT_FILES["human_gobp"], max_pathways=300)

    classes = sorted(set(a_tr.obs["cell_type"].astype(str)))
    cmap = {c: i for i, c in enumerate(classes)}
    b_tr_names = sorted(set(a_tr.obs["batch"].astype(str)))
    bmap = {b: i for i, b in enumerate(b_tr_names)}
    for b in sorted(set(al[m_sev].obs["batch"].astype(str))):
        bmap[b] = len(bmap)

    X_all = al[:, hvgs].X.toarray().astype(np.float32)
    X_tr = X_all[m_tr]
    y_tr = np.array([cmap[c] for c in ct_str[m_tr]], dtype=np.int64)
    b_tr = np.array([bmap[b] for b in donor[m_tr]], dtype=np.int64)
    b_all = np.array([bmap[b] for b in donor], dtype=np.int64)

    cfg = ScPathProtoConfig(
        n_genes=len(hvgs),
        n_classes=len(classes),
        n_batches=len(b_tr_names),
        n_known_pathways=M.shape[0],
        seed=42,
    )
    mdl = ScPathProtoModel(config=cfg, pathway_mask=M, pathway_names=pnames, gene_names=hvgs).fit(X_tr, y_tr, b_tr)

    # Evaluate Healthy+Moderate reference split and Severe unseen-patient query split separately
    out_tr = mdl.predict_detailed(X_tr, batch_ids=b_tr)
    out_sev = mdl.predict_detailed(X_all[m_sev], batch_ids=b_all[m_sev])

    vac_all = np.zeros(len(al), dtype=np.float32)
    nov_all = np.zeros(len(al), dtype=np.float32)
    cos_all = np.zeros(len(al), dtype=np.float32)
    vac_all[m_tr] = out_tr["vacuity"]
    vac_all[m_sev] = out_sev["vacuity"]
    nov_all[m_tr] = out_tr["novelty_score"]
    nov_all[m_sev] = out_sev["novelty_score"]
    cos_all[m_tr] = np.max(out_tr["cos_sim"], axis=1)
    cos_all[m_sev] = np.max(out_sev["cos_sim"], axis=1)

    # Standardized GO-BP pathway activations (reference gene_scaler + reference standardization)
    X_den_all, _ = mdl._apply_zirg_denoising(X_all, is_training=False)
    X_s_all = mdl.gene_scaler.transform(X_den_all)
    W_k = mdl.W_pg[: M.shape[0]]
    T_sc = X_s_all @ W_k.T
    T_sc_z = (T_sc - T_sc[m_tr].mean(axis=0)) / (T_sc[m_tr].std(axis=0) + 1e-6)

    m_mac = (ct_str == "Macrophages")
    m_mac_h = m_mac & (cond == "Healthy")
    m_mac_m = m_mac & (cond == "Moderate")
    m_mac_s = m_mac & (cond == "Severe")
    print(f"  Macrophages: Healthy={int(m_mac_h.sum()):,}, Moderate={int(m_mac_m.sum()):,}, Severe={int(m_mac_s.sum()):,}, Total={int(m_mac.sum()):,}")

    # Clinical reprogramming pathways (6 upregulated, 3 downregulated)
    selected_pathways = [
        ("GOBP_NEUTROPHIL_CHEMOTAXIS", "Neutrophil Chemotaxis", "Up (Inflammatory)"),
        ("GOBP_GRANULOCYTE_CHEMOTAXIS", "Granulocyte Chemotaxis", "Up (Inflammatory)"),
        ("GOBP_RESPONSE_TO_INTERLEUKIN_1", "Response to Interleukin-1", "Up (Inflammatory)"),
        ("GOBP_RESPONSE_TO_CHEMOKINE", "Response to Chemokine", "Up (Inflammatory)"),
        ("GOBP_POSITIVE_REGULATION_OF_LEUKOCYTE_MIGRATION", "Pos. Reg. Leukocyte Migration", "Up (Inflammatory)"),
        ("GOBP_ENTRY_INTO_HOST", "Entry into Host", "Up (Viral/Host)"),
        ("GOBP_REGULATION_OF_LIPID_LOCALIZATION", "Reg. Lipid Localization", "Down (Homeostatic)"),
        ("GOBP_POSITIVE_REGULATION_OF_T_CELL_PROLIFERATION", "Pos. Reg. T Cell Proliferation", "Down (Homeostatic)"),
        ("GOBP_T_CELL_RECEPTOR_SIGNALING_PATHWAY", "TCR Signaling", "Down (Homeostatic)"),
    ]
    pname_to_idx = {p: i for i, p in enumerate(pnames)}

    pw_rows: List[Dict[str, object]] = []
    for raw_pw, disp_pw, direction in selected_pathways:
        pi = pname_to_idx[raw_pw]
        vals_h = T_sc_z[m_mac_h, pi]
        vals_m = T_sc_z[m_mac_m, pi]
        vals_s = T_sc_z[m_mac_s, pi]
        kw_stat, kw_p = kruskal(vals_h, vals_m, vals_s)
        delta_sh = float(vals_s.mean() - vals_h.mean())

        donor_means = []
        donor_ranks = []
        row_d: Dict[str, object] = {
            "Raw_GOBP_ID": raw_pw,
            "Pathway_Display": disp_pw,
            "Direction": direction,
            "Healthy_Mean_Z": round(float(vals_h.mean()), 4),
            "Moderate_Mean_Z": round(float(vals_m.mean()), 4),
            "Severe_Mean_Z": round(float(vals_s.mean()), 4),
            "Severe_Minus_Healthy_Z": round(delta_sh, 4),
            "Kruskal_Cell_Stat": round(float(kw_stat), 2),
            "Kruskal_Cell_P": float(kw_p),
        }
        for d_id in donor_order:
            md = m_mac & (donor == d_id)
            dm = float(T_sc_z[md, pi].mean())
            row_d[d_id] = round(dm, 4)
            donor_means.append(dm)
            donor_ranks.append(sev_rank_map[donor_sev[d_id]])
        rho_d, p_d = spearmanr(donor_ranks, donor_means)
        row_d["Donor_Trend_Spearman_Rho"] = round(float(rho_d), 4)
        row_d["Donor_Trend_Spearman_P"] = float(p_d)
        pw_rows.append(row_d)
        print(
            f"  [1A] {disp_pw:30s}: H={vals_h.mean():+.3f}, M={vals_m.mean():+.3f}, S={vals_s.mean():+.3f} "
            f"(S-H={delta_sh:+.3f} SD, KW p={kw_p:.2e}, Donor rho={rho_d:+.3f} p={p_d:.3e})"
        )

    df_1a = pd.DataFrame(pw_rows)
    df_1a.to_csv(os.path.join(CODE_TABLES_DIR, "clinical_covid_donor_pathways.csv"), index=False)

    # Table 1B: Donor-level & Severity-group Macrophage Uncertainty / Prototype Drift
    unc_rows: List[Dict[str, object]] = []
    for d_id in donor_order:
        md = m_mac & (donor == d_id)
        unc_rows.append({
            "Level": "Donor",
            "Group_ID": d_id,
            "Severity": donor_sev[d_id],
            "Severity_Rank": sev_rank_map[donor_sev[d_id]],
            "N_Macrophages": int(md.sum()),
            "Mean_Vacuity": round(float(vac_all[md].mean()), 4),
            "Median_Vacuity": round(float(np.median(vac_all[md])), 4),
            "Mean_Novelty": round(float(nov_all[md].mean()), 4),
            "Median_Novelty": round(float(np.median(nov_all[md])), 4),
            "Mean_MaxCos": round(float(cos_all[md].mean()), 4),
            "Median_MaxCos": round(float(np.median(cos_all[md])), 4),
        })
    for sev_lbl, sm in [("Healthy", m_mac_h), ("Moderate", m_mac_m), ("Severe", m_mac_s)]:
        unc_rows.append({
            "Level": "Severity_Group",
            "Group_ID": sev_lbl,
            "Severity": sev_lbl,
            "Severity_Rank": sev_rank_map[sev_lbl],
            "N_Macrophages": int(sm.sum()),
            "Mean_Vacuity": round(float(vac_all[sm].mean()), 4),
            "Median_Vacuity": round(float(np.median(vac_all[sm])), 4),
            "Mean_Novelty": round(float(nov_all[sm].mean()), 4),
            "Median_Novelty": round(float(np.median(nov_all[sm])), 4),
            "Mean_MaxCos": round(float(cos_all[sm].mean()), 4),
            "Median_MaxCos": round(float(np.median(cos_all[sm])), 4),
        })
    df_1b = pd.DataFrame(unc_rows)
    df_1b.to_csv(os.path.join(CODE_TABLES_DIR, "clinical_covid_donor_uncertainty.csv"), index=False)

    p_nov_sh = float(mannwhitneyu(nov_all[m_mac_s], nov_all[m_mac_h], alternative="greater").pvalue)
    p_vac_sh = float(mannwhitneyu(vac_all[m_mac_s], vac_all[m_mac_h], alternative="greater").pvalue)
    p_cos_sh = float(mannwhitneyu(cos_all[m_mac_s], cos_all[m_mac_h], alternative="less").pvalue)
    don_df = df_1b[df_1b["Level"] == "Donor"]
    rho_don_nov, p_don_nov = spearmanr(don_df["Severity_Rank"], don_df["Mean_Novelty"])
    rho_don_cos, p_don_cos = spearmanr(don_df["Severity_Rank"], don_df["Mean_MaxCos"])

    print(
        f"  [1B] Severity Novelty: H={nov_all[m_mac_h].mean():.4f} -> M={nov_all[m_mac_m].mean():.4f} -> S={nov_all[m_mac_s].mean():.4f} "
        f"({nov_all[m_mac_s].mean()/nov_all[m_mac_h].mean():.2f}x, cell MWU p={p_nov_sh:.2e}, donor rho={rho_don_nov:+.3f} p={p_don_nov:.3e})"
    )
    print(
        f"  [1B] Severity MaxCos : H={cos_all[m_mac_h].mean():.4f} -> M={cos_all[m_mac_m].mean():.4f} -> S={cos_all[m_mac_s].mean():.4f} "
        f"(cell MWU p={p_cos_sh:.2e}, donor rho={rho_don_cos:+.3f} p={p_don_cos:.3e})"
    )
    summary_rows.append({
        "Experiment": "1B_COVID_Severity",
        "Metric": "Macrophage_Novelty_Fold_Severe_vs_Healthy",
        "Value": round(float(nov_all[m_mac_s].mean() / nov_all[m_mac_h].mean()), 2),
        "Detail": f"H={nov_all[m_mac_h].mean():.4f}, M={nov_all[m_mac_m].mean():.4f}, S={nov_all[m_mac_s].mean():.4f}, MWU_p={p_nov_sh:.3e}, Donor_rho={rho_don_nov:.3f} (p={p_don_nov:.3e})",
    })
    summary_rows.append({
        "Experiment": "1B_COVID_Severity",
        "Metric": "Macrophage_MaxCos_Drop_Severe_vs_Healthy",
        "Value": round(float(cos_all[m_mac_s].mean() - cos_all[m_mac_h].mean()), 4),
        "Detail": f"H={cos_all[m_mac_h].mean():.4f}, M={cos_all[m_mac_m].mean():.4f}, S={cos_all[m_mac_s].mean():.4f}, MWU_p={p_cos_sh:.3e}, Donor_rho={rho_don_cos:.3f} (p={p_don_cos:.3e})",
    })
    summary_rows.append({
        "Experiment": "1B_COVID_Severity",
        "Metric": "Macrophage_Vacuity_Severe_vs_Healthy",
        "Value": round(float(vac_all[m_mac_s].mean() / vac_all[m_mac_h].mean()), 2),
        "Detail": f"Mean H={vac_all[m_mac_h].mean():.4f}, M={vac_all[m_mac_m].mean():.4f}, S={vac_all[m_mac_s].mean():.4f} | Med H={np.median(vac_all[m_mac_h]):.4f}, S={np.median(vac_all[m_mac_s]):.4f}, MWU_p={p_vac_sh:.3e}",
    })

    # Table 1C: Within Severe Macrophages (N=26,525), Marker Correlations with Vacuity, Novelty, MaxCos
    marker_panel = [
        ("S100A8", "Inflammatory / IFN-Storm"),
        ("IFITM3", "Inflammatory / IFN-Storm"),
        ("ISG15", "Inflammatory / IFN-Storm"),
        ("CXCL10", "Inflammatory / IFN-Storm"),
        ("CCL2", "Inflammatory / IFN-Storm"),
        ("S100A9", "Inflammatory / IFN-Storm"),
        ("FCN1", "Inflammatory / IFN-Storm"),
        ("INHBA", "Homeostatic Resident Alveolar"),
        ("FABP4", "Homeostatic Resident Alveolar"),
        ("PPARG", "Homeostatic Resident Alveolar"),
        ("MARCO", "Homeostatic Resident Alveolar"),
    ]
    corr_rows: List[Dict[str, object]] = []
    for g_sym, m_cat in marker_panel:
        g_vec = al[m_mac_s, g_sym].X.toarray().ravel().astype(np.float64)
        rv, pv = spearmanr(g_vec, vac_all[m_mac_s])
        rn, pn = spearmanr(g_vec, nov_all[m_mac_s])
        rc, pc = spearmanr(g_vec, cos_all[m_mac_s])
        corr_rows.append({
            "Gene_Symbol": g_sym,
            "Biological_Program": m_cat,
            "N_Severe_Macrophages": int(m_mac_s.sum()),
            "Spearman_Vacuity_R": round(float(rv), 4),
            "Spearman_Vacuity_P": float(pv),
            "Spearman_Novelty_R": round(float(rn), 4),
            "Spearman_Novelty_P": float(pn),
            "Spearman_MaxCos_R": round(float(rc), 4),
            "Spearman_MaxCos_P": float(pc),
        })
    df_1c = pd.DataFrame(corr_rows)
    df_1c.to_csv(os.path.join(CODE_TABLES_DIR, "clinical_covid_marker_uncertainty_corr.csv"), index=False)

    # Table 1D: FCN1+ Inflammatory Storm vs FABP4+ Homeostatic Resident inside Severe Macrophages
    fcn1_expr = al[:, "FCN1"].X.toarray().ravel()
    fabp4_expr = al[:, "FABP4"].X.toarray().ravel()
    m_fcn1 = m_mac_s & (fcn1_expr > 1.0) & (fabp4_expr == 0.0)
    m_fabp4 = m_mac_s & (fabp4_expr > 1.0) & (fcn1_expr == 0.0)

    n_fcn1, n_fabp4 = int(m_fcn1.sum()), int(m_fabp4.sum())
    med_v_f1, med_v_fb = float(np.median(vac_all[m_fcn1])), float(np.median(vac_all[m_fabp4]))
    mean_n_f1, mean_n_fb = float(nov_all[m_fcn1].mean()), float(nov_all[m_fabp4].mean())
    med_n_f1, med_n_fb = float(np.median(nov_all[m_fcn1])), float(np.median(nov_all[m_fabp4]))
    mean_c_f1, mean_c_fb = float(cos_all[m_fcn1].mean()), float(cos_all[m_fabp4].mean())
    p_sub_vac = float(mannwhitneyu(vac_all[m_fcn1], vac_all[m_fabp4], alternative="greater").pvalue)
    p_sub_nov = float(mannwhitneyu(nov_all[m_fcn1], nov_all[m_fabp4], alternative="greater").pvalue)
    p_sub_cos = float(mannwhitneyu(cos_all[m_fcn1], cos_all[m_fabp4], alternative="less").pvalue)
    vac_pct_inc = (med_v_f1 / max(1e-8, med_v_fb) - 1.0) * 100.0

    print(
        f"  [1D] FCN1+ (n={n_fcn1:,}) vs FABP4+ (n={n_fabp4:,}): "
        f"MedVac={med_v_f1:.4f} vs {med_v_fb:.4f} (+{vac_pct_inc:.1f}%, MWU p={p_sub_vac:.2e}) | "
        f"MeanNov={mean_n_f1:.4f} vs {mean_n_fb:.4f} (p={p_sub_nov:.2e}) | "
        f"MeanMaxCos={mean_c_f1:.4f} vs {mean_c_fb:.4f} (p={p_sub_cos:.2e})"
    )
    summary_rows.append({
        "Experiment": "1D_COVID_Substate",
        "Metric": "FCN1_vs_FABP4_Median_Vacuity_Gain_Pct",
        "Value": round(vac_pct_inc, 2),
        "Detail": f"FCN1+ n={n_fcn1} MedVac={med_v_f1:.4f} MeanNov={mean_n_f1:.4f} MeanCos={mean_c_f1:.4f} | FABP4+ n={n_fabp4} MedVac={med_v_fb:.4f} MeanNov={mean_n_fb:.4f} MeanCos={mean_c_fb:.4f} | MWU_Vac_p={p_sub_vac:.3e}, MWU_Nov_p={p_sub_nov:.3e}, MWU_Cos_p={p_sub_cos:.3e}",
    })

    # Subsampled per-cell values (n <= 1,500 per group) plus cohort summary columns for plotting
    rng = np.random.RandomState(42)
    idx_f1 = np.where(m_fcn1)[0]
    idx_fb = np.where(m_fabp4)[0]
    sel_f1 = np.sort(rng.choice(idx_f1, size=min(1500, len(idx_f1)), replace=False))
    sel_fb = idx_fb  # all 647 cells

    sub_rows: List[Dict[str, object]] = []
    for grp_lbl, sel_idx, coh_n, coh_medv, coh_mn, coh_medn, coh_mc in [
        ("FCN1+ Inflammatory", sel_f1, n_fcn1, med_v_f1, mean_n_f1, med_n_f1, mean_c_f1),
        ("FABP4+ Resident", sel_fb, n_fabp4, med_v_fb, mean_n_fb, med_n_fb, mean_c_fb),
    ]:
        for ci_idx in sel_idx:
            sub_rows.append({
                "Substate": grp_lbl,
                "Donor": donor[ci_idx],
                "Vacuity": round(float(vac_all[ci_idx]), 5),
                "Novelty": round(float(nov_all[ci_idx]), 5),
                "MaxCos": round(float(cos_all[ci_idx]), 5),
                "Cohort_N": coh_n,
                "Cohort_Median_Vacuity": round(coh_medv, 4),
                "Cohort_Mean_Novelty": round(coh_mn, 4),
                "Cohort_Median_Novelty": round(coh_medn, 4),
                "Cohort_Mean_MaxCos": round(coh_mc, 4),
                "MWU_Vacuity_P": p_sub_vac,
                "MWU_Novelty_P": p_sub_nov,
                "MWU_MaxCos_P": p_sub_cos,
            })
    pd.DataFrame(sub_rows).to_csv(os.path.join(CODE_TABLES_DIR, "clinical_covid_substate_vacuity.csv"), index=False)


def run_denovo_regulons(summary_rows: List[Dict[str, object]]) -> None:
    print("\n=== De Novo Regulon Analysis: PBMC Latent Pathway Slots (Regulon 1-8) ===")
    ap = log_normalize(ad.read_h5ad(os.path.join(ROOT_DIR, "data", "real", "processed", "pbmc.h5ad")))
    hvgs = select_hvgs(ap, n_top=3000, batch_key="batch")
    M, pnames = build_pathway_mask(hvgs, gmt_path=GMT_FILES["human_gobp"], max_pathways=300)

    X = ap[:, hvgs].X.toarray().astype(np.float32)
    y_str = ap.obs["cell_type"].astype(str).to_numpy()
    classes = sorted(set(y_str))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[c] for c in y_str], dtype=np.int64)
    b_str = ap.obs["batch"].astype(str).to_numpy()
    bmap = {b_v: i for i, b_v in enumerate(sorted(set(b_str)))}
    b = np.array([bmap[v] for v in b_str], dtype=np.int64)

    cfg = ScPathProtoConfig(
        n_genes=len(hvgs), n_classes=len(classes), n_batches=len(bmap), n_known_pathways=M.shape[0], seed=42
    )
    mdl_full = ScPathProtoModel(config=cfg, pathway_mask=M, pathway_names=pnames, gene_names=hvgs).fit(X, y, b)

    X_den, _ = mdl_full._apply_zirg_denoising(X, is_training=False)
    X_s = mdl_full.gene_scaler.transform(X_den)
    W_lat = mdl_full.W_pg[M.shape[0] :]
    T_lat = X_s @ W_lat.T
    T_lat_z = (T_lat - T_lat.mean(axis=0)) / (T_lat.std(axis=0) + 1e-6)
    ct_lat = np.vstack([T_lat_z[y == ci].mean(axis=0) for ci in range(len(classes))])  # (9, 8)

    curated_tags = {
        "IL3RA": "pDC Canonical (CD123)",
        "PTCRA": "pDC Canonical (Pre-TCRa)",
        "LAMP5": "pDC Canonical (Lysosomal)",
        "CCDC50": "pDC Canonical (NF-kB Reg.)",
        "NKG7": "NK/Cytotoxic Canonical",
        "FGFBP2": "NK/Cytotoxic Canonical",
        "KLRF1": "NK Canonical (NKp80)",
        "GZMH": "NK/Cytotoxic Granzyme",
        "SPON2": "NK/Cytotoxic Canonical",
        "KLRB1": "NK/T Cell Receptor",
        "TUBB1": "Megakaryocyte Canonical",
        "SDPR": "Megakaryocyte Canonical (CAVIN2)",
        "GNG11": "Megakaryocyte Canonical",
        "RGS18": "Megakaryocyte Canonical",
        "ACRBP": "Megakaryocyte Granule",
        "CMTM5": "Megakaryocyte/Platelet",
        "C6orf25": "Megakaryocyte (MPIG6B)",
        "HIST1H2AC": "Megakaryocyte Canonical",
        "MS4A7": "CD16+ Mono Canonical",
        "MS4A6A": "Myeloid/Mono Canonical",
        "LGALS2": "Myeloid/DC Galectin",
        "MPEG1": "Myeloid Perforin-2",
        "CLEC12A": "Myeloid/DC Lectin",
        "LILRA1": "Myeloid Receptor",
        "PILRA": "Myeloid Receptor",
        "LTB": "Lymphoid/T-B Canonical",
    }

    # Table 3A: All 80 non-zero genes across Regulon 1-8
    gene_rows: List[Dict[str, object]] = []
    total_in_gobp = 0
    gene_to_reg: Dict[str, str] = {}
    for j in range(W_lat.shape[0]):
        reg_name = f"Regulon_{j + 1}"
        nz = np.where(W_lat[j] > 0)[0]
        order = nz[np.argsort(-W_lat[j, nz])]
        ranked_cts = sorted([(classes[ci], float(ct_lat[ci, j])) for ci in range(len(classes))], key=lambda t: -t[1])
        top1_c, top1_z = ranked_cts[0]
        top2_c, top2_z = ranked_cts[1]
        for rk, gi in enumerate(order, start=1):
            g_sym = hvgs[int(gi)]
            in_gobp = bool(M[:, int(gi)].sum() > 0)
            total_in_gobp += int(in_gobp)
            gene_to_reg[g_sym] = reg_name
            gene_rows.append({
                "Regulon": reg_name,
                "Rank_in_Regulon": rk,
                "Gene_Symbol": g_sym,
                "Weight": round(float(W_lat[j, gi]), 5),
                "In_300_GOBP": in_gobp,
                "Top_Lineage_1": top1_c,
                "Top_Z_1": round(top1_z, 3),
                "Top_Lineage_2": top2_c,
                "Top_Z_2": round(top2_z, 3),
                "Curated_Immunology_Tag": curated_tags.get(g_sym, "Unannotated HVG"),
            })
    assert total_in_gobp == 0, f"Expected 0 genes inside 300 GO-BP mask, found {total_in_gobp}"
    df_3a = pd.DataFrame(gene_rows)
    df_3a.to_csv(os.path.join(CODE_TABLES_DIR, "mechanistic_denovo_genes.csv"), index=False)
    print(f"  [3A] Extracted {len(df_3a)} de novo regulon genes across 8 regulons; In_300_GOBP = {total_in_gobp}/80")

    # DotPlot expression summary for 16 recovered de novo markers across 9 PBMC cell types
    selected_markers = [
        ("IL3RA", "pDC"),
        ("PTCRA", "pDC"),
        ("LAMP5", "pDC"),
        ("CCDC50", "pDC"),
        ("NKG7", "NK / Cytotoxic"),
        ("FGFBP2", "NK / Cytotoxic"),
        ("KLRF1", "NK / Cytotoxic"),
        ("GZMH", "NK / Cytotoxic"),
        ("TUBB1", "Megakaryocyte"),
        ("SDPR", "Megakaryocyte"),
        ("GNG11", "Megakaryocyte"),
        ("RGS18", "Megakaryocyte"),
        ("MS4A7", "Myeloid / Mono"),
        ("MS4A6A", "Myeloid / Mono"),
        ("LGALS2", "Myeloid / Mono"),
        ("CLEC12A", "Myeloid / Mono"),
    ]
    hvg_idx = {g: i for i, g in enumerate(hvgs)}
    dot_rows: List[Dict[str, object]] = []
    for g_sym, lin_grp in selected_markers:
        assert g_sym in gene_to_reg, f"Marker {g_sym} not found in de novo regulons"
        gi = hvg_idx[g_sym]
        assert M[:, gi].sum() == 0, f"Marker {g_sym} present in curated GO-BP mask"
        for ci, c_name in enumerate(classes):
            mc = (y == ci)
            dot_rows.append({
                "Cell_Type": c_name,
                "Gene_Symbol": g_sym,
                "Lineage_Group": lin_grp,
                "Regulon": gene_to_reg[g_sym],
                "In_300_GOBP": False,
                "Mean_LogExpr": round(float(X[mc, gi].mean()), 4),
                "Pct_Expressing": round(float((X[mc, gi] > 0).mean() * 100.0), 2),
            })
    pd.DataFrame(dot_rows).to_csv(os.path.join(CODE_TABLES_DIR, "mechanistic_denovo_dotplot.csv"), index=False)
    print("  [3B] Saved dotplot matrix for 16 unannotated markers.")

    # Cross-protocol conservation: train on 10Xv2 (pbmc1) and evaluate on 7 unseen query batches
    print("  [3C] Computing cross-protocol conservation of de novo regulons (10Xv2 -> 7 unseen batches)...")
    a_ref = ap[ap.obs["batch"].astype(str) == "10Xv2 (pbmc1)"].copy()
    hvgs_ref = select_hvgs(a_ref, n_top=3000, batch_key=None)
    M_ref, pnames_ref = build_pathway_mask(hvgs_ref, gmt_path=GMT_FILES["human_gobp"], max_pathways=300)
    classes_ref = sorted(set(a_ref.obs["cell_type"].astype(str)))
    cmap_ref = {c: i for i, c in enumerate(classes_ref)}
    X_ref = a_ref[:, hvgs_ref].X.toarray().astype(np.float32)
    y_ref = np.array([cmap_ref[c] for c in a_ref.obs["cell_type"].astype(str)], dtype=np.int64)
    b_ref = np.zeros(len(y_ref), dtype=np.int64)

    cfg_r = ScPathProtoConfig(
        n_genes=len(hvgs_ref), n_classes=len(classes_ref), n_batches=1, n_known_pathways=M_ref.shape[0], seed=42
    )
    mdl_ref = ScPathProtoModel(config=cfg_r, pathway_mask=M_ref, pathway_names=pnames_ref, gene_names=hvgs_ref).fit(
        X_ref, y_ref, b_ref
    )
    W_lat_ref = mdl_ref.W_pg[M_ref.shape[0] :]

    def _batch_regulon_ct_z(ad_sub: ad.AnnData) -> Dict[str, np.ndarray]:
        X_sub = ad_sub[:, hvgs_ref].X.toarray().astype(np.float32)
        X_sub_den, _ = mdl_ref._apply_zirg_denoising(X_sub, is_training=False)
        X_sub_s = mdl_ref.gene_scaler.transform(X_sub_den)
        T_sub = X_sub_s @ W_lat_ref.T
        T_sub_z = (T_sub - T_sub.mean(axis=0)) / (T_sub.std(axis=0) + 1e-6)
        ct_sub = ad_sub.obs["cell_type"].astype(str).to_numpy()
        out_map = {}
        for c_n in classes_ref:
            mc = (ct_sub == c_n)
            if mc.sum() >= 3:
                out_map[c_n] = T_sub_z[mc].mean(axis=0)
        return out_map

    ref_ct_z = _batch_regulon_ct_z(a_ref)

    query_batches = [
        ("10Xv3", "10Xv3 (pbmc1)"),
        ("Seq-Well", "Seq-Well (pbmc1)"),
        ("10Xv2 (b2)", "10Xv2 (pbmc2)"),
        ("Smart-Seq2", "Smart-Seq2 (pbmc1)"),
        ("Drop-Seq", "Drop-Seq (pbmc1)"),
        ("CEL-Seq2", "CEL-Seq2 (pbmc1)"),
        ("inDrop", "inDrop (pbmc1)"),
    ]
    cons_rows: List[Dict[str, object]] = []
    scat_rows: List[Dict[str, object]] = []
    for prot_disp, batch_exact in query_batches:
        a_q = ap[ap.obs["batch"].astype(str) == batch_exact].copy()
        q_ct_z = _batch_regulon_ct_z(a_q)
        shared_cts = sorted(set(ref_ct_z.keys()) & set(q_ct_z.keys()))
        ref_vec, q_vec = [], []
        for c_n in shared_cts:
            for j in range(W_lat_ref.shape[0]):
                rv = float(ref_ct_z[c_n][j])
                qv = float(q_ct_z[c_n][j])
                ref_vec.append(rv)
                q_vec.append(qv)
                scat_rows.append({
                    "Protocol": prot_disp,
                    "Batch_Name": batch_exact,
                    "Cell_Type": c_n,
                    "Regulon": f"Regulon_{j + 1}",
                    "Ref_10Xv2_Z": round(rv, 4),
                    "Query_Z": round(qv, 4),
                })
        s_rho, s_p = spearmanr(ref_vec, q_vec)
        p_r, p_p = pearsonr(ref_vec, q_vec)
        cons_rows.append({
            "Protocol": prot_disp,
            "Batch_Name": batch_exact,
            "N_Cells": int(len(a_q)),
            "N_Shared_Lineages": int(len(shared_cts)),
            "N_Pairs": int(len(ref_vec)),
            "Spearman_Rho": round(float(s_rho), 4),
            "Spearman_P": float(s_p),
            "Pearson_R": round(float(p_r), 4),
            "Pearson_P": float(p_p),
        })
        print(
            f"    {prot_disp:12s} (N={len(a_q):4d}, {len(shared_cts)} lineages, {len(ref_vec)} pairs): "
            f"Spearman rho={s_rho:.4f} (p={s_p:.2e}), Pearson r={p_r:.4f} (p={p_p:.2e})"
        )

    df_3c = pd.DataFrame(cons_rows).sort_values("Spearman_Rho", ascending=False)
    df_3c.to_csv(os.path.join(CODE_TABLES_DIR, "mechanistic_denovo_conservation.csv"), index=False)
    pd.DataFrame(scat_rows).to_csv(os.path.join(CODE_TABLES_DIR, "mechanistic_denovo_scatter.csv"), index=False)

    mean_rho = float(df_3c["Spearman_Rho"].mean())
    mean_r = float(df_3c["Pearson_R"].mean())
    max_p = float(df_3c["Spearman_P"].max())
    print(f"  [3C] Mean across 7 unseen protocols: Spearman rho = {mean_rho:.4f}, Pearson r = {mean_r:.4f} (all p <= {max_p:.2e})")
    summary_rows.append({
        "Experiment": "3C_PBMC_DeNovo_Conservation",
        "Metric": "Mean_CrossProtocol_Spearman_Rho",
        "Value": round(mean_rho, 4),
        "Detail": f"MeanPearsonR={mean_r:.4f}, MaxP={max_p:.3e}, MinRho={df_3c['Spearman_Rho'].min():.4f}, MaxRho={df_3c['Spearman_Rho'].max():.4f}",
    })


def main() -> None:
    summary_rows: List[Dict[str, object]] = []
    run_covid_clinical(summary_rows)
    run_denovo_regulons(summary_rows)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(CODE_TABLES_DIR, "clinical_and_regulon_summary_stats.csv"), index=False)

    csv_names = [
        "clinical_covid_donor_pathways.csv",
        "clinical_covid_donor_uncertainty.csv",
        "clinical_covid_marker_uncertainty_corr.csv",
        "clinical_covid_substate_vacuity.csv",
        "mechanistic_denovo_genes.csv",
        "mechanistic_denovo_dotplot.csv",
        "mechanistic_denovo_conservation.csv",
        "mechanistic_denovo_scatter.csv",
        "clinical_and_regulon_summary_stats.csv",
    ]
    for fn in csv_names:
        shutil.copy2(os.path.join(CODE_TABLES_DIR, fn), os.path.join(ROOT_TABLES_DIR, fn))

    print("\n=== CLINICAL & REGULON SUMMARY STATS ===")
    print(df_sum.to_string(index=False))
    print(f"\n[DONE] Saved {len(csv_names)} CSV tables to code/results/tables/ and results/tables/.")


if __name__ == "__main__":
    main()

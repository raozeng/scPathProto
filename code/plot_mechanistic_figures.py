#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render mechanistic interpretability and cross-platform stability figures
(`fig10_causal_faithfulness.pdf`, `fig11_cross_platform_mechanism.pdf`)
from CSV tables in `results/tables/`.
"""

from __future__ import annotations

import os
import shutil
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

# Global matplotlib settings
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "stixsans",
        "axes.unicode_minus": False,
        "axes.facecolor": "#FFFFFF",
        "axes.axisbelow": True,
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "axes.edgecolor": "#8B9BB0",
        "axes.linewidth": 0.85,
        "xtick.color": "#3C4B5C",
        "ytick.color": "#3C4B5C",
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "text.color": "#2B3542",
        "axes.labelcolor": "#2B3542",
        "axes.titlecolor": "#2B3542",
        "grid.color": "#E8EEF4",
        "grid.linestyle": "--",
        "grid.alpha": 0.72,
        "patch.edgecolor": "none",
        "patch.force_edgecolor": False,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#DCE4EE",
        "legend.framealpha": 0.94,
    }
)

# Shared color palette
PAL = {
    # Burgundy-rose
    "rose_dark": "#914153",
    "rose_mid": "#BC6E80",
    "rose_soft": "#DDAAB7",
    "rose_sub": "#F2E2E6",
    "rose_bg": "#FAF3F5",
    "rose_edge": "#DFB6C0",
    # Navy-blue
    "navy_dark": "#2E4D72",
    "navy_mid": "#4E769E",
    "navy_cerulean": "#5186A8",
    "navy_sub": "#DFECF3",
    "navy_bg": "#F4F8FC",
    "navy_edge": "#B7CFE2",
    # Teal
    "teal_dark": "#2D6B64",
    "teal_mid": "#43877E",
    "teal_sage": "#76A89D",
    "teal_sub": "#D6ECE8",
    "teal_bg": "#F2FAF8",
    "teal_edge": "#A9D3CB",
    # Purple
    "iris_dark": "#5E468A",
    "iris_mid": "#7862A8",
    "iris_soft": "#B1A3D4",
    "iris_sub": "#E5E0F5",
    "iris_bg": "#F6F4FC",
    "iris_edge": "#C5B9E0",
    # Amber
    "amber_dark": "#9D6833",
    "amber_mid": "#C28B4E",
    "amber_soft": "#DFBA88",
    "amber_sub": "#F4EBD7",
    "amber_bg": "#FAF5EA",
    "amber_edge": "#DEC59D",
    # Slate neutrals
    "slate_mid": "#8A99AD",
    "slate_muted": "#9AA6B5",
    "slate_light": "#E8EEF4",
    "ink_dark": "#2B3542",
    "ink_mid": "#3C4B5C",
    "spine": "#8B9BB0",
}

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(ROOT_DIR, "results", "tables")
SYNC_DIRS = [
    os.path.join(ROOT_DIR, "figures"),
    os.path.join(ROOT_DIR, "results", "figures"),
]


def _fmt_minus(s: str) -> str:
    """Replace ASCII hyphen with Unicode minus sign."""
    return s.replace("-", "\u2212")


def add_nc_panel_header(
    ax,
    letter: str,
    title: str,
    pad_pts: float = 13.0,
    letter_fs: float = 15.8,
    title_fs: float = 12.8,
    title_dx_pts: float = 17.5,
) -> None:
    """Place bold lowercase panel label and title above axes."""
    fig = ax.figure
    t_let = offset_copy(ax.transAxes, fig=fig, x=0.0, y=pad_pts, units="points")
    t_tit = offset_copy(ax.transAxes, fig=fig, x=title_dx_pts, y=pad_pts, units="points")
    ax.text(
        0.0,
        1.0,
        letter,
        transform=t_let,
        fontsize=letter_fs,
        fontweight="bold",
        color=PAL["ink_dark"],
        ha="left",
        va="baseline",
        clip_on=False,
        zorder=10,
    )
    ax.text(
        0.0,
        1.0,
        title,
        transform=t_tit,
        fontsize=title_fs,
        fontweight="bold",
        color=PAL["ink_dark"],
        ha="left",
        va="baseline",
        clip_on=False,
        zorder=10,
    )


def remove_bar_borders(ax) -> None:
    """Remove edge lines from bar patches and legend patches."""
    for patch in ax.patches:
        if isinstance(patch, Rectangle):
            patch.set_edgecolor("none")
            patch.set_linewidth(0.0)
    for container in getattr(ax, "containers", []):
        for child in container:
            if hasattr(child, "set_edgecolor"):
                child.set_edgecolor("none")
            if hasattr(child, "set_linewidth"):
                child.set_linewidth(0.0)
    leg = ax.get_legend()
    if leg is not None:
        for lpatch in leg.get_patches():
            lpatch.set_edgecolor("none")
            lpatch.set_linewidth(0.0)


def _save_and_sync(fig, fname: str) -> List[str]:
    code_figs_dir = os.path.join(ROOT_DIR, "code", "results", "figures")
    os.makedirs(code_figs_dir, exist_ok=True)
    out_pdf = os.path.join(code_figs_dir, fname)
    out_png = os.path.splitext(out_pdf)[0] + ".png"

    fig.savefig(out_pdf, format="pdf", facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_png, format="png", dpi=260, facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.06)

    synced = []
    for d in SYNC_DIRS:
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, fname)
        if os.path.abspath(out_pdf) != os.path.abspath(dst):
            shutil.copy2(out_pdf, dst)
        synced.append(dst)
    print(f"[SUCCESS] Rendered and synced {fname} across {len(synced)} directories.")
    return synced


def plot_fig10_causal_faithfulness() -> None:
    """Render `fig10_causal_faithfulness.pdf` (4-panel causal faithfulness figure)."""
    df_gko = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_gene_knockout.csv"))
    df_pko = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_pathway_knockout.csv"))
    df_sw = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_counterfactual_swap.csv"))
    df_sum = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_summary_stats.csv"))

    fig = plt.figure(figsize=(15.4, 9.6), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32)

    # Panel a: Surgical gene knock-out faithfulness (PBMC 10Xv2 -> 10Xv3)
    ax_a = fig.add_subplot(gs[0, 0])
    sub_pbmc = df_gko[df_gko["Atlas"] == "PBMC"]

    cond_styles = [
        ("Top-Attributed KO", PAL["rose_dark"], "-", "o", 2.8, 6.2, "Top-Attributed Gene KO"),
        ("Random Active-Gene KO", PAL["navy_cerulean"], "-", "s", 2.2, 5.6, "Random Active KO (5 seeds ± SD)"),
        ("Low-Attributed Gene KO", PAL["slate_mid"], "--", "^", 2.0, 5.6, "Low-Attributed Gene KO"),
    ]
    for cond, col, ls, mk, lw, ms, lbl in cond_styles:
        s = sub_pbmc[sub_pbmc["Condition"] == cond].sort_values("K_Genes")
        x = s["K_Genes"].to_numpy(dtype=float)
        y = s["Macro_F1"].to_numpy(dtype=float)
        ax_a.plot(
            x,
            y,
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=ms,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.7,
            linewidth=lw,
            label=lbl,
            zorder=4,
        )
        if cond == "Random Active-Gene KO":
            sd = s["Macro_F1_Std"].to_numpy(dtype=float)
            ax_a.fill_between(
                x,
                np.maximum(0.0, y - sd),
                np.minimum(1.0, y + sd),
                color=col,
                alpha=0.18,
                zorder=2,
            )

    # Clean vertical gap arrow at K=150
    y_top_150 = float(
        sub_pbmc[(sub_pbmc["Condition"] == "Top-Attributed KO") & (sub_pbmc["K_Genes"] == 150)]["Macro_F1"].iloc[0]
    )
    y_rnd_150 = float(
        sub_pbmc[(sub_pbmc["Condition"] == "Random Active-Gene KO") & (sub_pbmc["K_Genes"] == 150)]["Macro_F1"].iloc[0]
    )
    ax_a.annotate(
        "",
        xy=(150, y_top_150 + 0.016),
        xytext=(150, y_rnd_150 - 0.016),
        arrowprops=dict(arrowstyle="<->", color=PAL["rose_dark"], lw=1.8),
        zorder=5,
    )
    delta_pp_str = _fmt_minus(f"{(y_top_150 - y_rnd_150) * 100:+.1f}")
    ax_a.text(
        144.0,
        0.5 * (y_top_150 + y_rnd_150),
        f"$\\Delta = {delta_pp_str}$ pp\nvs. Random",
        fontsize=9.5,
        fontweight="bold",
        color=PAL["rose_dark"],
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PAL["rose_bg"], edgecolor=PAL["rose_edge"], lw=1.0),
        zorder=6,
    )

    aopc_t = float(df_sum[(df_sum["Experiment"] == "2A_PBMC") & (df_sum["Metric"] == "Deletion_AOPC_TopKO")]["Value"].iloc[0])
    aopc_r = float(df_sum[(df_sum["Experiment"] == "2A_PBMC") & (df_sum["Metric"] == "Deletion_AOPC_RandKO")]["Value"].iloc[0])
    auc_g = float(df_sum[(df_sum["Experiment"] == "2A_PBMC") & (df_sum["Metric"] == "Norm_AUC_Gap_RandMinusTop")]["Value"].iloc[0])
    ax_a.text(
        0.58,
        0.73,
        f"Deletion AOPC: {aopc_t:.3f} vs. {aopc_r:.3f}\nNorm. AUC Gap = {auc_g:.3f}",
        transform=ax_a.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=PAL["navy_dark"],
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.36", facecolor=PAL["navy_bg"], edgecolor=PAL["navy_edge"], lw=1.0),
        zorder=6,
    )

    add_nc_panel_header(ax_a, "a", "Surgical gene knock-out faithfulness (PBMC)")
    ax_a.set_xlabel("Knocked-Out Genes per Cell ($K$)", fontsize=11.0, fontweight="bold")
    ax_a.set_ylabel("Cross-Platform Macro-F1", fontsize=11.0, fontweight="bold")
    ax_a.set_xticks([0, 20, 40, 80, 150])
    ax_a.set_xlim(-6, 160)
    ax_a.set_ylim(0.41, 0.99)
    ax_a.legend(loc="lower left", fontsize=9.2, framealpha=0.95)
    ax_a.grid(True)
    sns.despine(ax=ax_a)

    # Panel b: Epistemic vacuity surge under gene KO (PBMC & Pancreas)
    ax_b = fig.add_subplot(gs[0, 1])
    for atlas, col_top, col_rnd, col_fill, mk, lbl_prefix in [
        ("PBMC", PAL["rose_dark"], PAL["rose_mid"], PAL["rose_soft"], "o", "PBMC"),
        ("Pancreas", PAL["teal_dark"], PAL["teal_mid"], PAL["teal_sage"], "D", "Pancreas"),
    ]:
        sub_a = df_gko[df_gko["Atlas"] == atlas]
        s_top = sub_a[sub_a["Condition"] == "Top-Attributed KO"].sort_values("K_Genes")
        s_rnd = sub_a[sub_a["Condition"] == "Random Active-Gene KO"].sort_values("K_Genes")
        x = s_top["K_Genes"].to_numpy(dtype=float)
        vt = s_top["Mean_Vacuity"].to_numpy(dtype=float)
        vr = s_rnd["Mean_Vacuity"].to_numpy(dtype=float)
        vsd = s_rnd["Mean_Vacuity_Std"].to_numpy(dtype=float)

        ax_b.plot(
            x,
            vt,
            color=col_top,
            linestyle="-",
            marker=mk,
            markersize=6.2 if mk == "o" else 5.8,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.7,
            linewidth=2.7,
            label=f"{lbl_prefix}: Top-Attributed KO",
            zorder=4,
        )
        ax_b.plot(
            x,
            vr,
            color=col_rnd,
            linestyle="--",
            marker=mk,
            markersize=4.6,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.5,
            linewidth=1.7,
            alpha=0.82,
            label=f"{lbl_prefix}: Random KO (± SD)",
            zorder=3,
        )
        ax_b.fill_between(x, np.maximum(0.0, vr - vsd), vr + vsd, color=col_fill, alpha=0.22, zorder=2)

        fold_150 = vt[-1] / max(1e-8, vt[0])
        if atlas == "PBMC":
            ax_b.annotate(
                "PBMC: 2.38$\\times$ surge\n($p < 10^{-300}$)",
                xy=(150, vt[-1]),
                xytext=(108, 0.335),
                fontsize=9.3,
                fontweight="bold",
                color=PAL["rose_dark"],
                ha="center",
                bbox=dict(boxstyle="round,pad=0.34", facecolor=PAL["rose_bg"], edgecolor=PAL["rose_edge"], lw=1.0),
                arrowprops=dict(arrowstyle="->", color=PAL["rose_dark"], lw=1.4, connectionstyle="arc3,rad=-0.12"),
                zorder=6,
            )
        else:
            ax_b.annotate(
                f"Pancreas: {fold_150:.2f}$\\times$ surge\n($p = 1.5 \\times 10^{{-197}}$)",
                xy=(150, vt[-1]),
                xytext=(108, 0.205),
                fontsize=9.3,
                fontweight="bold",
                color=PAL["teal_dark"],
                ha="center",
                bbox=dict(boxstyle="round,pad=0.34", facecolor=PAL["teal_bg"], edgecolor=PAL["teal_edge"], lw=1.0),
                arrowprops=dict(arrowstyle="->", color=PAL["teal_dark"], lw=1.4, connectionstyle="arc3,rad=0.14"),
                zorder=6,
            )

    add_nc_panel_header(ax_b, "b", "Epistemic vacuity surge under gene KO")
    ax_b.set_xlabel("Knocked-Out Genes per Cell ($K$)", fontsize=11.0, fontweight="bold")
    ax_b.set_ylabel("Mean Dirichlet Vacuity ($u_i = C / S_i$)", fontsize=11.0, fontweight="bold")
    ax_b.set_xticks([0, 20, 40, 80, 150])
    ax_b.set_xlim(-6, 160)
    ax_b.set_ylim(0.03, 0.382)
    ax_b.legend(loc="upper left", fontsize=8.9, ncol=1, framealpha=0.95)
    ax_b.grid(True)
    sns.despine(ax=ax_b)

    # Panel c: Curated GO-BP pathway-level knock-out (PBMC)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c2 = ax_c.twinx()

    lns_c = []
    for cond, col, ls, mk, lw, ms, lbl in [
        ("Top-Pathway KO", PAL["rose_dark"], "-", "o", 2.7, 6.0, "Top GO-BP Pathway KO (F1)"),
        ("Matched-Random KO", PAL["navy_cerulean"], "-", "s", 2.1, 5.5, "Matched-Size Random KO (F1 ± SD)"),
        ("Low-Relevance Pathway KO", PAL["slate_mid"], "--", "^", 1.9, 5.5, "Low-Relevance Pathway KO (F1)"),
    ]:
        s = df_pko[df_pko["Condition"] == cond].sort_values("K_Pathways")
        x = s["K_Pathways"].to_numpy(dtype=float)
        y = s["Macro_F1"].to_numpy(dtype=float)
        ln, = ax_c.plot(
            x,
            y,
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=ms,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.65,
            linewidth=lw,
            label=lbl,
            zorder=4,
        )
        lns_c.append(ln)
        if cond == "Matched-Random KO":
            sd = s["Macro_F1_Std"].to_numpy(dtype=float)
            ax_c.fill_between(
                x,
                np.maximum(0.0, y - sd),
                np.minimum(1.0, y + sd),
                color=col,
                alpha=0.16,
                zorder=2,
            )

    s_ptop = df_pko[df_pko["Condition"] == "Top-Pathway KO"].sort_values("K_Pathways")
    s_prnd = df_pko[df_pko["Condition"] == "Matched-Random KO"].sort_values("K_Pathways")
    ln_v1, = ax_c2.plot(
        s_ptop["K_Pathways"],
        s_ptop["Mean_Vacuity"],
        color=PAL["iris_dark"],
        linestyle="-.",
        marker="D",
        markersize=5.4,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.6,
        linewidth=2.3,
        label="Top-Pathway Vacuity ($u_i$, right)",
        zorder=5,
    )
    ln_v2, = ax_c2.plot(
        s_prnd["K_Pathways"],
        s_prnd["Mean_Vacuity"],
        color=PAL["iris_soft"],
        linestyle=":",
        marker="o",
        markersize=3.8,
        linewidth=1.8,
        label="Matched-Random Vacuity (right)",
        zorder=3,
    )
    lns_c.extend([ln_v1, ln_v2])

    # Annotate k_pw=20 gene count and drop cleanly in lower-right open space
    f1_20 = float(s_ptop[s_ptop["K_Pathways"] == 20]["Macro_F1"].iloc[0])
    f1_0 = float(s_ptop[s_ptop["K_Pathways"] == 0]["Macro_F1"].iloc[0])
    g_20 = float(s_ptop[s_ptop["K_Pathways"] == 20]["Mean_Perturbed_Genes"].iloc[0])
    df1_str = _fmt_minus(f"{(f1_20 - f1_0) * 100:+.1f}")
    ax_c.annotate(
        f"$k_{{\\mathrm{{pw}}}}=20$ (~{g_20:.0f} genes)\nF1: {df1_str} pp | Vac: 2.00$\\times$",
        xy=(20, f1_20),
        xytext=(15.4, 0.590),
        fontsize=9.0,
        fontweight="bold",
        color=PAL["rose_dark"],
        ha="center",
        bbox=dict(boxstyle="round,pad=0.34", facecolor=PAL["rose_bg"], edgecolor=PAL["rose_edge"], lw=1.0),
        arrowprops=dict(arrowstyle="->", color=PAL["rose_dark"], lw=1.4, connectionstyle="arc3,rad=0.14"),
        zorder=6,
    )

    add_nc_panel_header(ax_c, "c", "Curated GO-BP pathway knock-out (PBMC)")
    ax_c.set_xlabel("Knocked-Out GO-BP Pathways ($k_{\\mathrm{pw}}$)", fontsize=11.0, fontweight="bold")
    ax_c.set_ylabel("Cross-Platform Macro-F1", fontsize=11.0, fontweight="bold")
    ax_c2.set_ylabel("Mean Dirichlet Vacuity ($u_i$)", fontsize=11.0, fontweight="bold", color=PAL["iris_dark"])
    ax_c2.tick_params(axis="y", colors=PAL["iris_dark"], labelsize=10.0)
    ax_c2.spines["right"].set_color(PAL["iris_dark"])
    ax_c2.spines["right"].set_linewidth(0.95)
    ax_c2.spines["top"].set_visible(False)
    ax_c.spines["top"].set_visible(False)

    ax_c.set_xticks([0, 2, 5, 10, 20])
    ax_c.set_xlim(-0.8, 21.6)
    ax_c.set_ylim(0.49, 1.005)
    ax_c2.set_ylim(0.03, 0.305)
    ax_c.grid(True)
    ax_c.legend(lns_c, [l.get_label() for l in lns_c], loc="lower left", fontsize=8.4, framealpha=0.95)

    # Panel d: Counterfactual lineage swap trajectory (CD14+ Mono -> B cell)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d2 = ax_d.twinx()

    alp = df_sw["Alpha"].to_numpy(dtype=float)
    p_cd14 = df_sw["P_CD14"].to_numpy(dtype=float)
    p_b = df_sw["P_B"].to_numpy(dtype=float)
    vac = df_sw["Mean_Vacuity"].to_numpy(dtype=float)
    n_sw_g = int(df_sw["N_Swap_Genes"].iloc[0])
    pct_sw = float(df_sw["Pct_HVGs_Swapped"].iloc[0])

    l1, = ax_d.plot(
        alp,
        p_cd14,
        color=PAL["rose_dark"],
        marker="o",
        markersize=6.0,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.7,
        linewidth=2.8,
        label="$P(\\mathrm{CD14^+\\ Monocyte})$",
        zorder=5,
    )
    l2, = ax_d.plot(
        alp,
        p_b,
        color=PAL["teal_dark"],
        marker="s",
        markersize=5.8,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.7,
        linewidth=2.8,
        label="$P(\\mathrm{B\\ Cell})$",
        zorder=5,
    )

    l3, = ax_d2.plot(
        alp,
        vac,
        color=PAL["iris_dark"],
        linestyle="-.",
        marker="D",
        markersize=5.6,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.65,
        linewidth=2.5,
        label="Dirichlet Vacuity ($u_i$, right)",
        zorder=4,
    )
    ax_d2.fill_between(alp, 0.0, vac, color=PAL["iris_soft"], alpha=0.22, zorder=1)

    pk_idx = int(np.argmax(vac))
    a_pk, v_pk, v_0 = alp[pk_idx], vac[pk_idx], vac[0]
    ax_d2.axvline(a_pk, color=PAL["iris_dark"], linestyle=":", linewidth=1.4, alpha=0.80, zorder=2)
    ax_d2.annotate(
        f"{v_pk / v_0:.1f}$\\times$ Vacuity Ridge\n($u_i = {v_pk:.3f}$ at $\\alpha={a_pk:.1f}$)",
        xy=(a_pk, v_pk),
        xytext=(0.45, 0.438),
        fontsize=9.4,
        fontweight="bold",
        color=PAL["iris_dark"],
        ha="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PAL["iris_bg"], edgecolor=PAL["iris_edge"], lw=1.0),
        arrowprops=dict(arrowstyle="->", color=PAL["iris_dark"], lw=1.4, connectionstyle="arc3,rad=-0.12"),
        zorder=7,
    )

    ax_d.text(
        0.025,
        0.55,
        f"Perturbed: {n_sw_g} genes ({pct_sw:.1f}% HVGs)\nUntouched: {100.0 - pct_sw:.1f}% transcriptome",
        transform=ax_d.transAxes,
        fontsize=9.1,
        fontweight="bold",
        color=PAL["navy_dark"],
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.34", facecolor=PAL["navy_bg"], edgecolor=PAL["navy_edge"], lw=1.0),
        zorder=6,
    )

    add_nc_panel_header(ax_d, "d", "Counterfactual swap (CD14+ Mono $\\to$ B cell)")
    ax_d.set_xlabel("Interpolation Weight ($\\alpha$) on Attributed Lineage Genes", fontsize=11.0, fontweight="bold")
    ax_d.set_ylabel("Predicted Class Probability", fontsize=11.0, fontweight="bold")
    ax_d2.set_ylabel("Dirichlet Epistemic Vacuity ($u_i$)", fontsize=11.0, fontweight="bold", color=PAL["iris_dark"])
    ax_d2.tick_params(axis="y", colors=PAL["iris_dark"], labelsize=10.0)
    ax_d2.spines["right"].set_color(PAL["iris_dark"])
    ax_d2.spines["right"].set_linewidth(0.95)
    ax_d2.spines["top"].set_visible(False)
    ax_d.spines["top"].set_visible(False)

    ax_d.set_xlim(-0.04, 1.04)
    ax_d.set_ylim(-0.03, 1.06)
    ax_d2.set_ylim(0.0, 0.505)
    ax_d.grid(True)
    ax_d.legend(
        [l1, l2, l3],
        [l1.get_label(), l2.get_label(), l3.get_label()],
        loc="center left",
        bbox_to_anchor=(0.015, 0.30),
        fontsize=9.0,
        framealpha=0.95,
    )

    _save_and_sync(fig, "fig10_causal_faithfulness.pdf")
    plt.close(fig)


def plot_fig11_cross_platform_mechanism() -> None:
    """Render `fig11_cross_platform_mechanism.pdf` (4-panel cross-platform mechanism figure)."""
    df_snr = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_cross_platform_snr.csv"))
    df_ec = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_error_correction_pbmc.csv"))
    df_ss2 = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_smartseq2_rescues.csv"))
    df_sum = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_summary_stats.csv"))

    def _parse_mwu_two_tex(exp_key: str) -> str:
        detail = str(
            df_sum[(df_sum["Experiment"] == exp_key) & (df_sum["Metric"] == "SNR_Median_Fold_MPCA_vs_HVG")]["Detail"].iloc[0]
        )
        mwu_two = float(detail.split("MWU_two=")[1].split(",")[0].strip())
        mant, exp = f"{mwu_two:.1e}".split("e")
        return f"$p = {mant} \\times 10^{{{int(exp)}}}$"

    p_pbmc_two_tex = _parse_mwu_two_tex("4A_PBMC")
    p_panc_two_tex = _parse_mwu_two_tex("4A_Pancreas")

    fig = plt.figure(figsize=(15.8, 9.8), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.34)

    # Panel a: Biological vs. batch variance across 7 PBMC protocols
    ax_a = fig.add_subplot(gs[0, 0])
    sub_p = df_snr[df_snr["Atlas"] == "PBMC"]
    g_df = sub_p[sub_p["Feature_Space"] == "Individual HVGs"]
    p_df = sub_p[sub_p["Feature_Space"] == "MPCA Pathway Tokens"]

    ax_a.scatter(
        g_df["R2_Batch"],
        g_df["R2_Bio"],
        s=17,
        color=PAL["amber_mid"],
        alpha=0.38,
        edgecolors="none",
        label=f"Individual HVGs ($n={len(g_df):,}$)".replace(",", "{,}"),
        zorder=2,
    )
    ax_a.scatter(
        p_df["R2_Batch"],
        p_df["R2_Bio"],
        s=30,
        color=PAL["iris_dark"],
        alpha=0.80,
        edgecolors="#FFFFFF",
        linewidths=0.40,
        label=f"MPCA GO-BP Pathways ($n={len(p_df)}$)",
        zorder=4,
    )

    # Median star markers
    mg_x, mg_y = float(g_df["R2_Batch"].median()), float(g_df["R2_Bio"].median())
    mp_x, mp_y = float(p_df["R2_Batch"].median()), float(p_df["R2_Bio"].median())
    ax_a.scatter(
        [mg_x],
        [mg_y],
        marker="*",
        s=210,
        color=PAL["amber_dark"],
        edgecolors="#FFFFFF",
        linewidths=0.9,
        zorder=6,
        label="HVG Median",
    )
    ax_a.scatter(
        [mp_x],
        [mp_y],
        marker="*",
        s=230,
        color=PAL["rose_dark"],
        edgecolors="#FFFFFF",
        linewidths=0.9,
        zorder=7,
        label="MPCA Pathway Median",
    )

    # Equal-SNR (1x) and 10x SNR reference lines
    x_ref = np.linspace(0.0, 0.36, 100)
    ax_a.plot(x_ref, x_ref, "--", color=PAL["spine"], lw=1.25, zorder=1)
    ax_a.text(0.288, 0.252, "SNR = 1", fontsize=9.0, fontweight="bold", color=PAL["ink_mid"], rotation=20, ha="center")
    x_ref10 = np.linspace(0.0, 0.082, 100)
    ax_a.plot(x_ref10, 10.0 * x_ref10, ":", color=PAL["ink_mid"], lw=1.25, zorder=1)
    ax_a.text(0.072, 0.775, "SNR = 10", fontsize=9.0, fontweight="bold", color=PAL["ink_mid"], ha="left")

    ax_a.annotate(
        f"61$\\times$ Median $R^2_{{\\mathrm{{bio}}}}$\n({mp_y:.3f} vs. {mg_y:.3f})",
        xy=(mp_x, mp_y),
        xytext=(0.222, 0.495),
        fontsize=9.4,
        fontweight="bold",
        color=PAL["iris_dark"],
        ha="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PAL["iris_bg"], edgecolor=PAL["iris_edge"], lw=1.0),
        arrowprops=dict(arrowstyle="->", color=PAL["iris_dark"], lw=1.4, connectionstyle="arc3,rad=0.14"),
        zorder=8,
    )

    add_nc_panel_header(ax_a, "a", "Biological vs. batch variance (7 PBMC protocols)")
    ax_a.set_xlabel("Technical Batch Variance Explained ($R^2_{\\mathrm{batch}}$)", fontsize=10.8, fontweight="bold")
    ax_a.set_ylabel("Biological Cell-Type Variance Explained ($R^2_{\\mathrm{bio}}$)", fontsize=10.8, fontweight="bold")
    ax_a.set_xlim(-0.008, 0.365)
    ax_a.set_ylim(-0.02, 0.86)
    ax_a.legend(loc="upper right", fontsize=8.9, framealpha=0.95)
    ax_a.grid(True)
    sns.despine(ax=ax_a)

    # Panel b: Biological-to-batch SNR across PBMC & Pancreas (log-scale)
    ax_b = fig.add_subplot(gs[0, 1])
    df_snr_plot = df_snr.copy()
    df_snr_plot["SNR_Plot"] = np.clip(df_snr_plot["Bio_Batch_SNR"], 1e-2, 1e4)
    atlas_map = {"PBMC": "PBMC\n(7 Protocols, N = 23,154)", "Pancreas": "Pancreas\n(4 Protocols, N = 14,229)"}
    df_snr_plot["Atlas_Disp"] = df_snr_plot["Atlas"].map(atlas_map)
    atlas_order = [atlas_map["PBMC"], atlas_map["Pancreas"]]
    space_order = ["Individual HVGs", "MPCA Pathway Tokens"]
    space_pal = {"Individual HVGs": PAL["amber_mid"], "MPCA Pathway Tokens": PAL["iris_dark"]}

    sns.violinplot(
        data=df_snr_plot,
        x="Atlas_Disp",
        y="SNR_Plot",
        hue="Feature_Space",
        order=atlas_order,
        hue_order=space_order,
        palette=space_pal,
        saturation=1.0,
        inner=None,
        cut=0,
        linewidth=0.0,
        alpha=0.34,
        width=0.72,
        ax=ax_b,
        log_scale=True,
    )
    sns.boxplot(
        data=df_snr_plot,
        x="Atlas_Disp",
        y="SNR_Plot",
        hue="Feature_Space",
        order=atlas_order,
        hue_order=space_order,
        palette=space_pal,
        saturation=1.0,
        width=0.28,
        dodge=True,
        showfliers=False,
        boxprops={"edgecolor": "none", "linewidth": 0.0, "alpha": 0.92},
        whiskerprops={"color": PAL["ink_mid"], "linewidth": 1.2},
        capprops={"color": PAL["ink_mid"], "linewidth": 1.2},
        medianprops={"color": "#FFFFFF", "linewidth": 2.2},
        legend=False,
        ax=ax_b,
    )
    ax_b.axhline(1.0, color=PAL["spine"], linestyle="--", linewidth=1.2, zorder=1)
    ax_b.text(1.48, 0.78, "Bio = Batch", fontsize=8.8, fontweight="bold", color=PAL["ink_mid"], ha="right", va="top")

    # Clean statistical brackets with exact two-sided Mann-Whitney U p-values & fold-gains
    # Placed above Pancreas HVG upper whisker (1,269.0)
    bracket_specs = [
        (0, 2500.0, f"8.67$\\times$ median (10.58 vs. 1.22)\n{p_pbmc_two_tex}"),
        (1, 2500.0, f"1.50$\\times$ median (9.63 vs. 6.40)\n{p_panc_two_tex}"),
    ]
    for xi, y_top, txt in bracket_specs:
        x1, x2 = xi - 0.18, xi + 0.18
        ax_b.plot([x1, x1, x2, x2], [y_top * 0.68, y_top, y_top, y_top * 0.68], color=PAL["ink_dark"], lw=1.2)
        ax_b.text(
            xi,
            y_top * 1.25,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=PAL["ink_dark"],
        )

    add_nc_panel_header(ax_b, "b", "Biological-to-batch SNR (PBMC & Pancreas)")
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Bio / Batch SNR ($R^2_{\\mathrm{bio}} / (R^2_{\\mathrm{batch}} + 10^{-4})$)", fontsize=10.6, fontweight="bold")
    ax_b.tick_params(axis="x", labelsize=10.0)
    for lbl in ax_b.get_xticklabels():
        lbl.set_fontweight("bold")
    ax_b.set_ylim(0.01, 42000.0)
    ax_b.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    ax_b.legend(handles_b[:2], labels_b[:2], loc="lower right", fontsize=9.1, framealpha=0.95)
    remove_bar_borders(ax_b)
    ax_b.set_axisbelow(True)
    ax_b.grid(axis="y")
    sns.despine(ax=ax_b)

    # Panel c: Droplet-to-plate (10x v2 -> Smart-Seq2) rescue dissection
    ax_c = fig.add_subplot(gs[1, 0])
    short_ct = {
        "Cytotoxic T cell": "Cytotoxic T",
        "Natural killer cell": "NK cell",
        "CD14+ monocyte": "CD14+ Mono",
        "Dendritic cell": "DC",
        "CD4+ T cell": "CD4+ T",
    }
    labels_c, counts_c, colors_c, annots_c = [], [], [], []
    for _, r in df_ss2.iterrows():
        t_s = short_ct.get(r["True_Type"], r["True_Type"])
        w_s = short_ct.get(r["Wrong_Pred"], r["Wrong_Pred"])
        is_resc = str(r["Direction"]).startswith("Rescued")
        tag = "SVM err." if is_resc else "Ours err."
        labels_c.append(f"{t_s} $\\to$ {w_s} ({tag})")
        cnt = int(r["Cell_Count"])
        counts_c.append(cnt)
        colors_c.append(PAL["teal_mid"] if is_resc else PAL["slate_mid"])
        fold_j = float(r["HVG_Det_Fold_Jump"])
        unit = "cell" if cnt == 1 else "cells"
        if is_resc:
            annots_c.append(f"{cnt} {unit} ({fold_j:.1f}$\\times$ HVG det.)")
        else:
            annots_c.append(f"{cnt} {unit}")

    y_pos = np.arange(len(labels_c))
    ax_c.barh(y_pos, counts_c, height=0.66, color=colors_c, edgecolor="none", linewidth=0.0, alpha=0.92, zorder=3)
    ax_c.invert_yaxis()
    for yi, cnt, txt, col in zip(y_pos, counts_c, annots_c, colors_c):
        ax_c.text(
            cnt + 0.65,
            yi,
            txt,
            va="center",
            ha="left",
            fontsize=9.2,
            fontweight="bold" if col == PAL["teal_mid"] else "normal",
            color=PAL["teal_dark"] if col == PAL["teal_mid"] else PAL["ink_mid"],
        )

    # Subtle separator between Rescued (4 rows) and Opposite (2 rows)
    ax_c.axhline(3.5, color=PAL["spine"], linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)

    # Summary box in lower-right open space
    ax_c.text(
        0.985,
        0.05,
        "10x v2 $\\to$ Smart-seq2 ($N=253$)\n"
        "44 Rescued vs. 6 Opposite (7.3:1)\n"
        "Acc: 89.3% vs. 74.3% (+15.02 pp)\n"
        "HVG Detection Jump: 2.8$\\times$–4.5$\\times$",
        transform=ax_c.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=PAL["teal_dark"],
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.38", facecolor=PAL["teal_bg"], edgecolor=PAL["teal_edge"], lw=1.0),
        zorder=6,
    )

    leg_patches_c = [
        Rectangle((0, 0), 1, 1, facecolor=PAL["teal_mid"], edgecolor="none", label="Rescued by scPathProto (44 cells)"),
        Rectangle((0, 0), 1, 1, facecolor=PAL["slate_mid"], edgecolor="none", label="Opposite: SVM Correct (6 cells)"),
    ]
    ax_c.legend(handles=leg_patches_c, loc="center right", bbox_to_anchor=(0.99, 0.50), fontsize=8.9, framealpha=0.95)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels_c, fontsize=9.4, fontweight="bold")
    ax_c.set_xlim(0, 45.5)
    add_nc_panel_header(ax_c, "c", "10x v2 $\\to$ Smart-seq2 error-correction dissection")
    ax_c.set_xlabel("Discrepant Cells (Single-Cell Pairwise Audit)", fontsize=10.8, fontweight="bold")
    ax_c.set_axisbelow(True)
    ax_c.grid(axis="x")
    remove_bar_borders(ax_c)
    sns.despine(ax=ax_c)

    # Panel d: Cross-protocol rescue advantage across all 7 PBMC chemistries
    ax_d = fig.add_subplot(gs[1, 1])
    df_ec_s = df_ec.sort_values("Acc_Diff_pp", ascending=False).reset_index(drop=True)
    y_d = np.arange(len(df_ec_s))
    diffs = df_ec_s["Acc_Diff_pp"].to_numpy(dtype=float)
    bar_cols = [PAL["rose_dark"] if v >= 0 else PAL["slate_mid"] for v in diffs]

    ax_d.barh(y_d, diffs, height=0.64, color=bar_cols, edgecolor="none", linewidth=0.0, alpha=0.92, zorder=3)
    ax_d.axvline(0.0, color=PAL["ink_dark"], linewidth=1.15, zorder=4)
    ax_d.invert_yaxis()

    for yi, r in df_ec_s.iterrows():
        v = float(r["Acc_Diff_pp"])
        net = int(r["Net_Gain"])
        sp = float(r["Zero_Sparsity_Pct"])
        ratio = float(r["HVG_Detection_Ratio_vs_Ref"])
        u_net = "cell" if abs(net) == 1 else "cells"
        v_s = _fmt_minus(f"{v:+.2f}")
        net_s = _fmt_minus(f"{net:+d}")
        if r["Protocol"] == "Smart-Seq2":
            extra = f"{v_s} pp ({net_s} {u_net} | {ratio:.1f}$\\times$ det.)"
        elif r["Protocol"] in ("inDrop", "Seq-Well"):
            extra = f"{v_s} pp ({net_s} {u_net} | {sp:.1f}% zeros)"
        else:
            extra = f"{v_s} pp ({net_s} {u_net})"

        if v >= 0:
            ax_d.text(
                v + 0.55,
                yi,
                extra,
                va="center",
                ha="left",
                fontsize=9.0,
                fontweight="bold" if v >= 1.0 else "normal",
                color=PAL["rose_dark"] if v >= 1.0 else PAL["ink_dark"],
            )
        else:
            ax_d.text(
                0.55,
                yi,
                extra,
                va="center",
                ha="left",
                fontsize=9.0,
                color=PAL["ink_mid"],
            )

    prot_disp = {"Smart-Seq2": "Smart-seq2", "Drop-Seq": "Drop-seq", "10Xv3": "10x v3", "10Xv2 (b2)": "10x v2 (b2)"}
    prot_ylabels = [
        f"{prot_disp.get(r['Protocol'], r['Protocol'])}\n($N={int(r['N_Cells']):,}$)".replace(",", "{,}")
        for _, r in df_ec_s.iterrows()
    ]
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(prot_ylabels, fontsize=9.2, fontweight="bold")
    ax_d.set_xlim(-8.8, 44.0)
    ax_d.set_xticks([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40])
    add_nc_panel_header(ax_d, "d", "Cross-protocol advantage across 7 PBMC chemistries")
    ax_d.set_xlabel("Accuracy Gain: scPathProto $-$ SVM (percentage points, pp)", fontsize=10.6, fontweight="bold")
    ax_d.set_axisbelow(True)
    ax_d.grid(axis="x")

    leg_patches_d = [
        Rectangle((0, 0), 1, 1, facecolor=PAL["rose_dark"], edgecolor="none", label="scPathProto Advantage (5/7 Protocols)"),
        Rectangle((0, 0), 1, 1, facecolor=PAL["slate_mid"], edgecolor="none", label="SVM Advantage (2/7 Protocols)"),
    ]
    ax_d.legend(handles=leg_patches_d, loc="lower right", fontsize=8.9, framealpha=0.95)
    remove_bar_borders(ax_d)
    sns.despine(ax=ax_d)

    _save_and_sync(fig, "fig11_cross_platform_mechanism.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_fig10_causal_faithfulness()
    plot_fig11_cross_platform_mechanism()

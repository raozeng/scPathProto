#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render clinical COVID-19 macrophage reprogramming and PBMC de novo regulon
figures (`fig12_covid_clinical_discovery.pdf`, `fig13_denovo_regulon_decoding.pdf`)
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
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

CMAP_NC_DIV_RB = LinearSegmentedColormap.from_list(
    "nc_div_rb", ["#2E4D72", "#9EBAD4", "#F8FAFC", "#DDAAB7", "#914153"], N=256
)
CMAP_NC_ROSE = LinearSegmentedColormap.from_list(
    "nc_rose", ["#FAF3F5", "#F2E2E6", "#DDAAB7", "#BC6E80", "#914153"], N=256
)
CMAP_NC_AMBER = LinearSegmentedColormap.from_list(
    "nc_amber", ["#FAF5EA", "#F4EBD7", "#DFBA88", "#C28B4E", "#9D6833"], N=256
)
CMAP_NC_TEAL_NAVY = LinearSegmentedColormap.from_list(
    "nc_teal_navy", ["#F5F9EE", "#DBE9D8", "#8BBCC3", "#5186A8", "#2E4D72"], N=256
)

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


def style_heatmap_annot_contrast(ax, lum_threshold: float = 0.44) -> None:
    """Set heatmap cell text color based on background luminance."""
    if not ax.collections:
        return
    mesh = ax.collections[0]
    mesh.update_scalarmappable()
    fcs = mesh.get_facecolors()
    if fcs is None or len(fcs) == 0:
        return
    for idx, txt in enumerate(ax.texts):
        if idx >= len(fcs):
            break
        r, g, b = fcs[idx][:3]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        txt.set_color(PAL["ink_dark"] if lum > lum_threshold else "#FFFFFF")


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


def plot_fig12_covid_clinical_discovery() -> None:
    """Render `fig12_covid_clinical_discovery.pdf` (4-panel COVID-19 clinical figure)."""
    df_pw = pd.read_csv(os.path.join(TABLES_DIR, "clinical_covid_donor_pathways.csv"))
    df_unc = pd.read_csv(os.path.join(TABLES_DIR, "clinical_covid_donor_uncertainty.csv"))
    df_cor = pd.read_csv(os.path.join(TABLES_DIR, "clinical_covid_marker_uncertainty_corr.csv"))
    df_sub = pd.read_csv(os.path.join(TABLES_DIR, "clinical_covid_substate_vacuity.csv"))

    fig = plt.figure(figsize=(16.0, 10.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.24, 1.0], height_ratios=[1.06, 1.0], hspace=0.42, wspace=0.33)

    # Panel a: Macrophage GO-BP reprogramming across 13 COVID-19 donors
    ax_a = fig.add_subplot(gs[0, 0])
    donors = ["HC1", "HC2", "HC3", "HC4", "M1", "M2", "M3", "S1", "S2", "S3", "S4", "S5", "S6"]
    mat_df = df_pw.set_index("Pathway_Display")[donors]

    # Clean typographic annotation strings (avoid "-0.0", use U+2212 minus)
    annot_df = mat_df.map(
        lambda v: "0.0" if abs(round(float(v), 1)) == 0.0 else _fmt_minus(f"{float(v):.1f}")
    )

    hm = sns.heatmap(
        mat_df,
        cmap=CMAP_NC_DIV_RB,
        center=0.0,
        vmin=-1.4,
        vmax=6.2,
        annot=annot_df,
        fmt="",
        annot_kws={"size": 8.2, "weight": "bold"},
        linewidths=0.45,
        linecolor="#FFFFFF",
        cbar_kws={"label": "Pathway Z-Score (vs. Ref)", "pad": 0.02, "fraction": 0.042},
        ax=ax_a,
    )
    style_heatmap_annot_contrast(ax_a, lum_threshold=0.42)
    cbar_a = hm.collections[0].colorbar
    cbar_a.set_label("Pathway Z-Score (vs. Ref)", fontsize=10.4, fontweight="bold", color=PAL["ink_dark"])
    cbar_a.ax.tick_params(labelsize=9.5, colors=PAL["ink_mid"])
    cbar_a.outline.set_edgecolor(PAL["spine"])
    cbar_a.outline.set_linewidth(0.75)

    # Severity group separator lines
    ax_a.axvline(4.0, color=PAL["ink_dark"], linewidth=1.6)
    ax_a.axvline(7.0, color=PAL["ink_dark"], linewidth=1.6)
    ax_a.axhline(6.0, color=PAL["ink_mid"], linestyle="--", linewidth=1.15)

    # Severity group headers above donor columns
    trans_a = ax_a.get_xaxis_transform()
    for x0, w_seg, lbl, fc_seg, tc_seg in [
        (0.05, 3.90, "Healthy (HC1–HC4)", PAL["teal_sub"], PAL["teal_dark"]),
        (4.05, 2.90, "Moderate (M1–M3)", PAL["amber_sub"], PAL["amber_dark"]),
        (7.05, 5.90, "Severe COVID-19 (S1–S6, Unseen)", PAL["rose_sub"], PAL["rose_dark"]),
    ]:
        ax_a.add_patch(
            Rectangle(
                (x0, 1.022),
                w_seg,
                0.092,
                transform=trans_a,
                clip_on=False,
                facecolor=fc_seg,
                edgecolor="none",
                linewidth=0.0,
                zorder=4,
            )
        )
        ax_a.text(
            x0 + 0.5 * w_seg,
            1.068,
            lbl,
            transform=trans_a,
            clip_on=False,
            fontsize=8.6,
            fontweight="bold",
            color=tc_seg,
            ha="center",
            va="center",
            zorder=5,
        )

    add_nc_panel_header(ax_a, "a", "Macrophage GO-BP reprogramming across 13 donors", pad_pts=45.0)
    ax_a.set_xlabel("Patient Donor (Macrophages, $N = 49{,}417$)", fontsize=10.8, fontweight="bold")
    ax_a.set_ylabel("")
    ax_a.tick_params(axis="x", rotation=0, labelsize=9.2)
    ax_a.tick_params(axis="y", rotation=0, labelsize=9.4)
    for lbl in ax_a.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax_a.get_yticklabels():
        lbl.set_fontweight("bold")

    # Panel b: Macrophage epistemic novelty & prototype drift across severity
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b2 = ax_b.twinx()

    sev_order = ["Healthy", "Moderate", "Severe"]
    grp_df = df_unc[df_unc["Level"] == "Severity_Group"].set_index("Group_ID").loc[sev_order]
    don_df = df_unc[df_unc["Level"] == "Donor"]

    x_s = np.arange(3, dtype=float)
    nov_means = grp_df["Mean_Novelty"].to_numpy(dtype=float)
    cos_means = grp_df["Mean_MaxCos"].to_numpy(dtype=float)

    # Cohort means on left (Novelty, Berry Rose) & right (MaxCos, Forest Pine-Teal)
    ax_b.plot(
        x_s - 0.06,
        nov_means,
        color=PAL["rose_dark"],
        marker="o",
        markersize=8.4,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.8,
        linewidth=2.8,
        label="Cohort Mean Novelty (left)",
        zorder=5,
    )
    ax_b2.plot(
        x_s + 0.06,
        cos_means,
        color=PAL["teal_dark"],
        marker="s",
        markersize=8.0,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.8,
        linewidth=2.8,
        linestyle="--",
        label="Cohort Mean Max Cosine (right)",
        zorder=5,
    )

    # Overlay all 13 individual donors with clean horizontal offsets
    for si, sev in enumerate(sev_order):
        sub_d = don_df[don_df["Severity"] == sev]
        offs = np.linspace(-0.08, 0.08, len(sub_d))
        ax_b.scatter(
            np.full(len(sub_d), si - 0.10) + offs * 0.55,
            sub_d["Mean_Novelty"],
            s=46,
            color=PAL["rose_mid"],
            alpha=0.68,
            edgecolors="#FFFFFF",
            linewidths=0.7,
            label="Individual Donor Mean ($n = 13$)" if si == 0 else None,
            zorder=4,
        )
        ax_b2.scatter(
            np.full(len(sub_d), si + 0.10) + offs * 0.55,
            sub_d["Mean_MaxCos"],
            s=46,
            color=PAL["teal_mid"],
            alpha=0.68,
            edgecolors="#FFFFFF",
            linewidths=0.7,
            zorder=4,
        )

    # Quantitative summary callout boxes
    ax_b.annotate(
        "1.87$\\times$ Novelty Surge\n(0.069 $\\to$ 0.129)\nDonor $\\rho = +0.76$, $p = 0.0028$",
        xy=(2.0 - 0.06, nov_means[2]),
        xytext=(0.94, 0.222),
        fontsize=8.9,
        fontweight="bold",
        color=PAL["rose_dark"],
        ha="center",
        bbox=dict(boxstyle="round,pad=0.33", facecolor=PAL["rose_bg"], edgecolor=PAL["rose_edge"], lw=1.0),
        arrowprops=dict(arrowstyle="->", color=PAL["rose_dark"], lw=1.4, connectionstyle="arc3,rad=-0.16"),
        zorder=7,
    )
    ax_b2.annotate(
        "Prototype Drift: $-0.149$\n(0.732 $\\to$ 0.583)\nDonor $\\rho = -0.87$, $p = 1.2\\times 10^{-4}$",
        xy=(2.0 + 0.06, cos_means[2]),
        xytext=(0.76, 0.502),
        fontsize=8.9,
        fontweight="bold",
        color=PAL["teal_dark"],
        ha="center",
        bbox=dict(boxstyle="round,pad=0.33", facecolor=PAL["teal_bg"], edgecolor=PAL["teal_edge"], lw=1.0),
        arrowprops=dict(arrowstyle="->", color=PAL["teal_dark"], lw=1.4, connectionstyle="arc3,rad=0.14"),
        zorder=7,
    )

    xtick_b = [
        f"Healthy\n(n = {int(grp_df.loc['Healthy', 'N_Macrophages']):,})",
        f"Moderate\n(n = {int(grp_df.loc['Moderate', 'N_Macrophages']):,})",
        f"Severe\n(n = {int(grp_df.loc['Severe', 'N_Macrophages']):,})",
    ]
    ax_b.set_xticks(x_s)
    ax_b.set_xticklabels(xtick_b, fontsize=9.8, fontweight="bold")
    ax_b.set_xlim(-0.42, 2.42)
    ax_b.set_ylim(0.032, 0.272)
    ax_b2.set_ylim(0.44, 0.81)
    ax_b.set_ylabel("Macrophage Open-Set Novelty Score", fontsize=10.8, fontweight="bold", color=PAL["rose_dark"])
    ax_b.tick_params(axis="y", colors=PAL["rose_dark"], labelsize=9.8)
    ax_b2.set_ylabel("Max Prototype Cosine Similarity", fontsize=10.8, fontweight="bold", color=PAL["teal_dark"])
    ax_b2.tick_params(axis="y", colors=PAL["teal_dark"], labelsize=9.8)
    ax_b2.spines["right"].set_color(PAL["teal_dark"])
    ax_b2.spines["right"].set_linewidth(0.95)
    ax_b.spines["left"].set_color(PAL["rose_dark"])
    ax_b.spines["left"].set_linewidth(0.95)
    ax_b.spines["top"].set_visible(False)
    ax_b2.spines["top"].set_visible(False)
    ax_b.grid(axis="y")

    h1, lb1 = ax_b.get_legend_handles_labels()
    h2, lb2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(
        h1 + h2,
        lb1 + lb2,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.012),
        ncol=2,
        fontsize=8.3,
        columnspacing=0.9,
        handletextpad=0.45,
        borderpad=0.30,
        framealpha=0.95,
    )
    add_nc_panel_header(ax_b, "b", "Epistemic novelty & prototype drift across severity", pad_pts=45.0)

    # Panel c: Pathological marker correlations inside Severe macrophages
    ax_c = fig.add_subplot(gs[1, 0])
    df_cor_s = df_cor.sort_values("Spearman_Vacuity_R", ascending=False).reset_index(drop=True)
    y_c = np.arange(len(df_cor_s))
    rv = df_cor_s["Spearman_Vacuity_R"].to_numpy(dtype=float)
    rn = df_cor_s["Spearman_Novelty_R"].to_numpy(dtype=float)
    c_bars = [PAL["rose_dark"] if v >= 0 else PAL["teal_dark"] for v in rv]

    ax_c.barh(y_c, rv, height=0.64, color=c_bars, edgecolor="none", linewidth=0.0, alpha=0.92, zorder=3)
    ax_c.scatter(
        rn,
        y_c,
        marker="D",
        s=44,
        color=PAL["iris_dark"],
        edgecolors="#FFFFFF",
        linewidths=0.7,
        zorder=5,
        label="Spearman $\\rho$ with Novelty",
    )
    ax_c.axvline(0.0, color=PAL["ink_dark"], linewidth=1.15, zorder=4)
    ax_c.axhline(6.5, color=PAL["spine"], linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)
    ax_c.invert_yaxis()

    for yi, (v_r, n_r) in enumerate(zip(rv, rn)):
        v_txt = _fmt_minus(f"{v_r:+.3f}")
        if v_r >= 0:
            x_txt = max(v_r, n_r) + 0.009
            ax_c.text(x_txt, yi, v_txt, va="center", ha="left", fontsize=9.1, fontweight="bold", color=PAL["rose_dark"])
        else:
            x_txt = min(v_r, n_r) - 0.009
            ax_c.text(x_txt, yi, v_txt, va="center", ha="right", fontsize=9.1, fontweight="bold", color=PAL["teal_dark"])

    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(df_cor_s["Gene_Symbol"], fontsize=9.6, fontweight="bold")
    ax_c.set_xlim(-0.315, 0.355)
    ax_c.set_xlabel("Spearman Correlation ($\\rho$) inside Severe Macrophages ($N = 26{,}525$)", fontsize=10.6, fontweight="bold")
    add_nc_panel_header(ax_c, "c", "Pathological vs. homeostatic marker correlations")
    ax_c.set_axisbelow(True)
    ax_c.grid(axis="x")

    leg_c = [
        Rectangle((0, 0), 1, 1, facecolor=PAL["rose_dark"], edgecolor="none", label="Hyper-Inflammatory (Vacuity $\\rho$)"),
        Rectangle((0, 0), 1, 1, facecolor=PAL["teal_dark"], edgecolor="none", label="Homeostatic Resident (Vacuity $\\rho$)"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=PAL["iris_dark"],
            markeredgecolor="#FFFFFF",
            markersize=7.0,
            label="Composite Novelty ($\\rho$)",
        ),
    ]
    ax_c.legend(handles=leg_c, loc="lower right", fontsize=8.3, framealpha=0.95)
    ax_c.text(
        0.028,
        0.955,
        "Vacuity (all 11):\n$p < 10^{-30}$",
        transform=ax_c.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=PAL["navy_dark"],
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.34", facecolor=PAL["navy_bg"], edgecolor=PAL["navy_edge"], lw=1.0),
    )
    remove_bar_borders(ax_c)
    sns.despine(ax=ax_c)

    # Panel d: FCN1+ Inflammatory vs FABP4+ Resident Macrophage Vacuity
    ax_d = fig.add_subplot(gs[1, 1])
    sub_order = ["FABP4+ Resident", "FCN1+ Inflammatory"]
    sub_pal_violin = {"FABP4+ Resident": PAL["teal_mid"], "FCN1+ Inflammatory": PAL["rose_mid"]}
    sub_pal_box = {"FABP4+ Resident": PAL["teal_dark"], "FCN1+ Inflammatory": PAL["rose_dark"]}

    df_sub_plot = df_sub.copy()
    df_sub_plot["Vacuity_Plot"] = np.clip(df_sub_plot["Vacuity"], 0.0, 0.24)

    sns.violinplot(
        data=df_sub_plot,
        x="Substate",
        y="Vacuity_Plot",
        order=sub_order,
        hue="Substate",
        hue_order=sub_order,
        palette=sub_pal_violin,
        saturation=1.0,
        inner=None,
        cut=0,
        linewidth=0.0,
        alpha=0.35,
        width=0.62,
        legend=False,
        ax=ax_d,
    )
    sns.boxplot(
        data=df_sub_plot,
        x="Substate",
        y="Vacuity_Plot",
        order=sub_order,
        hue="Substate",
        hue_order=sub_order,
        palette=sub_pal_box,
        saturation=1.0,
        width=0.22,
        showfliers=False,
        boxprops={"edgecolor": "none", "linewidth": 0.0, "alpha": 0.92},
        whiskerprops={"color": PAL["ink_mid"], "linewidth": 1.25},
        capprops={"color": PAL["ink_mid"], "linewidth": 1.25},
        medianprops={"color": "#FFFFFF", "linewidth": 2.4},
        legend=False,
        ax=ax_d,
    )

    med_fb = float(df_sub[df_sub["Substate"] == "FABP4+ Resident"]["Cohort_Median_Vacuity"].iloc[0])
    med_f1 = float(df_sub[df_sub["Substate"] == "FCN1+ Inflammatory"]["Cohort_Median_Vacuity"].iloc[0])
    n_fb = int(df_sub[df_sub["Substate"] == "FABP4+ Resident"]["Cohort_N"].iloc[0])
    n_f1 = int(df_sub[df_sub["Substate"] == "FCN1+ Inflammatory"]["Cohort_N"].iloc[0])
    nov_fb = float(df_sub[df_sub["Substate"] == "FABP4+ Resident"]["Cohort_Mean_Novelty"].iloc[0])
    nov_f1 = float(df_sub[df_sub["Substate"] == "FCN1+ Inflammatory"]["Cohort_Mean_Novelty"].iloc[0])
    cos_fb = float(df_sub[df_sub["Substate"] == "FABP4+ Resident"]["Cohort_Mean_MaxCos"].iloc[0])
    cos_f1 = float(df_sub[df_sub["Substate"] == "FCN1+ Inflammatory"]["Cohort_Mean_MaxCos"].iloc[0])

    # Median value labels above upper whiskers
    ax_d.text(
        0.0,
        0.148,
        f"Median: {med_fb:.4f}",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=PAL["teal_dark"],
        bbox=dict(boxstyle="round,pad=0.24", facecolor=PAL["teal_bg"], edgecolor=PAL["teal_edge"], lw=0.85),
    )
    ax_d.text(
        1.0,
        0.165,
        f"Median: {med_f1:.4f}",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=PAL["rose_dark"],
        bbox=dict(boxstyle="round,pad=0.24", facecolor=PAL["rose_bg"], edgecolor=PAL["rose_edge"], lw=0.85),
    )

    # Significance bracket above the 0.24 clipped violin tail
    y_brk = 0.250
    ax_d.plot([0, 0, 1, 1], [y_brk - 0.008, y_brk, y_brk, y_brk - 0.008], color=PAL["ink_dark"], lw=1.25)
    ax_d.text(
        0.5,
        y_brk + 0.007,
        "+34.0% Median Vacuity (MWU $p < 10^{-300}$)\nZero Substate Labels Used in Training",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=PAL["ink_dark"],
    )

    # Secondary metrics summary box
    ax_d.text(
        0.50,
        0.025,
        f"Mean Novelty: {nov_f1:.3f} vs. {nov_fb:.3f} ($p < 10^{{-300}}$)\n"
        f"MaxCos: {cos_f1:.3f} vs. {cos_fb:.3f} ($p = 9.2 \\times 10^{{-11}}$)",
        transform=ax_d.transAxes,
        fontsize=8.6,
        fontweight="bold",
        color=PAL["navy_dark"],
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.33", facecolor=PAL["navy_bg"], edgecolor=PAL["navy_edge"], lw=0.95),
    )

    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(
        [
            f"FABP4+ Homeostatic Resident\n(n = {n_fb:,} cells)",
            f"FCN1+ Inflammatory Storm\n(n = {n_f1:,} cells)",
        ],
        fontsize=9.6,
        fontweight="bold",
    )
    ax_d.set_xlabel("Unsupervised Severe Macrophage Substate", fontsize=10.8, fontweight="bold")
    ax_d.set_ylabel("Dirichlet Epistemic Vacuity ($u_i = C / S_i$)", fontsize=10.8, fontweight="bold")
    ax_d.set_ylim(-0.048, 0.316)
    ax_d.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax_d.set_axisbelow(True)
    ax_d.grid(axis="y")
    add_nc_panel_header(ax_d, "d", "Unsupervised FCN1+ vs. FABP4+ substate vacuity")
    remove_bar_borders(ax_d)
    sns.despine(ax=ax_d)

    _save_and_sync(fig, "fig12_covid_clinical_discovery.pdf")
    plt.close(fig)


def plot_fig13_denovo_regulon_decoding() -> None:
    """Render `fig13_denovo_regulon_decoding.pdf` (4-panel de novo regulon figure)."""
    df_gen = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_denovo_genes.csv"))
    df_dot = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_denovo_dotplot.csv"))
    df_con = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_denovo_conservation.csv"))
    df_sca = pd.read_csv(os.path.join(TABLES_DIR, "mechanistic_denovo_scatter.csv"))

    fig = plt.figure(figsize=(16.2, 10.6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.10, 1.18], height_ratios=[1.05, 1.0], hspace=0.47, wspace=0.31)

    # Panel a: Decoded unannotated driver genes across Regulon 1-8
    ax_a = fig.add_subplot(gs[0, 0])
    reg_order = [f"Regulon_{i}" for i in range(1, 9)]
    short_lin = {
        "Plasmacytoid dendritic cell": "pDC",
        "Megakaryocyte": "Mega.",
        "CD16+ monocyte": "CD16+ Mono",
        "Natural killer cell": "NK",
        "Dendritic cell": "DC",
    }
    lin_col_map = {
        "Plasmacytoid dendritic cell": PAL["navy_dark"],
        "Megakaryocyte": PAL["rose_dark"],
        "CD16+ monocyte": PAL["amber_mid"],
    }
    reg_highlights = {
        "Regulon_1": "IL3RA (0.22), TUBB1 (0.16), CPVL",
        "Regulon_2": "TPM2 (0.19), KLRF1 (0.11), GZMH (0.07)",
        "Regulon_3": "SDPR (0.19), LAMP5 (0.17), LRRC25",
        "Regulon_4": "C1orf186, PTCRA (0.12), CMTM5, CLEC12A",
        "Regulon_5": "CCDC50 (0.20), MS4A7 (0.14), SPON2",
        "Regulon_6": "HIST1H2AC (0.20), PGRMC1, PILRA, LTB",
        "Regulon_7": "RGS18 (0.14), NKG7 (0.11), FGFBP2 (0.10)",
        "Regulon_8": "GNG11 (0.20), ACRBP (0.14), LGALS2, MS4A6A",
    }

    z_vals, bar_cols_a, y_labels_a, text_a = [], [], [], []
    for reg in reg_order:
        sub_r = df_gen[df_gen["Regulon"] == reg].sort_values("Rank_in_Regulon")
        lin1 = str(sub_r["Top_Lineage_1"].iloc[0])
        z1 = float(sub_r["Top_Z_1"].iloc[0])
        lin2 = str(sub_r["Top_Lineage_2"].iloc[0])
        z2 = float(sub_r["Top_Z_2"].iloc[0])
        z_vals.append(z1)
        bar_cols_a.append(lin_col_map.get(lin1, PAL["teal_mid"]))
        y_labels_a.append(reg.replace("_", " "))
        s1 = short_lin.get(lin1, lin1)
        s2 = short_lin.get(lin2, lin2)
        text_a.append(f"{s1} (+{z1:.2f}) / {s2} (+{z2:.2f})\n{reg_highlights[reg]}")

    y_a = np.arange(len(reg_order))
    ax_a.barh(y_a, z_vals, height=0.66, color=bar_cols_a, edgecolor="none", linewidth=0.0, alpha=0.92, zorder=3)
    ax_a.invert_yaxis()
    for yi, zv, txt in zip(y_a, z_vals, text_a):
        ax_a.text(
            zv + 0.16,
            yi,
            txt,
            va="center",
            ha="left",
            fontsize=8.1,
            fontweight="bold",
            color=PAL["ink_dark"],
            linespacing=1.18,
        )

    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels(y_labels_a, fontsize=9.6, fontweight="bold")
    ax_a.set_xlim(0.0, 15.6)
    ax_a.set_xticks([0, 1, 2, 3, 4])
    ax_a.set_xlabel("Peak Lineage Activation Z-Score (with Top Unannotated Genes)", fontsize=10.6, fontweight="bold")
    add_nc_panel_header(ax_a, "a", "Decoded unannotated drivers (Regulon 1–8)", pad_pts=48.0)
    ax_a.set_axisbelow(True)
    ax_a.grid(axis="x")

    leg_a = [
        Rectangle((0, 0), 1, 1, facecolor=PAL["navy_dark"], edgecolor="none", label="pDC-Enriched"),
        Rectangle((0, 0), 1, 1, facecolor=PAL["rose_dark"], edgecolor="none", label="Megakaryocyte-Enriched"),
        Rectangle((0, 0), 1, 1, facecolor=PAL["amber_mid"], edgecolor="none", label="CD16+ Mono-Enriched"),
    ]
    ax_a.legend(
        handles=leg_a,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=3,
        fontsize=8.4,
        columnspacing=0.9,
        handletextpad=0.45,
        borderpad=0.30,
        framealpha=0.95,
    )
    ax_a.text(
        0.992,
        0.985,
        "80 / 80 Genes Strictly Outside\n300 Curated GO-BP Masks",
        transform=ax_a.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color=PAL["amber_dark"],
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=PAL["amber_bg"], edgecolor=PAL["amber_edge"], lw=1.0),
    )
    remove_bar_borders(ax_a)
    sns.despine(ax=ax_a)

    # Panel b: Single-cell DotPlot of 16 recovered de novo unannotated markers
    ax_b = fig.add_subplot(gs[0, 1])
    genes_16 = [
        "IL3RA", "PTCRA", "LAMP5", "CCDC50",
        "NKG7", "FGFBP2", "KLRF1", "GZMH",
        "TUBB1", "SDPR", "GNG11", "RGS18",
        "MS4A7", "MS4A6A", "LGALS2", "CLEC12A",
    ]
    ct_order_b = [
        "Plasmacytoid dendritic cell",
        "Natural killer cell",
        "Cytotoxic T cell",
        "CD4+ T cell",
        "Megakaryocyte",
        "CD16+ monocyte",
        "CD14+ monocyte",
        "Dendritic cell",
        "B cell",
    ]
    ct_disp_b = {
        "Plasmacytoid dendritic cell": "Plasmacytoid DC",
        "Natural killer cell": "NK cell",
        "Cytotoxic T cell": "Cytotoxic T",
        "CD4+ T cell": "CD4+ T",
        "Megakaryocyte": "Megakaryocyte",
        "CD16+ monocyte": "CD16+ Mono",
        "CD14+ monocyte": "CD14+ Mono",
        "Dendritic cell": "Dendritic cell",
        "B cell": "B cell",
    }

    g_to_x = {g: j for j, g in enumerate(genes_16)}
    ct_to_y = {c: i for i, c in enumerate(ct_order_b)}
    sub_dot = df_dot[df_dot["Gene_Symbol"].isin(g_to_x) & df_dot["Cell_Type"].isin(ct_to_y)]

    xs = [g_to_x[g] for g in sub_dot["Gene_Symbol"]]
    ys = [ct_to_y[c] for c in sub_dot["Cell_Type"]]
    sizes = [max(10.0, float(p) * 2.25) for p in sub_dot["Pct_Expressing"]]
    colors = [float(v) for v in sub_dot["Mean_LogExpr"]]

    sc = ax_b.scatter(
        xs,
        ys,
        s=sizes,
        c=colors,
        cmap=CMAP_NC_AMBER,
        edgecolors=PAL["ink_mid"],
        linewidths=0.50,
        zorder=3,
    )
    for xv in [3.5, 7.5, 11.5]:
        ax_b.axvline(xv, color=PAL["spine"], linestyle="--", linewidth=0.95, alpha=0.70, zorder=1)

    # Top lineage-group headers
    for gx, glbl, gcol in [
        (1.5, "pDC", PAL["navy_dark"]),
        (5.5, "NK / Cytotoxic", PAL["teal_dark"]),
        (9.5, "Megakaryocyte", PAL["rose_dark"]),
        (13.5, "Myeloid", PAL["amber_dark"]),
    ]:
        ax_b.text(gx, -0.85, glbl, ha="center", va="center", fontsize=9.1, fontweight="bold", color=gcol)

    ax_b.set_xticks(range(len(genes_16)))
    ax_b.set_xticklabels(genes_16, rotation=38, ha="right", fontsize=8.8, fontweight="bold")
    ax_b.set_yticks(range(len(ct_order_b)))
    ax_b.set_yticklabels([ct_disp_b[c] for c in ct_order_b], fontsize=9.4, fontweight="bold")
    ax_b.set_xlim(-0.6, len(genes_16) - 0.4)
    ax_b.set_ylim(len(ct_order_b) - 0.38, -1.32)
    ax_b.grid(linestyle=":", alpha=0.55, zorder=0)

    cbar = plt.colorbar(sc, ax=ax_b, fraction=0.032, pad=0.02)
    cbar.set_label("Mean Log1p Expression", fontsize=9.8, fontweight="bold", color=PAL["ink_dark"])
    cbar.ax.tick_params(labelsize=9.2, colors=PAL["ink_mid"])
    cbar.outline.set_edgecolor(PAL["spine"])
    cbar.outline.set_linewidth(0.75)

    dot_handles = [
        plt.scatter([], [], s=pct * 2.25, facecolor=PAL["amber_soft"], edgecolor=PAL["ink_mid"], linewidth=0.65, label=f"{pct}%")
        for pct in [25, 50, 75, 100]
    ]
    leg_dot = ax_b.legend(
        handles=dot_handles,
        title="% Expressing:",
        loc="lower center",
        bbox_to_anchor=(0.48, 1.012),
        ncol=4,
        fontsize=8.3,
        title_fontsize=8.5,
        columnspacing=0.95,
        handletextpad=0.35,
        borderpad=0.28,
        framealpha=0.95,
    )
    leg_dot.get_title().set_fontweight("bold")
    leg_dot.get_title().set_color(PAL["ink_dark"])
    leg_dot.get_frame().set_edgecolor("#DCE4EE")
    leg_dot.get_frame().set_linewidth(0.8)
    add_nc_panel_header(ax_b, "b", "Lineage specificity of 16 de novo unannotated markers", pad_pts=48.0)
    sns.despine(ax=ax_b)

    # Panel c: Cross-protocol conservation across 7 unseen PBMC protocols
    ax_c = fig.add_subplot(gs[1, 0])
    df_con_s = df_con.sort_values("Spearman_Rho", ascending=False).reset_index(drop=True)
    y_c = np.arange(len(df_con_s))
    rhos = df_con_s["Spearman_Rho"].to_numpy(dtype=float)
    rs = df_con_s["Pearson_R"].to_numpy(dtype=float)
    mean_rho = float(np.mean(rhos))

    bar_cols_c = [PAL["teal_dark"] if v >= 0.90 else PAL["navy_cerulean"] for v in rhos]
    ax_c.barh(
        y_c,
        rhos,
        height=0.62,
        color=bar_cols_c,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.92,
        zorder=3,
        label="Spearman $\\rho$ (Rank)",
    )
    ax_c.scatter(
        rs,
        y_c,
        marker="D",
        s=54,
        color=PAL["rose_dark"],
        edgecolors="#FFFFFF",
        linewidths=0.8,
        zorder=5,
        label="Pearson $r$ (Linear)",
    )
    ax_c.axvline(
        mean_rho,
        color=PAL["iris_dark"],
        linestyle="--",
        linewidth=1.65,
        zorder=4,
        label=f"Mean $\\rho = {mean_rho:.3f}$",
    )
    ax_c.invert_yaxis()

    for yi, (rho_v, r_v) in enumerate(zip(rhos, rs)):
        x_end = max(rho_v, r_v)
        ax_c.text(
            x_end + 0.022,
            yi,
            f"$\\rho={rho_v:.3f}$ ($r={r_v:.3f}$)",
            va="center",
            ha="left",
            fontsize=9.1,
            fontweight="bold" if rho_v >= 0.90 else "normal",
            color=PAL["ink_dark"],
        )

    prot_disp = {"Smart-Seq2": "Smart-seq2", "Drop-Seq": "Drop-seq", "10Xv3": "10x v3", "10Xv2 (b2)": "10x v2 (b2)"}
    y_lbl_c = [
        f"{prot_disp.get(r['Protocol'], r['Protocol'])} ($N={int(r['N_Cells']):,}$)".replace(",", "{,}")
        for _, r in df_con_s.iterrows()
    ]
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(y_lbl_c, fontsize=9.4, fontweight="bold")
    ax_c.set_xlim(0.0, 1.58)
    ax_c.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_xlabel("Cross-Protocol Conservation (10x v2 Ref; All 7: $p \\leq 1.4 \\times 10^{-13}$)", fontsize=10.4, fontweight="bold")
    add_nc_panel_header(ax_c, "c", "Cross-protocol conservation of de novo regulons", pad_pts=30.0)
    ax_c.set_axisbelow(True)
    ax_c.grid(axis="x")
    ax_c.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=3,
        fontsize=8.3,
        columnspacing=0.85,
        handletextpad=0.40,
        borderpad=0.28,
        framealpha=0.95,
    )
    remove_bar_borders(ax_c)
    sns.despine(ax=ax_c)

    # Panel d: Reference (10x v2) vs Unseen-Protocol Regulon Scatter
    ax_d = fig.add_subplot(gs[1, 1])
    for prot, col, mk, lbl_tag in [
        ("10Xv3", PAL["teal_dark"], "o", "10x v3 (Droplet, 64 pairs)"),
        ("Smart-Seq2", PAL["rose_dark"], "D", "Smart-seq2 (Full-Length Plate, 48 pairs)"),
    ]:
        sub_s = df_sca[df_sca["Protocol"] == prot]
        xr = sub_s["Ref_10Xv2_Z"].to_numpy(dtype=float)
        yq = sub_s["Query_Z"].to_numpy(dtype=float)
        row_c = df_con[df_con["Protocol"] == prot].iloc[0]
        rho_v, r_v = float(row_c["Spearman_Rho"]), float(row_c["Pearson_R"])

        ax_d.scatter(
            xr,
            yq,
            s=46,
            color=col,
            marker=mk,
            alpha=0.82,
            edgecolors="#FFFFFF",
            linewidths=0.65,
            label=f"{lbl_tag}: $\\rho={rho_v:.3f}$, $r={r_v:.3f}$",
            zorder=4,
        )
        m_fit, b_fit = np.polyfit(xr, yq, 1)
        xg = np.linspace(xr.min(), xr.max(), 50)
        ax_d.plot(xg, m_fit * xg + b_fit, color=col, linewidth=2.2, alpha=0.88, zorder=3)

    lim_lo = (
        min(
            float(df_sca[df_sca["Protocol"].isin(["10Xv3", "Smart-Seq2"])]["Ref_10Xv2_Z"].min()),
            float(df_sca[df_sca["Protocol"].isin(["10Xv3", "Smart-Seq2"])]["Query_Z"].min()),
        )
        - 0.35
    )
    lim_hi = (
        max(
            float(df_sca[df_sca["Protocol"].isin(["10Xv3", "Smart-Seq2"])]["Ref_10Xv2_Z"].max()),
            float(df_sca[df_sca["Protocol"].isin(["10Xv3", "Smart-Seq2"])]["Query_Z"].max()),
        )
        + 0.50
    )
    ax_d.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color=PAL["spine"], linewidth=1.35, label="Identity ($y = x$)", zorder=1)

    ax_d.set_xlim(lim_lo, lim_hi)
    ax_d.set_ylim(lim_lo, lim_hi)
    ax_d.set_xlabel("Reference 10x v2 Regulon Activation (Z-Score)", fontsize=10.8, fontweight="bold")
    ax_d.set_ylabel("Unseen Query Protocol Regulon Activation (Z-Score)", fontsize=10.8, fontweight="bold")
    add_nc_panel_header(ax_d, "d", "Reference vs. unseen-protocol regulon alignment", pad_pts=14.0)
    ax_d.legend(loc="upper left", fontsize=8.8, framealpha=0.95)
    ax_d.grid(True)
    sns.despine(ax=ax_d)

    _save_and_sync(fig, "fig13_denovo_regulon_decoding.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_fig12_covid_clinical_discovery()
    plot_fig13_denovo_regulon_decoding()

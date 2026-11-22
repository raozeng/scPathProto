#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render multi-panel benchmark and interpretability figures (`fig1`–`fig9`)
for scPathProto from CSV tables in `results/tables/`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from matplotlib.transforms import ScaledTranslation

# Global matplotlib styling
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "stixsans",
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#8B9BB0",
        "axes.linewidth": 0.85,
        "xtick.color": "#3C4B5C",
        "ytick.color": "#3C4B5C",
        "text.color": "#2B3542",
        "axes.labelcolor": "#2B3542",
        "axes.titlecolor": "#2B3542",
        "grid.color": "#E8EEF4",
        "grid.linestyle": "--",
        "grid.alpha": 0.72,
        "axes.axisbelow": True,
        "patch.edgecolor": "none",
        "patch.force_edgecolor": False,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#D8E1EB",
        "legend.framealpha": 0.94,
    }
)

# Color palette
PAL: Dict[str, str] = {
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
    # Neutrals
    "slate": "#8A99AD",
    "slate_muted": "#9AA6B5",
    "ink_dark": "#2B3542",
    "ink_tick": "#3C4B5C",
    "ink_mid": "#526071",
    "spine": "#8B9BB0",
    "grid": "#E8EEF4",
}

METHODS_ORDER: List[str] = [
    "SingleR",
    "CellTypist",
    "scANVI",
    "TOSICA",
    "SVM",
    "scPathProto (Ours)",
]

METHOD_PALETTE: Dict[str, str] = {
    "SingleR": "#8A99AD",
    "CellTypist": "#5186A8",
    "scANVI": "#7862A8",
    "TOSICA": "#43877E",
    "SVM": "#C28B4E",
    "scPathProto (Ours)": "#914153",
}

ABLATION_PALETTE: Dict[str, str] = {
    "Full scPathProto": "#914153",
    "w/o ZIRG (Denoising Gate)": "#4E769E",
    "w/o MPCA (Pathway Prior)": "#C28B4E",
    "w/o HA-SGMP (Heterophily Graph)": "#43877E",
    "w/o Proto-Align": "#7862A8",
    "w/o Evidential Open-Set Head": "#8A99AD",
}

# Custom colormaps
CMAP_NC_TEAL = LinearSegmentedColormap.from_list(
    "nc_teal", ["#F2FAF8", "#D6ECE8", "#76A89D", "#43877E", "#2D6B64"], N=256
)
CMAP_NC_BLUE = LinearSegmentedColormap.from_list(
    "nc_blue", ["#F5F9EE", "#DBE9D8", "#8BBCC3", "#5186A8", "#2E4D72"], N=256
)
CMAP_NC_ROSE = LinearSegmentedColormap.from_list(
    "nc_rose", ["#FAF3F5", "#F2E2E6", "#DDAAB7", "#BC6E80", "#914153"], N=256
)
CMAP_NC_AMBER = LinearSegmentedColormap.from_list(
    "nc_amber", ["#FAF5EA", "#F4EBD7", "#DFBA88", "#C28B4E", "#9D6833"], N=256
)
CMAP_NC_DIV_RB = LinearSegmentedColormap.from_list(
    "nc_div_rb", ["#2E4D72", "#9EBAD4", "#F8FAFC", "#DDAAB7", "#914153"], N=256
)
CMAP_NC_DIV_TC = LinearSegmentedColormap.from_list(
    "nc_div_tc", ["#2D6B64", "#A9D3CB", "#F8FAFC", "#DFBA88", "#9D6833"], N=256
)

# 13-color cell-type palette
NC_CELLTYPE_PALETTE: List[str] = [
    "#914153", "#2E4D72", "#2D6B64", "#5E468A", "#9D6833",
    "#5186A8", "#BC6E80", "#76A89D", "#7862A8", "#C28B4E",
    "#7C8DA1", "#4E769E", "#A48B73",
]

# 7-protocol palette (disjoint from NC_CELLTYPE_PALETTE)
BATCH_PROTOCOL_PALETTE: List[str] = [
    "#3B6E8C",
    "#B57F42",
    "#3A8277",
    "#8B5E83",
    "#6B8EAD",
    "#A65A69",
    "#697785",
]


# Plotting helpers
def set_nc_panel_title(
    ax,
    letter: str,
    title: str,
    pad: float = 10.0,
    letter_size: float = 16.0,
    title_size: float = 12.8,
    x_letter_frac: float = 0.0,
    letter_gap_pts: float = 16.5,
) -> None:
    """Place a bold lowercase panel label and title above `ax`."""
    fig = ax.get_figure()
    trans_letter = ax.transAxes + ScaledTranslation(0.0, pad / 72.0, fig.dpi_scale_trans)
    trans_title = ax.transAxes + ScaledTranslation(letter_gap_pts / 72.0, pad / 72.0, fig.dpi_scale_trans)
    ax.text(
        x_letter_frac,
        1.0,
        letter,
        transform=trans_letter,
        fontsize=letter_size,
        fontweight="bold",
        color=PAL["ink_dark"],
        ha="left",
        va="baseline",
        clip_on=False,
        zorder=20,
    )
    ax.text(
        x_letter_frac,
        1.0,
        title,
        transform=trans_title,
        fontsize=title_size,
        fontweight="bold",
        color=PAL["ink_dark"],
        ha="left",
        va="baseline",
        clip_on=False,
        zorder=20,
    )


def style_legend(leg, border_color: str = "#D8E1EB", facecolor: str = "#FFFFFF", lw: float = 0.75) -> None:
    """Style legend frame and remove patch borders."""
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor(facecolor)
    frame.set_edgecolor(border_color)
    frame.set_linewidth(lw)
    frame.set_alpha(0.95)
    for lpatch in leg.get_patches():
        lpatch.set_edgecolor("none")
        lpatch.set_linewidth(0.0)


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


def style_heatmap_annot_contrast(ax, lum_threshold: float = 0.48, cbar_labelsize: float = 10.2, cbar_ticksize: float = 9.0) -> None:
    """Set heatmap annotation text color from cell luminance and style the colorbar."""
    if not ax.collections:
        return
    mesh = ax.collections[0]
    mesh.update_scalarmappable()
    fcs = mesh.get_facecolors()
    if fcs is not None and len(fcs) > 0:
        for idx, txt in enumerate(ax.texts):
            if idx >= len(fcs):
                break
            r, g, b = fcs[idx][:3]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            txt.set_color(PAL["ink_dark"] if lum > lum_threshold else "#FFFFFF")
    cbar = getattr(mesh, "colorbar", None)
    if cbar is not None:
        cbar.outline.set_edgecolor("#CBD5E1")
        cbar.outline.set_linewidth(0.65)
        cbar.ax.tick_params(labelsize=cbar_ticksize, colors=PAL["ink_tick"], width=0.7)
        cbar.ax.yaxis.label.set_size(cbar_labelsize)
        cbar.ax.yaxis.label.set_weight("bold")
        cbar.ax.yaxis.label.set_color(PAL["ink_dark"])


def draw_grouped_bars_clipped_err(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    std_col: str,
    x_order: List[str],
    hue_col: str,
    hue_order: List[str],
    palette: Dict[str, str],
    group_width: float = 0.78,
    alpha: float = 0.92,
    upper_bound: float = 1.0,
) -> None:
    """Draw grouped bars with asymmetric error bars clipped to [0.0, upper_bound]."""
    n_hue = len(hue_order)
    bar_w = group_width / n_hue
    x_base = np.arange(len(x_order), dtype=float)

    for h_i, h_val in enumerate(hue_order):
        offsets = x_base - 0.5 * group_width + (h_i + 0.5) * bar_w
        means, lows, highs = [], [], []
        for x_val in x_order:
            sub = df[(df[x_col] == x_val) & (df[hue_col] == h_val)]
            if len(sub) == 0:
                means.append(0.0)
                lows.append(0.0)
                highs.append(0.0)
            else:
                m_v = float(sub[y_col].iloc[0])
                s_v = float(sub[std_col].iloc[0]) if std_col in sub.columns and pd.notna(sub[std_col].iloc[0]) else 0.0
                means.append(m_v)
                lows.append(min(s_v, max(0.0, m_v)))
                highs.append(min(s_v, max(0.0, upper_bound - m_v)))
        means_arr = np.asarray(means)
        err_arr = np.vstack([lows, highs])
        ax.bar(
            offsets,
            means_arr,
            width=bar_w * 0.93,
            color=palette[h_val],
            edgecolor="none",
            linewidth=0.0,
            alpha=alpha,
            label=h_val,
            zorder=3,
        )
        has_err = (err_arr[0] > 1e-6) | (err_arr[1] > 1e-6)
        if np.any(has_err):
            ax.errorbar(
                offsets[has_err],
                means_arr[has_err],
                yerr=err_arr[:, has_err],
                fmt="none",
                ecolor=PAL["ink_mid"],
                elinewidth=0.90,
                capsize=2.2,
                capthick=0.90,
                zorder=4,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(x_order)


def save_pdf_and_png(fig, out_pdf: str) -> None:
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    out_png = os.path.splitext(out_pdf)[0] + ".png"
    fig.savefig(out_png, format="png", dpi=260, bbox_inches="tight", pad_inches=0.05)


# Multi-Panel Composite Figure Rendering Functions (fig1 to fig9)
def plot_fig1_intra_dataset(tables_dir: str, out_pdf: str) -> None:
    """
    Figure 2 in manuscript (`fig1_intra_dataset_performance.pdf`):
    a  Intra-dataset overall accuracy (mean ± clipped SD)
    b  Intra-dataset rare-subtype F1 (mean ± clipped SD)
    c  Multi-metric radar profile
    d  Intra-dataset accuracy vs. 126-task mean runtime (Pareto bubble plot)
    """
    df_sum = pd.read_csv(os.path.join(tables_dir, "perf_intra_dataset_summary.csv"))
    rt_path = os.path.join(tables_dir, "runtime_summary_126tasks.csv")
    df_rt = pd.read_csv(rt_path) if os.path.exists(rt_path) else None

    fig = plt.figure(figsize=(16.2, 9.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], height_ratios=[1.0, 1.0], hspace=0.44, wspace=0.26)

    ds_order = ["Pancreas", "PBMC", "Tabula Muris", "Lung COVID-19"]

    # Panel a: Overall Accuracy with clipped [0, 1] SD whiskers
    ax_a = fig.add_subplot(gs[0, 0])
    draw_grouped_bars_clipped_err(
        ax_a, df_sum, "Dataset", "Accuracy_Mean", "Accuracy_Std",
        ds_order, "Method", METHODS_ORDER, METHOD_PALETTE
    )
    set_nc_panel_title(ax_a, "a", "Intra-dataset overall accuracy", pad=34)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Overall Accuracy", fontsize=11.0, fontweight="bold")
    ax_a.set_ylim(0.20, 1.03)
    ax_a.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.tick_params(axis="both", labelsize=9.8)
    ax_a.grid(axis="y", zorder=0)
    leg_a = ax_a.legend(
        ncol=6, loc="lower center", bbox_to_anchor=(0.50, 1.02),
        fontsize=8.6, columnspacing=0.95, handletextpad=0.40, borderpad=0.35
    )
    style_legend(leg_a)
    remove_bar_borders(ax_a)
    sns.despine(ax=ax_a)

    # Panel b: Rare-Subtype F1 Score with clipped [0, 1] SD whiskers
    ax_b = fig.add_subplot(gs[1, 0])
    draw_grouped_bars_clipped_err(
        ax_b, df_sum, "Dataset", "Rare_F1_Mean", "Rare_F1_Std",
        ds_order, "Method", METHODS_ORDER, METHOD_PALETTE
    )
    set_nc_panel_title(ax_b, "b", "Intra-dataset rare-subtype F1", pad=12)
    ax_b.set_xlabel("Single-Cell Atlas", fontsize=11.0, fontweight="bold", labelpad=6)
    ax_b.set_ylabel("Rare-Subtype F1 Score", fontsize=11.0, fontweight="bold")
    ax_b.set_ylim(0.0, 1.05)
    ax_b.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.tick_params(axis="both", labelsize=9.8)
    ax_b.grid(axis="y", zorder=0)
    remove_bar_borders(ax_b)
    sns.despine(ax=ax_b)

    # Panel c: 6-Axis Radar Chart averaged across all 4 benchmark datasets
    ax_c = fig.add_subplot(gs[0, 1], polar=True)
    ax_c.spines["polar"].set_color("#CBD5E1")
    ax_c.spines["polar"].set_linewidth(0.90)
    ax_c.grid(color=PAL["grid"], linestyle="--", alpha=0.85, linewidth=0.8)
    radar_cols = ["Accuracy_Mean", "Macro_F1_Mean", "Median_F1_Mean", "Weighted_F1_Mean", "Kappa_Mean", "Rare_F1_Mean"]
    radar_labels = ["Accuracy", "Macro-F1", "Median-F1", "Weighted-F1", "Kappa", "Rare-F1"]
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]

    df_radar = df_sum.groupby("Method", as_index=False)[radar_cols].mean()
    for m_name in METHODS_ORDER:
        row_m = df_radar[df_radar["Method"] == m_name]
        if len(row_m) == 0:
            continue
        vals = [float(row_m[c].values[0]) for c in radar_cols]
        vals += vals[:1]
        is_ours = "Ours" in m_name
        lw = 2.6 if is_ours else 1.45
        alpha_fill = 0.18 if is_ours else 0.03
        ms = 5.0 if is_ours else 3.8
        ax_c.plot(
            angles, vals, color=METHOD_PALETTE[m_name], linewidth=lw,
            marker="o", markersize=ms, zorder=6 if is_ours else 4, label=m_name
        )
        ax_c.fill(angles, vals, color=METHOD_PALETTE[m_name], alpha=alpha_fill, zorder=5 if is_ours else 3)

    ax_c.set_xticks(angles[:-1])
    ax_c.set_xticklabels(radar_labels, fontsize=10.0, fontweight="bold", color=PAL["ink_dark"])
    ax_c.tick_params(axis="x", pad=21)
    ax_c.set_ylim(0.40, 1.00)
    ax_c.set_yticks([0.50, 0.65, 0.80, 0.95])
    ax_c.set_yticklabels(["0.50", "0.65", "0.80", "0.95"], fontsize=8.8, color=PAL["ink_mid"])
    set_nc_panel_title(ax_c, "c", "Multi-metric radar profile", pad=34, x_letter_frac=-0.28)

    # Panel d: 4-Dataset Mean Intra-Dataset Accuracy vs. 126-Task Mean Runtime Pareto Bubble Chart
    # Zoomed y-axis [0.84, 1.005] cleanly separates SVM (0.960, 51.7s) from CellTypist (0.952, 51.2s)
    ax_d = fig.add_subplot(gs[1, 1])
    df_eff = df_sum.groupby("Method", as_index=False).agg(
        Acc=("Accuracy_Mean", "mean"),
        Rare_F1=("Rare_F1_Mean", "mean"),
        Intra_Time=("Total_Time_Sec", "mean"),
    )
    if df_rt is not None:
        df_eff = df_eff.merge(df_rt[["Method", "Total_Sec_Mean"]], on="Method", how="left")
    else:
        df_eff["Total_Sec_Mean"] = df_eff["Intra_Time"]

    # Staggered label offsets and leader arrows to avoid overlap
    annot_pos = {
        "scPathProto (Ours)": (24.0, 0.980, "center", "bottom"),
        "SVM": (76.0, 0.974, "left", "bottom"),
        "CellTypist": (25.5, 0.916, "center", "top"),
        "TOSICA": (118.0, 0.922, "center", "top"),
        "scANVI": (260.0, 0.952, "center", "bottom"),
        "SingleR": (245.0, 0.870, "center", "top"),
    }

    # Pareto-optimal corner shading
    ax_d.axvspan(11.0, 38.0, ymin=0.68, ymax=1.0, facecolor=PAL["rose_bg"], edgecolor="none", alpha=0.65, zorder=1)

    for m_name in METHODS_ORDER:
        sub_r = df_eff[df_eff["Method"] == m_name]
        if len(sub_r) == 0:
            continue
        r = sub_r.iloc[0]
        x_t = max(1.0, float(r["Total_Sec_Mean"]))
        y_acc = float(r["Acc"])
        b_size = (float(r["Rare_F1"]) ** 2) * 440.0
        is_ours = "Ours" in m_name
        z_ord = 7 if is_ours else (6 if m_name == "SVM" else 5)
        ax_d.scatter(
            x_t, y_acc, s=b_size, color=METHOD_PALETTE.get(m_name, PAL["slate"]),
            alpha=0.92, edgecolors="#FFFFFF", linewidths=1.8, zorder=z_ord
        )
        tx, ty, ha_s, va_s = annot_pos.get(m_name, (x_t, y_acc + 0.015, "center", "bottom"))
        lbl_text = f"{m_name}\n(Acc={y_acc:.3f}, {x_t:.1f}s)"
        if is_ours:
            ax_d.annotate(
                lbl_text,
                xy=(x_t, y_acc + 0.004),
                xytext=(tx, ty - 0.002),
                fontsize=8.9,
                fontweight="bold",
                color=PAL["rose_dark"],
                ha=ha_s,
                va=va_s,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=PAL["rose_bg"], edgecolor=PAL["rose_edge"], lw=0.95),
                arrowprops=dict(arrowstyle="-", color=PAL["rose_mid"], lw=1.0, shrinkA=2, shrinkB=5),
                zorder=9,
            )
        else:
            ax_d.annotate(
                lbl_text,
                xy=(x_t, y_acc),
                xytext=(tx, ty),
                fontsize=8.7,
                fontweight="normal",
                color=PAL["ink_dark"],
                ha=ha_s,
                va=va_s,
                arrowprops=dict(arrowstyle="-", color="#9AA6B5", lw=0.9, shrinkA=3, shrinkB=7),
                zorder=8,
            )

    # Quantitative bubble-size legend for Rare-F1
    size_handles = [
        plt.scatter([], [], s=(rf ** 2) * 440.0, color=PAL["slate_muted"], alpha=0.60, edgecolors=PAL["ink_mid"], linewidths=0.8, label=f"{rf:.2f}")
        for rf in [0.55, 0.75, 0.90]
    ]
    leg_sz = ax_d.legend(
        handles=size_handles,
        title="Bubble Size: Rare-F1",
        loc="lower left",
        fontsize=8.6,
        title_fontsize=8.9,
        ncol=3,
        labelspacing=0.8,
        borderpad=0.50,
        columnspacing=0.95,
        handletextpad=0.45,
    )
    leg_sz.get_title().set_fontweight("bold")
    style_legend(leg_sz)

    ax_d.set_xscale("log")
    ax_d.set_xlim(11.0, 540.0)
    ax_d.xaxis.set_major_locator(FixedLocator([15, 25, 50, 100, 200, 400]))
    ax_d.xaxis.set_major_formatter(FixedFormatter(["15", "25", "50", "100", "200", "400"]))
    ax_d.xaxis.set_minor_locator(NullLocator())
    ax_d.set_ylim(0.840, 1.005)
    ax_d.set_yticks([0.85, 0.88, 0.91, 0.94, 0.97, 1.00])
    ax_d.tick_params(axis="both", labelsize=9.6)
    set_nc_panel_title(ax_d, "d", "Intra-dataset accuracy vs. runtime", pad=12)
    ax_d.set_xlabel("Mean Runtime across 126 Tasks (s, log scale)", fontsize=10.8, fontweight="bold", labelpad=6)
    ax_d.set_ylabel("4-Dataset Mean Accuracy", fontsize=10.8, fontweight="bold")
    ax_d.grid(True, zorder=0)
    sns.despine(ax=ax_d)

    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


def plot_fig2_cross_batch(tables_dir: str, out_pdf: str) -> None:
    """
    Figure 3 in manuscript (`fig2_cross_batch_transfer.pdf`):
    a  Cross-platform transfer Macro-F1 (mean ± clipped task SD)
    b  Rare-subtype F1 retention (in percentage points, pp)
    c  Transfer across 7 PBMC protocols
    d  Cross-platform lineage transfer matrix
    """
    df_intra = pd.read_csv(os.path.join(tables_dir, "perf_intra_dataset_summary.csv"))
    df_cross = pd.read_csv(os.path.join(tables_dir, "perf_cross_batch_transfer.csv"))
    df_tasks = pd.read_csv(os.path.join(tables_dir, "perf_cross_batch_tasks.csv"))
    df_flow = pd.read_csv(os.path.join(tables_dir, "perf_cross_platform_flow.csv")).set_index("Reference_True_Lineage")

    fig = plt.figure(figsize=(16.2, 9.4), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.16, 1.0], height_ratios=[1.0, 1.08], hspace=0.44, wspace=0.28)

    ds_order = ["Pancreas", "PBMC", "Tabula Muris", "Lung COVID-19"]

    # Panel a: Cross-Batch Macro-F1 Grouped Barplot with clipped SD whiskers
    ax_a = fig.add_subplot(gs[0, 0])
    draw_grouped_bars_clipped_err(
        ax_a, df_cross, "Dataset", "Macro_F1", "Macro_F1_Std",
        ds_order, "Method", METHODS_ORDER, METHOD_PALETTE
    )
    set_nc_panel_title(ax_a, "a", "Cross-platform transfer Macro-F1", pad=34)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Cross-Platform Macro-F1", fontsize=11.0, fontweight="bold")
    ax_a.set_ylim(0.25, 1.03)
    ax_a.set_yticks([0.3, 0.5, 0.7, 0.9, 1.0])
    ax_a.tick_params(axis="both", labelsize=9.8)
    ax_a.grid(axis="y", zorder=0)
    leg_a = ax_a.legend(
        ncol=6, loc="lower center", bbox_to_anchor=(0.50, 1.02),
        fontsize=8.4, columnspacing=0.85, handletextpad=0.35, borderpad=0.35
    )
    style_legend(leg_a)
    remove_bar_borders(ax_a)
    sns.despine(ax=ax_a)

    # Panel b: Rare-Subtype F1 Generalization Retention Dumbbell Plot (percentage points, pp)
    ax_b = fig.add_subplot(gs[0, 1])
    in_mean = df_intra.groupby("Method")["Rare_F1_Mean"].mean()
    cb_mean = df_cross.groupby("Method")["Rare_F1"].mean()
    y_pos = np.arange(len(METHODS_ORDER))

    for idx, m_name in enumerate(METHODS_ORDER):
        v_in = float(in_mean.get(m_name, 0.8))
        v_cb = float(cb_mean.get(m_name, 0.75))
        diff_pp = (v_cb - v_in) * 100.0
        is_ours = "Ours" in m_name
        ax_b.plot(
            [min(v_in, v_cb), max(v_in, v_cb)], [idx, idx],
            color=METHOD_PALETTE[m_name], linewidth=3.0 if is_ours else 2.5, zorder=3
        )
        ax_b.scatter(
            v_in, idx, color=METHOD_PALETTE[m_name], s=92, marker="o",
            edgecolors="#FFFFFF", linewidths=1.0, zorder=4, label="Intra-Dataset CV" if idx == 0 else ""
        )
        ax_b.scatter(
            v_cb, idx, color=METHOD_PALETTE[m_name], s=102, marker="D",
            edgecolors="#FFFFFF", linewidths=1.0, zorder=4, label="Cross-Platform" if idx == 0 else ""
        )
        lbl_x = max(v_in, v_cb) + 0.015
        txt_kw = dict(va="center", ha="left", fontsize=9.1)
        if is_ours:
            ax_b.text(
                lbl_x, idx, f"Cross={v_cb:.3f} ({diff_pp:+.1f} pp)",
                fontweight="bold", color=PAL["rose_dark"],
                bbox=dict(boxstyle="round,pad=0.20", facecolor=PAL["rose_bg"], edgecolor=PAL["rose_edge"], lw=0.8),
                **txt_kw,
            )
        else:
            ax_b.text(
                lbl_x, idx, f"Cross={v_cb:.3f} ({diff_pp:+.1f} pp)",
                fontweight="normal", color=PAL["ink_tick"], **txt_kw
            )

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(METHODS_ORDER, fontsize=10.0, fontweight="bold")
    ax_b.set_xlim(0.35, 1.25)
    ax_b.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax_b.tick_params(axis="x", labelsize=9.6)
    ax_b.set_xlabel("Mean Rare-Subtype F1", fontsize=10.8, fontweight="bold", labelpad=5)
    set_nc_panel_title(ax_b, "b", "Rare-subtype F1 retention", pad=12)
    ax_b.grid(axis="x")
    leg_b = ax_b.legend(loc="lower left", fontsize=9.0, borderpad=0.40)
    style_legend(leg_b)
    sns.despine(ax=ax_b)

    # Panel c: Per-Protocol Transfer Macro-F1 across the 7 PBMC Target Protocols
    ax_c = fig.add_subplot(gs[1, 0])
    df_pbmc_cb = df_tasks[df_tasks["Dataset_Key"] == "pbmc"].copy()
    df_pbmc_cb["Protocol"] = (
        df_pbmc_cb["Test_Batch"]
        .str.replace(" (pbmc1)", "", regex=False)
        .str.replace(" (pbmc2)", " (b2)", regex=False)
    )
    prot_disp_map = {
        "10Xv3": "10x v3",
        "10Xv2 (b2)": "10x v2 (b2)",
        "Smart-Seq2": "Smart-seq2",
        "Drop-Seq": "Drop-seq",
    }
    prot_order = ["10Xv3", "10Xv2 (b2)", "Smart-Seq2", "inDrop", "Drop-Seq", "Seq-Well", "CEL-Seq2"]
    sns.barplot(
        data=df_pbmc_cb, x="Protocol", y="Macro_F1", hue="Method",
        order=prot_order, hue_order=METHODS_ORDER, palette=METHOD_PALETTE,
        saturation=1.0, gap=0.07, edgecolor="none", linewidth=0.0,
        width=0.78, alpha=0.92, zorder=3, ax=ax_c
    )
    set_nc_panel_title(ax_c, "c", "Transfer across 7 PBMC protocols", pad=12)
    ax_c.set_xlabel("Target Query Protocol (10x v2 Reference)", fontsize=10.8, fontweight="bold", labelpad=5)
    ax_c.set_xticklabels([prot_disp_map.get(p, p) for p in prot_order])
    ax_c.set_ylabel("Macro-F1 Score", fontsize=11.0, fontweight="bold")
    ax_c.set_ylim(0.08, 1.04)
    ax_c.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.tick_params(axis="y", labelsize=9.6)
    ax_c.tick_params(axis="x", labelsize=9.4)
    plt.setp(ax_c.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")
    ax_c.get_legend().remove()
    remove_bar_borders(ax_c)
    ax_c.grid(axis="y")
    sns.despine(ax=ax_c)

    # Panel d: Cross-Platform Lineage Transfer Matrix on PBMC
    ax_d = fig.add_subplot(gs[1, 1])
    short_ct = {
        "Plasmacytoid dendritic cell": "Plasmacytoid DC",
        "Natural killer cell": "NK cell",
        "Cytotoxic T cell": "Cytotoxic T",
        "Dendritic cell": "Dendritic cell",
        "CD14+ monocyte": "CD14+ Mono",
        "CD16+ monocyte": "CD16+ Mono",
    }
    df_flow_disp = df_flow.rename(index=short_ct, columns=short_ct)
    sns.heatmap(
        df_flow_disp, cmap=CMAP_NC_TEAL, vmin=0.0, vmax=1.0,
        annot=True, fmt=".2f", annot_kws={"size": 8.4, "weight": "bold"},
        linewidths=0.45, linecolor="#FFFFFF", cbar_kws={"label": "Transfer Recall"}, ax=ax_d
    )
    style_heatmap_annot_contrast(ax_d)
    set_nc_panel_title(ax_d, "d", "Cross-platform lineage transfer matrix", pad=12)
    ax_d.set_xlabel("Predicted Query Lineage", fontsize=10.5, fontweight="bold", labelpad=5)
    ax_d.set_ylabel("True Reference Lineage", fontsize=10.5, fontweight="bold")
    ax_d.tick_params(axis="x", labelsize=8.9)
    plt.setp(ax_d.get_xticklabels(), rotation=32, ha="right", rotation_mode="anchor")
    ax_d.tick_params(axis="y", rotation=0, labelsize=8.9)

    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


def plot_fig3_rare_celltypes(tables_dir: str, out_pdf: str) -> None:
    """
    Figure 4 in manuscript (`fig3_rare_celltype_analysis.pdf`):
    a  Per-subtype F1 across 12 rare lineages (*intra-only)
    b  F1 across cell prevalence tiers
    c  Rare-F1 across protocol sparsity
    d  Subspace neighbor affinity (HA-SGMP cosine gate distribution)
    """
    df_rare = pd.read_csv(os.path.join(tables_dir, "perf_rare_celltypes.csv"))
    df_tiers = pd.read_csv(os.path.join(tables_dir, "perf_prevalence_tiers.csv"))
    df_drop = pd.read_csv(os.path.join(tables_dir, "ablation_dropout_sparsity.csv"))
    df_gate = pd.read_csv(os.path.join(tables_dir, "interpret_hasgmp_gate_distribution.csv"))

    fig = plt.figure(figsize=(16.4, 9.8), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.02], hspace=0.58, wspace=0.28)

    # Panel a: Top full-width barplot across 12 rare subpopulations (* = single-batch intra CV)
    ax_a = fig.add_subplot(gs[0, :])
    df_rare_plot = df_rare.copy()
    ds_short_map = {"Pancreas": "Panc.", "PBMC": "PBMC", "Tabula Muris": "TM", "Lung COVID-19": "Lung"}
    ct_short_map = {"cardiac muscle cell": "cardiac muscle", "luminal epithelial": "luminal epith."}
    df_rare_plot["Subtype_Label"] = (
        df_rare_plot["Dataset"].replace(ds_short_map) + "\n"
        + df_rare_plot["Rare_Cell_Type"].replace(ct_short_map)
        + " (" + df_rare_plot["Prevalence_Pct"].astype(str) + "%)"
    )
    sns.barplot(
        data=df_rare_plot, x="Subtype_Label", y="F1_Score", hue="Method",
        hue_order=METHODS_ORDER, palette=METHOD_PALETTE,
        saturation=1.0, gap=0.07, edgecolor="none", linewidth=0.0,
        width=0.78, alpha=0.92, zorder=3, ax=ax_a
    )
    set_nc_panel_title(ax_a, "a", "Per-subtype F1 across 12 rare lineages (*intra-only)", pad=34)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Rare Subtype F1-Score", fontsize=11.0, fontweight="bold")
    ax_a.set_ylim(0.0, 1.03)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.tick_params(axis="y", labelsize=9.8)
    ax_a.tick_params(axis="x", labelsize=8.9)
    plt.setp(ax_a.get_xticklabels(), rotation=14, ha="right", rotation_mode="anchor")
    ax_a.grid(axis="y")
    leg_a = ax_a.legend(
        ncol=6, loc="lower center", bbox_to_anchor=(0.50, 1.02),
        fontsize=9.2, columnspacing=1.25, handletextpad=0.45, borderpad=0.35
    )
    style_legend(leg_a)
    remove_bar_borders(ax_a)
    sns.despine(ax=ax_a)

    # Panel b: F1 Trajectory across Cell-Type Prevalence Tiers
    ax_b = fig.add_subplot(gs[1, 0])
    for m_name in METHODS_ORDER:
        sub = df_tiers[df_tiers["Method"] == m_name].sort_values("Tier_Order")
        is_ours = "Ours" in m_name
        lw = 2.8 if is_ours else 1.55
        ms = 6.0 if is_ours else 4.8
        ax_b.plot(
            sub["Prevalence_Tier"], sub["F1_Mean"], marker="o", markersize=ms,
            linewidth=lw, color=METHOD_PALETTE[m_name], zorder=6 if is_ours else 4, label=m_name
        )
    set_nc_panel_title(ax_b, "b", "F1 across cell prevalence tiers", pad=12, title_size=12.4)
    ax_b.set_xlabel("Cell Prevalence Tier", fontsize=10.5, fontweight="bold", labelpad=5)
    ax_b.set_ylabel("Mean Cross-Platform F1", fontsize=10.5, fontweight="bold")
    ax_b.set_ylim(0.32, 0.96)
    ax_b.tick_params(axis="y", labelsize=9.4)
    ax_b.tick_params(axis="x", labelsize=8.8)
    plt.setp(ax_b.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")
    ax_b.grid(True)
    sns.despine(ax=ax_b)

    # Panel c: Multi-protocol zero-inflation sparsity across 7 PBMC target protocols
    ax_c = fig.add_subplot(gs[1, 1])
    drop_palette = {
        "scPathProto (Ours)": PAL["rose_dark"],
        "scPathProto (w/o ZIRG)": PAL["teal_dark"],
        "SVM": PAL["amber_mid"],
        "CellTypist": PAL["navy_cerulean"],
        "scANVI": PAL["iris_mid"],
        "SingleR": PAL["slate"],
    }
    for m_name, c_hex in drop_palette.items():
        sub = df_drop[df_drop["Method"] == m_name].dropna(subset=["Rare_F1"]).sort_values("Zero_Sparsity_Pct")
        if len(sub) == 0:
            continue
        is_ours = "Ours" in m_name
        lw = 2.7 if is_ours else 1.65
        ls = "--" if "w/o ZIRG" in m_name else "-"
        mk = "o" if is_ours else "s"
        ax_c.plot(
            sub["Zero_Sparsity_Pct"], sub["Rare_F1"], marker=mk, markersize=5.2,
            linestyle=ls, linewidth=lw, color=c_hex, zorder=6 if is_ours else 4, label=m_name
        )
    set_nc_panel_title(ax_c, "c", "Rare-F1 across protocol sparsity", pad=12, title_size=12.4)
    ax_c.set_xlabel("Natural Query Zero-Inflation Sparsity (%)", fontsize=10.2, fontweight="bold", labelpad=5)
    ax_c.set_ylabel("PBMC Cross-Platform Rare-F1", fontsize=10.5, fontweight="bold")
    y_min_c = max(0.0, float(df_drop["Rare_F1"].dropna().min()) - 0.06)
    ax_c.set_ylim(y_min_c, 1.03)
    ax_c.tick_params(axis="both", labelsize=9.4)
    leg_c = ax_c.legend(loc="lower left", fontsize=8.0, ncol=2, columnspacing=0.65, handletextpad=0.35, borderpad=0.35)
    style_legend(leg_c)
    ax_c.grid(True)
    sns.despine(ax=ax_c)

    # Panel d: HA-SGMP Subspace Neighbor Cosine Similarity (\kappa_{ij}) Distribution
    ax_d = fig.add_subplot(gs[1, 2])
    ax_d.set_axisbelow(True)
    sns.violinplot(
        data=df_gate, x="Model", y="Gate_Weight", hue="Edge_Category",
        split=True, inner="quartile", cut=0, linewidth=0.90,
        saturation=1.0, alpha=0.90, zorder=3,
        palette={"Homophilic (Same Subtype)": PAL["teal_mid"], "Heterophilic (Rare Boundary)": PAL["rose_mid"]},
        ax=ax_d
    )
    for ln in ax_d.lines:
        ln.set_zorder(5)
        ln.set_alpha(0.95)
    ax_d.axhline(0.72, color=PAL["ink_mid"], linestyle="--", linewidth=1.35, zorder=6, label="Boundary Gate ($\\kappa = 0.72$)")
    set_nc_panel_title(ax_d, "d", "Subspace neighbor affinity", pad=36, title_size=12.4)
    ax_d.set_xlabel("")
    ax_d.set_ylabel("Subspace Cosine Similarity ($\\kappa_{ij}$)", fontsize=10.2, fontweight="bold")
    y_min_g = min(-0.05, float(df_gate["Gate_Weight"].min()) - 0.03)
    ax_d.set_ylim(y_min_g, 1.02)
    ax_d.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.tick_params(axis="y", labelsize=9.4)
    ax_d.tick_params(axis="x", labelsize=9.0)
    leg_d = ax_d.legend(
        loc="lower center", bbox_to_anchor=(0.50, 1.02), ncol=2,
        fontsize=7.8, columnspacing=0.65, handletextpad=0.35, borderpad=0.30
    )
    style_legend(leg_d)
    ax_d.grid(axis="y")
    sns.despine(ax=ax_d)

    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


def plot_fig4_openset(tables_dir: str, out_pdf: str) -> None:
    """
    Figure 5 in manuscript (`fig4_openset_roc_prc.pdf`):
    a  Open-set ROC curves (with ±1 SEM band)
    b  Open-set precision-recall curves
    c  Open-set harmonic H-score across datasets
    d  Confidence calibration on open-set cohorts (37/37 tasks, ECE=0.088)
    """
    df_os = pd.read_csv(os.path.join(tables_dir, "perf_openset_discovery.csv"))
    df_curves = pd.read_csv(os.path.join(tables_dir, "perf_openset_curves.csv"))
    df_ece = pd.read_csv(os.path.join(tables_dir, "perf_calibration_ece.csv"))

    fig = plt.figure(figsize=(16.0, 9.2), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.25)

    # Exact unrounded 4-dataset mean AUROC/AUPRC from perf_openset_discovery.csv
    # (prevents double-rounding CellTypist 0.244542 -> 0.2445 -> 0.244 instead of 0.245)
    auroc_exact = df_os.groupby("Method")["OpenSet_AUROC"].mean().to_dict()
    auprc_exact = df_os.groupby("Method")["OpenSet_AUPRC"].mean().to_dict()

    # Panel a: Open-set ROC curves (cross-dataset SEM)
    ax_a = fig.add_subplot(gs[0, 0])
    df_roc = df_curves[df_curves["Curve_Type"] == "ROC"]
    for m_name in METHODS_ORDER:
        sub = df_roc[df_roc["Method"] == m_name]
        if len(sub) == 0:
            continue
        auc_v = float(auroc_exact.get(m_name, sub["AUC"].iloc[0]))
        is_ours = "Ours" in m_name
        lw = 2.7 if is_ours else 1.55
        ax_a.plot(
            sub["X"], sub["Y"], color=METHOD_PALETTE[m_name], linewidth=lw,
            zorder=6 if is_ours else 4, label=f"{m_name} (AUROC={auc_v:.3f})"
        )
        if is_ours and "SEM" in sub.columns:
            sem_v = sub["SEM"].to_numpy(dtype=float)
            y_v = sub["Y"].to_numpy(dtype=float)
            ax_a.fill_between(
                sub["X"],
                np.maximum(0.0, y_v - sem_v),
                np.minimum(1.0, y_v + sem_v),
                color=METHOD_PALETTE[m_name],
                alpha=0.16,
                edgecolor="none",
                zorder=5,
                label="±1 SEM across 4 atlases",
            )
    ax_a.plot([0, 1], [0, 1], "--", color=PAL["slate_muted"], linewidth=1.1, zorder=2)
    set_nc_panel_title(ax_a, "a", "Open-set ROC curves", pad=12)
    ax_a.set_xlabel("False Positive Rate", fontsize=10.8, fontweight="bold", labelpad=5)
    ax_a.set_ylabel("True Positive Rate", fontsize=10.8, fontweight="bold")
    ax_a.tick_params(axis="both", labelsize=9.6)
    leg_a = ax_a.legend(loc="lower right", fontsize=8.8, borderpad=0.40)
    style_legend(leg_a)
    ax_a.grid(True)
    sns.despine(ax=ax_a)

    # Panel b: Open-Set Precision-Recall Curves
    ax_b = fig.add_subplot(gs[0, 1])
    df_prc = df_curves[df_curves["Curve_Type"] == "PRC"]
    for m_name in METHODS_ORDER:
        sub = df_prc[df_prc["Method"] == m_name]
        if len(sub) == 0:
            continue
        auc_v = float(auprc_exact.get(m_name, sub["AUC"].iloc[0]))
        is_ours = "Ours" in m_name
        lw = 2.7 if is_ours else 1.55
        ax_b.plot(
            sub["X"], sub["Y"], color=METHOD_PALETTE[m_name], linewidth=lw,
            zorder=6 if is_ours else 4, label=f"{m_name} (AUPRC={auc_v:.3f})"
        )
    set_nc_panel_title(ax_b, "b", "Open-set precision-recall curves", pad=12)
    ax_b.set_xlabel("Recall (Unseen Cell Types)", fontsize=10.8, fontweight="bold", labelpad=5)
    ax_b.set_ylabel("Precision (Unseen Cell Types)", fontsize=10.8, fontweight="bold")
    ax_b.set_ylim(0.02, 0.88)
    ax_b.tick_params(axis="both", labelsize=9.6)
    leg_b = ax_b.legend(loc="upper right", fontsize=8.8, borderpad=0.40)
    style_legend(leg_b)
    ax_b.grid(True)
    sns.despine(ax=ax_b)

    # Panel c: Open-Set Harmonic H-Score Across Datasets (with clipped task SD)
    ax_c = fig.add_subplot(gs[1, 0])
    ds_order = ["Pancreas", "PBMC", "Tabula Muris", "Lung COVID-19"]
    draw_grouped_bars_clipped_err(
        ax_c, df_os, "Dataset", "H_Score", "H_Score_Std",
        ds_order, "Method", METHODS_ORDER, METHOD_PALETTE
    )
    set_nc_panel_title(ax_c, "c", "Open-set harmonic H-score", pad=12)
    ax_c.set_xlabel("Single-Cell Atlas", fontsize=10.8, fontweight="bold", labelpad=5)
    ax_c.set_ylabel("Open-Set Harmonic H-Score", fontsize=10.8, fontweight="bold")
    ax_c.set_ylim(0.0, 1.06)
    ax_c.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.tick_params(axis="both", labelsize=9.6)
    leg_c = ax_c.legend(ncol=3, loc="upper right", fontsize=8.3, columnspacing=0.85, handletextpad=0.40, borderpad=0.35)
    style_legend(leg_c)
    remove_bar_borders(ax_c)
    ax_c.grid(axis="y", zorder=0)
    sns.despine(ax=ax_c)

    # Panel d: Reliability Diagram & Expected Calibration Error (all 6 methods including SingleR*)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot([0.1, 1.0], [0.1, 1.0], "--", color=PAL["slate_muted"], linewidth=1.3, zorder=2, label="Perfect Calibration")
    for m_name in METHODS_ORDER:
        sub = df_ece[df_ece["Method"] == m_name].sort_values("Confidence_Bin")
        if len(sub) == 0:
            continue
        ece_v = float(sub["ECE_Score"].iloc[0])
        is_ours = "Ours" in m_name
        lw = 2.7 if is_ours else 1.60
        ls = ":" if m_name == "SingleR" else "-"
        lbl_m = f"{m_name}* (ECE={ece_v:.3f})" if m_name == "SingleR" else f"{m_name} (ECE={ece_v:.3f})"
        ax_d.plot(
            sub["Confidence_Bin"], sub["Empirical_Accuracy"],
            marker="o", markersize=5.2 if is_ours else 4.5, linestyle=ls, linewidth=lw,
            color=METHOD_PALETTE[m_name], zorder=6 if is_ours else 4, label=lbl_m
        )
    set_nc_panel_title(ax_d, "d", "Confidence calibration on open-set cohorts", pad=12)
    ax_d.set_xlabel("Predicted Confidence (*SingleR: correlation score)", fontsize=10.4, fontweight="bold", labelpad=5)
    ax_d.set_ylabel("Empirical Accuracy (All Test Cells)", fontsize=10.6, fontweight="bold")
    ax_d.set_xlim(0.12, 1.01)
    ax_d.set_ylim(0.02, 1.06)
    ax_d.tick_params(axis="both", labelsize=9.6)
    leg_d = ax_d.legend(loc="upper left", fontsize=8.0, labelspacing=0.30, borderpad=0.34)
    style_legend(leg_d)
    ax_d.grid(True)
    sns.despine(ax=ax_d)

    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


def plot_fig5_ablation_and_sensitivity(tables_dir: str, out_pdf: str) -> None:
    """
    Figure 6 in manuscript (`fig5_ablation_and_sensitivity.pdf`):
    a  Module ablation across metrics (4-dataset mean ± SEM)
    b  Ablation impact on rare-subtype F1 (Oxford Navy <-> Berry Rose)
    c  Hyperparameter sensitivity surface (Warm Amber-Bronze)
    d  Sensitivity to rare margin scale m_0
    """
    df_ab = pd.read_csv(os.path.join(tables_dir, "ablation_study.csv"))
    df_sens = pd.read_csv(os.path.join(tables_dir, "hyperparam_sensitivity.csv"))
    df_grid = pd.read_csv(os.path.join(tables_dir, "ablation_2d_grid.csv"))

    fig = plt.figure(figsize=(16.2, 9.4), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    short_var_map = {
        "Full scPathProto": "Full scPathProto",
        "w/o ZIRG (Denoising Gate)": "w/o ZIRG",
        "w/o MPCA (Pathway Prior)": "w/o MPCA",
        "w/o HA-SGMP (Heterophily Graph)": "w/o HA-SGMP",
        "w/o Proto-Align": "w/o Proto-Align",
        "w/o Proto-OT Alignment": "w/o Proto-Align",
        "w/o Evidential Open-Set Head": "w/o Evidential",
    }
    short_ab_palette = {short_var_map[k]: v for k, v in ABLATION_PALETTE.items()}

    # Panel a: Grouped Ablation Barplot with 4-Dataset SEM whiskers
    ax_a = fig.add_subplot(gs[0, 0])
    df_ab_short = df_ab.copy()
    df_ab_short["Variant_Short"] = df_ab_short["Variant"].replace(short_var_map)
    df_melt = df_ab_short.melt(
        id_vars=["Dataset", "Variant_Short"],
        value_vars=["CrossBatch_Acc", "CrossBatch_Macro_F1", "CrossBatch_Rare_F1", "OpenSet_AUROC", "OpenSet_H_Score"],
        var_name="Metric",
        value_name="Score",
    )
    metric_labels = {
        "CrossBatch_Acc": "Cross Acc",
        "CrossBatch_Macro_F1": "Cross Macro-F1",
        "CrossBatch_Rare_F1": "Cross Rare-F1",
        "OpenSet_AUROC": "Open AUROC",
        "OpenSet_H_Score": "Open H-Score",
    }
    df_melt["Metric"] = df_melt["Metric"].map(metric_labels)
    sns.barplot(
        data=df_melt, x="Metric", y="Score", hue="Variant_Short",
        palette=short_ab_palette, saturation=1.0, gap=0.07,
        edgecolor="none", linewidth=0.0, width=0.78,
        errorbar="se", capsize=0.03, err_kws={"linewidth": 0.90, "color": PAL["ink_mid"]},
        alpha=0.92, zorder=3, ax=ax_a
    )
    set_nc_panel_title(ax_a, "a", "Module ablation across metrics", pad=12)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("4-Dataset Mean Score (± SEM)", fontsize=10.8, fontweight="bold")
    y_min_a = max(0.25, float(df_melt.groupby(["Metric", "Variant_Short"])["Score"].mean().min()) - 0.10)
    ax_a.set_ylim(y_min_a, 1.07)
    ax_a.set_yticks([t for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if t >= y_min_a - 0.02])
    ax_a.tick_params(axis="both", labelsize=9.5)
    leg_a = ax_a.legend(ncol=3, loc="upper right", fontsize=8.3, columnspacing=0.85, handletextpad=0.40, borderpad=0.35)
    style_legend(leg_a)
    remove_bar_borders(ax_a)
    ax_a.grid(axis="y")
    sns.despine(ax=ax_a)

    # Panel b: Dataset x Ablation Variant Heatmap on Cross-Batch Rare-F1
    ax_b = fig.add_subplot(gs[0, 1])
    piv_ab = df_ab_short.pivot(index="Variant_Short", columns="Dataset", values="CrossBatch_Rare_F1")
    variant_order = list(short_ab_palette.keys())
    ds_order = ["Pancreas", "PBMC", "Tabula Muris", "Lung COVID-19"]
    piv_ab = piv_ab.reindex(index=[v for v in variant_order if v in piv_ab.index], columns=ds_order)
    sns.heatmap(
        piv_ab, cmap=CMAP_NC_DIV_RB, annot=True, fmt=".4f", annot_kws={"size": 9.3, "weight": "bold"},
        linewidths=0.55, linecolor="#FFFFFF", cbar_kws={"label": "Cross-Platform Rare-F1"}, ax=ax_b
    )
    style_heatmap_annot_contrast(ax_b)
    set_nc_panel_title(ax_b, "b", "Ablation impact on rare-subtype F1", pad=12)
    ax_b.set_xlabel("Single-Cell Atlas", fontsize=10.8, fontweight="bold", labelpad=5)
    ax_b.set_ylabel("")
    ax_b.tick_params(axis="x", labelsize=9.5)
    ax_b.tick_params(axis="y", rotation=0, labelsize=9.6)

    # Panel c: 2D Hyperparameter Sensitivity Surface on PBMC Cross-Platform Transfer
    ax_c = fig.add_subplot(gs[1, 0])
    piv_grid = df_grid.pivot(index="n_latent_pathways", columns="knn_k", values="Macro_F1")
    sns.heatmap(
        piv_grid, cmap=CMAP_NC_AMBER, annot=True, fmt=".4f", annot_kws={"size": 9.2, "weight": "bold"},
        linewidths=0.50, linecolor="#FFFFFF", cbar_kws={"label": "PBMC Cross-Platform Macro-F1"}, ax=ax_c
    )
    style_heatmap_annot_contrast(ax_c)
    set_nc_panel_title(ax_c, "c", "Hyperparameter sensitivity surface", pad=12)
    ax_c.set_xlabel("Subspace Neighborhood Size ($k$)", fontsize=10.8, fontweight="bold", labelpad=5)
    ax_c.set_ylabel("De Novo Pathway Slots ($P_{\\mathrm{latent}}$)", fontsize=10.8, fontweight="bold")
    ax_c.tick_params(axis="both", labelsize=9.6)
    ax_c.tick_params(axis="y", rotation=0)

    # Panel d: Sensitivity to Rare-Class Margin Scale m_0
    ax_d = fig.add_subplot(gs[1, 1])
    df_m = df_sens[df_sens["Param_Name"] == "rare_margin_scale"]
    for ds_key, c_hex in [("Pancreas", PAL["rose_dark"]), ("PBMC", PAL["teal_dark"])]:
        sub = df_m[df_m["Dataset"] == ds_key].sort_values("Param_Value")
        if len(sub) > 0:
            ax_d.plot(sub["Param_Value"], sub["Macro_F1"], marker="o", markersize=5.5, linewidth=2.4, color=c_hex, label=f"{ds_key} (Macro-F1)")
            ax_d.plot(sub["Param_Value"], sub["Rare_F1"], marker="s", markersize=5.0, linestyle="--", linewidth=1.9, color=c_hex, label=f"{ds_key} (Rare-F1)")
    ax_d.axvline(0.18, color=PAL["ink_mid"], linestyle=":", linewidth=1.45, label="Default ($m_0 = 0.18$)")
    set_nc_panel_title(ax_d, "d", "Sensitivity to rare margin scale", pad=12)
    ax_d.set_xlabel("Rare-Class Margin Scale ($m_0$)", fontsize=10.8, fontweight="bold", labelpad=5)
    ax_d.set_ylabel("Cross-Platform F1 Score", fontsize=10.8, fontweight="bold")
    min_f1_d = float(min(df_m["Rare_F1"].min(), df_m["Macro_F1"].min()))
    max_f1_d = float(max(df_m["Rare_F1"].max(), df_m["Macro_F1"].max()))
    ax_d.set_ylim(max(0.0, min_f1_d - 0.035), min(1.005, max_f1_d + 0.03))
    ax_d.tick_params(axis="both", labelsize=9.6)
    leg_d = ax_d.legend(loc="center right", bbox_to_anchor=(0.985, 0.52), fontsize=8.8, ncol=1, borderpad=0.42)
    style_legend(leg_d)
    ax_d.grid(True)
    sns.despine(ax=ax_d)

    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


def plot_fig6_pathway_heatmap(tables_dir: str, out_pdf: str) -> None:
    """
    Figure 7 in manuscript (`fig6_interpret_pathway_heatmap.pdf`):
    a  Curated GO-BP pathway activation landscape (Oxford Navy <-> Berry Rose)
    b  De novo latent regulon activation (Forest Pine-Teal <-> Warm Amber-Bronze)
    c  Empirical pathway co-activation matrix (Forest Pine-Teal <-> Warm Amber-Bronze)
    """
    df_path = pd.read_csv(os.path.join(tables_dir, "interpret_pathway_attention.csv")).set_index("Cell_Type")
    df_denovo = pd.read_csv(os.path.join(tables_dir, "interpret_denovo_regulons.csv")).set_index("Cell_Type")
    df_inter = pd.read_csv(os.path.join(tables_dir, "interpret_pathway_interaction.csv")).set_index("Pathway")

    fig = plt.figure(figsize=(17.2, 11.6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.90, 1.10], height_ratios=[1.12, 1.02], hspace=0.80, wspace=0.72)

    # Panel a: Top full-width Curated GO-BP Pathway Activation Heatmap
    ax_a = fig.add_subplot(gs[0, :])
    var_cols = df_path.columns[:18]
    sns.heatmap(
        df_path[var_cols], cmap=CMAP_NC_DIV_RB, center=0.0, annot=True, fmt=".2f",
        annot_kws={"size": 8.0, "weight": "bold"}, linewidths=0.40, linecolor="#FFFFFF",
        cbar_kws={"label": "Pathway Activation (Z-Score)", "pad": 0.018}, ax=ax_a
    )
    style_heatmap_annot_contrast(ax_a, cbar_labelsize=10.5, cbar_ticksize=9.2)
    set_nc_panel_title(ax_a, "a", "Curated GO-BP pathway activation landscape", pad=12, title_size=13.2)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Immune Cell Types", fontsize=11.0, fontweight="bold")
    ax_a.tick_params(axis="x", labelsize=9.0)
    plt.setp(ax_a.get_xticklabels(), rotation=32, ha="right", rotation_mode="anchor")
    ax_a.tick_params(axis="y", rotation=0, labelsize=9.6)

    # Panel b: De Novo Latent Regulon Activation Profile
    ax_b = fig.add_subplot(gs[1, 0])
    df_denovo_disp = df_denovo.rename(columns=lambda c: str(c).replace("_", " "))
    sns.heatmap(
        df_denovo_disp, cmap=CMAP_NC_DIV_TC, center=0.0, annot=True, fmt=".2f",
        annot_kws={"size": 8.6, "weight": "bold"}, linewidths=0.45, linecolor="#FFFFFF",
        cbar_kws={"label": "De Novo Activation (Z-Score)", "pad": 0.025}, ax=ax_b
    )
    style_heatmap_annot_contrast(ax_b, cbar_labelsize=10.2, cbar_ticksize=9.0)
    set_nc_panel_title(ax_b, "b", "De novo latent regulon activation", pad=12, title_size=13.0)
    ax_b.set_xlabel("De Novo Pathway Slots ($P_{\\mathrm{latent}}=8$)", fontsize=10.6, fontweight="bold", labelpad=6)
    ax_b.set_ylabel("Immune Cell Types", fontsize=10.6, fontweight="bold")
    ax_b.tick_params(axis="x", labelsize=9.2)
    plt.setp(ax_b.get_xticklabels(), rotation=28, ha="right", rotation_mode="anchor")
    ax_b.tick_params(axis="y", rotation=0, labelsize=9.4)

    # Panel c: Empirical Pathway Co-Activation Matrix (Pearson r across PBMC cells)
    ax_c = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        df_inter, cmap=CMAP_NC_DIV_TC, center=0.0, vmin=-1.0, vmax=1.0, annot=True, fmt=".2f",
        annot_kws={"size": 7.6, "weight": "bold"}, linewidths=0.40, linecolor="#FFFFFF",
        cbar_kws={"label": "Co-Activation (Pearson $r$)", "pad": 0.025}, ax=ax_c
    )
    style_heatmap_annot_contrast(ax_c, cbar_labelsize=10.2, cbar_ticksize=9.0)
    set_nc_panel_title(ax_c, "c", "Empirical pathway co-activation matrix", pad=12, title_size=13.0)
    ax_c.set_xlabel("GO-BP Pathway", fontsize=10.4, fontweight="bold", labelpad=6)
    ax_c.set_ylabel("GO-BP Pathway", fontsize=10.4, fontweight="bold")
    ax_c.tick_params(axis="x", labelsize=8.5)
    plt.setp(ax_c.get_xticklabels(), rotation=32, ha="right", rotation_mode="anchor")
    ax_c.tick_params(axis="y", rotation=0, labelsize=8.5)

    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


def plot_fig7_marker_dotplot(tables_dir: str, out_pdf: str) -> None:
    """
    Figure 8 in manuscript (`fig7_interpret_marker_dotplot.pdf`):
    a  Attributed marker expression dotplot
    b  Top-2 gene attribution (*canonical)
    c  Tripartite gene-pathway-celltype attribution graph
    """
    df_markers = pd.read_csv(os.path.join(tables_dir, "interpret_top_marker_genes.csv"))
    df_dot = pd.read_csv(os.path.join(tables_dir, "interpret_dotplot_matrix.csv"))
    top2_df = df_markers[df_markers["Rank"] <= 2].copy()

    cell_types = list(dict.fromkeys(top2_df["Cell_Type"].tolist()))
    selected_genes = list(dict.fromkeys(top2_df["Gene_Symbol"].tolist()))

    fig = plt.figure(figsize=(16.6, 11.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.05, 1.32], hspace=0.48, wspace=0.34)

    # Panel a: Attributed Marker Expression DotPlot + Explicit Dot-Size Legend
    ax_a = fig.add_subplot(gs[0, 0])
    ct_to_y = {ct: i for i, ct in enumerate(cell_types)}
    g_to_x = {g: j for j, g in enumerate(selected_genes)}
    xs = [g_to_x[g] for g in df_dot["Gene_Symbol"] if g in g_to_x]
    ys = [ct_to_y[c] for c, g in zip(df_dot["Cell_Type"], df_dot["Gene_Symbol"]) if g in g_to_x]
    sizes = [max(12.0, float(p) * 2.3) for p, g in zip(df_dot["Pct_Expressing"], df_dot["Gene_Symbol"]) if g in g_to_x]
    colors = [float(v) for v, g in zip(df_dot["Mean_LogExpr"], df_dot["Gene_Symbol"]) if g in g_to_x]

    sc = ax_a.scatter(xs, ys, s=sizes, c=colors, cmap=CMAP_NC_ROSE, edgecolors=PAL["ink_mid"], linewidths=0.55, zorder=3)
    ax_a.set_xticks(range(len(selected_genes)))
    ax_a.set_xticklabels(selected_genes, rotation=36, ha="right", fontsize=9.4, fontweight="bold")
    ax_a.set_yticks(range(len(cell_types)))
    ax_a.set_yticklabels(cell_types, fontsize=9.6)
    ax_a.set_ylim(-0.55, len(cell_types) - 0.35)
    set_nc_panel_title(ax_a, "a", "Attributed marker expression dotplot", pad=46)
    ax_a.grid(linestyle=":", alpha=0.55, zorder=0)
    cbar = plt.colorbar(sc, ax=ax_a, fraction=0.03, pad=0.02)
    cbar.outline.set_edgecolor("#CBD5E1")
    cbar.outline.set_linewidth(0.65)
    cbar.set_label("Mean Log1p Expression", fontsize=9.8, fontweight="bold", color=PAL["ink_dark"])
    cbar.ax.tick_params(labelsize=8.8, colors=PAL["ink_tick"])

    # Quantitative dot-size legend (% Expressing: 25%, 50%, 75%, 100%)
    dot_handles = [
        plt.scatter([], [], s=pct * 2.3, facecolor=PAL["rose_sub"], edgecolor=PAL["rose_dark"], linewidth=0.7, label=f"{pct}%")
        for pct in [25, 50, 75, 100]
    ]
    leg_dot = ax_a.legend(
        handles=dot_handles,
        title="% Cells Expressing:",
        loc="lower center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=4,
        fontsize=8.6,
        title_fontsize=8.8,
        columnspacing=1.0,
        handletextpad=0.45,
        borderpad=0.35,
    )
    leg_dot.get_title().set_fontweight("bold")
    style_legend(leg_dot)
    sns.despine(ax=ax_a)

    # Panel b: Top-2 Attribution Ranking Barplot
    ax_b = fig.add_subplot(gs[0, 1])
    short_ct_map = {
        "Plasmacytoid dendritic cell": "Plasmacytoid DC",
        "Natural killer cell": "NK cell",
        "Cytotoxic T cell": "Cytotoxic T",
        "Dendritic cell": "Dendritic cell",
        "CD14+ monocyte": "CD14+ Mono",
        "CD16+ monocyte": "CD16+ Mono",
    }
    top2_plot = top2_df.copy()
    top2_plot["Short_Cell_Type"] = top2_plot["Cell_Type"].replace(short_ct_map)
    top2_plot["Rank_Label"] = top2_plot["Rank"].map({1: "Top-1 Gene", 2: "Top-2 Gene"})
    sns.barplot(
        data=top2_plot, x="Attribution_Score", y="Short_Cell_Type", hue="Rank_Label",
        saturation=1.0, gap=0.08, edgecolor="none", linewidth=0.0,
        width=0.74, alpha=0.92, zorder=3,
        palette={"Top-1 Gene": PAL["rose_dark"], "Top-2 Gene": PAL["navy_cerulean"]}, ax=ax_b
    )
    for idx, ct in enumerate(cell_types):
        sub_c = top2_plot[top2_plot["Cell_Type"] == ct].sort_values("Rank")
        r1, r2 = sub_c.iloc[0], sub_c.iloc[1]
        g1 = f"{r1['Gene_Symbol']}*" if bool(r1["Is_Canonical_Marker"]) else str(r1["Gene_Symbol"])
        g2 = f"{r2['Gene_Symbol']}*" if bool(r2["Is_Canonical_Marker"]) else str(r2["Gene_Symbol"])
        s2 = float(r2["Attribution_Score"])
        ax_b.text(1.015, idx - 0.19, g1, va="center", ha="left", fontsize=8.8, fontweight="bold", color=PAL["rose_dark"])
        ax_b.text(s2 + 0.015, idx + 0.20, g2, va="center", ha="left", fontsize=8.6, color=PAL["navy_dark"])
    set_nc_panel_title(ax_b, "b", "Top-2 gene attribution (*canonical)", pad=46)
    ax_b.set_xlabel("Normalized Two-Hop Attribution Score", fontsize=10.6, fontweight="bold", labelpad=5)
    ax_b.set_ylabel("")
    ax_b.set_xlim(0.0, 1.30)
    ax_b.tick_params(axis="both", labelsize=9.4)
    leg_b = ax_b.legend(
        title="Attribution Rank:",
        loc="lower center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=2,
        fontsize=8.6,
        title_fontsize=8.8,
        columnspacing=1.2,
        handletextpad=0.5,
        borderpad=0.35,
    )
    leg_b.get_title().set_fontweight("bold")
    style_legend(leg_b)
    remove_bar_borders(ax_b)
    ax_b.grid(axis="x")
    sns.despine(ax=ax_b)

    # Panel c: Bottom full-width tripartite attribution graph
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_xlim(0.0, 10.0)
    ax_c.set_ylim(-0.65, len(cell_types) + 0.45)
    ax_c.axis("off")
    set_nc_panel_title(ax_c, "c", "Tripartite biological attribution graph", pad=16, title_size=13.2)

    x_ct, x_pw, x_gn = 1.32, 5.02, 8.70
    ax_c.text(x_ct, len(cell_types) + 0.05, "Immune Lineage", fontsize=12.2, fontweight="bold", ha="center", color=PAL["rose_dark"])
    ax_c.text(x_pw, len(cell_types) + 0.05, "Dominant GO-BP Pathway", fontsize=12.2, fontweight="bold", ha="center", color=PAL["teal_dark"])
    ax_c.text(x_gn, len(cell_types) + 0.05, "Top-2 Attributed Genes", fontsize=12.2, fontweight="bold", ha="center", color=PAL["navy_dark"])

    for idx, ct in enumerate(reversed(cell_types)):
        sub_ct = top2_df[top2_df["Cell_Type"] == ct].sort_values("Rank")
        if len(sub_ct) == 0:
            continue
        dom_pw = str(sub_ct["Dominant_Pathway"].iloc[0])
        genes_str = ",  ".join(sub_ct["Gene_Symbol"].tolist())

        ax_c.annotate(
            "", xy=(x_pw - 1.88, idx), xytext=(x_ct + 1.28, idx),
            arrowprops=dict(arrowstyle="-|>", color=PAL["slate"], lw=1.7, mutation_scale=12),
            zorder=2,
        )
        ax_c.annotate(
            "", xy=(x_gn - 1.05, idx), xytext=(x_pw + 1.88, idx),
            arrowprops=dict(arrowstyle="-|>", color=PAL["teal_mid"], lw=1.7, mutation_scale=12),
            zorder=2,
        )

        ax_c.text(
            x_ct, idx, ct, ha="center", va="center", fontsize=10.2, fontweight="bold", color=PAL["rose_dark"],
            bbox=dict(boxstyle="round,pad=0.30", facecolor=PAL["rose_bg"], edgecolor=PAL["rose_edge"], lw=1.05),
            zorder=4,
        )
        ax_c.text(
            x_pw, idx, dom_pw, ha="center", va="center", fontsize=9.8, fontweight="bold", color=PAL["teal_dark"],
            bbox=dict(boxstyle="round,pad=0.30", facecolor=PAL["teal_bg"], edgecolor=PAL["teal_edge"], lw=1.05),
            zorder=4,
        )
        ax_c.text(
            x_gn, idx, genes_str, ha="center", va="center", fontsize=10.2, fontweight="bold", color=PAL["navy_dark"],
            bbox=dict(boxstyle="round,pad=0.30", facecolor=PAL["navy_bg"], edgecolor=PAL["navy_edge"], lw=1.05),
            zorder=4,
        )

    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


def plot_fig8_manifold_and_confusion(tables_dir: str, out_pdf: str) -> None:
    r"""
    Figure 9 in manuscript (`fig8_latent_manifold_and_confusion.pdf`):
    a  Raw unaligned expression manifold
    b  scPathProto hyperspherical manifold + learned unit prototypes \mu_c
    c  Cross-protocol latent batch mixing (7-protocol palette)
    """
    df_man = pd.read_csv(os.path.join(tables_dir, "interpret_latent_embedding.csv"))
    df_cm = pd.read_csv(os.path.join(tables_dir, "interpret_confusion_matrix.csv"))
    proto_path = os.path.join(tables_dir, "interpret_prototype_tsne.csv")
    df_proto = pd.read_csv(proto_path) if os.path.exists(proto_path) else None
    cm_mat = df_cm.set_index("True_Cell_Type") if "True_Cell_Type" in df_cm.columns else df_cm

    cell_types = list(cm_mat.index)
    ct_color = {ct: NC_CELLTYPE_PALETTE[i % len(NC_CELLTYPE_PALETTE)] for i, ct in enumerate(cell_types)}

    fig = plt.figure(figsize=(17.8, 5.7), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.26)

    # Panel a: Raw Noisy Expression Manifold
    ax_a = fig.add_subplot(gs[0, 0])
    sns.scatterplot(
        data=df_man, x="Raw_TSNE_1", y="Raw_TSNE_2", hue="Cell_Type",
        hue_order=cell_types, palette=ct_color, s=18, alpha=0.82, linewidth=0, legend=False, ax=ax_a
    )
    set_nc_panel_title(ax_a, "a", "Raw unaligned expression manifold", pad=12, title_size=12.6)
    ax_a.set_xlabel("t-SNE 1", fontsize=11.0, fontweight="bold")
    ax_a.set_ylabel("t-SNE 2", fontsize=11.0, fontweight="bold")
    ax_a.tick_params(axis="both", labelsize=9.6)
    sns.despine(ax=ax_a)

    # Panel b: scPathProto Hyperspherical Manifold + Jointly Projected Unit Prototypes \mu_c
    ax_b = fig.add_subplot(gs[0, 1])
    sns.scatterplot(
        data=df_man, x="scPathProto_TSNE_1", y="scPathProto_TSNE_2", hue="Cell_Type",
        hue_order=cell_types, palette=ct_color, s=18, alpha=0.85, linewidth=0, ax=ax_b
    )
    if df_proto is not None:
        px, py = df_proto["Proto_TSNE_1"], df_proto["Proto_TSNE_2"]
    else:
        med = df_man.groupby("Cell_Type")[["scPathProto_TSNE_1", "scPathProto_TSNE_2"]].median()
        px, py = med["scPathProto_TSNE_1"], med["scPathProto_TSNE_2"]
    ax_b.scatter(
        px, py,
        marker="*", s=250, c=PAL["amber_soft"], edgecolors=PAL["amber_dark"], linewidths=1.15, zorder=10,
        label="Learned Prototype ($\\boldsymbol{\\mu}_c$)",
    )
    set_nc_panel_title(ax_b, "b", "scPathProto hyperspherical manifold", pad=12, title_size=12.6)
    ax_b.set_xlabel("t-SNE 1", fontsize=11.0, fontweight="bold")
    ax_b.set_ylabel("t-SNE 2", fontsize=11.0, fontweight="bold")
    ax_b.tick_params(axis="both", labelsize=9.6)
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    ax_b.legend_.remove()
    sns.despine(ax=ax_b)

    # Panel c: scPathProto manifold colored by sequencing protocol
    ax_c = fig.add_subplot(gs[0, 2])
    batches_sorted = sorted(df_man["Batch"].unique())
    prot_color = {b: BATCH_PROTOCOL_PALETTE[i % len(BATCH_PROTOCOL_PALETTE)] for i, b in enumerate(batches_sorted)}
    sns.scatterplot(
        data=df_man, x="scPathProto_TSNE_1", y="scPathProto_TSNE_2", hue="Batch",
        hue_order=batches_sorted, palette=prot_color, s=18, alpha=0.80, linewidth=0, ax=ax_c
    )
    set_nc_panel_title(ax_c, "c", "Cross-protocol latent batch mixing", pad=12, title_size=12.6)
    ax_c.set_xlabel("t-SNE 1", fontsize=11.0, fontweight="bold")
    ax_c.set_ylabel("t-SNE 2", fontsize=11.0, fontweight="bold")
    ax_c.tick_params(axis="both", labelsize=9.6)
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    prot_leg_map = {"10Xv2": "10x v2", "10Xv3": "10x v3", "Drop-Seq": "Drop-seq", "Smart-Seq2": "Smart-seq2"}
    labels_c = [prot_leg_map.get(lb, lb) for lb in labels_c]
    ax_c.legend_.remove()
    sns.despine(ax=ax_c)

    leg_ab = fig.legend(
        handles_b, labels_b, loc="lower center", bbox_to_anchor=(0.33, -0.14),
        ncol=5, fontsize=9.2, title="Cell Type & Unit Prototype", title_fontsize=9.6,
        frameon=True, facecolor="#FFFFFF", edgecolor="#D8E1EB", borderpad=0.50
    )
    leg_ab.get_title().set_fontweight("bold")
    leg_ab.get_frame().set_linewidth(0.8)
    for h in leg_ab.legend_handles[:-1]:
        if hasattr(h, "set_sizes"):
            h.set_sizes([56])
        elif hasattr(h, "set_markersize"):
            h.set_markersize(7.4)

    leg_c = fig.legend(
        handles_c, labels_c, loc="lower center", bbox_to_anchor=(0.835, -0.14),
        ncol=4, fontsize=9.2, title="Sequencing Protocol (7 Platforms)", title_fontsize=9.6,
        frameon=True, facecolor="#FFFFFF", edgecolor="#D8E1EB", borderpad=0.50
    )
    leg_c.get_title().set_fontweight("bold")
    leg_c.get_frame().set_linewidth(0.8)
    for h in leg_c.legend_handles:
        if hasattr(h, "set_sizes"):
            h.set_sizes([56])
        elif hasattr(h, "set_markersize"):
            h.set_markersize(7.4)

    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


def plot_fig9_uncertainty_and_convergence(tables_dir: str, out_pdf: str) -> None:
    """
    Figure 10 in manuscript (`fig9_uncertainty_openset_analysis.pdf`):
    a  Open-set novelty score distribution + validation 95% threshold
    b  Prototype similarity vs. Dirichlet epistemic vacuity
    """
    df_unc = pd.read_csv(os.path.join(tables_dir, "interpret_uncertainty_distribution.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2), dpi=300)
    grp_order = [
        "Correctly Annotated Known Cell",
        "Misclassified / Boundary Cell",
        "Unseen Novel Cell Type",
    ]
    grp_palette = {
        "Correctly Annotated Known Cell": PAL["teal_mid"],
        "Misclassified / Boundary Cell": PAL["amber_mid"],
        "Unseen Novel Cell Type": PAL["rose_dark"],
    }

    counts_map = df_unc["Group"].value_counts().to_dict()
    short_xtick = [
        f"Correct Known\n(n={counts_map.get('Correctly Annotated Known Cell', 0)})",
        f"Misclassified / Boundary\n(n={counts_map.get('Misclassified / Boundary Cell', 0)})",
        f"Unseen Novel Type\n(n={counts_map.get('Unseen Novel Cell Type', 0)})",
    ]
    tau_open = float(df_unc["Rejection_Threshold"].iloc[0]) if "Rejection_Threshold" in df_unc.columns else 0.16

    # Panel a: Violin + Boxplot + Validation-Calibrated Rejection Threshold
    axes[0].set_axisbelow(True)
    sns.violinplot(
        data=df_unc, x="Group", y="OpenSet_Novelty_Score", order=grp_order, hue="Group",
        hue_order=grp_order, palette=grp_palette, inner="box", cut=0, linewidth=0.95,
        saturation=1.0, alpha=0.88, zorder=3, legend=False,
        inner_kws=dict(color=PAL["ink_dark"], alpha=1.0, zorder=5),
        ax=axes[0]
    )
    axes[0].axhline(
        tau_open, color=PAL["rose_dark"], linestyle="--", linewidth=1.65, zorder=6,
        label=f"Val 95%-Quantile Cutoff ($\\tau_{{\\mathrm{{open}}}}={tau_open:.2f}$)"
    )
    set_nc_panel_title(axes[0], "a", "Open-set novelty score distribution", pad=12)
    axes[0].set_xlabel("")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(short_xtick, fontsize=9.6, fontweight="bold")
    axes[0].set_ylabel("Open-Set Novelty Score ($s_i^{\\mathrm{nov}}$)", fontsize=11.0, fontweight="bold")
    axes[0].tick_params(axis="y", labelsize=9.6)
    leg_0 = axes[0].legend(loc="upper left", fontsize=9.2, borderpad=0.45)
    style_legend(leg_0, border_color=PAL["rose_edge"], facecolor=PAL["rose_bg"])
    axes[0].grid(axis="y")
    sns.despine(ax=axes[0])

    # Panel b: 2D Prototype Cosine vs. Epistemic Vacuity Landscape
    df_unc_leg = df_unc.copy()
    leg_map = {g: f"{g} (n={counts_map.get(g, 0)})" for g in grp_order}
    df_unc_leg["Group_N"] = df_unc_leg["Group"].map(leg_map)
    leg_order = [leg_map[g] for g in grp_order]
    leg_pal = {leg_map[g]: grp_palette[g] for g in grp_order}
    sns.scatterplot(
        data=df_unc_leg, x="Max_Prototype_Cosine", y="Dirichlet_Vacuity", hue="Group_N",
        hue_order=leg_order, palette=leg_pal, s=30, alpha=0.84, edgecolor="#FFFFFF", linewidth=0.35, ax=axes[1]
    )
    set_nc_panel_title(axes[1], "b", "Prototype similarity vs. vacuity", pad=12)
    axes[1].set_xlabel("Max Prototype Cosine Similarity", fontsize=11.0, fontweight="bold", labelpad=5)
    axes[1].set_ylabel("Dirichlet Epistemic Vacuity ($u_i = C/S_i$)", fontsize=11.0, fontweight="bold")
    axes[1].tick_params(axis="both", labelsize=9.6)
    leg_1 = axes[1].legend(title="Cell Category (Sample Size)", loc="upper right", fontsize=8.8, title_fontsize=9.2, borderpad=0.45)
    leg_1.get_title().set_fontweight("bold")
    style_legend(leg_1)
    axes[1].grid(True)
    sns.despine(ax=axes[1])

    plt.tight_layout(w_pad=3.2)
    save_pdf_and_png(fig, out_pdf)
    plt.close(fig)


# Batch figure generation
def generate_all_paper_figures(
    tables_dir: str = "results/tables",
    out_figs_dir: str = "figures",
    paper_figs_dir: str = "figures",
) -> List[str]:
    """Render figures `fig1`..`fig9` from CSV tables in `tables_dir`."""
    ws_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sync_dirs = list(dict.fromkeys([
        os.path.abspath(paper_figs_dir),
        os.path.join(ws_root, "figures"),
        os.path.join(ws_root, "results", "figures"),
    ]))
    out_figs_abs = os.path.abspath(out_figs_dir)
    if out_figs_abs not in sync_dirs:
        sync_dirs.append(out_figs_abs)
    for d in sync_dirs:
        os.makedirs(d, exist_ok=True)

    figure_jobs = [
        ("fig1_intra_dataset_performance.pdf", plot_fig1_intra_dataset),
        ("fig2_cross_batch_transfer.pdf", plot_fig2_cross_batch),
        ("fig3_rare_celltype_analysis.pdf", plot_fig3_rare_celltypes),
        ("fig4_openset_roc_prc.pdf", plot_fig4_openset),
        ("fig5_ablation_and_sensitivity.pdf", plot_fig5_ablation_and_sensitivity),
        ("fig6_interpret_pathway_heatmap.pdf", plot_fig6_pathway_heatmap),
        ("fig7_interpret_marker_dotplot.pdf", plot_fig7_marker_dotplot),
        ("fig8_latent_manifold_and_confusion.pdf", plot_fig8_manifold_and_confusion),
        ("fig9_uncertainty_openset_analysis.pdf", plot_fig9_uncertainty_and_convergence),
    ]

    generated_paths: List[str] = []
    for fname, plot_fn in figure_jobs:
        out_path = os.path.join(out_figs_abs, fname)
        plot_fn(tables_dir, out_path)
        for target_dir in sync_dirs:
            target_path = os.path.join(target_dir, fname)
            if os.path.abspath(out_path) != os.path.abspath(target_path):
                shutil.copy2(out_path, target_path)
        generated_paths.append(out_path)
        print(f"  [OK] Saved {fname}")

    return generated_paths


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_tables = os.path.join(ws_root, "results", "tables")
    default_figs = os.path.join(ws_root, "figures")
    default_paper_figs = os.path.join(ws_root, "figures")

    parser = argparse.ArgumentParser(description="Render benchmark and interpretability figures (fig1-fig9).")
    parser.add_argument("--tables_dir", type=str, default=default_tables)
    parser.add_argument("--out_figs_dir", type=str, default=default_figs)
    parser.add_argument("--paper_figs_dir", type=str, default=default_paper_figs)
    args = parser.parse_args()

    generate_all_paper_figures(
        tables_dir=args.tables_dir,
        out_figs_dir=args.out_figs_dir,
        paper_figs_dir=args.paper_figs_dir,
    )

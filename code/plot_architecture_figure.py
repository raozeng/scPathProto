#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render the scPathProto model architecture schematic (`fig0_model_architecture.pdf/.png`)
and graphical abstract (`graphical_abstract.pdf/.png`).
"""

import csv
import os
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
    Wedge,
)
from matplotlib.path import Path

# Global font and vector output settings
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "stixsans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Color palette
PAL = {
    # Navy-blue (panel a)
    "navy_dark": "#2E4D72",
    "navy_mid": "#4E769E",
    "navy_cerulean": "#5186A8",
    "navy_sub": "#DFECF3",
    "navy_bg": "#F4F8FC",
    "navy_edge": "#B7CFE2",
    # Teal (panel b)
    "teal_dark": "#2D6B64",
    "teal_mid": "#43877E",
    "teal_sage": "#76A89D",
    "teal_sub": "#D6ECE8",
    "teal_bg": "#F2FAF8",
    "teal_edge": "#A9D3CB",
    # Purple (panel c)
    "iris_dark": "#5E468A",
    "iris_mid": "#7862A8",
    "iris_soft": "#B1A3D4",
    "iris_sub": "#E5E0F5",
    "iris_bg": "#F6F4FC",
    "iris_edge": "#C5B9E0",
    # Amber (panel d)
    "amber_dark": "#9D6833",
    "amber_mid": "#C28B4E",
    "amber_soft": "#DFBA88",
    "amber_sub": "#F4EBD7",
    "amber_bg": "#FAF5EA",
    "amber_edge": "#DEC59D",
    # Burgundy-rose (panel e)
    "rose_dark": "#914153",
    "rose_mid": "#BC6E80",
    "rose_soft": "#DDAAB7",
    "rose_sub": "#F2E2E6",
    "rose_bg": "#FAF3F5",
    "rose_edge": "#DFB6C0",
    # Panel f and neutrals
    "out_bg": "#F4F9F6",
    "out_edge": "#B5D3C4",
    "ink_dark": "#2B3542",
    "ink_mid": "#526071",
    "slate_muted": "#9AA6B5",
    "slate_light": "#E8EEF4",
}

REF_CONTACT_CMAP = LinearSegmentedColormap.from_list(
    "ref_contact_teal_navy",
    ["#F5F9EE", "#DBE9D8", "#8BBCC3", "#5186A8", "#2E4D72"],
    N=256,
)


# Shared drawing primitives
def add_box(
    ax,
    x,
    y,
    w,
    h,
    fc="#FFFFFF",
    ec="#94A3B8",
    lw=1.1,
    r=0.10,
    alpha=1.0,
    ls="-",
    zorder=2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_banner(
    ax,
    x,
    y,
    w,
    h,
    title,
    badge="",
    fc="#DCE7F3",
    ec="#9BB4CC",
    tc="#243B53",
    bc="#335C81",
    fontsize=11.0,
    zorder=4,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.09",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.1,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if badge:
        ax.text(
            x + 0.10,
            y + h * 0.5,
            badge,
            fontsize=fontsize + 0.4,
            fontweight="bold",
            color=bc,
            va="center",
            ha="left",
            zorder=zorder + 1,
        )
        ax.text(
            x + 0.48,
            y + h * 0.5,
            title,
            fontsize=fontsize,
            fontweight="bold",
            color=tc,
            va="center",
            ha="left",
            zorder=zorder + 1,
        )
    else:
        ax.text(
            x + w * 0.5,
            y + h * 0.5,
            title,
            fontsize=fontsize,
            fontweight="bold",
            color=tc,
            va="center",
            ha="center",
            zorder=zorder + 1,
        )


def _draw_arrowhead_polygon(
    ax, x_tip, y_tip, ux, uy, head_len=0.11, head_width=0.095, notch=0.022, color="#3B5B78", halo=True, zorder=11
):
    px, py = -uy, ux
    bx = x_tip - head_len * ux
    by = y_tip - head_len * uy
    nx = x_tip - (head_len - notch) * ux
    ny = y_tip - (head_len - notch) * uy

    left_x = bx + 0.5 * head_width * px
    left_y = by + 0.5 * head_width * py
    right_x = bx - 0.5 * head_width * px
    right_y = by - 0.5 * head_width * py

    verts = np.array([[x_tip, y_tip], [left_x, left_y], [nx, ny], [right_x, right_y]])
    if halo:
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor="#FFFFFF",
                edgecolor="#FFFFFF",
                linewidth=1.5,
                joinstyle="round",
                alpha=0.90,
                zorder=zorder - 1,
            )
        )
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
            joinstyle="miter",
            zorder=zorder,
        )
    )


def add_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color="#3B5B78",
    lw=2.0,
    head_len=0.105,
    head_width=0.092,
    ls="-",
    halo=True,
    zorder=10,
):
    dx, dy = x2 - x1, y2 - y1
    dist = float(np.hypot(dx, dy))
    if dist < 1e-5:
        return
    ux, uy = dx / dist, dy / dist
    hl = min(head_len, dist * 0.52)
    hw = head_width * (hl / head_len)
    notch = 0.20 * hl

    x_shaft_end = x2 - (hl - notch * 0.5) * ux
    y_shaft_end = y2 - (hl - notch * 0.5) * uy

    if halo:
        ax.plot(
            [x1, x_shaft_end],
            [y1, y_shaft_end],
            color="#FFFFFF",
            lw=lw + 2.0,
            solid_capstyle="butt",
            alpha=0.90,
            zorder=zorder - 1,
        )
    ax.plot(
        [x1, x_shaft_end],
        [y1, y_shaft_end],
        color=color,
        lw=lw,
        linestyle=ls,
        solid_capstyle="butt",
        zorder=zorder,
    )
    _draw_arrowhead_polygon(
        ax, x2, y2, ux, uy, head_len=hl, head_width=hw, notch=notch, color=color, halo=halo, zorder=zorder + 1
    )


def add_ortho_arrow(
    ax,
    points,
    color="#3B5B78",
    lw=2.0,
    head_len=0.105,
    head_width=0.092,
    ls="-",
    halo=True,
    zorder=10,
):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return
    x_prev, y_prev = pts[-2]
    x_end, y_end = pts[-1]
    dx, dy = x_end - x_prev, y_end - y_prev
    seg_len = float(np.hypot(dx, dy))
    if seg_len < 1e-5:
        return
    ux, uy = dx / seg_len, dy / seg_len
    hl = min(head_len, seg_len * 0.58)
    hw = head_width * (hl / head_len)
    notch = 0.20 * hl

    shaft_end = (x_end - (hl - notch * 0.5) * ux, y_end - (hl - notch * 0.5) * uy)
    shaft_pts = np.vstack([pts[:-1], shaft_end])

    if halo:
        ax.plot(
            shaft_pts[:, 0],
            shaft_pts[:, 1],
            color="#FFFFFF",
            lw=lw + 2.0,
            solid_capstyle="butt",
            solid_joinstyle="round",
            alpha=0.90,
            zorder=zorder - 1,
        )
    ax.plot(
        shaft_pts[:, 0],
        shaft_pts[:, 1],
        color=color,
        lw=lw,
        linestyle=ls,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=zorder,
    )
    _draw_arrowhead_polygon(
        ax, x_end, y_end, ux, uy, head_len=hl, head_width=hw, notch=notch, color=color, halo=halo, zorder=zorder + 1
    )


# Vector panel drawing helpers
def draw_nc_panel(ax, x, y, w, h, letter, title, bg, edge, title_color):
    """Draw a stage panel container with bold lowercase panel label."""
    add_box(ax, x, y, w, h, fc=bg, ec=edge, lw=1.05, r=0.14, zorder=1)
    # Panel letter (a-f)
    ax.text(
        x + 0.14,
        y + h - 0.28,
        letter,
        fontsize=20.5,
        fontweight="bold",
        color=PAL["ink_dark"],
        ha="left",
        va="center",
        zorder=5,
    )
    # Stage title beside panel letter
    ax.text(
        x + 0.49,
        y + h - 0.28,
        title,
        fontsize=14.2,
        fontweight="bold",
        color=title_color,
        ha="left",
        va="center",
        zorder=5,
    )
    # Header divider line
    ax.plot(
        [x + 0.14, x + w - 0.14],
        [y + h - 0.54, y + h - 0.54],
        color=edge,
        lw=1.0,
        alpha=0.85,
        zorder=4,
    )


def draw_bezier_ribbon(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.32, zorder=4):
    """Draw a cubic Bezier Sankey ribbon."""
    xm = 0.5 * (x0 + x1)
    verts = [
        (x0, y0 + 0.5 * h0),
        (xm, y0 + 0.5 * h0),
        (xm, y1 + 0.5 * h1),
        (x1, y1 + 0.5 * h1),
        (x1, y1 - 0.5 * h1),
        (xm, y1 - 0.5 * h1),
        (xm, y0 - 0.5 * h0),
        (x0, y0 - 0.5 * h0),
        (x0, y0 + 0.5 * h0),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            Path(verts, codes),
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=alpha,
            zorder=zorder,
        )
    )


def draw_borderless_expr_matrix(ax, x0, y0, w, h, mat, obs_color, imp_color, show_zeros=True, zorder=4):
    """Draw a 6x5 expression matrix (observed=blue, recovered=amber, zero=gray)."""
    rows, cols = mat.shape
    cw = w / cols
    ch = h / rows
    gap_x = cw * 0.12
    gap_y = ch * 0.12
    for r in range(rows):
        for c in range(cols):
            val = float(mat[r, c])
            rx = x0 + c * cw + 0.5 * gap_x
            ry = y0 + (rows - 1 - r) * ch + 0.5 * gap_y
            tw = cw - gap_x
            th = ch - gap_y
            if val == 0.0:
                ax.add_patch(
                    Rectangle((rx, ry), tw, th, facecolor="#E8EEF4", edgecolor="none", linewidth=0.0, zorder=zorder)
                )
                if show_zeros:
                    ax.text(
                        rx + 0.5 * tw,
                        ry + 0.5 * th,
                        "0",
                        fontsize=8.8,
                        color="#94A3B8",
                        ha="center",
                        va="center",
                        zorder=zorder + 1,
                    )
            elif val < 0.0:
                ax.add_patch(
                    Rectangle(
                        (rx, ry), tw, th, facecolor=imp_color, edgecolor="none", linewidth=0.0, alpha=0.88, zorder=zorder
                    )
                )
                ax.text(
                    rx + 0.5 * tw,
                    ry + 0.5 * th,
                    "+",
                    fontsize=9.8,
                    fontweight="bold",
                    color="#FFFFFF",
                    ha="center",
                    va="center",
                    zorder=zorder + 1,
                )
            else:
                alpha_v = 0.36 + 0.58 * min(1.0, val)
                ax.add_patch(
                    Rectangle(
                        (rx, ry), tw, th, facecolor=obs_color, edgecolor="none", linewidth=0.0, alpha=alpha_v, zorder=zorder
                    )
                )


def draw_check_badge(ax, cx, cy, r=0.11, fc="#2D6B64", zorder=7):
    ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor="none", zorder=zorder))
    ax.plot(
        [cx - 0.42 * r, cx - 0.10 * r, cx + 0.48 * r],
        [cy - 0.02 * r, cy - 0.34 * r, cy + 0.34 * r],
        color="#FFFFFF",
        lw=1.7,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder + 1,
    )


def draw_stop_badge(ax, cx, cy, r=0.10, fc="#9AA6B5", zorder=7):
    ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor="none", zorder=zorder))
    d = 0.38 * r
    ax.plot([cx - d, cx + d], [cy - d, cy + d], color="#FFFFFF", lw=1.6, solid_capstyle="round", zorder=zorder + 1)
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color="#FFFFFF", lw=1.6, solid_capstyle="round", zorder=zorder + 1)


def draw_gate_x_marker(ax, cx, cy, r=0.085, color="#914153", zorder=8):
    ax.add_patch(Circle((cx, cy), r, facecolor="#FFFFFF", edgecolor=color, lw=1.0, zorder=zorder))
    d = 0.42 * r
    ax.plot([cx - d, cx + d], [cy - d, cy + d], color=color, lw=1.4, solid_capstyle="round", zorder=zorder + 1)
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color=color, lw=1.4, solid_capstyle="round", zorder=zorder + 1)


# Architecture figure (`fig0_model_architecture`)
def generate_architecture_figure(
    out_pdf_paper: str = "figures/fig0_model_architecture.pdf",
    out_png_paper: str = "figures/fig0_model_architecture.png",
    out_pdf_results: str = "results/figures/fig0_model_architecture.pdf",
    out_png_results: str = "results/figures/fig0_model_architecture.png",
):
    """Render the 6-panel architecture overview figure (16.6 x 9.6 in)."""
    W, H = 16.6, 9.6
    fig = plt.figure(figsize=(W, H), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0.0, W)
    ax.set_ylim(0.0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel grid layout (2 rows x 3 columns)

    # Panel a: Multi-Platform ZIRG Denoising
    draw_nc_panel(
        ax, 0.12, 5.18, 5.22, 4.30,
        "a", "Multi-Platform ZIRG Denoising",
        PAL["navy_bg"], PAL["navy_edge"], PAL["navy_dark"],
    )

    # Left Graphic: 3 Stylized Multi-Platform Cohort Icons (y: 8.15 -> 8.84)
    cohort_specs = [
        (0.45, 8.62, "10x", PAL["navy_mid"], "cluster"),
        (0.98, 8.62, "SS2", PAL["teal_mid"], "plate"),
        (1.51, 8.62, "BALF", PAL["rose_mid"], "balf"),
    ]
    for cx_c, cy_c, clbl, ccol, ckind in cohort_specs:
        ax.add_patch(Circle((cx_c, cy_c), 0.18, facecolor="#FFFFFF", edgecolor=ccol, lw=1.0, zorder=3))
        if ckind == "cluster":
            pts = [(-0.062, 0.042), (0.052, 0.052), (-0.032, -0.052), (0.062, -0.042), (0.0, 0.01)]
            for dx, dy in pts:
                ax.add_patch(Circle((cx_c + dx, cy_c + dy), 0.036, facecolor=ccol, edgecolor="none", alpha=0.85, zorder=4))
        elif ckind == "plate":
            for gx in [-0.064, 0.0, 0.064]:
                for gy in [-0.054, 0.054]:
                    ax.add_patch(Circle((cx_c + gx, cy_c + gy), 0.025, facecolor=ccol, edgecolor="none", alpha=0.88, zorder=4))
        else:
            pts = [(-0.058, 0.036), (0.052, 0.058), (0.01, -0.052), (-0.052, -0.042)]
            cols_b = [PAL["rose_mid"], PAL["amber_mid"], PAL["rose_dark"], PAL["teal_mid"]]
            for (dx, dy), cb in zip(pts, cols_b):
                ax.add_patch(Circle((cx_c + dx, cy_c + dy), 0.038, facecolor=cb, edgecolor="none", alpha=0.85, zorder=4))
        ax.text(cx_c, cy_c - 0.24, clbl, fontsize=9.8, fontweight="bold", color=PAL["ink_mid"], ha="center", va="top")

    # Small downward arrow from cohorts to Sparse Matrix
    add_arrow(ax, 0.98, 8.05, 0.98, 7.78, color=PAL["navy_dark"], lw=1.8, head_len=0.085, head_width=0.075, halo=False)

    # 6x5 Sparse Count Matrix Heatmap (x: 0.28 -> 1.68, y: 5.88 -> 7.70)
    raw_mat = np.array(
        [
            [0.92, 0.00, 0.72, 0.00, 0.55],
            [0.00, 0.85, 0.00, 0.64, 0.00],
            [0.68, 0.00, 0.90, 0.00, 0.78],
            [0.00, 0.58, 0.00, 0.82, 0.00],
            [0.80, 0.00, 0.00, 0.60, 0.88],
            [0.00, 0.76, 0.66, 0.00, 0.00],
        ]
    )
    add_box(ax, 0.26, 5.84, 1.44, 1.88, fc="#FFFFFF", ec=PAL["navy_edge"], lw=0.85, r=0.07, zorder=2)
    draw_borderless_expr_matrix(ax, 0.32, 5.90, 1.32, 1.76, raw_mat, PAL["navy_mid"], PAL["amber_mid"], show_zeros=True, zorder=4)
    ax.text(1.01, 5.60, "Log-Norm scRNA-seq $\\mathbf{X}$", fontsize=9.4, fontweight="bold", color=PAL["navy_dark"], ha="center", va="center")
    ax.text(0.98, 5.34, "80%–96% zeros", fontsize=9.6, color=PAL["ink_mid"], ha="center", va="center")

    # Arrow: Sparse Matrix -> Pathway Co-Expression Gate
    add_arrow(ax, 1.73, 6.80, 1.96, 6.80, color=PAL["navy_dark"], lw=2.0, head_len=0.095, head_width=0.085)

    # Middle Graphic: Pathway Co-Expression Gate (x: 1.98 -> 3.74)
    ax.text(2.86, 8.70, "Co-Expression Gate", fontsize=11.8, fontweight="bold", color=PAL["navy_dark"], ha="center", va="center")
    # Upper sub-illustration: Co-expressed dropout -> Recover
    ax.add_patch(
        Ellipse((2.72, 7.90), 1.30, 1.02, facecolor=PAL["navy_sub"], edgecolor=PAL["navy_edge"], lw=0.9, ls="--", alpha=0.80, zorder=2)
    )
    drop_pt = (2.72, 7.90)
    active_pts = [(2.32, 8.14), (2.40, 7.64), (3.08, 8.14), (3.08, 7.66)]
    for ax_p, ay_p in active_pts:
        ax.plot([ax_p, drop_pt[0]], [ay_p, drop_pt[1]], color=PAL["navy_mid"], lw=1.5, alpha=0.85, zorder=4)
        ax.add_patch(Circle((ax_p, ay_p), 0.088, facecolor=PAL["navy_mid"], edgecolor="#FFFFFF", lw=0.8, zorder=5))
    ax.add_patch(Circle(drop_pt, 0.125, facecolor=PAL["amber_mid"], edgecolor="#FFFFFF", lw=1.0, zorder=6))
    ax.text(drop_pt[0], drop_pt[1], "?", fontsize=10.8, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=7)
    draw_check_badge(ax, 3.52, 7.90, r=0.125, fc=PAL["teal_dark"], zorder=8)
    ax.text(2.86, 7.18, "Dropout → Recover", fontsize=10.0, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center")
    ax.text(2.86, 6.94, "Open gate", fontsize=9.5, color=PAL["ink_mid"], ha="center", va="center")

    # Lower sub-illustration: Isolated silent gene -> Keep zero
    ax.add_patch(
        Ellipse((2.72, 6.26), 1.24, 0.72, facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.85, ls=":", alpha=0.85, zorder=2)
    )
    iso_pt = (2.72, 6.26)
    silent_pts = [(2.32, 6.42), (2.40, 6.08), (3.10, 6.40)]
    for sx_p, sy_p in silent_pts:
        ax.plot([sx_p, iso_pt[0]], [sy_p, iso_pt[1]], color="#CBD5E1", lw=1.15, ls=":", zorder=4)
        ax.add_patch(Circle((sx_p, sy_p), 0.072, facecolor="#DCE4EC", edgecolor="#CBD5E1", lw=0.7, zorder=5))
    ax.add_patch(Circle(iso_pt, 0.105, facecolor="#CBD5E1", edgecolor="#94A3B8", lw=0.8, zorder=6))
    ax.text(iso_pt[0], iso_pt[1], "0", fontsize=9.6, fontweight="bold", color="#64748B", ha="center", va="center", zorder=7)
    draw_stop_badge(ax, 3.52, 6.26, r=0.115, fc=PAL["slate_muted"], zorder=8)
    ax.text(2.86, 5.62, "True silence → Keep 0", fontsize=9.8, fontweight="bold", color=PAL["ink_mid"], ha="center", va="center")
    ax.text(2.86, 5.36, "Closed gate", fontsize=9.5, color="#64748B", ha="center", va="center")

    # Arrow: Gate -> Denoised Matrix
    add_arrow(ax, 3.72, 6.80, 3.94, 6.80, color=PAL["navy_dark"], lw=2.0, head_len=0.095, head_width=0.085)

    # Right Graphic: Selectively Denoised Matrix (x: 3.96 -> 5.22)
    denoised_mat = raw_mat.copy()
    denoised_mat[0, 1] = -1.0
    denoised_mat[2, 3] = -1.0
    denoised_mat[4, 2] = -1.0
    add_box(ax, 3.96, 5.84, 1.24, 1.88, fc="#FFFFFF", ec=PAL["amber_edge"], lw=0.95, r=0.07, zorder=2)
    draw_borderless_expr_matrix(ax, 4.01, 5.90, 1.14, 1.76, denoised_mat, PAL["navy_mid"], PAL["amber_mid"], show_zeros=True, zorder=4)

    # Color legend above denoised matrix
    leg_items_a = [
        (8.66, PAL["navy_mid"], "Observed"),
        (8.34, PAL["amber_mid"], "Recovered"),
        (8.02, "#DDE5EE", "True zero"),
    ]
    for ly, lcol, llbl in leg_items_a:
        ax.add_patch(Rectangle((3.96, ly - 0.065), 0.18, 0.13, facecolor=lcol, edgecolor="none", linewidth=0.0, zorder=5))
        ax.text(4.20, ly, llbl, fontsize=9.8, fontweight="bold", color=PAL["ink_mid"], ha="left", va="center")

    ax.text(4.58, 5.60, "Denoised", fontsize=10.6, fontweight="bold", color=PAL["amber_dark"], ha="center", va="center")
    ax.text(4.58, 5.34, "Non-zeros kept", fontsize=9.6, color=PAL["ink_mid"], ha="center", va="center")

    # Connector a -> b (x: 5.34 -> 5.70)
    add_arrow(ax, 5.36, 7.26, 5.68, 7.26, color=PAL["navy_dark"], lw=2.4, head_len=0.12, head_width=0.105)

    # Panel b: Hybrid curated and de novo MPCA
    draw_nc_panel(
        ax, 5.70, 5.18, 5.22, 4.30,
        "b", "Hybrid Curated + De Novo MPCA",
        PAL["teal_bg"], PAL["teal_edge"], PAL["teal_dark"],
    )

    # Left: hybrid pathway mask heatmap and ANOVA weight bars
    ax.text(6.74, 8.70, "Hybrid Mask + F-Wt.", fontsize=11.6, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center")
    cur_mask = np.array(
        [
            [1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=float,
    )
    f_weights = [0.92, 0.74, 0.86, 0.62, 0.96, 0.78]
    mx0, my0, mw, mh = 5.86, 7.12, 1.14, 1.24
    cw_m = mw / 6.0
    ch_m = mh / 6.0
    for r in range(6):
        ry = my0 + (5 - r) * ch_m
        for c in range(6):
            rx = mx0 + c * cw_m
            fc_m = PAL["teal_mid"] if cur_mask[r, c] > 0 else "#E8F2EF"
            ax.add_patch(
                Rectangle((rx + 0.012, ry + 0.015), cw_m - 0.024, ch_m - 0.03, facecolor=fc_m, edgecolor="none", linewidth=0.0, zorder=4)
            )
        # Horizontal ANOVA F-weight bar beside each row
        bw = 0.46 * f_weights[r]
        ax.add_patch(
            Rectangle((mx0 + mw + 0.06, ry + 0.036), bw, ch_m - 0.072, facecolor=PAL["teal_dark"], edgecolor="none", linewidth=0.0, alpha=0.82, zorder=4)
        )
    ax.text(6.74, 6.90, "Curated Priors (≤300)", fontsize=9.8, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center")

    # Dashed divider between Curated & De Novo blocks
    ax.plot([5.84, 7.62], [6.70, 6.70], color=PAL["amber_mid"], lw=1.0, ls="--", zorder=4)

    # Bottom block: 8 de novo regulon slots
    denovo_y0, denovo_h = 6.12, 0.48
    ch_d = denovo_h / 2.0
    dn_weights = [0.88, 0.76]
    for r in range(2):
        ry = denovo_y0 + (1 - r) * ch_d
        for c in range(6):
            rx = mx0 + c * cw_m
            alpha_d = 0.48 + 0.44 * ((r + c) % 3 == 0 or (c % 2 == r))
            ax.add_patch(
                Rectangle((rx + 0.012, ry + 0.02), cw_m - 0.024, ch_d - 0.04, facecolor=PAL["amber_mid"], edgecolor="none", linewidth=0.0, alpha=alpha_d, zorder=4)
            )
        ax.add_patch(
            Rectangle((mx0 + mw + 0.06, ry + 0.045), 0.46 * dn_weights[r], ch_d - 0.09, facecolor=PAL["amber_dark"], edgecolor="none", linewidth=0.0, alpha=0.86, zorder=4)
        )
    ax.text(6.74, 5.86, "8 De Novo Slots", fontsize=10.2, fontweight="bold", color=PAL["amber_dark"], ha="center", va="center")
    ax.text(6.74, 5.58, "IL3RA · NKG7 · TUBB1", fontsize=9.6, fontweight="bold", color=PAL["ink_mid"], ha="center", va="center")
    ax.text(6.74, 5.34, "0/80 in 300-set mask", fontsize=9.5, color="#64748B", ha="center", va="center")

    # Arrow: Hybrid Mask -> Inter-Pathway Covariance Mini-Heatmap
    add_arrow(ax, 7.62, 7.26, 7.86, 7.26, color=PAL["teal_dark"], lw=2.0, head_len=0.095, head_width=0.085)

    # Middle Graphic: 5x5 Empirical Inter-Pathway Covariance Mini-Heatmap (x: 7.90 -> 9.26)
    ax.text(8.58, 8.54, "Pathway\nCovariance", fontsize=11.5, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center", linespacing=1.12)
    cov_mat = np.array(
        [
            [1.00, 0.78, 0.22, 0.14, 0.08],
            [0.78, 1.00, 0.28, 0.16, 0.10],
            [0.22, 0.28, 1.00, 0.72, 0.60],
            [0.14, 0.16, 0.72, 1.00, 0.80],
            [0.08, 0.10, 0.60, 0.80, 1.00],
        ]
    )
    add_box(ax, 7.94, 6.72, 1.28, 1.28, fc="#FFFFFF", ec=PAL["teal_edge"], lw=0.85, r=0.07, zorder=2)
    hx0, hy0, hw, hh = 8.00, 6.78, 1.16, 1.16
    tile = hw / 5.0
    for r in range(5):
        for c in range(5):
            rgba = REF_CONTACT_CMAP(cov_mat[r, c])
            ax.add_patch(
                Rectangle(
                    (hx0 + c * tile + 0.012, hy0 + (4 - r) * tile + 0.012),
                    tile - 0.024,
                    tile - 0.024,
                    facecolor=rgba,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=4,
                )
            )
    ax.text(8.58, 6.38, "Co-Activation", fontsize=10.2, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center")
    ax.text(8.58, 6.10, "Smoothing", fontsize=9.6, color=PAL["ink_mid"], ha="center", va="center")
    add_box(ax, 7.66, 5.29, 1.88, 0.54, fc=PAL["teal_sub"], ec="none", lw=0.0, r=0.09, zorder=3)
    ax.text(8.60, 5.56, "8.7× Bio/Batch SNR\n$P$ ≤ 308 vs. $G$ = 3,000", fontsize=9.4, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center", linespacing=1.16, zorder=4)

    # Right Graphic: Dual-Branch Spectral Projection Funnel (x: 9.30 -> 10.80)
    draw_bezier_ribbon(ax, 9.30, 7.72, 0.42, 10.36, 7.46, 0.56, PAL["teal_mid"], alpha=0.45, zorder=3)
    draw_bezier_ribbon(ax, 9.30, 6.66, 0.42, 10.36, 6.90, 0.56, PAL["navy_mid"], alpha=0.42, zorder=3)
    ax.text(9.84, 8.20, "Pathway\nSVD", fontsize=9.8, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center", linespacing=1.06)
    ax.text(9.82, 6.20, "HVG SVD", fontsize=9.8, fontweight="bold", color=PAL["navy_dark"], ha="center", va="center")

    # Cell embedding vector bar
    seg_cols = [PAL["teal_dark"], PAL["teal_mid"], PAL["teal_sage"], PAL["navy_cerulean"], PAL["navy_mid"], PAL["iris_mid"], PAL["amber_mid"], PAL["navy_dark"]]
    bar_x, bar_y0, bar_w, bar_h = 10.36, 6.46, 0.32, 1.64
    sh = bar_h / len(seg_cols)
    for si, sc in enumerate(seg_cols):
        ax.add_patch(
            Rectangle((bar_x, bar_y0 + si * sh + 0.01), bar_w, sh - 0.02, facecolor=sc, edgecolor="none", linewidth=0.0, alpha=0.90, zorder=5)
        )
    ax.text(10.36, 5.76, "Pathway\nSubspace", fontsize=10.2, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center", linespacing=1.12)

    # Connector b -> c (x: 10.92 -> 11.28)
    add_arrow(ax, 10.94, 7.26, 11.26, 7.26, color=PAL["teal_dark"], lw=2.4, head_len=0.12, head_width=0.105)

    # Panel c: Proto-Align and rare-boundary graph
    draw_nc_panel(
        ax, 11.28, 5.18, 5.20, 4.30,
        "c", "Proto-Align & Rare-Boundary Graph",
        PAL["iris_bg"], PAL["iris_edge"], PAL["iris_dark"],
    )

    # Left Half (x: 11.40 -> 13.78): Composition-Decoupled Moment Alignment (Before -> After)
    ax.text(12.58, 8.70, "Proto-Align", fontsize=11.8, fontweight="bold", color=PAL["iris_dark"], ha="center", va="center")

    # Mini-Scatter 1: BEFORE (y: 7.42 -> 8.48)
    add_box(ax, 11.44, 7.42, 2.28, 1.06, fc="#FFFFFF", ec=PAL["iris_edge"], lw=0.85, r=0.07, zorder=2)
    ax.text(11.52, 8.33, "Before", fontsize=9.6, fontweight="bold", color=PAL["ink_mid"], ha="left", va="center", zorder=4)
    # Protocol legend in upper-right corner
    for lx_p, lcol_p, ltxt_p in [(12.22, PAL["navy_mid"], "10x"), (12.74, PAL["amber_mid"], "SS2"), (13.24, PAL["rose_mid"], "inD")]:
        ax.add_patch(Circle((lx_p, 8.33), 0.044, facecolor=lcol_p, edgecolor="none", zorder=5))
        ax.text(lx_p + 0.062, 8.33, ltxt_p, fontsize=9.4, fontweight="bold", color=PAL["ink_mid"], ha="left", va="center", zorder=5)
    rng_c = np.random.RandomState(11)
    before_groups = [
        (11.84, 7.98, PAL["navy_mid"], "o"),
        (12.46, 8.00, PAL["amber_mid"], "o"),
        (13.10, 7.94, PAL["rose_mid"], "o"),
        (11.94, 7.58, PAL["navy_mid"], "^"),
        (12.58, 7.62, PAL["amber_mid"], "^"),
        (13.24, 7.58, PAL["rose_mid"], "^"),
    ]
    for gx, gy, gcol, gmk in before_groups:
        px = gx + rng_c.normal(0, 0.06, size=4)
        py = gy + rng_c.normal(0, 0.045, size=4)
        ax.scatter(px, py, s=24, marker=gmk, color=gcol, alpha=0.88, edgecolors="#FFFFFF", linewidths=0.4, zorder=5)
    add_arrow(ax, 12.96, 7.98, 12.56, 8.00, color=PAL["iris_dark"], lw=1.4, head_len=0.075, head_width=0.065, halo=False, zorder=6)
    add_arrow(ax, 11.98, 7.98, 12.34, 8.00, color=PAL["iris_dark"], lw=1.4, head_len=0.075, head_width=0.065, halo=False, zorder=6)

    # Downward connector Before -> After
    add_arrow(ax, 12.58, 7.36, 12.58, 7.08, color=PAL["iris_dark"], lw=1.8, head_len=0.08, head_width=0.072, halo=False)
    ax.text(12.70, 7.22, "Moment align", fontsize=9.5, fontweight="bold", color=PAL["iris_dark"], ha="left", va="center")

    # Mini-Scatter 2: AFTER (y: 5.96 -> 7.02)
    add_box(ax, 11.44, 5.96, 2.28, 1.06, fc="#FFFFFF", ec=PAL["iris_edge"], lw=0.85, r=0.07, zorder=2)
    ax.text(11.54, 6.87, "After", fontsize=9.6, fontweight="bold", color=PAL["teal_dark"], ha="left", va="center", zorder=4)
    ax.add_patch(Ellipse((12.14, 6.42), 0.72, 0.56, facecolor=PAL["navy_sub"], edgecolor="none", alpha=0.65, zorder=3))
    ax.add_patch(Ellipse((13.10, 6.40), 0.72, 0.56, facecolor=PAL["teal_sub"], edgecolor="none", alpha=0.65, zorder=3))
    for gcol in [PAL["navy_mid"], PAL["amber_mid"], PAL["rose_mid"]]:
        ax.scatter(12.14 + rng_c.normal(0, 0.11, 4), 6.42 + rng_c.normal(0, 0.085, 4), s=24, marker="o", color=gcol, alpha=0.88, edgecolors="#FFFFFF", lw=0.4, zorder=5)
        ax.scatter(13.10 + rng_c.normal(0, 0.11, 4), 6.40 + rng_c.normal(0, 0.085, 4), s=24, marker="^", color=gcol, alpha=0.88, edgecolors="#FFFFFF", lw=0.4, zorder=5)

    # Proto-Align / HA-SGMP summary box below aligned scatter
    add_box(ax, 11.42, 5.30, 2.32, 0.52, fc=PAL["iris_sub"], ec="none", lw=0.0, r=0.09, zorder=3)
    ax.text(12.58, 5.56, "Homophilic κ = 0.78\nvs. Hetero 0.53", fontsize=9.6, fontweight="bold", color=PAL["iris_dark"], ha="center", va="center", linespacing=1.15, zorder=4)

    # Subtle vertical separator inside Panel c
    ax.plot([13.86, 13.86], [5.32, 8.76], color=PAL["iris_edge"], lw=0.85, ls=":", zorder=2)

    # Right Half (x: 13.98 -> 16.36): Boundary-Gated Anchor k-NN Graph (HA-SGMP)
    ax.text(15.17, 8.70, "HA-SGMP $k$-NN", fontsize=11.8, fontweight="bold", color=PAL["iris_dark"], ha="center", va="center")
    add_box(ax, 14.00, 6.02, 2.34, 2.46, fc="#FFFFFF", ec=PAL["iris_edge"], lw=0.85, r=0.07, zorder=2)

    # Major lineage cluster (upper-left inside box)
    ax.add_patch(Ellipse((14.74, 7.52), 1.22, 1.06, facecolor=PAL["iris_sub"], edgecolor="none", alpha=0.55, zorder=3))
    major_nodes = [
        (14.34, 7.72), (14.70, 7.94), (15.12, 7.76),
        (14.42, 7.26), (14.82, 7.46), (15.08, 7.18),
    ]
    major_edges = [(0, 1), (1, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 5), (2, 5)]
    for u, v in major_edges:
        ax.plot(
            [major_nodes[u][0], major_nodes[v][0]],
            [major_nodes[u][1], major_nodes[v][1]],
            color=PAL["teal_dark"],
            lw=1.85,
            alpha=0.84,
            zorder=4,
        )
    for mx, my in major_nodes:
        ax.add_patch(Circle((mx, my), 0.098, facecolor=PAL["iris_mid"], edgecolor="#FFFFFF", lw=0.9, zorder=6))
    ax.text(14.74, 8.24, "Major Lineage", fontsize=10.0, fontweight="bold", color=PAL["iris_dark"], ha="center", va="center", zorder=6)

    # Rare lineage cluster (lower-right; solid internal homophilic edges)
    ax.add_patch(Ellipse((15.74, 6.62), 0.88, 0.78, facecolor=PAL["rose_sub"], edgecolor="none", alpha=0.72, zorder=3))
    rare_nodes = [(15.58, 6.76), (15.96, 6.68), (15.74, 6.40)]
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        ax.plot(
            [rare_nodes[u][0], rare_nodes[v][0]],
            [rare_nodes[u][1], rare_nodes[v][1]],
            color=PAL["teal_dark"],
            lw=1.70,
            alpha=0.86,
            zorder=4,
        )
    for rx, ry in rare_nodes:
        ax.add_patch(Circle((rx, ry), 0.094, facecolor=PAL["rose_mid"], edgecolor="#FFFFFF", lw=0.9, zorder=6))
    ax.text(15.72, 6.14, "Rare (<2.5%)", fontsize=9.8, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center", zorder=6)

    # Blocked heterophilic cross-boundary edges with X markers
    cross_edges = [(major_nodes[2], rare_nodes[0]), (major_nodes[5], rare_nodes[0]), (major_nodes[5], rare_nodes[2])]
    for (x1, y1), (x2, y2) in cross_edges:
        ax.plot([x1, x2], [y1, y2], color=PAL["rose_dark"], lw=1.35, ls="--", alpha=0.82, zorder=5)
        draw_gate_x_marker(ax, 0.5 * (x1 + x2), 0.5 * (y1 + y2), r=0.082, color=PAL["rose_dark"], zorder=8)

    # Clean large legend below graph (9.8 pt bold)
    ax.plot([14.04, 14.34], [5.72, 5.72], color=PAL["teal_dark"], lw=2.1)
    ax.text(14.42, 5.72, "Homophilic smooth", fontsize=9.8, fontweight="bold", color=PAL["teal_dark"], ha="left", va="center")
    ax.plot([14.04, 14.34], [5.40, 5.40], color=PAL["rose_dark"], lw=1.4, ls="--")
    draw_gate_x_marker(ax, 14.19, 5.40, r=0.062, color=PAL["rose_dark"], zorder=8)
    ax.text(14.42, 5.40, "Heterophilic blocked", fontsize=9.8, fontweight="bold", color=PAL["rose_dark"], ha="left", va="center")

    # Inter-row routing connectors
    # Track 2 (Under-pass White-Box Pathway Trace): Panel b -> Panel f (y = 4.60, lower zorder=8)
    add_ortho_arrow(
        ax,
        [(8.31, 5.16), (8.31, 4.60), (13.88, 4.60), (13.88, 4.46)],
        color=PAL["teal_dark"],
        lw=2.0,
        ls="--",
        head_len=0.115,
        head_width=0.10,
        halo=False,
        zorder=8,
    )
    add_box(ax, 9.06, 4.46, 4.08, 0.28, fc=PAL["teal_sub"], ec=PAL["teal_edge"], lw=0.9, r=0.09, zorder=13)
    ax.text(
        11.10, 4.60, "Pathway Activations & Weights",
        fontsize=10.8, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center", zorder=14,
    )

    # Primary route: Panel c -> Panel d (over-pass bridge over Track 2)
    add_ortho_arrow(
        ax,
        [(13.88, 5.16), (13.88, 4.96), (2.73, 4.96), (2.73, 4.46)],
        color=PAL["iris_dark"],
        lw=2.4,
        head_len=0.125,
        head_width=0.11,
        halo=False,
        zorder=10,
    )
    ax.plot(
        [7.60, 9.02],
        [4.96, 4.96],
        color=PAL["iris_dark"],
        lw=2.4,
        solid_capstyle="butt",
        path_effects=[patheffects.withStroke(linewidth=6.0, foreground="#FFFFFF")],
        zorder=11,
    )
    add_box(ax, 3.18, 4.81, 4.48, 0.30, fc=PAL["iris_sub"], ec=PAL["iris_edge"], lw=0.9, r=0.09, zorder=13)
    ax.text(
        5.42, 4.96, "Aligned, Rare-Preserved Embedding",
        fontsize=11.0, fontweight="bold", color=PAL["iris_dark"], ha="center", va="center", zorder=14,
    )

    # Panel d: Hyperspherical rare-margin fusion
    draw_nc_panel(
        ax, 0.12, 0.12, 5.22, 4.32,
        "d", "Hyperspherical Rare-Margin Fusion",
        PAL["amber_bg"], PAL["amber_edge"], PAL["amber_dark"],
    )

    # Left Half (x: 0.24 -> 2.74): 3D-Shaded Translucent Unit Hypersphere
    sx, sy, sr = 1.44, 2.18, 0.94
    for frac, fc_s, alp_s in [
        (1.00, "#EBD7B8", 0.55),
        (0.92, "#F4E8D4", 0.65),
        (0.76, "#FAF3E7", 0.72),
        (0.50, "#FFFDF9", 0.80),
    ]:
        ox = -0.18 * (1.0 - frac) * sr
        oy = 0.20 * (1.0 - frac) * sr
        ax.add_patch(Circle((sx + ox, sy + oy), sr * frac, facecolor=fc_s, edgecolor="none", alpha=alp_s, zorder=2))
    ax.add_patch(Circle((sx, sy), sr, facecolor="none", edgecolor=PAL["amber_mid"], lw=1.35, zorder=4))
    ax.add_patch(Arc((sx, sy), 2 * sr, 0.58 * sr, angle=0, theta1=0, theta2=180, edgecolor=PAL["amber_mid"], lw=0.95, ls="--", alpha=0.7, zorder=4))
    ax.add_patch(Arc((sx, sy), 2 * sr, 0.58 * sr, angle=0, theta1=180, theta2=360, edgecolor=PAL["amber_mid"], lw=1.1, alpha=0.85, zorder=4))
    ax.add_patch(Arc((sx, sy), 0.72 * sr, 2 * sr, angle=0, theta1=-90, theta2=90, edgecolor=PAL["amber_mid"], lw=0.85, ls=":", alpha=0.65, zorder=4))

    ax.add_patch(
        Wedge((sx, sy), sr, 18, 76, facecolor=PAL["rose_soft"], edgecolor=PAL["rose_mid"], lw=1.1, alpha=0.62, zorder=5)
    )
    ax.add_patch(Wedge((sx, sy), sr, 146, 174, facecolor=PAL["navy_sub"], edgecolor="none", alpha=0.72, zorder=5))
    ax.add_patch(Wedge((sx, sy), sr, 294, 324, facecolor=PAL["teal_sub"], edgecolor="none", alpha=0.72, zorder=5))

    rng_d = np.random.RandomState(5)
    proto_specs = [
        (47, PAL["rose_dark"], PAL["rose_mid"], "Rare", 0.16, 0.23, 4),
        (160, PAL["navy_dark"], PAL["navy_mid"], "Major 1", -0.10, 0.29, 6),
        (309, PAL["teal_dark"], PAL["teal_mid"], "Major 2", 0.18, -0.24, 6),
    ]
    for ang, col_star, col_dot, plbl, dx_l, dy_l, ndots in proto_specs:
        rad = np.deg2rad(ang)
        px = sx + sr * np.cos(rad)
        py = sy + sr * np.sin(rad)
        cx_dot = sx + 0.78 * sr * np.cos(rad) + rng_d.normal(0, 0.06, ndots)
        cy_dot = sy + 0.78 * sr * np.sin(rad) + rng_d.normal(0, 0.06, ndots)
        ax.scatter(cx_dot, cy_dot, s=18, color=col_dot, alpha=0.85, edgecolors="#FFFFFF", lw=0.4, zorder=6)
        ax.scatter([px], [py], marker="*", s=175, color=col_star, edgecolors="#FFFFFF", linewidths=0.7, zorder=8)
        ax.text(px + dx_l, py + dy_l, plbl, fontsize=10.0, fontweight="bold", color=col_star, ha="center", va="center", zorder=9)

    ax.annotate(
        "Wider rare\nmargin",
        xy=(sx + 0.56 * sr, sy + 0.52 * sr),
        xytext=(0.74, 3.44),
        fontsize=9.6,
        fontweight="bold",
        color=PAL["rose_dark"],
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=PAL["rose_sub"], edgecolor=PAL["rose_edge"], lw=0.8),
        arrowprops=dict(arrowstyle="->", color=PAL["rose_dark"], lw=1.25, connectionstyle="arc3,rad=-0.16"),
        zorder=9,
    )
    ax.text(1.48, 0.68, "Orthonormal/ETF Prototypes", fontsize=10.4, fontweight="bold", color=PAL["amber_dark"], ha="center", va="center")
    ax.text(1.48, 0.38, "Inverse-freq rare margin", fontsize=9.6, color=PAL["ink_mid"], ha="center", va="center")

    # Right half: adaptive dual-stream fusion and probability bars
    ax.text(4.04, 3.68, "Dual-Stream Fusion", fontsize=11.8, fontweight="bold", color=PAL["amber_dark"], ha="center", va="center")
    add_box(ax, 2.88, 2.74, 1.12, 0.68, fc=PAL["amber_sub"], ec=PAL["amber_edge"], lw=0.95, r=0.08, zorder=4)
    ax.text(3.44, 3.08, "Subspace\nPrototype", fontsize=10.2, fontweight="bold", color=PAL["amber_dark"], ha="center", va="center", linespacing=1.12, zorder=5)

    add_box(ax, 4.10, 2.74, 1.12, 0.68, fc=PAL["navy_sub"], ec=PAL["navy_edge"], lw=0.95, r=0.08, zorder=4)
    ax.text(4.66, 3.08, "Pathway\nMax-Margin", fontsize=10.2, fontweight="bold", color=PAL["navy_dark"], ha="center", va="center", linespacing=1.12, zorder=5)

    add_arrow(ax, 3.44, 2.70, 3.92, 2.28, color=PAL["amber_dark"], lw=1.7, head_len=0.085, head_width=0.075, halo=False)
    add_arrow(ax, 4.66, 2.70, 4.18, 2.28, color=PAL["navy_dark"], lw=1.7, head_len=0.085, head_width=0.075, halo=False)

    ax.add_patch(Circle((4.05, 2.15), 0.175, facecolor=PAL["amber_dark"], edgecolor="#FFFFFF", lw=1.1, zorder=6))
    ax.text(4.05, 2.15, "+", fontsize=13.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=7)
    add_arrow(ax, 4.05, 1.96, 4.05, 1.72, color=PAL["amber_dark"], lw=1.8, head_len=0.085, head_width=0.075, halo=False)

    # Horizontal class probability bars
    add_box(ax, 2.88, 0.64, 2.34, 1.02, fc="#FFFFFF", ec=PAL["amber_edge"], lw=0.85, r=0.07, zorder=2)
    prob_rows = [
        ("Type 1", 0.84, PAL["rose_mid"]),
        ("Type 2", 0.09, PAL["slate_muted"]),
        ("Rare", 0.05, PAL["slate_muted"]),
        ("Type C", 0.02, PAL["slate_muted"]),
    ]
    for pi, (plbl, pval, pcol) in enumerate(prob_rows):
        py = 1.50 - pi * 0.23
        ax.text(3.52, py, plbl, fontsize=9.6, fontweight="bold" if pi == 0 else "normal", color=PAL["ink_dark"], ha="right", va="center", zorder=5)
        bw = max(0.07, 1.12 * pval)
        ax.add_patch(Rectangle((3.58, py - 0.065), bw, 0.13, facecolor=pcol, edgecolor="none", linewidth=0.0, alpha=0.92, zorder=5))
        ax.text(3.64 + bw, py, f"{pval:.2f}", fontsize=9.6, fontweight="bold" if pi == 0 else "normal", color=PAL["rose_dark"] if pi == 0 else PAL["ink_mid"], ha="left", va="center", zorder=5)
    ax.text(4.05, 0.38, "Calibrated Posterior", fontsize=10.8, fontweight="bold", color=PAL["amber_dark"], ha="center", va="center")

    # Connector d -> e (x: 5.34 -> 5.70)
    add_arrow(ax, 5.36, 2.24, 5.68, 2.24, color=PAL["amber_dark"], lw=2.4, head_len=0.12, head_width=0.105)

    # Panel e: Evidential uncertainty and open-set rejection
    draw_nc_panel(
        ax, 5.70, 0.12, 5.22, 4.32,
        "e", "Evidential Uncertainty & Open-Set",
        PAL["rose_bg"], PAL["rose_edge"], PAL["rose_dark"],
    )

    # Left half: Dirichlet evidence simplex
    ax.text(7.00, 3.68, "Dirichlet Simplex", fontsize=11.8, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center")
    tri_top = np.array([7.00, 3.24])
    tri_bl = np.array([6.00, 1.52])
    tri_br = np.array([8.00, 1.52])
    tri_pts = np.vstack([tri_top, tri_bl, tri_br])
    ax.add_patch(Polygon(tri_pts, closed=True, facecolor="#FFFDFD", edgecolor=PAL["rose_mid"], lw=1.45, zorder=3))

    rng_e = np.random.RandomState(23)
    corners = [
        (tri_top, (7.00, 2.95), PAL["teal_sub"], PAL["teal_mid"], "Type 1", (7.00, 3.40)),
        (tri_bl, (6.26, 1.68), PAL["navy_sub"], PAL["navy_mid"], "Type 2", (6.02, 1.32)),
        (tri_br, (7.74, 1.68), PAL["amber_sub"], PAL["amber_mid"], "Type C", (7.98, 1.32)),
    ]
    for _, (bx, by), b_tint, b_dot, clbl, (lx, ly) in corners:
        ax.add_patch(Circle((bx, by), 0.20, facecolor=b_tint, edgecolor="none", alpha=0.80, zorder=4))
        ax.scatter(bx + rng_e.normal(0, 0.052, 5), by + rng_e.normal(0, 0.052, 5), s=17, color=b_dot, alpha=0.88, edgecolors="#FFFFFF", lw=0.35, zorder=5)
        ax.text(lx, ly, clbl, fontsize=10.0, fontweight="bold", color=PAL["ink_dark"], ha="center", va="center", zorder=6)

    cent = np.array([7.00, 2.34])
    ax.add_patch(Circle(cent, 0.30, facecolor=PAL["rose_soft"], edgecolor="none", alpha=0.55, zorder=4))
    ax.add_patch(Circle(cent, 0.165, facecolor=PAL["rose_dark"], edgecolor="#FFFFFF", lw=1.1, zorder=6))
    ax.text(cent[0], cent[1], "?", fontsize=12.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=7)
    ax.text(cent[0], 1.94, "Novel / Pathol.", fontsize=9.6, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center", zorder=7)
    ax.text(7.00, 0.86, "Vacuity: $u_i = C / S_i$", fontsize=10.8, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center")
    ax.text(7.00, 0.44, "Corners: Known | Center: Alert", fontsize=9.5, color=PAL["ink_mid"], ha="center", va="center")

    # Right Half (x: 8.28 -> 10.80): Overlapping Density Plot + Val 95% Cutoff Gate
    ax.text(9.54, 3.68, "Calibrated Cutoff Gate", fontsize=11.8, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center")
    add_box(ax, 8.32, 2.08, 2.44, 1.38, fc="#FFFFFF", ec=PAL["rose_edge"], lw=0.85, r=0.07, zorder=2)
    dx_g = np.linspace(0.0, 1.0, 120)
    y_known = np.exp(-0.5 * ((dx_g - 0.24) / 0.115) ** 2)
    y_novel = 0.86 * np.exp(-0.5 * ((dx_g - 0.74) / 0.145) ** 2)
    px_g = 8.42 + 2.24 * dx_g
    base_y = 2.18
    scale_y = 0.72
    ax.fill_between(px_g, base_y, base_y + scale_y * y_known, facecolor=PAL["teal_mid"], alpha=0.38, edgecolor="none", zorder=4)
    ax.plot(px_g, base_y + scale_y * y_known, color=PAL["teal_dark"], lw=1.7, zorder=5)
    ax.fill_between(px_g, base_y, base_y + scale_y * y_novel, facecolor=PAL["rose_mid"], alpha=0.42, edgecolor="none", zorder=4)
    ax.plot(px_g, base_y + scale_y * y_novel, color=PAL["rose_dark"], lw=1.7, zorder=5)
    ax.plot([8.42, 10.66], [base_y, base_y], color=PAL["ink_mid"], lw=0.85, zorder=4)

    x_cut = 8.42 + 2.24 * 0.47
    ax.plot([x_cut, x_cut], [base_y, base_y + 0.96], color=PAL["ink_dark"], lw=1.45, ls="--", zorder=6)
    ax.text(x_cut, base_y + 1.01, "Val 95% Cutoff", fontsize=9.8, fontweight="bold", color=PAL["ink_dark"], ha="center", va="bottom", zorder=7)
    ax.text(8.94, 3.00, "Known", fontsize=9.8, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center", zorder=7)
    ax.text(10.14, 2.94, "Novel", fontsize=9.8, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center", zorder=7)
    ax.text(9.54, 1.90, "Novelty Score", fontsize=9.6, color=PAL["ink_mid"], ha="center", va="center")

    # Branching arrows into 2 spacious decision pills
    add_arrow(ax, 8.91, 1.74, 8.91, 1.44, color=PAL["teal_dark"], lw=1.9, head_len=0.085, head_width=0.075, halo=False)
    add_arrow(ax, 10.17, 1.74, 10.17, 1.44, color=PAL["rose_dark"], lw=1.9, head_len=0.085, head_width=0.075, halo=False)

    add_box(ax, 8.32, 0.34, 1.17, 1.06, fc=PAL["teal_sub"], ec=PAL["teal_edge"], lw=0.95, r=0.08, zorder=4)
    ax.text(8.905, 1.06, "Assign\nKnown", fontsize=10.6, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center", linespacing=1.10, zorder=5)
    ax.text(8.905, 0.56, "0.088 ECE", fontsize=9.6, fontweight="bold", color=PAL["ink_mid"], ha="center", va="center", zorder=5)

    add_box(ax, 9.59, 0.34, 1.17, 1.06, fc=PAL["rose_sub"], ec=PAL["rose_edge"], lw=0.95, r=0.08, zorder=4)
    ax.text(10.175, 1.06, "Flag\nNovel", fontsize=10.6, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center", linespacing=1.10, zorder=5)
    ax.text(10.175, 0.56, "Clinical alert", fontsize=9.6, fontweight="bold", color=PAL["ink_mid"], ha="center", va="center", zorder=5)

    # Connector e -> f (x: 10.92 -> 11.28)
    add_arrow(ax, 10.94, 2.24, 11.26, 2.24, color=PAL["rose_dark"], lw=2.4, head_len=0.12, head_width=0.105)

    # Panel f: Attribution and clinical translation
    draw_nc_panel(
        ax, 11.28, 0.12, 5.20, 4.32,
        "f", "Attribution & Clinical Translation",
        PAL["out_bg"], PAL["out_edge"], PAL["teal_dark"],
    )

    # Top half: curved two-hop attribution ribbons
    ax.text(12.14, 3.72, "Driver Genes", fontsize=10.6, fontweight="bold", color=PAL["navy_dark"], ha="center", va="center")
    ax.text(13.98, 3.72, "GO-BP Pathway", fontsize=10.6, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center")
    ax.text(15.72, 3.72, "Lineage", fontsize=10.6, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center")

    sankey_rows = [
        (3.40, "CD74 / HLA-DRB1", "B Cell Med. Imm.", "B Cell", PAL["navy_mid"], PAL["teal_mid"]),
        (3.08, "S100A9 / S100A8", "Inflamm. Resp.", "CD14+ Mono", PAL["navy_mid"], PAL["teal_mid"]),
        (2.76, "GZMB / TYROBP", "Leuk. Cytotox.", "NK Cell", PAL["navy_mid"], PAL["teal_mid"]),
        (2.44, "PF4 / CLU", "Platelet Degran.", "Megakaryo.", PAL["amber_mid"], PAL["amber_mid"]),
    ]
    for ry, g_txt, p_txt, c_txt, rcol1, rcol2 in sankey_rows:
        draw_bezier_ribbon(ax, 12.90, ry, 0.14, 13.18, ry, 0.14, rcol1, alpha=0.35, zorder=3)
        draw_bezier_ribbon(ax, 14.78, ry, 0.14, 15.06, ry, 0.14, rcol2, alpha=0.35, zorder=3)

        # Gene pill (x: 11.38 -> 12.90, w=1.52)
        add_box(ax, 11.40, ry - 0.115, 1.48, 0.23, fc=PAL["navy_sub"], ec="none", lw=0.0, r=0.06, zorder=5)
        ax.text(12.14, ry, g_txt, fontsize=9.5, fontweight="bold", color=PAL["navy_dark"], ha="center", va="center", zorder=6)

        # Pathway pill (x: 13.18 -> 14.78, w=1.60)
        p_bg = PAL["amber_sub"] if "Platelet" in p_txt else PAL["teal_sub"]
        p_tc = PAL["amber_dark"] if "Platelet" in p_txt else PAL["teal_dark"]
        add_box(ax, 13.20, ry - 0.115, 1.56, 0.23, fc=p_bg, ec="none", lw=0.0, r=0.06, zorder=5)
        ax.text(13.98, ry, p_txt, fontsize=9.5, fontweight="bold", color=p_tc, ha="center", va="center", zorder=6)

        # Lineage pill (x: 15.06 -> 16.38, w=1.32)
        c_bg = PAL["amber_sub"] if "Mega" in c_txt else PAL["rose_sub"]
        c_tc = PAL["amber_dark"] if "Mega" in c_txt else PAL["rose_dark"]
        add_box(ax, 15.08, ry - 0.115, 1.28, 0.23, fc=c_bg, ec="none", lw=0.0, r=0.06, zorder=5)
        ax.text(15.72, ry, c_txt, fontsize=9.5, fontweight="bold", color=c_tc, ha="center", va="center", zorder=6)

    # Warm Amber De Novo Discovery Strip (y: 1.98 -> 2.26)
    add_box(ax, 11.40, 1.98, 4.96, 0.28, fc=PAL["amber_sub"], ec=PAL["amber_edge"], lw=0.85, r=0.07, zorder=4)
    ax.text(
        13.88, 2.12, "De Novo: IL3RA · NKG7 · TUBB1  (7 protocols)",
        fontsize=10.2, fontweight="bold", color=PAL["amber_dark"], ha="center", va="center", zorder=5,
    )

    # Bottom-Left Micro-Plot (x: 11.40 -> 13.82, y: 0.26 -> 1.88): Causal Faithfulness
    add_box(ax, 11.40, 0.26, 2.42, 1.62, fc="#FFFFFF", ec=PAL["iris_edge"], lw=0.85, r=0.08, zorder=2)
    ax.text(12.61, 1.72, "Causal Faithfulness", fontsize=11.2, fontweight="bold", color=PAL["iris_dark"], ha="center", va="center", zorder=4)
    mx_l, mx_r, my_b, my_t = 11.56, 13.68, 0.58, 1.38
    ax.plot([mx_l, mx_r], [my_b, my_b], color="#CBD5E1", lw=0.85, zorder=3)
    ax.plot([mx_l, mx_l], [my_b, my_t], color="#CBD5E1", lw=0.85, zorder=3)
    tx = np.linspace(0.0, 1.0, 60)
    px_m = mx_l + (mx_r - mx_l) * tx
    f1_y = my_b + (my_t - my_b) * (0.76 - 0.56 * (tx ** 1.35))
    vac_y = my_b + (my_t - my_b) * (0.12 + 0.62 * np.exp(-0.5 * ((tx - 0.72) / 0.16) ** 2))
    ax.fill_between(px_m, my_b, vac_y, facecolor=PAL["iris_mid"], alpha=0.22, edgecolor="none", zorder=4)
    ax.plot(px_m, vac_y, color=PAL["iris_dark"], lw=1.75, ls="-.", zorder=5)
    ax.plot(px_m, f1_y, color=PAL["rose_dark"], lw=1.85, zorder=5)
    ax.text(11.60, 1.44, "−42.6 pp", fontsize=9.5, fontweight="bold", color=PAL["rose_dark"], ha="left", va="center", zorder=6)
    ax.text(13.16, 1.44, "6.2× Ridge", fontsize=9.5, fontweight="bold", color=PAL["iris_dark"], ha="center", va="center", zorder=6)
    ax.text(12.61, 0.40, "KO & lineage swap", fontsize=9.5, color=PAL["ink_mid"], ha="center", va="center", zorder=5)

    # Bottom-Right Micro-Plot (x: 13.94 -> 16.36, y: 0.26 -> 1.88): COVID-19 Clinical Substate Alert
    add_box(ax, 13.94, 0.26, 2.42, 1.62, fc="#FFFFFF", ec=PAL["rose_edge"], lw=0.85, r=0.08, zorder=2)
    ax.text(15.15, 1.72, "Clinical Substate Alert", fontsize=11.2, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center", zorder=4)
    vy = np.linspace(0.0, 1.0, 60)
    x_v1 = 14.38
    w_v1 = 0.22 * np.exp(-0.5 * ((vy - 0.28) / 0.17) ** 2)
    py_v = 0.62 + 0.68 * vy
    ax.fill_betweenx(py_v, x_v1 - w_v1, x_v1 + w_v1, facecolor=PAL["teal_mid"], alpha=0.55, edgecolor="none", zorder=4)
    ax.plot([x_v1 - 0.13, x_v1 + 0.13], [0.62 + 0.68 * 0.28] * 2, color="#FFFFFF", lw=1.7, zorder=5)
    x_v2 = 15.18
    w_v2 = 0.23 * np.exp(-0.5 * ((vy - 0.62) / 0.19) ** 2)
    ax.fill_betweenx(py_v, x_v2 - w_v2, x_v2 + w_v2, facecolor=PAL["rose_mid"], alpha=0.62, edgecolor="none", zorder=4)
    ax.plot([x_v2 - 0.13, x_v2 + 0.13], [0.62 + 0.68 * 0.62] * 2, color="#FFFFFF", lw=1.7, zorder=5)
    add_arrow(ax, 14.58, 0.90, 14.94, 1.12, color=PAL["rose_dark"], lw=1.45, head_len=0.075, head_width=0.065, halo=False, zorder=6)
    ax.text(14.78, 1.44, "+34% vac.", fontsize=9.5, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center", zorder=6)
    ax.text(x_v1, 0.42, "FABP4+", fontsize=9.5, fontweight="bold", color=PAL["teal_dark"], ha="center", va="center")
    ax.text(x_v2, 0.42, "FCN1+", fontsize=9.5, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center")

    # Mini 3-tile vertical COVID severity heat-strip (HC -> Mod -> Sev +5.4 SD)
    sev_tiles = [(1.30, PAL["teal_sub"], "HC"), (1.02, PAL["amber_soft"], "Mod"), (0.74, PAL["rose_dark"], "Sev")]
    for ty, tc, tlbl in sev_tiles:
        ax.add_patch(Rectangle((15.68, ty - 0.11), 0.56, 0.22, facecolor=tc, edgecolor="none", linewidth=0.0, zorder=4))
        txt_c = "#FFFFFF" if tlbl == "Sev" else PAL["ink_dark"]
        ax.text(15.96, ty, tlbl, fontsize=9.4, fontweight="bold", color=txt_c, ha="center", va="center", zorder=5)
    ax.text(15.96, 0.42, "+5.4 SD", fontsize=9.5, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center")

    # Save PDF and 300-DPI PNG outputs
    extra_dirs = ["figures", "results/figures"]
    for p in [out_pdf_paper, out_png_paper, out_pdf_results, out_png_results]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    fig.savefig(out_pdf_paper, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png_paper, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    if os.path.abspath(out_pdf_paper) != os.path.abspath(out_pdf_results):
        shutil.copy2(out_pdf_paper, out_pdf_results)
    if os.path.abspath(out_png_paper) != os.path.abspath(out_png_results):
        shutil.copy2(out_png_paper, out_png_results)
    for d in extra_dirs:
        os.makedirs(d, exist_ok=True)
        for src_f, fname in [
            (out_pdf_paper, "fig0_model_architecture.pdf"),
            (out_png_paper, "fig0_model_architecture.png"),
        ]:
            dst_f = os.path.join(d, fname)
            if os.path.abspath(src_f) != os.path.abspath(dst_f):
                shutil.copy2(src_f, dst_f)
    plt.close(fig)
    print(f"[OK] Saved architecture figure to {out_pdf_paper}")


# Alias for external callers
plot_fig0_architecture = generate_architecture_figure


def _load_scpathproto_ece(default_ece: float = 0.0877) -> float:
    """Read scPathProto 37-task ECE from perf_calibration_ece.csv."""
    for rel_path in [
        "results/tables/perf_calibration_ece.csv",
        "code/results/tables/perf_calibration_ece.csv",
    ]:
        if os.path.exists(rel_path):
            with open(rel_path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("Method") == "scPathProto (Ours)":
                        return float(row["ECE_Score"])
    return default_ece


def generate_graphical_abstract(
    out_pdf_paper: str = "figures/graphical_abstract.pdf",
    out_png_paper: str = "figures/graphical_abstract.png",
):
    """Render the 3-column graphical abstract (13.8 x 5.2 in)."""
    W, H = 13.8, 5.2
    fig = plt.figure(figsize=(W, H), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0.0, W)
    ax.set_ylim(0.0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    def _draw_ga_column(x, y, w, h, num_str, title, bg, edge, dark_col):
        """Draw a column container with numbered badge and header divider."""
        add_box(ax, x, y, w, h, fc=bg, ec=edge, lw=1.1, r=0.14, zorder=1)
        cy_hdr = y + h - 0.28
        ax.add_patch(Circle((x + 0.25, cy_hdr), 0.145, facecolor=dark_col, edgecolor="none", zorder=5))
        ax.text(
            x + 0.25,
            cy_hdr - 0.005,
            num_str,
            fontsize=12.4,
            fontweight="bold",
            color="#FFFFFF",
            ha="center",
            va="center",
            zorder=6,
        )
        ax.text(
            x + 0.46,
            cy_hdr,
            title,
            fontsize=13.5,
            fontweight="bold",
            color=dark_col,
            ha="left",
            va="center",
            zorder=5,
        )
        ax.plot(
            [x + 0.14, x + w - 0.14],
            [y + h - 0.56, y + h - 0.56],
            color=edge,
            lw=1.0,
            alpha=0.85,
            zorder=4,
        )

    # Column 1: scRNA-seq annotation challenges
    _draw_ga_column(
        0.06, 0.08, 4.22, 5.04,
        "1", "Real-World scRNA-seq Challenges",
        PAL["navy_bg"], PAL["navy_edge"], PAL["navy_dark"],
    )

    c1_y0s = [3.46, 2.38, 1.30, 0.22]
    c1_h = 0.97
    c1_x, c1_w = 0.18, 3.98
    c1_cards = [
        (
            "Technical Dropout Sparsity",
            "80–96% zero counts",
            "Distorts gene-level distances",
            PAL["navy_dark"], PAL["navy_edge"], PAL["navy_sub"], "sparse",
        ),
        (
            "Cross-Platform Batch Shift",
            "10x · Smart-seq2 · inDrop",
            "Protocol shifts break transfer",
            PAL["teal_dark"], PAL["teal_edge"], PAL["teal_sub"], "shift",
        ),
        (
            "Rare Lineages (<2.5%)",
            "Severe class imbalance",
            "Eroded by naive $k$-NN smoothing",
            PAL["iris_dark"], PAL["iris_edge"], PAL["iris_sub"], "rare",
        ),
        (
            "Unseen & Black-Box Risk",
            "Overconfident softmax failure",
            "Opaque calls miss novel states",
            PAL["rose_dark"], PAL["rose_edge"], PAL["rose_sub"], "simplex",
        ),
    ]

    for y0, (ctitle, cline1, cline2, tcol, ecol, wcol, kind) in zip(c1_y0s, c1_cards):
        ymid = y0 + 0.5 * c1_h
        add_box(ax, c1_x, y0, c1_w, c1_h, fc="#FFFFFF", ec=ecol, lw=0.95, r=0.09, zorder=2)

        # Left visual micro-icon well (0.86 x 0.77 in)
        ix, iy, iw, ih = c1_x + 0.09, y0 + 0.10, 0.86, 0.77
        add_box(ax, ix, iy, iw, ih, fc=wcol, ec="none", lw=0.0, r=0.07, alpha=0.64, zorder=3)
        icx, icy = ix + 0.5 * iw, iy + 0.5 * ih

        if kind == "sparse":
            # 4x4 sparse expression count matrix icon
            add_box(ax, ix + 0.08, iy + 0.07, iw - 0.16, ih - 0.14, fc="#FFFFFF", ec=PAL["navy_edge"], lw=0.65, r=0.04, zorder=4)
            mat4 = np.array(
                [
                    [0.95, 0.0, 0.0, 0.65],
                    [0.0, 0.0, 0.82, 0.0],
                    [0.0, 0.74, 0.0, 0.0],
                    [0.60, 0.0, 0.0, 0.90],
                ]
            )
            gx0, gy0, gw, gh = ix + 0.11, iy + 0.10, iw - 0.22, ih - 0.20
            cw, ch = gw / 4.0, gh / 4.0
            for r in range(4):
                for c in range(4):
                    v = mat4[r, c]
                    fc_t = PAL["navy_dark"] if v > 0.85 else (PAL["navy_mid"] if v > 0 else "#E4ECF3")
                    alp_t = 0.94 if v > 0 else 1.0
                    ax.add_patch(
                        Rectangle(
                            (gx0 + c * cw + 0.013, gy0 + (3 - r) * ch + 0.013),
                            cw - 0.026,
                            ch - 0.026,
                            facecolor=fc_t,
                            edgecolor="none",
                            linewidth=0.0,
                            alpha=alp_t,
                            zorder=5,
                        )
                    )
        elif kind == "shift":
            # 3 shifted platform dot clusters separated by dashed shift vectors
            clusters = [
                (ix + 0.22, iy + 0.55, PAL["navy_mid"], "o", [(-0.06, 0.04), (0.04, 0.05), (-0.01, -0.05)]),
                (ix + 0.66, iy + 0.55, PAL["amber_mid"], "^", [(-0.05, 0.04), (0.05, 0.03), (0.0, -0.06)]),
                (ix + 0.44, iy + 0.21, PAL["rose_mid"], "s", [(-0.06, 0.03), (0.06, 0.02), (0.0, -0.05)]),
            ]
            for cx_p, cy_p, col_p, mk_p, offs in clusters:
                ax.add_patch(Circle((cx_p, cy_p), 0.145, facecolor="#FFFFFF", edgecolor="none", alpha=0.80, zorder=4))
                xs = [cx_p + dx for dx, _ in offs]
                ys = [cy_p + dy for _, dy in offs]
                ax.scatter(xs, ys, s=24, marker=mk_p, color=col_p, alpha=0.95, edgecolors="#FFFFFF", lw=0.4, zorder=6)
            ax.plot([ix + 0.36, ix + 0.52], [iy + 0.55, iy + 0.55], color=PAL["teal_dark"], lw=1.4, ls="--", zorder=5)
            ax.plot([ix + 0.29, ix + 0.38], [iy + 0.41, iy + 0.31], color=PAL["teal_dark"], lw=1.4, ls="--", zorder=5)
            ax.plot([ix + 0.59, ix + 0.50], [iy + 0.41, iy + 0.31], color=PAL["teal_dark"], lw=1.4, ls="--", zorder=5)
        elif kind == "rare":
            # Large dominant iris cluster encroaching on tiny rose rare cluster
            ax.add_patch(Ellipse((ix + 0.33, iy + 0.48), 0.50, 0.44, facecolor="#FFFFFF", edgecolor="none", alpha=0.85, zorder=4))
            maj_pts = [
                (ix + 0.19, iy + 0.56), (ix + 0.34, iy + 0.61), (ix + 0.46, iy + 0.52),
                (ix + 0.22, iy + 0.39), (ix + 0.36, iy + 0.43), (ix + 0.44, iy + 0.34),
            ]
            for px, py in maj_pts:
                ax.add_patch(Circle((px, py), 0.045, facecolor=PAL["iris_mid"], edgecolor="#FFFFFF", lw=0.6, zorder=6))
            ax.add_patch(Ellipse((ix + 0.67, iy + 0.22), 0.30, 0.25, facecolor=PAL["rose_soft"], edgecolor="none", alpha=0.72, zorder=4))
            for px, py in [(ix + 0.63, iy + 0.26), (ix + 0.72, iy + 0.19)]:
                ax.add_patch(Circle((px, py), 0.043, facecolor=PAL["rose_dark"], edgecolor="#FFFFFF", lw=0.6, zorder=6))
            ax.plot([ix + 0.46, ix + 0.61], [iy + 0.36, iy + 0.26], color=PAL["rose_dark"], lw=1.3, ls="--", zorder=5)
            ax.plot([ix + 0.48, ix + 0.63], [iy + 0.50, iy + 0.28], color=PAL["rose_dark"], lw=1.3, ls="--", zorder=5)
        else:
            # Mini Dirichlet simplex with corner dots cleanly inside triangle + Berry Rose '?' center
            tri = np.array([[icx, iy + 0.69], [ix + 0.10, iy + 0.11], [ix + iw - 0.10, iy + 0.11]])
            ax.add_patch(Polygon(tri, closed=True, facecolor="#FFFFFF", edgecolor=PAL["rose_mid"], lw=1.35, zorder=4))
            for (vx, vy), vc in zip(
                [(icx, iy + 0.55), (ix + 0.23, iy + 0.18), (ix + iw - 0.23, iy + 0.18)],
                [PAL["teal_mid"], PAL["navy_mid"], PAL["amber_mid"]],
            ):
                ax.add_patch(Circle((vx, vy), 0.046, facecolor=vc, edgecolor="#FFFFFF", lw=0.5, zorder=5))
            ax.add_patch(Circle((icx, iy + 0.33), 0.140, facecolor=PAL["rose_soft"], edgecolor="none", alpha=0.62, zorder=5))
            ax.add_patch(Circle((icx, iy + 0.33), 0.102, facecolor=PAL["rose_dark"], edgecolor="#FFFFFF", lw=0.9, zorder=6))
            ax.text(icx, iy + 0.325, "?", fontsize=10.8, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=7)

        # Right text block
        tx = ix + iw + 0.15
        ax.text(tx, ymid + 0.24, ctitle, fontsize=10.8, fontweight="bold", color=tcol, ha="left", va="center", zorder=5)
        ax.text(tx, ymid - 0.02, cline1, fontsize=9.6, fontweight="bold", color=PAL["ink_dark"], ha="left", va="center", zorder=5)
        ax.text(tx, ymid - 0.25, cline2, fontsize=9.5, color=PAL["ink_mid"], ha="left", va="center", zorder=5)

    # Connector 1 -> 2
    add_arrow(ax, 4.30, 2.55, 4.64, 2.55, color=PAL["navy_dark"], lw=2.6, head_len=0.13, head_width=0.115)
    ax.text(4.47, 2.78, "$\\mathbf{X}$", fontsize=11.5, fontweight="bold", color=PAL["navy_dark"], ha="center", va="center")

    # Column 2: scPathProto pipeline
    _draw_ga_column(
        4.66, 0.08, 4.22, 5.04,
        "2", "scPathProto White-Box Pipeline",
        PAL["teal_bg"], PAL["teal_edge"], PAL["teal_dark"],
    )

    c2_y0s = [3.51, 2.42, 1.33, 0.24]
    c2_h = 0.90
    c2_x, c2_w = 4.78, 3.98
    c2_stages = [
        (
            "1. ZIRG Co-Expression Gate",
            "Selective technical-zero recovery",
            "Preserves non-zeros & true silence",
            PAL["navy_dark"], PAL["navy_edge"], PAL["navy_sub"], "zirg",
        ),
        (
            "2. Hybrid MPCA Pathways",
            "Curated ≤300 + 8 de novo slots",
            "Pathway covariance & dual SVD",
            PAL["teal_dark"], PAL["teal_edge"], PAL["teal_sub"], "mpca",
        ),
        (
            "3. Proto-Align + HA-SGMP",
            "Moment align & rare-gate $k$-NN",
            "Blocks heterophilic rare erosion",
            PAL["iris_dark"], PAL["iris_edge"], PAL["iris_sub"], "hasgmp",
        ),
        (
            "4. Prototypes & Vacuity",
            "Rare angular margin + $u_i = C/S_i$",
            "Calibrated posterior & novelty gate",
            PAL["amber_dark"], PAL["amber_edge"], PAL["amber_sub"], "etf",
        ),
    ]

    for idx, (y0, (stitle, sline1, sline2, tcol, ecol, wcol, kind)) in enumerate(zip(c2_y0s, c2_stages)):
        ymid = y0 + 0.5 * c2_h
        add_box(ax, c2_x, y0, c2_w, c2_h, fc="#FFFFFF", ec=ecol, lw=0.95, r=0.09, zorder=2)

        # Left stage icon box
        ix, iy, iw, ih = c2_x + 0.09, y0 + 0.09, 0.94, 0.72
        add_box(ax, ix, iy, iw, ih, fc=wcol, ec="none", lw=0.0, r=0.07, alpha=0.64, zorder=3)
        icx, icy = ix + 0.5 * iw, iy + 0.5 * ih

        if kind == "zirg":
            # Pathway co-expression recovery icon
            hub = (ix + 0.37, icy)
            nbrs = [(ix + 0.16, iy + 0.54), (ix + 0.18, iy + 0.19), (ix + 0.56, iy + 0.54), (ix + 0.56, iy + 0.19)]
            for nx, ny in nbrs:
                ax.plot([nx, hub[0]], [ny, hub[1]], color=PAL["navy_mid"], lw=1.45, alpha=0.88, zorder=4)
                ax.add_patch(Circle((nx, ny), 0.056, facecolor=PAL["navy_mid"], edgecolor="#FFFFFF", lw=0.7, zorder=5))
            ax.add_patch(Circle(hub, 0.095, facecolor=PAL["amber_mid"], edgecolor="#FFFFFF", lw=0.9, zorder=6))
            ax.text(hub[0], hub[1] - 0.005, "+", fontsize=10.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=7)
            draw_check_badge(ax, ix + 0.79, icy, r=0.095, fc=PAL["teal_dark"], zorder=7)
        elif kind == "mpca":
            # Hybrid curated mask and de novo row icon
            cur3 = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=float)
            fw3 = [0.92, 0.72, 0.85]
            mx0, my0, mw, mh = ix + 0.08, iy + 0.28, 0.50, 0.36
            cw_m, ch_m = mw / 4.0, mh / 3.0
            for r in range(3):
                ry = my0 + (2 - r) * ch_m
                for c in range(4):
                    fc_m = PAL["teal_mid"] if cur3[r, c] > 0 else "#FFFFFF"
                    ax.add_patch(
                        Rectangle((mx0 + c * cw_m + 0.01, ry + 0.012), cw_m - 0.02, ch_m - 0.024, facecolor=fc_m, edgecolor="none", lw=0.0, zorder=5)
                    )
                # Horizontal F-weight bar
                ax.add_patch(
                    Rectangle((mx0 + mw + 0.04, ry + 0.022), 0.24 * fw3[r], ch_m - 0.044, facecolor=PAL["teal_dark"], edgecolor="none", lw=0.0, alpha=0.90, zorder=5)
                )
            ax.plot([ix + 0.08, ix + iw - 0.08], [iy + 0.23, iy + 0.23], color=PAL["amber_mid"], lw=1.05, ls="--", zorder=5)
            # De novo regulon row and weight bar
            dn_y, dn_h = iy + 0.07, 0.12
            for c in range(4):
                alp_d = 0.92 if c in (0, 2, 3) else 0.45
                ax.add_patch(
                    Rectangle((mx0 + c * cw_m + 0.01, dn_y), cw_m - 0.02, dn_h, facecolor=PAL["amber_mid"], edgecolor="none", lw=0.0, alpha=alp_d, zorder=5)
                )
            ax.add_patch(
                Rectangle((mx0 + mw + 0.04, dn_y + 0.015), 0.21, dn_h - 0.03, facecolor=PAL["amber_dark"], edgecolor="none", lw=0.0, alpha=0.92, zorder=5)
            )
        elif kind == "hasgmp":
            # Boundary-gated k-NN graph icon
            ax.add_patch(Ellipse((ix + 0.29, iy + 0.45), 0.44, 0.42, facecolor="#FFFFFF", edgecolor="none", alpha=0.72, zorder=3))
            m_nodes = [(ix + 0.17, iy + 0.53), (ix + 0.36, iy + 0.58), (ix + 0.41, iy + 0.37), (ix + 0.20, iy + 0.32)]
            for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
                ax.plot([m_nodes[u][0], m_nodes[v][0]], [m_nodes[u][1], m_nodes[v][1]], color=PAL["teal_dark"], lw=1.5, alpha=0.88, zorder=4)
            for mx, my in m_nodes:
                ax.add_patch(Circle((mx, my), 0.052, facecolor=PAL["iris_mid"], edgecolor="#FFFFFF", lw=0.65, zorder=6))
            ax.add_patch(Ellipse((ix + 0.74, iy + 0.23), 0.32, 0.28, facecolor=PAL["rose_soft"], edgecolor="none", alpha=0.68, zorder=3))
            r_nodes = [(ix + 0.69, iy + 0.28), (ix + 0.80, iy + 0.18)]
            ax.plot([r_nodes[0][0], r_nodes[1][0]], [r_nodes[0][1], r_nodes[1][1]], color=PAL["teal_dark"], lw=1.5, zorder=4)
            for rx, ry in r_nodes:
                ax.add_patch(Circle((rx, ry), 0.050, facecolor=PAL["rose_dark"], edgecolor="#FFFFFF", lw=0.65, zorder=6))
            for p1, p2 in [(m_nodes[1], r_nodes[0]), (m_nodes[2], r_nodes[0])]:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=PAL["rose_dark"], lw=1.25, ls="--", zorder=5)
                draw_gate_x_marker(ax, 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), r=0.052, color=PAL["rose_dark"], zorder=8)
        else:
            # Unit hypersphere with rare-class margin and prototypes
            sr = 0.29
            for frac, fc_s, alp_s in [
                (1.00, "#EBD7B8", 0.70),
                (0.86, "#F6ECD9", 0.78),
                (0.55, "#FFFDF9", 0.88),
            ]:
                ax.add_patch(
                    Circle(
                        (icx - 0.18 * (1.0 - frac) * sr, icy + 0.20 * (1.0 - frac) * sr),
                        sr * frac,
                        facecolor=fc_s,
                        edgecolor="none",
                        alpha=alp_s,
                        zorder=4,
                    )
                )
            ax.add_patch(Circle((icx, icy), sr, facecolor="none", edgecolor=PAL["amber_mid"], lw=1.2, zorder=5))
            ax.add_patch(Arc((icx, icy), 2 * sr, 0.58 * sr, angle=0, theta1=0, theta2=180, edgecolor=PAL["amber_mid"], lw=0.8, ls="--", alpha=0.65, zorder=5))
            ax.add_patch(Arc((icx, icy), 2 * sr, 0.58 * sr, angle=0, theta1=180, theta2=360, edgecolor=PAL["amber_mid"], lw=0.95, alpha=0.85, zorder=5))
            ax.add_patch(Arc((icx, icy), 0.72 * sr, 2 * sr, angle=0, theta1=-90, theta2=90, edgecolor=PAL["amber_mid"], lw=0.75, ls=":", alpha=0.60, zorder=5))
            ax.add_patch(Wedge((icx, icy), sr, 18, 78, facecolor=PAL["rose_soft"], edgecolor=PAL["rose_mid"], lw=0.95, alpha=0.76, zorder=6))
            ax.add_patch(Wedge((icx, icy), sr, 150, 178, facecolor=PAL["navy_sub"], edgecolor="none", alpha=0.82, zorder=6))
            ax.add_patch(Wedge((icx, icy), sr, 292, 322, facecolor=PAL["teal_sub"], edgecolor="none", alpha=0.82, zorder=6))
            for ang, scol, dcol in [
                (48, PAL["rose_dark"], PAL["rose_mid"]),
                (164, PAL["navy_dark"], PAL["navy_mid"]),
                (307, PAL["teal_dark"], PAL["teal_mid"]),
            ]:
                rad = np.deg2rad(ang)
                ax.scatter(
                    [icx + 0.70 * sr * np.cos(rad - 0.14), icx + 0.74 * sr * np.cos(rad + 0.12)],
                    [icy + 0.70 * sr * np.sin(rad - 0.14), icy + 0.74 * sr * np.sin(rad + 0.12)],
                    s=11,
                    color=dcol,
                    edgecolors="#FFFFFF",
                    lw=0.35,
                    zorder=7,
                )
                ax.scatter(
                    [icx + sr * np.cos(rad)],
                    [icy + sr * np.sin(rad)],
                    marker="*",
                    s=95,
                    color=scol,
                    edgecolors="#FFFFFF",
                    lw=0.55,
                    zorder=8,
                )

        # Right text block
        tx = ix + iw + 0.13
        ax.text(tx, ymid + 0.22, stitle, fontsize=10.8, fontweight="bold", color=tcol, ha="left", va="center", zorder=5)
        ax.text(tx, ymid - 0.03, sline1, fontsize=9.6, fontweight="bold", color=PAL["ink_dark"], ha="left", va="center", zorder=5)
        ax.text(tx, ymid - 0.25, sline2, fontsize=9.5, color=PAL["ink_mid"], ha="left", va="center", zorder=5)

        if idx < 3:
            # Downward stage connector
            add_arrow(
                ax,
                c2_x + 0.5 * c2_w,
                y0 - 0.015,
                c2_x + 0.5 * c2_w,
                c2_y0s[idx + 1] + c2_h + 0.015,
                color=PAL["teal_dark"],
                lw=1.9,
                head_len=0.085,
                head_width=0.075,
                halo=False,
                zorder=6,
            )

    # Connector 2 -> 3
    add_arrow(ax, 8.90, 2.55, 9.24, 2.55, color=PAL["rose_dark"], lw=2.6, head_len=0.13, head_width=0.115)
    ax.text(9.07, 2.78, "$\\hat{y}_i, u_i$", fontsize=11.0, fontweight="bold", color=PAL["rose_dark"], ha="center", va="center")

    # Column 3: benchmark and clinical summary
    _draw_ga_column(
        9.26, 0.08, 4.48, 5.04,
        "3", "Benchmark & Translational Impact",
        PAL["rose_bg"], PAL["rose_edge"], PAL["rose_dark"],
    )

    scpathproto_ece = _load_scpathproto_ece()
    c3_y0s = [3.46, 2.38, 1.30, 0.22]
    c3_h = 0.97
    c3_x, c3_w = 9.38, 4.24
    c3_cards = [
        (
            "Cross-Platform & Rare Transfer",
            "89.8% Acc · 8.7× Pathway SNR",
            "7.3:1 Smart-seq2 · 126 tasks",
            PAL["rose_dark"], PAL["rose_edge"], PAL["rose_bg"], PAL["ink_mid"], "snr",
        ),
        (
            "Causal & De Novo Discovery",
            "−42.6 pp KO · 6.2× Vacuity Ridge",
            "IL3RA · NKG7 · TUBB1 (7 protocols)",
            PAL["iris_dark"], PAL["iris_edge"], PAL["iris_bg"], PAL["amber_dark"], "causal",
        ),
        (
            "Calibrated Open-Set Gate",
            f"0.798 AUROC · {scpathproto_ece:.3f} ECE",
            "Fast CPU (2–12×) · 37 open-set tasks",
            PAL["teal_dark"], PAL["teal_edge"], PAL["teal_bg"], PAL["ink_mid"], "openset",
        ),
        (
            "Clinical COVID-19 Alerting",
            "+5.4 SD Chemotaxis Storm",
            "+34% FCN1+ Vacuity · BALF triage",
            PAL["amber_dark"], PAL["amber_edge"], PAL["amber_bg"], PAL["rose_dark"], "covid",
        ),
    ]

    for y0, (otitle, oline1, oline2, tcol, ecol, pbg, subcol, kind) in zip(c3_y0s, c3_cards):
        ymid = y0 + 0.5 * c3_h
        add_box(ax, c3_x, y0, c3_w, c3_h, fc="#FFFFFF", ec=ecol, lw=0.95, r=0.09, zorder=2)

        # Left quantitative data-driven micro-chart (1.08 x 0.77 in)
        px, py, pw, ph = c3_x + 0.09, y0 + 0.10, 1.08, 0.77
        add_box(ax, px, py, pw, ph, fc=pbg, ec=ecol, lw=0.72, r=0.07, alpha=0.86, zorder=3)

        if kind == "snr":
            # Proportional pathway SNR bars (from mechanistic_summary_stats.csv)
            # HVG=1.220, Unsup=4.018, MPCA=10.579 (8.67x gain)
            base_y = py + 0.12
            ax.plot([px + 0.09, px + pw - 0.09], [base_y, base_y], color="#B8C4D0", lw=0.95, zorder=4)
            bar_vals = [1.220, 4.018, 10.579]
            bar_cols = [PAL["slate_muted"], PAL["navy_mid"], PAL["teal_dark"]]
            bw_s = 0.21
            gap_s = 0.09
            x_start = px + 0.135
            max_h = 0.54
            for bi, (bv, bc) in enumerate(zip(bar_vals, bar_cols)):
                bh = max_h * (bv / 10.579)
                ax.add_patch(
                    Rectangle(
                        (x_start + bi * (bw_s + gap_s), base_y),
                        bw_s,
                        bh,
                        facecolor=bc,
                        edgecolor="none",
                        linewidth=0.0,
                        alpha=0.95,
                        zorder=5,
                    )
                )
            ax.text(
                px + 0.37,
                py + 0.57,
                "8.7×",
                fontsize=9.6,
                fontweight="bold",
                color=PAL["teal_dark"],
                ha="center",
                va="center",
                zorder=6,
            )
        elif kind == "causal":
            # Real PBMC Top-Attributed KO Macro-F1 drop + Counterfactual Swap 6.2x Vacuity Ridge
            ax_l, ax_r = px + 0.11, px + pw - 0.10
            ax_b, ax_t = py + 0.12, py + ph - 0.11
            ax.plot([ax_l, ax_r], [ax_b, ax_b], color="#B8C4D0", lw=0.9, zorder=4)
            ax.plot([ax_l, ax_l], [ax_b, ax_t], color="#B8C4D0", lw=0.9, zorder=4)

            # PBMC top-attributed KO Macro-F1 across K in {0, 5, 10, 20, 40, 80, 150}
            ko_x = np.linspace(0.0, 1.0, 7)
            ko_f1 = np.array([0.9282, 0.8787, 0.8731, 0.8433, 0.7688, 0.6298, 0.4920])
            ko_norm = (ko_f1 - 0.42) / (0.98 - 0.42)
            ax.plot(
                ax_l + (ax_r - ax_l) * ko_x,
                ax_b + (ax_t - ax_b) * ko_norm,
                color=PAL["rose_dark"],
                lw=2.0,
                solid_capstyle="round",
                zorder=6,
            )

            # PBMC counterfactual swap vacuity curve (alpha = 0.0..1.0, peak 6.16x at 0.9)
            sw_x = np.linspace(0.0, 1.0, 11)
            sw_vac = np.array([0.0632, 0.0691, 0.0760, 0.0863, 0.1021, 0.1264, 0.1644, 0.2264, 0.3132, 0.3894, 0.3002])
            sw_norm = (sw_vac - 0.04) / (0.43 - 0.04)
            sx_p = ax_l + (ax_r - ax_l) * sw_x
            sy_p = ax_b + (ax_t - ax_b) * sw_norm
            ax.fill_between(sx_p, ax_b, sy_p, facecolor=PAL["iris_mid"], alpha=0.32, edgecolor="none", zorder=4)
            ax.plot(sx_p, sy_p, color=PAL["iris_dark"], lw=1.8, ls="-.", zorder=5)
            ax.scatter([sx_p[9]], [sy_p[9]], s=22, color=PAL["iris_dark"], edgecolors="#FFFFFF", lw=0.5, zorder=7)
        elif kind == "openset":
            # Known vs. novel score densities with 95% validation cutoff
            base_y = py + 0.12
            gx = np.linspace(0.0, 1.0, 90)
            px_g = px + 0.08 + (pw - 0.16) * gx
            yk = np.exp(-0.5 * ((gx - 0.26) / 0.125) ** 2)
            yn = 0.88 * np.exp(-0.5 * ((gx - 0.74) / 0.145) ** 2)
            sy = 0.51
            ax.fill_between(px_g, base_y, base_y + sy * yk, facecolor=PAL["teal_mid"], alpha=0.46, edgecolor="none", zorder=4)
            ax.plot(px_g, base_y + sy * yk, color=PAL["teal_dark"], lw=1.7, zorder=5)
            ax.fill_between(px_g, base_y, base_y + sy * yn, facecolor=PAL["rose_mid"], alpha=0.48, edgecolor="none", zorder=4)
            ax.plot(px_g, base_y + sy * yn, color=PAL["rose_dark"], lw=1.7, zorder=5)
            ax.plot([px + 0.08, px + pw - 0.08], [base_y, base_y], color="#B8C4D0", lw=0.9, zorder=4)
            xcut = px + 0.08 + (pw - 0.16) * 0.48
            ax.plot([xcut, xcut], [base_y, py + ph - 0.08], color=PAL["ink_dark"], lw=1.45, ls="--", zorder=6)
        else:
            # FABP4+ vs. FCN1+ mini-violins and COVID severity strip
            vy = np.linspace(0.0, 1.0, 50)
            py_v = py + 0.10 + 0.57 * vy
            xv1, xv2 = px + 0.21, px + 0.56
            wv1 = 0.098 * np.exp(-0.5 * ((vy - 0.30) / 0.18) ** 2)
            wv2 = 0.104 * np.exp(-0.5 * ((vy - 0.66) / 0.19) ** 2)
            ax.fill_betweenx(py_v, xv1 - wv1, xv1 + wv1, facecolor=PAL["teal_mid"], alpha=0.68, edgecolor="none", zorder=4)
            ax.plot([xv1 - 0.062, xv1 + 0.062], [py + 0.10 + 0.57 * 0.30] * 2, color="#FFFFFF", lw=1.6, zorder=5)
            ax.fill_betweenx(py_v, xv2 - wv2, xv2 + wv2, facecolor=PAL["rose_mid"], alpha=0.74, edgecolor="none", zorder=4)
            ax.plot([xv2 - 0.062, xv2 + 0.062], [py + 0.10 + 0.57 * 0.66] * 2, color="#FFFFFF", lw=1.6, zorder=5)
            add_arrow(
                ax,
                xv1 + 0.095,
                py + 0.29,
                xv2 - 0.105,
                py + 0.47,
                color=PAL["rose_dark"],
                lw=1.45,
                head_len=0.062,
                head_width=0.056,
                halo=False,
                zorder=6,
            )
            # 3-tile severity escalation strip (HC -> Mod -> Sev)
            for ti, tcol_s in enumerate([PAL["teal_sub"], PAL["amber_soft"], PAL["rose_dark"]]):
                ty_s = py + 0.49 - ti * 0.19
                ax.add_patch(
                    Rectangle((px + 0.78, ty_s), 0.20, 0.16, facecolor=tcol_s, edgecolor="none", linewidth=0.0, zorder=5)
                )

        # Right metrics summary block
        tx = px + pw + 0.15
        ax.text(tx, ymid + 0.24, otitle, fontsize=10.8, fontweight="bold", color=tcol, ha="left", va="center", zorder=5)
        ax.text(tx, ymid - 0.02, oline1, fontsize=9.6, fontweight="bold", color=PAL["ink_dark"], ha="left", va="center", zorder=5)
        ax.text(
            tx,
            ymid - 0.25,
            oline2,
            fontsize=9.5,
            fontweight="bold" if subcol != PAL["ink_mid"] else "normal",
            color=subcol,
            ha="left",
            va="center",
            zorder=5,
        )

    for p in [out_pdf_paper, out_png_paper]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(out_pdf_paper, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png_paper, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    for d in ["figures", "results/figures"]:
        os.makedirs(d, exist_ok=True)
        for src_f, fname in [
            (out_pdf_paper, "graphical_abstract.pdf"),
            (out_png_paper, "graphical_abstract.png"),
        ]:
            dst_f = os.path.join(d, fname)
            if os.path.abspath(src_f) != os.path.abspath(dst_f):
                shutil.copy2(src_f, dst_f)
    plt.close(fig)
    print(f"[OK] Saved graphical abstract to {out_pdf_paper}")


plot_graphical_abstract = generate_graphical_abstract


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate scPathProto architecture figure and/or graphical abstract.")
    parser.add_argument("--only-ga", action="store_true", help="Render only graphical_abstract.pdf/.png (preserve fig0).")
    args = parser.parse_args()
    if not args.only_ga:
        generate_architecture_figure()
    generate_graphical_abstract()

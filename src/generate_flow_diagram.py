"""Render the project flow diagram as a PNG using matplotlib (no external deps).

Output: reports/flow_diagram.png

This is a mandatory deliverable per the EST Project Statements PDF
(page 2: 'A flow diagram must be given with Pre-processing and visualization
steps clearly documented').
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import REPORTS_DIR

# Color palette (matches Amazon_Project / academic-tool dark theme convention)
BG = "#0f172a"
SURFACE = "#1e293b"
TEAL = "#2dd4bf"
YELLOW = "#fbbf24"
GREEN = "#34d399"
INDIGO = "#312e81"
EMERALD = "#064e3b"
TEXT = "#f1f5f9"


def add_box(ax, x, y, w, h, title, body, fill, edge):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(box)
    cx = x + w / 2
    title_y = y + h - 0.22
    body_y = y + h - 0.55
    ax.text(
        cx,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
        zorder=3,
    )
    ax.text(
        cx,
        body_y,
        body,
        ha="center",
        va="center",
        fontsize=8.2,
        color=TEXT,
        zorder=3,
        linespacing=1.35,
    )


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color=TEAL,
        zorder=1,
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 14))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        5,
        13.4,
        "UCS321 EST Project — Malaria RBC Classification",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=YELLOW,
    )
    ax.text(
        5,
        13.0,
        "Statement #4 (Biocon Ltd.) · Pre-processing → CNN → Grad-CAM → Dashboard",
        ha="center",
        fontsize=9.5,
        color=TEAL,
    )

    # Layer 1: data
    add_box(
        ax, 3.0, 11.4, 4.0, 1.2,
        "NIH Malaria Cell Images",
        "27,558 RBC images · 50/50 balanced",
        SURFACE, TEAL,
    )
    # Layer 2: pre-processing
    add_box(
        ax, 3.0, 9.6, 4.0, 1.4,
        "Pre-processing",
        "Resize 128x128 · Normalize [0,1]\nStratified 70/15/15 split",
        SURFACE, TEAL,
    )
    # Layer 3: augmentation
    add_box(
        ax, 3.0, 7.8, 4.0, 1.4,
        "Data Augmentation",
        "H/V flip · rotation · brightness\ncontrast · Gaussian noise",
        SURFACE, TEAL,
    )
    # Layer 4: two models
    add_box(
        ax, 0.3, 5.6, 4.0, 1.6,
        "Baseline CNN",
        "3 conv blocks + dense head\n~500K params · sigmoid output",
        INDIGO, YELLOW,
    )
    add_box(
        ax, 5.7, 5.6, 4.0, 1.6,
        "ResNet50 Transfer",
        "ImageNet weights · frozen base\n+ fine-tune last block",
        INDIGO, YELLOW,
    )
    # Layer 5: training
    add_box(
        ax, 3.0, 3.6, 4.0, 1.5,
        "Training",
        "Adam · binary crossentropy\nEarlyStopping · ModelCheckpoint",
        INDIGO, YELLOW,
    )
    # Layer 6: evaluation
    add_box(
        ax, 0.0, 1.4, 4.7, 1.6,
        "Evaluation Metrics",
        "Accuracy · Precision · Recall · F1\nROC-AUC · PR curve · Confusion matrix",
        EMERALD, GREEN,
    )
    # Layer 7: interpretability
    add_box(
        ax, 5.3, 1.4, 4.7, 1.6,
        "Grad-CAM Interpretability",
        "Heatmap overlays for\nTrue/False · Pos/Neg samples",
        EMERALD, GREEN,
    )
    # Layer 8: dashboard
    add_box(
        ax, 3.0, -0.7, 4.0, 1.5,
        "Streamlit Dashboard",
        "Upload image → prediction\n+ Grad-CAM heatmap",
        EMERALD, GREEN,
    )

    # Arrows
    add_arrow(ax, 5, 11.4, 5, 11.0)
    add_arrow(ax, 5, 9.6, 5, 9.2)
    add_arrow(ax, 5, 7.8, 2.3, 7.2)
    add_arrow(ax, 5, 7.8, 7.7, 7.2)
    add_arrow(ax, 2.3, 5.6, 4.5, 5.1)
    add_arrow(ax, 7.7, 5.6, 5.5, 5.1)
    add_arrow(ax, 5, 3.6, 2.5, 3.0)
    add_arrow(ax, 5, 3.6, 7.5, 3.0)
    add_arrow(ax, 2.5, 1.4, 4.2, 0.8)
    add_arrow(ax, 7.5, 1.4, 5.8, 0.8)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=SURFACE, edgecolor=TEAL, label="Data / Pre-processing"),
        mpatches.Patch(facecolor=INDIGO, edgecolor=YELLOW, label="Modeling / Training"),
        mpatches.Patch(facecolor=EMERALD, edgecolor=GREEN, label="Evaluation / Output"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        labelcolor=TEXT,
        fontsize=9,
    )

    out_path = REPORTS_DIR / "flow_diagram.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved flow diagram: {out_path}")


if __name__ == "__main__":
    main()

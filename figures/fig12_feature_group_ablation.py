#!/usr/bin/env python3
"""
Fig. 12 -- feature-group ablation (spectral only / +temporal / +spatial / full model), random
resampling vs. HUC4 station holdout, 35 runs each, mean R2 +/- SD.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

# Produced by validation/ablation_random_split_mc35.py
RANDOM_FILE = (
    RESULTS_DIR
    / "xgb_monte_carlo_35"
    / "feature_group_ablation"
    / "MC35_ablation_summary.csv"
)

# Produced by validation/ablation_huc4_split_35seed.py
HUC4_FILE = (
    RESULTS_DIR
    / "spatial_validation_diagnostics"
    / "HUC4_feature_group_ablation_35seeds"
    / "HUC4_ablation_35seed_summary.csv"
)

OUT_DIR = RESULTS_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure12_feature_group_ablation_R2_FINAL.png"
OUT_PDF = OUT_DIR / "Figure12_feature_group_ablation_R2_FINAL.pdf"
OUT_SVG = OUT_DIR / "Figure12_feature_group_ablation_R2_FINAL.svg"


# =============================================================================
# LOAD
# =============================================================================

random_df = pd.read_csv(RANDOM_FILE)
huc4_df = pd.read_csv(HUC4_FILE)

order = [
    "Spectral only",
    "Spectral + temporal",
    "Spectral + spatial",
    "Full model",
]

random_df = (
    random_df
    .set_index("configuration")
    .loc[order]
    .reset_index()
)

huc4_df = (
    huc4_df
    .set_index("configuration")
    .loc[order]
    .reset_index()
)


# =============================================================================
# VALUES
# =============================================================================

labels = [
    "Spectral\nonly",
    "Spectral +\ntemporal",
    "Spectral +\nspatial",
    "Full\nmodel",
]

x = np.arange(4)

random_r2 = random_df["r2_mean"].to_numpy()
random_sd = random_df["r2_sd"].to_numpy()

huc4_r2 = huc4_df["r2_mean"].to_numpy()
huc4_sd = huc4_df["r2_sd"].to_numpy()


def relative_improvement(values):
    baseline = values[0]
    return 100.0 * (values - baseline) / baseline


random_rel = relative_improvement(random_r2)
huc4_rel = relative_improvement(huc4_r2)


# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14.0, 6.8),
    sharey=True,
)

# Restrained monochromatic progression
BAR_COLORS = [
    "#BDD7E7",
    "#8FBFDC",
    "#579AC7",
    "#2878B5",
]

ACCENT = "#B22222"
BAR_WIDTH = 0.60

YMIN = 0.0
YMAX = 0.82


# =============================================================================
# PANEL FUNCTION
# =============================================================================

def draw_panel(
    ax,
    values,
    errors,
    relative,
    title,
):

    baseline = values[0]

    # -------------------------------------------------------------------------
    # Dashed spectral-only reference.
    # Drawn FIRST so it remains behind all bars.
    # -------------------------------------------------------------------------

    ax.axhline(
        baseline,
        color=ACCENT,
        linestyle="--",
        linewidth=1.25,
        alpha=0.75,
        zorder=1,
    )

    # -------------------------------------------------------------------------
    # Bars
    # -------------------------------------------------------------------------

    bars = ax.bar(
        x,
        values,
        width=BAR_WIDTH,
        yerr=errors,
        capsize=5,
        color=BAR_COLORS,
        edgecolor="black",
        linewidth=0.75,
        zorder=3,
        error_kw={
            "elinewidth": 1.5,
            "capthick": 1.5,
            "zorder": 4,
        },
    )

    # -------------------------------------------------------------------------
    # Panel title
    # -------------------------------------------------------------------------

    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=42,
    )

    # -------------------------------------------------------------------------
    # One explanation of red percentages
    # -------------------------------------------------------------------------

    ax.text(
        0.5,
        1.025,
        r"Relative $R^2$ improvement over spectral-only",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
    )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    for i, (bar, value, error) in enumerate(
        zip(bars, values, errors)
    ):

        xpos = (
            bar.get_x()
            + bar.get_width() / 2
        )

        y_top = (
            value
            + error
        )

        # -------------------------------------------------------------
        # Spectral-only reference bar
        # -------------------------------------------------------------

        if i == 0:

            ax.text(
                xpos,
                y_top + 0.012,
                f"{value:.3f}\n(baseline)",
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color="black",
                linespacing=1.05,
                zorder=6,
            )

        # -------------------------------------------------------------
        # Enhanced configurations
        # -------------------------------------------------------------

        else:

            # Mean R²
            ax.text(
                xpos,
                y_top + 0.010,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black",
                zorder=6,
            )

            # Relative improvement
            ax.text(
                xpos,
                y_top + 0.050,
                f"+{relative[i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=ACCENT,
                zorder=6,
            )

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        fontsize=12.5,
    )

    ax.set_xlim(
        -0.52,
        3.48,
    )

    ax.set_ylim(
        YMIN,
        YMAX,
    )

    ax.set_ylabel(
        r"$R^2$",
        fontsize=16,
    )

    # Important: y tick labels on BOTH panels
    ax.tick_params(
        axis="y",
        labelsize=12.5,
        labelleft=True,
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.75,
        alpha=0.35,
        zorder=0,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# DRAW PANELS
# =============================================================================

draw_panel(
    axes[0],
    random_r2,
    random_sd,
    random_rel,
    "(a) Random resampling (35 runs)",
)

draw_panel(
    axes[1],
    huc4_r2,
    huc4_sd,
    huc4_rel,
    "(b) HUC4 station holdout (35 runs)",
)

# Force numeric y labels on panel b despite sharey=True
axes[1].tick_params(
    axis="y",
    labelleft=True,
)


# =============================================================================
# FOOTNOTE
# =============================================================================

fig.text(
    0.5,
    0.018,
    r"Bars show mean $R^2$ ± SD across repeated validation runs.",
    ha="center",
    va="bottom",
    fontsize=11.5,
)


# =============================================================================
# SAVE
# =============================================================================

plt.tight_layout(
    rect=[0, 0.055, 1, 0.98],
    w_pad=3.5,
)

plt.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    OUT_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    OUT_SVG,
    bbox_inches="tight",
    facecolor="white",
)

plt.close()


# =============================================================================
# PRINT RESULTS
# =============================================================================

print("=" * 100)
print("FINAL FIGURE 11 CREATED")
print("=" * 100)

print("\nRANDOM RESAMPLING")
for name, value, rel in zip(order, random_r2, random_rel):

    if name == "Spectral only":
        print(
            f"{name:22s}: "
            f"R² = {value:.4f} (baseline)"
        )
    else:
        print(
            f"{name:22s}: "
            f"R² = {value:.4f} | "
            f"relative improvement = +{rel:.1f}%"
        )


print("\nHUC4 STATION HOLDOUT")
for name, value, rel in zip(order, huc4_r2, huc4_rel):

    if name == "Spectral only":
        print(
            f"{name:22s}: "
            f"R² = {value:.4f} (baseline)"
        )
    else:
        print(
            f"{name:22s}: "
            f"R² = {value:.4f} | "
            f"relative improvement = +{rel:.1f}%"
        )


print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_SVG)

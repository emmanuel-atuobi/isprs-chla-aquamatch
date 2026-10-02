#!/usr/bin/env python3
"""
Fig. 10 -- concentration-dependent prediction bias and squared-error share, random holdout vs.
HUC4 station holdout. Reads the CSVs written by validation/discussion_error_diagnostics.py.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))
# Written by validation/discussion_error_diagnostics.py
DIAGNOSTICS_DIR = RESULTS_DIR / "discussion_error_diagnostics"

HOLDOUT_FILE = (
    DIAGNOSTICS_DIR / "fixed_holdout_error_by_chla_bin.csv"
)

HUC4_FILE = (
    DIAGNOSTICS_DIR / "huc4_error_by_chla_bin_summary.csv"
)

OUT_DIR = DIAGNOSTICS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUT_DIR / "Fig10_concentration_dependent_errors.png"
PDF_OUT = OUT_DIR / "Fig10_concentration_dependent_errors.pdf"
TIFF_OUT = OUT_DIR / "Fig10_concentration_dependent_errors.tif"


# =============================================================================
# LOAD DATA
# =============================================================================

hold = pd.read_csv(HOLDOUT_FILE)
huc = pd.read_csv(HUC4_FILE)

BIN_ORDER = [
    "<2",
    "2–5",
    "5–10",
    "10–20",
    "20–50",
    "50–100",
    "≥100",
]

hold["chl_bin"] = pd.Categorical(
    hold["chl_bin"],
    categories=BIN_ORDER,
    ordered=True,
)

huc["chl_bin"] = pd.Categorical(
    huc["chl_bin"],
    categories=BIN_ORDER,
    ordered=True,
)

hold = (
    hold
    .sort_values("chl_bin")
    .reset_index(drop=True)
)

huc = (
    huc
    .sort_values("chl_bin")
    .reset_index(drop=True)
)


# =============================================================================
# VALUES
# =============================================================================

x = np.arange(len(BIN_ORDER))
width = 0.34

# Panel A
hold_bias = hold["bias"].to_numpy(dtype=float)
huc_bias = huc["bias_mean"].to_numpy(dtype=float)
huc_bias_sd = huc["bias_sd"].to_numpy(dtype=float)

# Panel B
hold_sse = hold[
    "share_total_squared_error_pct"
].to_numpy(dtype=float)

huc_sse = huc[
    "share_total_squared_error_pct_mean"
].to_numpy(dtype=float)

huc_sse_sd = huc[
    "share_total_squared_error_pct_sd"
].to_numpy(dtype=float)

hold_n = hold["n"].to_numpy(dtype=int)

tick_labels = [
    f"{label}\n$n$={n}"
    for label, n in zip(BIN_ORDER, hold_n)
]


# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
# FIGURE
# =============================================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(15.5, 6.8),
)

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.18,
    top=0.90,
    wspace=0.20,
)


# =============================================================================
# PANEL A — PREDICTION BIAS
# =============================================================================

ax1.bar(
    x - width / 2,
    hold_bias,
    width,
    label="Random holdout (seed 42)",
)

ax1.bar(
    x + width / 2,
    huc_bias,
    width,
    yerr=huc_bias_sd,
    capsize=4,
    error_kw={
        "elinewidth": 1.3,
        "capthick": 1.3,
    },
    label="HUC4 station holdout (35-run mean)",
)

ax1.axhline(
    0,
    linestyle="--",
    linewidth=1.4,
)

ax1.set_title(
    "(a) Prediction bias",
    pad=12,
)

ax1.set_ylabel(
    "Mean prediction bias (µg/L)"
)

ax1.set_xlabel(
    "Observed Chl-a range (µg/L)"
)

ax1.set_xticks(x)
ax1.set_xticklabels(
    tick_labels,
    linespacing=1.25,
)

bias_min = min(
    np.nanmin(hold_bias),
    np.nanmin(huc_bias - huc_bias_sd),
)

bias_max = max(
    np.nanmax(hold_bias),
    np.nanmax(huc_bias + huc_bias_sd),
)

# Extra top headroom so the legend stays well above the bars and error bars
ax1.set_ylim(
    bias_min - 4,
    bias_max + 11,
)

# Pushed fully to the top-right corner with 0 right-padding
ax1.legend(
    loc="upper right",
    bbox_to_anchor=(1.0, 0.99),
    borderaxespad=0.0,
    frameon=False,
)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# =============================================================================
# PANEL B — CONTRIBUTION TO TOTAL SQUARED ERROR
# =============================================================================

ax2.bar(
    x - width / 2,
    hold_sse,
    width,
    label="Random holdout (seed 42)",
)

ax2.bar(
    x + width / 2,
    huc_sse,
    width,
    yerr=huc_sse_sd,
    capsize=4,
    error_kw={
        "elinewidth": 1.3,
        "capthick": 1.3,
    },
    label="HUC4 station holdout (35-run mean)",
)

ax2.set_title(
    "(b) Contribution to total squared error",
    pad=12,
)

ax2.set_ylabel(
    "Share of total squared error (%)"
)

ax2.set_xlabel(
    "Observed Chl-a range (µg/L)"
)

ax2.set_xticks(x)
ax2.set_xticklabels(
    tick_labels,
    linespacing=1.25,
)

ax2.legend(
    loc="upper left",
    frameon=False,
)

ymax = max(
    np.nanmax(hold_sse),
    np.nanmax(huc_sse + huc_sse_sd),
)

ax2.set_ylim(
    0,
    ymax * 1.12,
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


# =============================================================================
# SAVE
# =============================================================================

fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
)

fig.savefig(
    TIFF_OUT,
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# CONFIRM
# =============================================================================

print("\nSaved:")
print(PNG_OUT)
print(PDF_OUT)
print(TIFF_OUT)

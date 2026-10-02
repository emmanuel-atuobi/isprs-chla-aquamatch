#!/usr/bin/env python3
"""
Fig. 1 -- Chl-a histogram + ECDF, with quartile annotations and summary stats box.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

INPUT = DATA_DIR / "raw" / "aquamatch_sitesr_joined.csv"
OUT = RESULTS_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT, low_memory=False).rename(columns={
    "harmonized_value": "chl_a",
    "med_Blue": "blue",
    "med_Green": "green",
    "med_Red": "red",
    "med_Nir": "nir",
    "med_Swir1": "swir1",
    "med_Swir2": "swir2",
    "lon": "long",
})

numeric = [
    "chl_a",
    "harmonized_value_cv",
    "harmonized_row_count",
    "depth_flag",
    "harmonized_discrete_depth_value",
    "harmonized_top_depth_value",
    "harmonized_bottom_depth_value",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "pCount_dswe1",
    "timediff",
    "lat",
    "long",
]

for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# FINAL QUALITY FILTERS
# ============================================================

df = df[df["ResolvedMonitoringLocationTypeName"] == "Lake, Reservoir, Impoundment"].copy()
df = df[df["mission"].isin(["LT05", "LE07", "LC08", "LC09"])].copy()
df = df[np.isfinite(df["chl_a"]) & (df["chl_a"] > 0) & (df["chl_a"] <= 200)].copy()
df = df[df["mdl_flag"] == 0].copy()
df = df[(df["harmonized_row_count"] == 1) | (df["harmonized_value_cv"] <= 0.5)].copy()

discrete = (
    (df["depth_flag"] == 1)
    & (df["harmonized_discrete_depth_value"] >= 0)
    & (df["harmonized_discrete_depth_value"] <= 2)
)

integrated = (
    (df["depth_flag"] == 2)
    & (df["harmonized_top_depth_value"] >= 0)
    & (df["harmonized_top_depth_value"] <= 0.5)
    & (df["harmonized_bottom_depth_value"] >= df["harmonized_top_depth_value"])
    & (df["harmonized_bottom_depth_value"] <= 2)
)

df = df[discrete | integrated].copy()

for c in ["blue", "green", "red", "nir", "swir1", "swir2"]:
    df = df[np.isfinite(df[c]) & (df[c] > 0)].copy()

df = df[df["pCount_dswe1"] >= 8].copy()
df["timediff"] = df["timediff"].abs()
df = df[np.isfinite(df["timediff"]) & (df["timediff"] <= 2)].copy()
df = df.reset_index(drop=True)

# ============================================================
# STATISTICAL METRICS
# ============================================================

assert len(df) == 14125, f"Expected N = 14,125; got {len(df):,}"

chl = df["chl_a"].to_numpy()
N = len(chl)
chl_min = np.min(chl)
chl_max = np.max(chl)
chl_med = np.median(chl)
chl_mean = np.mean(chl)

q1 = np.quantile(chl, 0.25)
q3 = np.quantile(chl, 0.75)
iqr = q3 - q1

# ECDF calculations
chl_sorted = np.sort(chl)
ecdf = np.arange(1, N + 1) / N * 100

# ============================================================
# PUBLICATION FIGURE CONFIGURATION
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10.0,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "xtick.minor.width": 0.55,
    "ytick.minor.width": 0.55,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
})

# Color palette
HIST_COLOR = "#2C5E8A"   # Slate blue
ECDF_COLOR = "#B83A24"   # Crimson / terracotta
GUIDE_COLOR = "#555555"  # Clean dark grey for reference dashes
GRID_COLOR = "#ECEFF1"

fig, ax1 = plt.subplots(figsize=(7.8, 4.8))

# ============================================================
# HISTOGRAM (AXIS 1 - LEFT)
# ============================================================

BIN_WIDTH = 2.0
bins = np.arange(0, 200 + BIN_WIDTH, BIN_WIDTH)

counts, _, _ = ax1.hist(
    chl,
    bins=bins,
    color=HIST_COLOR,
    alpha=0.72,
    edgecolor="white",
    linewidth=0.25,
    zorder=2
)

ax1.set_xlim(0, 200)
ax1.set_ylim(0, counts.max() * 1.12)

# Black titles and tick labels
ax1.set_xlabel("Chlorophyll-$a$ (µg L$^{-1}$)", color="black", fontweight="medium")
ax1.set_ylabel("Number of Observations", color="black", fontweight="medium")
ax1.tick_params(axis="both", colors="black")

ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(400))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(100))

ax1.grid(True, which="major", axis="both", linestyle="--", linewidth=0.55, color=GRID_COLOR, zorder=0)

# ============================================================
# ECDF (AXIS 2 - RIGHT)
# ============================================================

ax2 = ax1.twinx()

ax2.step(
    chl_sorted,
    ecdf,
    where="post",
    color=ECDF_COLOR,
    linewidth=2.0,
    zorder=4
)

ax2.set_ylim(0, 100)

# Black titles and tick labels
ax2.set_ylabel("Cumulative Observations (%)", color="black", fontweight="medium")
ax2.tick_params(axis="y", colors="black")

ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5))

# ============================================================
# QUARTILE ANNOTATIONS & THICKENED SHORT DASH GUIDES
# ============================================================

quartile_points = [
    (q1, 25, f"$Q_1$ = {q1:.2f} µg L$^{{-1}}$ (25%)", (32, 22)),
    (chl_med, 50, f"Median = {chl_med:.2f} µg L$^{{-1}}$ (50%)", (40, 48)),
    (q3, 75, f"$Q_3$ = {q3:.2f} µg L$^{{-1}}$ (75%)", (50, 73)),
]

for val, pct, label, text_pos in quartile_points:
    # Thickened short-dash guideline to axes
    ax2.plot(
        [0, val, val],
        [pct, pct, 0],
        color=GUIDE_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=1.35,
        alpha=0.85,
        zorder=3
    )
    
    # Point on curve
    ax2.plot(
        val, pct,
        marker="o",
        markersize=5.4,
        markerfacecolor=ECDF_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.1,
        zorder=6
    )
    
    # Annotation arrow
    ax2.annotate(
        label,
        xy=(val, pct),
        xytext=text_pos,
        arrowprops=dict(
            arrowstyle="->",
            color=ECDF_COLOR,
            lw=0.95,
            shrinkA=3,
            shrinkB=3
        ),
        fontsize=9.2,
        fontweight="medium",
        color=ECDF_COLOR,
        va="center",
        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=7
    )

# Layer axes properly
ax1.set_zorder(1)
ax1.patch.set_visible(False)
ax2.set_zorder(2)

# ============================================================
# STATISTICAL SUMMARY BOX (TOP RIGHT)
# ============================================================

stats_text = (
    f"$N$ = {N:,}\n"
    f"Median = {chl_med:.2f} µg L$^{{-1}}$\n"
    f"IQR = {iqr:.2f} µg L$^{{-1}}$\n"
    f"$Q_1$–$Q_3$ = [{q1:.2f}, {q3:.2f}]\n"
    f"Mean = {chl_mean:.2f} µg L$^{{-1}}$\n"
    f"Range = [{chl_min:.2f}, {chl_max:.2f}]"
)

bbox_props = dict(
    boxstyle="square,pad=0.70",
    facecolor="white",
    edgecolor="#B0BEC5",
    linewidth=0.85,
    alpha=0.96
)

ax1.text(
    0.96,
    0.93,
    stats_text,
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=9.5,
    linespacing=1.45,
    bbox=bbox_props,
    zorder=8
)

# ============================================================
# EXPORT
# ============================================================

plt.tight_layout()

png = OUT / "Fig1_Chla_distribution_ECDF_journal.png"
pdf = OUT / "Fig1_Chla_distribution_ECDF_journal.pdf"

plt.savefig(png, dpi=600, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print(f"Figure saved successfully in publication format:\nPNG: {png}\nPDF: {pdf}")
print("=" * 60)

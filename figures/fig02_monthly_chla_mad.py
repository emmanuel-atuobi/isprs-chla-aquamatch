#!/usr/bin/env python3
"""
Fig. 2 -- monthly Chl-a 75th-percentile with median-absolute-deviation (MAD) error bars.
"""

import numpy as np
import pandas as pd
import calendar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

# 1. Lakes / reservoirs / impoundments
df = df[
    df["ResolvedMonitoringLocationTypeName"]
    == "Lake, Reservoir, Impoundment"
].copy()

# 2. Landsat 5, 7, 8, 9
df = df[
    df["mission"].isin(["LT05", "LE07", "LC08", "LC09"])
].copy()

# 3. Chl-a > 0 and <= 200 µg/L
df = df[
    np.isfinite(df["chl_a"])
    & (df["chl_a"] > 0)
    & (df["chl_a"] <= 200)
].copy()

# 4. Detection-limit screening
df = df[
    df["mdl_flag"] == 0
].copy()

# 5. Replicate agreement
df = df[
    (df["harmonized_row_count"] == 1)
    | (df["harmonized_value_cv"] <= 0.5)
].copy()

# 6. Near-surface sampling depth
discrete = (
    (df["depth_flag"] == 1)
    & (df["harmonized_discrete_depth_value"] >= 0)
    & (df["harmonized_discrete_depth_value"] <= 2)
)

integrated = (
    (df["depth_flag"] == 2)
    & (df["harmonized_top_depth_value"] >= 0)
    & (df["harmonized_top_depth_value"] <= 0.5)
    & (
        df["harmonized_bottom_depth_value"]
        >= df["harmonized_top_depth_value"]
    )
    & (df["harmonized_bottom_depth_value"] <= 2)
)

df = df[
    discrete | integrated
].copy()

# 7. All six SR bands finite and positive
for c in [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2"
]:
    df = df[
        np.isfinite(df[c])
        & (df[c] > 0)
    ].copy()

# 8. At least 8 confident-water pixels
df = df[
    df["pCount_dswe1"] >= 8
].copy()

# 9. Absolute satellite–in-situ temporal separation <= 2 h
df["timediff"] = df["timediff"].abs()

df = df[
    np.isfinite(df["timediff"])
    & (df["timediff"] <= 2)
].copy()

df = df.reset_index(drop=True)

# ============================================================
# VERIFY FINAL DATASET
# ============================================================

print("=" * 80)
print("FINAL QUALITY-FILTERED DATASET")
print("=" * 80)
print(f"N = {len(df):,}")

assert len(df) == 14125, (
    f"Expected N = 14,125 but obtained {len(df):,}"
)

# ============================================================
# DATE / MONTH / YEAR
# ============================================================

df["sample_date"] = pd.to_datetime(
    df["ActivityStartDate"],
    errors="coerce"
)

df = df[
    df["sample_date"].notna()
].copy()

df["year"] = df["sample_date"].dt.year
df["month"] = df["sample_date"].dt.month

months = np.arange(1, 13)
labels = [calendar.month_abbr[m] for m in months]

# ============================================================
# MONTHLY SAMPLE COUNTS
# ============================================================

counts = (
    df["month"]
    .value_counts()
    .reindex(months, fill_value=0)
)

assert counts.sum() == 14125

# ============================================================
# YEAR × MONTH 75TH PERCENTILE
# ============================================================

year_month_q75 = (
    df.groupby(["year", "month"])["chl_a"]
    .quantile(0.75)
    .reset_index(name="year_month_p75")
)

# ============================================================
# MONTHLY SUMMARY:
# point = median across years
# error bars = raw MAD across years
# ============================================================

rows = []

for month in months:

    values = year_month_q75.loc[
        year_month_q75["month"] == month,
        "year_month_p75"
    ].dropna().to_numpy()

    med = np.median(values)
    raw_mad = np.median(np.abs(values - med))

    rows.append({
        "month": month,
        "month_name": labels[month - 1],
        "n_samples": int(counts.loc[month]),
        "n_years": len(values),
        "p75_median": med,
        "raw_mad": raw_mad,
        "raw_mad_low": med - raw_mad,
        "raw_mad_high": med + raw_mad,
        "p75_min": np.min(values),
        "p75_max": np.max(values),
    })

summary = pd.DataFrame(rows)

print("\n" + "=" * 110)
print("MONTHLY 75TH-PERCENTILE SUMMARY WITH RAW MAD")
print("=" * 110)
print(summary.round(2).to_string(index=False))

summary_file = OUT / "monthly_P75_rawMAD_summary.csv"
summary.to_csv(summary_file, index=False)

print("\nSaved summary:")
print(summary_file)

# ============================================================
# FIGURE
# ============================================================

fig, ax1 = plt.subplots(figsize=(10, 6))

# ------------------------------------------------------------
# LEFT AXIS: sample counts
# ------------------------------------------------------------

bars = ax1.bar(
    months,
    summary["n_samples"],
    width=0.72,
    alpha=0.30,
    edgecolor="none",
    zorder=1
)

ax1.set_xlim(0.5, 12.5)
ax1.set_xticks(months, labels)
ax1.set_xlabel("Month", fontsize=12)
ax1.set_ylabel("Number of samples", fontsize=12)

count_max = summary["n_samples"].max()
ax1.set_ylim(0, count_max * 1.15)

for x, value in zip(months, summary["n_samples"]):
    ax1.text(
        x,
        value + count_max * 0.015,
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        zorder=5
    )

ax1.grid(
    axis="y",
    alpha=0.14,
    linewidth=0.6,
    zorder=0
)

# ------------------------------------------------------------
# RIGHT AXIS: Chl-a
# ------------------------------------------------------------

ax2 = ax1.twinx()

y = summary["p75_median"].to_numpy()
mad = summary["raw_mad"].to_numpy()

yerr = np.vstack([mad, mad])

# MAD error bars only
ax2.errorbar(
    months,
    y,
    yerr=yerr,
    fmt="none",
    ecolor="tab:orange",
    capsize=3.5,
    capthick=1.0,
    elinewidth=1.0,
    zorder=4
)

# Actual monthly P75 points + short-dashed connecting line
line, = ax2.plot(
    months,
    y,
    color="tab:orange",
    linestyle=(0, (4, 2)),
    linewidth=2.4,
    marker="o",
    markersize=6,
    zorder=5
)

ax2.set_ylabel(
    "Chlorophyll-a (µg L$^{-1}$)",
    fontsize=12
)

# Axis limit based on median + MAD
chl_ymax = np.ceil(((summary["raw_mad_high"].max()) * 1.06) / 5) * 5
ax2.set_ylim(0, chl_ymax)
ax2.set_yticks(np.arange(0, chl_ymax + 0.1, 5))

print(f"\nChl-a axis: 0 to {chl_ymax:.0f} µg/L")

# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------

sample_handle = Patch(
    facecolor=bars.patches[0].get_facecolor(),
    edgecolor="none",
    label="Samples (count)"
)

p75_handle = Line2D(
    [0], [0],
    color="tab:orange",
    linestyle=(0, (4, 2)),
    linewidth=2.4,
    marker="o",
    markersize=6,
    label="75th-percentile Chl-a"
)

mad_handle = ax2.errorbar(
    [np.nan],
    [np.nan],
    yerr=[[1], [1]],
    fmt="none",
    ecolor="tab:orange",
    capsize=3.5,
    elinewidth=1.0,
    label="MAD"
)

ax1.legend(
    handles=[sample_handle, p75_handle, mad_handle],
    labels=["Samples (count)", "75th-percentile Chl-a", "MAD"],
    loc="upper left",
    frameon=False,
    fontsize=10.5
)

# ------------------------------------------------------------
# CLEAN APPEARANCE
# ------------------------------------------------------------

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax1.tick_params(axis="both", labelsize=10.5)
ax2.tick_params(axis="y", labelsize=10.5)

plt.tight_layout()

# ============================================================
# EXPORT
# ============================================================

png = OUT / "Fig2_monthly_Chla_rawMAD.png"
pdf = OUT / "Fig2_monthly_Chla_rawMAD.pdf"

plt.savefig(
    png,
    dpi=500,
    bbox_inches="tight"
)

plt.savefig(
    pdf,
    bbox_inches="tight"
)

plt.show()

print("\nSaved figure:")
print(png)
print(pdf)

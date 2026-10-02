#!/usr/bin/env python3
"""
Fig. 4 -- (a) cyclical sin/cos day-of-year encoding and (b) polar plot of monthly
75th-percentile Chl-a.
"""
from pathlib import Path
import calendar

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

import os

# ============================================================
# PATHS
# ============================================================
DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

INPUT = DATA_DIR / "raw" / "aquamatch_sitesr_joined.csv"

OUTDIR = RESULTS_DIR / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "Fig4_temporal_encoding_polar_chla.png"
OUT_PDF = OUTDIR / "Fig4_temporal_encoding_polar_chla.pdf"
OUT_SVG = OUTDIR / "Fig4_temporal_encoding_polar_chla.svg"
OUT_CSV = OUTDIR / "Fig4_monthly_p75_summary.csv"


# ============================================================
# HELPER
# ============================================================
def find_first_present(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"\nCould not find column for: {label}\n"
        f"Tried:\n{candidates}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# LOAD RAW AQUAMATCH + siteSR JOIN
# ============================================================
print("=" * 90)
print("LOADING RAW AQUAMATCH + siteSR DATA")
print("=" * 90)

df = pd.read_csv(INPUT, low_memory=False)

df = df.rename(columns={
    "harmonized_value": "chl_a",
    "med_Blue": "blue",
    "med_Green": "green",
    "med_Red": "red",
    "med_Nir": "nir",
    "med_Swir1": "swir1",
    "med_Swir2": "swir2",
    "lon": "long",
})

print(f"Raw rows: {len(df):,}")


# ============================================================
# FIELD-SAMPLING DATE/TIME
# ============================================================
date_col = find_first_present(
    df,
    [
        "harmonized_utc",
        "harmonized_datetime",
        "sample_datetime",
        "sample_date",
        "date",
        "datetime",
        "ActivityStartDate",
        "activity_start_date",
        "Date",
    ],
    "field sampling date/time"
)

print(f"Date/time column used: {date_col}")


# ============================================================
# TYPE CLEANING
# ============================================================
numeric_cols = [
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

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce",
    utc=True
)


# ============================================================
# FINAL QUALITY FILTERING
# ============================================================
print("\n" + "=" * 90)
print("APPLYING FINAL QUALITY FILTERS")
print("=" * 90)


# 1. Lakes / reservoirs / impoundments
df = df[
    df["ResolvedMonitoringLocationTypeName"]
    == "Lake, Reservoir, Impoundment"
].copy()

print(f"After waterbody type: {len(df):,}")


# 2. Landsat 5, 7, 8, 9
df = df[
    df["mission"].isin(["LT05", "LE07", "LC08", "LC09"])
].copy()

print(f"After mission filter: {len(df):,}")


# 3. Chl-a >0 and <=200 µg/L
df = df[
    np.isfinite(df["chl_a"])
    & (df["chl_a"] > 0)
    & (df["chl_a"] <= 200)
].copy()

print(f"After Chl-a range: {len(df):,}")


# 4. Measurement/model flag
df = df[
    df["mdl_flag"] == 0
].copy()

print(f"After mdl_flag: {len(df):,}")


# 5. Replicate variability
df = df[
    (df["harmonized_row_count"] == 1)
    | (df["harmonized_value_cv"] <= 0.5)
].copy()

print(f"After replicate CV: {len(df):,}")


# 6. Near-surface depth, 0–2 m
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

print(f"After depth filter: {len(df):,}")


# 7. All six Landsat SR bands finite and positive
for col in [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]:
    df = df[
        np.isfinite(df[col])
        & (df[col] > 0)
    ].copy()

print(f"After positive-band filter: {len(df):,}")


# 8. Minimum DSWE water pixels
df = df[
    np.isfinite(df["pCount_dswe1"])
    & (df["pCount_dswe1"] >= 8)
].copy()

print(f"After DSWE pixel filter: {len(df):,}")


# 9. Absolute satellite–in-situ temporal separation <=2 h
df["timediff"] = df["timediff"].abs()

df = df[
    np.isfinite(df["timediff"])
    & (df["timediff"] <= 2)
].copy()

print(f"After <=2 h temporal filter: {len(df):,}")


# Valid date required
df = df[
    df[date_col].notna()
].copy()

df = df.reset_index(drop=True)


print("\n" + "=" * 90)
print("FINAL DATASET")
print("=" * 90)
print(f"Final N = {len(df):,}")

if len(df) != 14125:
    print(
        f"WARNING: expected final quality-filtered N = 14,125, "
        f"but obtained {len(df):,}."
    )


# ============================================================
# DATE FIELDS
# ============================================================
df["year"] = df[date_col].dt.year.astype(int)
df["month"] = df[date_col].dt.month.astype(int)


# ============================================================
# MONTH / DOY REFERENCE STRUCTURE
# ============================================================
months = np.arange(1, 13)

month_names = [
    calendar.month_abbr[m]
    for m in months
]

month_starts = []
month_ends = []
month_centers = []

current_doy = 1

for month in months:

    n_days = calendar.monthrange(
        2021,
        month
    )[1]

    start = current_doy
    end = current_doy + n_days - 1
    center = start + (n_days - 1) / 2

    month_starts.append(start)
    month_ends.append(end)
    month_centers.append(center)

    current_doy += n_days


month_starts = np.asarray(month_starts)
month_ends = np.asarray(month_ends)
month_centers = np.asarray(month_centers)


polar_labels = [
    f"{month_names[i]}\n"
    f"(DOY {month_starts[i]}-{month_ends[i]})"
    for i in range(12)
]


# ============================================================
# PANEL A:
# CYCLICAL ENCODING OF DAY OF YEAR
# ============================================================
doy = np.arange(1, 366)

sin_doy = np.sin(
    2 * np.pi * (doy - 1) / 365.0
)

cos_doy = np.cos(
    2 * np.pi * (doy - 1) / 365.0
)


# ============================================================
# PANEL B:
# MONTHLY 75TH-PERCENTILE Chl-a
#
# P75 calculated separately within each year/month,
# followed by the median across years.
# ============================================================
year_month_p75 = (
    df
    .groupby(["year", "month"])["chl_a"]
    .quantile(0.75)
    .unstack("year")
    .reindex(index=months)
)

monthly_p75 = year_month_p75.median(axis=1)

n_years = year_month_p75.notna().sum(axis=1)


summary = pd.DataFrame({
    "month": months,
    "month_name": month_names,
    "doy_start": month_starts,
    "doy_end": month_ends,
    "n_years": n_years.values,
    "monthly_p75_median_across_years": monthly_p75.values,
})

summary.to_csv(
    OUT_CSV,
    index=False
)


print("\n" + "=" * 90)
print("MONTHLY VALUES USED IN POLAR PANEL")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# INTERPOLATE MONTHLY VALUES THROUGH ANNUAL CYCLE
# ============================================================
x_month = month_centers.astype(float)
y_month = monthly_p75.to_numpy(dtype=float)

# Periodic extension prevents Jan/Dec edge artifact
x_extended = np.concatenate([
    x_month - 365,
    x_month,
    x_month + 365
])

y_extended = np.concatenate([
    y_month,
    y_month,
    y_month
])

x_daily = np.arange(1, 366)

y_interpolated = np.interp(
    x_daily,
    x_extended,
    y_extended
)

# Light smoothing for visualization only
y_smooth = gaussian_filter1d(
    y_interpolated,
    sigma=6,
    mode="wrap"
)


# ============================================================
# POLAR COORDINATES
# ============================================================
theta_daily = (
    2 * np.pi * (x_daily - 1) / 365.0
)

theta_month = (
    2 * np.pi * (month_centers - 1) / 365.0
)


# ============================================================
# PEAK LOCATION
# ============================================================
peak_index = int(
    np.nanargmax(y_smooth)
)

peak_theta = theta_daily[peak_index]
peak_value = y_smooth[peak_index]


# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,

    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "legend.fontsize": 10,

    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
})


# ============================================================
# CREATE COMBINED FIGURE
# ============================================================
fig = plt.figure(
    figsize=(15.0, 5.8)
)

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.34, 1.00],
    wspace=0.22
)


# ============================================================
# PANEL (a)
# CYCLICAL DOY ENCODING
# ============================================================
ax1 = fig.add_subplot(
    gs[0, 0]
)

# Flatter proportions
ax1.set_box_aspect(0.60)


# Month boundaries
for start in month_starts:

    ax1.axvline(
        start,
        linestyle="--",
        linewidth=0.8,
        alpha=0.65,
        zorder=0
    )


# sin_doy
ax1.plot(
    doy,
    sin_doy,
    linewidth=2.0,
    label="sin_doy"
)


# cos_doy
ax1.plot(
    doy,
    cos_doy,
    linewidth=2.0,
    linestyle="--",
    label="cos_doy"
)


ax1.set_xlim(
    1,
    365
)

ax1.set_ylim(
    -1.10,
    1.10
)


# Bottom axis
ax1.set_xlabel(
    "Day of year"
)

ax1.set_ylabel(
    "Sine/Cosine of Normalized DOY"
)


ax1.set_xticks([
    1,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    365
])


# ------------------------------------------------------------
# Top axis = calendar month context
# ------------------------------------------------------------
ax1_top = ax1.secondary_xaxis(
    "top"
)

ax1_top.set_xticks(
    month_centers
)

ax1_top.set_xticklabels(
    month_names
)

ax1_top.tick_params(
    axis="x",
    length=0,
    pad=5
)


# Light grid
ax1.grid(
    True,
    axis="y",
    alpha=0.18,
    linewidth=0.7
)

ax1.grid(
    True,
    axis="x",
    alpha=0.08,
    linewidth=0.6
)


ax1.legend(
    loc="lower left",
    frameon=True
)


# Panel label
ax1.text(
    -0.035,
    1.035,
    "(a)",
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    clip_on=False
)


# ============================================================
# PANEL (b)
# POLAR MONTHLY Chl-a SUMMARY
# ============================================================
ax2 = fig.add_subplot(
    gs[0, 1],
    projection="polar"
)


# January at top
ax2.set_theta_offset(
    np.pi / 2.0
)

# Calendar progresses clockwise
ax2.set_theta_direction(
    -1
)


# ------------------------------------------------------------
# Monthly 75th-percentile curve
# ------------------------------------------------------------
ax2.plot(
    theta_daily,
    y_smooth,
    linewidth=2.1,
    label="Monthly 75th-percentile Chl-a"
)

ax2.fill(
    theta_daily,
    y_smooth,
    alpha=0.16
)


# ------------------------------------------------------------
# Month + DOY-range labels
# ------------------------------------------------------------
ax2.set_xticks(
    theta_month
)

ax2.set_xticklabels(
    polar_labels
)

for label in ax2.get_xticklabels():
    label.set_fontsize(9)


# ============================================================
# RADIAL SCALE
# ============================================================
curve_max = float(
    np.nanmax(y_smooth)
)

# One complete 5 µg/L interval of headroom
rmax = (
    np.ceil(curve_max / 5.0) * 5.0
    + 5.0
)

rmax = max(
    rmax,
    30.0
)

ax2.set_ylim(
    0,
    rmax
)


# Radial ticks every 5, excluding the outer border value
rticks = np.arange(
    5,
    rmax,
    5
)

ax2.set_yticks(
    rticks
)

ax2.set_yticklabels([
    f"{int(v)}"
    for v in rticks
])

ax2.set_rlabel_position(
    22
)


ax2.grid(
    alpha=0.30,
    linewidth=0.8
)


# ============================================================
# Chl-a LABEL INSIDE BLUE REGION
# ============================================================
ax2.text(
    0.40,
    0.50,
    "Chlorophyll-a (µg L$^{-1}$)",
    transform=ax2.transAxes,
    rotation=90,
    ha="center",
    va="center",
    fontsize=11.5
)


# ============================================================
# PEAK ANNOTATION
# ============================================================
ax2.scatter(
    [peak_theta],
    [peak_value],
    s=28,
    color="black",
    zorder=5
)

ax2.annotate(
    "Peak",
    xy=(peak_theta, peak_value),
    xytext=(5, -8),
    textcoords="offset points",
    ha="left",
    va="center",
    fontsize=9
)


# ============================================================
# LEGEND
# ------------------------------------------------------------
# Positioned above and to the RIGHT of January so it does
# not overlap the Jan / DOY 1-31 label.
# ============================================================
ax2.legend(
    loc="lower left",
    bbox_to_anchor=(0.68, 1.12),
    frameon=True,
    borderaxespad=0.0
)


# Panel label
ax2.text(
    -0.035,
    1.025,
    "(b)",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    clip_on=False
)


# ============================================================
# FINAL LAYOUT
# ============================================================
fig.subplots_adjust(
    left=0.055,
    right=0.975,
    bottom=0.11,
    top=0.90,
    wspace=0.22
)


# ============================================================
# SAVE
# ============================================================
fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT_SVG,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# ============================================================
# OUTPUT SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("FIGURE SAVED")
print("=" * 90)

print(f"PNG : {OUT_PNG}")
print(f"PDF : {OUT_PDF}")
print(f"SVG : {OUT_SVG}")
print(f"CSV : {OUT_CSV}")

print("\nPolar scale:")
print(
    f"  Seasonal curve maximum = "
    f"{curve_max:.2f} µg/L"
)
print(
    f"  Radial axis maximum     = "
    f"{rmax:.0f} µg/L"
)

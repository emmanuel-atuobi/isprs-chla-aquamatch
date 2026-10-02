#!/usr/bin/env python3
"""
Figs. 11 & 15 -- SHAP global feature-importance bar chart (Fig. 11) and beeswarm plot (Fig. 15)
for the top 15 predictors of the final XGBoost model. Reshapes and plots the SHAP values
computed by interpretation/shap_compute_final_xgb.py.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))
# Written by interpretation/shap_compute_final_xgb.py
ROOT = RESULTS_DIR / "shap_final_xgb"

IMPORTANCE_FILE = ROOT / "final_xgb_shap_global_importance.csv"
SHAP_VALUES_FILE = ROOT / "final_xgb_holdout_shap_values.csv"

OUT_BEE_PNG = ROOT / "final_xgb_shap_beeswarm_top15_clean.png"
OUT_BEE_PDF = ROOT / "final_xgb_shap_beeswarm_top15_clean.pdf"

OUT_BAR_PNG = ROOT / "final_xgb_shap_global_importance_top15_clean.png"
OUT_BAR_PDF = ROOT / "final_xgb_shap_global_importance_top15_clean.pdf"

OUT_TABLE = ROOT / "final_xgb_shap_global_importance_top15_clean.csv"

DPI = 600


# ============================================================
# CHECK FILES
# ============================================================

if not IMPORTANCE_FILE.exists():
    raise FileNotFoundError(f"Missing file:\n{IMPORTANCE_FILE}")

if not SHAP_VALUES_FILE.exists():
    raise FileNotFoundError(f"Missing file:\n{SHAP_VALUES_FILE}")


# ============================================================
# LOAD
# ============================================================

importance = pd.read_csv(IMPORTANCE_FILE)
shap_df = pd.read_csv(SHAP_VALUES_FILE)

print("=" * 90)
print("LOADED SAVED SHAP FILES")
print("=" * 90)
print(f"Importance rows: {len(importance)}")
print(f"SHAP rows:       {len(shap_df)}")


# ============================================================
# SHORTER DISPLAY NAMES
# ============================================================

display_name = {
    "lat": "Latitude",
    "long": "Longitude",
    "TGI": "TGI",
    "RBI": "RBI",
    "SWI": "SWI",
    "BLRDGR": "(B − R)/G Index",
    "sin_doy": "sin(DOY)",
    "cos_doy": "cos(DOY)",
    "MNDWI": "MNDWI",
    "FAI": "FAI",
    "AFAI": "AFAI",
    "red": "Red reflectance",
    "NDAVI": "NDAVI",
    "NDTI": "NDTI",
    "NDVI": "NDVI",
    "nir": "NIR reflectance",
    "GBI": "GBI",
    "GNRI": "GNRI",
    "RNI": "RNI",
    "NIRGI": "NIRGI",
    "blue": "Blue reflectance",
    "GDVI": "GDVI",
}


# ============================================================
# TOP 15
# ============================================================

TOP_N = 15

top15 = (
    importance
    .sort_values("mean_abs_shap", ascending=False)
    .head(TOP_N)
    .copy()
)

top_features = top15["feature"].tolist()

print("\nTop 15 predictors:")
for i, feat in enumerate(top_features, start=1):
    val = top15.loc[top15["feature"] == feat, "mean_abs_shap"].iloc[0]
    print(f"{i:2d}. {feat:8s}  mean|SHAP| = {val:.6f}")


# ============================================================
# BUILD TOP-15 MATRICES
# ============================================================

value_cols = [f"VALUE_{f}" for f in top_features]
shap_cols = [f"SHAP_{f}" for f in top_features]

missing_value_cols = [c for c in value_cols if c not in shap_df.columns]
missing_shap_cols = [c for c in shap_cols if c not in shap_df.columns]

if missing_value_cols:
    raise KeyError(
        "Missing VALUE columns:\n  " + "\n  ".join(missing_value_cols)
    )

if missing_shap_cols:
    raise KeyError(
        "Missing SHAP columns:\n  " + "\n  ".join(missing_shap_cols)
    )

X_top = shap_df[value_cols].copy()
X_top.columns = top_features

S_top = shap_df[shap_cols].copy()
S_top.columns = top_features

n_obs = len(X_top)

# keep the ranking order
X_top = X_top[top_features]
S_top = S_top[top_features]

pretty_labels = [display_name.get(f, f) for f in top_features]


# ============================================================
# BEESWARM
# ============================================================

plt.figure(figsize=(10.4, 9.4))

shap.summary_plot(
    S_top.to_numpy(),
    X_top,
    feature_names=pretty_labels,
    plot_type="dot",
    max_display=TOP_N,
    show=False,
    plot_size=None,
)

ax = plt.gca()

# Remove huge header; use one cleaner title only
ax.set_title(
    f"Top {TOP_N} predictors by mean |SHAP| value  |  n = {n_obs:,}",
    fontsize=17,
    pad=18,
)

ax.set_xlabel(
    "SHAP value (impact on predicted chlorophyll-a, µg/L)",
    fontsize=15,
    labelpad=12,
)

ax.set_ylabel("")

ax.tick_params(axis="x", labelsize=12.5)
ax.tick_params(axis="y", labelsize=13.5)

# subtle vertical reference already there at 0, add light x-grid
ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.30)

# colorbar cleanup
fig = plt.gcf()
if len(fig.axes) > 1:
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=12.5)
    cbar_ax.set_ylabel("Feature value", fontsize=15, rotation=90, labelpad=18)

# More room for labels and top title
plt.tight_layout()
plt.subplots_adjust(left=0.23, right=0.96, top=0.92, bottom=0.10)

plt.savefig(
    OUT_BEE_PNG,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    OUT_BEE_PDF,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


# ============================================================
# GLOBAL IMPORTANCE BAR PLOT
# ============================================================

bar_df = top15.copy()
bar_df["label"] = bar_df["feature"].map(lambda x: display_name.get(x, x))

# plot smallest to largest so largest appears at top
bar_df = bar_df.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10.0, 9.2))

bars = ax.barh(
    bar_df["label"],
    bar_df["mean_abs_shap"],
    edgecolor="black",
    linewidth=0.55,
)

max_val = bar_df["mean_abs_shap"].max()
x_offset = max_val * 0.015

# numeric labels at bar ends, in dark red
for bar, value in zip(bars, bar_df["mean_abs_shap"]):
    ax.text(
        bar.get_width() + x_offset,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color="#b22222",
    )

# Cleaner single title only
ax.set_title(
    f"Top {TOP_N} predictors by mean |SHAP| value  |  n = {n_obs:,}",
    fontsize=17,
    pad=18,
)

ax.set_xlabel(
    "Mean |SHAP value|",
    fontsize=15,
    labelpad=10,
)

ax.set_ylabel("")

ax.tick_params(axis="x", labelsize=12.5)
ax.tick_params(axis="y", labelsize=13.5)

ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.30)

# extend x limit to make space for labels
ax.set_xlim(0, max_val * 1.20)

plt.tight_layout()
plt.subplots_adjust(left=0.26, right=0.96, top=0.92, bottom=0.10)

plt.savefig(
    OUT_BAR_PNG,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    OUT_BAR_PDF,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


# ============================================================
# SAVE CLEAN TOP-15 TABLE
# ============================================================

table_df = (
    top15[["rank", "feature", "mean_abs_shap", "mean_shap"]]
    .copy()
)

table_df["display_name"] = table_df["feature"].map(lambda x: display_name.get(x, x))

table_df = table_df[
    ["rank", "feature", "display_name", "mean_abs_shap", "mean_shap"]
]

table_df.to_csv(OUT_TABLE, index=False)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 90)
print("CLEAN TOP-15 SHAP FIGURES CREATED")
print("=" * 90)
print(OUT_BEE_PNG)
print(OUT_BEE_PDF)
print(OUT_BAR_PNG)
print(OUT_BAR_PDF)
print(OUT_TABLE)

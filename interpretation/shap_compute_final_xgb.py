#!/usr/bin/env python3
"""
Computes TreeSHAP values for the final XGBoost model over the untouched holdout set.
Produces final_xgb_shap_global_importance.csv and final_xgb_holdout_shap_values.csv, which
figures/fig11_15_shap_importance_and_beeswarm.py reads and plots. No model is fitted or
tuned here; it only loads the frozen model and explains it.
"""
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

MODEL_FILE = Path(os.environ.get("AQUAMATCH_MODEL_FILE", "./model/xgboost_frozen_v1.json"))

HOLDOUT_FILE = DATA_DIR / "model_comparison_6models_final" / "split" / "holdout_test.csv"

OUT_DIR = RESULTS_DIR / "shap_final_xgb"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600


# ============================================================
# EXPECTED FINAL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "red", "nir", "blue", "lat", "long", "NDVI", "NDTI", "RNI", "GBI",
    "BLRDGR", "GNRI", "RBI", "NIRGI", "GDVI", "NDAVI", "FAI", "MNDWI",
    "SWI", "TGI", "AFAI", "sin_doy", "cos_doy",
]


# ============================================================
# IMPORT SHAP
# ============================================================

try:
    import shap
except ImportError:
    print("\nERROR: shap is not installed in the current environment.")
    print("Try:")
    print("pip install shap")
    sys.exit(1)


print("=" * 90)
print("FINAL XGBOOST SHAP ANALYSIS")
print("=" * 90)
print(f"SHAP version: {shap.__version__}")


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Final model not found:\n{MODEL_FILE}")

if not HOLDOUT_FILE.exists():
    raise FileNotFoundError(f"Holdout file not found:\n{HOLDOUT_FILE}")


# ============================================================
# LOAD FINAL SAVED MODEL
# ============================================================

model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)

print("\nLoaded:")
print(MODEL_FILE)

saved_features = model.get_booster().feature_names
features = list(saved_features) if saved_features is not None else EXPECTED_FEATURES.copy()


print(f"\nModel type: {type(model)}")
print(f"Number of model features: {len(features)}")

print("\nFeatures:")
for i, feature in enumerate(features, start=1):
    print(f"{i:2d}. {feature}")


# ============================================================
# VERIFY THIS IS THE EXPECTED FINAL FEATURE SET
# ============================================================

if features != EXPECTED_FEATURES:
    print("\nWARNING:")
    print("Features stored with the saved model differ from the expected 22-feature list.")
    print("\nUsing the feature order stored with the model, because that is the authoritative order.")


# ============================================================
# LOAD UNTOUCHED HOLDOUT
# ============================================================

holdout = pd.read_csv(HOLDOUT_FILE, low_memory=False)

print(f"\nHoldout rows loaded: {len(holdout):,}")

missing = [feature for feature in features if feature not in holdout.columns]
if missing:
    raise KeyError("The holdout file is missing model features:\n  " + "\n  ".join(missing))

X = holdout[features].apply(pd.to_numeric, errors="coerce")

# SHAP cannot use missing/inf values here.
finite_mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

if not finite_mask.all():
    removed = int((~finite_mask).sum())
    print(f"Removing {removed:,} holdout rows with non-finite predictors before SHAP.")

    holdout = holdout.loc[finite_mask].reset_index(drop=True)
    X = X.loc[finite_mask].reset_index(drop=True)

print(f"Holdout observations used for SHAP: {len(X):,}")


# ============================================================
# CONFIRM MODEL PREDICTION WORKS
# ============================================================

pred = model.predict(X)

print(f"Prediction range: {np.min(pred):.4f} to {np.max(pred):.4f}")


# ============================================================
# COMPUTE TREE SHAP
# ============================================================

print("\n" + "=" * 90)
print("COMPUTING TREE SHAP VALUES")
print("=" * 90)

explainer = shap.TreeExplainer(model)

explanation = explainer(X)

shap_values = np.asarray(explanation.values)

if shap_values.ndim != 2:
    raise ValueError(f"Unexpected SHAP array shape: {shap_values.shape}")

print(f"SHAP matrix shape: {shap_values.shape}")


# ============================================================
# GLOBAL MEAN ABSOLUTE SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
mean_shap = np.mean(shap_values, axis=0)

importance = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap": mean_shap,
})

importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
importance["rank"] = np.arange(len(importance)) + 1
importance = importance[["rank", "feature", "mean_abs_shap", "mean_shap"]]

IMPORTANCE_FILE = OUT_DIR / "final_xgb_shap_global_importance.csv"
importance.to_csv(IMPORTANCE_FILE, index=False)

print("\n" + "=" * 90)
print("FINAL GLOBAL SHAP RANKING")
print("=" * 90)
print(importance.to_string(
    index=False,
    formatters={
        "mean_abs_shap": "{:.6f}".format,
        "mean_shap": "{:+.6f}".format,
    }
))


# ============================================================
# SAVE OBSERVATION-LEVEL SHAP VALUES
# ============================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=[f"SHAP_{feature}" for feature in features]
)

feature_df = X.copy()
feature_df.columns = [f"VALUE_{feature}" for feature in features]

output_rows = pd.concat(
    [
        pd.DataFrame({
            "holdout_row": np.arange(1, len(X) + 1),
            "prediction": pred,
        }),
        feature_df.reset_index(drop=True),
        shap_df.reset_index(drop=True),
    ],
    axis=1
)

SHAP_VALUES_FILE = OUT_DIR / "final_xgb_holdout_shap_values.csv"
output_rows.to_csv(SHAP_VALUES_FILE, index=False)


# ============================================================
# GLOBAL IMPORTANCE BAR PLOT
# ============================================================

plot_importance = importance.sort_values("mean_abs_shap", ascending=True).copy()

fig, ax = plt.subplots(figsize=(8.2, 8.4))

ax.barh(
    plot_importance["feature"],
    plot_importance["mean_abs_shap"],
    edgecolor="black",
    linewidth=0.45,
)

ax.set_xlabel("Mean |SHAP value|", fontsize=12)
ax.set_ylabel("Predictor", fontsize=12)
ax.tick_params(axis="both", labelsize=10.5)
ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.35)

# No figure title for manuscript use
fig.tight_layout()

BAR_PNG = OUT_DIR / "final_xgb_shap_global_importance.png"
BAR_PDF = OUT_DIR / "final_xgb_shap_global_importance.pdf"

fig.savefig(BAR_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(BAR_PDF, bbox_inches="tight", facecolor="white")

plt.close(fig)


# ============================================================
# SHAP BEESWARM
# ============================================================

# Keep all 22 predictors.
# max_display can be changed later if we decide top 15 or top 20
# is visually cleaner for the manuscript.

plt.figure(figsize=(9.5, 9.5))

shap.summary_plot(
    shap_values,
    X,
    feature_names=features,
    plot_type="dot",
    max_display=len(features),
    show=False,
    plot_size=None,
)

current_ax = plt.gca()

current_ax.set_xlabel("SHAP value (impact on predicted Chl-a)", fontsize=12)
current_ax.tick_params(axis="both", labelsize=10.5)

plt.tight_layout()

BEESWARM_PNG = OUT_DIR / "final_xgb_shap_beeswarm.png"
BEESWARM_PDF = OUT_DIR / "final_xgb_shap_beeswarm.pdf"

plt.savefig(BEESWARM_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig(BEESWARM_PDF, bbox_inches="tight", facecolor="white")

plt.close()


# ============================================================
# TOP-10 TEXT SUMMARY
# ============================================================

TOP_FILE = OUT_DIR / "final_xgb_shap_top10.txt"

with open(TOP_FILE, "w", encoding="utf-8") as f:
    f.write("FINAL XGBOOST SHAP ANALYSIS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Model: {MODEL_FILE}\n")
    f.write(f"Dataset explained: {HOLDOUT_FILE}\n")
    f.write(f"Holdout observations: {len(X):,}\n")
    f.write(f"Predictors: {len(features)}\n\n")
    f.write("TOP 10 GLOBAL PREDICTORS\n")
    f.write("-" * 70 + "\n")

    for _, row in importance.head(10).iterrows():
        f.write(
            f"{int(row['rank']):2d}. {row['feature']:12s} "
            f"Mean |SHAP| = {row['mean_abs_shap']:.6f}\n"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 90)
print("SHAP ANALYSIS COMPLETE")
print("=" * 90)

print("\nFiles saved:")
for path in [IMPORTANCE_FILE, SHAP_VALUES_FILE, BAR_PNG, BAR_PDF, BEESWARM_PNG, BEESWARM_PDF, TOP_FILE]:
    print(path)

print("\nIMPORTANT: No model was fitted or tuned in this script.")

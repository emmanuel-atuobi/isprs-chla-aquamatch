#!/usr/bin/env python3
"""
Feature-group ablation, random 75/25 splits (35 seeds) -- "Random" row of the Section 3.4
ablation table / Fig. 12. Reuses the per-seed tuned hyperparameters from the 35-seed Monte
Carlo XGBoost experiment (xgb_monte_carlo_35/) and holds them fixed while refitting on
reduced feature-group configurations (Spectral only / +temporal / +spatial); the "Full
model" row is taken directly from the existing Monte Carlo results, not refit.

Output feeds figures/fig12_feature_group_ablation.py directly (RANDOM_FILE).
"""
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from scipy.stats import linregress
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

INPUT_FILE = DATA_DIR / "raw" / "aquamatch_sitesr_joined.csv"

# Produced by validation/xgb_monte_carlo_35_tuned.py
MC_DIR = RESULTS_DIR / "xgb_monte_carlo_35"

FULL_MODEL_RESULTS_FILE = MC_DIR / "monte_carlo_xgb_results.csv"

BEST_PARAMS_FILE = MC_DIR / "monte_carlo_xgb_best_parameters.csv"

OUT_DIR = MC_DIR / "feature_group_ablation"

OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 35
RUN_SEEDS = list(range(42, 77))
TEST_SIZE = 0.25
TIME_DIFFERENCE_HOURS = 2


# =============================================================================
# FEATURE GROUPS
# =============================================================================

SPECTRAL = [
    "red", "nir", "blue", "NDVI", "NDTI", "RNI", "GBI", "BLRDGR", "GNRI",
    "RBI", "NIRGI", "GDVI", "NDAVI", "FAI", "MNDWI", "SWI", "TGI", "AFAI",
]

TEMPORAL = ["sin_doy", "cos_doy"]

SPATIAL = ["lat", "long"]

FULL = SPECTRAL + SPATIAL + TEMPORAL

# Only the reduced configurations need to be newly fitted.
# Full-model results already exist and are loaded below (not refit).
ABLATIONS = {
    "Spectral only": SPECTRAL,
    "Spectral + temporal": SPECTRAL + TEMPORAL,
    "Spectral + spatial": SPECTRAL + SPATIAL,
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def require_columns(data, columns):
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError("Missing required columns:\n  " + "\n  ".join(missing))


def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1.0 - ((1.0 - r2) * (n - 1.0) / (n - p - 1.0))


def calculate_metrics(y_true, y_pred, n_features):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred))
    fit = linregress(y_true, y_pred)
    return {
        "r2": r2,
        "adjusted_r2": float(adjusted_r2(r2, len(y_true), n_features)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "pearson_r": float(fit.rvalue),
    }


def clean_best_params(row):
    """Convert one row of the saved Monte Carlo best-parameter table into valid XGBRegressor args."""
    integer_params = {"n_estimators", "max_depth", "max_bin"}
    float_params = {
        "learning_rate", "subsample", "colsample_bytree", "min_child_weight",
        "gamma", "reg_alpha", "reg_lambda",
    }
    params = {}
    for key in integer_params:
        if key not in row.index:
            raise KeyError(f"Best-parameter file does not contain {key}")
        params[key] = int(round(float(row[key])))
    for key in float_params:
        if key not in row.index:
            raise KeyError(f"Best-parameter file does not contain {key}")
        params[key] = float(row[key])
    return params


# =============================================================================
# CHECK FILES
# =============================================================================

banner("CHECKING INPUTS")

for file in [INPUT_FILE, FULL_MODEL_RESULTS_FILE, BEST_PARAMS_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Missing required file:\n{file}")

print("All required files found.")


# =============================================================================
# LOAD FULL-MODEL MONTE CARLO RESULTS
# =============================================================================

banner("LOADING FULL-MODEL 35-SEED MONTE CARLO RESULTS")

full_model = pd.read_csv(FULL_MODEL_RESULTS_FILE)
best_params = pd.read_csv(BEST_PARAMS_FILE)

print(f"Full-model results: {len(full_model)} runs")
print(f"Best-parameter rows: {len(best_params)}")


# =============================================================================
# VERIFY SEEDS
# =============================================================================

required_full_model_cols = ["split_seed", "test_r2", "test_rmse", "test_mae", "test_bias"]
require_columns(full_model, required_full_model_cols)
require_columns(
    best_params,
    [
        "split_seed", "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
        "max_bin",
    ],
)

observed_seeds = sorted(full_model["split_seed"].astype(int).tolist())
if observed_seeds != RUN_SEEDS:
    raise RuntimeError(
        "Full-model Monte Carlo seeds do not match the expected range 42-76.\n"
        f"Found: {observed_seeds}"
    )

param_seeds = sorted(best_params["split_seed"].astype(int).tolist())
if param_seeds != RUN_SEEDS:
    raise RuntimeError("Best-parameter table does not contain exactly seeds 42-76.")

print(f"Confirmed full-model split seeds: {RUN_SEEDS[0]}-{RUN_SEEDS[-1]}")
print()
print("Full-model Monte Carlo performance:")
print(f"R2   = {full_model['test_r2'].mean():.4f} +/- {full_model['test_r2'].std(ddof=1):.4f}")
print(f"RMSE = {full_model['test_rmse'].mean():.3f} +/- {full_model['test_rmse'].std(ddof=1):.3f} ug/L")
print(f"MAE  = {full_model['test_mae'].mean():.3f} +/- {full_model['test_mae'].std(ddof=1):.3f} ug/L")
print(f"Bias = {full_model['test_bias'].mean():+.3f} +/- {full_model['test_bias'].std(ddof=1):.3f} ug/L")


# =============================================================================
# LOAD AND FILTER AQUAMATCH
#
# This reproduces the full-model Monte Carlo's data preparation exactly.
# =============================================================================

banner("LOADING AND FILTERING AQUAMATCH")

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Starting rows: {len(df):,}")

df = df.rename(columns={
    "med_Blue": "blue",
    "med_Green": "green",
    "med_Red": "red",
    "med_Nir": "nir",
    "med_Swir1": "swir1",
    "med_Swir2": "swir2",
    "harmonized_value": "chl_a",
    "lon": "long",
})

required = [
    "ResolvedMonitoringLocationTypeName", "mission", "chl_a", "mdl_flag",
    "harmonized_row_count", "harmonized_value_cv", "depth_flag",
    "harmonized_discrete_depth_value", "harmonized_top_depth_value",
    "harmonized_bottom_depth_value", "blue", "green", "red", "nir", "swir1",
    "swir2", "pCount_dswe1", "timediff", "lat", "long",
]
require_columns(df, required)

numeric_columns = [
    "chl_a", "harmonized_value_cv", "harmonized_discrete_depth_value",
    "harmonized_top_depth_value", "harmonized_bottom_depth_value", "blue",
    "green", "red", "nir", "swir1", "swir2", "pCount_dswe1", "timediff",
    "lat", "long",
]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# =============================================================================
# QUALITY FILTERING, MATCHING THE MONTE CARLO SPLITS
# =============================================================================

df = df[df["ResolvedMonitoringLocationTypeName"] == "Lake, Reservoir, Impoundment"].copy()
df = df[df["mission"].isin(["LT05", "LE07", "LC08", "LC09"])].copy()
df = df[np.isfinite(df["chl_a"]) & (df["chl_a"] > 0) & (df["chl_a"] <= 200)].copy()
df = df[df["mdl_flag"] == 0].copy()

replicate_mask = (
    (df["harmonized_row_count"] == 1) | (df["harmonized_value_cv"] <= 0.5)
)
df = df[replicate_mask].copy()

discrete_mask = (
    (df["depth_flag"] == 1)
    & (df["harmonized_discrete_depth_value"] >= 0)
    & (df["harmonized_discrete_depth_value"] <= 2)
)
integrated_mask = (
    (df["depth_flag"] == 2)
    & (df["harmonized_top_depth_value"] >= 0)
    & (df["harmonized_top_depth_value"] <= 0.5)
    & (df["harmonized_bottom_depth_value"] >= df["harmonized_top_depth_value"])
    & (df["harmonized_bottom_depth_value"] <= 2)
)
df = df[discrete_mask | integrated_mask].copy()

for column in ["blue", "green", "red", "nir", "swir1", "swir2"]:
    df = df[np.isfinite(df[column]) & (df[column] > 0)].copy()

df = df[df["pCount_dswe1"] >= 8].copy()
df = df[np.isfinite(df["timediff"]) & (df["timediff"] <= TIME_DIFFERENCE_HOURS)].copy()


# =============================================================================
# FEATURE ENGINEERING
#
# Same index formulas as elsewhere in this repo, without the "+ eps" denominator
# guard those scripts use -- safe here since all six reflectance bands are already
# filtered to be strictly positive upstream.
# =============================================================================

with np.errstate(divide="ignore", invalid="ignore"):
    df["NDVI"] = (df["nir"] - df["red"]) / (df["nir"] + df["red"])
    df["NDTI"] = (df["red"] - df["green"]) / (df["red"] + df["green"])
    df["RNI"] = df["red"] / df["nir"]
    df["GBI"] = df["green"] / df["blue"]
    df["BLRDGR"] = (df["blue"] - df["red"]) / df["green"]
    df["GNRI"] = df["green"] - (df["green"] / df["red"])
    df["RBI"] = df["red"] / df["blue"]
    df["NIRGI"] = df["nir"] / df["green"]
    df["GDVI"] = df["nir"] - df["green"]
    df["NDAVI"] = (df["nir"] - df["blue"]) / (df["nir"] + df["blue"])
    baseline = df["red"] + ((df["swir1"] - df["red"]) * (0.86 - 0.66) / (1.60 - 0.66))
    df["FAI"] = df["nir"] - baseline
    df["MNDWI"] = (df["green"] - df["swir1"]) / (df["green"] + df["swir1"])
    df["SWI"] = (df["nir"] - df["swir1"]) / (df["nir"] + df["swir1"])
    df["TGI"] = -0.5 * (120 * (df["red"] - df["green"]) - 190 * (df["red"] - df["blue"]))
    df["AFAI"] = (df["nir"] - df["red"]) + 0.5 * (df["swir1"] - df["red"])


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

date_column = "ActivityStartDate" if "ActivityStartDate" in df.columns else "date"
require_columns(df, [date_column])

df["_date"] = pd.to_datetime(df[date_column], errors="coerce", format="mixed", dayfirst=False)
day_of_year = df["_date"].dt.dayofyear

df["sin_doy"] = np.sin(2 * np.pi * (day_of_year - 1) / 365.0)
df["cos_doy"] = np.cos(2 * np.pi * (day_of_year - 1) / 365.0)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FULL + ["chl_a"]).reset_index(drop=True)


# =============================================================================
# SANITY CHECK
# =============================================================================

banner("MODEL-READY DATASET")

print(f"Final observations: {len(df):,}")
print(f"Predictors in full model: {len(FULL)}")

if len(df) != 14125:
    raise RuntimeError(f"Expected exactly 14,125 model-ready observations. Found {len(df):,}.")

X = df[FULL].copy()
y = df["chl_a"].copy()


# =============================================================================
# INDEX LOOKUPS FOR FULL-MODEL RESULTS / PARAMETERS
# =============================================================================

full_model_lookup = full_model.set_index("split_seed")
params_lookup = best_params.set_index("split_seed")


# =============================================================================
# RUN REDUCED-MODEL ABLATION
# =============================================================================

banner("RUNNING RANDOM-SPLIT FEATURE-GROUP ABLATION")

print("35 random 75/25 splits using seeds 42-76.")
print("For each seed, its previously selected full-model hyperparameters are held fixed across all reduced configurations.")
print("NO new hyperparameter tuning is performed.")
print("New fits: 35 seeds x 3 reduced configurations = 105")

result_rows = []
overall_start = time.perf_counter()

for run_number, seed in enumerate(RUN_SEEDS, start=1):
    print()
    print("-" * 110)
    print(f"SEED {seed} ({run_number:02d}/35)")
    print("-" * 110)

    # Reproduce the same 75/25 split used for the full-model run
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=seed,
    )

    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    if len(train_idx) != 10593:
        raise RuntimeError(f"Seed {seed}: expected 10,593 training observations, found {len(train_idx):,}.")
    if len(test_idx) != 3532:
        raise RuntimeError(f"Seed {seed}: expected 3,532 test observations, found {len(test_idx):,}.")

    # Hyperparameters selected previously for THIS full-model MC run
    param_row = params_lookup.loc[seed]
    fixed_params = clean_best_params(param_row)

    # Fit each reduced predictor configuration with those parameters unchanged
    for config_name, features in ABLATIONS.items():
        start = time.perf_counter()

        X_train = df[features].iloc[train_idx]
        X_test = df[features].iloc[test_idx]

        model = XGBRegressor(
            **fixed_params,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=seed,
            tree_method="hist",
            device="cpu",
            n_jobs=16,
            verbosity=0,
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, pred, len(features))
        elapsed = time.perf_counter() - start

        print(
            f"{config_name:24s} | R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.3f} | "
            f"MAE={metrics['mae']:.3f} | Bias={metrics['bias']:+.3f} | {elapsed:.1f}s"
        )

        result_rows.append({
            "run": run_number,
            "split_seed": seed,
            "configuration": config_name,
            "n_features": len(features),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **metrics,
            "fit_seconds": elapsed,
        })


# =============================================================================
# ADD FULL-MODEL RESULTS
# =============================================================================

banner("ADDING EXISTING FULL-MODEL MONTE CARLO RESULTS")

reduced = pd.DataFrame(result_rows)

full_rows = []
for run_number, seed in enumerate(RUN_SEEDS, start=1):
    row = full_model_lookup.loc[seed]
    full_rows.append({
        "run": run_number,
        "split_seed": seed,
        "configuration": "Full model",
        "n_features": 22,
        "n_train": int(row["n_train"]),
        "n_test": int(row["n_test"]),
        "r2": float(row["test_r2"]),
        "adjusted_r2": float(row["test_adjusted_r2"]),
        "rmse": float(row["test_rmse"]),
        "mae": float(row["test_mae"]),
        "bias": float(row["test_bias"]),
        "slope": np.nan,
        "intercept": np.nan,
        "pearson_r": np.nan,
        "fit_seconds": np.nan,
    })

full_df = pd.DataFrame(full_rows)
all_metrics = pd.concat([reduced, full_df], ignore_index=True)


# =============================================================================
# SUMMARY BY FEATURE CONFIGURATION
# =============================================================================

order = ["Spectral only", "Spectral + temporal", "Spectral + spatial", "Full model"]

summary_rows = []
for configuration in order:
    subset = all_metrics[all_metrics["configuration"] == configuration]
    row = {
        "configuration": configuration,
        "n_features": int(subset["n_features"].iloc[0]),
    }
    for metric in ["r2", "adjusted_r2", "rmse", "mae", "bias"]:
        values = subset[metric].astype(float)
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_sd"] = values.std(ddof=1)
        row[f"{metric}_median"] = values.median()
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)


# =============================================================================
# PAIRED CONTRASTS
# =============================================================================

wide = all_metrics.pivot(
    index="split_seed", columns="configuration", values=["r2", "rmse", "mae"],
)

paired_rows = []


def add_contrast(contrast, base, enhanced):
    for seed in RUN_SEEDS:
        base_r2 = float(wide.loc[seed, ("r2", base)])
        enhanced_r2 = float(wide.loc[seed, ("r2", enhanced)])
        base_rmse = float(wide.loc[seed, ("rmse", base)])
        enhanced_rmse = float(wide.loc[seed, ("rmse", enhanced)])
        base_mae = float(wide.loc[seed, ("mae", base)])
        enhanced_mae = float(wide.loc[seed, ("mae", enhanced)])

        paired_rows.append({
            "split_seed": seed,
            "contrast": contrast,
            "base": base,
            "enhanced": enhanced,
            # Positive = better
            "delta_r2": enhanced_r2 - base_r2,
            # Positive = lower error
            "rmse_reduction": base_rmse - enhanced_rmse,
            "rmse_reduction_pct": 100 * (base_rmse - enhanced_rmse) / base_rmse,
            "mae_reduction": base_mae - enhanced_mae,
            "mae_reduction_pct": 100 * (base_mae - enhanced_mae) / base_mae,
        })


add_contrast("Temporal added to spectral", "Spectral only", "Spectral + temporal")
add_contrast("Spatial added to spectral", "Spectral only", "Spectral + spatial")
add_contrast("Spatial + temporal added to spectral", "Spectral only", "Full model")
add_contrast("Temporal added after spatial", "Spectral + spatial", "Full model")

paired = pd.DataFrame(paired_rows)


# =============================================================================
# PAIRED SUMMARY
# =============================================================================

paired_summary_rows = []
for contrast in paired["contrast"].drop_duplicates():
    subset = paired[paired["contrast"] == contrast]
    row = {"contrast": contrast}
    for metric in ["delta_r2", "rmse_reduction", "rmse_reduction_pct", "mae_reduction", "mae_reduction_pct"]:
        values = subset[metric].astype(float)
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_sd"] = values.std(ddof=1)
        row[f"{metric}_median"] = values.median()
        row[f"{metric}_positive_n"] = int((values > 0).sum())
    paired_summary_rows.append(row)

paired_summary = pd.DataFrame(paired_summary_rows)


# =============================================================================
# SAVE
# =============================================================================

METRICS_FILE = OUT_DIR / "MC35_ablation_metrics.csv"
SUMMARY_FILE = OUT_DIR / "MC35_ablation_summary.csv"
PAIRED_FILE = OUT_DIR / "MC35_ablation_paired_changes.csv"
PAIRED_SUMMARY_FILE = OUT_DIR / "MC35_ablation_paired_summary.csv"

all_metrics.to_csv(METRICS_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)
paired.to_csv(PAIRED_FILE, index=False)
paired_summary.to_csv(PAIRED_SUMMARY_FILE, index=False)


# =============================================================================
# PRINT MAIN RESULTS
# =============================================================================

banner("FINAL 35-SEED RANDOM-SPLIT ABLATION SUMMARY")

for _, row in summary.iterrows():
    print(
        f"{row['configuration']:24s} | R2 = {row['r2_mean']:.4f} +/- {row['r2_sd']:.4f} | "
        f"RMSE = {row['rmse_mean']:.3f} +/- {row['rmse_sd']:.3f} | "
        f"MAE = {row['mae_mean']:.3f} +/- {row['mae_sd']:.3f} | "
        f"Bias = {row['bias_mean']:+.3f} +/- {row['bias_sd']:.3f}"
    )

banner("PAIRED FEATURE-GROUP CONTRIBUTIONS ACROSS SAME 35 RANDOM SPLITS")

for _, row in paired_summary.iterrows():
    print()
    print(row["contrast"])
    print(f"  dR2: {row['delta_r2_mean']:+.4f} +/- {row['delta_r2_sd']:.4f}")
    print(f"  RMSE reduction: {row['rmse_reduction_mean']:+.3f} +/- {row['rmse_reduction_sd']:.3f} ug/L")
    print(f"  Relative RMSE reduction: {row['rmse_reduction_pct_mean']:+.2f}% +/- {row['rmse_reduction_pct_sd']:.2f}%")
    print(f"  MAE reduction: {row['mae_reduction_mean']:+.3f} +/- {row['mae_reduction_sd']:.3f} ug/L")
    print(f"  Relative MAE reduction: {row['mae_reduction_pct_mean']:+.2f}% +/- {row['mae_reduction_pct_sd']:.2f}%")
    print(f"  R2 improved in {int(row['delta_r2_positive_n'])}/35 splits")
    print(f"  RMSE improved in {int(row['rmse_reduction_positive_n'])}/35 splits")

banner("OUTPUTS")

print(METRICS_FILE)
print(SUMMARY_FILE)
print(PAIRED_FILE)
print(PAIRED_SUMMARY_FILE)

print()
print(f"Total ablation runtime: {(time.perf_counter() - overall_start) / 60:.1f} minutes")
print()
print("DONE")

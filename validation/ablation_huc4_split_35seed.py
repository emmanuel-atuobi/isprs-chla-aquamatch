#!/usr/bin/env python3
"""
Feature-group ablation, HUC4 station holdout splits (35 seeds) -- "HUC4" row of the Section
3.4 ablation table / Fig. 12. Reuses the saved atomic station-group manifests from
validation/run_huc4_fixed_35seeds.py (spatial_validation_diagnostics/
repeated_HUC4_fixed_35seeds/splits/) and the same fixed XGBoost configuration for every
seed, refitting on reduced feature-group configurations (Spectral only / +temporal /
+spatial); the "Full model" row is taken directly from the existing
HUC4_fixed_35seed_metrics.csv, not refit.

Output feeds figures/fig12_feature_group_ablation.py directly (HUC4_FILE).
"""
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from scipy.stats import linregress

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SEEDS = list(range(1, 36))
COORD_DECIMALS = 6

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

INPUT = DATA_DIR / "spatial_validation_diagnostics" / "huc8_field_comparison.csv"

# Produced by validation/run_huc4_fixed_35seeds.py
HUC_ROOT = RESULTS_DIR / "spatial_validation_diagnostics" / "repeated_HUC4_fixed_35seeds"

MANIFEST_DIR = HUC_ROOT / "splits"

EXISTING_FULL_METRICS = HUC_ROOT / "HUC4_fixed_35seed_metrics.csv"

OUT = RESULTS_DIR / "spatial_validation_diagnostics" / "HUC4_feature_group_ablation_35seeds"

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "predictions").mkdir(parents=True, exist_ok=True)


# =============================================================================
# FEATURE GROUPS
# =============================================================================

SPECTRAL = [
    "red", "nir", "blue", "NDVI", "NDTI", "RNI", "GBI", "BLRDGR", "GNRI",
    "RBI", "NIRGI", "GDVI", "NDAVI", "FAI", "MNDWI", "SWI", "TGI", "AFAI",
]

SPATIAL = ["lat", "long"]

TEMPORAL = ["sin_doy", "cos_doy"]

FULL = SPECTRAL + SPATIAL + TEMPORAL

# Only these three require new fits.
ABLATIONS = {
    "Spectral only": SPECTRAL,
    "Spectral + temporal": SPECTRAL + TEMPORAL,
    "Spectral + spatial": SPECTRAL + SPATIAL,
}


# =============================================================================
# EXACT FIXED PARAMETERS FROM EXISTING 35-SEED HUC4 EXPERIMENT
# =============================================================================

FIXED_PARAMS = {
    "colsample_bytree": 0.8,
    "gamma": 0.5,
    "learning_rate": 0.02,
    "max_bin": 512,
    "max_depth": 7,
    "min_child_weight": 1,
    "n_estimators": 2200,
    "reg_alpha": 0.1,
    "reg_lambda": 4.0,
    "subsample": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": 16,
    "verbosity": 0,
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def normalize_huc8(series):
    s = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False)
    return s.str.zfill(8)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fit = linregress(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "pearson_r": float(fit.rvalue),
    }


# =============================================================================
# FEATURE ENGINEERING MATCHING run_huc4_fixed_35seeds.py
# =============================================================================

def add_features(df):
    df = df.copy()

    rename_map = {
        "harmonized_value": "chl_a",
        "med_Blue": "blue",
        "med_Green": "green",
        "med_Red": "red",
        "med_Nir": "nir",
        "med_Swir1": "swir1",
        "med_Swir2": "swir2",
        "lon": "long",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    numeric_cols = ["chl_a", "blue", "green", "red", "nir", "swir1", "swir2", "lat", "long"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    eps = np.finfo(float).eps

    df["NDVI"] = (df["nir"] - df["red"]) / (df["nir"] + df["red"] + eps)
    df["NDTI"] = (df["red"] - df["green"]) / (df["red"] + df["green"] + eps)
    df["RNI"] = df["red"] / (df["nir"] + eps)
    df["GBI"] = df["green"] / (df["blue"] + eps)
    df["BLRDGR"] = (df["blue"] - df["red"]) / (df["green"] + eps)
    df["GNRI"] = df["green"] - (df["green"] / (df["red"] + eps))
    df["RBI"] = df["red"] / (df["blue"] + eps)
    df["NIRGI"] = df["nir"] / (df["green"] + eps)
    df["GDVI"] = df["nir"] - df["green"]
    df["NDAVI"] = (df["nir"] - df["blue"]) / (df["nir"] + df["blue"] + eps)
    baseline = df["red"] + ((df["swir1"] - df["red"]) * (0.86 - 0.66) / (1.60 - 0.66))
    df["FAI"] = df["nir"] - baseline
    df["MNDWI"] = (df["green"] - df["swir1"]) / (df["green"] + df["swir1"] + eps)
    df["SWI"] = (df["nir"] - df["swir1"]) / (df["nir"] + df["swir1"] + eps)
    df["TGI"] = -0.5 * (120 * (df["red"] - df["green"]) - 190 * (df["red"] - df["blue"]))
    df["AFAI"] = (df["nir"] - df["red"]) + 0.5 * (df["swir1"] - df["red"])

    # Cyclical DOY
    if "ActivityStartDate" in df.columns:
        date_col = "ActivityStartDate"
    elif "date" in df.columns:
        date_col = "date"
    else:
        raise KeyError("Could not identify date column.")

    date = pd.to_datetime(df[date_col], errors="coerce", format="mixed", dayfirst=False)
    doy = date.dt.dayofyear

    df["sin_doy"] = np.sin(2 * np.pi * (doy - 1) / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * (doy - 1) / 365.0)

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =============================================================================
# CHECK FILES
# =============================================================================

banner("CHECKING INPUT FILES")

if not INPUT.exists():
    raise FileNotFoundError(f"Missing quality-filtered input:\n{INPUT}")

if not EXISTING_FULL_METRICS.exists():
    raise FileNotFoundError(f"Missing existing HUC4 metrics:\n{EXISTING_FULL_METRICS}")

for seed in SEEDS:
    path = MANIFEST_DIR / f"HUC4_seed_{seed:02d}_station_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest:\n{path}")

print("All 35 station manifests found.")


# =============================================================================
# LOAD QUALITY-FILTERED DATA
# =============================================================================

banner("LOADING QUALITY-FILTERED DATASET")

df = pd.read_csv(INPUT, low_memory=False, dtype={"HUC8_assigned": "string"})

print(f"Rows: {len(df):,}")

if len(df) != 14125:
    raise RuntimeError(f"Expected 14,125 observations; found {len(df):,}")


# =============================================================================
# STANDARDIZE CORE FIELDS
# =============================================================================

if "lon" in df.columns and "long" not in df.columns:
    df = df.rename(columns={"lon": "long"})

if "harmonized_value" in df.columns and "chl_a" not in df.columns:
    df = df.rename(columns={"harmonized_value": "chl_a"})

df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["long"] = pd.to_numeric(df["long"], errors="coerce")


# =============================================================================
# HUC8
# =============================================================================

huc_candidates = ["HUC8_assigned", "assigned_HUC", "HUC8", "HUCEightDigitCode"]

HUC8_FIELD = None
for candidate in huc_candidates:
    if candidate in df.columns:
        HUC8_FIELD = candidate
        break

if HUC8_FIELD is None:
    raise KeyError("Could not find HUC8 field.")

print(f"HUC8 field: {HUC8_FIELD}")

df["_HUC8"] = normalize_huc8(df[HUC8_FIELD])


# =============================================================================
# RECONSTRUCT EXACT ATOMIC STATION GROUPS
# =============================================================================

banner("RECONSTRUCTING ATOMIC STATION GROUPS")

df["_station_id"] = df["MonitoringLocationIdentifier"].astype("string")

missing_id = df["_station_id"].isna()
if missing_id.any():
    df.loc[missing_id, "_station_id"] = [
        f"MISSING_ID_ROW_{i}" for i in df.index[missing_id]
    ]

df["_coord_key"] = (
    df["lat"].round(COORD_DECIMALS).map(lambda x: f"{x:.{COORD_DECIMALS}f}")
    + "_"
    + df["long"].round(COORD_DECIMALS).map(lambda x: f"{x:.{COORD_DECIMALS}f}")
)

n = len(df)
parent = np.arange(n, dtype=int)
rank = np.zeros(n, dtype=int)


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra = find(a)
    rb = find(b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


first_id = {}
first_coord = {}

for pos, (station_id, coord_key) in enumerate(zip(df["_station_id"], df["_coord_key"])):
    station_id = str(station_id)
    coord_key = str(coord_key)

    if station_id in first_id:
        union(pos, first_id[station_id])
    else:
        first_id[station_id] = pos

    if coord_key in first_coord:
        union(pos, first_coord[coord_key])
    else:
        first_coord[coord_key] = pos


roots = np.array([find(i) for i in range(n)])

unique_roots = {
    root: number for number, root in enumerate(np.unique(roots), start=1)
}

df["station_group"] = [f"STATION_{unique_roots[root]:05d}" for root in roots]

n_stations = df["station_group"].nunique()

print(f"Atomic station groups: {n_stations:,}")

if n_stations != 5147:
    raise RuntimeError(f"Expected 5,147 stations; found {n_stations:,}")


# =============================================================================
# ENGINEER EXACT 22 FEATURES
# =============================================================================

banner("ENGINEERING FINAL PREDICTORS")

df = add_features(df)

df = df.dropna(subset=FULL + ["chl_a", "station_group"]).reset_index(drop=True)

print(f"Model-ready observations: {len(df):,}")

if len(df) != 14125:
    raise RuntimeError("Feature engineering changed the expected sample size.")


# =============================================================================
# LOAD EXISTING FULL-MODEL METRICS
# =============================================================================

banner("LOADING EXISTING FULL HUC4 RESULTS")

full_existing = pd.read_csv(EXISTING_FULL_METRICS)

required_full = ["seed", "test_r2", "test_rmse", "test_mae", "test_bias", "test_slope"]

missing = [c for c in required_full if c not in full_existing.columns]
if missing:
    raise KeyError("Existing metrics file is missing:\n" + "\n".join(missing))

full_existing = full_existing[required_full].copy()
full_existing["configuration"] = "Full model"

full_existing = full_existing.rename(columns={
    "test_r2": "r2",
    "test_rmse": "rmse",
    "test_mae": "mae",
    "test_bias": "bias",
    "test_slope": "slope",
})

print(f"Existing full-model seeds: {len(full_existing):,}")

if set(full_existing["seed"]) != set(SEEDS):
    raise RuntimeError("Existing full metrics do not contain exactly seeds 1-35.")

print("Existing full model:")
print(f"R2   = {full_existing['r2'].mean():.4f} +/- {full_existing['r2'].std(ddof=1):.4f}")
print(f"RMSE = {full_existing['rmse'].mean():.3f} +/- {full_existing['rmse'].std(ddof=1):.3f}")
print(f"MAE  = {full_existing['mae'].mean():.3f} +/- {full_existing['mae'].std(ddof=1):.3f}")


# =============================================================================
# RUN 35 x 3 REDUCED MODELS
# =============================================================================

banner("RUNNING 35-SEED HUC4 FEATURE-GROUP ABLATION")

print("New model fits: 35 seeds x 3 reduced configurations = 105")
print("Full-model results will be reused from the existing HUC4 experiment.")
print()
print("Fixed HUC4 XGBoost configuration:")

for key, value in FIXED_PARAMS.items():
    if key not in ["objective", "eval_metric", "verbosity", "n_jobs", "tree_method"]:
        print(f"  {key:20s}: {value}")

all_results = []
all_predictions = []

overall_start = time.perf_counter()

for seed in SEEDS:
    print()
    print("-" * 110)
    print(f"SEED {seed:02d}/35")
    print("-" * 110)

    manifest_file = MANIFEST_DIR / f"HUC4_seed_{seed:02d}_station_manifest.csv"
    manifest = pd.read_csv(manifest_file)

    required_manifest = ["station_group", "split"]
    missing_manifest = [c for c in required_manifest if c not in manifest.columns]
    if missing_manifest:
        raise KeyError(f"Manifest seed {seed} missing: {missing_manifest}")

    if manifest["station_group"].duplicated().any():
        raise RuntimeError(f"Duplicate station groups in seed {seed} manifest.")

    split_lookup = manifest.set_index("station_group")["split"]

    work = df.copy()
    work["split"] = work["station_group"].map(split_lookup)

    if work["split"].isna().any():
        raise RuntimeError(f"Seed {seed}: some observations were not found in the saved station manifest.")

    train = work[work["split"] == "train"].copy()
    test = work[work["split"] == "test"].copy()

    train_stations = train["station_group"].nunique()
    test_stations = test["station_group"].nunique()

    if train_stations != 3860:
        raise RuntimeError(f"Seed {seed}: expected 3,860 train stations; found {train_stations:,}")
    if test_stations != 1287:
        raise RuntimeError(f"Seed {seed}: expected 1,287 validation stations; found {test_stations:,}")

    y_train = train["chl_a"].astype(float)
    y_test = test["chl_a"].astype(float)

    print(f"Train observations/stations: {len(train):,} / {train_stations:,}")
    print(f"Test observations/stations : {len(test):,} / {test_stations:,}")

    for config_name, features in ABLATIONS.items():
        start = time.perf_counter()

        X_train = train[features]
        X_test = test[features]

        model = XGBRegressor(**FIXED_PARAMS)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        elapsed = time.perf_counter() - start

        m = calculate_metrics(y_test, pred)

        print(
            f"{config_name:24s} | R2={m['r2']:.4f} | RMSE={m['rmse']:.3f} | "
            f"MAE={m['mae']:.3f} | Bias={m['bias']:+.3f} | {elapsed:.1f}s"
        )

        all_results.append({
            "seed": seed,
            "configuration": config_name,
            "n_features": len(features),
            "train_n": len(train),
            "test_n": len(test),
            "train_stations": train_stations,
            "test_stations": test_stations,
            **m,
            "fit_seconds": elapsed,
        })

        pred_df = pd.DataFrame({
            "seed": seed,
            "configuration": config_name,
            "station_group": test["station_group"].to_numpy(),
            "observed_chl_a": y_test.to_numpy(),
            "predicted_chl_a": pred,
            "residual": pred - y_test.to_numpy(),
        })

        all_predictions.append(pred_df)


# =============================================================================
# COMBINE WITH EXISTING FULL MODEL
# =============================================================================

banner("COMBINING REDUCED MODELS WITH EXISTING FULL MODEL")

reduced = pd.DataFrame(all_results)

full_for_merge = full_existing[
    ["seed", "configuration", "r2", "rmse", "mae", "bias", "slope"]
].copy()

full_for_merge["n_features"] = 22
full_for_merge["fit_seconds"] = np.nan

all_metrics = pd.concat([reduced, full_for_merge], ignore_index=True, sort=False)

order = ["Spectral only", "Spectral + temporal", "Spectral + spatial", "Full model"]

all_metrics["configuration"] = pd.Categorical(
    all_metrics["configuration"], categories=order, ordered=True,
)

all_metrics = all_metrics.sort_values(["seed", "configuration"]).reset_index(drop=True)


# =============================================================================
# SUMMARY BY CONFIGURATION
# =============================================================================

summary_rows = []
for config in order:
    x = all_metrics[all_metrics["configuration"] == config].copy()
    row = {
        "configuration": config,
        "n_features": int(x["n_features"].dropna().iloc[0]),
    }
    for metric in ["r2", "rmse", "mae", "bias", "slope"]:
        values = x[metric].astype(float)
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_sd"] = values.std(ddof=1)
        row[f"{metric}_median"] = values.median()
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)


# =============================================================================
# PAIRED COMPARISONS
# =============================================================================

wide = all_metrics.pivot(index="seed", columns="configuration", values=["r2", "rmse", "mae"])

paired_rows = []


def add_contrast(name, base, enhanced):
    for seed in SEEDS:
        base_r2 = float(wide.loc[seed, ("r2", base)])
        enhanced_r2 = float(wide.loc[seed, ("r2", enhanced)])
        base_rmse = float(wide.loc[seed, ("rmse", base)])
        enhanced_rmse = float(wide.loc[seed, ("rmse", enhanced)])
        base_mae = float(wide.loc[seed, ("mae", base)])
        enhanced_mae = float(wide.loc[seed, ("mae", enhanced)])

        paired_rows.append({
            "seed": seed,
            "contrast": name,
            "base": base,
            "enhanced": enhanced,
            # Positive means improvement
            "delta_r2": enhanced_r2 - base_r2,
            # Positive means error was reduced
            "rmse_reduction": base_rmse - enhanced_rmse,
            "rmse_reduction_pct": 100 * (base_rmse - enhanced_rmse) / base_rmse,
            "mae_reduction": base_mae - enhanced_mae,
            "mae_reduction_pct": 100 * (base_mae - enhanced_mae) / base_mae,
        })


add_contrast(name="Temporal added to spectral", base="Spectral only", enhanced="Spectral + temporal")
add_contrast(name="Spatial added to spectral", base="Spectral only", enhanced="Spectral + spatial")
add_contrast(name="Spatial + temporal added to spectral", base="Spectral only", enhanced="Full model")
add_contrast(name="Temporal added after spatial", base="Spectral + spatial", enhanced="Full model")

paired = pd.DataFrame(paired_rows)


# =============================================================================
# SUMMARIZE PAIRED CHANGES
# =============================================================================

paired_summary_rows = []
for contrast in paired["contrast"].drop_duplicates():
    x = paired[paired["contrast"] == contrast].copy()
    row = {"contrast": contrast}
    for metric in ["delta_r2", "rmse_reduction", "rmse_reduction_pct", "mae_reduction", "mae_reduction_pct"]:
        values = x[metric].astype(float)
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_sd"] = values.std(ddof=1)
        row[f"{metric}_median"] = values.median()
        # Useful descriptive check: number of 35 splits for which enhanced model improved.
        row[f"{metric}_positive_n"] = int((values > 0).sum())
    paired_summary_rows.append(row)

paired_summary = pd.DataFrame(paired_summary_rows)


# =============================================================================
# SAVE EVERYTHING
# =============================================================================

METRICS_FILE = OUT / "HUC4_ablation_35seed_metrics.csv"
SUMMARY_FILE = OUT / "HUC4_ablation_35seed_summary.csv"
PAIRED_FILE = OUT / "HUC4_ablation_35seed_paired_changes.csv"
PAIRED_SUMMARY_FILE = OUT / "HUC4_ablation_35seed_paired_summary.csv"
PRED_FILE = OUT / "HUC4_ablation_35seed_reduced_predictions.csv"

all_metrics.to_csv(METRICS_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)
paired.to_csv(PAIRED_FILE, index=False)
paired_summary.to_csv(PAIRED_SUMMARY_FILE, index=False)

pd.concat(all_predictions, ignore_index=True).to_csv(PRED_FILE, index=False)


# =============================================================================
# PRINT MANUSCRIPT-USEFUL RESULTS
# =============================================================================

banner("FINAL 35-SEED HUC4 ABLATION SUMMARY")

for _, row in summary.iterrows():
    print(
        f"{row['configuration']:24s} | R2 = {row['r2_mean']:.4f} +/- {row['r2_sd']:.4f} | "
        f"RMSE = {row['rmse_mean']:.3f} +/- {row['rmse_sd']:.3f} | "
        f"MAE = {row['mae_mean']:.3f} +/- {row['mae_sd']:.3f} | "
        f"Bias = {row['bias_mean']:+.3f} +/- {row['bias_sd']:.3f}"
    )

banner("PAIRED FEATURE-GROUP CONTRIBUTIONS ACROSS THE SAME 35 SPLITS")

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
print(PRED_FILE)

total_minutes = (time.perf_counter() - overall_start) / 60.0

print()
print(f"Total runtime: {total_minutes:.1f} minutes")
print()
print("DONE")

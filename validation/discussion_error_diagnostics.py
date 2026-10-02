#!/usr/bin/env python3
"""
Discussion-section error diagnostics: concentration-dependent bias and error-share tables
across the fixed holdout, 35-seed Monte Carlo, HUC4 station holdout, and Lake Erie <=1h
external validation, plus a validation-hierarchy summary and diagnostic flags. Produces the
CSVs that figures/fig10_concentration_dependent_error.py plots.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

HOLDOUT_FILE = (
    DATA_DIR
    / "model_comparison_6models_final"
    / "models"
    / "xgb"
    / "predictions_holdout.csv"
)

# Produced by validation/xgb_monte_carlo_35_tuned.py
MC_FILE = (
    RESULTS_DIR
    / "xgb_monte_carlo_35"
    / "monte_carlo_xgb_results.csv"
)

# Produced by validation/run_huc4_fixed_35seeds.py
HUC4_DIR = (
    RESULTS_DIR
    / "spatial_validation_diagnostics"
    / "repeated_HUC4_fixed_35seeds"
)

HUC4_PRED_FILE = (
    HUC4_DIR
    / "HUC4_fixed_35seed_all_predictions.csv"
)

HUC4_METRICS_FILE = (
    HUC4_DIR
    / "HUC4_fixed_35seed_metrics.csv"
)

ERIE_FILE = (
    DATA_DIR
    / "lake_erie_validation"
    / "DSWE1_ValidTime_FixedScene_final"
    / "predictions"
    / "LakeErie_XGBoost_predictions_le1h.csv"
)

OUT = RESULTS_DIR / "discussion_error_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPECTED SAMPLE SIZES
# =============================================================================

EXPECTED_HOLDOUT_N = 3532
EXPECTED_ERIE_N = 116  # DSWE1_ValidTime_FixedScene_final, le1h window: n=116, R2=0.532.
# An earlier, separate "positive_spectra_only" external-validation run produced n=120 for
# the same le1h window but R2=-0.17; that run is not used here or anywhere else in this
# pipeline, only the DSWE1_ValidTime_FixedScene_final run above is.
EXPECTED_HUC4_SEEDS = 35
EXPECTED_MC_RUNS = 35


# =============================================================================
# OBSERVED CHL-A BINS
#
# Do NOT calculate R² inside these bins.
# The response variance becomes artificially restricted, making within-bin R²
# unstable and difficult to interpret.
# =============================================================================

BIN_EDGES = [
    -np.inf,
    2,
    5,
    10,
    20,
    50,
    100,
    np.inf,
]

BIN_LABELS = [
    "<2",
    "2–5",
    "5–10",
    "10–20",
    "20–50",
    "50–100",
    "≥100",
]


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def exact_overall_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    good = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[good]
    predicted = predicted[good]

    residual = predicted - observed

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum(
        (observed - np.mean(observed)) ** 2
    )

    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))
    bias = np.mean(residual)

    fit = linregress(observed, predicted)

    return {
        "n": len(observed),
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "pearson_r": fit.rvalue,
        "obs_min": np.min(observed),
        "obs_max": np.max(observed),
        "pred_min": np.min(predicted),
        "pred_max": np.max(predicted),
    }


def prepare_prediction_data(
    df,
    observed_col,
    predicted_col,
):
    work = df.copy()

    work["observed"] = numeric(
        work[observed_col]
    )

    work["predicted"] = numeric(
        work[predicted_col]
    )

    work = work.loc[
        np.isfinite(work["observed"])
        & np.isfinite(work["predicted"])
        & (work["observed"] > 0)
    ].copy()

    work["residual"] = (
        work["predicted"]
        - work["observed"]
    )

    work["absolute_error"] = (
        work["residual"].abs()
    )

    work["squared_error"] = (
        work["residual"] ** 2
    )

    work["absolute_percent_error"] = (
        100.0
        * work["absolute_error"]
        / work["observed"]
    )

    work["pred_obs_ratio"] = (
        work["predicted"]
        / work["observed"]
    )

    work["chl_bin"] = pd.cut(
        work["observed"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=False,
        ordered=True,
    )

    return work


def error_by_bin(
    df,
    observed_col,
    predicted_col,
):
    work = prepare_prediction_data(
        df,
        observed_col,
        predicted_col,
    )

    total_n = len(work)

    total_abs_error = (
        work["absolute_error"].sum()
    )

    total_squared_error = (
        work["squared_error"].sum()
    )

    rows = []

    for label in BIN_LABELS:

        g = work.loc[
            work["chl_bin"].astype(str) == label
        ].copy()

        if len(g) == 0:
            rows.append({
                "chl_bin": label,
                "n": 0,
            })
            continue

        residual = g["residual"]
        ae = g["absolute_error"]
        se = g["squared_error"]
        ratio = g["pred_obs_ratio"]

        rows.append({

            "chl_bin":
                label,

            "n":
                len(g),

            "sample_pct":
                100.0 * len(g) / total_n,

            "obs_min":
                g["observed"].min(),

            "obs_max":
                g["observed"].max(),

            "obs_mean":
                g["observed"].mean(),

            "obs_median":
                g["observed"].median(),

            "pred_mean":
                g["predicted"].mean(),

            "pred_median":
                g["predicted"].median(),

            # Positive = overprediction
            # Negative = underprediction
            "bias":
                residual.mean(),

            "median_error":
                residual.median(),

            "mae":
                ae.mean(),

            "median_abs_error":
                ae.median(),

            "rmse":
                np.sqrt(se.mean()),

            # Useful diagnostically, but interpret cautiously at
            # very low observed concentrations.
            "median_ape_pct":
                g[
                    "absolute_percent_error"
                ].median(),

            "median_pred_obs_ratio":
                ratio.median(),

            "pred_mean_over_obs_mean":
                (
                    g["predicted"].mean()
                    / g["observed"].mean()
                ),

            "pct_overpredicted":
                100.0
                * np.mean(residual > 0),

            "pct_underpredicted":
                100.0
                * np.mean(residual < 0),

            "pct_within_5":
                100.0
                * np.mean(ae <= 5.0),

            "pct_within_10":
                100.0
                * np.mean(ae <= 10.0),

            # Prediction within a factor of two of observation.
            "pct_within_factor2":
                100.0
                * np.mean(
                    (ratio >= 0.5)
                    & (ratio <= 2.0)
                ),

            # How much this Chl-a range contributes to total MAE burden.
            "share_total_abs_error_pct":
                (
                    100.0
                    * ae.sum()
                    / total_abs_error
                    if total_abs_error > 0
                    else np.nan
                ),

            # How much this range contributes to total squared error,
            # and therefore to the overall RMSE.
            "share_total_squared_error_pct":
                (
                    100.0
                    * se.sum()
                    / total_squared_error
                    if total_squared_error > 0
                    else np.nan
                ),
        })

    out = pd.DataFrame(rows)

    out["chl_bin"] = pd.Categorical(
        out["chl_bin"],
        categories=BIN_LABELS,
        ordered=True,
    )

    return (
        out
        .sort_values("chl_bin")
        .reset_index(drop=True)
    )


def print_bin_table(title, table):
    banner(title)

    columns = [
        "chl_bin",
        "n",
        "sample_pct",
        "obs_mean",
        "pred_mean",
        "bias",
        "mae",
        "rmse",
        "median_ape_pct",
        "median_pred_obs_ratio",
        "pct_overpredicted",
        "pct_underpredicted",
        "pct_within_5",
        "pct_within_10",
        "pct_within_factor2",
        "share_total_abs_error_pct",
        "share_total_squared_error_pct",
    ]

    use = [
        c for c in columns
        if c in table.columns
    ]

    print(
        table[use].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def summarize_huc4_bins(per_seed):
    metrics_to_summarize = [
        "n",
        "sample_pct",
        "obs_mean",
        "pred_mean",
        "bias",
        "mae",
        "rmse",
        "median_ape_pct",
        "median_pred_obs_ratio",
        "pct_overpredicted",
        "pct_underpredicted",
        "pct_within_5",
        "pct_within_10",
        "pct_within_factor2",
        "share_total_abs_error_pct",
        "share_total_squared_error_pct",
    ]

    rows = []

    for label in BIN_LABELS:

        g = per_seed.loc[
            per_seed["chl_bin"].astype(str)
            == label
        ].copy()

        row = {
            "chl_bin": label,
            "n_seeds": g["seed"].nunique(),
        }

        for metric in metrics_to_summarize:

            if metric not in g.columns:
                continue

            vals = numeric(g[metric]).dropna()

            if len(vals) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_sd"] = np.nan
            else:
                row[f"{metric}_mean"] = vals.mean()
                row[f"{metric}_sd"] = vals.std(ddof=1)

        rows.append(row)

    out = pd.DataFrame(rows)

    out["chl_bin"] = pd.Categorical(
        out["chl_bin"],
        categories=BIN_LABELS,
        ordered=True,
    )

    return (
        out
        .sort_values("chl_bin")
        .reset_index(drop=True)
    )


# =============================================================================
# CHECK FILES
# =============================================================================

for file in [
    HOLDOUT_FILE,
    MC_FILE,
    HUC4_PRED_FILE,
    HUC4_METRICS_FILE,
    ERIE_FILE,
]:
    require_file(file)


# =============================================================================
# 1. FINAL FIXED HOLDOUT
# =============================================================================

banner("LOADING FINAL FIXED XGBOOST HOLDOUT")

holdout = pd.read_csv(
    HOLDOUT_FILE,
    low_memory=False,
)

required = [
    "observed_chl_a",
    "predicted_chl_a",
]

missing = [
    c for c in required
    if c not in holdout.columns
]

if missing:
    raise KeyError(
        f"Holdout prediction file is missing: {missing}\n"
        f"Columns available:\n{list(holdout.columns)}"
    )

if len(holdout) != EXPECTED_HOLDOUT_N:
    raise RuntimeError(
        f"Expected final holdout n={EXPECTED_HOLDOUT_N:,}, "
        f"but found n={len(holdout):,}.\n"
        "STOPPING so an old prediction file is not accidentally analyzed."
    )

print(
    f"Final fixed holdout observations: {len(holdout):,}"
)

holdout_overall = exact_overall_metrics(
    holdout["observed_chl_a"],
    holdout["predicted_chl_a"],
)

holdout_bins = error_by_bin(
    holdout,
    "observed_chl_a",
    "predicted_chl_a",
)

holdout_bins.to_csv(
    OUT / "fixed_holdout_error_by_chla_bin.csv",
    index=False,
)


# =============================================================================
# 2. MONTE CARLO RANDOM VALIDATION
#
# We use this only for OVERALL performance context because the original
# Monte Carlo script saved run-level metrics, not all outer-test predictions.
# =============================================================================

banner("LOADING 35-RUN RANDOM MONTE CARLO METRICS")

mc = pd.read_csv(
    MC_FILE,
    low_memory=False,
)

if len(mc) != EXPECTED_MC_RUNS:
    raise RuntimeError(
        f"Expected {EXPECTED_MC_RUNS} Monte Carlo runs, "
        f"found {len(mc)}."
    )

required_mc = [
    "test_r2",
    "test_rmse",
    "test_mae",
    "test_bias",
]

missing = [
    c for c in required_mc
    if c not in mc.columns
]

if missing:
    raise KeyError(
        f"Monte Carlo file missing: {missing}\n"
        f"Columns:\n{list(mc.columns)}"
    )

print(
    "Monte Carlo predictions were not saved per run, "
    "so concentration-bin analysis is intentionally NOT "
    "performed for the 35 random splits."
)


# =============================================================================
# 3. HUC4 STATION-LEVEL VALIDATION
# =============================================================================

banner("LOADING 35-SEED HUC4 VALIDATION")

huc = pd.read_csv(
    HUC4_PRED_FILE,
    low_memory=False,
)

huc_metrics = pd.read_csv(
    HUC4_METRICS_FILE,
    low_memory=False,
)

required_huc = [
    "seed",
    "observed_chl_a",
    "predicted_chl_a",
]

missing = [
    c for c in required_huc
    if c not in huc.columns
]

if missing:
    raise KeyError(
        f"HUC4 prediction file missing: {missing}\n"
        f"Columns available:\n{list(huc.columns)}"
    )

n_huc_seeds = huc["seed"].nunique()

if n_huc_seeds != EXPECTED_HUC4_SEEDS:
    raise RuntimeError(
        f"Expected {EXPECTED_HUC4_SEEDS} HUC4 seeds, "
        f"found {n_huc_seeds}."
    )

print(
    f"HUC4 validation predictions: {len(huc):,} rows "
    f"across {n_huc_seeds} seeds"
)

# -------------------------------------------------------------------------
# Analyze EACH seed independently.
#
# This avoids pretending that repeated predictions from overlapping
# resampling experiments are independent observations.
# -------------------------------------------------------------------------

huc_seed_tables = []

for seed in sorted(
    huc["seed"].dropna().unique()
):

    g = huc.loc[
        huc["seed"] == seed
    ].copy()

    t = error_by_bin(
        g,
        "observed_chl_a",
        "predicted_chl_a",
    )

    t.insert(
        0,
        "seed",
        int(seed),
    )

    huc_seed_tables.append(t)

huc_per_seed = pd.concat(
    huc_seed_tables,
    ignore_index=True,
)

huc_per_seed.to_csv(
    OUT / "huc4_error_by_chla_bin_per_seed.csv",
    index=False,
)

huc_summary = summarize_huc4_bins(
    huc_per_seed
)

huc_summary.to_csv(
    OUT / "huc4_error_by_chla_bin_summary.csv",
    index=False,
)

# Also create a pooled table only as a descriptive diagnostic.
# Do NOT use pooled HUC4 values as inferential replicates in the paper.
huc_pooled = error_by_bin(
    huc,
    "observed_chl_a",
    "predicted_chl_a",
)

huc_pooled.to_csv(
    OUT / "huc4_error_by_chla_bin_pooled_DIAGNOSTIC_ONLY.csv",
    index=False,
)


# =============================================================================
# 4. FINAL LAKE ERIE ≤1 HOUR EXTERNAL VALIDATION
# =============================================================================

banner("LOADING FINAL LAKE ERIE ≤1 H EXTERNAL VALIDATION")

erie = pd.read_csv(
    ERIE_FILE,
    low_memory=False,
)

required_erie = [
    "chl_a",
    "predicted_chl_a",
]

missing = [
    c for c in required_erie
    if c not in erie.columns
]

if missing:
    raise KeyError(
        f"Lake Erie file missing: {missing}\n"
        f"Columns available:\n{list(erie.columns)}"
    )

if len(erie) != EXPECTED_ERIE_N:
    raise RuntimeError(
        f"\nExpected FINAL Lake Erie ≤1 h n={EXPECTED_ERIE_N}, "
        f"but found n={len(erie)}.\n\n"
        "This likely means an older Lake Erie output has overwritten "
        "the final file. Do NOT interpret the results until this is resolved."
    )

print(
    f"Final Lake Erie ≤1 h observations: {len(erie):,}"
)

erie_overall = exact_overall_metrics(
    erie["chl_a"],
    erie["predicted_chl_a"],
)

erie_bins = error_by_bin(
    erie,
    "chl_a",
    "predicted_chl_a",
)

erie_bins.to_csv(
    OUT / "lake_erie_le1h_error_by_chla_bin.csv",
    index=False,
)


# =============================================================================
# 5. OVERALL VALIDATION HIERARCHY
# =============================================================================

overall_rows = []


# Fixed holdout
overall_rows.append({
    "validation":
        "Fixed random holdout (seed 42)",

    "n_mean":
        holdout_overall["n"],

    "r2_mean":
        holdout_overall["r2"],

    "r2_sd":
        np.nan,

    "rmse_mean":
        holdout_overall["rmse"],

    "rmse_sd":
        np.nan,

    "mae_mean":
        holdout_overall["mae"],

    "mae_sd":
        np.nan,

    "bias_mean":
        holdout_overall["bias"],

    "bias_sd":
        np.nan,

    "slope_mean":
        holdout_overall["slope"],

    "slope_sd":
        np.nan,

    "pearson_r_mean":
        holdout_overall["pearson_r"],

    "obs_min":
        holdout_overall["obs_min"],

    "obs_max":
        holdout_overall["obs_max"],

    "pred_min":
        holdout_overall["pred_min"],

    "pred_max":
        holdout_overall["pred_max"],
})


# 35-run random MC
overall_rows.append({
    "validation":
        "Random resampling (35 runs)",

    "n_mean":
        (
            mc["n_test"].mean()
            if "n_test" in mc.columns
            else np.nan
        ),

    "r2_mean":
        mc["test_r2"].mean(),

    "r2_sd":
        mc["test_r2"].std(ddof=1),

    "rmse_mean":
        mc["test_rmse"].mean(),

    "rmse_sd":
        mc["test_rmse"].std(ddof=1),

    "mae_mean":
        mc["test_mae"].mean(),

    "mae_sd":
        mc["test_mae"].std(ddof=1),

    "bias_mean":
        mc["test_bias"].mean(),

    "bias_sd":
        mc["test_bias"].std(ddof=1),

    "slope_mean":
        np.nan,

    "slope_sd":
        np.nan,

    "pearson_r_mean":
        np.nan,

    "obs_min":
        np.nan,

    "obs_max":
        np.nan,

    "pred_min":
        np.nan,

    "pred_max":
        np.nan,
})


# HUC4 35-run metrics
required_huc_metrics = [
    "test_r2",
    "test_rmse",
    "test_mae",
    "test_bias",
    "test_slope",
]

missing = [
    c for c in required_huc_metrics
    if c not in huc_metrics.columns
]

if missing:
    raise KeyError(
        f"HUC4 metrics file missing: {missing}"
    )

overall_rows.append({
    "validation":
        "HUC4 station holdout (35 runs)",

    "n_mean":
        (
            huc_metrics["test_n"].mean()
            if "test_n" in huc_metrics.columns
            else np.nan
        ),

    "r2_mean":
        huc_metrics["test_r2"].mean(),

    "r2_sd":
        huc_metrics["test_r2"].std(ddof=1),

    "rmse_mean":
        huc_metrics["test_rmse"].mean(),

    "rmse_sd":
        huc_metrics["test_rmse"].std(ddof=1),

    "mae_mean":
        huc_metrics["test_mae"].mean(),

    "mae_sd":
        huc_metrics["test_mae"].std(ddof=1),

    "bias_mean":
        huc_metrics["test_bias"].mean(),

    "bias_sd":
        huc_metrics["test_bias"].std(ddof=1),

    "slope_mean":
        huc_metrics["test_slope"].mean(),

    "slope_sd":
        huc_metrics["test_slope"].std(ddof=1),

    "pearson_r_mean":
        (
            huc_metrics["test_pearson_r"].mean()
            if "test_pearson_r"
            in huc_metrics.columns
            else np.nan
        ),

    "obs_min":
        np.nan,

    "obs_max":
        np.nan,

    "pred_min":
        np.nan,

    "pred_max":
        np.nan,
})


# Lake Erie
overall_rows.append({
    "validation":
        "Lake Erie external ≤1 h",

    "n_mean":
        erie_overall["n"],

    "r2_mean":
        erie_overall["r2"],

    "r2_sd":
        np.nan,

    "rmse_mean":
        erie_overall["rmse"],

    "rmse_sd":
        np.nan,

    "mae_mean":
        erie_overall["mae"],

    "mae_sd":
        np.nan,

    "bias_mean":
        erie_overall["bias"],

    "bias_sd":
        np.nan,

    "slope_mean":
        erie_overall["slope"],

    "slope_sd":
        np.nan,

    "pearson_r_mean":
        erie_overall["pearson_r"],

    "obs_min":
        erie_overall["obs_min"],

    "obs_max":
        erie_overall["obs_max"],

    "pred_min":
        erie_overall["pred_min"],

    "pred_max":
        erie_overall["pred_max"],
})


overall = pd.DataFrame(
    overall_rows
)

overall.to_csv(
    OUT / "overall_validation_comparison.csv",
    index=False,
)


# =============================================================================
# 6. DIAGNOSTIC FLAGS
# =============================================================================

def get_extreme_bin(
    table,
    metric,
    mode="max",
):
    x = table.dropna(
        subset=[metric]
    ).copy()

    if len(x) == 0:
        return None

    if mode == "max":
        row = x.loc[
            x[metric].idxmax()
        ]
    else:
        row = x.loc[
            x[metric].idxmin()
        ]

    return (
        str(row["chl_bin"]),
        float(row[metric]),
    )


def print_flags(
    name,
    table,
):
    banner(
        f"KEY ERROR FLAGS — {name}"
    )

    positive_bias = get_extreme_bin(
        table,
        "bias",
        "max",
    )

    negative_bias = get_extreme_bin(
        table,
        "bias",
        "min",
    )

    largest_rmse = get_extreme_bin(
        table,
        "rmse",
        "max",
    )

    largest_ape = get_extreme_bin(
        table,
        "median_ape_pct",
        "max",
    )

    largest_sse = get_extreme_bin(
        table,
        "share_total_squared_error_pct",
        "max",
    )

    largest_ae = get_extreme_bin(
        table,
        "share_total_abs_error_pct",
        "max",
    )

    print(
        "Largest positive bias:           ",
        positive_bias,
    )

    print(
        "Largest negative bias:           ",
        negative_bias,
    )

    print(
        "Largest RMSE:                    ",
        largest_rmse,
    )

    print(
        "Largest median percentage error: ",
        largest_ape,
    )

    print(
        "Largest share of total AE:       ",
        largest_ae,
    )

    print(
        "Largest share of total SE/RMSE:  ",
        largest_sse,
    )


# =============================================================================
# 7. PRINT RESULTS
# =============================================================================

banner(
    "A. OVERALL VALIDATION HIERARCHY"
)

overall_print_cols = [
    "validation",
    "n_mean",
    "r2_mean",
    "r2_sd",
    "rmse_mean",
    "rmse_sd",
    "mae_mean",
    "mae_sd",
    "bias_mean",
    "bias_sd",
    "slope_mean",
    "slope_sd",
    "pearson_r_mean",
]

print(
    overall[
        overall_print_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print_bin_table(
    "B. FIXED RANDOM HOLDOUT — ERROR BY OBSERVED CHL-A RANGE",
    holdout_bins,
)


banner(
    "C. HUC4 STATION HOLDOUT — ERROR BY OBSERVED CHL-A RANGE"
)
print(
    "Values below are means across the 35 HUC4 validation splits.\n"
    "SDs are saved in the CSV and selected SDs are also shown here.\n"
)

huc_print_cols = [
    "chl_bin",
    "n_seeds",
    "n_mean",
    "sample_pct_mean",
    "obs_mean_mean",
    "pred_mean_mean",
    "bias_mean",
    "bias_sd",
    "mae_mean",
    "mae_sd",
    "rmse_mean",
    "rmse_sd",
    "median_ape_pct_mean",
    "median_pred_obs_ratio_mean",
    "pct_overpredicted_mean",
    "pct_underpredicted_mean",
    "pct_within_5_mean",
    "pct_within_10_mean",
    "pct_within_factor2_mean",
    "share_total_abs_error_pct_mean",
    "share_total_squared_error_pct_mean",
]

print(
    huc_summary[
        huc_print_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print_bin_table(
    "D. LAKE ERIE ≤1 H — ERROR BY OBSERVED CHL-A RANGE",
    erie_bins,
)


print_flags(
    "FIXED RANDOM HOLDOUT",
    holdout_bins,
)

print_flags(
    "LAKE ERIE ≤1 H",
    erie_bins,
)


# HUC4 flag table using mean-across-seed quantities
huc_flag_table = pd.DataFrame({
    "chl_bin":
        huc_summary["chl_bin"],

    "bias":
        huc_summary["bias_mean"],

    "rmse":
        huc_summary["rmse_mean"],

    "median_ape_pct":
        huc_summary[
            "median_ape_pct_mean"
        ],

    "share_total_abs_error_pct":
        huc_summary[
            "share_total_abs_error_pct_mean"
        ],

    "share_total_squared_error_pct":
        huc_summary[
            "share_total_squared_error_pct_mean"
        ],
})

print_flags(
    "HUC4 STATION HOLDOUT",
    huc_flag_table,
)


# =============================================================================
# 8. SAVE A COMPACT TEXT REPORT
# =============================================================================

report_file = (
    OUT
    / "PASTE_BACK_discussion_error_summary.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "=" * 120
        + "\nA. OVERALL VALIDATION HIERARCHY\n"
        + "=" * 120
        + "\n"
    )

    f.write(
        overall[
            overall_print_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    f.write(
        "\n\n"
        + "=" * 120
        + "\nB. FIXED HOLDOUT — ERROR BY CHL-A RANGE\n"
        + "=" * 120
        + "\n"
    )

    f.write(
        holdout_bins.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    f.write(
        "\n\n"
        + "=" * 120
        + "\nC. HUC4 — MEAN ERROR BY CHL-A RANGE ACROSS 35 SPLITS\n"
        + "=" * 120
        + "\n"
    )

    f.write(
        huc_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    f.write(
        "\n\n"
        + "=" * 120
        + "\nD. LAKE ERIE ≤1 H — ERROR BY CHL-A RANGE\n"
        + "=" * 120
        + "\n"
    )

    f.write(
        erie_bins.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# =============================================================================
# OUTPUTS
# =============================================================================

banner("FILES SAVED")

for file in sorted(OUT.iterdir()):
    print(file)

print()
print("=" * 120)
print("SUMMARY REPORT")
print("=" * 120)
print(report_file)
print()
print(
    "NOTE: Per-bin R² was intentionally excluded because restricting "
    "the observed concentration range makes R² unstable and potentially "
    "misleading. Bias, MAE, RMSE, error direction, ratios, and error-share "
    "metrics are more appropriate for this diagnostic."
)


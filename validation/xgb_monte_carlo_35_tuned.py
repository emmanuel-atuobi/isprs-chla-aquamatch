#!/usr/bin/env python3
"""
Monte Carlo validation for tuned XGBoost AquaMatch Chl-a model
===============================================================

Design:
- Quality filtering
- timediff <= 2 hours
- no Green/Blue target-conditioned filter
- no additional cloud-percentage filter
- 22 predictors
- 35 repeated random 75/25 train/test splits
- independent RandomizedSearchCV inside each training split
- 5-fold inner cross-validation
- untouched outer test set for each run

Outputs:
- monte_carlo_xgb_results.csv
- monte_carlo_xgb_summary.csv
- monte_carlo_xgb_best_parameters.csv
- monte_carlo_xgb_r2_boxplot.png/.pdf
- monte_carlo_xgb_metrics_boxplot.png/.pdf
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from packaging.version import Version
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))
INPUT_FILE = DATA_DIR / "raw" / "aquamatch_sitesr_joined.csv"
RESULT_DIR = RESULTS_DIR / "xgb_monte_carlo_35"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 35
TEST_SIZE = 0.25
TIME_DIFFERENCE_HOURS = 2

# Each outer split gets its own independent hyperparameter search.
INNER_CV_FOLDS = 5
SEARCH_ITERATIONS_PER_RUN = 30

# Reproducible outer-split seeds.
RUN_SEEDS = list(range(42, 42 + N_RUNS))

USE_GPU = True
GPU_SEARCH_JOBS = 1
CPU_SEARCH_JOBS = -1
XGB_CPU_THREADS = 8

FIGURE_DPI = 600


# =============================================================================
# FEATURES
# =============================================================================

FEATURES = [
    "red",
    "nir",
    "blue",
    "lat",
    "long",
    "NDVI",
    "NDTI",
    "RNI",
    "GBI",
    "BLRDGR",
    "GNRI",
    "RBI",
    "NIRGI",
    "GDVI",
    "NDAVI",
    "FAI",
    "MNDWI",
    "SWI",
    "TGI",
    "AFAI",
    "sin_doy",
    "cos_doy",
]


# =============================================================================
# HELPERS
# =============================================================================

def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  " + "\n  ".join(missing)
        )


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return np.nan
    return 1.0 - ((1.0 - r2) * (n - 1.0) / (n - p - 1.0))


def evaluate(
    observed: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    n_features: int,
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    r2 = float(r2_score(observed, predicted))
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    mae = float(mean_absolute_error(observed, predicted))
    bias = float(np.mean(predicted - observed))

    return {
        "n": int(len(observed)),
        "r2": r2,
        "adjusted_r2": float(adjusted_r2(r2, len(observed), n_features)),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
    }


def save_figure(figure: plt.Figure, basename: str) -> None:
    png_path = RESULT_DIR / f"{basename}.png"
    pdf_path = RESULT_DIR / f"{basename}.pdf"

    figure.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =============================================================================
# LOAD, FILTER, AND ENGINEER FEATURES
# =============================================================================

print("=" * 100)
print("LOADING AND FILTERING AQUAMATCH")
print("=" * 100)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Starting joined rows: {len(df):,}")

df = df.rename(
    columns={
        "med_Blue": "blue",
        "med_Green": "green",
        "med_Red": "red",
        "med_Nir": "nir",
        "med_Swir1": "swir1",
        "med_Swir2": "swir2",
        "harmonized_value": "chl_a",
        "lon": "long",
    }
)

required = [
    "ResolvedMonitoringLocationTypeName",
    "mission",
    "chl_a",
    "mdl_flag",
    "harmonized_row_count",
    "harmonized_value_cv",
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
require_columns(df, required)

numeric_columns = [
    "chl_a",
    "harmonized_value_cv",
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

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df[
    df["ResolvedMonitoringLocationTypeName"]
    == "Lake, Reservoir, Impoundment"
].copy()

df = df[
    df["mission"].isin(["LT05", "LE07", "LC08", "LC09"])
].copy()

df = df[
    np.isfinite(df["chl_a"])
    & (df["chl_a"] > 0)
    & (df["chl_a"] <= 200)
].copy()

df = df[df["mdl_flag"] == 0].copy()

replicate_mask = (
    (df["harmonized_row_count"] == 1)
    | (df["harmonized_value_cv"] <= 0.5)
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
    & (
        df["harmonized_bottom_depth_value"]
        >= df["harmonized_top_depth_value"]
    )
    & (df["harmonized_bottom_depth_value"] <= 2)
)

df = df[discrete_mask | integrated_mask].copy()

for column in ["blue", "green", "red", "nir", "swir1", "swir2"]:
    df = df[
        np.isfinite(df[column])
        & (df[column] > 0)
    ].copy()

df = df[df["pCount_dswe1"] >= 8].copy()

df = df[
    np.isfinite(df["timediff"])
    & (df["timediff"] <= TIME_DIFFERENCE_HOURS)
].copy()

with np.errstate(divide="ignore", invalid="ignore"):
    df["NDVI"] = (
        (df["nir"] - df["red"])
        / (df["nir"] + df["red"])
    )

    df["NDTI"] = (
        (df["red"] - df["green"])
        / (df["red"] + df["green"])
    )

    df["RNI"] = df["red"] / df["nir"]
    df["GBI"] = df["green"] / df["blue"]
    df["BLRDGR"] = (df["blue"] - df["red"]) / df["green"]
    df["GNRI"] = df["green"] - (df["green"] / df["red"])
    df["RBI"] = df["red"] / df["blue"]
    df["NIRGI"] = df["nir"] / df["green"]
    df["GDVI"] = df["nir"] - df["green"]

    df["NDAVI"] = (
        (df["nir"] - df["blue"])
        / (df["nir"] + df["blue"])
    )

    baseline = (
        df["red"]
        + (
            (df["swir1"] - df["red"])
            * (0.86 - 0.66)
            / (1.60 - 0.66)
        )
    )
    df["FAI"] = df["nir"] - baseline

    df["MNDWI"] = (
        (df["green"] - df["swir1"])
        / (df["green"] + df["swir1"])
    )

    df["SWI"] = (
        (df["nir"] - df["swir1"])
        / (df["nir"] + df["swir1"])
    )

    df["TGI"] = -0.5 * (
        (120 * (df["red"] - df["green"]))
        - (190 * (df["red"] - df["blue"]))
    )

    df["AFAI"] = (
        (df["nir"] - df["red"])
        + 0.5 * (df["swir1"] - df["red"])
    )

date_column = (
    "ActivityStartDate"
    if "ActivityStartDate" in df.columns
    else "date"
)
require_columns(df, [date_column])

df["_date"] = pd.to_datetime(
    df[date_column],
    errors="coerce",
    format="mixed",
    dayfirst=False,
)

day_of_year = df["_date"].dt.dayofyear
df["sin_doy"] = np.sin(
    2 * np.pi * (day_of_year - 1) / 365.0
)
df["cos_doy"] = np.cos(
    2 * np.pi * (day_of_year - 1) / 365.0
)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + ["chl_a"]).reset_index(drop=True)

X = df[FEATURES].copy()
y = df["chl_a"].copy()

print(f"Final model-ready observations: {len(df):,}")
print(f"Predictors: {len(FEATURES)}")
print(f"Monte Carlo runs: {N_RUNS}")
print(
    f"Independent tuning per run: "
    f"{SEARCH_ITERATIONS_PER_RUN} candidates × "
    f"{INNER_CV_FOLDS} folds"
)
print(
    f"Total planned CV fits: "
    f"{N_RUNS * SEARCH_ITERATIONS_PER_RUN * INNER_CV_FOLDS:,}"
)


# =============================================================================
# XGBOOST RUNTIME
# =============================================================================

if USE_GPU:
    if Version(xgb.__version__) >= Version("2.0.0"):
        runtime_parameters = {
            "tree_method": "hist",
            "device": "cuda",
        }
    else:
        runtime_parameters = {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
        }

    search_jobs = GPU_SEARCH_JOBS
else:
    if Version(xgb.__version__) >= Version("2.0.0"):
        runtime_parameters = {
            "tree_method": "hist",
            "device": "cpu",
        }
    else:
        runtime_parameters = {
            "tree_method": "hist",
        }

    search_jobs = CPU_SEARCH_JOBS

print(f"XGBoost version: {xgb.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"Runtime parameters: {runtime_parameters}")
print(f"Search jobs: {search_jobs}")


# =============================================================================
# HYPERPARAMETER SPACE
# =============================================================================

parameter_distributions = {
    "n_estimators": randint(300, 1801),
    "max_depth": randint(3, 11),
    "learning_rate": loguniform(0.015, 0.20),
    "subsample": uniform(0.65, 0.35),
    "colsample_bytree": uniform(0.55, 0.45),
    "min_child_weight": loguniform(0.5, 20.0),
    "gamma": uniform(0.0, 1.0),
    "reg_alpha": loguniform(1e-5, 5.0),
    "reg_lambda": loguniform(0.10, 20.0),
    "max_bin": [128, 256, 512],
}


# =============================================================================
# MONTE CARLO VALIDATION
# =============================================================================

result_rows = []
parameter_rows = []

overall_start = time.perf_counter()

for run_number, split_seed in enumerate(RUN_SEEDS, start=1):
    print("\n" + "=" * 100)
    print(
        f"MONTE CARLO RUN {run_number:02d}/{N_RUNS} "
        f"| OUTER SPLIT SEED = {split_seed}"
    )
    print("=" * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=split_seed,
    )

    inner_cv = KFold(
        n_splits=INNER_CV_FOLDS,
        shuffle=True,
        random_state=split_seed,
    )

    base_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=split_seed,
        n_jobs=XGB_CPU_THREADS,
        verbosity=0,
        **runtime_parameters,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_distributions,
        n_iter=SEARCH_ITERATIONS_PER_RUN,
        scoring={
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        refit="r2",
        cv=inner_cv,
        random_state=split_seed,
        n_jobs=search_jobs,
        verbose=1,
        return_train_score=False,
        error_score="raise",
        pre_dispatch=1 if USE_GPU else "2*n_jobs",
    )

    run_start = time.perf_counter()
    search.fit(X_train, y_train)
    run_seconds = time.perf_counter() - run_start

    best_model = search.best_estimator_
    test_predictions = best_model.predict(X_test)
    train_predictions = best_model.predict(X_train)

    train_metrics = evaluate(
        y_train,
        train_predictions,
        len(FEATURES),
    )
    test_metrics = evaluate(
        y_test,
        test_predictions,
        len(FEATURES),
    )

    cv_table = pd.DataFrame(search.cv_results_)
    best_index = int(search.best_index_)

    best_cv_r2 = float(
        cv_table.loc[best_index, "mean_test_r2"]
    )
    best_cv_r2_sd = float(
        cv_table.loc[best_index, "std_test_r2"]
    )
    best_cv_rmse = float(
        -cv_table.loc[best_index, "mean_test_rmse"]
    )
    best_cv_mae = float(
        -cv_table.loc[best_index, "mean_test_mae"]
    )

    result_row = {
        "run": run_number,
        "split_seed": split_seed,
        "search_seed": split_seed,
        "n_total": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "search_iterations": SEARCH_ITERATIONS_PER_RUN,
        "inner_cv_folds": INNER_CV_FOLDS,
        "best_cv_r2": best_cv_r2,
        "best_cv_r2_sd": best_cv_r2_sd,
        "best_cv_rmse": best_cv_rmse,
        "best_cv_mae": best_cv_mae,
        "train_r2": train_metrics["r2"],
        "train_adjusted_r2": train_metrics["adjusted_r2"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_bias": train_metrics["bias"],
        "test_r2": test_metrics["r2"],
        "test_adjusted_r2": test_metrics["adjusted_r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_bias": test_metrics["bias"],
        "run_minutes": run_seconds / 60.0,
    }
    result_rows.append(result_row)

    parameter_row = {
        "run": run_number,
        "split_seed": split_seed,
        **search.best_params_,
    }
    parameter_rows.append(parameter_row)

    # Save after every run so progress is not lost if a job stops.
    pd.DataFrame(result_rows).to_csv(
        RESULT_DIR / "monte_carlo_xgb_results.csv",
        index=False,
    )
    pd.DataFrame(parameter_rows).to_csv(
        RESULT_DIR / "monte_carlo_xgb_best_parameters.csv",
        index=False,
    )

    cv_table.to_csv(
        RESULT_DIR / f"run_{run_number:02d}_seed_{split_seed}_cv_results.csv",
        index=False,
    )

    best_model.save_model(
        RESULT_DIR / f"run_{run_number:02d}_seed_{split_seed}_best_model.json"
    )

    print(
        f"Best CV R²: {best_cv_r2:.4f} ± {best_cv_r2_sd:.4f}"
    )
    print(
        "Outer test: "
        f"R²={test_metrics['r2']:.4f}, "
        f"Adjusted R²={test_metrics['adjusted_r2']:.4f}, "
        f"RMSE={test_metrics['rmse']:.2f}, "
        f"MAE={test_metrics['mae']:.2f}, "
        f"Bias={test_metrics['bias']:+.2f}"
    )
    print(f"Run time: {run_seconds / 60.0:.2f} minutes")


# =============================================================================
# SUMMARIES
# =============================================================================

results = pd.DataFrame(result_rows)
parameters = pd.DataFrame(parameter_rows)

metric_columns = [
    "test_r2",
    "test_adjusted_r2",
    "test_rmse",
    "test_mae",
    "test_bias",
    "best_cv_r2",
]

summary_rows = []

for metric in metric_columns:
    values = results[metric].dropna()

    summary_rows.append(
        {
            "metric": metric,
            "n_runs": len(values),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "sem": values.sem(ddof=1),
            "median": values.median(),
            "q1": values.quantile(0.25),
            "q3": values.quantile(0.75),
            "minimum": values.min(),
            "maximum": values.max(),
            "ci95_lower_normal": (
                values.mean() - 1.96 * values.sem(ddof=1)
            ),
            "ci95_upper_normal": (
                values.mean() + 1.96 * values.sem(ddof=1)
            ),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    RESULT_DIR / "monte_carlo_xgb_summary.csv",
    index=False,
)

run_metadata = {
    "n_runs": N_RUNS,
    "run_seeds": RUN_SEEDS,
    "test_size": TEST_SIZE,
    "inner_cv_folds": INNER_CV_FOLDS,
    "search_iterations_per_run": SEARCH_ITERATIONS_PER_RUN,
    "total_cv_fits": (
        N_RUNS
        * SEARCH_ITERATIONS_PER_RUN
        * INNER_CV_FOLDS
    ),
    "n_total": len(X),
    "features": FEATURES,
    "runtime_parameters": runtime_parameters,
    "xgboost_version": xgb.__version__,
    "sklearn_version": sklearn.__version__,
    "total_hours": (
        time.perf_counter() - overall_start
    ) / 3600.0,
}

with open(
    RESULT_DIR / "monte_carlo_run_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(run_metadata, file, indent=2)


# =============================================================================
# R² BOXPLOT
# =============================================================================

r2_values = results["test_r2"].to_numpy()

figure, axis = plt.subplots(figsize=(7.2, 7.6))

box = axis.boxplot(
    r2_values,
    vert=True,
    widths=0.38,
    patch_artist=True,
    showmeans=True,
    meanline=True,
    boxprops={
        "facecolor": "#DDEAF7",
        "edgecolor": "#222222",
        "linewidth": 1.6,
    },
    medianprops={
        "color": "#222222",
        "linewidth": 2.2,
    },
    whiskerprops={
        "color": "#222222",
        "linewidth": 1.5,
    },
    capprops={
        "color": "#222222",
        "linewidth": 1.5,
    },
    meanprops={
        "color": "#C62828",
        "linewidth": 2.0,
    },
    flierprops={
        "marker": "o",
        "markersize": 5,
        "markerfacecolor": "none",
        "markeredgecolor": "#444444",
        "alpha": 0.8,
    },
)

rng = np.random.default_rng(42)
jitter = rng.normal(
    loc=1.0,
    scale=0.035,
    size=len(r2_values),
)

axis.scatter(
    jitter,
    r2_values,
    s=34,
    alpha=0.72,
    edgecolors="white",
    linewidths=0.45,
    zorder=3,
)

mean_r2 = results["test_r2"].mean()
std_r2 = results["test_r2"].std(ddof=1)
median_r2 = results["test_r2"].median()

axis.set_title(
    "Monte Carlo Validation of Tuned XGBoost",
    fontsize=18,
    fontweight="bold",
    pad=16,
)
axis.text(
    0.5,
    1.01,
    (
        f"35 independently tuned train/test splits | "
        f"Mean R² = {mean_r2:.3f} ± {std_r2:.3f} | "
        f"Median = {median_r2:.3f}"
    ),
    transform=axis.transAxes,
    ha="center",
    va="bottom",
    fontsize=11.5,
)

axis.set_ylabel("Outer-test R²", fontsize=14)
axis.set_xticks([1])
axis.set_xticklabels(["XGBoost"], fontsize=12)
axis.tick_params(axis="y", labelsize=12)
axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.8,
    alpha=0.35,
)

figure.tight_layout()
save_figure(
    figure,
    "monte_carlo_xgb_r2_boxplot",
)
plt.close(figure)


# =============================================================================
# MULTI-METRIC BOXPLOT
# =============================================================================

figure, axes = plt.subplots(
    1,
    3,
    figsize=(15.5, 6.2),
)

metric_specs = [
    ("test_r2", "Outer-test R²", "R²"),
    ("test_rmse", "Outer-test RMSE", "RMSE (µg/L)"),
    ("test_bias", "Outer-test Bias", "Bias (µg/L)"),
]

for axis, (column, title, ylabel) in zip(axes, metric_specs):
    values = results[column].to_numpy()

    axis.boxplot(
        values,
        vert=True,
        widths=0.38,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops={
            "facecolor": "#DDEAF7",
            "edgecolor": "#222222",
            "linewidth": 1.4,
        },
        medianprops={
            "color": "#222222",
            "linewidth": 2.0,
        },
        whiskerprops={
            "color": "#222222",
            "linewidth": 1.3,
        },
        capprops={
            "color": "#222222",
            "linewidth": 1.3,
        },
        meanprops={
            "color": "#C62828",
            "linewidth": 1.8,
        },
    )

    jitter = rng.normal(
        loc=1.0,
        scale=0.035,
        size=len(values),
    )
    axis.scatter(
        jitter,
        values,
        s=28,
        alpha=0.70,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_ylabel(ylabel, fontsize=12)
    axis.set_xticks([])
    axis.tick_params(axis="y", labelsize=11)
    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.32,
    )

figure.suptitle(
    "Monte Carlo Validation Across 35 Independently Tuned XGBoost Models",
    fontsize=18,
    fontweight="bold",
    y=1.02,
)

figure.tight_layout()
save_figure(
    figure,
    "monte_carlo_xgb_metrics_boxplot",
)
plt.close(figure)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("MONTE CARLO VALIDATION COMPLETE")
print("=" * 100)

print(
    f"Outer-test R²: "
    f"{mean_r2:.4f} ± {std_r2:.4f}"
)
print(
    f"Outer-test RMSE: "
    f"{results['test_rmse'].mean():.4f} ± "
    f"{results['test_rmse'].std(ddof=1):.4f}"
)
print(
    f"Outer-test Bias: "
    f"{results['test_bias'].mean():+.4f} ± "
    f"{results['test_bias'].std(ddof=1):.4f}"
)
print(
    f"Total elapsed time: "
    f"{(time.perf_counter() - overall_start) / 3600.0:.2f} hours"
)
print(f"Results directory: {RESULT_DIR}")

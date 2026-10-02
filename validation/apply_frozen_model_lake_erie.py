#!/usr/bin/env python3
"""
Applies the frozen final XGBoost Chl-a model to the Lake Erie matchup CSV, producing the
per-time-window parity plots and metrics used for external validation.

IMPORTANT: requires XGBoost >= 3.3.0 -- older versions silently produce wrong predictions
from this model file, with no error. This script refuses to run under an older XGBoost
rather than fail silently.

Usage:
    python apply_frozen_model_lake_erie.py --input-csv /path/to/matchups.csv \
        --output-dir /path/to/results [--force]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from packaging.version import Version

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import linregress, pearsonr, spearmanr
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

MIN_XGBOOST_VERSION = "3.3.0"

MODEL_FILE = Path(os.environ.get("AQUAMATCH_MODEL_FILE", "./model/xgboost_frozen_v1.json"))

FEATURES = [
    "red", "nir", "blue", "lat", "long",
    "NDVI", "NDTI", "RNI", "GBI", "BLRDGR", "GNRI", "RBI",
    "NIRGI", "GDVI", "NDAVI", "FAI", "MNDWI", "SWI", "TGI", "AFAI",
    "sin_doy", "cos_doy",
]
BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

WINDOWS = [
    ("le1h", "≤ 1 h", 1.0),
    ("le2h", "≤ 2 h", 2.0),
    ("le6h", "≤ 6 h", 6.0),
    ("le24h", "≤ 24 h", 24.0),
    ("all", "All matched", np.inf),
]

DPI = 600
PARITY_MIN = 0.5
PARITY_MAX = 250.0
PARITY_TICKS = [1.0, 10.0, 100.0]
PARITY_TICK_LABELS = ["1", "10", "100"]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_xgboost_version():
    if Version(xgb.__version__) < Version(MIN_XGBOOST_VERSION):
        sys.exit(
            f"ERROR: XGBoost {xgb.__version__} detected, but this model requires "
            f">= {MIN_XGBOOST_VERSION} to predict correctly (older versions silently "
            f"produce wrong predictions from this file -- no error, just bad numbers).\n"
            f"Install a newer XGBoost in your environment and retry."
        )


def load_model():
    print("XGBoost version:", xgb.__version__)
    print("Model file:", MODEL_FILE)
    print("Model MD5:", md5_file(MODEL_FILE))

    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)

    saved_features = model.get_booster().feature_names
    if saved_features is not None and list(saved_features) != FEATURES:
        raise RuntimeError("Saved model feature order does not match FEATURES. Refusing to predict.")

    print("Model class:", type(model))
    return model


def load_and_screen(input_csv):
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Original rows: {len(df):,}")

    df = df.loc[df["match_status"].astype(str).str.lower().eq("matched")].copy()
    print(f"Matched rows: {len(df):,}")

    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Exact duplicates removed: {before - len(df):,}")

    for b in BANDS:
        df[b] = pd.to_numeric(df[b], errors="coerce")

    df["time_diff_hours"] = pd.to_numeric(df["time_diff_hours"], errors="coerce")
    df["abs_time_diff_hours"] = df["time_diff_hours"].abs()

    positive = (df[BANDS] > 0).all(axis=1)
    print(f"Rows with >=1 non-positive band: {(~positive).sum():,}")
    df = df.loc[positive].copy()
    print(f"Positive-spectrum rows: {len(df):,}")

    df = df.rename(columns={
        "Chlorophyll (µg/L)": "chl_a", "longitude": "long", "latitude": "lat",
        "Blue": "blue", "Green": "green", "Red": "red",
        "NIR": "nir", "SWIR1": "swir1", "SWIR2": "swir2",
    })
    numeric = ["chl_a", "long", "lat", "blue", "green", "red", "nir", "swir1", "swir2"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def engineer_features(df):
    df = df.copy()
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
    baseline = df["red"] + (df["swir1"] - df["red"]) * (0.86 - 0.66) / (1.60 - 0.66)
    df["FAI"] = df["nir"] - baseline
    df["MNDWI"] = (df["green"] - df["swir1"]) / (df["green"] + df["swir1"] + eps)
    df["SWI"] = (df["nir"] - df["swir1"]) / (df["nir"] + df["swir1"] + eps)
    df["TGI"] = -0.5 * (120 * (df["red"] - df["green"]) - 190 * (df["red"] - df["blue"]))
    df["AFAI"] = (df["nir"] - df["red"]) + 0.5 * (df["swir1"] - df["red"])

    sample_dt = pd.to_datetime(df["sample_datetime_local"], errors="coerce")
    doy = sample_dt.dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * (doy - 1) / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * (doy - 1) / 365.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    complete = df[FEATURES + ["chl_a", "abs_time_diff_hours"]].notna().all(axis=1)
    print(f"Incomplete engineered rows: {(~complete).sum():,}")
    return df.loc[complete].copy().reset_index(drop=True)


def calculate_metrics(data):
    y = data["chl_a"].to_numpy(dtype=float)
    p = data["predicted_chl_a"].to_numpy(dtype=float)
    fit = linregress(y, p)
    return {
        "n": int(len(y)),
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "bias": float(np.mean(p - y)),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "pearson_r": float(pearsonr(y, p)[0]),
        "spearman_rho": float(spearmanr(y, p).statistic),
        "observed_min": float(np.min(y)), "observed_max": float(np.max(y)),
        "predicted_min": float(np.min(p)), "predicted_max": float(np.max(p)),
    }


def relative_density(observed, predicted, bins=55, sigma=1.10):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    xp = np.clip(observed, PARITY_MIN, PARITY_MAX)
    yp = np.clip(predicted, PARITY_MIN, PARITY_MAX)
    lx, ly = np.log10(xp), np.log10(yp)
    lo, hi = np.log10(PARITY_MIN), np.log10(PARITY_MAX)
    H, xedges, yedges = np.histogram2d(lx, ly, bins=bins, range=[[lo, hi], [lo, hi]])
    H = gaussian_filter(H.astype(float), sigma=sigma)
    ix = np.clip(np.searchsorted(xedges, lx, side="right") - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.searchsorted(yedges, ly, side="right") - 1, 0, H.shape[1] - 1)
    density = np.log1p(H[ix, iy])
    dmin, dmax = np.nanmin(density), np.nanmax(density)
    return (density - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(density)


def format_parity_axis(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(PARITY_MIN, PARITY_MAX); ax.set_ylim(PARITY_MIN, PARITY_MAX)
    ax.set_xticks(PARITY_TICKS); ax.set_yticks(PARITY_TICKS)
    ax.set_xticklabels(PARITY_TICK_LABELS); ax.set_yticklabels(PARITY_TICK_LABELS)
    ax.set_box_aspect(1.0)
    ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=1.0)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.7)
    ax.grid(True, which="major", linewidth=0.7, alpha=0.18)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.07)


def plot_parity_panel(ax, observed, predicted, title):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    metrics_local = {
        "n": len(observed), "r2": r2_score(observed, predicted),
        "rmse": np.sqrt(mean_squared_error(observed, predicted)),
        "mae": mean_absolute_error(observed, predicted),
        "bias": np.mean(predicted - observed),
    }
    fit = linregress(observed, predicted)
    observed_plot = np.clip(observed, PARITY_MIN, PARITY_MAX)
    predicted_plot = np.clip(predicted, PARITY_MIN, PARITY_MAX)
    density = relative_density(observed, predicted)
    order = np.argsort(density)

    scatter = ax.scatter(observed_plot[order], predicted_plot[order], c=density[order],
                          cmap="viridis", norm=Normalize(0, 1), s=27, alpha=0.84,
                          edgecolors="none", rasterized=True, zorder=2)
    ax.plot([PARITY_MIN, PARITY_MAX], [PARITY_MIN, PARITY_MAX], linestyle="--", color="black",
            linewidth=1.45, alpha=0.80, label="1:1", zorder=4)

    x_fit = np.logspace(np.log10(PARITY_MIN), np.log10(PARITY_MAX), 600)
    y_fit = fit.slope * x_fit + fit.intercept
    visible = y_fit > 0
    ax.plot(x_fit[visible], y_fit[visible], color="#C62828", linewidth=1.75,
            label=f"Linear fit: y = {fit.slope:.2f}x{fit.intercept:+.1f}", zorder=5)

    metric_text = (
        f"n = {metrics_local['n']:,}\n"
        f"R² = {metrics_local['r2']:.3f}\n"
        f"Pearson r = {fit.rvalue:.3f}\n"
        f"RMSE = {metrics_local['rmse']:.2f} µg/L\n"
        f"MAE = {metrics_local['mae']:.2f} µg/L\n"
        f"Bias = {metrics_local['bias']:+.2f} µg/L\n"
        f"Slope = {fit.slope:.3f}"
    )
    ax.text(0.035, 0.965, metric_text, transform=ax.transAxes, ha="left", va="top",
            fontsize=10.6, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.55",
                      linewidth=0.8, alpha=0.94), zorder=10)

    format_parity_axis(ax)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    ax.set_xlabel("Measured Chl-a (µg/L)", fontsize=14, labelpad=7)
    ax.set_ylabel("Predicted Chl-a (µg/L)", fontsize=14, labelpad=7)
    ax.legend(loc="lower right", fontsize=9.3, frameon=True, framealpha=0.95,
              borderpad=0.55, labelspacing=0.36, handlelength=2.0)
    return scatter


def make_individual_figure(data, key, label, fig_dir):
    fig = plt.figure(figsize=(8.2, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    scatter = plot_parity_panel(ax, data["chl_a"], data["predicted_chl_a"],
                                 f"Lake Erie external validation ({label})")
    colorbar = fig.colorbar(scatter, cax=cax)
    colorbar.set_label("Relative point density", fontsize=13.5, labelpad=10)
    colorbar.ax.tick_params(labelsize=11)
    for ext in ["png", "pdf", "svg"]:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = DPI
        fig.savefig(fig_dir / f"LakeErie_XGBoost_parity_{key}.{ext}", **kwargs)
    plt.close(fig)


def make_combined_figure(window_data, fig_dir):
    fig = plt.figure(figsize=(17.3, 11.5))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.040], wspace=0.28, hspace=0.31)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    blank = fig.add_subplot(grid[1, 2]); blank.axis("off")
    cax = fig.add_subplot(grid[:, 3])
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    scatter = None
    for ax, panel, (key, label, _) in zip(axes, panel_labels, WINDOWS):
        g = window_data[key]
        scatter = plot_parity_panel(ax, g["chl_a"], g["predicted_chl_a"], f"{panel} {label}")
    colorbar = fig.colorbar(scatter, cax=cax)
    colorbar.set_label("Relative point density", fontsize=15.5, labelpad=11)
    colorbar.ax.tick_params(labelsize=11.5)
    for ext in ["png", "pdf", "svg"]:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = DPI
        fig.savefig(fig_dir / f"LakeErie_XGBoost_parity_all_time_windows.{ext}", **kwargs)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    check_xgboost_version()

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        sys.exit(f"ERROR: {out_dir} exists and is non-empty. Refusing to overwrite. Use --force to override.")
    fig_dir = out_dir / "figures"
    pred_dir = out_dir / "predictions"
    for d in [out_dir, fig_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 105)
    print("ENVIRONMENT")
    print("=" * 105)
    print("Python:", sys.executable)
    print("XGBoost:", xgb.__version__, " NumPy:", np.__version__, " Pandas:", pd.__version__)
    print("Input CSV:", args.input_csv)
    print("Input CSV MD5:", md5_file(args.input_csv))

    print("\n" + "=" * 105)
    print("LOAD MODEL")
    print("=" * 105)
    model = load_model()

    print("\n" + "=" * 105)
    print("LOAD + CLEAN DATA")
    print("=" * 105)
    df = load_and_screen(args.input_csv)
    df = engineer_features(df)

    print("\n" + "=" * 105)
    print("PREDICT")
    print("=" * 105)
    df["predicted_chl_a"] = np.asarray(model.predict(df[FEATURES]), dtype=float).reshape(-1)
    df["residual"] = df["predicted_chl_a"] - df["chl_a"]
    df["absolute_error"] = df["residual"].abs()
    df.to_csv(pred_dir / "LakeErie_XGBoost_predictions_all_positive_spectra.csv", index=False)

    print("\n" + "=" * 115)
    print("EXTERNAL VALIDATION BY TIME WINDOW")
    print("=" * 115)

    results = []
    window_data = {}
    for key, label, hours in WINDOWS:
        g = df.copy() if not np.isfinite(hours) else df.loc[df["abs_time_diff_hours"] <= hours].copy()
        g = g.sort_values(["abs_time_diff_hours", "sample_datetime_local"]).reset_index(drop=True)
        window_data[key] = g
        g.to_csv(pred_dir / f"LakeErie_XGBoost_predictions_{key}.csv", index=False)

        metrics = calculate_metrics(g)
        metrics.update({"window_key": key, "window": label,
                         "mean_abs_time_diff_hours": float(g["abs_time_diff_hours"].mean()),
                         "median_abs_time_diff_hours": float(g["abs_time_diff_hours"].median())})
        results.append(metrics)

        print(f"\n{label}")
        print("-" * 70)
        print(f"n: {metrics['n']:,}  R2: {metrics['r2']:.4f}  RMSE: {metrics['rmse']:.3f}  "
              f"MAE: {metrics['mae']:.3f}  Bias: {metrics['bias']:+.3f}")

        make_individual_figure(g, key, label, fig_dir)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(out_dir / "LakeErie_XGBoost_metrics_by_time_window.csv", index=False)
    (out_dir / "LakeErie_XGBoost_metrics_by_time_window.json").write_text(json.dumps(results, indent=2))

    make_combined_figure(window_data, fig_dir)

    print("\n" + "=" * 120)
    print("COMPACT SUMMARY")
    print("=" * 120)
    print(metrics_df[["window", "n", "r2", "rmse", "mae", "bias", "pearson_r",
                       "spearman_rho", "slope", "intercept"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOutputs:", out_dir)
    print("COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

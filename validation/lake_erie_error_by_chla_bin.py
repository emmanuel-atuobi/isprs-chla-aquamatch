#!/usr/bin/env python3
"""
Concentration-dependent (observed Chl-a bin) error analysis for the Lake Erie <=1h external
validation, as presented in the supplementary material. Standalone counterpart to the Lake
Erie section of discussion_error_diagnostics.py (same bin edges, same metrics).

Usage: python lake_erie_error_by_chla_bin.py [--input-csv /path/to/le1h_predictions.csv]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

DEFAULT_ERIE_FILE = (
    DATA_DIR / "lake_erie_validation" / "DSWE1_ValidTime_FixedScene_final"
    / "predictions" / "LakeErie_XGBoost_predictions_le1h.csv"
)
OUT = RESULTS_DIR / "discussion_error_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

BIN_EDGES = [-np.inf, 2, 5, 10, 20, 50, 100, np.inf]
BIN_LABELS = ["<2", "2–5", "5–10", "10–20", "20–50", "50–100", "≥100"]


def banner(text):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def exact_overall_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    good = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[good], predicted[good]
    residual = predicted - observed
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    fit = linregress(observed, predicted)
    return {
        "n": len(observed), "r2": 1.0 - ss_res / ss_tot,
        "rmse": np.sqrt(np.mean(residual ** 2)), "mae": np.mean(np.abs(residual)),
        "bias": np.mean(residual), "slope": fit.slope, "intercept": fit.intercept,
        "pearson_r": fit.rvalue,
        "obs_min": np.min(observed), "obs_max": np.max(observed),
        "pred_min": np.min(predicted), "pred_max": np.max(predicted),
    }


def prepare_prediction_data(df, observed_col, predicted_col):
    work = df.copy()
    work["observed"] = numeric(work[observed_col])
    work["predicted"] = numeric(work[predicted_col])
    work = work.loc[
        np.isfinite(work["observed"]) & np.isfinite(work["predicted"]) & (work["observed"] > 0)
    ].copy()
    work["residual"] = work["predicted"] - work["observed"]
    work["absolute_error"] = work["residual"].abs()
    work["squared_error"] = work["residual"] ** 2
    work["absolute_percent_error"] = 100.0 * work["absolute_error"] / work["observed"]
    work["pred_obs_ratio"] = work["predicted"] / work["observed"]
    work["chl_bin"] = pd.cut(work["observed"], bins=BIN_EDGES, labels=BIN_LABELS,
                              right=False, ordered=True)
    return work


def error_by_bin(df, observed_col, predicted_col):
    work = prepare_prediction_data(df, observed_col, predicted_col)
    total_n = len(work)
    total_abs_error = work["absolute_error"].sum()
    total_squared_error = work["squared_error"].sum()

    rows = []
    for label in BIN_LABELS:
        g = work.loc[work["chl_bin"].astype(str) == label].copy()
        if len(g) == 0:
            rows.append({"chl_bin": label, "n": 0})
            continue
        residual, ae, se, ratio = g["residual"], g["absolute_error"], g["squared_error"], g["pred_obs_ratio"]
        rows.append({
            "chl_bin": label, "n": len(g), "sample_pct": 100.0 * len(g) / total_n,
            "obs_min": g["observed"].min(), "obs_max": g["observed"].max(),
            "obs_mean": g["observed"].mean(), "obs_median": g["observed"].median(),
            "pred_mean": g["predicted"].mean(), "pred_median": g["predicted"].median(),
            "bias": residual.mean(), "median_error": residual.median(),
            "mae": ae.mean(), "median_abs_error": ae.median(), "rmse": np.sqrt(se.mean()),
            "median_ape_pct": g["absolute_percent_error"].median(),
            "median_pred_obs_ratio": ratio.median(),
            "pred_mean_over_obs_mean": g["predicted"].mean() / g["observed"].mean(),
            "pct_overpredicted": 100.0 * np.mean(residual > 0),
            "pct_underpredicted": 100.0 * np.mean(residual < 0),
            "pct_within_5": 100.0 * np.mean(ae <= 5.0),
            "pct_within_10": 100.0 * np.mean(ae <= 10.0),
            "pct_within_factor2": 100.0 * np.mean((ratio >= 0.5) & (ratio <= 2.0)),
            "share_total_abs_error_pct": (100.0 * ae.sum() / total_abs_error if total_abs_error > 0 else np.nan),
            "share_total_squared_error_pct": (100.0 * se.sum() / total_squared_error if total_squared_error > 0 else np.nan),
        })
    out = pd.DataFrame(rows)
    out["chl_bin"] = pd.Categorical(out["chl_bin"], categories=BIN_LABELS, ordered=True)
    return out.sort_values("chl_bin").reset_index(drop=True)


def print_bin_table(title, table):
    banner(title)
    columns = ["chl_bin", "n", "sample_pct", "obs_mean", "pred_mean", "bias", "mae", "rmse",
               "median_ape_pct", "median_pred_obs_ratio", "pct_overpredicted", "pct_underpredicted",
               "pct_within_5", "pct_within_10", "pct_within_factor2",
               "share_total_abs_error_pct", "share_total_squared_error_pct"]
    use = [c for c in columns if c in table.columns]
    print(table[use].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", default=str(DEFAULT_ERIE_FILE))
    args = p.parse_args()
    erie_file = Path(args.input_csv)

    banner("LOADING LAKE ERIE <=1 H EXTERNAL VALIDATION")
    if not erie_file.exists():
        raise FileNotFoundError(erie_file)
    erie = pd.read_csv(erie_file, low_memory=False)

    required = ["chl_a", "predicted_chl_a"]
    missing = [c for c in required if c not in erie.columns]
    if missing:
        raise KeyError(f"Lake Erie file missing: {missing}\nColumns available:\n{list(erie.columns)}")

    print(f"Lake Erie <=1 h observations: {len(erie):,}  (source: {erie_file})")

    overall = exact_overall_metrics(erie["chl_a"], erie["predicted_chl_a"])
    banner("OVERALL <=1 H METRICS")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    bins = error_by_bin(erie, "chl_a", "predicted_chl_a")
    print_bin_table("D. LAKE ERIE <=1 H -- ERROR BY OBSERVED CHL-A RANGE", bins)

    out_path = OUT / "lake_erie_le1h_error_by_chla_bin.csv"
    bins.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

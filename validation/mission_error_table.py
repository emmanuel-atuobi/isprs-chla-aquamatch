#!/usr/bin/env python3
"""
Supplementary table: final XGBoost holdout error grouped by Landsat mission and by
atmospheric-correction processor (LEDAPS = L5+L7, LaSRC = L8+L9). Also runs Mann-Whitney U
tests and a bootstrap CI on the LEDAPS-vs-LaSRC error difference, to assess
atmospheric-correction consistency across processors.

Uses the already-saved holdout predictions (model_comparison_6models_final/models/xgb/
predictions_holdout.csv) rather than a new model fit, so these numbers are consistent with
the headline R²=0.7182 in the manuscript. 'mission' isn't in predictions_holdout.csv (only
qc_row_id, observed, predicted), so this reconstructs a qc_row_id -> mission lookup by
re-running the quality filter chain, keeping 'mission' instead of dropping it, then joins on
qc_row_id.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = os.environ.get("AQUAMATCH_DATA_DIR", "./data")
RESULTS_DIR = os.environ.get("AQUAMATCH_RESULTS_DIR", "./results")
RAW_FILE = f"{DATA_DIR}/raw/aquamatch_sitesr_joined.csv"
HOLDOUT_FILE = f"{DATA_DIR}/model_comparison_6models_final/models/xgb/predictions_holdout.csv"
OUT_CSV = f"{RESULTS_DIR}/discussion_error_diagnostics/holdout_error_by_mission.csv"
os.makedirs(f"{RESULTS_DIR}/discussion_error_diagnostics", exist_ok=True)

FEATURES = [
    "red", "nir", "blue", "lat", "long",
    "NDVI", "NDTI", "RNI", "GBI", "BLRDGR", "GNRI", "RBI", "NIRGI",
    "GDVI", "NDAVI", "FAI", "MNDWI", "SWI", "TGI", "AFAI", "sin_doy", "cos_doy",
]

# ---- Reconstruct qc_row_id -> mission, via the quality filter chain ----
df = pd.read_csv(RAW_FILE, low_memory=False)
df = df.rename(columns={
    "med_Blue": "blue", "med_Green": "green", "med_Red": "red",
    "med_Nir": "nir", "med_Swir1": "swir1", "med_Swir2": "swir2",
    "harmonized_value": "chl_a", "lon": "long",
})

df = df[df["ResolvedMonitoringLocationTypeName"] == "Lake, Reservoir, Impoundment"].copy()
df = df[df["mission"].isin(["LT05", "LE07", "LC08", "LC09"])].copy()
df["chl_a"] = pd.to_numeric(df["chl_a"], errors="coerce")
df = df[(df["chl_a"] > 0) & (df["chl_a"] <= 200)].copy()
df = df[df["mdl_flag"] == 0].copy()
mask_cv = (df["harmonized_row_count"] == 1) | (df["harmonized_value_cv"] <= 0.5)
df = df[mask_cv].copy()
discrete = ((df["depth_flag"] == 1) & (df["harmonized_discrete_depth_value"] >= 0) & (df["harmonized_discrete_depth_value"] <= 2))
integrated = ((df["depth_flag"] == 2) & (df["harmonized_top_depth_value"] >= 0) & (df["harmonized_top_depth_value"] <= 0.5)
              & (df["harmonized_bottom_depth_value"] >= df["harmonized_top_depth_value"]) & (df["harmonized_bottom_depth_value"] <= 2))
df = df[discrete | integrated].copy()
for col in ["blue", "green", "red", "nir", "swir1", "swir2"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
positive = np.ones(len(df), dtype=bool)
for col in ["blue", "green", "red", "nir", "swir1", "swir2"]:
    positive &= np.isfinite(df[col].to_numpy()) & (df[col].to_numpy() > 0)
df = df.loc[positive].copy()
df["pCount_dswe1"] = pd.to_numeric(df["pCount_dswe1"], errors="coerce")
df = df[df["pCount_dswe1"] >= 8].copy()
df["timediff"] = pd.to_numeric(df["timediff"], errors="coerce")
df = df[np.isfinite(df["timediff"]) & (df["timediff"].abs() <= 2)].copy()
df = df.reset_index(drop=True)

for col in ["lat", "long"]:
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
baseline = df["red"] + (df["swir1"] - df["red"]) * (0.86 - 0.66) / (1.60 - 0.66)
df["FAI"] = df["nir"] - baseline
df["MNDWI"] = (df["green"] - df["swir1"]) / (df["green"] + df["swir1"] + eps)
df["SWI"] = (df["nir"] - df["swir1"]) / (df["nir"] + df["swir1"] + eps)
df["TGI"] = -0.5 * (120.0 * (df["red"] - df["green"]) - 190.0 * (df["red"] - df["blue"]))
df["AFAI"] = (df["nir"] - df["red"]) + 0.5 * (df["swir1"] - df["red"])
date_col = "ActivityStartDate" if "ActivityStartDate" in df.columns else "date"
dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed", dayfirst=False)
doy = dates.dt.dayofyear
df["sin_doy"] = np.sin(2.0 * np.pi * (doy - 1.0) / 365.0)
df["cos_doy"] = np.cos(2.0 * np.pi * (doy - 1.0) / 365.0)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + ["chl_a"]).reset_index(drop=True)

if len(df) != 14125:
    raise RuntimeError(f"Expected quality-filtered N=14,125 while reconstructing mission lookup, got {len(df):,}. "
                        f"Upstream data may have changed -- stop and check before trusting this table.")

df["qc_row_id"] = np.arange(len(df), dtype=int)
# Carry chl_a through too, so the join can be verified against the real holdout values,
# not just checked for row count / no-NaN (which wouldn't catch a silent misalignment).
mission_lookup = df[["qc_row_id", "mission", "chl_a"]].rename(
    columns={"chl_a": "reconstructed_chl_a"}
)

# ---- Join to the actual, already-reported holdout predictions ----
holdout = pd.read_csv(HOLDOUT_FILE)
merged = holdout.merge(mission_lookup, on="qc_row_id", how="left")
if merged["mission"].isna().any():
    raise RuntimeError(f"{merged['mission'].isna().sum()} holdout rows failed to match a mission -- stop.")

if not np.allclose(
    merged["observed_chl_a"], merged["reconstructed_chl_a"], rtol=0, atol=1e-10
):
    raise RuntimeError("qc_row_id reconstruction does not match the holdout file -- row alignment is wrong, stop.")
print("Row-alignment check passed: reconstructed chl_a matches the holdout file exactly for all rows.")

print(f"Holdout rows: {len(merged):,} (matches expected N=3,532: {len(merged) == 3532})")


def metrics(g):
    y, p = g["observed_chl_a"].to_numpy(), g["predicted_chl_a"].to_numpy()
    resid = p - y
    return {
        "n": len(g), "r2": r2_score(y, p), "rmse": np.sqrt(mean_squared_error(y, p)),
        "mae": mean_absolute_error(y, p), "bias": np.mean(resid),
    }


rows = []
mission_names = {"LT05": "Landsat 5 (LEDAPS)", "LE07": "Landsat 7 (LEDAPS)",
                  "LC08": "Landsat 8 (LaSRC)", "LC09": "Landsat 9 (LaSRC)"}
for code, name in mission_names.items():
    g = merged[merged["mission"] == code]
    if len(g) == 0:
        continue
    rows.append({"sensor_processor": name, **metrics(g)})

rows.append({"sensor_processor": "LEDAPS: L5 + L7", **metrics(merged[merged["mission"].isin(["LT05", "LE07"])])})
rows.append({"sensor_processor": "LaSRC: L8 + L9", **metrics(merged[merged["mission"].isin(["LC08", "LC09"])])})

table = pd.DataFrame(rows)
for c in ["r2", "rmse", "mae", "bias"]:
    table[c] = table[c].round(4)

print("\n" + table.to_string(index=False))
table.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

# ---- Significance checks: location (signed residual, i.e. bias) AND magnitude
# (absolute error, i.e. accuracy) -- these test different things. The signed-residual
# test only rules out a systematic directional shift in bias between groups; the
# absolute-error test is the one that speaks to accuracy/error magnitude. ----
from scipy.stats import mannwhitneyu
merged["residual"] = merged["predicted_chl_a"] - merged["observed_chl_a"]
merged["abs_error"] = merged["residual"].abs()

res_ledaps = merged.loc[merged["mission"].isin(["LT05", "LE07"]), "residual"]
res_lasrc = merged.loc[merged["mission"].isin(["LC08", "LC09"]), "residual"]
_, pval = mannwhitneyu(res_ledaps, res_lasrc, alternative="two-sided")
print(f"\nMann-Whitney U on signed residuals / bias (LEDAPS vs LaSRC): p={pval:.4f}")
print(f"  Median residual LEDAPS: {res_ledaps.median():+.3f} ug/L")
print(f"  Median residual LaSRC:  {res_lasrc.median():+.3f} ug/L")
print("  " + ("No significant difference in bias" if pval > 0.05 else "Significant difference in bias"))

ae_ledaps = merged.loc[merged["mission"].isin(["LT05", "LE07"]), "abs_error"]
ae_lasrc = merged.loc[merged["mission"].isin(["LC08", "LC09"]), "abs_error"]
_, p_abs = mannwhitneyu(ae_ledaps, ae_lasrc, alternative="two-sided")
print(f"\nMann-Whitney U on absolute errors / accuracy (LEDAPS vs LaSRC): p={p_abs:.4f}")
print(f"  Median absolute error LEDAPS: {ae_ledaps.median():.3f} ug/L")
print(f"  Median absolute error LaSRC:  {ae_lasrc.median():.3f} ug/L")
print("  " + ("No significant difference in error magnitude" if p_abs > 0.05 else "Significant difference in error magnitude"))

# ---- Persist the significance-test results to a companion CSV ----
SIG_CSV = f"{RESULTS_DIR}/discussion_error_diagnostics/holdout_significance_ledaps_vs_lasrc.csv"
sig_rows = [
    {
        "test": "signed_residual_bias",
        "description": "Mann-Whitney U on signed residuals (predicted - observed): tests for a systematic directional shift in bias between groups.",
        "group1": "LEDAPS: L5 + L7", "n1": len(res_ledaps), "median1": round(res_ledaps.median(), 4),
        "group2": "LaSRC: L8 + L9", "n2": len(res_lasrc), "median2": round(res_lasrc.median(), 4),
        "p_value": round(pval, 4),
        "significant_at_0.05": bool(pval <= 0.05),
    },
    {
        "test": "absolute_error_magnitude",
        "description": "Mann-Whitney U on absolute errors: tests whether error magnitude/accuracy differs between groups.",
        "group1": "LEDAPS: L5 + L7", "n1": len(ae_ledaps), "median1": round(ae_ledaps.median(), 4),
        "group2": "LaSRC: L8 + L9", "n2": len(ae_lasrc), "median2": round(ae_lasrc.median(), 4),
        "p_value": round(p_abs, 4),
        "significant_at_0.05": bool(p_abs <= 0.05),
    },
]
pd.DataFrame(sig_rows).to_csv(SIG_CSV, index=False)
print(f"\nSaved: {SIG_CSV}")

# ---- Bootstrap uncertainty for processor-group differences in MAE and RMSE ----
# Delta = LaSRC - LEDAPS, so a positive value means the LaSRC group has larger error.
rng = np.random.default_rng(42)
n_boot = 10000

ledaps = merged[merged["mission"].isin(["LT05", "LE07"])].copy()
lasrc = merged[merged["mission"].isin(["LC08", "LC09"])].copy()

y_led = ledaps["observed_chl_a"].to_numpy()
p_led = ledaps["predicted_chl_a"].to_numpy()

y_las = lasrc["observed_chl_a"].to_numpy()
p_las = lasrc["predicted_chl_a"].to_numpy()

obs_mae_led = np.mean(np.abs(p_led - y_led))
obs_mae_las = np.mean(np.abs(p_las - y_las))

obs_rmse_led = np.sqrt(np.mean((p_led - y_led) ** 2))
obs_rmse_las = np.sqrt(np.mean((p_las - y_las) ** 2))

obs_delta_mae = obs_mae_las - obs_mae_led
obs_delta_rmse = obs_rmse_las - obs_rmse_led

boot_delta_mae = np.empty(n_boot)
boot_delta_rmse = np.empty(n_boot)

for b in range(n_boot):
    idx_led = rng.integers(0, len(ledaps), len(ledaps))
    idx_las = rng.integers(0, len(lasrc), len(lasrc))

    e_led = p_led[idx_led] - y_led[idx_led]
    e_las = p_las[idx_las] - y_las[idx_las]

    mae_led = np.mean(np.abs(e_led))
    mae_las = np.mean(np.abs(e_las))
    rmse_led = np.sqrt(np.mean(e_led ** 2))
    rmse_las = np.sqrt(np.mean(e_las ** 2))

    boot_delta_mae[b] = mae_las - mae_led
    boot_delta_rmse[b] = rmse_las - rmse_led

ci_mae = np.percentile(boot_delta_mae, [2.5, 97.5])
ci_rmse = np.percentile(boot_delta_rmse, [2.5, 97.5])

print(f"\nBootstrap (n={n_boot:,}, seed=42), Delta = LaSRC - LEDAPS:")
print(f"  Delta MAE:  {obs_delta_mae:+.3f} ug/L   95% CI [{ci_mae[0]:+.3f}, {ci_mae[1]:+.3f}]")
print(f"  Delta RMSE: {obs_delta_rmse:+.3f} ug/L   95% CI [{ci_rmse[0]:+.3f}, {ci_rmse[1]:+.3f}]")

boot_rows = [
    {
        "metric": "MAE", "group1": "LEDAPS: L5 + L7", "value1": round(obs_mae_led, 4),
        "group2": "LaSRC: L8 + L9", "value2": round(obs_mae_las, 4),
        "delta_lasrc_minus_ledaps": round(obs_delta_mae, 4),
        "ci_95_low": round(ci_mae[0], 4), "ci_95_high": round(ci_mae[1], 4),
        "ci_excludes_zero": bool(ci_mae[0] > 0 or ci_mae[1] < 0),
    },
    {
        "metric": "RMSE", "group1": "LEDAPS: L5 + L7", "value1": round(obs_rmse_led, 4),
        "group2": "LaSRC: L8 + L9", "value2": round(obs_rmse_las, 4),
        "delta_lasrc_minus_ledaps": round(obs_delta_rmse, 4),
        "ci_95_low": round(ci_rmse[0], 4), "ci_95_high": round(ci_rmse[1], 4),
        "ci_excludes_zero": bool(ci_rmse[0] > 0 or ci_rmse[1] < 0),
    },
]
BOOT_CSV = f"{RESULTS_DIR}/discussion_error_diagnostics/holdout_bootstrap_ledaps_vs_lasrc.csv"
pd.DataFrame(boot_rows).to_csv(BOOT_CSV, index=False)
print(f"\nSaved: {BOOT_CSV}")

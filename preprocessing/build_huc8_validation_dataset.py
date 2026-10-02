"""
build_huc8_validation_dataset.py
=========================
Applies the quality-filter chain (same 9 steps as model_development/prepare_6models_final.py)
to the joined AquaMatch + siteSR dataset, keeping the NHD-derived HUC8 watershed assignment
that preprocessing/join_aquamatch_sitesr.py carries through from the siteSR sites list.
Produces data/spatial_validation_diagnostics/huc8_field_comparison.csv, the input
validation/run_huc4_fixed_35seeds.py and validation/ablation_huc4_split_35seed.py expect.

No separate spatial join against watershed boundaries is needed: the sites list's own
`assigned_HUC` column (NHD-derived) already matches `HUC8_assigned`/`flag_HUC8`.

Usage:
  Run after producing data/raw/aquamatch_sitesr_joined.csv (see preprocessing/
  join_aquamatch_sitesr.py and preprocessing/finalize_aquamatch_sitesr_join.py).
  python build_huc8_validation_dataset.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
INPUT_FILE = DATA_DIR / "raw" / "aquamatch_sitesr_joined.csv"
OUTPUT_FILE = DATA_DIR / "spatial_validation_diagnostics" / "huc8_field_comparison.csv"


def counts(label, frame):
    print(f"{label:<38s}: {len(frame):,}")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing: {INPUT_FILE}\nRun the join scripts first.")

    print("=" * 100)
    print("BUILDING HUC8 VALIDATION DATASET")
    print("=" * 100)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    counts("Raw joined rows", df)

    df = df.rename(columns={
        "med_Blue": "blue", "med_Green": "green", "med_Red": "red",
        "med_Nir": "nir", "med_Swir1": "swir1", "med_Swir2": "swir2",
        "harmonized_value": "chl_a", "lon": "long",
        "HUCEightDigitCode": "HUC8_original", "assigned_HUC": "HUC8_assigned",
    })

    has_huc = "HUC8_assigned" in df.columns
    if not has_huc:
        raise KeyError(
            "No HUC8_assigned column found. Re-run preprocessing/join_aquamatch_sitesr.py "
            "(it now carries 'assigned_HUC' through from the sites list) before this script."
        )

    # Quality filtering, steps 1-9, identical to model_development/prepare_6models_final.py.
    df = df[df["ResolvedMonitoringLocationTypeName"] == "Lake, Reservoir, Impoundment"].copy()
    counts("1. Lakes/reservoirs/impoundments", df)

    df = df[df["mission"].isin(["LT05", "LE07", "LC08", "LC09"])].copy()
    counts("2. Landsat 5/7/8/9", df)

    df["chl_a"] = pd.to_numeric(df["chl_a"], errors="coerce")
    df = df[(df["chl_a"] > 0) & (df["chl_a"] <= 200)].copy()
    counts("3. Chl-a >0 and <=200", df)

    df = df[df["mdl_flag"] == 0].copy()
    counts("4. mdl_flag = 0", df)

    mask_cv = (df["harmonized_row_count"] == 1) | (df["harmonized_value_cv"] <= 0.5)
    df = df[mask_cv].copy()
    counts("5. Replicate CV <=0.5 or single", df)

    discrete = (
        (df["depth_flag"] == 1)
        & (df["harmonized_discrete_depth_value"] >= 0)
        & (df["harmonized_discrete_depth_value"] <= 2)
    )
    integrated = (
        (df["depth_flag"] == 2)
        & (df["harmonized_top_depth_value"] >= 0)
        & (df["harmonized_top_depth_value"] <= 0.5)
        & (df["harmonized_bottom_depth_value"] >= df["harmonized_top_depth_value"])
        & (df["harmonized_bottom_depth_value"] <= 2)
    )
    df = df[discrete | integrated].copy()
    counts("6. Near-surface depth 0-2 m", df)

    for col in ["blue", "green", "red", "nir", "swir1", "swir2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    positive = np.ones(len(df), dtype=bool)
    for col in ["blue", "green", "red", "nir", "swir1", "swir2"]:
        positive &= np.isfinite(df[col].to_numpy()) & (df[col].to_numpy() > 0)
    df = df.loc[positive].copy()
    counts("7. Six bands finite and positive", df)

    df["pCount_dswe1"] = pd.to_numeric(df["pCount_dswe1"], errors="coerce")
    df = df[df["pCount_dswe1"] >= 8].copy()
    counts("8. pCount_dswe1 >= 8", df)

    df["timediff"] = pd.to_numeric(df["timediff"], errors="coerce")
    df = df[np.isfinite(df["timediff"]) & (df["timediff"].abs() <= 2)].copy()
    counts("9. Absolute exact time difference <=2 h", df)

    df = df.reset_index(drop=True)

    if len(df) != 14125:
        raise RuntimeError(f"Expected quality-filtered N=14,125, found {len(df):,}.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"  {len(df):,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()

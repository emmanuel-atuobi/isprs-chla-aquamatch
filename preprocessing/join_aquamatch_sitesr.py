"""
join_aquamatch_sitesr.py
========================
Stage 1 of 2 in building the matchup dataset: joins AquaMatch Chl-a in-situ data to siteSR
Landsat Collection 2 surface reflectance by site and calendar date, producing a `sensor`
column (e.g. "Landsat5") per matched row. `mission` (e.g. "LT05") comes through unchanged,
since siteSR's own reflectance records already carry it. Also carries through the sites
list's own NHD-derived HUC8 watershed assignment (`HUCEightDigitCode`, `assigned_HUC`,
`flag_HUC8`), used later by `preprocessing/build_huc8_validation_dataset.py`.

This is NOT yet the file downstream scripts expect: `preprocessing/
finalize_aquamatch_sitesr_join.py` (Stage 2) still needs to merge in the exact scene-center
acquisition time (from preprocessing/gee/gee_scene_overpass_lookup_time.js) to compute the
real `timediff` (hours) before `timediff_window_sweep.py` or anything in
`model_development/` can run. See data/README.md.

Pipeline:
  AquaMatch Chl-a --> sites list (via loc_id) --> siteSR reflectance (via siteSR_id + date)

Usage:
  1. Set SOURCE_DATA_DIR below (or the AQUAMATCH_SOURCE_DIR env var) to a local directory
     containing the AquaMatch Chl-a file, the siteSR sites list, and the siteSR reflectance
     feather files. These are large upstream source files and are not included in this
     repository; obtain them from the AquaMatch/siteSR project directly.
  2. Run:  python join_aquamatch_sitesr.py
  3. Phase 1 prints column schemas; use --discovery-only to stop after inspection
  4. Phase 2 does the join and profiles the distribution
"""

import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
import gc
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# =====================================================================
# CONFIG — edit these paths if your source files use different names
# =====================================================================

SOURCE_DATA_DIR = Path(os.environ.get("AQUAMATCH_SOURCE_DIR", "./source_data"))
DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))

# AquaMatch Chl-a in-situ data
AQUAMATCH_CHL_PATH = str(SOURCE_DATA_DIR / "chla_harmonized_final.csv")

# siteSR sites list — download file #3 from EDI if not yet present
SITES_LIST_PATH = str(SOURCE_DATA_DIR / "siteSR_collated_WQP_NWIS_sites_with_NHD_info_2025-06-04.csv")

# siteSR reflectance feather files — DSWE1 only (not DSWE1a)
SITESR_REFLECTANCE_PATHS = [
    str(SOURCE_DATA_DIR / "siteSR_Landsat4_DSWE1_2025-06-06.feather"),
    str(SOURCE_DATA_DIR / "siteSR_Landsat5_DSWE1_2025-06-06.feather"),
    str(SOURCE_DATA_DIR / "siteSR_Landsat7_DSWE1_2025-06-06.feather"),
    str(SOURCE_DATA_DIR / "siteSR_Landsat8_DSWE1_2025-06-06.feather"),
    str(SOURCE_DATA_DIR / "siteSR_Landsat9_DSWE1_2025-06-06.feather"),
]

# Named _stage1, not aquamatch_sitesr_joined.csv: this output is missing timediff
# (Stage 2, see module docstring) and would be mistaken for the complete file otherwise.
OUTPUT_PATH = str(DATA_DIR / "raw" / "aquamatch_sitesr_joined_stage1.csv")

# --- Column name mappings ---
# siteSR feather files (confirmed from UserGuide):
#   siteSR_id, date, sat_id, med_Blue, med_Green, med_Red, med_Nir,
#   med_Swir1, med_Swir2, med_SurfaceTemp, prop_clouds, pCount_dswe1

# Sites list (confirmed):
#   siteSR_id, loc_id, org_id, MonitoringLocationTypeName, + lat/lon

# Expected AquaMatch column mappings (WQP conventions):
CHL_SITE_COL = "MonitoringLocationIdentifier"
CHL_DATE_COL = "ActivityStartDate"
CHL_VALUE_COL = "harmonized_value"

# =====================================================================


def read_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if p.suffix == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path, low_memory=False)


def discover(df, name, n=3):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"{'='*60}")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
        sample = df[col].dropna().iloc[0] if df[col].notna().any() else "NaN"
        print(f"  {i:>3}. {col:<45} {str(dtype):<12} e.g. {str(sample)[:60]}")
    print()


def profile_chl(values, label=""):
    valid = values[(values > 0) & (values <= 1000)].dropna()
    n = len(valid)
    if n == 0:
        print(f"  No valid chl-a values found.")
        return

    pct = valid.quantile([0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"\n  {label} — {n:,} valid values")
    print(f"    min     {valid.min():10.3f}")
    print(f"    5th     {pct.loc[0.05]:10.3f}")
    print(f"    25th    {pct.loc[0.25]:10.3f}")
    print(f"    median  {pct.loc[0.50]:10.3f}")
    print(f"    mean    {valid.mean():10.3f}")
    print(f"    75th    {pct.loc[0.75]:10.3f}")
    print(f"    90th    {pct.loc[0.90]:10.3f}")
    print(f"    95th    {pct.loc[0.95]:10.3f}")
    print(f"    99th    {pct.loc[0.99]:10.3f}")
    print(f"    max     {valid.max():10.3f}")

    BINS = [-np.inf, 2, 7, 30, np.inf]
    LABELS = ["Oligotrophic", "Mesotrophic", "Eutrophic", "Hypereutrophic"]
    NLA = {"Oligotrophic": 7.0, "Mesotrophic": 20.0,
           "Eutrophic": 43.0, "Hypereutrophic": 30.0}

    cats = pd.cut(valid, bins=BINS, labels=LABELS, right=True)
    counts = cats.value_counts().reindex(LABELS)
    perc = (counts / n * 100).round(1)

    print(f"\n  {'Trophic state':<16}{'n':>10}{'%':>10}{'NLA 2022':>10}{'Diff':>10}")
    print(f"  {'-'*56}")
    for lab in LABELS:
        print(f"  {lab:<16}{counts[lab]:>10,}{perc[lab]:>9.1f}%{NLA[lab]:>9.1f}%{perc[lab]-NLA[lab]:>+9.1f}")

    below_7 = (valid <= 7).mean() * 100
    above_30 = (valid > 30).mean() * 100
    print(f"\n  oligo+meso (<=7):       {below_7:.1f}%")
    print(f"  eutrophic+ (>7):        {100-below_7:.1f}%")
    print(f"  hypereutrophic (>30):   {above_30:.1f}%")


def main():
    # ==================================================================
    # PHASE 1: DISCOVERY
    # ==================================================================
    print("\n" + "#"*60)
    print("#  PHASE 1: DISCOVERY")
    print("#"*60)

    # --- AquaMatch Chl-a ---
    print("\nLoading AquaMatch Chl-a...")
    chl = read_file(AQUAMATCH_CHL_PATH)
    discover(chl, "AquaMatch Chl-a")

    # --- Sites list (just read columns, don't load full 245 MiB yet) ---
    print("Loading siteSR sites list (first 5 rows for schema)...")
    sites_preview = pd.read_csv(SITES_LIST_PATH, nrows=5, low_memory=False)
    discover(sites_preview, "siteSR Sites List")

    # --- One reflectance file (schema only, don't load full file) ---
    # Use the smallest file (Landsat 4, 33 MiB) for schema discovery
    sr_files_exist = [p for p in SITESR_REFLECTANCE_PATHS if Path(p).exists()]
    if sr_files_exist:
        # Pick smallest file for preview
        smallest = min(sr_files_exist, key=lambda p: Path(p).stat().st_size)
        print(f"Loading smallest reflectance file for schema: {Path(smallest).name}")
        sr_preview = pd.read_feather(smallest, columns=None).head(5)
        discover(sr_preview, f"siteSR Reflectance ({Path(smallest).name})")
        del sr_preview

    # --- Checkpoint ---
    print("="*60)
    print("  CHECKPOINT")
    print(f"  Chl-a site column:  {CHL_SITE_COL}")
    print(f"  Chl-a date column:  {CHL_DATE_COL}")
    print(f"  Chl-a value column: {CHL_VALUE_COL}")
    print()
    print("  If these don't match what you see above,")
    print("  update the CONFIG and re-run with --discovery-only flag.")
    print("="*60)

    if "--discovery-only" in sys.argv:
        print("\n  Discovery-only mode. Stopping here.")
        return

    # ==================================================================
    # PHASE 2: JOIN
    # ==================================================================
    print("\n" + "#"*60)
    print("#  PHASE 2: JOIN")
    print("#"*60)

    # --- Step 1: Load sites list and build loc_id -> siteSR_id map ---
    print("\nStep 1: Loading full sites list and building site map...")
    sites = pd.read_csv(SITES_LIST_PATH, low_memory=False)
    print(f"  Sites list: {len(sites):,} rows")

    # Keep only the columns we need for the join
    map_cols = ["siteSR_id", "loc_id"]
    # Add lat/lon if present
    for c in sites.columns:
        if c.lower() in ["latitude", "longitude", "lat", "long", "lon"]:
            map_cols.append(c)
    # Carry through the sites list's own NHD-derived HUC8 watershed assignment -- needed by
    # validation/run_huc4_fixed_35seeds.py and validation/ablation_huc4_split_35seed.py.
    for c in ["HUCEightDigitCode", "assigned_HUC", "flag_HUC8"]:
        if c in sites.columns:
            map_cols.append(c)
    site_map = sites[map_cols].drop_duplicates(subset=["loc_id"])
    print(f"  Unique loc_id entries: {len(site_map):,}")
    del sites
    gc.collect()

    # --- Step 2: Map AquaMatch to siteSR_id ---
    print("\nStep 2: Mapping AquaMatch Chl-a to siteSR_id...")

    if CHL_SITE_COL not in chl.columns:
        print(f"  ERROR: '{CHL_SITE_COL}' not in AquaMatch columns.")
        print(f"  Available: {[c for c in chl.columns if 'loc' in c.lower() or 'monitor' in c.lower() or 'site' in c.lower() or 'id' in c.lower()]}")
        raise SystemExit("Fix CHL_SITE_COL in CONFIG.")

    chl_mapped = chl.merge(site_map, left_on=CHL_SITE_COL, right_on="loc_id", how="inner")
    print(f"  AquaMatch rows:          {len(chl):,}")
    print(f"  Mapped to siteSR_id:     {len(chl_mapped):,}")
    print(f"  Lost (no siteSR match):  {len(chl) - len(chl_mapped):,}")
    del chl
    gc.collect()

    # --- Step 3: Parse dates ---
    print("\nStep 3: Parsing dates...")
    chl_mapped["_join_date"] = pd.to_datetime(
        chl_mapped[CHL_DATE_COL], errors="coerce"
    ).dt.date
    bad_dates = chl_mapped["_join_date"].isna().sum()
    print(f"  Bad dates in AquaMatch: {bad_dates:,}")
    chl_mapped = chl_mapped.dropna(subset=["_join_date"])

    # --- Step 4: Join with reflectance, one sensor at a time ---
    print("\nStep 4: Joining with reflectance (one sensor at a time)...")

    all_joined = []
    for sr_path in SITESR_REFLECTANCE_PATHS:
        p = Path(sr_path)
        if not p.exists():
            print(f"  WARNING: {sr_path} not found, skipping.")
            continue

        sensor = p.stem.split("_")[1]  # e.g. "Landsat5"
        print(f"\n  Loading {p.name} ({p.stat().st_size / 1e9:.1f} GB)...")
        sr = read_file(sr_path)
        print(f"    Rows: {len(sr):,}")

        # Parse date in reflectance
        sr["_join_date"] = pd.to_datetime(sr["date"], errors="coerce").dt.date
        bad = sr["_join_date"].isna().sum()
        if bad > 0:
            print(f"    Bad dates: {bad:,}")
        sr = sr.dropna(subset=["_join_date"])

        # Join
        joined = chl_mapped.merge(
            sr,
            on=["siteSR_id", "_join_date"],
            how="inner",
            suffixes=("_chl", "_sr")
        )
        joined["sensor"] = sensor
        print(f"    Matched (same-day): {len(joined):,}")
        all_joined.append(joined)

        del sr, joined
        gc.collect()

    if not all_joined:
        raise SystemExit("No matchups found across any sensor. Check column names.")

    final = pd.concat(all_joined, ignore_index=True)
    del all_joined
    gc.collect()

    print(f"\n  Total matchups across all sensors: {len(final):,}")
    print(f"  Unique sites:  {final['siteSR_id'].nunique():,}")
    print(f"  Unique dates:  {final['_join_date'].nunique():,}")

    # Sensor breakdown
    print(f"\n  Per-sensor breakdown:")
    for s, cnt in final["sensor"].value_counts().items():
        print(f"    {s}: {cnt:,}")

    # --- Step 5: Profile chl-a distribution ---
    print("\n" + "#"*60)
    print("#  CHL-A DISTRIBUTION PROFILE")
    print("#"*60)

    chl_col = CHL_VALUE_COL
    if chl_col not in final.columns:
        candidates = [c for c in final.columns if "chl" in c.lower() or "harmon" in c.lower() or "value" in c.lower()]
        if chl_col + "_chl" in final.columns:
            chl_col = chl_col + "_chl"
        elif candidates:
            print(f"  '{CHL_VALUE_COL}' not found. Candidates: {candidates}")
            chl_col = input("  Enter correct column name: ").strip()
        else:
            print(f"  No chl-a column found. Available: {list(final.columns)}")
            raise SystemExit()

    chl_vals = pd.to_numeric(final[chl_col], errors="coerce")
    profile_chl(chl_vals, "AquaMatch + siteSR joined")

    # --- Step 6: List band columns present ---
    band_cols = [c for c in final.columns if c.startswith("med_")]
    if band_cols:
        print(f"\n  Band/measurement columns found:")
        for c in band_cols:
            print(f"    {c}")

    # --- Step 7: Save ---
    print(f"\nStep 7: Saving...")
    final = final.drop(columns=["_join_date"], errors="ignore")
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved: {OUTPUT_PATH}")
    print(f"  Size:  {len(final):,} rows x {final.shape[1]} columns")
    print("\nDone.")


if __name__ == "__main__":
    main()

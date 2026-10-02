"""
finalize_aquamatch_sitesr_join.py
==================================
Stage 2 of 2 in building the matchup dataset: merges the GEE scene-time lookup
(preprocessing/gee/gee_scene_overpass_lookup_time.js's export) onto Stage 1's output
(preprocessing/join_aquamatch_sitesr.py) to compute the exact `timediff` (hours between the
field sample time and the Landsat scene's acquisition time) that every downstream script
filters on.

Stage 1's output already carries `mission` through unchanged from siteSR (siteSR's own
reflectance records include a `mission` column), so this stage only needs to add `timediff`.
The join key is `sat_id`, which both Stage 1's output (via siteSR) and the GEE scene-time
export share natively -- no reconciliation between different ID schemes is needed.

Usage:
  1. Run preprocessing/join_aquamatch_sitesr.py first.
  2. Set SOURCE_DATA_DIR below (or the AQUAMATCH_SOURCE_DIR env var) to the directory
     containing scene_center_times.csv, the CSV exported by
     preprocessing/gee/gee_scene_overpass_lookup_time.js.
  3. Run: python finalize_aquamatch_sitesr_join.py
"""
import os
from pathlib import Path

import pandas as pd

SOURCE_DATA_DIR = Path(os.environ.get("AQUAMATCH_SOURCE_DIR", "./source_data"))
DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))

STAGE1_PATH = DATA_DIR / "raw" / "aquamatch_sitesr_joined_stage1.csv"
SCENE_TIMES_PATH = SOURCE_DATA_DIR / "scene_center_times.csv"
OUTPUT_PATH = DATA_DIR / "raw" / "aquamatch_sitesr_joined.csv"


def main():
    if not STAGE1_PATH.exists():
        raise FileNotFoundError(
            f"Missing Stage 1 output: {STAGE1_PATH}\nRun join_aquamatch_sitesr.py first."
        )
    if not SCENE_TIMES_PATH.exists():
        raise FileNotFoundError(
            f"Missing GEE scene-time export: {SCENE_TIMES_PATH}\n"
            "Run preprocessing/gee/gee_scene_overpass_lookup_time.js in the GEE code editor "
            "and download its export here."
        )

    print(f"Loading Stage 1 output: {STAGE1_PATH}")
    stage1 = pd.read_csv(STAGE1_PATH, low_memory=False)
    print(f"  Rows: {len(stage1):,}")

    if "mission" not in stage1.columns:
        raise KeyError(
            "Stage 1 output has no 'mission' column -- expected it to pass through from "
            "siteSR's own 'mission' field. Check join_aquamatch_sitesr.py's merge."
        )
    if "harmonized_utc" not in stage1.columns:
        raise KeyError(
            "Stage 1 output has no 'harmonized_utc' column -- expected it to pass through "
            "from the raw AquaMatch Chl-a file."
        )

    print(f"\nLoading GEE scene-time export: {SCENE_TIMES_PATH}")
    scene = pd.read_csv(
        SCENE_TIMES_PATH, low_memory=False,
        usecols=["sat_id", "acquisition_datetime_utc"],
    )
    print(f"  Rows: {len(scene):,}  Unique sat_id: {scene['sat_id'].nunique():,}")

    dupes = scene["sat_id"].duplicated().sum()
    if dupes:
        raise RuntimeError(
            f"{dupes:,} duplicate sat_id rows in the scene-time export -- the merge below "
            "would silently fan out. Deduplicate before proceeding."
        )

    merged = stage1.merge(scene, on="sat_id", how="left")

    unmatched = merged["acquisition_datetime_utc"].isna().sum()
    if unmatched:
        print(
            f"\n  WARNING: {unmatched:,} / {len(merged):,} rows have no matching sat_id in "
            "the scene-time export. These rows will have timediff = NaN and should be "
            "investigated before using this file -- every row is expected to match."
        )
    else:
        print(f"\n  All {len(merged):,} rows matched a scene time.")

    field_time = pd.to_datetime(merged["harmonized_utc"], utc=True, errors="coerce")
    scene_time = pd.to_datetime(merged["acquisition_datetime_utc"], utc=True, errors="coerce")
    merged["timediff"] = (field_time - scene_time).abs().dt.total_seconds() / 3600.0

    merged = merged.drop(columns=["acquisition_datetime_utc"])

    print(f"\ntimediff summary (hours):")
    print(merged["timediff"].describe())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  {len(merged):,} rows x {merged.shape[1]} columns")


if __name__ == "__main__":
    main()

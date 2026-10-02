# Data

The primary source datasets used in this study are not redistributed in this repository.

- **AquaMatch chlorophyll-a** — Environmental Data Initiative:
  https://doi.org/10.6073/pasta/2f750544112e5408928dd9a61e6ace30
- **siteSR Landsat surface reflectance** — Environmental Data Initiative:
  https://doi.org/10.6073/pasta/f85622d6d32ef7fe6cff8d63c3b947c9
- **Western Lake Erie independent validation observations** — NOAA NCEI:
  https://doi.org/10.25921/7c57-gf98

Download these from their original repositories.

Building `data/raw/aquamatch_sitesr_joined.csv` from those source files takes three steps,
run in order:

1. **Join** (`preprocessing/join_aquamatch_sitesr.py`) joins AquaMatch Chl-a to siteSR
   reflectance by site (`siteSR_id`) and calendar date, writing
   `data/raw/aquamatch_sitesr_joined_stage1.csv`. `mission` and the NHD-derived HUC8
   watershed code (used below) both come through unchanged from siteSR's own data at this
   step.
2. **Scene times** (`preprocessing/gee/gee_scene_overpass_lookup_time.js`, run in the Earth
   Engine code editor) looks up the exact acquisition time of each Landsat scene matched at
   step 1. It needs a small input table built first, since a GEE script can only read Earth
   Engine assets, not local files:
   - From step 1's output, keep one row per unique `sat_id`.
   - Add a `gee_id` column: strip `sat_id`'s leading numeric prefix and prepend
     `LANDSAT/<mission>/C02/T1_L2/`. Example: `sat_id` `1_1_LT04_033036_19920604` becomes
     `gee_id` `LANDSAT/LT04/C02/T1_L2/LT04_033036_19920604`.
   - Upload that `gee_id`, `sat_id`, `mission` table to Earth Engine as a FeatureCollection,
     then point the script's `LOOKUP_ASSET` variable at it.
3. **Finalize** (`preprocessing/finalize_aquamatch_sitesr_join.py`) merges step 2's export
   back onto step 1's output on `sat_id` and computes `timediff` (hours between the field
   sample time and the scene's acquisition time), producing the final
   `data/raw/aquamatch_sitesr_joined.csv`.

Point `AQUAMATCH_DATA_DIR` (see the top-level `README.md`) at the result. The scripts in
`preprocessing/` from that point onward reproduce the quality filtering and matchup
processing used in the study.

One validation setting needs an extra input. The HUC4-stratified spatial validation
(`validation/run_huc4_fixed_35seeds.py` and `validation/ablation_huc4_split_35seed.py`)
groups stations into HUC4 strata and withholds ~25% of stations per stratum for validation
(a given station's observations stay entirely on one side; a HUC4 region can appear on
both). Both scripts read `data/spatial_validation_diagnostics/huc8_field_comparison.csv`,
built by running `preprocessing/build_huc8_validation_dataset.py` on
`data/raw/aquamatch_sitesr_joined.csv` — the same quality-filter chain as
`model_development/prepare_6models_final.py`, plus each station's watershed code carried
through unchanged from step 1. It's tagged at the HUC8 (8-digit sub-basin) level only
because that's the granularity siteSR's sites list provides; the validation scripts
truncate it to the first 4 digits to get HUC4.

## Contents

- `lake_erie_validation/DSWE1_ValidTime_FixedScene_final/LakeErie_XGBoost_metrics_by_time_window.{csv,json}`
  — aggregate accuracy metrics (R², RMSE, MAE, bias) per temporal matchup window, produced by
  `validation/apply_frozen_model_lake_erie.py`. The corresponding figures are included in
  the manuscript.

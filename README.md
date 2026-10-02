# isprs-chla-aquamatch

Code and the final trained model associated with the manuscript "Enhancing Large-Scale
Chlorophyll-a Estimation Across U.S. Lakes through Spectral and Spatio-Temporal Feature
Integration: A Machine Learning Approach Using AquaMatch" (PHOTO-D-26-00534), submitted to
the ISPRS Journal of Photogrammetry and Remote Sensing.

The manuscript estimates lake chlorophyll-a from Landsat surface reflectance augmented with
spatial and temporal context, using AquaMatch + siteSR (Collection 2) in-situ/satellite
matchups. Six candidate models were compared, with XGBoost selected as the best model.
Feature-group ablation and SHAP analysis show that adding this geographic and seasonal
context to the spectral predictors improves prediction accuracy.

This is a curated release containing the central analysis scripts supporting the paper's
reported results, rather than every script written during the project. The final
22-predictor set used by the frozen model is documented in the manuscript and embedded in
the model file; exploratory feature-selection scripts are not included in this release.

## Setup

```bash
pip install -r requirements.txt
```

Scripts use three common environment variables for data, results, and the frozen model,
with two workflow-specific variables required by individual scripts (each defaults to a
folder relative to the current directory, shown below). **Run scripts from the repository
root**, or set these to absolute paths:

| Variable | Default | Purpose |
|---|---|---|
| `AQUAMATCH_DATA_DIR` | `./data` | Input data — mostly *not* included; see "Data availability" |
| `AQUAMATCH_RESULTS_DIR` | `./results` | Generated outputs (not included; created on first run) |
| `AQUAMATCH_MODEL_FILE` | `./model/xgboost_frozen_v1.json` | The frozen final model |
| `AQUAMATCH_SOURCE_DIR` | `./source_data` | Used by the two join scripts in `preprocessing/` |
| `AQUAMATCH_LAKE_ERIE_IMAGES_DIR` | `./lake_erie_images` | Used only by `validation/lake_erie_seasonal_application.py` |

## Data availability

This repository does **not** redistribute the primary source datasets — see
`data/README.md` for the three source DOIs (AquaMatch chlorophyll-a, siteSR Landsat
reflectance, and the NOAA NCEI western Lake Erie validation observations). `data/` itself
ships with only this study's own derived summary outputs (aggregate Lake Erie accuracy
metrics). Everything else a script reads from `AQUAMATCH_DATA_DIR` needs to be produced
locally by running the pipeline stages below, starting from the join (step 1).

`validation/lake_erie_seasonal_application.py` additionally needs three Landsat scene
GeoTIFFs (spring/summer/fall) exported from Google Earth Engine; see `preprocessing/gee/`.

## Model

`model/xgboost_frozen_v1.json` stores the final trained model in XGBoost's native JSON
format (not Python pickle/joblib). It was trained with XGBoost 3.4.1, so `requirements.txt`
pins `xgboost==3.4.1` to match; the Lake Erie validation script also checks the installed
XGBoost version before loading the model.

Load it with:
```python
import xgboost as xgb
model = xgb.XGBRegressor()
model.load_model("model/xgboost_frozen_v1.json")
```

The model expects 22 engineered predictors in a fixed order (embedded in the file as
`feature_names`, and also listed as `FEATURES`/`EXPECTED_FEATURES` in each script that loads
it): `red, nir, blue, lat, long, NDVI, NDTI, RNI, GBI, BLRDGR, GNRI, RBI, NIRGI, GDVI, NDAVI,
FAI, MNDWI, SWI, TGI, AFAI, sin_doy, cos_doy`. The spectral indices and cyclical day-of-year
encoding are computed inline in every script that needs them — see e.g.
`model_development/prepare_6models_final.py` for the reference implementation.

## Pipeline, in order

1. **Join Chl-a to reflectance** (`preprocessing/join_aquamatch_sitesr.py`) — joins
   AquaMatch Chl-a to siteSR Landsat reflectance by site and calendar date. `mission`
   comes through directly from siteSR's own data, as does the siteSR sites list's
   NHD-derived HUC8 watershed code for each station (used at step 6). Writes
   `data/raw/aquamatch_sitesr_joined_stage1.csv`.
2. **Scene acquisition times** (`preprocessing/gee/gee_scene_overpass_lookup_time.js`) —
   Google Earth Engine script that looks up the exact scene-center acquisition time for
   each Landsat scene matched at step 1.
3. **Finalize the join** (`preprocessing/finalize_aquamatch_sitesr_join.py`) — merges
   step 2's scene-time export back onto step 1's output (on `sat_id`) and computes
   `timediff`, the hours between the field sample time and the scene's acquisition time.
   Produces `data/raw/aquamatch_sitesr_joined.csv`.
4. **Time-window selection** (`preprocessing/timediff_window_sweep.py`) — sweeps the
   satellite-field matchup window and justifies the ≤2 h satellite–field matchup threshold
   used in the final quality-control workflow (N = 14,125).
5. **Model development** (`model_development/`) — `prepare_6models_final.py` writes the
   fixed train/holdout split; `fit_6models_one_final.py` tunes and fits the six candidate
   models (PLS, KPLS, MLP, RealMLP, RF, and XGBoost), with the script executed separately
   for each `--model` option. `figures/final_model_comparison_figures.py` builds the
   comparison figures.
6. **Robustness and ablation** (`validation/`):
   - `xgb_monte_carlo_35_tuned.py` (35-seed repeated random-split validation) and
     `run_huc4_fixed_35seeds.py` (35-seed station-level spatial validation stratified by
     HUC4; a station's observations stay wholly on one side of a split) each evaluate
     XGBoost under a different repeated-validation framework.
   - `ablation_random_split_mc35.py` and `ablation_huc4_split_35seed.py` reuse those
     splits and hyperparameters to run the feature-group ablation reported in Section 3.4;
     `figures/fig12_feature_group_ablation.py` plots the combined result.
   - The HUC4 scripts need each station's watershed code first: run
     `preprocessing/build_huc8_validation_dataset.py` on step 3's output to produce
     `data/spatial_validation_diagnostics/huc8_field_comparison.csv` (see
     `data/README.md` for why it's tagged at the HUC8, not HUC4, level).
7. **Interpretation** (`interpretation/shap_compute_final_xgb.py` →
   `figures/fig11_15_shap_importance_and_beeswarm.py`) — SHAP values for the final model.
8. **External validation** (`validation/`) —
   `preprocessing/gee/gee_lake_erie_matchup_extraction.js` pulls the separate Lake Erie
   matchups; `apply_frozen_model_lake_erie.py` applies the frozen model to them across
   several time windows. `lake_erie_error_by_chla_bin.py` analyzes the ≤1 h matchup
   predictions from that script's output, while `lake_erie_seasonal_application.py`
   separately applies the same frozen model to three water-masked Landsat scenes.
9. **Error and sensor diagnostics** (`validation/discussion_error_diagnostics.py`,
   `validation/mission_error_table.py` → `figures/fig10_concentration_dependent_error.py`) —
   concentration-dependent error analysis and the LEDAPS-vs-LaSRC atmospheric-correction
   comparison.
10. **Dataset-overview figures** (`figures/fig01_chla_histogram_ecdf.py`,
    `fig02_monthly_chla_mad.py`, `fig04_temporal_encoding_polar_chla.py`) — descriptive
    figures of the cleaned dataset.

## Structure

```
preprocessing/        data acquisition (gee/), the join, the time-window sweep, and the
                       watershed-tagged dataset needed by the HUC4 validation scripts
model_development/    the six-model comparison chain
validation/           repeated random + HUC4-stratified station validation,
                       ablation, Lake Erie external validation, and error/sensor diagnostics
interpretation/       SHAP computation
figures/              plotting scripts
data/                 included data and source links (see "Data availability")
model/                the frozen final model
```

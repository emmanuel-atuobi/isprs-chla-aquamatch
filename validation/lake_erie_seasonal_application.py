#!/usr/bin/env python3
"""
Lake Erie Seasonal Application -- Per-Pixel Chl-a Prediction
==============================================================
Applies the final saved XGBoost model to three Landsat scenes over
western Lake Erie (spring / peak-summer / fall), predicting Chl-a on
every valid (Pekel water-masked) pixel, then builds a 3-row x 2-column
figure: RGB | Predicted Chl-a, one shared horizontal colorbar.

Uses the same spectral-index formulas and FEATURES order as
apply_frozen_model_lake_erie.py, but handles edge cases differently. The
point-matchup script requires all six bands strictly positive and uses
machine epsilon in the index denominators; a continuous raster has far more
near-zero and negative atmospherically-corrected reflectance values, which
that strict filter would drop as gaps across the map. This script instead
keeps every water pixel and floors each index's denominator magnitude at
safe_divide()'s min_abs (see below). These are the per-season Chl-a ranges
reported for Fig. 14.
"""

import os
from pathlib import Path
import re
from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import xy as raster_xy
from pyproj import Transformer
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

# Landsat scene GeoTIFFs exported from GEE for the three seasonal dates below;
# these are not included in this repository (large raster files).
IMAGE_DIR = Path(os.environ.get("AQUAMATCH_LAKE_ERIE_IMAGES_DIR", "./lake_erie_images"))

MODEL_FILE = Path(os.environ.get("AQUAMATCH_MODEL_FILE", "./model/xgboost_frozen_v1.json"))

OUTPUT_DIR = RESULTS_DIR / "lake_erie_application_maps"
PRED_RASTER_DIR = OUTPUT_DIR / "predicted_rasters"
FIG_DIR = OUTPUT_DIR / "figures"

for p in [OUTPUT_DIR, PRED_RASTER_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SCENES -- must match the filenames actually exported from GEE
# (WesternErie_<label>_RGB.tif and WesternErie_<label>_6band_watermasked.tif)
# =============================================================================

SCENES = [
    {"label": "spring_2023-04-12", "row_title": "Spring", "date": "2023-04-12"},
    {"label": "summer_2024-08-12", "row_title": "Summer", "date": "2024-08-12"},
    {"label": "fall_2024-11-08",   "row_title": "Fall",   "date": "2024-11-08"},
]

RGB_SUFFIX = "_RGB.tif"
BAND_SUFFIX = "_6band_watermasked.tif"

# Band order as exported from GEE (Blue, Green, Red, NIR, SWIR1, SWIR2)
BAND_NAMES = ["blue", "green", "red", "nir", "swir1", "swir2"]

# EXACT final feature order -- must match training / external validation script
FEATURES = [
    "red", "nir", "blue", "lat", "long",
    "NDVI", "NDTI", "RNI", "GBI", "BLRDGR", "GNRI", "RBI",
    "NIRGI", "GDVI", "NDAVI", "FAI", "MNDWI", "SWI", "TGI", "AFAI",
    "sin_doy", "cos_doy",
]

DPI = 600


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)

    saved_features = model.get_booster().feature_names
    if saved_features is not None and list(saved_features) != FEATURES:
        raise RuntimeError(
            "Saved model feature order does not match the expected final feature order."
        )

    print(f"Loaded model from: {MODEL_FILE}")
    return model


# =============================================================================
# FEATURE ENGINEERING -- exact match to engineer_features() in the
# external validation script, adapted for full-image arrays instead of
# a DataFrame of point matchups.
# =============================================================================

def safe_divide(numerator, denominator, min_abs=1e-4):
    """
    Division that stays numerically stable even when the denominator is
    negative or very close to zero (both now possible since the strict
    positive-band filter was relaxed). Preserves the denominator's sign
    while flooring its magnitude, rather than relying on machine epsilon
    alone (which barely guards against literal zero, not near-zero).
    """
    denom_floored = np.where(
        denominator >= 0,
        np.maximum(denominator, min_abs),
        np.minimum(denominator, -min_abs),
    )
    return numerator / denom_floored


def engineer_features_array(bands, lat, lon, doy):
    """
    bands: dict of band_name -> 2D np.array (already scaled reflectance)
    lat, lon: 2D np.array, same shape as bands
    doy: scalar day-of-year for this scene's acquisition date
    Returns: dict of feature_name -> 2D np.array, matching FEATURES order/formulas.

    NOTE: the strict "all bands positive" filter has been relaxed for this
    application step (unlike training/validation, which required strictly
    positive reflectance on all 6 bands). Pixels with negative reflectance
    -- physically common for clear water in Landsat C2 SR after atmospheric
    correction -- are now included, using safe_divide() to keep index
    computation numerically stable. This means predictions on those pixels
    are an extrapolation beyond the model's training domain; flag this in
    the figure caption / limitations text.
    """
    blue = bands["blue"]
    green = bands["green"]
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]

    f = {}
    f["red"] = red
    f["nir"] = nir
    f["blue"] = blue
    f["lat"] = lat
    f["long"] = lon

    f["NDVI"] = safe_divide(nir - red, nir + red)
    f["NDTI"] = safe_divide(red - green, red + green)
    f["RNI"] = safe_divide(red, nir)
    f["GBI"] = safe_divide(green, blue)
    f["BLRDGR"] = safe_divide(blue - red, green)
    f["GNRI"] = green - safe_divide(green, red)
    f["RBI"] = safe_divide(red, blue)
    f["NIRGI"] = safe_divide(nir, green)
    f["GDVI"] = nir - green
    f["NDAVI"] = safe_divide(nir - blue, nir + blue)

    fai_baseline = red + (swir1 - red) * (0.86 - 0.66) / (1.60 - 0.66)
    f["FAI"] = nir - fai_baseline

    f["MNDWI"] = safe_divide(green - swir1, green + swir1)
    f["SWI"] = safe_divide(nir - swir1, nir + swir1)

    f["TGI"] = -0.5 * (120 * (red - green) - 190 * (red - blue))

    f["AFAI"] = (nir - red) + 0.5 * (swir1 - red)

    sin_doy = np.sin(2 * np.pi * (doy - 1) / 365.0)
    cos_doy = np.cos(2 * np.pi * (doy - 1) / 365.0)
    f["sin_doy"] = np.full_like(red, sin_doy)
    f["cos_doy"] = np.full_like(red, cos_doy)

    return f


# =============================================================================
# PER-SCENE PROCESSING
# =============================================================================

def get_pixel_lat_lon(src):
    """Compute per-pixel lon/lat arrays (EPSG:4326) from raster transform + CRS."""
    height, width = src.height, src.width
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    xs, ys = raster_xy(src.transform, rows.flatten(), cols.flatten())
    xs = np.array(xs)
    ys = np.array(ys)

    transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)

    lons = lons.reshape(height, width)
    lats = lats.reshape(height, width)
    return lons, lats


def process_scene(scene):
    label = scene["label"]
    date_str = scene["date"]
    doy = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday

    band_path = IMAGE_DIR / f"WesternErie_{label}{BAND_SUFFIX}"
    rgb_path = IMAGE_DIR / f"WesternErie_{label}{RGB_SUFFIX}"
    pred_out_path = PRED_RASTER_DIR / f"WesternErie_{label}_predicted_chla.tif"

    print(f"\n{'=' * 90}")
    print(f"Processing: {label}  (DOY {doy}, date {date_str})")
    print(f"{'=' * 90}")

    with rasterio.open(rgb_path) as rgb_src:
        rgb = rgb_src.read([1, 2, 3]).astype(np.float64)

    # CACHE: if the prediction raster already exists on disk, load it
    # directly instead of re-running the model over 4M+ pixels again.
    if pred_out_path.exists():
        print(f"  Found existing prediction raster -- loading from cache "
              f"(delete the file to force a re-run): {pred_out_path}")
        with rasterio.open(pred_out_path) as src:
            pred_raster = src.read(1).astype(np.float32)
        preds = pred_raster[np.isfinite(pred_raster)]
        print(f"  Loaded {len(preds):,} valid predicted pixels from cache")
        print(f"  Predicted Chl-a range: {preds.min():.2f} - {preds.max():.2f} ug/L "
              f"(mean {preds.mean():.2f})")
        return {
            "label": label,
            "row_title": scene["row_title"],
            "date": date_str,
            "rgb": rgb,
            "prediction": pred_raster,
            "stats": {
                "min": float(preds.min()),
                "mean": float(preds.mean()),
                "median": float(np.median(preds)),
                "max": float(preds.max()),
                "n": int(len(preds)),
            },
        }

    if not band_path.exists():
        raise FileNotFoundError(f"6-band file not found: {band_path}")
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")

    with rasterio.open(band_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        raw_bands = {}
        for i, name in enumerate(BAND_NAMES, start=1):
            raw_bands[name] = src.read(i).astype(np.float64)

        lon, lat = get_pixel_lat_lon(src)

    # Valid-pixel mask: finite reflectance on all 6 bands, water-masked
    # (Pekel, via nodata/NaN in the export). Positivity is NO LONGER
    # required -- relaxed per decision to keep clear-water pixels with
    # negative post-atmospheric-correction reflectance (common for NIR/SWIR
    # over clear water in Landsat C2 SR). Index computation uses
    # safe_divide() to stay numerically stable for these pixels.
    nodata_mask = np.ones((height, width), dtype=bool)
    if nodata is not None:
        for name in BAND_NAMES:
            nodata_mask &= (raw_bands[name] != nodata)

    finite_bands_mask = np.ones((height, width), dtype=bool)
    for name in BAND_NAMES:
        finite_bands_mask &= np.isfinite(raw_bands[name])

    valid = nodata_mask & finite_bands_mask

    total_px = height * width
    n_nodata_excluded = int((~nodata_mask).sum())
    n_nonfinite_band_excluded = int((~finite_bands_mask & nodata_mask).sum())
    n_negative_included = int(
        (valid & np.any([raw_bands[b] < 0 for b in BAND_NAMES], axis=0)).sum()
    )
    n_valid = int(valid.sum())

    print(f"  Raster size: {height}x{width} = {total_px} pixels")
    print(f"  Excluded -- nodata/Pekel-masked:      {n_nodata_excluded}")
    print(f"  Excluded -- non-finite band value:    {n_nonfinite_band_excluded}")
    print(f"  Valid pixels (finite, water-masked):  {n_valid}")
    print(f"    of which have ≥1 negative band:     {n_negative_included} "
          f"(extrapolation beyond training domain)")

    if n_valid == 0:
        raise RuntimeError(f"No valid pixels found for scene {label}")

    features = engineer_features_array(raw_bands, lat, lon, doy)

    # Build the (n_valid, n_features) matrix in the exact FEATURES order
    X = np.column_stack([features[f][valid] for f in FEATURES])

    finite_mask = np.isfinite(X).all(axis=1)
    n_dropped = int((~finite_mask).sum())
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} pixels with non-finite feature values")

    valid_idx = np.argwhere(valid)
    valid_idx = valid_idx[finite_mask]
    X = X[finite_mask]

    print(f"  Predicting Chl-a for {len(X)} pixels...")
    model = scene["_model"]
    preds = model.predict(X).astype(np.float64)

    neg = (preds < 0).sum()
    if neg > 0:
        print(f"  Capping {neg} negative predictions at 0.01 ug/L")
        preds[preds < 0] = 0.01

    print(f"  Predicted Chl-a range: {preds.min():.2f} - {preds.max():.2f} ug/L "
          f"(mean {preds.mean():.2f})")

    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pvals = np.percentile(preds, pctiles)
    print("  Percentile breakdown:")
    for p, v in zip(pctiles, pvals):
        print(f"    p{p:>2}: {v:7.2f} ug/L")

    pred_raster = np.full((height, width), np.nan, dtype=np.float32)
    pred_raster[valid_idx[:, 0], valid_idx[:, 1]] = preds

    # Save prediction raster
    out_path = PRED_RASTER_DIR / f"WesternErie_{label}_predicted_chla.tif"
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=height, width=width, count=1,
        dtype=rasterio.float32, crs=crs, transform=transform,
        nodata=np.nan, compress="lzw",
    ) as dst:
        dst.write(pred_raster, 1)
        dst.update_tags(
            Description="Predicted chlorophyll-a (ug/L)",
            AcquisitionDate=date_str,
            DOY=str(doy),
            Model="Final XGBoost (22 features)",
            ValidPixels=str(len(X)),
        )
    print(f"  Saved: {out_path}")

    return {
        "label": label,
        "row_title": scene["row_title"],
        "date": date_str,
        "rgb": rgb,
        "prediction": pred_raster,
        "stats": {
            "min": float(preds.min()),
            "mean": float(preds.mean()),
            "median": float(np.median(preds)),
            "max": float(preds.max()),
            "n": int(len(preds)),
        },
    }


# =============================================================================
# FIGURE: 3 rows (spring/summer/fall) x 2 cols (RGB | predicted),
# one shared horizontal colorbar underneath.
# =============================================================================

class PiecewiseNorm(matplotlib.colors.Normalize):
    """
    Maps values to color via piecewise-linear interpolation between fixed
    control points, allocating disproportionate color-space to whichever
    value range you choose -- unlike log/symlog/power norms, which follow
    a single fixed mathematical curve regardless of where your data
    actually clusters. Implements inverse() (interpolating the swapped
    control arrays), which matplotlib's colorbar needs internally for
    tick placement.
    """
    def __init__(self, control_values, control_positions, clip=False):
        self.control_values = np.asarray(control_values, dtype=float)
        self.control_positions = np.asarray(control_positions, dtype=float)
        super().__init__(vmin=float(self.control_values[0]),
                          vmax=float(self.control_values[-1]), clip=clip)

    def __call__(self, value, clip=None):
        data = np.ma.asarray(value, dtype=float)
        raw = data.filled(np.nan) if np.ma.is_masked(data) else np.asarray(data)
        result = np.interp(raw, self.control_values, self.control_positions)
        return np.ma.masked_invalid(result)

    def inverse(self, value):
        data = np.asarray(value, dtype=float)
        return np.interp(data, self.control_positions, self.control_values)


def add_stats_box(ax, stats, x=0.98, y=0.02):
    """Place min/mean/max text anchored in the bottom-right corner, fully
    outside the plotted data region (in the blank triangular gap)."""
    text = (f"Min:  {stats['min']:.1f}\n"
            f"Mean: {stats['mean']:.1f}\n"
            f"Max:  {stats['max']:.1f}\n"
            f"(µg/L)")
    # multialignment="left" is the fix: with ha="right", each line of a
    # multi-line string otherwise right-aligns INDIVIDUALLY, staggering
    # the left edges of "Min:"/"Mean:"/"Max:" since they have different
    # total widths (different value digit counts) despite a monospace font.
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11,
            va="bottom", ha="right", multialignment="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#999999"))


def make_scene_norm(preds, cmap_name="turbo"):
    """
    Per-scene rank-based piecewise norm: control points are THIS scene's
    own percentiles, mapped to evenly-spaced color positions. Guarantees
    maximum pattern visibility for each individual scene's own
    distribution, rather than one shared scale trying to compromise
    across scenes with genuinely different spreads (e.g. fall's tighter,
    lower distribution vs. summer's wider one with a much higher max).
    """
    finite = preds[np.isfinite(preds)]
    pctiles = np.array([0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    control_values = np.percentile(finite, pctiles)
    control_positions = pctiles / 100.0

    # de-duplicate any tied values (flat spots in the distribution), keeping
    # arrays sorted and monotonic as PiecewiseNorm/np.interp requires
    control_values, unique_idx = np.unique(control_values, return_index=True)
    control_positions = control_positions[unique_idx]

    cmap = matplotlib.colormaps[cmap_name]
    norm = PiecewiseNorm(control_values, control_positions)
    return norm, cmap, control_values


def make_figure(results, mode="percentile"):
    """
    mode: 'percentile' -- per-scene rank-based color scale, own colorbar
          per row (default -- see make_scene_norm)
          'nla'         -- discrete NLA trophic-category classification,
          still uses one shared/pooled scale since categories are fixed
          thresholds, not data-adaptive
    """
    n_rows = len(results)
    fig = plt.figure(figsize=(10.5, 4.3 * n_rows))

    # 3 columns per row: RGB | Prediction | thin per-row colorbar
    gs = gridspec.GridSpec(
        n_rows, 3,
        width_ratios=[1.0, 1.0, 0.05],
        wspace=0.08, hspace=0.25,
    )

    if mode == "nla":
        all_preds = np.concatenate([
            r["prediction"][np.isfinite(r["prediction"])] for r in results
        ])
        bounds = [0, 2, 7, 30, float(np.nanmax(all_preds)) + 1]
        colors = ["#2166ac", "#67a9cf", "#f4a582", "#b2182b"]
        shared_cmap = matplotlib.colors.ListedColormap(colors)
        shared_norm = matplotlib.colors.BoundaryNorm(bounds, shared_cmap.N)
        cbar_labels = ["Oligotrophic\n(≤2)", "Mesotrophic\n(2–7)",
                       "Eutrophic\n(7–30)", "Hypereutrophic\n(>30)"]

    for i, r in enumerate(results):
        ax_rgb = fig.add_subplot(gs[i, 0])
        rgb = r["rgb"]
        rgb_disp = np.clip(rgb / (np.nanpercentile(rgb, 98) + 1e-9), 0, 1)
        rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
        ax_rgb.imshow(rgb_disp)
        ax_rgb.set_ylabel(f"{r['row_title']}\n{r['date']}", fontsize=20, fontweight="bold")
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])
        if i == 0:
            ax_rgb.set_title("True color (RGB)", fontsize=20)

        ax_pred = fig.add_subplot(gs[i, 1])

        if mode == "nla":
            cmap, norm = shared_cmap, shared_norm
        else:
            # Own scale per row, built from THIS scene's own percentiles --
            # maximizes pattern visibility for each scene's actual spread
            # instead of compromising across scenes with different ranges.
            norm, cmap, _ = make_scene_norm(r["prediction"])

        im = ax_pred.imshow(r["prediction"], cmap=cmap, norm=norm)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        if i == 0:
            ax_pred.set_title("Predicted Chl-a", fontsize=20)

        add_stats_box(ax_pred, r["stats"], x=0.98, y=0.02)

        # Per-row colorbar (vertical, in the thin third column)
        cax = fig.add_subplot(gs[i, 2])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=10)

        if mode == "nla":
            if i == 0:
                tick_positions = [(bounds[k] + bounds[k + 1]) / 2 for k in range(len(bounds) - 1)]
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels(cbar_labels)
            else:
                cbar.set_ticks([])
        else:
            # Label at this row's own percentiles (0/10/25/50/75/90/100)
            p = np.percentile(r["prediction"][np.isfinite(r["prediction"])],
                               [0, 10, 25, 50, 75, 90, 100])
            p = sorted(set(round(v, 1) for v in p))
            cbar.set_ticks(p)
            cbar.set_ticklabels([f"{v:g}" for v in p])

    if mode != "nla":
        fig.text(0.5, -0.01, "Predicted Chl-a (µg/L) -- each row scaled to its own distribution",
                  ha="center", fontsize=13)

    suffix = "_nla" if mode == "nla" else "_percentile"
    for ext in ["png", "pdf"]:
        path = FIG_DIR / f"WesternErie_seasonal_application{suffix}.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = DPI
        fig.savefig(path, **kwargs)

    plt.close(fig)
    print(f"\nSaved combined figure ({mode}) to: {FIG_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_cached = all(
        (PRED_RASTER_DIR / f"WesternErie_{s['label']}_predicted_chla.tif").exists()
        for s in SCENES
    )
    model = None if all_cached else load_model()

    results = []
    for scene in SCENES:
        scene["_model"] = model
        results.append(process_scene(scene))

    make_figure(results, mode="percentile")
    make_figure(results, mode="nla")

    print("\nDone.")
    print(f"Predicted rasters: {PRED_RASTER_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()

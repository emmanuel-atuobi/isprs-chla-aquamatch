"""
Sweeps the satellite-field matchup time window (timediff) while holding every other
quality-filter criterion fixed, to justify the final <=2h cutoff used in the rest of the
pipeline.

Applying the same filter chain (steps 1-8 below) with timediff<=2h reproduces the 14,125-row
dataset used throughout the manuscript. Only step 9, the timediff cutoff, is swept here
instead of fixed, so every other criterion stays constant across windows.

Reads:  DATA_DIR/raw/aquamatch_sitesr_joined.csv  (raw join, untouched)
Writes: RESULTS_DIR/timediff_sweep_results.csv
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get("AQUAMATCH_DATA_DIR", "./data")
RESULTS_DIR = os.environ.get("AQUAMATCH_RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)
INPUT_FILE = f"{DATA_DIR}/raw/aquamatch_sitesr_joined.csv"

TIMEDIFF_WINDOWS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]  # matches manuscript: not tested beyond 6h

FEATURES = [
    'red', 'nir', 'blue', 'lat', 'long',
    'NDVI', 'NDTI', 'RNI', 'GBI', 'BLRDGR', 'GNRI', 'RBI', 'NIRGI',
    'GDVI', 'NDAVI', 'FAI', 'MNDWI', 'SWI', 'TGI', 'AFAI',
    'sin_doy', 'cos_doy',
]

# ── Load and apply the quality filter chain, holding everything fixed except timediff ─────
print("Loading raw joined AquaMatch + siteSR data...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"  Raw joined rows: {len(df):,}")

df = df.rename(columns={
    'med_Blue': 'blue', 'med_Green': 'green', 'med_Red': 'red',
    'med_Nir': 'nir', 'med_Swir1': 'swir1', 'med_Swir2': 'swir2',
    'harmonized_value': 'chl_a', 'lon': 'long',
})

# 1. Lakes/reservoirs/impoundments only
df = df[df['ResolvedMonitoringLocationTypeName'] == 'Lake, Reservoir, Impoundment'].copy()

# 2. Landsat 5/7/8/9
df = df[df['mission'].isin(['LT05', 'LE07', 'LC08', 'LC09'])].copy()

# 3. Chl-a in range
df['chl_a'] = pd.to_numeric(df['chl_a'], errors='coerce')
df = df[(df['chl_a'] > 0) & (df['chl_a'] <= 200)].copy()

# 4. Below method detection limit excluded
df = df[df['mdl_flag'] == 0].copy()

# 5. Replicate consistency: single observation, or CV <= 0.5 across replicates
mask_cv = (df['harmonized_row_count'] == 1) | (df['harmonized_value_cv'] <= 0.5)
df = df[mask_cv].copy()

# 6. Near-surface depth (0-2 m), discrete or integrated sample
discrete = (
    (df['depth_flag'] == 1)
    & (df['harmonized_discrete_depth_value'] >= 0)
    & (df['harmonized_discrete_depth_value'] <= 2)
)
integrated = (
    (df['depth_flag'] == 2)
    & (df['harmonized_top_depth_value'] >= 0)
    & (df['harmonized_top_depth_value'] <= 0.5)
    & (df['harmonized_bottom_depth_value'] >= df['harmonized_top_depth_value'])
    & (df['harmonized_bottom_depth_value'] <= 2)
)
df = df[discrete | integrated].copy()

# 7. All six bands finite and positive
for col in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
positive = np.ones(len(df), dtype=bool)
for col in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']:
    positive &= np.isfinite(df[col].to_numpy()) & (df[col].to_numpy() > 0)
df = df.loc[positive].copy()

# 8. Minimum water-pixel count
df['pCount_dswe1'] = pd.to_numeric(df['pCount_dswe1'], errors='coerce')
df = df[df['pCount_dswe1'] >= 8].copy()

# timediff stays as a column so it can be swept below (step 9)
df['timediff'] = pd.to_numeric(df['timediff'], errors='coerce')
df = df[np.isfinite(df['timediff'])].copy()
df = df.reset_index(drop=True)

for col in ['lat', 'long']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Spectral indices + cyclical day-of-year encoding ─────────────────────────────────
eps = np.finfo(float).eps
df['NDVI'] = (df['nir'] - df['red']) / (df['nir'] + df['red'] + eps)
df['NDTI'] = (df['red'] - df['green']) / (df['red'] + df['green'] + eps)
df['RNI'] = df['red'] / (df['nir'] + eps)
df['GBI'] = df['green'] / (df['blue'] + eps)
df['BLRDGR'] = (df['blue'] - df['red']) / (df['green'] + eps)
df['GNRI'] = df['green'] - (df['green'] / (df['red'] + eps))
df['RBI'] = df['red'] / (df['blue'] + eps)
df['NIRGI'] = df['nir'] / (df['green'] + eps)
df['GDVI'] = df['nir'] - df['green']
df['NDAVI'] = (df['nir'] - df['blue']) / (df['nir'] + df['blue'] + eps)
baseline = df['red'] + (df['swir1'] - df['red']) * (0.86 - 0.66) / (1.60 - 0.66)
df['FAI'] = df['nir'] - baseline
df['MNDWI'] = (df['green'] - df['swir1']) / (df['green'] + df['swir1'] + eps)
df['SWI'] = (df['nir'] - df['swir1']) / (df['nir'] + df['swir1'] + eps)
df['TGI'] = -0.5 * ((120.0 * (df['red'] - df['green'])) - (190.0 * (df['red'] - df['blue'])))
df['AFAI'] = (df['nir'] - df['red']) + 0.5 * (df['swir1'] - df['red'])

date_col = 'ActivityStartDate' if 'ActivityStartDate' in df.columns else 'date'
dates = pd.to_datetime(df[date_col], errors='coerce', format='mixed', dayfirst=False)
doy = dates.dt.dayofyear
df['sin_doy'] = np.sin(2.0 * np.pi * (doy - 1.0) / 365.0)
df['cos_doy'] = np.cos(2.0 * np.pi * (doy - 1.0) / 365.0)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + ['chl_a']).reset_index(drop=True)

pool_n = len(df)
n_at_2h = (df['timediff'].abs() <= 2).sum()
print(f"  Pre-timediff-window pool (all other quality criteria applied): {pool_n:,}")
print(f"  Sanity check -- timediff<=2h subset: {n_at_2h:,} (should be 14,125)")
if n_at_2h != 14125:
    print("  WARNING: does not match the expected 14,125 -- check upstream data hasn't changed.")

# ── Hyperparameter search space ──────────────────────────────────────────────────────
param_dist = {
    'n_estimators':     [300, 400, 500, 600, 700, 800],
    'max_depth':        [4, 5, 6, 7, 8, 9],
    'learning_rate':    [0.05, 0.1, 0.15, 0.2],
    'subsample':        [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 5],
    'gamma':            [0, 0.1, 0.2, 0.3],
    'reg_alpha':        [0.0, 0.25, 0.5, 1.0],
    'reg_lambda':       [0.5, 1.0, 1.5, 2.0],
}

# ── Sweep ─────────────────────────────────────────────────────────────────────────
results = []
for td in TIMEDIFF_WINDOWS:
    sub = df[df['timediff'].abs() <= td].copy()
    n = len(sub)
    if n < 100:
        print(f"\n  timediff<={td}h: only {n} rows, skipping")
        continue

    X = sub[FEATURES]
    y = sub['chl_a']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
        param_distributions=param_dist,
        n_iter=50, scoring='r2',
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        verbose=0, random_state=42, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)

    y_pred = search.best_estimator_.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    bias = np.mean(y_pred - y_test)

    print(f"\ntimediff<={td}h  N={n:,}  CV_R2={search.best_score_:.4f}  "
          f"Holdout_R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  Bias={bias:+.2f}")

    results.append({
        'timediff_window_h': td, 'n_rows': n,
        'cv_r2': round(search.best_score_, 4),
        'holdout_r2': round(r2, 4), 'holdout_rmse': round(rmse, 4),
        'holdout_mae': round(mae, 4), 'holdout_bias': round(bias, 4),
        'best_params': search.best_params_,
    })

rdf = pd.DataFrame(results)
out_path = f'{RESULTS_DIR}/timediff_sweep_results.csv'
rdf.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(rdf[['timediff_window_h', 'n_rows', 'cv_r2', 'holdout_r2', 'holdout_rmse']].to_string(index=False))

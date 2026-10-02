#!/usr/bin/env python3
"""
Applies the quality-filter chain and final feature engineering to the raw AquaMatch join,
then writes the single fixed 75/25 train/holdout split (development_train.csv,
holdout_test.csv) used by every model in the six-model comparison.
"""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.25
DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
INPUT_FILE = DATA_DIR / 'raw' / 'aquamatch_sitesr_joined.csv'
OUT = DATA_DIR / 'model_comparison_6models_final'
SPLIT_DIR = OUT / 'split'
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'red', 'nir', 'blue', 'lat', 'long',
    'NDVI', 'NDTI', 'RNI', 'GBI', 'BLRDGR', 'GNRI', 'RBI', 'NIRGI',
    'GDVI', 'NDAVI', 'FAI', 'MNDWI', 'SWI', 'TGI', 'AFAI',
    'sin_doy', 'cos_doy',
]

def counts(label, frame):
    print(f'{label:<38s}: {len(frame):,}')

print('=' * 100)
print('PREPARING COMMON SIX-MODEL AQUAMATCH DATASET')
print('=' * 100)
print('Input:', INPUT_FILE)
print('Output:', SPLIT_DIR)

df = pd.read_csv(INPUT_FILE, low_memory=False)
counts('Raw joined rows', df)

df = df.rename(columns={
    'med_Blue': 'blue',
    'med_Green': 'green',
    'med_Red': 'red',
    'med_Nir': 'nir',
    'med_Swir1': 'swir1',
    'med_Swir2': 'swir2',
    'harmonized_value': 'chl_a',
    'lon': 'long',
})

# Quality filtering, locked for the final comparison.
df = df[df['ResolvedMonitoringLocationTypeName'] == 'Lake, Reservoir, Impoundment'].copy()
counts('1. Lakes/reservoirs/impoundments', df)

df = df[df['mission'].isin(['LT05', 'LE07', 'LC08', 'LC09'])].copy()
counts('2. Landsat 5/7/8/9', df)

df['chl_a'] = pd.to_numeric(df['chl_a'], errors='coerce')
df = df[(df['chl_a'] > 0) & (df['chl_a'] <= 200)].copy()
counts('3. Chl-a >0 and <=200', df)

df = df[df['mdl_flag'] == 0].copy()
counts('4. mdl_flag = 0', df)

mask_cv = (df['harmonized_row_count'] == 1) | (df['harmonized_value_cv'] <= 0.5)
df = df[mask_cv].copy()
counts('5. Replicate CV <=0.5 or single', df)

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
counts('6. Near-surface depth 0-2 m', df)

for col in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
positive = np.ones(len(df), dtype=bool)
for col in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']:
    positive &= np.isfinite(df[col].to_numpy()) & (df[col].to_numpy() > 0)
df = df.loc[positive].copy()
counts('7. Six bands finite and positive', df)

df['pCount_dswe1'] = pd.to_numeric(df['pCount_dswe1'], errors='coerce')
df = df[df['pCount_dswe1'] >= 8].copy()
counts('8. pCount_dswe1 >= 8', df)

df['timediff'] = pd.to_numeric(df['timediff'], errors='coerce')
df = df[np.isfinite(df['timediff']) & (df['timediff'].abs() <= 2)].copy()
counts('9. Absolute exact time difference <=2 h', df)

df = df.reset_index(drop=True)

for col in ['lat', 'long']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

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

if len(df) != 14125:
    raise RuntimeError(f'Expected quality-filtered N=14,125, found {len(df):,}.')

df['qc_row_id'] = np.arange(len(df), dtype=int)

train_idx, holdout_idx = train_test_split(
    np.arange(len(df)),
    test_size=TEST_SIZE,
    random_state=SEED,
)

keep = ['qc_row_id'] + FEATURES + ['chl_a']
train = df.iloc[train_idx][keep].copy()
holdout = df.iloc[holdout_idx][keep].copy()

if len(train) != 10593 or len(holdout) != 3532:
    raise RuntimeError(
        f'Expected 10,593 development and 3,532 holdout rows; got {len(train):,}/{len(holdout):,}.'
    )

if set(train['qc_row_id']) & set(holdout['qc_row_id']):
    raise RuntimeError('Train/holdout row overlap detected.')

train.to_csv(SPLIT_DIR / 'development_train.csv', index=False)
holdout.to_csv(SPLIT_DIR / 'holdout_test.csv', index=False)

manifest = pd.DataFrame({
    'qc_row_id': df['qc_row_id'],
    'split': np.where(df['qc_row_id'].isin(set(holdout['qc_row_id'])), 'holdout_test', 'development_train'),
})
manifest.to_csv(SPLIT_DIR / 'split_manifest.csv', index=False)

metadata = {
    'seed': SEED,
    'test_size': TEST_SIZE,
    'n_total': len(df),
    'n_train': len(train),
    'n_holdout': len(holdout),
    'n_features': len(FEATURES),
    'features': FEATURES,
    'chl_a_min': float(df['chl_a'].min()),
    'chl_a_max': float(df['chl_a'].max()),
    'chl_a_mean': float(df['chl_a'].mean()),
    'chl_a_median': float(df['chl_a'].median()),
}
(SPLIT_DIR / 'split_metadata.json').write_text(json.dumps(metadata, indent=2))

print('\nFINAL COMMON SPLIT')
print('-' * 100)
print(f'Quality-filtered rows: {len(df):,}')
print(f'Development:  {len(train):,}')
print(f'Holdout test: {len(holdout):,}')
print(f'Predictors:   {len(FEATURES)}')
print('Saved:', SPLIT_DIR)

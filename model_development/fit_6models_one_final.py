#!/usr/bin/env python3
"""
Tunes and fits one of the six final models (PLS, KPLS, MLP, RealMLP, RF, XGBoost) on the
fixed train/holdout split from prepare_6models_final.py, and saves the fitted model,
holdout predictions, and cross-validation search results. Run once per model (--model).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform, pearsonr, spearmanr, linregress
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost as xgb

SEED = 42
CV_FOLDS = 5
XGB_SEARCH_ITERATIONS = 150
RF_SEARCH_ITERATIONS = 120
MLP_SEARCH_ITERATIONS = 120
KPLS_SEARCH_ITERATIONS = 60
REALMLP_HPO_STEPS = 50

DATA_DIR = Path(os.environ.get('AQUAMATCH_DATA_DIR', './data'))
OUT = DATA_DIR / 'model_comparison_6models_final'
SPLIT_DIR = OUT / 'split'
MODEL_ROOT = OUT / 'models'

FEATURES = [
    'red', 'nir', 'blue', 'lat', 'long',
    'NDVI', 'NDTI', 'RNI', 'GBI', 'BLRDGR', 'GNRI', 'RBI', 'NIRGI',
    'GDVI', 'NDAVI', 'FAI', 'MNDWI', 'SWI', 'TGI', 'AFAI',
    'sin_doy', 'cos_doy',
]

DISPLAY = {
    'pls': 'PLS',
    'kpls': 'RBF kernel-feature PLS (Nystroem approximation)',
    'mlp': 'MLP',
    'realmlp': 'RealMLP',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=list(DISPLAY))
a = parser.parse_args()
MODEL = a.model
MODEL_NAME = DISPLAY[MODEL]
CPU_JOBS = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
MODEL_DIR = MODEL_ROOT / MODEL
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    fit = linregress(y_true, y_pred)
    rp, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    nz = y_true != 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0)
    return {
        'n': int(len(y_true)),
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mape': mape,
        'bias': float(np.mean(y_pred - y_true)),
        'pearson_r': float(rp),
        'spearman_rho': float(rho),
        'slope': float(fit.slope),
        'intercept': float(fit.intercept),
    }


def print_metrics(label, m):
    print(f'\n{label}')
    print(f"  n:          {m['n']:,}")
    print(f"  R2:         {m['r2']:.4f}")
    print(f"  RMSE:       {m['rmse']:.3f} ug/L")
    print(f"  MAE:        {m['mae']:.3f} ug/L")
    print(f"  Bias:       {m['bias']:+.3f} ug/L")
    print(f"  MAPE:       {m['mape']:.1f}%")
    print(f"  Pearson r:  {m['pearson_r']:.4f}")
    print(f"  Spearman r: {m['spearman_rho']:.4f}")
    print(f"  Slope:      {m['slope']:.4f}")
    print(f"  Intercept:  {m['intercept']:+.3f}")


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f'{MODEL_NAME} job requested GPU but CUDA is not available.')
    print('GPU:', torch.cuda.get_device_name(0))


train = pd.read_csv(SPLIT_DIR / 'development_train.csv')
holdout = pd.read_csv(SPLIT_DIR / 'holdout_test.csv')
if len(train) != 10593 or len(holdout) != 3532:
    raise RuntimeError(f'Unexpected split sizes: {len(train):,}/{len(holdout):,}')

X_train = train[FEATURES].copy()
y_train = train['chl_a'].astype(float).copy()
X_hold = holdout[FEATURES].copy()
y_hold = holdout['chl_a'].astype(float).copy()
CV = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

print('=' * 100)
print(f'SIX-MODEL FINAL COMPARISON: {MODEL_NAME}')
print('=' * 100)
print(f'Development rows: {len(train):,}')
print(f'Holdout rows:     {len(holdout):,}')
print(f'Predictors:       {len(FEATURES)}')
print(f'CPUs allocated:   {CPU_JOBS}')

search = None
final_model = None
best_params = None
cv_best_score = None
cv_best_sd = None
tuning_method = None
tuning_objective = None
search_budget = None

if MODEL == 'pls':
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('pls', PLSRegression(scale=False, max_iter=2000, tol=1e-6)),
    ])
    search = GridSearchCV(
        pipeline,
        {'pls__n_components': list(range(1, len(FEATURES) + 1))},
        scoring='r2', cv=CV, n_jobs=CPU_JOBS, refit=True,
        verbose=1, return_train_score=False,
    )
    tuning_method = 'GridSearchCV'
    tuning_objective = 'R2'
    search_budget = len(FEATURES)

elif MODEL == 'kpls':
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('kernel', Nystroem(kernel='rbf', random_state=SEED)),
        ('pls', PLSRegression(scale=False, max_iter=2000, tol=1e-6)),
    ])
    search = RandomizedSearchCV(
        pipeline,
        {
            'kernel__gamma': loguniform(1e-3, 1.0),
            'kernel__n_components': [128, 192, 256, 384, 512, 768],
            'pls__n_components': randint(2, 21),
        },
        n_iter=KPLS_SEARCH_ITERATIONS, scoring='r2', cv=CV,
        random_state=SEED, n_jobs=min(4, CPU_JOBS), refit=True,
        verbose=1, return_train_score=False,
    )
    tuning_method = 'RandomizedSearchCV'
    tuning_objective = 'R2'
    search_budget = KPLS_SEARCH_ITERATIONS

elif MODEL == 'mlp':
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('mlp', MLPRegressor(
            random_state=SEED, solver='adam', max_iter=1500,
            early_stopping=True, validation_fraction=0.10,
            n_iter_no_change=30,
        )),
    ])
    search = RandomizedSearchCV(
        pipeline,
        {
            'mlp__hidden_layer_sizes': [
                (64,), (128,), (256,), (512,),
                (64, 32), (128, 64), (256, 128), (512, 256),
                (128, 64, 32), (256, 128, 64), (512, 256, 128),
                (128, 128), (256, 256),
            ],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__alpha': loguniform(1e-6, 1e-1),
            'mlp__learning_rate': ['constant', 'adaptive'],
            'mlp__learning_rate_init': loguniform(1e-4, 2e-2),
            'mlp__batch_size': [32, 64, 128, 256, 512],
            'mlp__beta_1': [0.85, 0.90, 0.95],
            'mlp__beta_2': [0.99, 0.995, 0.999],
        },
        n_iter=MLP_SEARCH_ITERATIONS, scoring='r2', cv=CV,
        random_state=SEED, n_jobs=min(6, CPU_JOBS), refit=True,
        verbose=1, return_train_score=False,
    )
    tuning_method = 'RandomizedSearchCV'
    tuning_objective = 'R2'
    search_budget = MLP_SEARCH_ITERATIONS

elif MODEL == 'rf':
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=1),
        {
            'n_estimators': randint(300, 1501),
            'max_depth': [None, 8, 10, 12, 15, 18, 20, 25, 30, 40],
            'min_samples_split': randint(2, 15),
            'min_samples_leaf': randint(1, 8),
            'max_features': ['sqrt', 'log2', 0.25, 0.35, 0.50, 0.65, 0.80, 1.0],
            'max_samples': [0.60, 0.70, 0.80, 0.90, 1.0],
            'bootstrap': [True],
        },
        n_iter=RF_SEARCH_ITERATIONS, scoring='r2', cv=CV,
        random_state=SEED, n_jobs=CPU_JOBS, refit=True,
        verbose=1, return_train_score=False,
    )
    tuning_method = 'RandomizedSearchCV'
    tuning_objective = 'R2'
    search_budget = RF_SEARCH_ITERATIONS

elif MODEL == 'xgb':
    require_cuda()
    from packaging.version import Version
    base = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': SEED,
        'verbosity': 0,
        'n_jobs': CPU_JOBS,
    }
    if Version(xgb.__version__) >= Version('2.0.0'):
        base.update({'tree_method': 'hist', 'device': 'cuda'})
    else:
        base.update({'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'})
    search = RandomizedSearchCV(
        XGBRegressor(**base),
        {
            'n_estimators': randint(300, 1801),
            'max_depth': randint(3, 11),
            'learning_rate': loguniform(0.015, 0.20),
            'subsample': uniform(0.65, 0.35),
            'colsample_bytree': uniform(0.55, 0.45),
            'min_child_weight': loguniform(0.5, 20.0),
            'gamma': uniform(0.0, 1.0),
            'reg_alpha': loguniform(1e-5, 5.0),
            'reg_lambda': loguniform(0.10, 20.0),
            'max_bin': [128, 256, 512],
        },
        n_iter=XGB_SEARCH_ITERATIONS, scoring='r2', cv=CV,
        random_state=SEED, n_jobs=1, refit=True,
        verbose=1, return_train_score=False,
    )
    tuning_method = 'RandomizedSearchCV'
    tuning_objective = 'R2'
    search_budget = XGB_SEARCH_ITERATIONS

elif MODEL == 'realmlp':
    require_cuda()
    try:
        from pytabkit import RealMLP_HPO_Regressor
    except Exception:
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_HPO_Regressor

    tmp = MODEL_DIR / 'tmp_realmlp'
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    # Use PyTabKit's native/default HPO space used for the reported comparison.
    # n_refit=1 (one final model refit on all development data, matching how the
    # other five algorithms here are finalized) raises
    # NotImplementedError("Refit is not fully implemented...") in this PyTabKit
    # version for RealMLP's refit interface. n_refit=0 is used instead: the final
    # RealMLP model is PyTabKit's own n_cv=5 cross-validation ensemble rather than
    # a single refit model. This is the configuration that produced the reported
    # holdout R2 = 0.685.
    realmlp_kwargs = {
        'n_hyperopt_steps': REALMLP_HPO_STEPS,
        'n_cv': CV_FOLDS,
        'n_refit': 0,
        'val_metric_name': 'rmse',
        'device': 'cuda',
        'random_state': SEED,
        'n_threads': min(8, CPU_JOBS),
        'tmp_folder': str(tmp),
        'verbosity': 1,
    }
    final_model = RealMLP_HPO_Regressor(**realmlp_kwargs)
    tuning_method = 'RealMLP native HPO'
    tuning_objective = 'RMSE'
    search_budget = REALMLP_HPO_STEPS
    best_params = realmlp_kwargs

print(f'Tuning method:    {tuning_method}')
print(f'Tuning objective: {tuning_objective}')
print(f'Search budget:    {search_budget}')
print(f'CV folds:         {CV_FOLDS}')

started = time.perf_counter()
if search is not None:
    search.fit(X_train, y_train)
    final_model = search.best_estimator_
    best_params = search.best_params_
    cv_best_score = float(search.best_score_)
    cv_best_sd = float(search.cv_results_['std_test_score'][search.best_index_])
    pd.DataFrame(search.cv_results_).sort_values('rank_test_score').to_csv(
        MODEL_DIR / 'cv_search_results.csv', index=False
    )
else:
    final_model.fit(X_train, y_train)
runtime_minutes = (time.perf_counter() - started) / 60.0

pred_train = np.asarray(final_model.predict(X_train)).reshape(-1)
pred_hold = np.asarray(final_model.predict(X_hold)).reshape(-1)
train_metrics = evaluate(y_train, pred_train)
hold_metrics = evaluate(y_hold, pred_hold)

print(f'\nRuntime: {runtime_minutes:.2f} min')
if cv_best_score is not None:
    print(f'Best CV R2: {cv_best_score:.4f} +/- {cv_best_sd:.4f}')
print_metrics('TRAINING', train_metrics)
print_metrics('UNTOUCHED HOLDOUT TEST', hold_metrics)
print('\nBest/configuration parameters:')
for k, v in sorted(best_params.items()):
    if k != 'tmp_folder':
        print(f'  {k}: {v}')

pd.DataFrame({
    'qc_row_id': train['qc_row_id'].astype(int),
    'observed_chl_a': y_train.to_numpy(),
    'predicted_chl_a': pred_train,
}).to_csv(MODEL_DIR / 'predictions_train.csv', index=False)

pd.DataFrame({
    'qc_row_id': holdout['qc_row_id'].astype(int),
    'observed_chl_a': y_hold.to_numpy(),
    'predicted_chl_a': pred_hold,
}).to_csv(MODEL_DIR / 'predictions_holdout.csv', index=False)

metadata = {
    'model_key': MODEL,
    'model_name': MODEL_NAME,
    'seed': SEED,
    'features': FEATURES,
    'n_train': len(train),
    'n_holdout': len(holdout),
    'tuning_method': tuning_method,
    'tuning_objective': tuning_objective,
    'cv_folds': CV_FOLDS,
    'search_budget': search_budget,
    'cv_best_score': cv_best_score,
    'cv_best_score_sd': cv_best_sd,
    'best_params': best_params,
    'runtime_minutes': runtime_minutes,
    'train_metrics': train_metrics,
    'holdout_metrics': hold_metrics,
}
(MODEL_DIR / 'metrics.json').write_text(json.dumps(json_safe(metadata), indent=2))

# Save fitted model. Do not discard a successful expensive run just because a
# specific serializer is temperamental; RealMLP keeps its tmp folder as well.
if MODEL == 'realmlp':
    save_ok = False
    try:
        with open(MODEL_DIR / 'model_realmlp.pkl', 'wb') as f:
            pickle.dump({'model': final_model, 'features': FEATURES, 'config': best_params}, f)
        save_ok = True
    except Exception as exc1:
        print('Standard pickle failed:', repr(exc1))
        try:
            import cloudpickle
            with open(MODEL_DIR / 'model_realmlp_cloudpickle.pkl', 'wb') as f:
                cloudpickle.dump({'model': final_model, 'features': FEATURES, 'config': best_params}, f)
            save_ok = True
        except Exception as exc2:
            (MODEL_DIR / 'MODEL_SERIALIZATION_WARNING.txt').write_text(
                f'pickle failed: {repr(exc1)}\ncloudpickle failed: {repr(exc2)}\n'
                f'PyTabKit tmp folder retained at: {best_params.get("tmp_folder")}\n'
            )
            print('WARNING: RealMLP serialization failed; tmp folder retained.')
    print('RealMLP serialized:', save_ok)
else:
    joblib.dump(
        {'model': final_model, 'features': FEATURES, 'best_params': best_params, 'seed': SEED},
        MODEL_DIR / f'model_{MODEL}.joblib',
        compress=3,
    )
    if MODEL == 'xgb':
        final_model.save_model(str(MODEL_DIR / 'model_xgboost.json'))

(MODEL_DIR / 'DONE.txt').write_text(
    f'model={MODEL_NAME}\n'
    f'holdout_r2={hold_metrics["r2"]:.8f}\n'
    f'runtime_minutes={runtime_minutes:.3f}\n'
)

print('\nSaved results to:', MODEL_DIR)
print('DONE')

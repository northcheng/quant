# -*- coding: utf-8 -*-
"""
Training entry points for the ML ranking layer.

Two scopes:
  - train_single: train on a single symbol's history. Useful for debugging
    features and labels without running the full pipeline.
  - train_pool:   train a single model on the pooled (X, y) of all symbols
    in a pool. This is the production path; the model can then be served
    to rank any symbol in the universe.

Models are persisted as pickle files under:
    git/quant/ml/models/{market}_{pool}_h{horizon}_{train_date}.pkl
  (the directory is intentionally git-ignored; a .gitkeep file marks its
  existence for git).

CLI quickstart:
    python -m quant.ml.train --pool us --horizon 5 --train-date 2026-06-04
    python -m quant.ml.train --pool us --symbol AAPL  # single-symbol debug

:author: Beichen Chen
"""
from __future__ import annotations

import argparse
import datetime
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
  import lightgbm as lgb
  _HAS_LGB = True
except ImportError:  # pragma: no cover - explicit, helpful error at runtime
  lgb = None
  _HAS_LGB = False

from quant.ml.dataset import (
  build_label,
  walk_forward_split,
  align_xy,
  make_symbol_dataset,
)


# ----- paths ------------------------------------------------------------------

# git/quant/ml/train.py -> git/quant/ml/ -> git/quant/ -> git/
GIT_PATH   = Path(__file__).resolve().parent.parent.parent
QUANT_ROOT = Path(__file__).resolve().parent.parent  # git/quant  (code repo)

# Models are persisted under <quant_path>/ta_model/ml_models/ by default,
# keeping trained artifacts next to the existing ta_data.pkl / result.pkl
# files and out of the code repo. The directory is created on first save.
DEFAULT_MODELS_SUBDIR = 'ml_models'

DEFAULT_PARAMS: dict[str, Any] = {
  'objective':           'multiclass',
  'num_class':           3,
  'metric':              'multi_logloss',
  'learning_rate':       0.05,
  'num_leaves':          31,
  'feature_fraction':    0.8,
  'bagging_fraction':    0.8,
  'bagging_freq':        5,
  'min_data_in_leaf':    50,
  'lambda_l1':           0.1,
  'lambda_l2':           0.1,
  'verbosity':           -1,
  'num_threads':         4,
}

# ----- logger -----------------------------------------------------------------

logger = logging.getLogger('quant.ml.train')


# ----- path resolution --------------------------------------------------------

def resolve_models_dir(config: dict | None) -> Path:
  """
  Resolve the model output directory. Falls back to <quant_path>/ta_model/
  ml_models/ when a config is provided, else to git/quant/ml/models/.
  """
  if config is not None and 'result_path' in config:
    return Path(config['result_path']) / DEFAULT_MODELS_SUBDIR
  return QUANT_ROOT / 'ml' / 'models'


# ----- helpers ----------------------------------------------------------------

def _normalize_ta_data_keys(ta_data: dict) -> dict:
  """
  bc_technical_analysis.load_data stores per-symbol entries with a '_day'
  suffix in the key (e.g. 'AAPL_day'). Strip that suffix here so the rest
  of the ML code can use clean symbol names. Empty / None entries are
  removed.

  :param ta_data: the inner dict (mapping 'symbol_day' -> DataFrame) as
                  loaded from a <pool>_day_ta_data.pkl.
  :returns: a new dict mapping 'symbol' -> DataFrame.
  """
  if not isinstance(ta_data, dict):
    return {}
  out: dict = {}
  for k, v in ta_data.items():
    sym = str(k)
    if sym.endswith('_day'):
      sym = sym[:-4]
    if v is None or (hasattr(v, '__len__') and len(v) == 0):
      continue
    out[sym] = v
  return out


def _ensure_models_dir(models_dir: Path) -> Path:
  models_dir.mkdir(parents=True, exist_ok=True)
  return models_dir


def _model_filename(market: str, pool: str, horizon: int, train_date: str) -> str:
  return f'{market}_{pool}_h{horizon}_{train_date}.pkl'


def save_model(
  model_bundle: dict,
  market: str,
  pool: str,
  horizon: int,
  train_date: str,
  models_dir: Path | None = None,
) -> Path:
  out_dir = resolve_models_dir(None) if models_dir is None else models_dir
  _ensure_models_dir(out_dir)
  path = out_dir / _model_filename(market, pool, horizon, train_date)
  with open(path, 'wb') as f:
    pickle.dump(model_bundle, f)
  logger.info(f'[save]: {path}')
  return path


def load_model_file(path: str | Path) -> dict:
  with open(path, 'rb') as f:
    return pickle.load(f)


def list_models(
  market: str | None = None,
  pool: str | None = None,
  horizon: int | None = None,
  models_dir: Path | None = None,
) -> list[Path]:
  if models_dir is None:
    models_dir = resolve_models_dir(None)
  if not models_dir.exists():
    return []
  out: list[Path] = []
  for p in models_dir.glob('*.pkl'):
    name = p.stem
    parts = name.split('_')
    if len(parts) < 4:
      continue
    m, pl, h, _date = parts[0], parts[1], parts[2], '_'.join(parts[3:])
    h_n = int(h.lstrip('h')) if h.startswith('h') else None
    if market is not None and m != market:
      continue
    if pool is not None and pl != pool:
      continue
    if horizon is not None and h_n != horizon:
      continue
    out.append(p)
  return sorted(out)


# ----- single-symbol training (debug) -----------------------------------------

def train_single(
  symbol: str,
  df: pd.DataFrame,
  benchmark_df: pd.DataFrame | None = None,
  market: str = 'us',
  horizon: int = 5,
  params: dict | None = None,
  train_window: int = 250,
  test_window: int = 60,
  save: bool = False,
  models_dir: Path | None = None,
) -> dict:
  """
  Train and (optionally) save a single-symbol model. Returns a result dict
  with the model, validation metrics, and (if save=True) the persisted path.
  """
  if not _HAS_LGB:
    raise ImportError('lightgbm is required. pip install lightgbm')

  X, y = make_symbol_dataset(
    df=df,
    benchmark_df=benchmark_df,
    market=market,
    horizon=horizon,
  )
  if len(X) < train_window:
    logger.warning(f'{symbol}: insufficient rows ({len(X)}) for train_window={train_window}')
    return {'symbol': symbol, 'ok': False, 'n_rows': len(X)}

  splits = walk_forward_split(X.index, train_window=train_window, test_window=test_window)
  if not splits:
    return {'symbol': symbol, 'ok': False, 'reason': 'no splits'}

  use_params = {**DEFAULT_PARAMS, **(params or {})}
  booster, last_metrics = _train_with_splits(X, y, splits, use_params)

  bundle = {
    'model':       booster,
    'feature_columns': list(X.columns),
    'label_method':    '3class',
    'horizon':         horizon,
    'market':          market,
    'pool':            'single',
    'symbol':          symbol,
    'train_date':      datetime.date.today().isoformat(),
    'params':          use_params,
    'metrics':         last_metrics,
  }

  result = {'symbol': symbol, 'ok': True, 'n_rows': len(X), 'metrics': last_metrics}
  if save:
    result['path'] = str(save_model(bundle, market=market, pool='single', horizon=horizon, train_date=bundle['train_date'], models_dir=models_dir))
  return result


# ----- pool training (production) ---------------------------------------------

def train_pool(
  ta_data: dict[str, pd.DataFrame],
  benchmark_df: pd.DataFrame | None,
  market: str = 'us',
  pool: str = 'us',
  horizon: int = 5,
  params: dict | None = None,
  train_window: int = 250,
  test_window: int = 60,
  min_history: int = 200,
  label_method: str = '3class',
  label_thresholds: dict[str, float] | None = None,
  save: bool = True,
  train_date: str | None = None,
  models_dir: Path | None = None,
) -> dict:
  """
  Train a single lightgbm model on the pooled (X, y) of all symbols in ta_data.

  :param ta_data: dict mapping symbol -> ta_data DataFrame.
  :param benchmark_df: shared benchmark ta_data.
  :param market, pool, horizon, label_method, label_thresholds, params: see
                     build_label / DEFAULT_PARAMS.
  :param min_history: skip symbols with fewer than this many rows.
  :param save: if True, persist the model under ml/models/ with a date stamp.
  :param train_date: date string used in the persisted filename. Defaults to
                     today.
  :returns: a result dict with the model, feature columns, metrics, and
            (if save=True) the persisted path.
  """
  if not _HAS_LGB:
    raise ImportError('lightgbm is required. pip install lightgbm')

  if train_date is None:
    train_date = datetime.date.today().isoformat()

  frames_X: list[pd.DataFrame] = []
  frames_y: list[pd.Series]   = []
  skipped:  list[str]         = []

  for symbol, df in ta_data.items():
    if df is None or len(df) < min_history or 'Close' not in df.columns:
      skipped.append(symbol)
      continue
    try:
      X, y = make_symbol_dataset(
        df=df,
        benchmark_df=benchmark_df,
        market=market,
        horizon=horizon,
        method=label_method,
        label_thresholds=label_thresholds,
      )
    except Exception as e:
      logger.warning(f'{symbol}: dataset build failed ({e})')
      skipped.append(symbol)
      continue
    if len(X) < train_window:
      skipped.append(symbol)
      continue
    X = X.copy()
    X['__symbol__'] = symbol
    # use a (symbol, date) MultiIndex so concatenating across symbols is
    # unambiguous (dates alone would collide across symbols).
    X.index = pd.MultiIndex.from_arrays(
      [[symbol] * len(X), X.index],
      names=['symbol', 'date'],
    )
    y.index = X.index
    frames_X.append(X)
    frames_y.append(y)

  if not frames_X:
    return {'ok': False, 'reason': 'no symbols had enough data', 'skipped': skipped}

  X_pool = pd.concat(frames_X).sort_index()
  y_pool = pd.concat(frames_y).reindex(X_pool.index)
  # X_pool is now indexed by (symbol, date); drop the per-row __symbol__ col
  # because it's redundant with the index level.
  if '__symbol__' in X_pool.columns:
    X_pool = X_pool.drop(columns=['__symbol__'])

  # align_xy again after concat
  X_pool, y_pool = align_xy(X_pool, y_pool, dropna=True)
  if len(X_pool) < train_window:
    return {'ok': False, 'reason': 'pooled X too small', 'n_rows': len(X_pool), 'skipped': skipped}

  # X_pool is indexed by a (symbol, date) MultiIndex after the concat;
  # walk_forward_split only needs the positional length, so feed it a
  # plain range of dates to avoid MultiIndex / DatetimeIndex coercion.
  splits = walk_forward_split(
    pd.date_range(start='2000-01-01', periods=len(X_pool)),
    train_window=train_window,
    test_window=test_window,
  )

  use_params = {**DEFAULT_PARAMS, **(params or {})}
  if label_method != '3class' and 'num_class' in use_params:
    use_params = {k: v for k, v in use_params.items() if k != 'num_class'}
    use_params['objective'] = 'regression' if label_method == 'regression' else 'binary'

  booster, metrics = _train_with_splits(X_pool, y_pool, splits, use_params)

  bundle = {
    'model':       booster,
    'feature_columns': [c for c in X_pool.columns if c != '__symbol__'],
    'label_method':    label_method,
    'horizon':         horizon,
    'market':          market,
    'pool':            pool,
    'train_date':      train_date,
    'params':          use_params,
    'metrics':         metrics,
    'skipped':         skipped,
  }

  result = {
    'ok':       True,
    'n_rows':   len(X_pool),
    'n_symbols':len(frames_X),
    'skipped':  skipped,
    'metrics':  metrics,
  }
  if save:
    result['path'] = str(save_model(bundle, market=market, pool=pool, horizon=horizon, train_date=train_date, models_dir=models_dir))
  return result


# ----- core training routine --------------------------------------------------

def _train_with_splits(
  X: pd.DataFrame,
  y: pd.Series,
  splits: list[tuple[np.ndarray, np.ndarray]],
  params: dict,
) -> tuple[Any, dict]:
  """
  Train a lightgbm model on the union of the first N-1 splits' training
  portions, with the last split used for early-stopping validation.
  Returns the booster and a dict of metrics.
  """
  if not splits:
    raise ValueError('splits is empty')

  # Use all splits except the last for training, the last for early stop.
  if len(splits) >= 2:
    train_idx = np.concatenate([tr for tr, _ in splits[:-1]])
    val_idx   = splits[-1][1]
  else:
    train_idx, val_idx = splits[0]

  # Drop the __symbol__ column if present (string, not a feature)
  feature_cols = [c for c in X.columns if c != '__symbol__']
  X_train = X.iloc[train_idx][feature_cols]
  y_train = y.iloc[train_idx]
  X_val   = X.iloc[val_idx][feature_cols]
  y_val   = y.iloc[val_idx]

  train_set = lgb.Dataset(X_train, label=y_train)
  val_set   = lgb.Dataset(X_val,   label=y_val, reference=train_set)

  callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
  booster = lgb.train(
    params=params,
    train_set=train_set,
    num_boost_round=2000,
    valid_sets=[val_set],
    valid_names=['val'],
    callbacks=callbacks,
  )

  val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
  metrics = _compute_metrics(y_val.values, val_pred, params.get('objective', 'multiclass'))

  return booster, metrics


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, objective: str) -> dict:
  """Compute a small dict of regression / classification metrics."""
  out: dict[str, float] = {}
  if objective in ('multiclass',):
    pred_class = np.argmax(y_pred, axis=1)
    out['val_accuracy'] = float((pred_class == y_true).mean())
  elif objective == 'binary':
    out['val_accuracy'] = float(((y_pred > 0.5).astype(int) == y_true).mean())
  else:  # regression
    out['val_mse'] = float(np.mean((y_pred - y_true) ** 2))
  return out


# ----- diagnostics ------------------------------------------------------------

def diagnose_pool(
  ta_data: dict[str, pd.DataFrame],
  market: str = 'us',
  horizon: int = 5,
  benchmark_df: pd.DataFrame | None = None,
  min_history: int = 100,
  train_window: int = 80,
) -> pd.DataFrame:
  """
  Walk every symbol in ta_data and report, per symbol:
    - raw_row_count     : rows in the source ta_data
    - feats_row_count   : rows after build_features
    - aligned_row_count : rows after build_label + align_xy
    - reason            : why the symbol is or is not usable
  Returns a DataFrame sorted by aligned_row_count descending.
  Useful when a pool-wide training returns 'no symbols had enough data'.
  """
  rows = []
  for symbol, df in ta_data.items():
    if df is None or len(df) == 0:
      rows.append({'symbol': symbol, 'raw_row_count': 0, 'feats_row_count': 0, 'aligned_row_count': 0, 'reason': 'empty df'})
      continue
    if 'Close' not in df.columns:
      rows.append({'symbol': symbol, 'raw_row_count': len(df), 'feats_row_count': 0, 'aligned_row_count': 0, 'reason': 'no Close column'})
      continue
    if len(df) < min_history:
      rows.append({'symbol': symbol, 'raw_row_count': len(df), 'feats_row_count': 0, 'aligned_row_count': 0, 'reason': f'raw rows < min_history({min_history})'})
      continue
    try:
      X, y = make_symbol_dataset(df=df, benchmark_df=benchmark_df, market=market, horizon=horizon)
    except Exception as e:
      rows.append({'symbol': symbol, 'raw_row_count': len(df), 'feats_row_count': 0, 'aligned_row_count': 0, 'reason': f'dataset build failed: {e}'})
      continue
    feats_rows = len(X)
    if feats_rows == 0:
      rows.append({'symbol': symbol, 'raw_row_count': len(df), 'feats_row_count': 0, 'aligned_row_count': 0, 'reason': 'no rows after features+align'})
      continue
    if feats_rows < train_window:
      rows.append({'symbol': symbol, 'raw_row_count': len(df), 'feats_row_count': feats_rows, 'aligned_row_count': feats_rows, 'reason': f'aligned rows < train_window({train_window})'})
      continue
    rows.append({'symbol': symbol, 'raw_row_count': len(df), 'feats_row_count': feats_rows, 'aligned_row_count': feats_rows, 'reason': 'ok'})
  out = pd.DataFrame(rows)
  if not out.empty:
    out = out.sort_values('aligned_row_count', ascending=False).reset_index(drop=True)
  return out


# ----- CLI --------------------------------------------------------------------

def _cli():
  p = argparse.ArgumentParser(description='Train ML ranking models (lightgbm).')
  p.add_argument('--pool',         type=str,   default='us',    help='pool name, e.g. us / a / company_300 / a_etf')
  p.add_argument('--market',       type=str,   default='us',    help='us / a / hk')
  p.add_argument('--horizon',      type=int,   default=5,       help='forward horizon in trading days')
  p.add_argument('--train-date',   type=str,   default=None,    help='ISO date for the saved model file (default: today)')
  p.add_argument('--symbol',       type=str,   default=None,    help='if set, train on a single symbol only (debug)')
  p.add_argument('--config',       type=str,   default=None,    help='path to ta_config.json (default: ~/git/quant/ta_config.json)')
  p.add_argument('--no-save',      action='store_true',         help='do not persist the trained model')
  p.add_argument('--min-history',  type=int,   default=100,     help='skip symbols with fewer raw rows (default: 100)')
  p.add_argument('--train-window', type=int,   default=80,      help='walk-forward training window (default: 80)')
  p.add_argument('--diagnose',     action='store_true',         help='do not train, just print per-symbol stats and exit')
  p.add_argument('--check',        action='store_true',         help='print the loaded config / data structure and exit (no features, no training)')
  args = p.parse_args()

  logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

  # Build root_paths the same way technical_analyst.py / automatic_trader.py
  # do, so the resulting config['quant_path'] matches the user's local
  # data directory (C:\Users\Dell\quant) where ta_data_*.pkl lives.
  sys.path.insert(0, str(GIT_PATH))
  from quant import bc_technical_analysis as ta_util  # type: ignore

  home_path = Path.home()
  root_paths = {'home_path': home_path, 'git_path': home_path / 'git'}
  if args.config:
    # Trust user-supplied config; still rely on its derived paths.
    import json
    with open(args.config, 'r', encoding='utf-8') as f:
      config = json.load(f)
    # ensure the derived keys are populated even if the user-supplied JSON
    # was minimal.
    for k, v in ta_util.load_config(root_paths).items():
      config.setdefault(k, v)
  else:
    config = ta_util.load_config(root_paths)
  target_list = config['selected_sec_list']

  if args.symbol:
    target_list = {args.pool: [args.symbol]}

  data = ta_util.load_data(target_list=target_list, config=config, interval='day', load_derived_data=True)
  raw_ta_data = data['ta_data'].get(f'{args.pool}_day', {})
  ta_data_pool = _normalize_ta_data_keys(raw_ta_data)

  benchmark_df = None
  benchmark_sym = 'SPY' if args.market == 'us' else 'SH000300'
  if benchmark_sym in ta_data_pool:
    benchmark_df = ta_data_pool[benchmark_sym]

  models_dir = resolve_models_dir(config)

  if args.check:
    print('\n=== check: config ===')
    print(f"home_path    = {config.get('home_path')}")
    print(f"git_path     = {config.get('git_path')}")
    print(f"config_path  = {config.get('config_path')}")
    print(f"quant_path   = {config.get('quant_path')}")
    print(f"data_path    = {config.get('data_path')}")
    print(f"result_path  = {config.get('result_path')}")
    sl = config.get('selected_sec_list', {})
    print(f"\nselected_sec_list keys = {list(sl.keys())}")
    for k, v in sl.items():
      print(f"  {k}: {len(v)} symbols  -> first 5: {list(v)[:5]}")
    print('\n=== check: loaded data ===')
    print(f"data['ta_data'].keys() = {list(data['ta_data'].keys())}")
    for ti, pool in data['ta_data'].items():
      n = len(pool) if hasattr(pool, '__len__') else 'n/a'
      sample = list(pool.keys())[:5] if isinstance(pool, dict) and len(pool) > 0 else []
      print(f"  {ti}: {n} symbols  -> first 5 keys: {sample}")
      if isinstance(pool, dict):
        for s, df in list(pool.items())[:3]:
          print(f"    {s}: type={type(df).__name__}, len={len(df) if df is not None else 'None'}")
    print('\n=== check: after key normalization (_day suffix stripped) ===')
    print(f"ta_data_pool (for --pool {args.pool}): {len(ta_data_pool)} symbols")
    print(f"  first 5 symbols: {list(ta_data_pool.keys())[:5]}")
    print(f"  benchmark {benchmark_sym}: {'present (len=' + str(len(benchmark_df)) + ')' if benchmark_df is not None else 'absent'}")
    return {'check': True}

  if args.diagnose:
    report = diagnose_pool(
      ta_data=ta_data_pool,
      market=args.market,
      horizon=args.horizon,
      benchmark_df=benchmark_df,
      min_history=args.min_history,
      train_window=args.train_window,
    )
    print('\n=== diagnose ===')
    print(report.to_string(index=False))
    n_ok = (report['reason'] == 'ok').sum() if not report.empty else 0
    print(f'\nusable: {n_ok} / {len(report)}')
    return {'diagnose': True, 'n_usable': int(n_ok), 'n_total': int(len(report))}

  if args.symbol:
    res = train_single(
      symbol=args.symbol,
      df=ta_data_pool[args.symbol],
      benchmark_df=benchmark_df,
      market=args.market,
      horizon=args.horizon,
      save=not args.no_save,
      models_dir=models_dir,
      train_window=args.train_window,
    )
    print(res)
  else:
    res = train_pool(
      ta_data=ta_data_pool,
      benchmark_df=benchmark_df,
      market=args.market,
      pool=args.pool,
      horizon=args.horizon,
      save=not args.no_save,
      train_date=args.train_date,
      models_dir=models_dir,
      min_history=args.min_history,
      train_window=args.train_window,
    )
    print({k: v for k, v in res.items() if k != 'metrics' and k != 'skipped'})
    print('metrics:', res.get('metrics'))
  return res


if __name__ == '__main__':
  _cli()

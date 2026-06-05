# -*- coding: utf-8 -*-
"""
Inference for the ML ranking layer.

Two scopes:
  - load_latest_model: find the most-recently-trained model in ml/models/
    matching (market, pool, horizon).
  - predict: apply the model to a single ta_data DataFrame and return the
    probabilities (p_down / p_neutral / p_up) plus a discrete ml_signal.

Integration point (for the next phase, not implemented in this file):
  - bc_technical_analysis.calculate_ta_signal is expected to call predict
    at the end and emit columns ml_proba_up / ml_proba_neutral /
    ml_proba_down / ml_signal that downstream consumers (postprocess,
    signal_trade) read.

:author: Beichen Chen
"""
from __future__ import annotations

import datetime
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
  import lightgbm as lgb  # noqa: F401  (kept for type checks downstream)
  _HAS_LGB = True
except ImportError:  # pragma: no cover
  _HAS_LGB = False

from quant.ml.features import build_features
from quant.ml.train import resolve_models_dir, _normalize_ta_data_keys


# ----- paths ------------------------------------------------------------------

QUANT_ROOT = Path(__file__).resolve().parent.parent
GIT_PATH   = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_SUBDIR = 'ml_models'

logger = logging.getLogger('quant.ml.predict')


# ----- model loading ----------------------------------------------------------

def list_models(
  market: str | None = None,
  pool: str | None = None,
  horizon: int | None = None,
  models_dir: Path | None = None,
) -> list[Path]:
  """Return the list of persisted model files, optionally filtered."""
  if models_dir is None:
    models_dir = resolve_models_dir(None)
  if not models_dir.exists():
    return []
  out: list[Path] = []
  for p in models_dir.glob('*.pkl'):
    # filename pattern: <market>_<pool...>_h<horizon>_<date>
    # pool may itself contain underscores (e.g. company_300), so we parse
    # from both ends instead of by fixed position.
    parts = p.stem.split('_')
    if len(parts) < 4:
      continue
    m = parts[0]
    h_part = parts[-2]
    if not h_part.startswith('h'):
      continue
    try:
      h_n = int(h_part.lstrip('h'))
    except ValueError:
      continue
    pl = '_'.join(parts[1:-2])
    if market  is not None and m  != market:  continue
    if pool    is not None and pl != pool:    continue
    if horizon is not None and h_n != horizon: continue
    out.append(p)
  return sorted(out)


def load_latest_model(
  market: str,
  pool: str,
  horizon: int,
  models_dir: Path | None = None,
) -> dict | None:
  """
  Return the most recent model bundle matching (market, pool, horizon), or
  None if no such model exists.

  The "most recent" criterion is the lexically-largest train_date suffix
  in the filename, which works because dates are ISO-formatted.
  """
  candidates = list_models(market=market, pool=pool, horizon=horizon, models_dir=models_dir)
  if not candidates:
    return None
  latest = candidates[-1]
  with open(latest, 'rb') as f:
    bundle = pickle.load(f)
  logger.info(f'[load]: {latest} (train_date={bundle.get("train_date")})')
  return bundle


# ----- prediction -------------------------------------------------------------

def predict(
  df: pd.DataFrame,
  model_bundle: dict | None = None,
  market: str = 'us',
  pool: str = 'us',
  horizon: int = 5,
  benchmark_df: pd.DataFrame | None = None,
  signal_threshold: dict[str, float] | None = None,
  models_dir: Path | None = None,
) -> pd.DataFrame:
  """
  Apply the model to a single symbol's ta_data and return probabilities +
  ml_signal. If `model_bundle` is None, the function will try to load the
  latest model from disk via load_latest_model.

  :param df: ta_data DataFrame for a single symbol (must have 'Close').
  :param model_bundle: pre-loaded model dict. Saves a disk round-trip when
                       the caller is in a hot loop.
  :param market, pool, horizon: used for model selection and feature config.
  :param benchmark_df: optional benchmark series, forwarded to build_features.
  :param signal_threshold: dict with keys 'buy' and 'sell' (probabilities
                          for p_up and p_down). Default: {'buy': 0.5,
                          'sell': 0.5}. The ml_signal column uses the same
                          b/s/n encoding as bc_technical_analysis:
                            'b' if p_up >= buy
                            's' if p_down >= sell
                            ''  otherwise
  :returns: a pd.DataFrame indexed like df with columns
              ml_proba_down, ml_proba_neutral, ml_proba_up, ml_signal
            (NaN for rows where the model is missing features).
  """
  if not _HAS_LGB:
    raise ImportError('lightgbm is required. pip install lightgbm')

  if signal_threshold is None:
    signal_threshold = {'buy': 0.5, 'sell': 0.5}

  if model_bundle is None:
    model_bundle = load_latest_model(market=market, pool=pool, horizon=horizon, models_dir=models_dir)
  if model_bundle is None:
    logger.warning(f'no model found for {market}/{pool}/h{horizon}; returning NaN frame')
    out = pd.DataFrame(index=df.index)
    for c in ('ml_proba_down', 'ml_proba_neutral', 'ml_proba_up', 'ml_signal'):
      out[c] = np.nan
    return out

  booster = model_bundle['model']
  feature_cols = model_bundle['feature_columns']

  feats = build_features(
    df=df,
    benchmark_df=benchmark_df,
    market=market,
  )
  # align column order with the model's expectation
  missing = [c for c in feature_cols if c not in feats.columns]
  for c in missing:
    feats[c] = np.nan
  feats = feats[feature_cols]

  # lightgbm returns a (n, 3) matrix for multiclass objective
  proba = booster.predict(feats, num_iteration=getattr(booster, 'best_iteration', None))

  # ensure we get a 2D array even for binary / regression models
  if proba.ndim == 1:
    if model_bundle.get('label_method') == '2class':
      proba = np.stack([1.0 - proba, proba], axis=1)
    else:
      proba = np.stack([np.zeros_like(proba), np.zeros_like(proba), proba], axis=1)

  out = pd.DataFrame(index=df.index)
  out['ml_proba_down']    = proba[:, 0]
  out['ml_proba_neutral'] = proba[:, 1]
  out['ml_proba_up']      = proba[:, 2]
  out['ml_signal'] = ''
  out.loc[out['ml_proba_up']   >= signal_threshold['buy'],  'ml_signal'] = 'b'
  out.loc[out['ml_proba_down'] >= signal_threshold['sell'], 'ml_signal'] = 's'
  return out


def predict_pool(
  ta_data: dict[str, pd.DataFrame],
  benchmark_df: pd.DataFrame | None = None,
  market: str = 'us',
  pool: str = 'us',
  horizon: int = 5,
  signal_threshold: dict[str, float] | None = None,
  models_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
  """
  Run predict for every symbol in ta_data and return a dict {symbol: frame}.
  Useful for nightly batch inference; the per-symbol frame contains the
  ml_proba_* and ml_signal columns ready to be merged back into ta_data.
  """
  model_bundle = load_latest_model(market=market, pool=pool, horizon=horizon, models_dir=models_dir)
  out: dict[str, pd.DataFrame] = {}
  for symbol, df in ta_data.items():
    if df is None or len(df) == 0 or 'Close' not in df.columns:
      continue
    try:
      out[symbol] = predict(
        df=df,
        model_bundle=model_bundle,
        market=market,
        pool=pool,
        horizon=horizon,
        benchmark_df=benchmark_df,
        signal_threshold=signal_threshold,
      )
    except Exception as e:
      logger.warning(f'{symbol}: predict failed ({e})')
      continue
  return out


# ----- CLI --------------------------------------------------------------------

def _cli():
  import argparse
  p = argparse.ArgumentParser(description='Run ML inference for a pool.')
  p.add_argument('--pool',   type=str, default='us', help='pool name')
  p.add_argument('--market', type=str, default='us', help='us / a / hk')
  p.add_argument('--horizon', type=int, default=5,   help='forward horizon')
  p.add_argument('--symbol', type=str, default=None, help='if set, predict for a single symbol only')
  p.add_argument('--config',    type=str, default=None, help='path to ta_config.json (default: ~/git/quant/ta_config.json)')
  args = p.parse_args()

  logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

  import sys
  sys.path.insert(0, str(GIT_PATH))
  from quant import bc_technical_analysis as ta_util  # type: ignore

  home_path = Path.home()
  root_paths = {'home_path': home_path, 'git_path': home_path / 'git'}
  if args.config:
    import json
    with open(args.config, 'r', encoding='utf-8') as f:
      config = json.load(f)
    for k, v in ta_util.load_config(root_paths).items():
      config.setdefault(k, v)
  else:
    config = ta_util.load_config(root_paths)
  target_list = config['selected_sec_list']
  if args.symbol:
    target_list = {args.pool: [args.symbol]}

  data = ta_util.load_data(target_list=target_list, config=config, interval='day', load_derived_data=True)
  ta_data_pool = _normalize_ta_data_keys(data['ta_data'].get(f'{args.pool}_day', {}))
  benchmark_sym = 'SPY' if args.market == 'us' else 'SH000300'
  benchmark_df = ta_data_pool.get(benchmark_sym)
  models_dir = resolve_models_dir(config)

  if args.symbol:
    bundle = load_latest_model(
      market=args.market, pool=args.pool, horizon=args.horizon, models_dir=models_dir,
    )
    if bundle is None:
      print(f"[error] no trained model found under {models_dir} for "
            f"market={args.market} pool={args.pool} horizon={args.horizon}")
      return None
    out = predict(
      df=ta_data_pool[args.symbol],
      model_bundle=bundle,
      market=args.market,
      pool=args.pool,
      horizon=args.horizon,
      benchmark_df=benchmark_df,
    )
    print(out.tail())
  else:
    out = predict_pool(
      ta_data=ta_data_pool,
      benchmark_df=benchmark_df,
      market=args.market,
      pool=args.pool,
      horizon=args.horizon,
      models_dir=models_dir,
    )
    for sym, frame in list(out.items())[:5]:
      print(sym, frame.tail(1))
  return out


if __name__ == '__main__':
  _cli()

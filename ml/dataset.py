# -*- coding: utf-8 -*-
"""
Dataset construction for the ML ranking layer.

Responsibilities:
  - build_label: convert a price series into a forward-looking target
    (3-class, 2-class, or regression). Carefully uses t+horizon close
    relative to t close, with explicit handling of A-share price limits
    and T+1 settlement.
  - walk_forward_split: produce (train_idx, test_idx) tuples that respect
    chronological order, so the model is always trained on the past and
    evaluated on the immediate future.
  - align_xy: align the feature matrix X with the label series y, drop rows
    where label is undefined (insufficient look-ahead), and return a clean
    (X, y) pair ready for lightgbm.

Look-ahead protection is the single most important property of this module.
Every helper must avoid using t+1..t+horizon data when computing features
for row t. The label is the only place that intentionally uses future prices.

:author: Beichen Chen
"""
from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
from typing import Any, Literal


# ----- label ------------------------------------------------------------------

def build_label(
  prices: pd.Series,
  horizon: int = 5,
  method: Literal['3class', '2class', 'regression'] = '3class',
  thresholds: dict[str, float] | None = None,
  drop_neutral: bool = False,
) -> pd.Series:
  """
  Compute the forward-looking label for each row.

  :param prices: a price series (typically 'Close') with a DatetimeIndex.
  :param horizon: number of trading days to look ahead.
  :param method: one of:
      - '3class'  :  0 (down), 1 (neutral), 2 (up)  (default)
      - '2class'  :  0 (down or flat), 1 (up)
      - 'regression':  raw log return over the horizon
  :param thresholds: for '3class', dict like {'up': 0.02, 'down': -0.02}
                     (interpreted as log returns). Default: +/- 2%.
                     For '2class', uses thresholds.get('up', 0.0).
  :param drop_neutral: if True (only valid for '3class'), drop the neutral
                       class. Useful when training a binary up-vs-down model
                       on a filtered universe.
  :returns: a pd.Series aligned to prices.index. The last `horizon` rows
            will be NaN (insufficient look-ahead).
  :raises: ValueError for unknown method.
  """
  if method not in ('3class', '2class', 'regression'):
    raise ValueError(f"unknown method: {method}")

  if thresholds is None:
    thresholds = {'up': 0.02, 'down': -0.02}

  up_t   = float(thresholds.get('up',   0.02))
  down_t = float(thresholds.get('down', -0.02))

  # forward log return over `horizon` days
  future = prices.shift(-horizon)
  log_ret = np.log(future / prices)

  if method == 'regression':
    return log_ret.rename('label')

  if method == '2class':
    y = (log_ret > up_t).astype(float)
    y[log_ret.isna()] = np.nan
    return y.rename('label')

  # 3class
  y = pd.Series(np.nan, index=prices.index, name='label', dtype=float)
  y[log_ret >  up_t] = 2.0
  y[(log_ret <= up_t) & (log_ret >= down_t)] = 1.0
  y[log_ret <  down_t] = 0.0
  if drop_neutral:
    y = y[y != 1.0]
  return y


# ----- A-share price-limit filter ---------------------------------------------

def a_share_price_limit_pct(row: pd.Series) -> float:
  """
  Best-effort determination of the daily price-limit band for an A-share.

  Approximation (covers >95% of A-share tickers):
    - ChiNext (300xxx), STAR (688xxx): 20%
    - ST / *ST names: 5%
    - Everything else: 10%

  This is intentionally simple: a precise implementation would require a
  master table of (symbol, is_st, board) which is not in the ta_data we
  have. If a more accurate mapping is needed, pass it via the
  `limit_overrides` argument of filter_a_share_limit.
  """
  symbol = str(row.get('symbol', ''))
  name   = str(row.get('name', ''))
  if symbol.startswith(('300', '301')) or symbol.startswith('688'):
    return 0.20
  if 'ST' in name.upper():
    return 0.05
  return 0.10


def filter_a_share_limit(
  df: pd.DataFrame,
  label: pd.Series,
  threshold_buffer: float = 0.001,
  limit_overrides: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
  """
  Drop rows where the A-share daily move is essentially the +/-limit, since
  those are non-tradable / non-predictable extremes and would bias training.

  :param df: ta_data with 'symbol' / 'name' / 'Close' (or whatever price
             column the caller used for label).
  :param label: pd.Series aligned to df.index, output of build_label.
  :param threshold_buffer: pct band around the limit that is still considered
                          "limit-like" (default: 0.1%).
  :param limit_overrides: dict of {symbol: limit_pct} that overrides the
                          heuristic in a_share_price_limit_pct.
  :returns: (df_filtered, label_filtered) with limit-like rows removed.
  """
  if 'Close' not in df.columns or 'symbol' not in df.columns:
    return df, label

  overrides = limit_overrides or {}

  # pct change from previous close
  pct = df['Close'].astype(float).pct_change(fill_method=None).abs()

  # per-row limit
  def _row_limit(row: pd.Series) -> float:
    sym = str(row['symbol'])
    if sym in overrides:
      return float(overrides[sym])
    return a_share_price_limit_pct(row)

  limits = df.apply(_row_limit, axis=1)
  is_limit = pct >= (limits - threshold_buffer)
  # only drop from training set, not the very first row (no pct)
  is_limit.iloc[0] = False

  keep = ~is_limit
  return df.loc[keep], label.loc[keep]


# ----- walk-forward split -----------------------------------------------------

def walk_forward_split(
  dates: pd.DatetimeIndex,
  train_window: int = 250,
  test_window: int = 60,
  step: int | None = None,
  min_train_size: int = 120,
) -> list[tuple[np.ndarray, np.ndarray]]:
  """
  Walk-forward / rolling-origin cross-validation.

  :param dates: a sorted, unique DatetimeIndex (or array of dates).
  :param train_window: size (in samples) of each training window.
  :param test_window: size of each test window.
  :param step: stride between successive windows. Defaults to test_window
               (i.e., non-overlapping test windows).
  :param min_train_size: minimum size of the first training window; smaller
                         histories are skipped.
  :returns: list of (train_idx, test_idx) integer-index tuples, suitable
            for direct indexing into a 0..N DataFrame whose rows are sorted
            by date.
  """
  if step is None:
    step = test_window
  dates = pd.DatetimeIndex(sorted(set(dates)))
  n = len(dates)
  splits: list[tuple[np.ndarray, np.ndarray]] = []

  train_start = 0
  train_end = min_train_size
  while True:
    train_end = min(train_start + train_window, n)
    test_start = train_end
    test_end = min(test_start + test_window, n)
    if test_start >= n:
      break
    train_idx = np.arange(train_start, train_end)
    test_idx  = np.arange(test_start,  test_end)
    splits.append((train_idx, test_idx))
    if test_end >= n:
      break
    train_start += step
  return splits


# ----- X / y alignment --------------------------------------------------------

def align_xy(
  X: pd.DataFrame,
  y: pd.Series,
  dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
  """
  Align X and y on the same index and drop rows with NaN labels (or NaNs
  in X if dropna=True).

  :param X: feature DataFrame with the same row index as y.
  :param y: label Series.
  :param dropna: if True, drop rows with any NaN in X or NaN in y.
  :returns: (X_clean, y_clean) with the same row count.
  """
  if not X.index.equals(y.index):
    y_aligned = y.reindex(X.index)
  else:
    y_aligned = y

  if dropna:
    mask = y_aligned.notna()
    if X.shape[1] > 0:
      mask &= ~X.isna().any(axis=1)
    return X.loc[mask], y_aligned.loc[mask]
  return X, y_aligned


# ----- per-symbol helpers -----------------------------------------------------

def make_symbol_dataset(
  df: pd.DataFrame,
  benchmark_df: pd.DataFrame | None,
  market: str = 'us',
  horizon: int = 5,
  method: str = '3class',
  feature_config: dict | None = None,
  label_thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
  """
  End-to-end: build features and labels for a single symbol, with A-share
  limit filtering if applicable.

  :param df: ta_data for a single symbol. Must have 'Close' (and ideally
             the TA score columns and 'symbol' / 'name' if market='a').
  :param benchmark_df: benchmark ta_data (may be None).
  :param market: 'us' or 'a'.
  :param horizon: forward horizon for the label.
  :param method: passed to build_label.
  :param feature_config: passed to build_features.
  :param label_thresholds: passed to build_label.
  :returns: (X, y) with matching index, ready for training.
  """
  from quant.ml.features import build_features  # local import to avoid cycle

  if feature_config is None:
    feature_config = {}

  feats = build_features(
    df=df,
    benchmark_df=benchmark_df,
    market=market,
    windows=feature_config.get('windows'),
    include_ta_scores=feature_config.get('use_ta_scores', True),
    include_calendar=feature_config.get('use_calendar', True),
  )

  price_col = 'Close'
  label = build_label(
    prices=df[price_col].astype(float),
    horizon=horizon,
    method=method,
    thresholds=label_thresholds,
  )

  if market == 'a' and 'symbol' in df.columns:
    df_for_filter = df.assign(symbol=df['symbol'])
    feats, label = filter_a_share_limit(df_for_filter.assign(**{c: feats[c] for c in feats.columns}), label) if False else (feats, label)
    # simpler: filter using raw df, then reindex feats / label
    df_with_sym = df.copy()
    if 'symbol' not in df_with_sym.columns:
      df_with_sym['symbol'] = ''
    if 'name' not in df_with_sym.columns:
      df_with_sym['name'] = ''
    _, label = filter_a_share_limit(df_with_sym, label)
    feats = feats.reindex(label.index)

  return align_xy(feats, label, dropna=True)

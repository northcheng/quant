# -*- coding: utf-8 -*-
"""
Feature engineering for the ML ranking layer.

Two layers of features:
  1. Direct consumption: 8 technical scores (position/support/resistant/
     break/boundary/trigger/pattern/trend) + signal + potential + trend
     produced by bc_technical_analysis.calculate_ta_signal. These already
     encode trend / momentum / volatility / pattern information in a
     human-readable form.
  2. Engineered: relative strength vs benchmark, distance to 52w high/low,
     realized volatility, volume structure, calendar features. These add
     cross-sectional and time-series information the TA layer does not
     capture.

The output of build_features is the X matrix (a pd.DataFrame with stable
column names) that the train/predict layer consumes. The target y is
computed by quant.ml.dataset.build_label separately so that the two can
be aligned with explicit look-ahead protection.

:author: Beichen Chen
"""
from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
from typing import Any

# ----- column constants -------------------------------------------------------

# Scores produced by bc_technical_analysis.calculate_ta_signal.
# These are safe to consume directly (no look-ahead, all built on data
# available at the time of computation).
TA_SCORE_COLS: list[str] = [
  'position_score',
  'support_score',
  'resistant_score',
  'break_score',
  'boundary_score',
  'trigger_score',
  'pattern_score',
  'trend_score',
  'signal_score',
  'potential_score',
  'total_score',
]

# Discrete state columns (categorical)
TA_STATE_COLS: list[str] = [
  'trend',          # 'u' / 'd' / '' / 'n'
  'signal',         # 'b' / 's' / '' / 'n'
  'potential',      # 'b' / 's' / '' / 'n'
  'trend_strength_symbol',  # strong/weak indicator
]

# Default look-back windows for engineered features
DEFAULT_WINDOWS: list[int] = [5, 20, 60]


# ----- helpers ----------------------------------------------------------------

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
  """Coerce the index to a DatetimeIndex, sort it, and drop duplicate dates."""
  if not isinstance(df.index, pd.DatetimeIndex):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
  df = df.sort_index()
  df = df[~df.index.duplicated(keep='last')]
  return df


def _safe_pct_change(s: pd.Series, periods: int) -> pd.Series:
  """Percentage change that handles zero / nan prices without warnings."""
  return s.pct_change(periods=periods, fill_method=None)


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
  """Rolling Pearson correlation that propagates nans consistently."""
  return a.rolling(window=window, min_periods=max(2, window // 2)).corr(b)


# ----- main entry point -------------------------------------------------------

def build_features(
  df: pd.DataFrame,
  benchmark_df: pd.DataFrame | None = None,
  market: str = 'us',
  windows: list[int] | None = None,
  include_ta_scores: bool = True,
  include_calendar: bool = True,
) -> pd.DataFrame:
  """
  Build the feature matrix for a single symbol from its ta_data.

  :param df: ta_data for a single symbol. Must contain a 'Close' column and
             (optionally) the TA score / state columns listed in TA_SCORE_COLS
             and TA_STATE_COLS. The index is interpreted as a date.
  :param benchmark_df: ta_data for the benchmark (SPY for us, SH000300 / 000300
                       for a-share). Must contain a 'Close' column with a
                       date-indexed series covering at least the date range of
                       df. If None, relative-strength features are skipped.
  :param market: 'us' or 'a'. Affects only calendar / timezone conventions;
                 the feature set is otherwise identical.
  :param windows: look-back windows (in trading days) for return / volatility
                  / volume features. Default: [5, 20, 60].
  :param include_ta_scores: if True (default), copy TA scores into the output
                            so that the model can learn linear / non-linear
                            combinations of them.
  :param include_calendar: if True, append dayofweek / month / quarter columns.
  :returns: a pd.DataFrame with stable column names and the same DatetimeIndex
            as df. NaNs are preserved (train.py handles dropping them).
  :raises: ValueError if 'Close' column is missing.
  """
  if 'Close' not in df.columns:
    raise ValueError("df must contain a 'Close' column")

  if windows is None:
    windows = DEFAULT_WINDOWS

  df = _ensure_dt_index(df)
  out = pd.DataFrame(index=df.index)

  # ----- 1) raw TA scores (only the columns that actually exist) ----------
  if include_ta_scores:
    for col in TA_SCORE_COLS:
      if col in df.columns:
        out[f'ta_{col}'] = df[col].astype(float)

  # one-hot encode the discrete state columns so the model can pick them up
  for col in TA_STATE_COLS:
    if col in df.columns:
      state = df[col].fillna('').astype(str)
      # Map known codes to fixed column names; unknown codes are bucketed
      # as 'other' to keep cardinality bounded.
      mapping = {
        'trend':        {'u': 1, 'd': -1, 'n': 0, '': 0},
        'signal':       {'b': 1, 's': -1, 'n': 0, '': 0},
        'potential':    {'b': 1, 's': -1, 'n': 0, '': 0},
        'trend_strength_symbol': {'strong': 1, 'weak': -1, '': 0},
      }
      if col in mapping:
        out[f'state_{col}'] = state.map(mapping[col]).fillna(0).astype(float)
      else:
        out[f'state_{col}'] = state.replace({'': 'none'}).astype('category').cat.codes.astype(float)

  # ----- 2) price-based features ------------------------------------------
  close = df['Close'].astype(float)
  high  = df['High'].astype(float)  if 'High'  in df.columns else close
  low   = df['Low'].astype(float)   if 'Low'   in df.columns else close
  vol   = df['Volume'].astype(float) if 'Volume' in df.columns else pd.Series(np.nan, index=df.index)

  # returns over multiple horizons
  for w in windows:
    out[f'ret_{w}d']   = _safe_pct_change(close, w)
    out[f'logret_{w}d'] = np.log(close / close.shift(w))

  # distance to rolling max / min (52w ~ 252 trading days for both us and a-share)
  for w in (60, 252):
    if len(close) >= w:
      rolling_max = close.rolling(window=w, min_periods=w // 2).max()
      rolling_min = close.rolling(window=w, min_periods=w // 2).min()
      out[f'pct_from_{w}d_high'] = (close - rolling_max) / rolling_max
      out[f'pct_from_{w}d_low']  = (close - rolling_min) / rolling_min

  # realized volatility
  log_ret_1d = np.log(close / close.shift(1))
  for w in (5, 20, 60):
    if len(close) >= w:
      out[f'vol_{w}d'] = log_ret_1d.rolling(window=w, min_periods=max(2, w // 2)).std()

  # ----- 3) volume-based features -----------------------------------------
  if 'Volume' in df.columns:
    vol = df['Volume'].astype(float)
    for w in (5, 20):
      if len(vol) >= w:
        vol_ma = vol.rolling(window=w, min_periods=max(2, w // 2)).mean()
        out[f'vol_ma_ratio_{w}d'] = vol / vol_ma.replace(0, np.nan)
        vol_z = (vol - vol_ma) / vol.rolling(window=w, min_periods=max(2, w // 2)).std().replace(0, np.nan)
        out[f'vol_zscore_{w}d'] = vol_z

    # volume * return correlation (smart-money proxy)
    if 'vol' not in dir():  # guard against earlier reassignment
      pass
    log_ret_1d_safe = np.log(close / close.shift(1))
    out['vol_ret_corr_20d'] = _rolling_corr(vol, log_ret_1d_safe.abs(), 20)

  # ----- 4) benchmark-relative features -----------------------------------
  if benchmark_df is not None and 'Close' in benchmark_df.columns:
    bench = _ensure_dt_index(benchmark_df)['Close'].astype(float)
    # align on inner join
    aligned = pd.concat({'s': close, 'b': bench}, axis=1, join='inner')
    aligned.columns = ['s', 'b']
    for w in windows:
      bench_ret = _safe_pct_change(aligned['b'], w)
      stock_ret = _safe_pct_change(aligned['s'], w)
      out[f'rel_strength_{w}d'] = (stock_ret - bench_ret)
    # beta: cov(stock, bench) / var(bench), 60d rolling
    if len(aligned) >= 60:
      cov = aligned['s'].pct_change().rolling(60).cov(aligned['b'].pct_change())
      var = aligned['b'].pct_change().rolling(60).var()
      out['beta_60d'] = (cov / var.replace(0, np.nan))

  # ----- 5) calendar features ---------------------------------------------
  if include_calendar:
    idx = df.index
    out['dow']    = pd.Series(idx.dayofweek,   index=idx).astype(float)
    out['month']  = pd.Series(idx.month,       index=idx).astype(float)
    out['quarter'] = pd.Series(idx.quarter,    index=idx).astype(float)
    out['is_month_end'] = pd.Series(idx.is_month_end.astype(int), index=idx).astype(float)
    out['is_quarter_end'] = pd.Series(idx.is_quarter_end.astype(int), index=idx).astype(float)

  return out


# ----- multi-symbol batch -----------------------------------------------------

def build_pool_features(
  ta_data: dict[str, pd.DataFrame],
  benchmark_df: pd.DataFrame | None,
  market: str = 'us',
  windows: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
  """
  Apply build_features to a pool of symbols, then concatenate the results.

  Each row is tagged with a 'symbol' column so the trainer can split by symbol
  if needed. The output index is a MultiIndex (date, symbol).

  :param ta_data: dict mapping symbol -> ta_data DataFrame.
  :param benchmark_df: benchmark ta_data shared by all symbols.
  :param market: 'us' or 'a'.
  :param windows: passed through to build_features.
  :returns: (X, idx) where X is the concatenated feature matrix (with a
            'symbol' column) and idx is a MultiIndex of (date, symbol).
  """
  frames: list[pd.DataFrame] = []
  for symbol, df in ta_data.items():
    if df is None or len(df) == 0 or 'Close' not in df.columns:
      continue
    feats = build_features(df=df, benchmark_df=benchmark_df, market=market, windows=windows)
    if feats is None or len(feats) == 0:
      continue
    feats = feats.copy()
    feats['__symbol__'] = symbol
    frames.append(feats)

  if not frames:
    return pd.DataFrame(), pd.Series(dtype=object)

  X = pd.concat(frames, axis=0).sort_index()
  # build a clean multiindex (date, symbol) for downstream alignment
  X = X.reset_index().rename(columns={'index': 'date'})
  X = X.set_index(['date', '__symbol__']).sort_index()
  return X, X.index


# ----- sanity-check helper ----------------------------------------------------

def feature_summary(X: pd.DataFrame) -> pd.DataFrame:
  """
  Return a summary DataFrame (per column: dtype, n_nan, n_unique) for quick
  sanity-checking of the engineered features.
  """
  rows = []
  for col in X.columns:
    s = X[col]
    rows.append({
      'column':  col,
      'dtype':   str(s.dtype),
      'n_nan':   int(s.isna().sum()),
      'n_total': int(len(s)),
      'n_unique': int(s.nunique(dropna=True)),
    })
  return pd.DataFrame(rows).sort_values('n_nan', ascending=False).reset_index(drop=True)

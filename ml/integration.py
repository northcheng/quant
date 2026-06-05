# -*- coding: utf-8 -*-
"""
Bridge between bc_technical_analysis and the ML ranking layer.

The single public entry point is :func:`attach_ml_scores`, which appends
five standardized columns to a per-symbol ta_data DataFrame:

    ml_proba_down      probability of price going down     (0..1)
    ml_proba_neutral   probability of price staying flat   (0..1)
    ml_proba_up        probability of price going up       (0..1)
    ml_signal          discrete 'b' / 's' / ''  (mirrors existing 'signal')
    ml_score           continuous ranking score (default = ml_proba_up)

The integration is intentionally opt-in: callers pass `signal_source` =
{'ml', 'total', 'auto', 'off'}. When the model is missing or the source
is 'off', the five columns are still appended (as NaN / empty string) so
downstream code (postprocess, bc_trader) can rely on their presence.

Usage from bc_technical_analysis.calculate_ta_signal:

    from quant.ml.integration import attach_ml_scores
    df = attach_ml_scores(
        df          = df,
        market      = 'us',
        pool        = 'company_300',
        horizon     = 5,
        config      = config,
        signal_source = 'auto',
    )

:author: Beichen Chen
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.ml.predict import load_latest_model, predict

logger = logging.getLogger('quant.ml.integration')


#: The five standardized ML columns.  Order matters for downstream readers.
ML_COLUMNS: tuple[str, ...] = (
    'ml_proba_down',
    'ml_proba_neutral',
    'ml_proba_up',
    'ml_signal',
    'ml_score',
)

#: Valid signal_source values
VALID_SOURCES: tuple[str, ...] = ('ml', 'total', 'auto', 'off')


def _resolve_signal_source(
  signal_source: str,
  market: str,
  pool: str,
  horizon: int,
  models_dir: Path | None = None,
) -> str:
  """
  Normalize the user-supplied signal_source and resolve 'auto' to a concrete
  choice based on whether an ML model is available for (market, pool, horizon).
  'auto' prefers 'ml' if a model is found, else falls back to 'total' (which
  means "no ML columns are filled in; downstream uses the existing total_score").
  """
  if signal_source not in VALID_SOURCES:
    logger.warning(f'unknown signal_source={signal_source!r}, defaulting to "auto"')
    signal_source = 'auto'
  if signal_source != 'auto':
    return signal_source

  # auto: probe for a model
  try:
    bundle = load_latest_model(market=market, pool=pool, horizon=horizon, models_dir=models_dir)
  except Exception as e:  # pragma: no cover - defensive
    logger.warning(f'auto-resolve: load_latest_model raised {e!r}; falling back to "total"')
    return 'total'
  if bundle is None:
    return 'total'
  return 'ml'


def _empty_ml_frame(index: pd.Index) -> pd.DataFrame:
  """Return a DataFrame with the 5 ML columns filled with NaN / empty string."""
  out = pd.DataFrame(index=index)
  for c in ML_COLUMNS:
    if c == 'ml_signal':
      out[c] = ''
    else:
      out[c] = np.nan
  return out


def attach_ml_scores(
  df: pd.DataFrame,
  market: str = 'us',
  pool: str = 'us',
  horizon: int = 5,
  config: dict | None = None,
  models_dir: Path | None = None,
  signal_source: str = 'auto',
  benchmark_df: pd.DataFrame | None = None,
  signal_threshold: dict[str, float] | None = None,
) -> pd.DataFrame:
  """
  Append the 5 standardized ML columns to ``df`` in place and return ``df``.

  :param df: per-symbol ta_data DataFrame. Must have a 'Close' column
             (already the case for outputs of calculate_ta_signal).
  :param market: market tag used to look up the trained model.
  :param pool: pool tag used to look up the trained model.
  :param horizon: forecast horizon (days) used to look up the trained model.
  :param config: optional config dict. If provided, models_dir is derived
                 from it via ``resolve_models_dir`` (in quant.ml.train).
  :param models_dir: explicit path to the model directory; overrides config.
  :param signal_source: one of 'ml', 'total', 'auto', 'off'.
             - 'ml'  : always try to attach ML probabilities.
             - 'auto': try ML; if no model is found, attach NaN columns
                       (behaves like 'total' downstream).
             - 'off' : attach NaN columns (no model invocation).
             - 'total': legacy behavior, attach NaN columns so downstream
                       code keeps working with the existing total_score.
  :param benchmark_df: optional benchmark series forwarded to build_features.
  :param signal_threshold: dict {'buy': float, 'sell': float} for the
             discrete ml_signal column.  Default = {'buy': 0.5, 'sell': 0.5}.
  :returns: the same DataFrame ``df`` with 5 new columns appended.
  :raises: never - all errors are logged and result in NaN ML columns so
           the rest of the pipeline keeps working.
  """
  if df is None or len(df) == 0:
    return df

  # resolve models_dir early so 'auto' can probe for a model
  if models_dir is None and config is not None:
    try:
      from quant.ml.train import resolve_models_dir
      models_dir = resolve_models_dir(config)
    except Exception as e:
      logger.warning(f'could not resolve models_dir from config: {e!r}')

  # resolve source
  source = _resolve_signal_source(signal_source, market, pool, horizon, models_dir=models_dir)
  if source == 'off' or source == 'total':
    # no ML: still attach empty columns for schema stability
    empty = _empty_ml_frame(df.index)
    for c in ML_COLUMNS:
      df[c] = empty[c]
    return df

  # source == 'ml' : actually call the model
  try:
    # resolve models_dir
    if models_dir is None and config is not None:
      try:
        from quant.ml.train import resolve_models_dir
        models_dir = resolve_models_dir(config)
      except Exception as e:
        logger.warning(f'could not resolve models_dir from config: {e!r}')

    bundle = load_latest_model(
      market=market, pool=pool, horizon=horizon, models_dir=models_dir,
    )
    if bundle is None:
      logger.warning(f'no ML model for {market}/{pool}/h{horizon}; '
                     f'attaching empty ML columns (fallback to total_score)')
      empty = _empty_ml_frame(df.index)
      for c in ML_COLUMNS:
        df[c] = empty[c]
      return df

    pred = predict(
      df=df,
      model_bundle=bundle,
      market=market,
      pool=pool,
      horizon=horizon,
      benchmark_df=benchmark_df,
      signal_threshold=signal_threshold,
      models_dir=models_dir,
    )
    # predict returns the 4 raw columns.  Derive ml_score = ml_proba_up.
    if pred is None or len(pred) == 0:
      raise RuntimeError('predict() returned empty frame')

    df['ml_proba_down']    = pred['ml_proba_down']
    df['ml_proba_neutral'] = pred['ml_proba_neutral']
    df['ml_proba_up']      = pred['ml_proba_up']
    df['ml_signal']        = pred['ml_signal'].fillna('')
    df['ml_score']         = df['ml_proba_up']
    return df

  except Exception as e:
    logger.warning(f'attach_ml_scores failed: {e!r}; attaching empty ML columns')
    empty = _empty_ml_frame(df.index)
    for c in ML_COLUMNS:
      df[c] = empty[c]
    return df

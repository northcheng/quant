# -*- coding: utf-8 -*-
"""
Evaluation utilities for the ML ranking layer.

The key diagnostic for a ranking model is the **information coefficient
(IC)**: the Spearman (or Pearson) correlation between the model's predicted
score and the realized forward return. A consistently positive IC is the
hallmark of a useful alpha signal.

Two flavors:
  - ic_score: per-date IC averaged across dates (cross-sectional).
  - layered_backtest: split predictions into N quantile buckets and report
    the mean forward return per bucket. A monotonically increasing series
    of bucket means is a robust sanity check that the model is ranking
    well.

These helpers are intentionally lightweight; they accept numpy arrays or
pandas Series and are easy to call from a notebook or a debug script.

:author: Beichen Chen
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


# ----- pointwise metrics ------------------------------------------------------

def ic_score(
  y_pred: np.ndarray | pd.Series,
  y_true: np.ndarray | pd.Series,
  method: str = 'spearman',
) -> float:
  """
  Compute the (Spearman or Pearson) correlation between prediction and label.

  :param y_pred: predicted score (e.g. p_up, or expected log return).
  :param y_true: realized forward return or 3-class label.
  :param method: 'spearman' (rank IC, recommended for ranking models) or
                 'pearson' (linear IC).
  :returns: the IC value. NaN if either side is degenerate (constant or
            all-NaN).
  """
  y_pred = np.asarray(y_pred, dtype=float)
  y_true = np.asarray(y_true, dtype=float)
  mask = ~(np.isnan(y_pred) | np.isnan(y_true))
  y_pred = y_pred[mask]
  y_true = y_true[mask]
  if len(y_pred) < 2:
    return float('nan')
  if np.std(y_pred) == 0 or np.std(y_true) == 0:
    return float('nan')

  if method == 'spearman':
    return float(pd.Series(y_pred).rank().corr(pd.Series(y_true).rank()))
  if method == 'pearson':
    return float(pd.Series(y_pred).corr(pd.Series(y_true)))
  raise ValueError(f"unknown method: {method}")


def ic_by_date(
  y_pred: pd.Series,
  y_true: pd.Series,
  method: str = 'spearman',
) -> pd.Series:
  """
  Compute IC per date, then return a time series of IC values.
  Both series must share a (date, symbol) MultiIndex.
  """
  if not isinstance(y_pred.index, pd.MultiIndex) or not isinstance(y_true.index, pd.MultiIndex):
    raise ValueError('y_pred and y_true must have a (date, symbol) MultiIndex')

  dates = y_pred.index.get_level_values(0).unique().sort_values()
  out = []
  for d in dates:
    p = y_pred.xs(d, level=0)
    t = y_true.xs(d, level=0)
    aligned = pd.concat([p, t], axis=1, join='inner').dropna()
    if len(aligned) < 5:
      continue
    out.append((d, ic_score(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values, method=method)))
  return pd.Series([v for _, v in out], index=[d for d, _ in out], name=f'ic_{method}')


# ----- layered backtest --------------------------------------------------------

def layered_backtest(
  y_pred: np.ndarray | pd.Series,
  y_true: np.ndarray | pd.Series,
  n_layers: int = 5,
) -> pd.DataFrame:
  """
  Split predictions into n_layers quantile buckets and report per-bucket
  mean / std of y_true. A well-calibrated ranking model should produce a
  monotonically increasing series of means.

  :returns: a DataFrame with columns ['layer', 'n', 'mean', 'std'].
  """
  y_pred = pd.Series(y_pred).reset_index(drop=True)
  y_true = pd.Series(y_true).reset_index(drop=True)
  df = pd.concat([y_pred.rename('pred'), y_true.rename('true')], axis=1).dropna()
  if len(df) == 0:
    return pd.DataFrame(columns=['layer', 'n', 'mean', 'std'])

  try:
    df['layer'] = pd.qcut(df['pred'], q=n_layers, labels=False, duplicates='drop') + 1
  except ValueError:
    # too few unique values; fall back to rank-based bucketing
    df = df.sort_values('pred').reset_index(drop=True)
    df['layer'] = (np.arange(len(df)) * n_layers // len(df)) + 1

  grouped = df.groupby('layer')['true'].agg(['count', 'mean', 'std']).reset_index()
  grouped.columns = ['layer', 'n', 'mean', 'std']
  return grouped


# ----- report -----------------------------------------------------------------

def quick_report(
  y_pred: np.ndarray | pd.Series,
  y_true: np.ndarray | pd.Series,
  n_layers: int = 5,
) -> dict:
  """
  Convenience function that returns the most important metrics as a dict.
  """
  return {
    'ic_spearman': ic_score(y_pred, y_true, method='spearman'),
    'ic_pearson':  ic_score(y_pred, y_true, method='pearson'),
    'layered':     layered_backtest(y_pred, y_true, n_layers=n_layers).to_dict(orient='records'),
  }


# ----- CLI --------------------------------------------------------------------

def _cli():
  import argparse, json, logging, pickle
  from pathlib import Path
  from quant import bc_technical_analysis as ta_util
  from quant.ml.train import resolve_models_dir, _normalize_ta_data_keys
  from quant.ml.predict import predict_pool, list_models, load_latest_model
  from quant.ml.dataset import build_label

  p = argparse.ArgumentParser(description='Evaluate a trained ML model on a pool.')
  p.add_argument('--pool', required=True)
  p.add_argument('--market', default='us')
  p.add_argument('--horizon', type=int, default=5)
  p.add_argument('--config', default=None)
  p.add_argument('--n-layers', type=int, default=5)
  p.add_argument('--oos-tail', type=int, default=20,
                 help='only score the last N rows of each symbol (out-of-sample test window)')
  p.add_argument('--show-insample', action='store_true',
                 help='also print the in-sample numbers (likely inflated)')
  p.add_argument('--model-pool', default=None,
                 help='which pool\'s model to load for prediction. '
                      'Defaults to --pool (in-distribution eval). '
                      'Set to a different pool for cross-pool transfer eval.')
  args = p.parse_args()

  logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

  # ---- config + data --------------------------------------------------------
  import json
  from pathlib import Path
  root_paths = {
    'home_path': Path.home(),
    'git_path':  Path.home() / 'git',
  }
  if args.config:
    with open(args.config, 'r', encoding='utf-8') as f:
      config = json.load(f)
    for k, v in ta_util.load_config(root_paths).items():
      config.setdefault(k, v)
  else:
    config = ta_util.load_config(root_paths)

  target_list = config['selected_sec_list']
  data = ta_util.load_data(target_list=target_list, config=config, interval='day', load_derived_data=True)
  ta_data_pool = _normalize_ta_data_keys(data['ta_data'].get(f'{args.pool}_day', {}))
  # result.pkl may be a per-symbol dict (older pools) or a single cross-sectional
  # DataFrame (newer pools, e.g. company_300). Handle both.
  _raw_result = data.get('result', {}).get(f'{args.pool}_day', None)
  if isinstance(_raw_result, dict):
    result_pool = _normalize_ta_data_keys(_raw_result)
  elif _raw_result is not None and hasattr(_raw_result, 'columns'):
    result_pool = _raw_result.copy()  # cross-sectional DataFrame
  else:
    result_pool = None
  print(f'[load] ta_data_pool: {len(ta_data_pool)} symbols')
  print(f'[load] result_pool:  {"DataFrame " + str(len(result_pool)) + " rows" if hasattr(result_pool, "__len__") and result_pool is not None else "0 symbols"}')
  models_dir = resolve_models_dir(config)

  model_pool = args.model_pool or args.pool
  is_transfer = model_pool != args.pool
  bundle = load_latest_model(market=args.market, pool=model_pool, horizon=args.horizon, models_dir=models_dir)
  if bundle is None:
    print(f'[error] no model under {models_dir} for {args.market}/{model_pool}/h{args.horizon}')
    return None
  tag = f'TRANSFER (model={model_pool} -> data={args.pool})' if is_transfer else f'in-distribution (pool={args.pool})'
  print(f'[load] model: {tag}  train_date={bundle.get("train_date")}  features={len(bundle.get("feature_columns", []))}')

  benchmark_df = ta_data_pool.get('SPY' if args.market == 'us' else 'SH000300')

  # ---- predictions ----------------------------------------------------------
  if is_transfer:
    # Cross-pool transfer: load model from model_pool, apply on args.pool data.
    from quant.ml.predict import predict as _predict_one
    pred_dict = {}
    for sym, df in ta_data_pool.items():
      if df is None or 'Close' not in df.columns:
        continue
      try:
        p = _predict_one(
          df=df,
          model_bundle=bundle,
          market=args.market,
          pool=model_pool,
          horizon=args.horizon,
          benchmark_df=benchmark_df,
        )
        pred_dict[sym] = p
      except Exception as e:
        logger.warning(f'predict failed for {sym}: {e}')
  else:
    pred_dict = predict_pool(
      ta_data=ta_data_pool,
      benchmark_df=benchmark_df,
      market=args.market,
      pool=args.pool,
      horizon=args.horizon,
      models_dir=models_dir,
    )

  # ---- realized forward returns --------------------------------------------
  rows = []
  for sym, df in ta_data_pool.items():
    if df is None or 'Close' not in df.columns:
      continue
    rets = build_label(df['Close'], horizon=args.horizon, method='regression')  # log return t -> t+h
    rets = rets.rename('fwd_ret')
    p = pred_dict.get(sym)
    if p is None or len(p) == 0:
      continue
    merged = pd.concat([rets, p[['ml_proba_up']]], axis=1).dropna()
    merged['symbol'] = sym
    # be tolerant of whatever the source index was called ('Date', 'index', ...)
    merged.index.name = 'date'
    rows.append(merged.reset_index())
  if not rows:
    print('[error] no aligned (pred, return) rows')
    return None
  panel = pd.concat(rows, ignore_index=True)
  panel['date'] = pd.to_datetime(panel['date'])
  print(f'[panel] rows={len(panel)}  symbols={panel["symbol"].nunique()}  dates={panel["date"].nunique()}')

  def _report(sub: pd.DataFrame, label: str):
    if len(sub) < 100:
      print(f'[{label}] not enough rows ({len(sub)}), skip')
      return
    ic_s = ic_score(sub['ml_proba_up'], sub['fwd_ret'], method='spearman')
    ic_p = ic_score(sub['ml_proba_up'], sub['fwd_ret'], method='pearson')
    by_date = sub.groupby('date').apply(
      lambda g: ic_score(g['ml_proba_up'], g['fwd_ret'], method='spearman'),
      include_groups=False,
    ).dropna()
    sub = sub.copy()
    sub['layer'] = sub.groupby('date')['ml_proba_up'].transform(
      lambda s: pd.qcut(s.rank(method='first'), q=args.n_layers, labels=False, duplicates='drop'),
    )
    layer_means = sub.groupby('layer')['fwd_ret'].mean()
    spread = (layer_means.iloc[-1] - layer_means.iloc[0]) if len(layer_means) >= 2 else float('nan')
    print(f'\n========== {label}  rows={len(sub)}  dates={sub["date"].nunique()} ==========')
    print(f'  spearman IC = {ic_s:+.4f}    pearson IC = {ic_p:+.4f}')
    if len(by_date) > 0:
      print(f'  daily IC:    mean={by_date.mean():+.4f}  std={by_date.std():.4f}  '
            f'IC>0={(by_date>0).mean():.1%}  N={len(by_date)}')
    print(f'  layer 0 (lowest  p_up):  mean_fwd_ret={layer_means.iloc[0]:+.4f}' if len(layer_means) else '  no layers')
    if len(layer_means) >= 2:
      print(f'  layer {len(layer_means)-1} (highest p_up):  mean_fwd_ret={layer_means.iloc[-1]:+.4f}')
      print(f'  long-short spread (top - bottom): {spread:+.4f}')

  # ---- OOS: only the tail per symbol (the walk-forward test window) -------
  if args.oos_tail and args.oos_tail > 0:
    panel_sorted = panel.sort_values(['symbol', 'date'])
    oos = panel_sorted.groupby('symbol').tail(args.oos_tail)
    _report(oos, f'OOS (last {args.oos_tail} rows / symbol)')
  if args.show_insample:
    _report(panel, 'IN-SAMPLE (all rows)')

  # ---- ml_signal win rate ---------------------------------------------------
  p2 = pred_dict
  win = {'b': 0, 'b_n': 0, 's': 0, 's_n': 0}
  for sym, df in p2.items():
    if df is None or len(df) == 0:
      continue
    sym_ret = build_label(ta_data_pool[sym]['Close'], horizon=args.horizon, method='regression')
    joined = df.join(sym_ret.rename('fwd_ret'), how='inner')
    b = joined[joined['ml_signal'] == 'b']
    s = joined[joined['ml_signal'] == 's']
    win['b'] += int((b['fwd_ret'] > 0).sum())
    win['b_n'] += int(len(b))
    win['s'] += int((s['fwd_ret'] < 0).sum())
    win['s_n'] += int(len(s))
  print('\n--- ml_signal win rate ---')
  if win['b_n']:
    print(f"  buy  signals:  hit rate = {win['b']/win['b_n']:.2%}  (N={win['b_n']})")
  else:
    print('  buy  signals:  none')
  if win['s_n']:
    print(f"  sell signals:  hit rate = {win['s']/win['s_n']:.2%}  (N={win['s_n']})")
  else:
    print('  sell signals:  none')

  # ---- compare to total_score over the full history -------------------------
  # total_score lives in ta_data.pkl (per-symbol DataFrame, column 'total_score'),
  # so we can build a (date, symbol, total_score, fwd_ret) panel and compute
  # the same pooled + daily IC the ML panel uses.
  if ta_data_pool:
    rows3 = []
    for sym, df in ta_data_pool.items():
      if df is None or len(df) == 0 or 'total_score' not in df.columns:
        continue
      ts_df = df[['total_score']].copy()
      ts_df.index.name = 'date'
      ts_df['symbol'] = sym
      rows3.append(ts_df.reset_index())
    if rows3:
      ts_panel = pd.concat(rows3, ignore_index=True)
      ts_panel['date'] = pd.to_datetime(ts_panel['date'])
      # merge fwd_ret from existing panel
      ts_panel = ts_panel.merge(
        panel[['date', 'symbol', 'fwd_ret']].drop_duplicates(['date', 'symbol']),
        on=['date', 'symbol'], how='inner',
      )
      ts_panel = ts_panel.dropna(subset=['total_score', 'fwd_ret'])
      print(f'\n--- total_score vs forward return  rows={len(ts_panel)}  '
            f'symbols={ts_panel["symbol"].nunique()}  dates={ts_panel["date"].nunique()} ---')
      ts_pool_s = ic_score(ts_panel['total_score'], ts_panel['fwd_ret'], method='spearman')
      ts_pool_p = ic_score(ts_panel['total_score'], ts_panel['fwd_ret'], method='pearson')
      print(f'  pooled   spearman = {ts_pool_s:+.4f}   pearson = {ts_pool_p:+.4f}')
      by_date_ts = ts_panel.groupby('date').apply(
        lambda g: ic_score(g['total_score'], g['fwd_ret'], method='spearman'),
        include_groups=False,
      ).dropna()
      if len(by_date_ts) > 0:
        print(f'  daily    spearman: mean={by_date_ts.mean():+.4f}  std={by_date_ts.std():.4f}  '
              f'IC>0={(by_date_ts>0).mean():.1%}  N={len(by_date_ts)}')
      # layer-by-date by total_score
      ts_panel_sorted = ts_panel.copy()
      ts_panel_sorted['layer'] = ts_panel_sorted.groupby('date')['total_score'].transform(
        lambda s: pd.qcut(s.rank(method='first'), q=args.n_layers, labels=False, duplicates='drop'),
      )
      layer_means_ts = ts_panel_sorted.groupby('layer')['fwd_ret'].mean()
      if len(layer_means_ts) >= 2:
        print(f'  layered  spread (top - bottom): '
              f'{layer_means_ts.iloc[-1] - layer_means_ts.iloc[0]:+.4f}')
        for L in sorted(ts_panel_sorted['layer'].dropna().unique()):
          print(f'    layer {int(L)}: mean_fwd_ret = {layer_means_ts.get(L, float("nan")):+.4f}')

      # ---- side-by-side on the OOS subset --------------------------------
      if 'oos' in dir() and oos is not None and len(oos) > 100:
        sub = oos.merge(
          ts_panel[['date', 'symbol', 'total_score']],
          on=['date', 'symbol'], how='left',
        ).dropna(subset=['total_score'])
        if len(sub) > 100:
          ml_s  = ic_score(sub['ml_proba_up'], sub['fwd_ret'], method='spearman')
          ml_p  = ic_score(sub['ml_proba_up'], sub['fwd_ret'], method='pearson')
          tot_s = ic_score(sub['total_score'], sub['fwd_ret'], method='spearman')
          tot_p = ic_score(sub['total_score'], sub['fwd_ret'], method='pearson')
          # rank-blend
          sub_b = sub.copy()
          sub_b['ml_rank']     = sub_b.groupby('date')['ml_proba_up'].rank(pct=True)
          sub_b['total_rank']  = sub_b.groupby('date')['total_score'].rank(pct=True)
          sub_b['blend']       = 0.5 * sub_b['ml_rank'] + 0.5 * sub_b['total_rank']
          blend_s = ic_score(sub_b['blend'], sub_b['fwd_ret'], method='spearman')
          print('\n========== OOS head-to-head (same dates, same symbols) ==========')
          print(f'  rows = {len(sub)}, dates = {sub["date"].nunique()}')
          print(f'  ml      spearman = {ml_s:+.4f}   pearson = {ml_p:+.4f}')
          print(f'  total   spearman = {tot_s:+.4f}   pearson = {tot_p:+.4f}')
          print(f'  blend   spearman = {blend_s:+.4f}   (50/50 rank avg of ml+total)')
          print(f'  delta  (ml - total)  spearman = {ml_s - tot_s:+.4f}')
          print(f'  delta  (blend - ml) spearman = {blend_s - ml_s:+.4f}')
          print(f'  delta  (blend - total) spearman = {blend_s - tot_s:+.4f}')

  return None


if __name__ == '__main__':
  _cli()

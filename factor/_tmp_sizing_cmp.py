# -*- coding: utf-8 -*-
"""
临时脚本: sizing 对照(equal / tier / linear) —— 为 signal_generator 的 position_weight 定默认口径
================================================================================================
只做一件事: 用 factor_config.json 里各池的**冻结权重(default_scheme)**, 在 IS / OOS / FULL
三个窗口上分别跑 equal / tier / linear 三种 sizing, 对比绩效指标, 看哪个更好.

红线: IS 用于选(定默认口径); OOS / FULL 只作汇报参考.
本脚本临时用, 不属于 factor/ 正式 7 脚本.

用法:
  cd ~/git && python -u -m quant.factor._tmp_sizing_cmp
  cd ~/git && python -u -m quant.factor._tmp_sizing_cmp --pools etf_3x,hs300
"""
import argparse
import json

import pandas as pd

from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor.data import load_pool

WINDOWS = (('is', cfg.TRADE_START, cfg.IS_END),
           ('oos', cfg.OOS_START, None),
           ('full', cfg.TRADE_START, None))
KEYS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'vol',
        'ann_turnover', 'n_trades', 'win_rate']


def load_pools_conf() -> dict:
  with open(cfg.FACTOR_CONFIG, encoding='utf-8') as f:
    d = json.load(f)
  return d.get('pools') or {}


def main():
  ap = argparse.ArgumentParser(description='sizing 对照(equal/tier/linear)')
  ap.add_argument('--pools', default='', help='逗号分隔池名; 空=config 里全部')
  ap.add_argument('--out', default=None, help='可选: 结果 CSV 落盘路径')
  a = ap.parse_args()
  want = [s.strip() for s in a.pools.split(',') if s.strip()]

  log = cfg.get_logger('tmp_sizing_cmp', prefix='tmp_sizing_cmp')
  conf = load_pools_conf()
  rows = []
  for pool, c in conf.items():
    if want and pool not in want:
      continue
    p = c.get('params') or {}
    prep = p.get('prep', 'rank')
    min_cs = p.get('min_cs', cfg.MIN_CS)
    top_k = int(p.get('top_k', cfg.TOP_K))
    exit_rank = int(p.get('exit_rank', cfg.EXIT_RANK))
    schemes = {k: {n: float(w) for n, w in v.items()} for k, v in c['schemes'].items()}
    default = p.get('default_scheme')
    if default not in schemes:
      default = 'greedy' if 'greedy' in schemes else next(iter(schemes))
    w = schemes[default]

    ds = load_pool(pool)
    sigs = cmb.build_signals(ds, prep, sorted(w), min_cs)
    comp = cmb.combine_signals(sigs, w, min_cs)
    ow, cw = ds.wide('open'), ds.wide('close')
    log.info(f'[{pool}] scheme={default} top_k={top_k} exit_rank={exit_rank} '
             f'因子 {len(w)} 个, 数据 {ds.dates.min():%Y-%m-%d}~{ds.dates.max():%Y-%m-%d}')

    for sizing in bt.SIZINGS:
      params = bt.EngineParams(top_k=top_k, exit_rank=exit_rank, sizing=sizing,
                               max_exposure=1.0, per_symbol_cap=0.30,
                               cost_bps=cfg.COST_BPS, band=0.05)
      for win, ws, we in WINDOWS:
        res = bt.run_engine(bt._win(ow, ws, we), bt._win(cw, ws, we),
                            bt._win(comp, ws, we), params)
        s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
        rows.append({'pool': pool, 'scheme': default, 'top_k': top_k,
                     'sizing': sizing, 'window': win,
                     **{k: s.get(k) for k in KEYS}})

  df = pd.DataFrame(rows)
  fmt = lambda x: f'{x:g}'  # noqa: E731
  keys = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'ann_turnover']
  for win in ('is', 'oos', 'full'):
    sub = df[df['window'] == win]
    if sub.empty:
      continue
    print(f'\n===== window = {win} =====')
    for (pool, scheme, top_k), g in sub.groupby(['pool', 'scheme', 'top_k'], sort=False):
      print(f'--- {pool}  scheme={scheme}  top_k={top_k} ---')
      print(g.set_index('sizing')[KEYS].to_string(float_format=fmt))

  # IS 判优: 逐池看 sharpe / calmar / total_ret 谁最高
  is_sub = df[df['window'] == 'is']
  if not is_sub.empty:
    print('\n===== IS 判优(每池各指标最优 sizing) =====')
    tally = {m: {} for m in ('sharpe', 'calmar', 'total_ret')}
    for pool, g in is_sub.groupby('pool'):
      line = []
      for m in ('sharpe', 'calmar', 'total_ret'):
        b = g.loc[g[m].idxmax()]
        line.append(f'{m}->{b["sizing"]}({b[m]:.2f})')
        tally[m][b['sizing']] = tally[m].get(b['sizing'], 0) + 1
      print(f'{pool:14s} ' + '  '.join(line))
    print()
    for m, t in tally.items():
      print(f'{m} 最优次数: {t}')
    print('\n三档平均(IS):')
    print(is_sub.groupby('sizing')[['total_ret', 'sharpe', 'calmar']].mean()
          .to_string(float_format=lambda x: f'{x:.3f}'))

  if a.out and not df.empty:
    df.to_csv(a.out, index=False, encoding='utf-8-sig')
    log.info(f'结果已写入: {a.out}')


if __name__ == '__main__':
  main()

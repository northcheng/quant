# -*- coding: utf-8 -*-
"""
factor.wf_step3_sens — Step3 **临时脚本**: 关键超参敏感性与跨池稳健性
====================================================================
Step1 结论: 把接受判据换成 WF 折外 sharpe 后, 搜索器(fold 目标)确实能把 FULL 表现做上去,
  但 IS->OOS 缺口在最重的 company 池反而更大 —— "只换判据不够".
Step2 结论: 换权重方案(逆波动 inv_vol / 风险平价 rp / 坐标下降 cd_ic)里, 只有以折外目标
  驱动的 cd_ic 能在 FULL/OOS 平均意义上超过旧预设, 代价是 company 池缺口变大; 纯风险类
  加权(inv_vol/rp)反而普遍弱于预设.

Step3 要回答三个问题(全部只用 IS 定参; OOS 仅事后汇报):
  A) **shrink**(信任 IS 实测 IC 的程度): cur_ic 的 shrink 从 0 扫到 1, 缺口/表现怎么变?
     -> 是否存在"少信 IS"更稳的拐点.
  B) **thresh**(冗余阈值): 去重门槛 0.5/0.7/0.9, 候选集大小与稳健性的权衡.
  C) **wf_folds / agg**(折外目标的口径): 折数 4/6 与聚合 mean/min 对 cd_ic 的
     折外过拟合有无抑制? (min 要求每折都不差, 理论上更保守)
  D) **跨池同规则**: 由 A/B/C 选出"一套规则", 逐池套用, 看它相对各池预设是否全面不劣
     —— 这是"多池共用一套权重/规则"红线的落地检验.

这是**临时脚本**: 只放一次性编排与报表. 通用能力(inv_vol/rp/coord_descent/ic 收缩/
  折外目标)已落在 combine.py / backtest.py.

自检:
  cd ~/git && python -u -m quant.factor.wf_step3_sens --pools etf_3x --parts shrink
  cd ~/git && python -u -m quant.factor.wf_step3_sens --out out.csv
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor import evaluate as ev
from quant.factor import repro_old_presets as repro          # 复用临时脚本的信号/口径
from quant.factor.data import load_pool

SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
THRESH_GRID = (0.5, 0.7, 0.9)
FOLDS_GRID = (4, 6)
AGG_GRID = ('mean', 'min')
BASE_SHRINK = 0.5
BASE_THRESH = 0.7
BASE_FOLDS = 4
PARTS = ('shrink', 'thresh', 'folds')


# ---------------------------- 测量 ----------------------------

def _measure(open_w, close_w, comp, p, folds, windows, agg='mean') -> dict:
  """一份组合在给定折集/聚合下的折外目标 + 4 窗口指标(与 repro 同口径)."""
  fo = bt.make_fold_objective(open_w, close_w, p, folds, metric='sharpe', agg=agg)
  out = {'fold_obj': round(float(fo({}, comp)), 4)}
  for win, ws, we in windows:
    ow, cw = bt._win(open_w, ws, we), bt._win(close_w, ws, we)
    res = bt.run_engine(ow, cw, bt._win(comp, ws, we), p)
    out[win] = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
  return out


def _ic_weights(sigs, fwd_is, names, shrink: float) -> dict:
  """ic 加权 + shrink 收缩(在临时脚本内实现, 因旧预设名不在 factor 注册表). 见 Step2 说明."""
  w = cmb.ic_weights(sigs, fwd_is, names=names, shrink=0.0)
  n = len(w)
  if n == 0:
    return {}
  eq = 1.0 / n
  return {k: round((1.0 - shrink) * v + shrink * eq, 6) for k, v in w.items()}


def _cands(sigs, fwd_is, thresh: float) -> tuple:
  scores = cmb.rank_ic_scores(sigs, fwd_is, min_cs=cfg.MIN_CS)
  kept, dropped = cmb.dedupe(sigs, scores, thresh=thresh, min_cs=cfg.MIN_CS)
  return (kept or list(sigs.keys())), dropped


# ---------------------------- 单池 ----------------------------

def run_pool(pool: str, parts: tuple, exclude=None, cd_passes: int = 3) -> list:
  ds = load_pool(pool, exclude=exclude)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  sigs = repro.native_signals(ds)
  preset_w, preset_k = repro.PRESETS[pool]
  windows = repro.WINDOWS
  p = repro.engine_params(preset_k)
  fwd = ev.forward_return(ds, h=1)
  fwd_is = bt._win(fwd, cfg.TRADE_START, cfg.IS_END)
  folds4 = cmb.split_folds(ds.dates, cfg.TRADE_START, cfg.IS_END, BASE_FOLDS)

  rows = []

  def add(part, scheme, dim, value, w, folds, agg='mean'):
    comp = cmb.combine_signals(sigs, w, min_cs=cfg.MIN_CS)
    m = _measure(open_w, close_w, comp, p, folds, windows, agg)
    rows.append({'pool': pool, 'part': part, 'scheme': scheme, 'dim': dim,
                 'value': value, 'n_w': len(w), 'agg': agg,
                 'fold_obj': m['fold_obj'],
                 **{win: m[win]['sharpe'] for win in ('is', 'oos', 'full', 'ytd26')},
                 'sh_full': m['full']['sharpe'], 'full_ret': m['full']['total_ret']})

  # --- 参照: preset / buyhold(@ 预设 top_k) ---
  add('ref', 'preset', 'preset', np.nan, dict(preset_w), folds4)
  fw = next((ws, we) for win, ws, we in windows if win == 'full')   # buyhold 仅 full 单行
  s = bt.perf_stats(*repro._bh_args(
      bt.buyhold(bt._win(open_w, fw[0], fw[1]), bt._win(close_w, fw[0], fw[1]))))
  rows.append({'pool': pool, 'part': 'ref', 'scheme': 'buyhold', 'dim': 'buyhold',
               'value': np.nan, 'n_w': np.nan, 'agg': np.nan, 'fold_obj': np.nan,
               'is': np.nan, 'oos': np.nan, 'full': np.nan, 'ytd26': np.nan,
               'sh_full': s['sharpe'], 'full_ret': s['total_ret']})

  # --- A) shrink 扫描: cur_ic @ thresh=0.7, folds=4 ---
  if 'shrink' in parts:
    cands, _ = _cands(sigs, fwd_is, BASE_THRESH)
    for s in SHRINK_GRID:
      w = _ic_weights(sigs, fwd_is, cands, s)
      if w:
        add('shrink', 'cur_ic', 'shrink', s, w, folds4)

  # --- B) 冗余阈值扫描: cur_ic @ shrink=0.5, folds=4 ---
  if 'thresh' in parts:
    for th in THRESH_GRID:
      cands, _ = _cands(sigs, fwd_is, th)
      w = _ic_weights(sigs, fwd_is, cands, BASE_SHRINK)
      if w:
        add('thresh', 'cur_ic', 'thresh', th, w, folds4)

  # --- C) 折外目标口径扫描: cd_ic(初值 cur_ic @ shrink=0.5) ---
  if 'folds' in parts:
    cands, _ = _cands(sigs, fwd_is, BASE_THRESH)
    w0 = _ic_weights(sigs, fwd_is, cands, BASE_SHRINK)
    if w0:
      for nf in FOLDS_GRID:
        folds = cmb.split_folds(ds.dates, cfg.TRADE_START, cfg.IS_END, nf)
        for agg in AGG_GRID:
          cache = {}

          def obj(ww, comp, folds=folds, agg=agg):
            key = tuple(sorted((n, round(float(x), 6)) for n, x in ww.items()))
            if key not in cache:
              f = bt.make_fold_objective(open_w, close_w, p, folds,
                                         metric='sharpe', agg=agg)
              cache[key] = f(ww, comp)
            return cache[key]

          w_cd, _ = cmb.coord_descent_weights(sigs, w0, obj, min_cs=cfg.MIN_CS,
                                              max_passes=cd_passes)
          if w_cd:
            add('folds', 'cd_ic', f'folds={nf}|agg={agg}', nf, w_cd, folds, agg)

  return rows


# ---------------------------- 报表 ----------------------------

def _table(rows: list) -> pd.DataFrame:
  df = pd.DataFrame(rows)
  df['gap_is_oos'] = df['is'] - df['oos']
  return df


def _part_summary(df: pd.DataFrame, part: str):
  sub = df[(df['part'] == part)].copy()
  if sub.empty:
    return
  print(f"\n===== [{part}] 逐池逐档 (gap_is_oos = is-oos, 越小越稳) =====")
  cols = ['pool', 'scheme', 'dim', 'value', 'n_w', 'agg', 'fold_obj',
          'is', 'oos', 'gap_is_oos', 'sh_full', 'full_ret']
  print(sub[cols].to_string(index=False, float_format=lambda x: f'{x:g}'))
  g = sub.pivot_table(index='pool', columns=['dim', 'value'], values='gap_is_oos',
                      aggfunc='first')
  print(f"  [{part}] gap_is_oos 透视:")
  print(g.to_string(float_format=lambda x: f'{x:g}'))


def main():
  ap = argparse.ArgumentParser(description='Step3 临时脚本: 超参敏感性 + 跨池稳健性')
  ap.add_argument('--pools', default=','.join(repro.PRESETS.keys()))
  ap.add_argument('--parts', default=','.join(PARTS), help='逗号分隔: shrink,thresh,folds')
  ap.add_argument('--cd-passes', type=int, default=3, help='坐标下降最大轮数')
  ap.add_argument('--exclude', default=None)
  ap.add_argument('--out', default=None)
  a = ap.parse_args()

  pools = [s.strip() for s in a.pools.split(',') if s.strip()]
  parts = tuple(s.strip() for s in a.parts.split(',') if s.strip())
  exclude = [s.strip() for s in a.exclude.split(',')] if a.exclude else None
  log = cfg.get_logger('wfstep3')

  rows = []
  for pool in pools:
    if pool not in repro.PRESETS:
      log.warning(f'跳过未知池: {pool}')
      continue
    rows.extend(run_pool(pool, parts, exclude, a.cd_passes))

  df = _table(rows)
  if df.empty:
    print('无结果')
    return
  if a.out:
    df.to_csv(a.out, index=False, encoding='utf-8-sig')
    log.info(f'结果已写入: {a.out}')

  for part in parts:
    _part_summary(df, part)

  # --- D) 跨池同规则: 逐池用预设基准对比"候选规则"的缺口/表现 ---
  print('\n===== [D] 跨池对照 (基准=各池 preset) =====')
  ref = df[df['scheme'] == 'preset'].set_index('pool')
  cand = df[df['part'] != 'ref'].copy()
  if not cand.empty:
    cand['gap_ref'] = cand['pool'].map(ref['gap_is_oos'])
    cand['d_gap'] = cand['gap_is_oos'] - cand['gap_ref']             # <0 = 比预设更稳
    cand['sh_ref'] = cand['pool'].map(ref['sh_full'])
    cand['d_sh_full'] = cand['sh_full'] - cand['sh_ref']
    piv_g = cand.pivot_table(index='pool', columns=['part', 'dim', 'value'],
                             values='d_gap', aggfunc='first')
    print('  各池 gap 相对预设的变化(d_gap = 候选 - preset; 负=更稳):')
    print(piv_g.to_string(float_format=lambda x: f'{x:g}'))
    piv_s = cand.pivot_table(index='pool', columns=['part', 'dim', 'value'],
                             values='d_sh_full', aggfunc='first')
    print('  各池 FULL sharpe 相对预设的变化(d_sh_full; 正=更好):')
    print(piv_s.to_string(float_format=lambda x: f'{x:g}'))


if __name__ == '__main__':
  main()

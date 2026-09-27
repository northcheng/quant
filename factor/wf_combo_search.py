# -*- coding: utf-8 -*-
"""
factor.wf_combo_search — Step1 **临时脚本**: 以"WF 折外指标"为判据的组合搜索(三方对照)
================================================================================
背景: 旧系统(bc_combo_search)的权重/组合搜索**只以 IS sharpe 为接受判据**, 容易过拟合
  样本内那一段. 本脚本把"接受判据"换成 **IS 期内部切成若干折后的折外均值 sharpe**
  (combine.greedy_weight_search + backtest.make_fold_objective), 再看它能否在
  IS/OOS/FULL 上同时不劣于现有两套方案.

三方对照(同信号、同口径, 仅"权重+top_k 的来源"不同):
  - preset  : 旧系统 5 组预设权重 @ 预设 top_k(基准).
  - cur_ic  : 当前系统 ic 加权(shrink=0.5)@ 预设 top_k.
  - new     : 本脚本搜索出的权重 @ 搜索出的 top_k(用 WF 折外 sharpe 定参).
  另附 equal(全候选等权 @ 预设 top_k) 与 buyhold(等权买入持有) 作参照.

红线(所有定参只用 IS; OOS 仅事后汇报):
  (a) new 的折外 sharpe 不低于 preset / cur_ic(即"换成折外判据"没有牺牲折外表现);
  (b) new 在 FULL 窗对 buyhold 有超额(total_ret / sharpe).

这是**临时脚本**: 只放"三步探索"特有的一次性编排, 通用能力(搜索骨架/折外目标)已落在
  combine.greedy_weight_search / refine_weights 与 backtest.folded_perf / make_fold_objective.

自检:
  cd ~/git && python -m quant.factor.wf_combo_search --pools etf_3x
  cd ~/git && python -m quant.factor.wf_combo_search --out out.csv
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

TOPK_GRID = (5, 8, 12)                                       # 联搜的 top_k 档位(旧系统同款)
N_FOLDS = 4                                                  # IS 期内部折数
STAT_COLS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'vol',
             'ann_turnover', 'n_trades', 'win_rate', 'avg_days', 'n_days']


# ---------------------------- 单方案评估 ----------------------------

def _stats(open_w, close_w, comp, p, ws, we) -> dict:
  """在某一窗口回测并取指标(与 repro_old_presets 同口径)."""
  ow, cw = bt._win(open_w, ws, we), bt._win(close_w, ws, we)
  res = bt.run_engine(ow, cw, bt._win(comp, ws, we), p)
  return bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])


def _fold_obj_cached(open_w, close_w, p, folds):
  """折外目标(带缓存): 同一权重组合不重复跑引擎 —— 贪心/精修会大量重复试算."""
  cache = {}

  def objective(weights, comp):
    key = tuple(sorted((n, round(float(w), 6)) for n, w in weights.items()))
    if key not in cache:
      base = bt.make_fold_objective(open_w, close_w, p, folds, metric='sharpe')
      cache[key] = base(weights, comp)
    return cache[key]
  return objective


def _cur_ic_weights(sigs, fwd_is, shrink: float) -> dict:
  """
  当前系统 ic 加权(shrink 收缩). 这里在临时脚本内实现收缩而非直接调 cmb.ic_weights,
  原因是本脚本的信号键名沿用旧预设名(F_er20 ...), 不在 factor 注册表里, 而
  cmb.ic_weights 的收缩目标 dir_equal_weights 需要按注册表查 direction.

  本脚本这 10 个信号**先验方向全为 +1**(N_range20 / N_idiovol60 已取负, 其余皆"越大越好"),
  故 dir_equal == equal, 收缩目标就是等权 1/N —— 与 cmb.ic_weights(shrink) 在本信号集上等价.
  """
  w = cmb.ic_weights(sigs, fwd_is, names=list(sigs.keys()), shrink=0.0)
  n = len(w)
  if n == 0:
    return {}
  eq = 1.0 / n
  return {k: round((1.0 - shrink) * v + shrink * eq, 6) for k, v in w.items()}


def search_weights(sigs, open_w, close_w, candidates, top_k, folds,
                   signed: bool, max_n: int, min_gain: float) -> tuple:
  """某一 top_k 下: 贪心离散权重搜索 + 精修 -> (weights, fold_obj)."""
  p = repro.engine_params(top_k)
  obj = _fold_obj_cached(open_w, close_w, p, folds)
  w, _ = cmb.greedy_weight_search(sigs, candidates, obj, min_cs=cfg.MIN_CS,
                                  weight_grid=cmb.WEIGHT_GRID, max_n=max_n,
                                  min_gain=min_gain, signed=signed, first_min=-np.inf)
  if not w:
    return {}, -np.inf
  w, _ = cmb.refine_weights(sigs, w, obj, min_cs=cfg.MIN_CS)
  return w, obj(w, cmb.combine_signals(sigs, w, cfg.MIN_CS))


# ---------------------------- 单池全流程 ----------------------------

def run_pool(pool: str, signed: bool, max_n: int, min_gain: float,
             exclude=None, n_folds: int = N_FOLDS) -> tuple:
  """跑一个池: 三方(及 equal/buyhold) x 4 窗口; 返回 (结果行, 搜索详情, 区间描述)."""
  ds = load_pool(pool, exclude=exclude)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  sigs = repro.native_signals(ds)                            # 10 个信号(键名对齐旧预设)
  preset_w, preset_k = repro.PRESETS[pool]
  windows = repro.WINDOWS
  folds = cmb.split_folds(ds.dates, cfg.TRADE_START, cfg.IS_END, n_folds)

  # --- 只依赖 IS 的定参材料 ---
  fwd = ev.forward_return(ds, h=1)
  fwd_is = bt._win(fwd, cfg.TRADE_START, cfg.IS_END)
  scores = cmb.rank_ic_scores(sigs, fwd_is, min_cs=cfg.MIN_CS)
  kept, dropped = cmb.dedupe(sigs, scores, thresh=cmb.DEDUPE_RHO, min_cs=cfg.MIN_CS)
  cands = kept or list(sigs.keys())                          # 冗余去重后的候选

  # --- 三套权重方案 ---
  schemes = {'preset': (preset_w, preset_k)}
  w_ic = _cur_ic_weights(sigs, fwd_is, 0.5)                  # 固定 0.5: 对齐 Step1 文档口径(不随 IC_SHRINK 漂移)
  schemes['cur_ic'] = (w_ic, preset_k)
  schemes['equal'] = (cmb.equal_weights(list(sigs.keys())), preset_k)

  # new: 联合搜 top_k × weights(以折外 sharpe 定参)
  best = None                                                # (obj, top_k, weights)
  tk_hist = []
  n_sym = next(iter(sigs.values())).shape[1]
  for tk in [k for k in TOPK_GRID if k <= n_sym] or [preset_k]:
    w, o = search_weights(sigs, open_w, close_w, cands, tk, folds, signed, max_n, min_gain)
    tk_hist.append({'top_k': tk, 'fold_obj': round(float(o), 4), 'weights': w})
    if best is None or o > best[0]:
      best = (o, tk, w)
  if best and best[2]:
    schemes['new'] = (best[2], best[1])

  # --- 逐方案逐窗口回测 ---
  rows = []
  for name, (w, tk) in schemes.items():
    comp = cmb.combine_signals(sigs, w, min_cs=cfg.MIN_CS)
    p = repro.engine_params(tk)
    obj = bt.make_fold_objective(open_w, close_w, p, folds, metric='sharpe')
    fobj = obj(w, comp)
    for win, ws, we in windows:
      s = _stats(open_w, close_w, comp, p, ws, we)
      rows.append({'pool': pool, 'scheme': name, 'window': win, 'top_k': tk,
                   'n_w': len(w), 'fold_obj': round(float(fobj), 4),
                   'weights': _wstr(w), **{k: s.get(k) for k in STAT_COLS}})
  # buyhold
  for win, ws, we in windows:
    ow, cw = bt._win(open_w, ws, we), bt._win(close_w, ws, we)
    bh = bt.buyhold(ow, cw)
    s = bt.perf_stats(*repro._bh_args(bh))
    rows.append({'pool': pool, 'scheme': 'buyhold', 'window': win, 'top_k': np.nan,
                 'n_w': np.nan, 'fold_obj': np.nan, 'weights': '',
                 **{k: s.get(k) for k in STAT_COLS}})

  detail = {'pool': pool, 'preset_w': preset_w, 'preset_k': preset_k,
            'kept': kept, 'dropped': dropped,
            'cur_ic_w': w_ic, 'new_w': best[2] if best else {},
            'new_k': best[1] if best else None, 'topk_hist': tk_hist,
            'folds': [(f'{a:%Y-%m-%d}', f'{b:%Y-%m-%d}') for a, b in folds]}
  actual = f'{ds.dates.min():%Y-%m-%d}~{ds.dates.max():%Y-%m-%d}'
  return rows, detail, actual


def _wstr(w: dict) -> str:
  return ';'.join(f'{n}:{w[n]:g}' for n in w)


# ---------------------------- 输出 ----------------------------

def _print_pool(detail: dict, actual: str, rows: list):
  pool = detail['pool']
  print(f"\n===== {pool}  数据 {actual} =====")
  print(f"  预设权重: {detail['preset_w']} @ top_k={detail['preset_k']}")
  print(f"  IS 去重保留: {detail['kept']}")
  if detail['dropped']:
    print(f"  IS 去重丢弃: {detail['dropped']}")
  print(f"  cur_ic 权重: {_wstr(detail['cur_ic_w'])}")
  for h in detail['topk_hist']:
    print(f"  [top_k={h['top_k']}] fold_obj={h['fold_obj']}  "
          f"new_w={_wstr(h['weights']) or '(空)'}")
  df = pd.DataFrame(rows)
  cols = ['scheme', 'window', 'top_k', 'n_w', 'fold_obj', 'total_ret',
          'sharpe', 'max_dd', 'ann_turnover', 'n_days']
  print(df[cols].to_string(index=False, float_format=lambda x: f'{x:g}'))


def _verdict(table: pd.DataFrame) -> pd.DataFrame:
  """逐池判定: (a) new 的折外 sharpe 不劣于 preset/cur_ic; (b) new FULL 超 buyhold."""
  out = []
  for pool, g in table.groupby('pool', sort=False):
    piv = g.pivot_table(index='scheme', columns='window', values='sharpe', aggfunc='first')
    ret = g.pivot_table(index='scheme', columns='window', values='total_ret', aggfunc='first')
    fo = g.groupby('scheme')['fold_obj'].first()
    if 'new' not in piv.index:
      out.append({'pool': pool, 'note': 'new 搜索为空, 跳过'})
      continue
    a_ok = (fo.get('new', -np.inf) >= fo.get('preset', -np.inf) and
            fo.get('new', -np.inf) >= fo.get('cur_ic', -np.inf))
    b_sh = piv.loc['new', 'full'] > piv.loc['buyhold', 'full']
    b_ret = ret.loc['new', 'full'] > ret.loc['buyhold', 'full']
    out.append({
        'pool': pool,
        'foldobj_new': round(float(fo.get('new', np.nan)), 3),
        'foldobj_preset': round(float(fo.get('preset', np.nan)), 3),
        'foldobj_cur_ic': round(float(fo.get('cur_ic', np.nan)), 3),
        'is_sh_new': round(float(piv.loc['new', 'is']), 3),
        'oos_sh_new': round(float(piv.loc['new', 'oos']), 3),
        'is_oos_gap_new': round(float(piv.loc['new', 'is'] - piv.loc['new', 'oos']), 3),
        'full_sh_new': round(float(piv.loc['new', 'full']), 3),
        'full_sh_bh': round(float(piv.loc['buyhold', 'full']), 3),
        '(a)折外不劣': 'YES' if a_ok else 'NO',
        '(b)超buyhold': 'YES' if (b_sh and b_ret) else ('part' if (b_sh or b_ret) else 'NO'),
    })
  return pd.DataFrame(out)


def main():
  ap = argparse.ArgumentParser(description='Step1 临时脚本: WF 折外判据的组合搜索(三方对照)')
  ap.add_argument('--pools', default=','.join(repro.PRESETS.keys()), help='逗号分隔池名')
  ap.add_argument('--signed', action='store_true', default=True,
                  help='允许负权重(默认开); 用 --no-signed 关闭')
  ap.add_argument('--no-signed', dest='signed', action='store_false')
  ap.add_argument('--max-n', type=int, default=4, help='最多入选成分数')
  ap.add_argument('--min-gain', type=float, default=0.05, help='折外 sharpe 增量下限')
  ap.add_argument('--folds', type=int, default=N_FOLDS, help='IS 期折数')
  ap.add_argument('--exclude', default=None, help='剔除标的(逗号分隔)')
  ap.add_argument('--out', default=None, help='结果 CSV 输出路径')
  a = ap.parse_args()

  pools = [s.strip() for s in a.pools.split(',') if s.strip()]
  exclude = [s.strip() for s in a.exclude.split(',')] if a.exclude else None
  log = cfg.get_logger('wfsearch')

  frames = []
  for pool in pools:
    if pool not in repro.PRESETS:
      log.warning(f'跳过未知池: {pool}')
      continue
    rows, detail, actual = run_pool(pool, a.signed, a.max_n, a.min_gain, exclude, a.folds)
    _print_pool(detail, actual, rows)
    frames.append(pd.DataFrame(rows))

  table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
  if a.out and not table.empty:
    table.to_csv(a.out, index=False, encoding='utf-8-sig')
    log.info(f'结果已写入: {a.out}')

  print('\n===== 判定小结(逐池) =====')
  v = _verdict(table)
  print(v.to_string(index=False, float_format=lambda x: f'{x:g}'))


if __name__ == '__main__':
  main()

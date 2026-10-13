# -*- coding: utf-8 -*-
"""
factor.wf_strat_search — 4 池「有效权重 × 有效策略」研究搜索
================================================================
目标: 针对 etf_3x / company_300 / hs300 / a_etf_all 四个池, 在**生产 quant.factor
底座 + 70 因子注册表**上, 回答两个问题:
  (1) 哪种"权重方案"在这些池上真正有效?
  (2) 配什么"策略参数"(top_k / exit_rank / sizing)才有效?

红线(与 factor_config.json / 既有 wf_* 研究一致):
  - 所有权重 / 参数 **只在 IS 期(2021-01-01 ~ 2024-12-31)估计与选择**;
  - OOS(2025)/RECENT(2026)/FULL 只做事后汇报, 绝不参与定参;
  - 不做空不加杠杆(引擎敞口恒 <= 1); 缺失填补/标准化统计量只在 fit 样本上估.

设计(两阶段, 逐池):
  Stage 1 — 权重方案对照(参考参数): 在同一候选集(IS 去重保留)上比较 7 种定权:
    equal / dir / ic(shrink) / inv_vol / rp / greedy / cd_ic(=以 ic 为初值的坐标下降,
    目标=IS 折外 sharpe). 附 buyhold 作参照. 用**IS 折外 sharpe**(fold_obj)选出该池
    最优方案 —— 这是唯一合法的选择依据.
  Stage 2 — 策略参数扫描: 对 Stage 1 选出的方案, 扫 top_k × exit_rank(=top_k+buf),
    仍以 IS 折外 sharpe 选参; 再对参考参数做 sizing(equal/tier/linear)小对照.

输出: 逐行结果 CSV + 控制台小结(逐池: 方案透视 / 参数透视 / 推荐配置).

自检:
  cd ~/git && python -u -m quant.factor.wf_strat_search --pools etf_3x
  cd ~/git && python -u -m quant.factor.wf_strat_search --out out.csv
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor import evaluate as ev
from quant.factor.data import load_pool, window as _win

# ---- 口径 ----
# 修正(2026-09): 原来这里**硬编码**了第三套窗口(2021-2024 / 2025 / 2026+), 与
# factor/config.py 的全局默认(2011-2019 / 2020-2023)以及各池 factor_config.json 的
# meta 都不一致 —— 于是"按哪套窗口选参"取决于跑的是哪个脚本, 结论不可比.
# 现在统一由 cfg.resolve_windows(pool) 提供(优先该池 config 的 meta, 否则全局默认).
IS_START, IS_END = cfg.TRADE_START, cfg.IS_END         # 占位默认; 实际用 pool_windows(pool)
OOS_START, OOS_END = cfg.OOS_START, getattr(cfg, 'OOS_END', None)
RECENT_START = getattr(cfg, 'RECENT_START', None)


def pool_windows(pool: str) -> list:
  """
  取某池的窗口定义 -> [('is', s, e), ('oos', s, e), ('recent', s, e)?].

  单一真源: cfg.resolve_windows(pool)(优先 factor_config.json 该池的 meta.is_window /
  oos_window, 缺失则回退全局默认). 所有 wf_* 脚本都应走这里, 不要再各自硬编码.

  **recent 可能不存在**: 若该池 OOS 是开放式的(如 etf_3x 的 2025-01-01~None),
  OOS 已覆盖到数据末尾 -> 不返回 recent 项(而不是返回 (None,None) 被下游当成全窗).

  :param pool: 池名
  :returns: 窗口列表(元素为 (name, start, end)); recent 的 end 通常为 None(至数据末)
  :raises: none
  """
  w = cfg.resolve_windows(pool)
  out = [('is', w['is'][0], w['is'][1]),
         ('oos', w['oos'][0], w['oos'][1])]
  # recent 只在**该池确实有独立 recent 窗**时才加:
  #   若 OOS 是开放式的(如 etf_3x 的 2025-01-01~None), cfg 会给出空区间哨兵;
  #   旧实现直接透传 (None, None), 下游 _win(df, None, None) 会把它当成**不限**= 全窗,
  #   于是"recent"那一段实际算的是全历史(2026-09-30 修复)。
  rs, re_ = w['recent']
  if rs not in (None, '9999-12-31'):
    out.append(('recent', rs, re_))
  return out


# 兼容旧引用: 保留模块级 WINDOWS 作为"未指定池时"的默认(仍来自全局口径, 不再硬编码)
WINDOWS = [('is', IS_START, IS_END),
           ('oos', OOS_START, OOS_END),
           ('recent', RECENT_START, None)]

N_FOLDS = 4                                            # IS 内折数(折外目标)
SHRINK = 0.25                                          # ic 加权收缩
THRESH = 0.7                                           # 冗余去重阈值
TOPK_GRID = (5, 8, 12, 20)                             # 策略参数网格
EXIT_BUFS = (4, 8)                                     # exit_rank = top_k + buf
SIZINGS = ('equal', 'tier', 'linear')
# 参考参数(Step2 legacy 口径): 只用于 Stage1 隔离"权重方案"差异
REF = {'etf_3x': (5, 12), 'company_300': (12, 16), 'hs300': (5, 12), 'a_etf_all': (5, 12)}
SCHEME_ORDER = ('equal', 'dir', 'ic', 'inv_vol', 'rp', 'greedy', 'cd_ic')
STAT_COLS = ['n_days', 'total_ret', 'sharpe', 'max_dd', 'vol', 'ann_turnover',
             'n_trades', 'win_rate']


def _wstr(w: dict) -> str:
  return ';'.join(f'{n}:{w[n]:g}' for n in sorted(w))


def _row(pool, stage, scheme, window, top_k, exit_rank, n_w, fold_obj, w, st) -> dict:
  r = {'pool': pool, 'stage': stage, 'scheme': scheme, 'window': window,
       'top_k': top_k, 'exit_rank': exit_rank, 'n_w': n_w,
       'fold_obj': round(float(fold_obj), 4) if pd.notna(fold_obj) else np.nan,
       'weights': _wstr(w)}
  r.update({k: st.get(k) for k in STAT_COLS})
  return r


def _stats(open_w, close_w, comp, p, ws, we) -> dict:
  ow, cw = _win(open_w, ws, we), _win(close_w, ws, we)
  res = bt.run_engine(ow, cw, _win(comp, ws, we), p)
  return bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])


# ---------------------------- Stage 1: 权重方案 ----------------------------

def fit_schemes(sigs, fwd_is, cands, scores, ref_p, folds, open_w, close_w,
                args, log) -> dict:
  """在候选集 cands 上拟合 7 种权重方案(全部只用 IS); 拟合失败/为空的直接丢弃."""
  out = {'equal': cmb.equal_weights(cands), 'dir': cmb.dir_equal_weights(cands)}
  trials = [
      ('ic', lambda: cmb.ic_weights(sigs, fwd_is, cands, shrink=args.shrink)),
      ('inv_vol', lambda: cmb.inv_vol_weights(sigs, fwd_is, cands)),
      ('rp', lambda: cmb.risk_parity_weights(sigs, fwd_is, cands)),
  ]
  for name, fn in trials:
    try:
      w = fn()
      if w:
        out[name] = w
    except ValueError as e:
      log.warning('方案 %s 定权失败: %s', name, e)
  try:
    chosen, _ = cmb.greedy_forward(sigs, fwd_is, cands, max_n=args.greedy_n)
    if chosen:
      out['greedy'] = cmb.equal_weights(chosen)
  except ValueError as e:
    log.warning('greedy 定权失败: %s', e)
  # cd_ic: 只取 |IC| 最强的前 cd_n 个候选, 控制坐标下降的引擎评估次数
  top = sorted([n for n in cands if pd.notna(scores.get(n))],
               key=lambda n: -abs(float(scores[n])))[:args.cd_n]
  if top:
    try:
      w0 = cmb.ic_weights(sigs, fwd_is, top, shrink=args.shrink)
      obj = bt.cached_fold_objective(open_w, close_w, ref_p, folds, metric='sharpe')
      w_cd, _ = cmb.coord_descent_weights(sigs, w0, obj, min_cs=cfg.MIN_CS)
      if w_cd:
        out['cd_ic'] = w_cd
    except ValueError as e:
      log.warning('cd_ic 定权失败: %s', e)
  return {k: out[k] for k in SCHEME_ORDER if k in out}


def _pick_best(rows, scheme_names, stage) -> str:
  """在给定 stage 的行里, 按 IS 折外 sharpe(fold_obj) 最大的方案/配置定序."""
  best, best_v = None, -np.inf
  for r in rows:
    if r['stage'] != stage or r['scheme'] not in scheme_names:
      continue
    v = r['fold_obj']
    if pd.notna(v) and v > best_v:
      best, best_v = r['scheme'], v
  return best


# ---------------------------- 单池全流程 ----------------------------

def run_pool(pool: str, args) -> tuple:
  log = cfg.get_logger('wfstrat')
  ds = load_pool(pool)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  # 窗口单一真源: 该池 config 的 meta 优先, 否则全局默认(不再用本脚本硬编码值)
  windows = pool_windows(pool)
  is_s, is_e = windows[0][1], windows[0][2]
  sigs = cmb.build_signals(ds)                       # 70 因子, 全史计算 -> 评估时截窗
  fwd = ev.forward_return(ds, h=1)
  fwd_is = _win(fwd, is_s, is_e)
  scores = cmb.rank_ic_scores(sigs, fwd_is, min_cs=cfg.MIN_CS)
  kept, dropped = cmb.dedupe(sigs, scores, thresh=args.thresh, min_cs=cfg.MIN_CS)
  cands = kept or list(sigs.keys())
  folds = cmb.split_folds(ds.dates, is_s, is_e, args.folds)
  ref_k, ref_e = REF.get(pool, (cfg.TOP_K, cfg.EXIT_RANK))
  ref_p = bt.EngineParams(top_k=ref_k, exit_rank=ref_e)
  log.info('%s: 标的 %d, 数据 %s~%s, IS %s~%s, 去重保留 %d/%d(丢 %d), 参考参数 top_k=%d exit=%d',
           pool, len(ds.symbols), f'{ds.dates.min():%Y-%m-%d}',
           f'{ds.dates.max():%Y-%m-%d}', is_s, is_e, len(cands), len(sigs), len(dropped),
           ref_k, ref_e)

  schemes = fit_schemes(sigs, fwd_is, cands, scores, ref_p, folds,
                        open_w, close_w, args, log)

  # --- Stage 1: 逐方案逐窗口(参考参数) ---
  rows = []
  fit_obj = bt.cached_fold_objective(open_w, close_w, ref_p, folds, metric='sharpe')
  for name in SCHEME_ORDER:
    w = schemes.get(name)
    if not w:
      continue
    comp = cmb.combine_signals(sigs, w, min_cs=cfg.MIN_CS)
    fo = fit_obj(w, comp)
    for win, ws, we in windows:
      rows.append(_row(pool, 'scheme', name, win, ref_k, ref_e, len(w), fo, w,
                       _stats(open_w, close_w, comp, ref_p, ws, we)))
  for win, ws, we in windows:
    bh = bt.buyhold(_win(open_w, ws, we), _win(close_w, ws, we))
    st = bt.perf_stats(bh['equity'], bh['ann_turnover'], bh['trades'])
    rows.append(_row(pool, 'scheme', 'buyhold', win, np.nan, np.nan, np.nan,
                     np.nan, {}, st))

  # --- Stage 2: 最佳方案 x 策略参数 ---
  best = _pick_best(rows, schemes.keys(), 'scheme')
  if best:
    w_best = schemes[best]
    comp = cmb.combine_signals(sigs, w_best, min_cs=cfg.MIN_CS)
    for tk in TOPK_GRID:
      if tk > max(2, len(ds.symbols) // 2):          # 池太小: 跳过过大的 top_k
        continue
      for buf in EXIT_BUFS:
        e = tk + buf
        p = ref_p.copy(top_k=tk, exit_rank=e)
        obj = bt.cached_fold_objective(open_w, close_w, p, folds, metric='sharpe')
        fo = obj(w_best, comp)
        for win, ws, we in windows:
          rows.append(_row(pool, 'param', best, win, tk, e, len(w_best), fo, w_best,
                           _stats(open_w, close_w, comp, p, ws, we)))
    for sz in SIZINGS:                               # sizing 小对照(参考参数)
      p = ref_p.copy(sizing=sz)
      for win, ws, we in windows:
        rows.append(_row(pool, 'sizing', f'{best}/{sz}', win, ref_k, ref_e,
                         len(w_best), np.nan, w_best,
                         _stats(open_w, close_w, comp, p, ws, we)))

  detail = {'pool': pool, 'n_sym': len(ds.symbols), 'kept': cands, 'dropped': dropped,
            'schemes': {k: _wstr(v) for k, v in schemes.items()},
            'best': best, 'ref': (ref_k, ref_e),
            'scores': {k: (round(float(scores[k]), 4) if pd.notna(scores[k]) else None)
                       for k in cands}}
  return rows, detail


# ---------------------------- 输出 ----------------------------

def _param_label(g: pd.DataFrame) -> pd.DataFrame:
  """给参数行加一列 'param'(= 'top_k/exit_rank'), 供透视与分组用同一把键."""
  g = g.copy()
  g['param'] = (g['top_k'].astype('Int64').astype(str) + '/'
                + g['exit_rank'].astype('Int64').astype(str))
  return g


def _pivot(table, stage, col, value, scheme=None):
  g = table[table['stage'] == stage]
  if scheme is not None:
    g = g[g['scheme'] == scheme]
  if col == 'param':
    g = _param_label(g)
    idx = 'param'
  else:
    idx = 'scheme'
  return g.pivot_table(index=idx, columns='window', values=value, aggfunc='first')


def _print_pool(table, detail):
  pool = detail['pool']
  print(f"\n================= {pool} (标的 {detail['n_sym']}, "
        f"参考 {detail['ref'][0]}/{detail['ref'][1]}) =================")
  print(f"  IS 候选(去重保留 {len(detail['kept'])}): {detail['kept']}")
  if detail['dropped']:
    d = detail['dropped']
    print(f"  IS 去重丢弃 {len(d)}: " +
          ', '.join(f"{x['factor']}({x['rho']:+g}~{x['conflict_with']})" for x in d[:12]))
  for k in SCHEME_ORDER:
    if k in detail['schemes']:
      print(f"  [{k:8s}] {detail['schemes'][k]}")

  sh = _pivot(table, 'scheme', 'scheme', 'sharpe')
  ret = _pivot(table, 'scheme', 'scheme', 'total_ret')
  fo = table[table['stage'] == 'scheme'].groupby('scheme')['fold_obj'].first()
  order = [s for s in list(SCHEME_ORDER) + ['buyhold'] if s in sh.index]
  sh = sh.reindex(order)
  ret = ret.reindex(order)
  show = pd.concat([fo.reindex(order).rename('fold_obj'),
                    sh.add_prefix('sh_'), ret.add_prefix('ret_')], axis=1)
  print('\n  -- Stage1 权重方案(sharpe / total_ret; fold_obj=IS折外sharpe 为唯一选择依据) --')
  print(show.to_string(float_format=lambda x: f'{x:g}'))

  gp = _param_label(table[table['stage'] == 'param'])
  psh = gp.pivot_table(index='param', columns='window', values='sharpe', aggfunc='first')
  pret = gp.pivot_table(index='param', columns='window', values='total_ret', aggfunc='first')
  pfo = gp.groupby('param')['fold_obj'].first()
  pshow = pd.concat([pfo.rename('fold_obj'),
                     psh.add_prefix('sh_'), pret.add_prefix('ret_')], axis=1)
  pshow = pshow.sort_values('fold_obj', ascending=False)
  print(f"\n  -- Stage2 策略参数({detail['best']} 方案; 按 fold_obj 降序) --")
  print(pshow.to_string(float_format=lambda x: f'{x:g}'))

  ssz = _pivot(table, 'sizing', 'scheme', 'sharpe')
  rsz = _pivot(table, 'sizing', 'scheme', 'total_ret')
  print('\n  -- sizing 小对照(参考参数) --')
  print(pd.concat([ssz.add_prefix('sh_'), rsz.add_prefix('ret_')], axis=1)
        .to_string(float_format=lambda x: f'{x:g}'))


def _recommend(table) -> pd.DataFrame:
  """逐池推荐: Stage1 最优方案 + Stage2 最优(top_k/exit_rank), 附 IS/OOS/RECENT."""
  out = []
  for pool, g in table.groupby('pool', sort=False):
    gs = g[g['stage'] == 'scheme']
    gp = g[g['stage'] == 'param']
    best_scheme = (gs.groupby('scheme')['fold_obj'].first()
                   .drop(labels=['buyhold'], errors='ignore').idxmax())
    if gp.empty:
      out.append({'pool': pool, 'scheme': best_scheme, 'top_k': np.nan,
                  'exit_rank': np.nan, 'fold_obj': np.nan})
      continue
    b = gp[gp['window'] == 'is'].copy()
    b = b.sort_values('fold_obj', ascending=False).iloc[0]
    psh = gp[gp['top_k'].eq(b['top_k']) & gp['exit_rank'].eq(b['exit_rank'])]
    sh = psh.set_index('window')['sharpe']
    rt = psh.set_index('window')['total_ret']

    def _g(s, k):
      return round(float(s.get(k, np.nan)), 3) if k in s else np.nan
    out.append({'pool': pool, 'scheme': best_scheme, 'top_k': int(b['top_k']),
                'exit_rank': int(b['exit_rank']), 'fold_obj': round(float(b['fold_obj']), 3),
                'sh_is': _g(sh, 'is'), 'sh_oos': _g(sh, 'oos'), 'sh_recent': _g(sh, 'recent'),
                'ret_is': _g(rt, 'is'), 'ret_oos': _g(rt, 'oos'), 'ret_recent': _g(rt, 'recent')})
  return pd.DataFrame(out)


def main():
  ap = argparse.ArgumentParser(description='4 池「有效权重 × 有效策略」研究搜索')
  ap.add_argument('--pools', default='etf_3x,company_300,hs300,a_etf_all', help='逗号分隔池名')
  ap.add_argument('--thresh', type=float, default=THRESH, help='冗余去重阈值')
  ap.add_argument('--shrink', type=float, default=SHRINK, help='ic 加权收缩比例')
  ap.add_argument('--folds', type=int, default=N_FOLDS, help='IS 折外目标折数')
  ap.add_argument('--greedy-n', type=int, default=8, help='greedy 最多入选因子数')
  ap.add_argument('--cd-n', type=int, default=10, help='cd_ic 坐标下降的候选上限')
  ap.add_argument('--out', default=str(cfg.OUT_DIR / 'wf_strat_search.csv'),
                  help='结果 CSV 输出路径')
  a = ap.parse_args()

  pools = [s.strip() for s in a.pools.split(',') if s.strip()]
  log = cfg.get_logger('wfstrat')
  frames, details = [], []
  for pool in pools:
    rows, detail = run_pool(pool, a)
    _print_pool(pd.DataFrame(rows), detail)
    frames.append(pd.DataFrame(rows))
    details.append(detail)

  table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
  if a.out and not table.empty:
    table.to_csv(a.out, index=False, encoding='utf-8-sig')
    log.info('结果已写入: %s', a.out)

  if not table.empty:
    print('\n===== 逐池推荐(方案按 IS 折外 sharpe 选, 参数按 IS 折外 sharpe 选) =====')
    print(_recommend(table).to_string(index=False, float_format=lambda x: f'{x:g}'))


if __name__ == '__main__':
  main()
# -*- coding: utf-8 -*-
"""
factor.wf_step2_search — Step2 **临时脚本**: 权重方案扩展对照(风险类加权 / 坐标下降)
================================================================================
背景: Step1 把"接受判据"从 IS sharpe 换成 WF 折外 sharpe, 结论是"只换判据不够" —— 新搜
  索器常塌缩到 1~2 个因子, 且在最重要的 company_300 上 IS->OOS 缺口反而更差. Step2 换思路:
  **不动"选谁", 只换"怎么定权"**, 在同一批候选因子上比较几种更抗过拟合的权重方案.

本脚本对照(同信号、同候选、同 top_k, 仅"权重来源"不同):
  - preset  : 旧系统预设权重 @ 预设 top_k(外部基准, 固定不动).
  - equal   : 候选因子等权 @ 预设 top_k(参照).
  - cur_ic  : 当前系统 ic 加权(shrink=0.5)@ 预设 top_k.
  - inv_vol : 逆波动率加权(combine.inv_vol_weights).
  - rp      : 风险平价/ERC(combine.risk_parity_weights).
  - ic_iv   : ic 权重向 inv_vol 收缩(shrink=0.5, combine.ic_weights(target=...)).
  - cd_ic   : 以 cur_ic 为初值做连续坐标下降(combine.coord_descent_weights, 目标=折外 sharpe).
  另附 buyhold 作参照.

关键设计:
  1) **top_k 固定在预设值** —— 本步只隔离"权重方案"的差异, 不再联合搜 top_k(那是 Step1).
  2) 所有"需拟合"的方案(cur_ic/inv_vol/rp/ic_iv/cd_ic)统一在**同一候选集** `cands` 上定权
     (cands = IS 冗余去重后的保留因子), 保证可比. preset 是固定外部基准, 可含候选集外因子.
  3) 定权只用 IS(cfg.TRADE_START~cfg.IS_END); OOS/FULL/ytd26 仅事后汇报(红线).

判定(逐池): (a) 折外 sharpe 不低于 preset/cur_ic; (b) FULL 窗对 buyhold 有超额;
  并重点看 **gap_is_oos = is_sharpe - oos_sharpe**(越小越稳).

这是**临时脚本**: 只放一次性的编排与报表; 通用能力(inv_vol/rp/coord_descent/ic 收缩目标)
  已落在 combine.py.

自检:
  cd ~/git && python -u -m quant.factor.wf_step2_search --pools etf_3x
  cd ~/git && python -u -m quant.factor.wf_step2_search --out out.csv
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

N_FOLDS = 4                                                  # IS 期内部折数
SCHEMES = ('preset', 'equal', 'cur_ic', 'inv_vol', 'rp', 'ic_iv', 'cd_ic')
STAT_COLS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'vol',
             'ann_turnover', 'n_trades', 'win_rate', 'avg_days', 'n_days']


# ---------------------------- 权重方案 ----------------------------

def _wstr(w: dict) -> str:
  return ';'.join(f'{n}:{w[n]:g}' for n in w)


def _ic_weights(sigs, fwd_is, names, shrink: float) -> dict:
  """
  ic 加权(shrink 收缩). 在临时脚本内实现收缩而非直接调 cmb.ic_weights(shrink>0),
  原因是本脚本信号键名沿用旧预设名(F_er20 ...), 不在 factor 注册表里, 而 ic_weights 的
  默认收缩目标 dir_equal_weights 需按注册表查 direction. 这批信号先验方向全为 +1(N_* 已取负),
  故 dir_equal == equal, 收缩目标即等权 1/N —— 与 cmb.ic_weights(shrink) 在本信号集上等价.
  """
  w = cmb.ic_weights(sigs, fwd_is, names=names, shrink=0.0)
  n = len(w)
  if n == 0:
    return {}
  eq = 1.0 / n
  return {k: round((1.0 - shrink) * v + shrink * eq, 6) for k, v in w.items()}


def _safe(fn, *args, **kw) -> dict:
  try:
    return fn(*args, **kw) or {}
  except ValueError:
    return {}


def build_schemes(sigs, fwd_is, cands, preset_w, shrink: float,
                  obj) -> dict:
  """构造 {方案名: 权重}(全部只用 IS 定权); 拟合失败/为空的方案直接丢弃."""
  out = {'preset': dict(preset_w), 'equal': cmb.equal_weights(cands)}
  w_ic = _ic_weights(sigs, fwd_is, cands, shrink)
  if w_ic:
    out['cur_ic'] = w_ic
  w_iv = _safe(cmb.inv_vol_weights, sigs, fwd_is, cands, cfg.MIN_CS)
  if w_iv:
    out['inv_vol'] = w_iv
  w_rp = _safe(cmb.risk_parity_weights, sigs, fwd_is, cands, cfg.MIN_CS)
  if w_rp:
    out['rp'] = w_rp
  if w_iv:                                                   # ic 向 inv_vol 收缩
    w_iiv = _safe(cmb.ic_weights, sigs, fwd_is, names=cands, shrink=shrink,
                  target=w_iv)
    if w_iiv:
      out['ic_iv'] = w_iiv
  if w_ic:                                                   # 以 cur_ic 为初值坐标下降
    w_cd, _ = cmb.coord_descent_weights(sigs, w_ic, obj, min_cs=cfg.MIN_CS)
    if w_cd:
      out['cd_ic'] = w_cd
  return out


# ---------------------------- 单池全流程 ----------------------------

def run_pool(pool: str, shrink: float, exclude=None, n_folds: int = N_FOLDS) -> tuple:
  """跑一个池: 各权重方案 x 4 窗口; 返回 (结果行, 搜索详情, 区间描述)."""
  ds = load_pool(pool, exclude=exclude)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  sigs = repro.native_signals(ds)                            # 10 个信号(键名对齐旧预设)
  preset_w, preset_k = repro.PRESETS[pool]
  windows = repro.WINDOWS
  folds = cmb.split_folds(ds.dates, cfg.TRADE_START, cfg.IS_END, n_folds)
  p = repro.engine_params(preset_k)                          # top_k 固定在预设值

  # --- 只依赖 IS 的定权材料 ---
  fwd = ev.forward_return(ds, h=1)
  fwd_is = bt._win(fwd, cfg.TRADE_START, cfg.IS_END)
  scores = cmb.rank_ic_scores(sigs, fwd_is, min_cs=cfg.MIN_CS)
  kept, dropped = cmb.dedupe(sigs, scores, thresh=cmb.DEDUPE_RHO, min_cs=cfg.MIN_CS)
  cands = kept or list(sigs.keys())

  # --- 折外目标(带缓存; cd_ic 用它当接受判据) ---
  cache = {}

  def obj(weights, comp):
    key = tuple(sorted((n, round(float(w), 6)) for n, w in weights.items()))
    if key not in cache:
      cache[key] = bt.make_fold_objective(open_w, close_w, p, folds,
                                          metric='sharpe')(weights, comp)
    return cache[key]

  schemes = build_schemes(sigs, fwd_is, cands, preset_w, shrink, obj)

  # --- 逐方案逐窗口回测 ---
  fold_obj = bt.make_fold_objective(open_w, close_w, p, folds, metric='sharpe')
  rows = []
  for name, w in schemes.items():
    comp = cmb.combine_signals(sigs, w, min_cs=cfg.MIN_CS)
    fobj = fold_obj(w, comp)
    for win, ws, we in windows:
      ow, cw = bt._win(open_w, ws, we), bt._win(close_w, ws, we)
      res = bt.run_engine(ow, cw, bt._win(comp, ws, we), p)
      s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      rows.append({'pool': pool, 'scheme': name, 'window': win, 'top_k': preset_k,
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
            'kept': kept, 'dropped': dropped, 'cands': cands,
            'weights': {k: v for k, v in schemes.items()},
            'folds': [(f'{a:%Y-%m-%d}', f'{b:%Y-%m-%d}') for a, b in folds]}
  actual = f'{ds.dates.min():%Y-%m-%d}~{ds.dates.max():%Y-%m-%d}'
  return rows, detail, actual


# ---------------------------- 输出 ----------------------------

def _print_pool(detail: dict, actual: str, rows: list):
  pool = detail['pool']
  print(f"\n===== {pool}  数据 {actual}  top_k={detail['preset_k']} =====")
  print(f"  预设权重: {detail['preset_w']}")
  print(f"  IS 候选(去重保留): {detail['kept']}")
  if detail['dropped']:
    print(f"  IS 去重丢弃: {detail['dropped']}")
  for k, w in detail['weights'].items():
    if k != 'preset':
      print(f"  [{k:8s}] {_wstr(w)}")
  df = pd.DataFrame(rows)
  cols = ['scheme', 'window', 'n_w', 'fold_obj', 'total_ret',
          'sharpe', 'max_dd', 'ann_turnover', 'n_days']
  print(df[cols].to_string(index=False, float_format=lambda x: f'{x:g}'))


def _verdict(table: pd.DataFrame) -> pd.DataFrame:
  """逐池: 各方案的折外目标 + IS/OOS 缺口 + FULL 是否超 buyhold."""
  out = []
  for pool, g in table.groupby('pool', sort=False):
    piv = g.pivot_table(index='scheme', columns='window', values='sharpe', aggfunc='first')
    ret = g.pivot_table(index='scheme', columns='window', values='total_ret', aggfunc='first')
    fo = g.groupby('scheme')['fold_obj'].first()
    base = max(fo.get('preset', -np.inf), fo.get('cur_ic', -np.inf))
    for name in [s for s in SCHEMES if s in piv.index]:
      a_ok = fo.get(name, -np.inf) >= base
      b_sh = piv.loc[name, 'full'] > piv.loc['buyhold', 'full']
      b_ret = ret.loc[name, 'full'] > ret.loc['buyhold', 'full']
      out.append({
          'pool': pool, 'scheme': name,
          'fold_obj': round(float(fo.get(name, np.nan)), 3),
          'sh_is': round(float(piv.loc[name, 'is']), 3),
          'sh_oos': round(float(piv.loc[name, 'oos']), 3),
          'gap_is_oos': round(float(piv.loc[name, 'is'] - piv.loc[name, 'oos']), 3),
          'sh_full': round(float(piv.loc[name, 'full']), 3),
          'full_ret': round(float(ret.loc[name, 'full']), 2),
          '(a)折外不劣': 'YES' if a_ok else 'NO',
          '(b)超buyhold': 'YES' if (b_sh and b_ret) else ('part' if (b_sh or b_ret) else 'NO'),
      })
  return pd.DataFrame(out)


def main():
  ap = argparse.ArgumentParser(description='Step2 临时脚本: 权重方案扩展对照')
  ap.add_argument('--pools', default=','.join(repro.PRESETS.keys()), help='逗号分隔池名')
  ap.add_argument('--shrink', type=float, default=0.5,
                  help='ic 收缩比例(固定 0.5: 对齐 Step2 文档口径, 不随 IC_SHRINK 漂移)')
  ap.add_argument('--folds', type=int, default=N_FOLDS, help='IS 期折数')
  ap.add_argument('--exclude', default=None, help='剔除标的(逗号分隔)')
  ap.add_argument('--out', default=None, help='结果 CSV 输出路径')
  a = ap.parse_args()

  pools = [s.strip() for s in a.pools.split(',') if s.strip()]
  exclude = [s.strip() for s in a.exclude.split(',')] if a.exclude else None
  log = cfg.get_logger('wfstep2')

  frames = []
  for pool in pools:
    if pool not in repro.PRESETS:
      log.warning(f'跳过未知池: {pool}')
      continue
    rows, detail, actual = run_pool(pool, a.shrink, exclude, a.folds)
    _print_pool(detail, actual, rows)
    frames.append(pd.DataFrame(rows))

  table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
  if a.out and not table.empty:
    table.to_csv(a.out, index=False, encoding='utf-8-sig')
    log.info(f'结果已写入: {a.out}')

  print('\n===== 判定小结(逐池逐方案) =====')
  v = _verdict(table)
  print(v.to_string(index=False, float_format=lambda x: f'{x:g}'))

  if not v.empty:
    print('\n===== gap_is_oos 透视(行=池, 列=方案; 越小越稳) =====')
    g = v.pivot(index='pool', columns='scheme', values='gap_is_oos')
    print(g.to_string(float_format=lambda x: f'{x:g}'))


if __name__ == '__main__':
  main()

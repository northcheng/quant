# -*- coding: utf-8 -*-
"""
combo_search.py — B3 组合层优化: 冗余检查 + 权重/门槛搜索 + IS/OOS 分窗
====================================================================

背景(research_summary_20260920.md 附录 B3): 信号层单因子挖掘已收敛(6 个可上线
信号), 下一层在 score_backtest 引擎上直接搜「权重组合 + 门槛」, 而非继续单因子
排序 top-k. 本脚本三步:

  1. 冗余检查(自实现, g_redundancy_check.py 实际不存在于 research/):
     6 可上线信号 + G_vr20 过滤器两两「日截面 Spearman」(逐日 rank 后按行 Pearson,
     截面有效数 >= MIN_CS 才计). |rho_mean| >= min_rho 判高冗余 → 组合搜索里
     二选一(留 IS 单信号更强者; 单信号评估不受影响, 全部照跑).
     重点对: (K_tmqr, C_tmqmom) —— 同素材(mom121×er20)异构造(秩乘积 vs 秩加法).
  2. 权重搜索(IS 窗 start~is_end 选参):
     a. 单信号基线(7 候选) + gold4 参照组合(上轮挖矿最优先例);
     b. 贪心前向: 从 IS 最强可上线单信号出发, 每步在「剩余候选 × 权重网格」
        (普通 {0.5, 1.0}; G_vr20 允许 ±{0.5, 1.0})里选 IS sharpe 提升最大者
        加入(提升须 > margin 防噪声), 最多 max_add 步;
     c. 精修一遍: 对已选成分试 ×0.5 / ×2 / 剔除;
     d. 门槛搜索: 最终组合 top_k ∈ {5, 8, 12}(exit_rank 固定 12).
  3. OOS 窗(oos_start~) 复验 + 全窗终验(2021+ 至今) + shuffle/buyhold 对照
     (shuffle 为窗内置换: 负 sharpe = 分数含真实时序信息).

引擎与口径: bt_core.BacktestKit + score_backtest 引擎层零改动复用;
  mode='rank'(Σ wᵢ×当日截面 rank_pct, NaN→0.5, 即 compute_composite 逐位口径,
  快速路径经 np.allclose 断言与 canonical 一致后才启用); EngineParams 全默认
  (top_k=5, exit_rank=12, tier 1.5/1.0/0.5, 单边 10bps, cap 0.30).
防前视: 信号全历史构建 → 截交易窗口 → s 收盘决策 → d=s+1 开盘成交.

候选公式(逐行核对自源模块):
  H_trendmag_alpha = _alphaize(trend_magnitude)                    [alpha_mining L112]
  H_ichimoku_alpha = _alphaize(ichimoku_distance)                  [alpha_mining L111]
  C_tmqmom         = rk(F_mom121) + rk(F_er20)                     [alpha_mining L162]
  K_tmqr           = rk(F_mom121) * rk(F_er20)                     [conditional_mining L183]
  F_er20/F_mom121/F_idiovol60/F_obv20 = build_mined(panel)         [factor_mining]
  D_mom250         = close.pct_change(250)                         [signal_search L56]
  G_vr20           = var(pct_change(20)) / (20*var(ret1))          [factor_mining2 L140-144]
  gold4 参照        = {F_mom121:.5, F_er20:.3, F_idiovol60:-.2, F_obv20:.1}  [score_backtest GOLD_SETS]

输出: research/output/combo_{run_id}/
  summary.csv(全部配置×窗口)  redundancy_{pool}.csv  greedy_{pool}.csv(试验史)
  {pool}_final/(胜出配置 3件套)  report.txt
"""
import argparse
import os
import sys
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_mining import build_mined                         # noqa: E402
from alpha_mining import _alphaize                            # noqa: E402
from bt_core import BacktestKit, BacktestResult               # noqa: E402
from score_backtest import (                                  # noqa: E402
    EngineParams, run_config, shuffled_composite)

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))

# ================================================================ 参数 ================================================================ #
CAND_SIGNALS = ['H_trendmag_alpha', 'H_ichimoku_alpha', 'C_tmqmom',
                'K_tmqr', 'F_er20', 'D_mom250']     # 6 个可上线信号(6.1 节)
FILTER_CAND = 'G_vr20'                              # 过滤器候选(允许负权)
RED_NAMES = CAND_SIGNALS + [FILTER_CAND]            # 冗余检查/单信号评估集合(7)
GOLD4 = {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2, 'F_obv20': 0.1}
GOLD4_EXTRA = ['F_mom121', 'F_idiovol60', 'F_obv20']
COMBO_COLS = RED_NAMES + GOLD4_EXTRA                # 需注入 panel 的全部成分列
TOPK_GRID = [5, 8, 12]                             # 门槛(top_k)搜索网格
MIN_CS = 10                                        # 日截面 Spearman 最少有效数
STAT_KEYS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'vol',
             'ann_turnover', 'n_trades', 'win_rate', 'avg_ret', 'avg_days', 'n_days']


def log(msg: str):
    print(msg, flush=True)


def _sh(stats: dict) -> float:
    """sharpe 取值, NaN/缺失 → -1e9(比较安全)."""
    v = stats.get('sharpe')
    return -1e9 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def wdesc(weights: dict) -> str:
    """权重描述: 按 |w| 降序, 如 'K_tmqr:1,H_trendmag_alpha:0.5'."""
    items = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
    return ','.join(f'{k}:{v:g}' for k, v in items)


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is None and end is None:
        return df
    if start is None:
        return df.loc[:end]
    if end is None:
        return df.loc[start:]
    return df.loc[start:end]


def make_row(pool: str, window: str, payload, weights: str) -> dict:
    s = {'pool': pool, 'window': window, 'config': payload['name'], 'weights': weights}
    st = payload['stats']
    s.update({k: st.get(k) for k in STAT_KEYS})
    return s


# ================================================================ 候选构建 ================================================================ #
def build_combo_cands(panel: pd.DataFrame) -> dict:
    """B3 全部候选(全历史构建, 因果安全; 公式逐行核对自源模块, 见模块 docstring)."""
    close = panel['Close'].unstack('symbol').sort_index()
    ret1 = close.pct_change()
    mined = build_mined(panel)          # F_ 族底料(整块复用, 无口径漂移)
    rk = lambda v: v.rank(axis=1, pct=True)  # noqa: E731

    def w(col):
        return pd.to_numeric(panel[col], errors='coerce').unstack('symbol').sort_index()

    out = {}
    out['H_trendmag_alpha'] = _alphaize(w('trend_magnitude'))
    out['H_ichimoku_alpha'] = _alphaize(w('ichimoku_distance'))
    out['C_tmqmom'] = rk(mined['F_mom121']) + rk(mined['F_er20'])
    out['K_tmqr'] = rk(mined['F_mom121']) * rk(mined['F_er20'])
    out['F_er20'] = mined['F_er20']
    out['D_mom250'] = close.pct_change(250)
    # G_vr20: 方差比(20), rolling(120, min_periods=60)
    var20 = close.pct_change(20).rolling(120, min_periods=60).var()
    var1 = ret1.rolling(120, min_periods=60).var()
    out['G_vr20'] = var20 / (20.0 * var1.replace(0, np.nan))
    for f in GOLD4_EXTRA:
        out[f] = mined[f]
    return out


# ================================================================ 冗余检查 ================================================================ #
def daily_spearman(ranks_a: pd.DataFrame, ranks_b: pd.DataFrame,
                   min_cs: int) -> pd.Series:
    """逐日截面 Spearman = 对两列已按行 rank 的矩阵做按行 Pearson(成对完备).
    有效数 < min_cs 的日子 → NaN."""
    ok = ranks_a.notna() & ranks_b.notna()
    n = ok.sum(axis=1)
    va, vb = ranks_a.where(ok), ranks_b.where(ok)
    da = va.sub(va.mean(axis=1), axis=0)
    db = vb.sub(vb.mean(axis=1), axis=0)
    cov = da.mul(db).sum(axis=1)
    den = np.sqrt(da.pow(2).sum(axis=1) * db.pow(2).sum(axis=1)).replace(0, np.nan)
    return (cov / den).where(n >= min_cs)


def redundancy_report(cands: dict, names: list, start: str, min_cs: int) -> pd.DataFrame:
    ranks = {n: cands[n].loc[start:].rank(axis=1) for n in names}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = daily_spearman(ranks[a], ranks[b], min_cs).dropna()
            rows.append({
                'pair': f'{a}~{b}', 'days': len(rho),
                'mean': round(float(rho.mean()), 3) if len(rho) else np.nan,
                'median': round(float(rho.median()), 3) if len(rho) else np.nan,
                'q10': round(float(rho.quantile(0.10)), 3) if len(rho) else np.nan,
                'q90': round(float(rho.quantile(0.90)), 3) if len(rho) else np.nan,
            })
    return pd.DataFrame(rows).sort_values('mean', key=lambda s: s.abs(),
                                          ascending=False)


# ================================================================ 回测执行器 ================================================================ #
class PoolRunner:
    """一个池的快速执行器: 预缓存各成分当日截面 rank_pct, 组合分数 = Σ(w/Σ|w|)·pct
    再 fillna(0.5) —— 与 compute_composite 逐位一致(构造时断言校验)."""

    def __init__(self, kit: BacktestKit):
        self.kit = kit
        kit._get_cands()                     # 触发注入(注册的候选列进 panel)
        self._pct = {}
        for col in COMBO_COLS:
            wide = pd.to_numeric(kit.study[col], errors='coerce') \
                         .unstack('symbol').sort_index()
            self._pct[col] = wide.rank(axis=1, pct=True)
        # 口径断言: 快速路径 vs canonical kit._composite(走 compute_composite)
        # 探针含负权, 覆盖 G_vr20 负向路径的符号处理
        probe = {COMBO_COLS[0]: 1.0, COMBO_COLS[3]: -0.7}
        ref = kit._composite(dict(probe), 'rank')
        fast = self.fast_composite(probe).reindex(index=ref.index, columns=ref.columns)
        if not np.allclose(ref.to_numpy(), fast.to_numpy(), equal_nan=True):
            raise RuntimeError('快速组合口径与 compute_composite 不一致, 禁用快速路径')

    def fast_composite(self, weights: dict) -> pd.DataFrame:
        total = float(sum(abs(w) for w in weights.values()))
        if total <= 0:
            raise ValueError('权重绝对值和为 0')
        comp = None
        for col, w in weights.items():
            part = (w / total) * self._pct[col]
            comp = part if comp is None else comp.add(part, fill_value=0.0)
        return comp.fillna(0.5)

    def run_window(self, weights: dict, name: str, start=None, end=None,
                   engine_params: EngineParams = None,
                   shuffle_seed: int = None) -> dict:
        comp = self.fast_composite(weights)
        ow = _slice(self.kit.open_wide, start, end)
        cw = _slice(self.kit.close_wide, start, end)
        comp = comp.reindex(index=ow.index, columns=ow.columns)
        if shuffle_seed is not None:          # 窗内置换(对照与配置同窗, 公平)
            comp = shuffled_composite(comp, shuffle_seed)
        g = pd.DataFrame(True, index=ow.index, columns=ow.columns)
        aw = (_slice(self.kit.atr_wide, start, end)
              if self.kit.atr_wide is not None else None)
        p = engine_params or self.kit.engine_params
        return run_config(name, ow, cw, comp, g, p, atr_wide=aw)

    def buyhold_window(self, name: str = 'buyhold_pool', start=None, end=None) -> dict:
        ow = _slice(self.kit.open_wide, start, end)
        cw = _slice(self.kit.close_wide, start, end)
        comp = pd.DataFrame(0.5, index=ow.index, columns=ow.columns)
        g = pd.DataFrame(True, index=ow.index, columns=ow.columns)
        p = self.kit.engine_params.copy(top_k=9999, exit_rank=99999, sizing='equal')
        aw = (_slice(self.kit.atr_wide, start, end)
              if self.kit.atr_wide is not None else None)
        return run_config(name, ow, cw, comp, g, p, atr_wide=aw)


# ================================================================ 搜索流程 ================================================================ #
def _weight_grid(cand: str) -> list:
    """普通候选正权网格; G_vr20 方向不确定 → 正负都试."""
    return [-1.0, -0.5, 0.5, 1.0] if cand == FILTER_CAND else [0.5, 1.0]


# ================================================================ 逐池主流程 ================================================================ #
def run_pool(pool: str, start: str, is_end: str, oos_start: str, min_rho: float,
             margin: float, max_add: int, shuffle_seed: int, out_dir: str):
    """一个池的完整 B3 流程: 冗余 → 单信号基线 → 贪心 → 精修 → 门槛 → OOS/终验/对照.
    返回 (rows, greedy_log, report_lines, final_info)."""
    log(f'\n{"=" * 70}\n===== pool={pool} =====')
    t0 = time.time()
    kit = BacktestKit(pool=pool, start=None)          # 全历史构建信号, 交易窗口逐次截
    kit.register(build_combo_cands, label='B3候选')
    runner = PoolRunner(kit)
    n_sym = kit.open_wide.shape[1]
    log(f'  数据 {kit.open_wide.index.min():%Y-%m-%d} ~ {kit.open_wide.index.max():%Y-%m-%d}, '
        f'{n_sym} 标的 | 引擎 {kit.engine_params}')
    rows, greedy_log, lines = [], [], []
    lines.append(f'## pool={pool}  ({n_sym} 标的, '
                 f'{kit.open_wide.index.min():%Y-%m-%d}~{kit.open_wide.index.max():%Y-%m-%d})')

    # ---------- 1. 冗余检查(日截面 Spearman) ----------
    cands = kit._get_cands()
    red = redundancy_report(cands, RED_NAMES, start, MIN_CS)
    red.to_csv(os.path.join(out_dir, f'redundancy_{pool}.csv'),
               index=False, encoding='utf-8-sig')
    if len(red):
        log('\n  [冗余检查] 日截面 Spearman(|rho_mean| 降序, 前 8 对):')
        log(red.head(8).to_string(index=False))
        lines.append('冗余(|rho_mean| 前 5 对): '
                     + '; '.join(f"{r['pair']}={r['mean']}" for _, r in red.head(5).iterrows()))

    # ---------- 2a. 单信号 IS 基线(7 候选, G_vr20 双向) ----------
    singles = {}                                     # name → (sharpe, best_w, payload)
    for name in RED_NAMES:
        grids = [(1.0, 'pos'), (-1.0, 'neg')] if name == FILTER_CAND else [(1.0, 'pos')]
        for w, tag in grids:
            pay = runner.run_window({name: w}, f'single:{name}[{tag}]', start, is_end)
            rows.append(make_row(pool, 'is', pay, f'{name}:{w:g}'))
            sh = _sh(pay['stats'])
            if name not in singles or sh > singles[name][0]:
                singles[name] = (sh, w, pay)
    order = sorted(RED_NAMES, key=lambda n: -singles[n][0])
    log('\n  [单信号 IS 基线]')
    for n in order:
        log(f'    {n:<20s} sharpe={singles[n][0]:6.2f}')
    lines.append('单信号 IS: ' + ', '.join(f'{n}={singles[n][0]:.2f}' for n in order))

    # 冗余二选一: |rho_mean| ≥ min_rho 的对留 IS 更强者, 弱者退出贪心候选池
    drop = set()
    for _, r in red.iterrows():
        if r['mean'] is None or pd.isna(r['mean']) or abs(r['mean']) < min_rho:
            continue
        a, b = r['pair'].split('~')
        loser = b if singles[a][0] >= singles[b][0] else a
        drop.add(loser)
        log(f'    [冗余] {r["pair"]} rho_mean={r["mean"]} >= {min_rho} → 贪心剔除 {loser}')
    if drop:
        lines.append(f'冗余剔除(退出贪心池): {sorted(drop)}')
    greedy_pool = [n for n in CAND_SIGNALS + [FILTER_CAND] if n not in drop]
    if not greedy_pool:                              # 病态兜底: 至少保留 6 信号
        greedy_pool = list(CAND_SIGNALS)

    # ---------- 2b. 贪心前向(IS 窗) ----------
    start_cand = max(greedy_pool, key=lambda n: singles[n][0])
    cur = {start_cand: singles[start_cand][1]}
    cur_sh, cur_pay = singles[start_cand][0], singles[start_cand][2]
    remaining = [n for n in greedy_pool if n != start_cand]
    log(f'\n  [贪心] 起点 {start_cand}:{cur[start_cand]:g} (IS sharpe={cur_sh:.2f}), '
        f'候选池 {remaining}')
    step = 0
    while step < max_add and remaining:
        step += 1
        trials = []
        for cand in remaining:
            for w in _weight_grid(cand):
                trial = dict(cur)
                trial[cand] = w
                pay = runner.run_window(trial, f'greedy_s{step}_{cand}_{w:g}',
                                        start, is_end)
                trials.append({'sh': _sh(pay['stats']), 'cand': cand, 'w': w,
                               'pay': pay, 'weights': wdesc(trial)})
        trials.sort(key=lambda t: -t['sh'])
        best = trials[0]
        for t in trials:
            greedy_log.append({'step': f'add{step}', 'cand': t['cand'],
                               'weight': t['w'], 'sharpe': round(t['sh'], 3),
                               'improve': round(t['sh'] - cur_sh, 3),
                               'accepted': t is best, 'weights': t['weights']})
        if best['sh'] <= cur_sh + margin:
            log(f'    step{step}: 最优 {best["cand"]}:{best["w"]:g} sharpe={best["sh"]:.2f} '
                f'(提升 {best["sh"] - cur_sh:+.2f} <= margin={margin}) → 停止')
            break
        cur[best['cand']] = best['w']
        remaining.remove(best['cand'])
        cur_sh, cur_pay = best['sh'], best['pay']
        log(f'    step{step}: +{best["cand"]}:{best["w"]:g} → sharpe={cur_sh:.2f} '
            f'[{wdesc(cur)}]')
    pay = dict(cur_pay)
    pay['name'] = 'greedy_final'
    rows.append(make_row(pool, 'is', pay, wdesc(cur)))
    lines.append(f'贪心终态(IS sharpe={cur_sh:.2f}): {wdesc(cur)}')

    # ---------- 2c. 精修一遍(×0.5 / ×2 / 剔除) ----------
    for comp in list(cur.keys()):
        if comp not in cur:                          # 精修中可能已被剔除
            continue
        variants = []
        for act, wv in (('x0.5', cur[comp] * 0.5), ('x2', cur[comp] * 2.0)):
            trial = dict(cur)
            trial[comp] = wv
            pay = runner.run_window(trial, f'refine_{comp}_{act}', start, is_end)
            variants.append({'act': act, 'w': wv, 'sh': _sh(pay['stats']),
                             'pay': pay, 'weights': wdesc(trial)})
        if len(cur) >= 2:
            trial = {k: v for k, v in cur.items() if k != comp}
            pay = runner.run_window(trial, f'refine_{comp}_drop', start, is_end)
            variants.append({'act': 'drop', 'w': None, 'sh': _sh(pay['stats']),
                             'pay': pay, 'weights': wdesc(trial)})
        best = max(variants, key=lambda v: v['sh'])
        for v in variants:
            greedy_log.append({'step': f'refine_{comp}', 'cand': comp,
                               'weight': v['w'] if v['w'] is not None else 'drop',
                               'sharpe': round(v['sh'], 3),
                               'improve': round(v['sh'] - cur_sh, 3),
                               'accepted': v is best and v['sh'] > cur_sh,
                               'weights': v['weights']})
        if best['sh'] > cur_sh:
            if best['act'] == 'drop':
                del cur[comp]
            else:
                cur[comp] = best['w']
            cur_sh, cur_pay = best['sh'], best['pay']
            log(f'    [精修] {comp} {best["act"]} → sharpe={cur_sh:.2f} [{wdesc(cur)}]')
    pay = dict(cur_pay)
    pay['name'] = 'combo_is'
    rows.append(make_row(pool, 'is', pay, wdesc(cur)))
    lines.append(f'精修后(IS sharpe={cur_sh:.2f}): {wdesc(cur)}')

    # ---------- 2d. 门槛搜索(top_k ∈ {5,8,12}, exit_rank 固定 12) ----------
    topk_best = None
    for k in TOPK_GRID:
        p_k = kit.engine_params.copy(top_k=k)
        pay = runner.run_window(cur, f'combo_topk{k}', start, is_end, engine_params=p_k)
        rows.append(make_row(pool, 'is', pay, wdesc(cur)))
        sh = _sh(pay['stats'])
        if topk_best is None or sh > topk_best[0]:
            topk_best = (sh, k)
    best_k, is_final_sh = topk_best[1], topk_best[0]
    log(f'\n  [门槛] top_k={best_k} → IS sharpe={is_final_sh:.2f}')
    lines.append(f'门槛: top_k={best_k} → IS sharpe={is_final_sh:.2f}')
    final_p = kit.engine_params.copy(top_k=best_k)

    # ---------- 3. OOS 复验 + 全窗终验 + shuffle/buyhold 对照 ----------
    pay_oos = runner.run_window(cur, 'combo_final', oos_start, None,
                                engine_params=final_p)
    rows.append(make_row(pool, 'oos', pay_oos, wdesc(cur)))
    pay_full = runner.run_window(cur, 'combo_final', start, None, engine_params=final_p)
    rows.append(make_row(pool, 'full', pay_full, wdesc(cur)))
    pay_shuf = runner.run_window(cur, 'combo_shuffle', start, None,
                                 engine_params=final_p, shuffle_seed=shuffle_seed)
    rows.append(make_row(pool, 'full_shuf', pay_shuf, wdesc(cur)))
    pay_bh = runner.buyhold_window('buyhold_pool', start, None)
    rows.append(make_row(pool, 'full', pay_bh, '(等权持有)'))
    # 最强单信号 OOS 参照(与终验同引擎参数, 公平对照)
    pay_sb = runner.run_window({start_cand: singles[start_cand][1]},
                               f'single_best[{start_cand}]', oos_start, None,
                               engine_params=final_p)
    rows.append(make_row(pool, 'oos', pay_sb,
                         f'{start_cand}:{singles[start_cand][1]:g}'))
    # gold4 参照组合(is/oos/full)
    for tag, ws, we in (('is', start, is_end), ('oos', oos_start, None),
                        ('full', start, None)):
        pay = runner.run_window(GOLD4, f'gold4_{tag}', ws, we)
        rows.append(make_row(pool, tag, pay, wdesc(GOLD4)))

    # 终验三件套落盘
    BacktestResult(pay_full).save(os.path.join(out_dir, f'{pool}_final'))

    sh_o, sh_f = _sh(pay_oos['stats']), _sh(pay_full['stats'])
    log(f'\n  [终验] IS={is_final_sh:.2f} OOS={sh_o:.2f} FULL={sh_f:.2f} | '
        f'shuffle={_sh(pay_shuf["stats"]):.2f} buyhold={_sh(pay_bh["stats"]):.2f} | '
        f'单信号OOS={_sh(pay_sb["stats"]):.2f} | 耗时 {time.time() - t0:.0f}s')
    lines.append(f'终验: IS={is_final_sh:.2f} OOS={sh_o:.2f} FULL={sh_f:.2f} '
                 f'shuffle={_sh(pay_shuf["stats"]):.2f} buyhold={_sh(pay_bh["stats"]):.2f} '
                 f'单信号OOS={_sh(pay_sb["stats"]):.2f}')
    final_info = {'pool': pool, 'weights': wdesc(cur), 'top_k': best_k,
                  'is': round(is_final_sh, 3), 'oos': round(sh_o, 3),
                  'full': round(sh_f, 3),
                  'shuffle': round(_sh(pay_shuf['stats']), 3),
                  'buyhold': round(_sh(pay_bh['stats']), 3)}
    return rows, greedy_log, lines, final_info


# ================================================================ CLI ================================================================ #
def main():
    ap = argparse.ArgumentParser(
        description='B3 组合层搜索: 冗余检查 + 贪心权重搜索 + 门槛 top_k, IS/OOS 分窗')
    ap.add_argument('--pool', default=None, help='单池名(默认三池 etf_3x/company_300/company_1000)')
    ap.add_argument('--start', default='2021-01-01', help='交易窗口起点')
    ap.add_argument('--is-end', default='2024-12-31', help='IS 窗终点(选参)')
    ap.add_argument('--oos-start', default='2025-01-01', help='OOS 窗起点(复验)')
    ap.add_argument('--min-rho', type=float, default=0.70, help='冗余阈值 |rho_mean|')
    ap.add_argument('--margin', type=float, default=0.10, help='贪心纳入的最小 IS sharpe 提升')
    ap.add_argument('--max-add', type=int, default=4, help='贪心最多新增成分数')
    ap.add_argument('--shuffle-seed', type=int, default=42)
    ap.add_argument('--tag', default=None, help='输出目录后缀')
    args = ap.parse_args()

    pools = [args.pool] if args.pool else ['etf_3x', 'company_300', 'company_1000']
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(BASE, 'output',
                           f'combo_{run_id}' + (f'_{args.tag}' if args.tag else ''))
    os.makedirs(out_dir, exist_ok=True)
    log(f'B3 组合层搜索 | pools={pools} | IS={args.start}~{args.is_end} '
        f'OOS={args.oos_start}~ | min_rho={args.min_rho} margin={args.margin} '
        f'max_add={args.max_add} | 输出 {out_dir}')

    all_rows, report, finals, t_all = [], [], [], time.time()
    for pool in pools:
        try:
            rows, glog, lines, info = run_pool(pool, args.start, args.is_end,
                                               args.oos_start, args.min_rho, args.margin,
                                               args.max_add, args.shuffle_seed, out_dir)
            all_rows.extend(rows)
            if glog:
                pd.DataFrame(glog).to_csv(os.path.join(out_dir, f'greedy_{pool}.csv'),
                                          index=False, encoding='utf-8-sig')
            report.extend(lines + [''])
            finals.append(info)
        except Exception:
            msg = f'## pool={pool} 失败: {traceback.format_exc()}'
            log(msg)
            report.append(msg)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, 'summary.csv'),
                                      index=False, encoding='utf-8-sig')
    if finals:
        ft = pd.DataFrame(finals)
        log('\n===== 汇总(胜出组合) =====')
        log(ft.to_string(index=False))
        ft.to_csv(os.path.join(out_dir, 'finals.csv'), index=False, encoding='utf-8-sig')
        report.append('## 汇总(胜出组合)\n' + ft.to_string(index=False))
    report.append(f'\n总耗时 {time.time() - t_all:.0f}s | 输出 {out_dir}')
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    log(f'\n完成: {out_dir}')


if __name__ == '__main__':
    main()
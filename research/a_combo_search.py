# -*- coding: utf-8 -*-
"""
a_combo_search.py — A 股组合层优化: 冗余检查 + 权重/门槛搜索 + IS/OOS 分窗
==========================================================================
combo_search.py 的 A 股版(池=hs300/a_etf_all): 框架(PoolRunner/冗余/贪心/精修/
门槛/OOS+shuffle+buyhold)原样复用, 仅替换候选清单与构建公式。

候选依据(两轨挖掘, 2026-09-20 跑):
  alpha_mine_20260920_225353(alpha_mining --pools hs300,a_etf_all):
    两池同号 C 弱头部: H_pos_alpha(h60)/H_ichimoku_alpha/H_er_alpha/H_bbw_alpha/F_er20;
    a_etf 单池: N_range20(h60 exc 3.74% ICIR .312 q=.061 换手 4.9%),
                H_adxstr_alpha(h20 q=.042), F_er20;
    hs300 单池: F_beta60(h60 8.94%, IC 负 → 预期负权), C_tmqmom, N_range20.
  g_mine_20260920_225505(factor_mining2 --pkl 注入两池):
    G_amiasym20 两池同号 C 弱; G_oviv20 a_etf h60 王牌(exc 3.68% ICIR .389
    换手 7.1%); G_dnbeta60 hs300 h60 ICIR .350; G_vr20/G_chop20 过滤器素材.
  score_backtest 历史 PASS: D_sharpe20(a_etf)/D_ma20_dist(hs300).

方向处理: A 股候选方向多数未经组合层确认(美股版 6 信号已在美股池定向, 故只有
G_vr20 双向) → 本版全部候选双向网格(±0.5/±1.0), 单信号基线也双向, 贪心自选向。

候选公式(逐行核对自源模块):
  F_er20/F_beta60/F_mom121 = build_mined(panel)                     [factor_mining]
  H_ichimoku_alpha = _alphaize(ichimoku_distance)                   [alpha_mining L112]
  H_er_alpha       = _alphaize(mined F_er20)                        [alpha_mining L135]
  H_adxstr_alpha   = _alphaize(adx_strength)                        [alpha_mining L115]
  N_range20        = -(high-low).rolling(20).mean()/close           [alpha_mining L156-158 取反 F_range20(factor_mining L116)]
  C_tmqmom         = rk(F_mom121) + rk(F_er20)                      [alpha_mining L162]
  G_oviv20         = std20(overnight)/std20(intraday)               [factor_mining2 L128-131]
  G_amiasym20      = mean20(illiq|up) - mean20(illiq|dn)            [factor_mining2 L162-165]
  G_dnbeta60       = beta_dn - beta_up(池等权收益分段)               [factor_mining2 L187-200]
  G_vr20           = var(pct20,120)/ (20*var(ret1,120))             [factor_mining2 L142-146]
  G_chop20         = -flip(ret1 sign).rolling(20).sum()             [factor_mining2 L182-184]
  D_sharpe20       = pct20 / (std20(ret1)*sqrt(20))                 [signal_search L65]
  D_ma20_dist      = close/ma20 - 1                                 [signal_search L68]
  goldA4 参照       = {N_range20:1, F_er20:1, H_ichimoku_alpha:1, G_amiasym20:1}

A 股适配: 引擎只做多(天然吻合 A 股约束); T+1 与回测口径(s 收盘决策 d+1 开盘成交)
吻合; 单边 10bps 保守(ETF 实际更低); 涨跌停截断波动尾部。
池接入: BacktestKit(pool=...) 走 alpha_mining.POOLS 的 research pkl, 零改动。

输出: research/output/a_combo_{run_id}[_{tag}]/
  summary.csv(全部配置×窗口)  redundancy_{pool}.csv  greedy_{pool}.csv(试验史)
  {pool}_final/(胜出配置 3件套)  report.txt  finals.csv
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
CAND_SIGNALS = [
    'F_er20',            # 两池同号 C 弱(alpha 轨)
    'H_ichimoku_alpha',  # 两池同号 C 弱
    'H_er_alpha',        # 两池同号 C 弱(与 F_er20 同素材, 冗余二选一)
    'N_range20',         # 两池最稳低波(a_etf h60 ICIR .312 / hs300 .317)
    'H_adxstr_alpha',    # a_etf 最显著(q=.042)
    'G_oviv20',          # a_etf h60 王牌(ICIR .389, T+1 隔夜微观结构)
    'G_amiasym20',       # 两池同号 C 弱(g 轨)
    'G_dnbeta60',        # hs300 h60 ICIR .350
    'F_beta60',          # hs300 王牌(IC 负 → 预期负权/低beta)
    'C_tmqmom',          # hs300 h20 动量复合
    'D_sharpe20',        # a_etf PASS(score_backtest)
    'D_ma20_dist',       # hs300 PASS(score_backtest)
]
FILTER_CANDS = ['G_vr20', 'G_chop20']              # 过滤器候选(方向不确定)
RED_NAMES = CAND_SIGNALS + FILTER_CANDS            # 冗余检查/单信号评估集合(14)
GOLD_A4 = {'N_range20': 1.0, 'F_er20': 1.0,
           'H_ichimoku_alpha': 1.0, 'G_amiasym20': 1.0}   # A 股参照(两池同号 4 因子)
COMBO_COLS = RED_NAMES                              # 需注入 panel 的全部成分列
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
def build_a_combo_cands(panel: pd.DataFrame) -> dict:
    """A 股版全部候选(全历史构建, 因果安全; 公式逐行核对自源模块, 见模块 docstring)."""
    close = panel['Close'].unstack('symbol').sort_index()
    open_ = panel['Open'].unstack('symbol').sort_index()
    high = panel['High'].unstack('symbol').sort_index()
    low = panel['Low'].unstack('symbol').sort_index()
    volume = panel['Volume'].unstack('symbol').sort_index()
    ret1 = close.pct_change()
    mined = build_mined(panel)          # F_ 族底料(整块复用, 无口径漂移)
    rk = lambda v: v.rank(axis=1, pct=True)  # noqa: E731

    def w(col):
        return pd.to_numeric(panel[col], errors='coerce').unstack('symbol').sort_index()

    out = {}
    # ---- alpha_mining H/N/C 族 ----
    out['H_ichimoku_alpha'] = _alphaize(w('ichimoku_distance'))
    out['H_er_alpha'] = _alphaize(mined['F_er20'])
    out['H_adxstr_alpha'] = _alphaize(w('adx_strength'))
    out['N_range20'] = -(high - low).rolling(20).mean() / close
    out['C_tmqmom'] = rk(mined['F_mom121']) + rk(mined['F_er20'])
    out['F_er20'] = mined['F_er20']
    out['F_beta60'] = mined['F_beta60']
    # ---- factor_mining2 G 族 ----
    overnight = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    out['G_oviv20'] = overnight.rolling(20, min_periods=10).std() \
        / intraday.rolling(20, min_periods=10).std().replace(0, np.nan)
    up_m, dn_m = ret1 > 0, ret1 < 0
    dvusd = (close * volume).replace(0, np.nan)
    illiq = ret1.abs() / dvusd
    out['G_amiasym20'] = illiq.where(up_m).rolling(20, min_periods=5).mean() \
        - illiq.where(dn_m).rolling(20, min_periods=5).mean()
    pr = ret1.mean(axis=1)                     # 池等权收益(仅用 t 及之前)
    m_dn = pd.DataFrame(np.repeat((pr.values < 0)[:, None], ret1.shape[1], axis=1),
                        index=ret1.index, columns=ret1.columns)
    m_up = pd.DataFrame(np.repeat((pr.values > 0)[:, None], ret1.shape[1], axis=1),
                        index=ret1.index, columns=ret1.columns)
    r_dn = ret1.where(m_dn)
    r_up = ret1.where(m_up)
    pr_dn = pr.where(pr < 0)
    pr_up = pr.where(pr > 0)
    beta_dn = r_dn.rolling(60, min_periods=15).cov(pr_dn) \
        .div(pr_dn.rolling(60, min_periods=15).var().replace(0, np.nan), axis=0)
    beta_up = r_up.rolling(60, min_periods=15).cov(pr_up) \
        .div(pr_up.rolling(60, min_periods=15).var().replace(0, np.nan), axis=0)
    out['G_dnbeta60'] = beta_dn - beta_up
    var20 = close.pct_change(20).rolling(120, min_periods=60).var()
    var1 = ret1.rolling(120, min_periods=60).var()
    out['G_vr20'] = var20 / (20.0 * var1.replace(0, np.nan))
    sgn = np.sign(ret1)
    flip = sgn.diff().abs().eq(2.0).astype(float)
    out['G_chop20'] = -flip.rolling(20, min_periods=10).sum()
    # ---- signal_search D 族 ----
    out['D_sharpe20'] = close.pct_change(20) / (ret1.rolling(20).std() * np.sqrt(20))
    out['D_ma20_dist'] = close / close.rolling(20).mean() - 1.0
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
        # 探针含负权, 覆盖负向路径的符号处理
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
    """A 股版: 全部候选双向(方向未经组合层确认, 数据自选向)."""
    return [-1.0, -0.5, 0.5, 1.0]


# ================================================================ 逐池主流程 ================================================================ #
def run_pool(pool: str, start: str, is_end: str, oos_start: str, min_rho: float,
             margin: float, max_add: int, shuffle_seed: int, out_dir: str):
    """一个池的完整流程: 冗余 → 单信号基线 → 贪心 → 精修 → 门槛 → OOS/终验/对照.
    返回 (rows, greedy_log, report_lines, final_info)."""
    log(f'\n{"=" * 70}\n===== pool={pool} =====')
    t0 = time.time()
    kit = BacktestKit(pool=pool, start=None)          # 全历史构建信号, 交易窗口逐次截
    kit.register(build_a_combo_cands, label='A股组合候选')
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

    # ---------- 2a. 单信号 IS 基线(全候选双向) ----------
    singles = {}                                     # name → (sharpe, best_w, payload)
    for name in RED_NAMES:
        for wv, tag in ((1.0, 'pos'), (-1.0, 'neg')):
            pay = runner.run_window({name: wv}, f'single:{name}[{tag}]', start, is_end)
            rows.append(make_row(pool, 'is', pay, f'{name}:{wv:g}'))
            sh = _sh(pay['stats'])
            if name not in singles or sh > singles[name][0]:
                singles[name] = (sh, wv, pay)
    order = sorted(RED_NAMES, key=lambda n: -singles[n][0])
    log('\n  [单信号 IS 基线]')
    for n in order:
        log(f'    {n:<20s} w={singles[n][1]:+g} sharpe={singles[n][0]:6.2f}')
    lines.append('单信号 IS: ' + ', '.join(f'{n}({singles[n][1]:+g})={singles[n][0]:.2f}'
                                           for n in order))

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
    greedy_pool = [n for n in RED_NAMES if n not in drop]
    if not greedy_pool:                              # 病态兜底
        greedy_pool = list(CAND_SIGNALS)

    # ---------- 2b. 贪心前向(IS 窗) ----------
    start_cand = max(greedy_pool, key=lambda n: singles[n][0])
    cur = {start_cand: singles[start_cand][1]}
    cur_sh, cur_pay = singles[start_cand][0], singles[start_cand][2]
    remaining = [n for n in greedy_pool if n != start_cand]
    log(f'\n  [贪心] 起点 {start_cand}:{cur[start_cand]:g} (IS sharpe={cur_sh:.2f}), '
        f'候选池 {len(remaining)} 个')
    step = 0
    while step < max_add and remaining:
        step += 1
        trials = []
        for cand in remaining:
            for wv in _weight_grid(cand):
                trial = dict(cur)
                trial[cand] = wv
                pay = runner.run_window(trial, f'greedy_s{step}_{cand}_{wv:g}',
                                        start, is_end)
                trials.append({'sh': _sh(pay['stats']), 'cand': cand, 'w': wv,
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
    # goldA4 参照组合(is/oos/full)
    for tag, ws, we in (('is', start, is_end), ('oos', oos_start, None),
                        ('full', start, None)):
        pay = runner.run_window(GOLD_A4, f'goldA4_{tag}', ws, we)
        rows.append(make_row(pool, tag, pay, wdesc(GOLD_A4)))

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
        description='A 股组合层搜索: 冗余检查 + 贪心权重搜索 + 门槛 top_k, IS/OOS 分窗')
    ap.add_argument('--pool', default=None, help='单池名(默认两池 hs300/a_etf_all)')
    ap.add_argument('--start', default='2021-01-01', help='交易窗口起点')
    ap.add_argument('--is-end', default='2024-12-31', help='IS 窗终点(选参)')
    ap.add_argument('--oos-start', default='2025-01-01', help='OOS 窗起点(复验)')
    ap.add_argument('--min-rho', type=float, default=0.70, help='冗余阈值 |rho_mean|')
    ap.add_argument('--margin', type=float, default=0.10, help='贪心纳入的最小 IS sharpe 提升')
    ap.add_argument('--max-add', type=int, default=4, help='贪心最多新增成分数')
    ap.add_argument('--shuffle-seed', type=int, default=42)
    ap.add_argument('--tag', default=None, help='输出目录后缀')
    args = ap.parse_args()

    pools = [args.pool] if args.pool else ['hs300', 'a_etf_all']
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(BASE, 'output',
                           f'a_combo_{run_id}' + (f'_{args.tag}' if args.tag else ''))
    os.makedirs(out_dir, exist_ok=True)
    log(f'A 股组合层搜索 | pools={pools} | IS={args.start}~{args.is_end} '
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

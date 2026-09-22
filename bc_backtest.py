# -*- coding: utf-8 -*-
"""
bc_backtest.py — 合并自 research 平铺模块(6 个: score_backtest, bt_core, signal_timeseries, signal_backtest, stop_loss_eval, exec_price_ab_test).

生成: _dbg_build_bc.py 自动拼接 + AST 精确改名(同名冲突加模块前缀, 未经人工改动).
规则:
  - 值完全相同的重复常量仅保留首处定义;
  - 被跨模块 import 的符号保留原名, 私有冲突符号加模块前缀(见各段内改名注释);
  - 源模块保留于 research/ 目录未删除, 供新旧一致性对拍.
"""
from datetime import datetime
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
import warnings
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from quant.bc_factor_search import POOLS, build_alpha_cands, build_derived, load_panel, normalize_causal

# 原 research 模块目录(合并文件位于 git/quant/, 输出路径保持与源模块一致)
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research')
BASE = HERE  # combo_search/a_combo_search/exec_price_ab_test 输出路径


# ==========================================================================
# ==== 源自 score_backtest.py ====  
# ==== 改名: main->sb_main
# ==========================================================================
WEIGHT_PRESETS = {
    'default': {'position_score': 0.25, 'rsi': 0.20, 'pattern_score_alpha': 0.20, 'trigger_score': -0.15, 'boundary_score': -0.10, 'candle_position_score': -0.10},
    'rank_only': {'position_score': 0.45, 'rsi': 0.35, 'pattern_score_alpha': 0.20},
    'flip_only': {'pattern_score_alpha': 0.20, 'trigger_score': -0.30, 'boundary_score': -0.25, 'candle_position_score': -0.25},
}

MOMENTUM_SETS = {
    'M1_mom250': {'D_mom250': 1.0},
    'M2_mom120': {'D_mom120': 1.0},
    'M3_ma200dist': {'D_ma200_dist': 1.0},
    'M4_adxstrength': {'adx_strength': 1.0},
    'M5_mom250+ma200': {'D_mom250': 0.6, 'D_ma200_dist': 0.4},
    'M6_mix': {'D_mom250': 0.4, 'D_mom120': 0.2, 'D_ma200_dist': 0.2, 'D_sharpe120': 0.1, 'adx_strength': 0.1},
    # 方向性对照(不是候选策略): 反向动量=买过去一年最弱者. 若 mom250 含真实 alpha,
    # 此配置须显著劣于 M1_mom250, 用于排除"任意排序都能赚"的伪信号解释.
    'M1_neg_mom250': {'D_mom250': -1.0},
    # ---- 跨池样本外(etf_3x/company_300/hs300/a_etf_all)筛出的因子族候选 ----
    # 波动族: hs300 真伪检验 PASS(D_vol20/60/10, D_atr_pct20). 两个方向都测,
    # 因筛查口径是"信号值最大者做多", 而回测历史上低波动倾斜(R3/R4)也有效 -> 需实验判定.
    'M7_vol20_low': {'D_vol20': -1.0},        # 买低 20 日波动
    'M8_vol20_high': {'D_vol20': 1.0},        # 买高 20 日波动(筛查方向)
    'M9_atr_low': {'D_atr_pct20': -1.0},      # 买低 ATR%/价格
    # Kijun 距离: 唯一四池筛查 excess_k 全正的信号; 同时测反向做对照
    'M10_kijun': {'Low_to_kijun': 1.0},
    'M10_kijun_neg': {'Low_to_kijun': -1.0},
    # 反转族: company_300 真伪检验 PASS(D_rev3)
    'M11_rev3': {'D_rev3': 1.0},
    # 趋势位置族: hs300 PASS(D_ma20_dist), a_etf_all PASS(D_sharpe20)
    'M12_ma20dist': {'D_ma20_dist': 1.0},
    'M13_sharpe20': {'D_sharpe20': 1.0},
    # 组合: 跨池唯一稳定因子 ADX 强度 + 波动族去冗余
    'M14_adx_vol': {'adx_strength': 0.6, 'D_vol20': -0.4},
    'M15_adx_kijun': {'adx_strength': 0.5, 'Low_to_kijun': 0.5},
    # ---- indicator_value_review_20260918 8.3 提议的 B 级技术指标组合 ----
    # 成员取四池全正/最可靠的 B 级; M16/M17 依赖 ichimoku_distance_alpha
    # (新版生产代码已删该列, 由 build_alpha_synths 自动按旧口径复现注入, 见下)
    'M16_kijun_ichimoku': {'Low_to_kijun': 0.5, 'ichimoku_distance_alpha': 0.5},
    'M17_adx_ichimoku':   {'adx_power': 0.5, 'ichimoku_distance_alpha': 0.5},
    'M18_ichimoku_only':  {'ichimoku_distance_alpha': 1.0},
}

WEIGHT_PRESETS.update(MOMENTUM_SETS)

GOLD_SETS = {
    'gold4': {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2, 'F_obv20': 0.1},
    'gold3': {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2},
}

WEIGHT_PRESETS.update(GOLD_SETS)

def build_alpha_synths(panel: pd.DataFrame) -> dict:
    """复现旧版生产代码产出、新版已删除的因果归一化列(当前仅 ichimoku_distance_alpha).

    背景: indicator_value_review_20260918 的 B 级评估基于旧版 208 列 pkl; 新版
    bc_technical_analysis.py(205 列)不再产出该列. 口径按 m_trend_score 首项同源:
    sign(ichimoku_distance) * normalize_causal(|ichimoku_distance|, 252, 60).
    pkl 已含该列时注入器自动跳过(不覆盖), 只用 t 及之前数据, 无前视.
    """
    out = {}
    if 'ichimoku_distance' in panel.columns:
        d = pd.to_numeric(panel['ichimoku_distance'], errors='coerce') \
              .unstack('symbol').sort_index()
        out['ichimoku_distance_alpha'] = np.sign(d) * normalize_causal(
            d.abs(), window=252, min_periods=60)
    return out

def compute_composite(study: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """组合分数 = Σ w_i * 成分当日截面百分位排名(rank pct, 0~1).
    成分缺失(NaN)以中性 0.5 参与; 单一成分整列缺失时该成分退化为常数, 无排序贡献."""
    total = float(sum(abs(w) for w in weights.values()))
    if total <= 0:
        raise ValueError('权重绝对值和为 0')
    comp = None
    for col, w in weights.items():
        if col not in study.columns:
            raise ValueError(f'组合分数成分列不存在: {col}')
        wide = pd.to_numeric(study[col], errors='coerce').unstack('symbol').sort_index()
        pct = wide.rank(axis=1, pct=True)  # 当日截面, NaN 保留
        part = (w / total) * pct
        comp = part if comp is None else comp.add(part, fill_value=0.0)
    return comp.fillna(0.5)

def shuffled_composite(comp: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """随机对照: 整张截面按交易日置换(某日的截面值来自随机其他交易日), 破坏分数-时间对齐."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(comp))
    return comp.iloc[perm].set_axis(comp.index)

class EngineParams:
    def __init__(self, top_k=5, exit_rank=12, sizing='tier', max_exposure=1.0,
                 per_symbol_cap=0.30, cost_bps=10.0, band=0.05,
                 stop_loss=None, take_profit=None, trail_stop=None,
                 stop_atr=None, trail_atr=None):
        self.top_k = top_k
        self.exit_rank = exit_rank
        self.sizing = sizing
        self.max_exposure = max_exposure
        self.per_symbol_cap = per_symbol_cap
        self.cost_rate = cost_bps / 10000.0  # 单边
        self.band = band
        # 价格止盈止损(默认关闭). 触发时间线与 rank 退出一致(防前视):
        # 信号日 s 收盘价较 entry_px 判定 -> 执行日 d=s+1 开盘卖出.
        # stop_loss  如 0.08: s 收盘较 entry 跌幅 >= 8% 触发
        # take_profit 如 0.15: 涨幅 >= 15% 触发
        # trail_stop  如 0.20: 自持仓以来最高收盘回落 >= 20% 触发(先更新高点再判定)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trail_stop = trail_stop
        # ATR 动态止损(波动自适应, k 为 ATR 倍数; 需引擎传入 atr_wide, 缺失则该股不判):
        # stop_atr  如 2.0: s 收盘 <= entry_px - 2*ATR(s) 触发(高波动股自动放宽)
        # trail_atr 如 3.0: 吊灯止损, s 收盘 <= 持仓最高收盘 - 3*ATR(s) 触发
        self.stop_atr = stop_atr
        self.trail_atr = trail_atr

    def copy(self, **kw):
        p = EngineParams(self.top_k, self.exit_rank, self.sizing, self.max_exposure,
                         self.per_symbol_cap, self.cost_rate * 10000.0, self.band,
                         self.stop_loss, self.take_profit, self.trail_stop,
                         self.stop_atr, self.trail_atr)
        for k, v in kw.items():
            if k == 'cost_bps':  # 引擎读 cost_rate, 需换算而非 setattr 新属性
                p.cost_rate = float(v) / 10000.0
            else:
                setattr(p, k, v)
        return p

    def brief(self):
        stops = []
        if self.stop_loss is not None:
            stops.append(f'SL={self.stop_loss:.0%}')
        if self.take_profit is not None:
            stops.append(f'TP={self.take_profit:.0%}')
        if self.trail_stop is not None:
            stops.append(f'TR={self.trail_stop:.0%}')
        if self.stop_atr is not None:
            stops.append(f'SLA={self.stop_atr:g}xATR')
        if self.trail_atr is not None:
            stops.append(f'CH={self.trail_atr:g}xATR')
        return (f'K={self.top_k}, exit_rank={self.exit_rank}, sizing={self.sizing}, '
                f'exposure={self.max_exposure}, cap={self.per_symbol_cap}, '
                f'cost={self.cost_rate * 10000:.0f}bps(单边), band={self.band}'
                + (f', stops[{",".join(stops)}]' if stops else ''))

def _sizing_coef(sub: pd.Series, sizing: str) -> pd.Series:
    """已选持仓内部的仓位系数: tier 三档 1.5/1.0/0.5, linear 0.5~1.5, equal 1.0.
    sub 为持仓的组合分数(高分=好); 返回与 sub 同索引的系数."""
    n = len(sub)
    if n <= 1:
        return pd.Series(1.0, index=sub.index)
    frac = (n - sub.rank(ascending=False, method='first')) / (n - 1)  # 最好=1, 最差=0
    if sizing == 'equal':
        return pd.Series(1.0, index=sub.index)
    if sizing == 'linear':
        return 0.5 + frac
    tier = np.ceil(frac * 3.0)  # 0..3, 最好=3
    coef = np.where(tier >= 3, 1.5, np.where(tier >= 2, 1.0, 0.5))
    return pd.Series(coef, index=sub.index)

def run_engine(open_wide: pd.DataFrame, close_wide: pd.DataFrame,
               composite: pd.DataFrame, gate: pd.DataFrame, p: EngineParams,
               atr_wide: pd.DataFrame = None):
    """逐日事件引擎. 返回 (daily_df, positions_df, trades_df, equity_series).

    时间线(防前视核心):
      信号日 s=dates[i-1] 收盘: 取 composite.loc[s] / gate.loc[s] 与当时持仓 -> 目标权重
      执行日 d=dates[i] 开盘: 结算(open s -> open d), 漂移, 换仓, 收成本
    分数与门只用 s 行; 目标只作用于 d 开盘之后; 收益只从成交开盘起算.
    近似说明: 决策时用的"当前权重"是漂移到 d 开盘的值(而非 s 收盘盯市),
    仅影响 band 微调与权重小数, 不影响进出场决策(持仓集合是确定的).
    """
    dates = open_wide.index
    rets = open_wide.pct_change()
    weights = {}        # symbol -> 最新执行后的权重(以执行日开盘市值计)
    open_trades = {}    # symbol -> 持仓段信息
    trades, daily, pos_rows = [], [], []
    equity, turnover_sum = 1.0, 0.0

    for i in range(1, len(dates)):
        d, s = dates[i], dates[i - 1]
        # --- A. 结算: weights 于 open(s) 建立, 持有至 open(d) ---
        gross = 0.0
        if weights:
            gross = float(sum(w * (rets.at[d, sym] if pd.notna(rets.at[d, sym]) else 0.0)
                              for sym, w in weights.items()))
        equity *= (1.0 + gross)
        if weights and gross > -1.0:
            weights = {sym: w * (1.0 + (rets.at[d, sym] if pd.notna(rets.at[d, sym]) else 0.0)) / (1.0 + gross)
                       for sym, w in weights.items()}
        # --- B. 信号日 s 收盘决策 ---
        comp = composite.loc[s]
        gate_on = gate.loc[s] if s in gate.index else None
        held = list(weights.keys())
        order = comp.rank(ascending=False, method='first') if len(comp) else pd.Series(dtype=float)
        exit_set, exit_reason = [], {}
        for sym in held:
            r = order.get(sym, np.nan)
            gate_off = (gate_on is not None) and (not bool(gate_on.get(sym, False)))
            rank_bad = pd.isna(r) or (r > p.exit_rank)
            reason = 'gate' if gate_off else ('rank' if rank_bad else None)
            # 价格止损/止盈: s 收盘价较 entry_px 判定 -> d 开盘执行(与 rank 退出同一时间线, 防前视)
            t = open_trades.get(sym)
            if reason is None and t is not None and t['entry_px'] > 0:
                px_s = close_wide.at[s, sym] if sym in close_wide.columns else np.nan
                if pd.notna(px_s):
                    px_s = float(px_s)
                    t['peak_px'] = max(t['peak_px'], px_s)
                    chg = px_s / t['entry_px'] - 1.0
                    # ATR(信号日 s, 价格单位) 供动态止损; 缺失则该股当日不判 ATR 类止损
                    atr_s = np.nan
                    if atr_wide is not None and s in atr_wide.index and sym in atr_wide.columns:
                        _v = atr_wide.at[s, sym]
                        if pd.notna(_v):
                            atr_s = float(_v)
                    if p.stop_loss is not None and chg <= -p.stop_loss:
                        reason = 'stop'
                    elif p.take_profit is not None and chg >= p.take_profit:
                        reason = 'take'
                    elif (p.trail_stop is not None
                          and t['peak_px'] > 0
                          and px_s <= t['peak_px'] * (1.0 - p.trail_stop)):
                        reason = 'trail'
                    # ATR 固定止损(波动自适应): s 收盘 <= entry_px - k*ATR
                    elif (p.stop_atr is not None and atr_s > 0
                          and px_s <= t['entry_px'] - p.stop_atr * atr_s):
                        reason = 'stopatr'
                    # 吊灯止损: s 收盘自持仓最高收盘回落 k*ATR
                    elif (p.trail_atr is not None and atr_s > 0
                          and px_s <= t['peak_px'] - p.trail_atr * atr_s):
                        reason = 'chand'
            if reason is not None:
                exit_set.append(sym)
                exit_reason[sym] = reason
        for sym in exit_set:  # 平仓(出场开盘价缺失时退回当日收盘价)
            px = open_wide.at[d, sym]
            if pd.isna(px) and sym in close_wide.columns:
                px = close_wide.at[d, sym]
            t = open_trades.pop(sym, None)
            if t is not None and pd.notna(px):
                trades.append({'symbol': sym, 'entry_exec': t['entry_exec'], 'exit_exec': d,
                               'entry_px': t['entry_px'], 'exit_px': float(px),
                               'entry_w': t['entry_w'], 'peak_w': t['peak_w'],
                               'days': i - t['entry_i'], 'ret': float(px) / t['entry_px'] - 1.0,
                               'exit_reason': exit_reason.get(sym, 'rank'), 'completed': True})
            elif t is not None:
                open_trades[sym] = t  # 价格缺失, 留待下一日再平
        # 进入候选: 门开 & rank<=top_k & 未持有(含刚退出者) & 执行日有开盘价
        cand = []
        if len(order):
            for sym, r in order.items():
                if sym in weights or pd.isna(r) or r > p.top_k:
                    continue
                if gate_on is not None and not bool(gate_on.get(sym, False)):
                    continue
                if not pd.notna(open_wide.at[d, sym]):
                    continue
                cand.append((float(r), sym))
            cand.sort()
        slots = max(0, p.top_k - (len(held) - len(exit_set)))
        entries = [sym for _, sym in cand[:slots]]
        new_held = [sym for sym in held if sym not in exit_set] + entries
        # --- C. 目标权重 ---
        target = {}
        if new_held:
            sub = comp.reindex(new_held)
            coef = _sizing_coef(sub, p.sizing)
            tot = float(coef.sum())
            if tot > 0:
                target = {sym: min(float(coef[sym] / tot * p.max_exposure), p.per_symbol_cap)
                          for sym in new_held}
        # band: 无进出场且最大偏差 < band -> 保持漂移权重(不交易)
        if not exit_set and not entries and weights:
            maxdev = max((abs(target.get(sym, 0.0) - weights.get(sym, 0.0))
                          for sym in set(target) | set(weights)), default=0.0)
            if maxdev < p.band:
                target = dict(weights)
        # --- D. 换手与成本 ---
        turnover = float(sum(abs(target.get(sym, 0.0) - weights.get(sym, 0.0))
                             for sym in set(target) | set(weights)))
        cost_drag = turnover * p.cost_rate
        equity *= (1.0 - cost_drag)
        turnover_sum += turnover
        weights = {sym: w for sym, w in target.items() if w > 1e-9}
        for sym in entries:
            if sym in weights:
                open_trades[sym] = {'entry_exec': d, 'entry_px': float(open_wide.at[d, sym]),
                                    'entry_w': weights[sym], 'peak_w': weights[sym],
                                    'peak_px': float(open_wide.at[d, sym]), 'entry_i': i}
        for sym, t in open_trades.items():
            t['peak_w'] = max(t['peak_w'], weights.get(sym, 0.0))
        daily.append({'date': d, 'equity': equity, 'gross_ret': gross, 'cost_drag': cost_drag,
                      'turnover': turnover, 'n_pos': len(weights),
                      'exposure': float(sum(weights.values()))})
        pos_rows += [(d, sym, w) for sym, w in weights.items()]

    # --- 期末: 未平仓段以最后交易日收盘标记; 最后执行日建的仓按收盘结算 ---
    last_d = dates[-1]
    for sym in list(open_trades.keys()):
        t = open_trades.pop(sym)
        px = close_wide.at[last_d, sym] if sym in close_wide.columns else np.nan
        if pd.notna(px) and t['entry_px'] > 0:
            trades.append({'symbol': sym, 'entry_exec': t['entry_exec'], 'exit_exec': last_d,
                           'entry_px': t['entry_px'], 'exit_px': float(px),
                           'entry_w': t['entry_w'], 'peak_w': t['peak_w'],
                           'days': len(dates) - 1 - t['entry_i'],
                           'ret': float(px) / t['entry_px'] - 1.0,
                           'exit_reason': 'open', 'completed': False})
    if weights and daily:
        final = sum(w * (close_wide.at[last_d, sym] / open_wide.at[last_d, sym] - 1.0)
                    for sym, w in weights.items()
                    if sym in close_wide.columns and pd.notna(close_wide.at[last_d, sym])
                    and pd.notna(open_wide.at[last_d, sym]) and open_wide.at[last_d, sym] > 0)
        if final != 0:
            daily[-1]['equity'] *= (1.0 + final)

    daily_df = pd.DataFrame(daily).set_index('date')
    pos_df = pd.DataFrame(pos_rows, columns=['date', 'symbol', 'weight'])
    trades_df = pd.DataFrame(trades)
    n_days = max(len(daily_df), 1)
    ann_turnover = turnover_sum / n_days * 252.0
    return daily_df, pos_df, trades_df, daily_df['equity'], ann_turnover

def perf_stats(equity: pd.Series, ann_turnover: float = np.nan,
               trades_df: pd.DataFrame = None) -> dict:
    if len(equity) < 2:
        return {'n_days': len(equity)}
    ret = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 else -1.0
    vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else np.nan
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) > 1 and ret.std() > 0 else np.nan
    dd = float((equity / equity.cummax() - 1.0).min())
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    out = {'n_days': len(equity), 'total_ret': round(total, 3), 'cagr': round(cagr, 3),
           'vol': round(vol, 3), 'sharpe': round(sharpe, 2), 'max_dd': round(dd, 3),
           'calmar': round(calmar, 2) if pd.notna(calmar) else np.nan,
           'ann_turnover': round(ann_turnover, 1) if pd.notna(ann_turnover) else np.nan}
    if trades_df is not None and len(trades_df):
        done = trades_df[trades_df['completed']] if 'completed' in trades_df.columns else trades_df
        if len(done):
            out.update({'n_trades': int(len(done)),
                        'win_rate': round(float((done['ret'] > 0).mean()), 3),
                        'avg_ret': round(float(done['ret'].mean()), 4),
                        'avg_days': round(float(done['days'].mean()), 1)})
    return out

def yearly_returns(equity: pd.Series) -> pd.Series:
    if len(equity) < 2:
        return pd.Series(dtype=float)
    yr = equity.groupby(equity.index.year)
    return (yr.last() / yr.first() - 1.0).round(3)

def run_config(name: str, open_wide, close_wide, composite, gate, p: EngineParams,
               atr_wide=None) -> dict:
    daily_df, pos_df, trades_df, equity, ann_to = run_engine(open_wide, close_wide,
                                                             composite, gate, p,
                                                             atr_wide=atr_wide)
    stats = perf_stats(equity, ann_to, trades_df)
    stats['config'] = name
    return {'name': name, 'stats': stats, 'daily': daily_df, 'pos': pos_df,
            'trades': trades_df, 'equity': equity, 'ann_turnover': ann_to}

def summary_table(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        s = dict(r['stats'])
        rows.append(s)
    cols = ['config', 'total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'vol',
            'ann_turnover', 'n_trades', 'win_rate', 'avg_ret', 'avg_days', 'n_days']
    t = pd.DataFrame(rows)
    return t[[c for c in cols if c in t.columns]]

def sb_main():
    ap = argparse.ArgumentParser(description='独立只读研究: 组合分数 + 动态仓位组合回测')
    ap.add_argument('--pool', default='etf_3x', help='池名, 默认 etf_3x')
    ap.add_argument('--interval', default='day', help='数据频率, 默认 day')
    ap.add_argument('--pkl-dir', default=os.path.join(os.path.expanduser('~'), 'quant'), help='pkl 所在目录')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl 路径(优先, 用于读取带后缀的 pkl)')
    ap.add_argument('--start', default='2021-01-01', help='回测起始(信号)日, 默认 2021-01-01(避开252日warmup)')
    ap.add_argument('--end', default=None, help='回测结束日')
    ap.add_argument('--gate-col', default='trend_magnitude_day', help='门列, >0 视为开; none=关闭')
    ap.add_argument('--weights', default='default', help='权重预设名(default/rank_only/flip_only)或JSON')
    ap.add_argument('--top-k', type=int, default=5, help='持仓数上限(进入条件)')
    ap.add_argument('--exit-rank', type=int, default=12, help='滞回退出名次(>该名次退出)')
    ap.add_argument('--sizing', default='tier', choices=['tier', 'linear', 'equal'], help='仓位分档方式')
    ap.add_argument('--max-exposure', type=float, default=1.0, help='总敞口上限')
    ap.add_argument('--per-symbol-cap', type=float, default=0.30, help='单票权重上限')
    ap.add_argument('--cost-bps', type=float, default=10.0, help='单边成本(bps)')
    ap.add_argument('--band', type=float, default=0.05, help='重平衡带宽(权重偏差小于该值不调仓)')
    ap.add_argument('--sensitivity', action='store_true', help='附消融/敏感性矩阵')
    ap.add_argument('--skip-baselines', action='store_true', help='不跑基线与shuffle对照')
    ap.add_argument('--derived', action='store_true',
                    help='注入 signal_search 的衍生信号(D_mom*/D_ma*_dist/D_sharpe*/adx_strength 等), '
                         '使其可直接用于 --weights 与敏感性矩阵')
    ap.add_argument('--mined', action='store_true',
                    help='注入 factor_mining 的挖矿信号(F_mom121/F_er*/F_idiovol* 等 38 个), '
                         '使其可直接用于 --weights')
    args = ap.parse_args()

    pkl_path = args.pkl_path or os.path.join(args.pkl_dir, f'{args.pool}_{args.interval}_ta_data.pkl')
    if not os.path.exists(pkl_path):
        print(f'[ERROR] 找不到 pkl: {pkl_path}')
        sys.exit(1)

    # 权重
    if args.weights in WEIGHT_PRESETS:
        W = dict(WEIGHT_PRESETS[args.weights])
    else:
        try:
            W = json.loads(args.weights)
            assert isinstance(W, dict) and W
        except Exception:
            print(f'[ERROR] --weights 既不是预设{list(WEIGHT_PRESETS)}也不是合法JSON: {args.weights}')
            sys.exit(1)
    if args.exit_rank < args.top_k:
        print('[ERROR] exit_rank 应 >= top_k (滞回缓冲)')
        sys.exit(1)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output',
                           f'{args.pool}_bt_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    # ---- 套件构建(bt_core 薄预设; 引擎/统计仍为本文件底层实现, 口径逐位一致) ----
    # 延迟导入: bt_core 顶层 import 本模块引擎层, 函数内导入避免循环依赖
    pass  # (合并至本文件: BacktestKit)
    p = EngineParams(args.top_k, args.exit_rank, args.sizing, args.max_exposure,
                     args.per_symbol_cap, args.cost_bps, args.band)
    gate_spec = None if args.gate_col.lower() == 'none' else args.gate_col
    try:
        kit = BacktestKit(pool=args.pool, pkl_path=pkl_path, interval=args.interval,
                          start=args.start, end=args.end, gate_col=gate_spec,
                          engine_params=p)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))
    if gate_spec is not None and gate_spec not in kit.panel_full.columns:
        print(f'[ERROR] 门列不存在: {args.gate_col}')
        sys.exit(1)
    if args.derived:
        kit.register(build_derived, label='衍生信号')
    if args.mined:
        from quant.bc_factor_search import build_mined
        kit.register(build_mined, label='挖矿信号')
    kit._get_cands()   # 立即构建全部注册信号并注入列(全历史, rolling 需 warmup)

    for label in ['alpha复现'] + (['衍生信号'] if args.derived else []) \
            + (['挖矿信号'] if args.mined else []):
        n = sum(1 for lb, _ in kit._inject_log if lb == label)
        print(f'== {label} == 注入 {n} 列')

    study = kit.study               # 交易窗口内 panel(含已注入的衍生/挖矿列)
    dates = study.index.get_level_values('date')
    symbols = sorted(study.index.get_level_values('symbol').unique())
    print(f'== 数据 == {args.pool}: {len(symbols)} 标的, 研究窗口 '
          f'{dates.min().date()}~{dates.max().date()}, panel {len(study)} 行')
    gate_name = 'none(常开)' if gate_spec is None else args.gate_col
    gate_w = kit._resolve_gate(gate_spec, kit.open_wide)
    print(f'== 门 == {gate_name} > 0, 覆盖率 {float(gate_w.values.mean()):.1%}')
    print(f'== 组合分数 == {W}')
    print(f'== 引擎参数 == {p.brief()}')

    # ---------------- 主策略 ----------------
    print('\n== 主策略回测 ==')
    res = kit.run(W, mode='rank', name='strategy')
    print(summary_table([res]).to_string(index=False))
    res.save(out_dir)
    yr = yearly_returns(res['equity'])
    print('\n[分年收益] strategy:')
    print(yr.to_string())

    results = [res]

    # ---------------- 对照与基线(同一引擎) ----------------
    if not args.skip_baselines:
        print('\n== 对照与基线(同一引擎核算) ==')
        results.append(kit.shuffle(res))            # 组合分数按交易日整体置换
        results.append(kit.gate_equal())            # 门内全等权: 常数分数+大K
        results.append(kit.buyhold())               # 全池等权买入持有: 常数分数+常开门
        for bench in ['TQQQ', 'SPY']:               # 单标的买入持有基准(池内首个)
            if bench in kit.open_wide.columns:
                results.append(kit.buyhold_symbol(bench))
                break
        print(summary_table(results).to_string(index=False))

    # ---------------- 消融/敏感性 ----------------
    sens_results = []
    if args.sensitivity:
        print('\n== 消融/敏感性矩阵 ==')
        cfgs = [
            ('base', W, p),
            ('no_rank(去position+rsi)', {k: v for k, v in W.items()
                                         if k not in ('position_score', 'rsi')}, p),
            ('no_flip(去三个负权)', {k: v for k, v in W.items()
                                    if k not in ('trigger_score', 'boundary_score', 'candle_position_score')}, p),
            ('no_gate', W, p),
            ('equal_size', W, p.copy(sizing='equal')),
            ('linear_size', W, p.copy(sizing='linear')),
            ('k3', W, p.copy(top_k=3, exit_rank=8)),
            ('k8', W, p.copy(top_k=8, exit_rank=20)),
            ('k12', W, p.copy(top_k=12, exit_rank=28)),
            ('cost0', W, p.copy(cost_bps=0.0)),
            ('cost25', W, p.copy(cost_bps=25.0)),
            ('band01', W, p.copy(band=0.10)),
        ]
        for preset in ('rank_only', 'flip_only'):
            if dict(WEIGHT_PRESETS[preset]) != W:
                cfgs.append((f'preset:{preset}', dict(WEIGHT_PRESETS[preset]), p))
        # 持仓周期假设检验: 因子 IC 有效期为 h20~h60, 但基准引擎平均持仓仅 ~2 天.
        # 拉宽退出名次(exit_rank=28, 近似仅极端垫底才退出)+去门 -> 慢轮换, 对齐因子有效期
        cfgs += [
            ('k5wide+no_gate', W, p.copy(exit_rank=28)),
            ('k12wide+no_gate', W, p.copy(top_k=12, exit_rank=28)),
            ('rank_only+no_gate', dict(WEIGHT_PRESETS['rank_only']), p),
            ('rank_only+no_gate+wide', dict(WEIGHT_PRESETS['rank_only']), p.copy(exit_rank=28)),
            ('rank_only+no_gate+wide+cost0', dict(WEIGHT_PRESETS['rank_only']),
             p.copy(exit_rank=28, cost_bps=0.0)),
        ]
        # 经 signal_search 真伪检验通过的动态信号(见 signal_search.py --validate-top):
        # D_mom250 / D_mom120 / D_ma200_dist / adx_strength / D_sharpe120 的 dyn_minus_static>0
        # 且 n_eff_symbols>=0.7(非静态身份效应), 故按"慢轮换(exit_rank=28)+无门"回测
        if args.derived:
            for nm, wm in MOMENTUM_SETS.items():
                if nm in ('M16_kijun_ichimoku', 'M17_adx_ichimoku', 'M18_ichimoku_only'):
                    continue  # B 级组合不依赖衍生列, 由下方独立块统一追加
                cfgs.append((f'{nm}+no_gate+wide', wm, p.copy(exit_rank=28)))
            cfgs.append(('M6_mix+gate+wide', MOMENTUM_SETS['M6_mix'], p.copy(exit_rank=28)))
            cfgs.append(('M6_mix+no_gate+default', MOMENTUM_SETS['M6_mix'], p))
            # 可轮换版: 若动量组合只是"一次性静态选票", 收窄 exit_rank + 等权应无改善;
            # 若含真实动态 alpha, 轮换应能保留/提升表现
            for nm in ('M1_mom250', 'M6_mix'):
                cfgs.append((f'{nm}+no_gate+rot', MOMENTUM_SETS[nm],
                             p.copy(exit_rank=12, sizing='equal')))
            # 参数网格内的最优邻域(k=5, exit_rank=7: 滞回缓冲仅 2 档 -> 紧贴当期动量)
            for nm in ('M1_mom250', 'M2_mom120', 'M4_adxstrength', 'M6_mix'):
                cfgs.append((f'{nm}+no_gate+rot7', MOMENTUM_SETS[nm],
                             p.copy(exit_rank=7, sizing='equal')))
            # 方向性对照: 反向动量在同样参数下应显著变差(否则说明是伪信号)
            cfgs.append(('CTRL_neg_mom250+rot7', MOMENTUM_SETS['M1_neg_mom250'],
                         p.copy(exit_rank=7, sizing='equal')))
            # rank_only(前一轮最优) 叠加已验证动态信号: 检验动量/低波动是否带来增量
            cfgs += [
                ('R1_rank+mom250+no_gate+wide',
                 {'position_score': 0.30, 'rsi': 0.20, 'pattern_score_alpha': 0.15,
                  'D_mom250': 0.20}, p.copy(exit_rank=28)),
                ('R2_rank+mom+ma200+no_gate+wide',
                 {'position_score': 0.30, 'rsi': 0.20, 'pattern_score_alpha': 0.10,
                  'D_mom250': 0.20, 'D_ma200_dist': 0.20}, p.copy(exit_rank=28)),
                ('R3_rank+lowvol+no_gate+wide',
                 {'position_score': 0.30, 'rsi': 0.20, 'pattern_score_alpha': 0.15,
                  'D_vol20': -0.35}, p.copy(exit_rank=28)),
                ('R4_rank+mom+lowvol+no_gate+wide',
                 {'position_score': 0.25, 'rsi': 0.20, 'pattern_score_alpha': 0.10,
                  'D_mom250': 0.20, 'D_vol20': -0.25}, p.copy(exit_rank=28)),
            ]
        # B 级技术指标组合(indicator_value_review 8.3): 不依赖 --derived,
        # ichimoku_distance_alpha 已由 build_alpha_synths 无条件注入
        for nm in ('M16_kijun_ichimoku', 'M17_adx_ichimoku', 'M18_ichimoku_only'):
            cfgs.append((f'{nm}+no_gate+wide', MOMENTUM_SETS[nm], p.copy(exit_rank=28)))
        # 挖矿黄金组合对照(需 --mined 注入 F_* 列), 与技术指标组合同参数公平对比
        if args.mined:
            for nm in ('gold4', 'gold3'):
                cfgs.append((f'{nm}+no_gate+wide', GOLD_SETS[nm], p.copy(exit_rank=28)))
        for name, w, pp in cfgs:
            try:
                # 'no_gate' 名字变体走常开门, 其余用 kit 默认门(= --gate-col)
                r = kit.run(w, mode='rank', name=name,
                            gate='none' if 'no_gate' in name else None, engine_params=pp)
                sens_results.append(r)
                s = r['stats']
                print(f"  {name:28s} cagr={s.get('cagr')} sharpe={s.get('sharpe')} "
                      f"maxdd={s.get('max_dd')} turnover={s.get('ann_turnover')} "
                      f"trades={s.get('n_trades', 0)}")
            except Exception as e:
                print(f'  {name}: [SKIP] {e}')
        if sens_results:
            st = summary_table(sens_results)
            st.to_csv(os.path.join(out_dir, 'sensitivity.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 报告 ----------------
    lines = []
    lines.append(f'score_backtest 报告 | pool={args.pool} | run={run_id}')
    lines.append(f'研究窗口: {dates.min().date()}~{dates.max().date()}, {len(symbols)} 标的, {len(study)} 行')
    lines.append(f'门: {gate_name} > 0, 覆盖率 {float(gate_w.values.mean()):.1%}')
    lines.append(f'组合分数权重: {W}')
    lines.append(f'引擎: {p.brief()}')
    lines.append('\n== 主结果(同一引擎) ==')
    lines.append(summary_table(results).to_string(index=False))
    lines.append('\n== strategy 分年收益 ==')
    lines.append(yr.to_string())
    if len(res['trades']):
        td = res['trades']
        lines.append(f"\n== strategy 交易明细统计 == n={len(td)}, 完成={(td['completed']).sum() if 'completed' in td.columns else 'NA'}")
    if sens_results:
        lines.append('\n== 消融/敏感性 ==')
        lines.append(summary_table(sens_results).to_string(index=False))
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')
    print('  ' + ', '.join(sorted(os.listdir(out_dir))))


# ==========================================================================
# ==== 源自 bt_core.py ====
# ==========================================================================
def to_unit01(name: str, sig: pd.DataFrame) -> pd.DataFrame:
    """把信号统一到 [0,1] 尺度: 已知尺度的直接映射, 未知的因果归一化兜底.

    (自 signal_timeseries 迁入, 两处共用同一实现, 保证可视化与回测口径一致)
    """
    if name.startswith(('H_', 'P_')) and name.endswith('_alpha'):
        return sig.clip(0, 1)                       # 构造上已在 [0,1]
    if name.startswith('F_er'):
        return sig.clip(0, 1)                       # 效率比天然 [0,1]
    if name.startswith('C_'):
        return (sig / 2.0).clip(0, 1)               # 两个 rank_pct 之和, 上限 2
    # 兜底: 逐标的因果归一化(与 alpha 列同口径)
    out = {}
    for c in sig.columns:
        out[c] = normalize_causal(sig[c].astype(float), window=252, min_periods=60)
    return pd.DataFrame(out)[sig.columns]

def monthly_returns(equity: pd.Series) -> pd.Series:
    """月度收益: 月末净值/月初净值 - 1. (自 signal_backtest 迁入)"""
    if len(equity) < 2:
        return pd.Series(dtype=float)
    g = equity.groupby([equity.index.year, equity.index.month])
    out = (g.last() / g.first() - 1.0).round(4)
    out.index = [f'{y}-{m:02d}' for y, m in out.index]
    return out

class BacktestResult:
    """
    统一回测结果对象.

    属性访问 + dict 兼容(__getitem__ 委托内部 payload), 可直接传 summary_table
    等既有接收 run_config dict 的函数。
    """

    def __init__(self, payload: dict, comp: pd.DataFrame = None,
                 gate: pd.DataFrame = None, p: EngineParams = None):
        self._d = payload                     # run_config 输出: name/stats/daily/pos/trades/equity/ann_turnover
        self._comp, self._gate, self._p = comp, gate, p   # 生成材料(shuffle 对照复用)

    # ---- 字段(委托 payload) ----
    @property
    def name(self):
        return self._d['name']

    @property
    def stats(self) -> dict:
        return self._d['stats']

    @property
    def daily(self) -> pd.DataFrame:
        return self._d['daily']

    @property
    def pos(self) -> pd.DataFrame:
        return self._d['pos']

    @property
    def trades(self) -> pd.DataFrame:
        return self._d['trades']

    @property
    def equity(self) -> pd.Series:
        return self._d['equity']

    @property
    def ann_turnover(self) -> float:
        return self._d['ann_turnover']

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def __repr__(self):
        s = self.stats
        return (f"<BacktestResult '{self.name}' total={s.get('total_ret')} "
                f"sharpe={s.get('sharpe')} maxdd={s.get('max_dd')} "
                f"trades={s.get('n_trades', 0)}>")

    # ---- 派生报告 ----
    def yearly(self) -> pd.Series:
        return yearly_returns(self.equity)

    def monthly(self) -> pd.Series:
        return monthly_returns(self.equity)

    def _done_trades(self) -> pd.DataFrame:
        td = self.trades
        if len(td) and 'completed' in td.columns:
            return td[td['completed']]
        return td

    def top_trades(self, n: int = 5) -> pd.DataFrame:
        """盈利最大 n 笔(已完成)."""
        done = self._done_trades()
        if not len(done):
            return pd.DataFrame()
        cols = ['symbol', 'entry_exec', 'exit_exec', 'days', 'ret']
        return done.sort_values('ret', ascending=False)[cols].round({'ret': 4}).head(n)

    def bottom_trades(self, n: int = 5) -> pd.DataFrame:
        """亏损最大 n 笔(已完成)."""
        done = self._done_trades()
        if not len(done):
            return pd.DataFrame()
        cols = ['symbol', 'entry_exec', 'exit_exec', 'days', 'ret']
        return done.sort_values('ret', ascending=False)[cols].round({'ret': 4}).tail(n)

    # ---- 落盘 ----
    def save(self, out_dir: str) -> str:
        """落盘 3 件套: equity_curve.csv / positions.csv / trades.csv(文件名与 CLI 一致)."""
        os.makedirs(out_dir, exist_ok=True)
        self.daily.to_csv(os.path.join(out_dir, 'equity_curve.csv'), encoding='utf-8-sig')
        self.pos.to_csv(os.path.join(out_dir, 'positions.csv'), index=False, encoding='utf-8-sig')
        if len(self.trades):
            self.trades.to_csv(os.path.join(out_dir, 'trades.csv'), index=False, encoding='utf-8-sig')
        return out_dir

class BacktestKit:
    """
    通用回测套件: 一个池一个窗口一套引擎参数, 注册任意信号源后 run/compare/run_many.

    窗口语义(与 signal_backtest 一致): 价格宽表在全历史上 unstack 后截行,
    信号在全历史构建后截行; 窗口内无数据的标的以 NaN 列参与, 不占截面名次。
    """

    def __init__(self, pool: str = None, pkl_path: str = None, interval: str = 'day',
                 start: str = None, end: str = None, gate_col: str = None,
                 engine_params: EngineParams = None):
        """pool 与 pkl_path 至少给一个(直接指定 pkl 优先); start/end 为交易窗口
        (None=全历史; 信号一律全历史构建, 只截交易窗口, 无 warmup 污染);
        gate_col=None 常开(通用默认), 或 panel 列名(>0 视为开)。"""
        self.pkl_path = pkl_path or POOLS.get(pool)
        if not self.pkl_path or not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f'pkl 不存在: {self.pkl_path}(pool={pool})')
        self.pool = pool or os.path.basename(self.pkl_path)
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None
        self.gate_col = gate_col
        self.engine_params = engine_params or EngineParams()

        _, panel = load_panel(self.pkl_path, interval)
        self.panel_full = panel          # 全历史(注入后含注册信号列)
        self._inject_log = []            # [(label, 列名)] 记录已注入列(CLI 打印用)
        self._inject([('alpha复现', build_alpha_synths)])   # 与 score_backtest 一致: 无条件注入
        self._builders = []              # [(label, builder)]
        self._cands = None               # 惰性缓存: {信号名: 全历史宽表}
        self._study = None               # 惰性: 交易窗口内的 panel

        ow = self.panel_full['Open'].unstack('symbol').sort_index()
        cw = self.panel_full['Close'].unstack('symbol').sort_index()
        self.open_wide = ow.loc[self.start:self.end]
        self.close_wide = cw.loc[self.start:self.end]
        # ATR 宽表(价格单位, 供动态止损 stop_atr/trail_atr); pkl 缺列时为 None(引擎自动跳过)
        if 'atr' in self.panel_full.columns:
            self.atr_wide = (self.panel_full['atr'].unstack('symbol').sort_index()
                             .loc[self.start:self.end])
        else:
            self.atr_wide = None
        if len(self.open_wide) < 2:
            raise ValueError(f'交易窗口不足 2 个交易日: '
                             f'{self.open_wide.index.min()}~{self.open_wide.index.max()}')

    # ---------------- 信号注册 ---------------- #
    def register(self, obj, fn=None, label: str = None):
        """注册信号源, 两种形式:
          kit.register(builder)       builder(panel)->{信号名: 宽表}(如 build_alpha_cands)
          kit.register('名字', fn)     fn(panel)->宽表(单信号, 任意自定义函数)
        builder 输出会注入 panel 列(已存在则跳过), 使 rank 模式可直接引用这些信号名。"""
        if fn is None:
            self._builders.append((label or getattr(obj, '__name__', str(obj)), obj))
        else:
            wrap = lambda panel, _fn=fn, _n=obj: {_n: _fn(panel)}  # noqa: E731
            self._builders.append((label or str(obj), wrap))
        self._cands = None   # 失效缓存
        self._study = None   # 新列注入后窗口 study 需重建
        return self

    def _inject(self, injectors: list):
        """在完整历史上计算信号再按 panel 行序注入, 避免错位(score_backtest 同口径)。
        已存在列跳过(不覆盖), 全 NaN 列不注入; 注入结果记入 _inject_log。"""
        panel = self.panel_full
        mi = panel.index
        key = pd.MultiIndex.from_arrays([mi.get_level_values('date'), mi.get_level_values('symbol')],
                                        names=['date', 'symbol'])
        for label, builder in injectors:
            for name, wide in builder(panel).items():
                if name in panel.columns:
                    continue
                vals = wide.stack().reindex(key).to_numpy(dtype=float)
                if not np.isnan(vals).all():
                    panel = panel.assign(**{name: vals})
                    self._inject_log.append((label, name))
        self.panel_full = panel

    @property
    def study(self) -> pd.DataFrame:
        """交易窗口内的 panel(rank 模式成分列的来源)。"""
        if self._study is None:
            st = self.panel_full
            if self.start is not None:
                st = st[st.index.get_level_values('date') >= self.start]
            if self.end is not None:
                st = st[st.index.get_level_values('date') <= self.end]
            self._study = st
        return self._study

    def _get_cands(self) -> dict:
        """构建并缓存全部注册信号(全历史, 惰性; 同时注入 panel 列)。"""
        if self._cands is None:
            inject, cands = [], {}
            for label, builder in self._builders:
                out = builder(self.panel_full)
                for name, wide in out.items():
                    if name not in cands:
                        cands[name] = wide
                inject.append((label, lambda p, _o=out: _o))
            if inject:
                self._inject(inject)
                self._study = None
            self._cands = cands
        return self._cands

    # ---------------- 分数构造 ---------------- #
    def _composite(self, spec, mode: str) -> pd.DataFrame:
        """spec → 全历史/窗口组合分数宽表。"""
        if isinstance(spec, pd.DataFrame):
            return spec                                        # 现成宽表: 原样(截行在 run 统一做)
        if isinstance(spec, str):
            if mode == 'unit01':
                cands = self._get_cands()
                if spec not in cands:
                    raise KeyError(f'未注册信号: {spec}(已注册: {sorted(cands)})')
                return to_unit01(spec, cands[spec])
            return self._rank_composite({spec: 1.0})           # rank 口径单信号 = 截面 rank_pct
        if isinstance(spec, (list, tuple)):
            spec = {s: 1.0 for s in spec}
        if not isinstance(spec, dict) or not spec:
            raise TypeError(f'run 只接受 str/list/dict/DataFrame, 收到: {type(spec)}')
        if mode == 'unit01':
            # 汇总信号口径: 分量归一化到 [0,1] 后加权 skipna 均值 —— 逐格 Σ wᵢxᵢ / Σ|wᵢ|
            # (有效分量)。等权时与 build_symbol_view 的 groupby.mean(skipna) 逐位一致:
            # 某分量缺失当日不参与, 全分量缺失才 NaN; 负权 = 反向分量(不做 clip)。
            cands = self._get_cands()
            missing = [k for k in spec if k not in cands]
            if missing:
                raise KeyError(f'未注册信号: {missing}(已注册: {sorted(cands)})')
            num = den = None
            for k, w in spec.items():
                u = to_unit01(k, cands[k])
                v = u.fillna(0.0) * w
                d = u.notna().astype(float) * abs(w)
                num = v if num is None else num.add(v, fill_value=0.0)
                den = d if den is None else den.add(d, fill_value=0.0)
            return num / den.replace(0.0, np.nan)
        return self._rank_composite(spec)

    def _rank_composite(self, weights: dict) -> pd.DataFrame:
        """rank 口径: 成分全部在窗口 panel 列时逐位走 compute_composite(与 score_backtest
        完全一致); 否则成分可来自注册信号(公式相同, 成分源更宽)。"""
        if all(k in self.study.columns for k in weights):
            return compute_composite(self.study, weights)
        cands = self._get_cands()
        total = float(sum(abs(w) for w in weights.values()))
        if total <= 0:
            raise ValueError('权重绝对值和为 0')
        comp = None
        for col, w in weights.items():
            if col in self.panel_full.columns:
                wide = pd.to_numeric(self.panel_full[col], errors='coerce') \
                              .unstack('symbol').sort_index()
            elif col in cands:
                wide = cands[col].astype(float)
            else:
                raise KeyError(f'组合分数成分不存在(既非 panel 列也非注册信号): {col}')
            pct = wide.rank(axis=1, pct=True)                  # 当日截面, NaN 保留
            part = (w / total) * pct
            comp = part if comp is None else comp.add(part, fill_value=0.0)
        return comp.fillna(0.5)

    def _resolve_gate(self, gate, like: pd.DataFrame) -> pd.DataFrame:
        """gate 参数 → 布尔宽表: None=kit 默认 / 'none'=常开 / 列名(>0 开) / DataFrame。"""
        g = self.gate_col if gate is None else gate
        if g is None or (isinstance(g, str) and g.lower() == 'none'):
            return pd.DataFrame(True, index=like.index, columns=like.columns)
        if isinstance(g, pd.DataFrame):
            return g.reindex(index=like.index, columns=like.columns).fillna(False)
        if g not in self.panel_full.columns:
            raise KeyError(f'门列不存在: {g}')
        gw = pd.to_numeric(self.panel_full[g], errors='coerce').unstack('symbol').sort_index() > 0
        return gw.reindex(index=like.index, columns=like.columns).fillna(False)

    @staticmethod
    def _auto_name(spec, mode: str) -> str:
        if isinstance(spec, pd.DataFrame):
            return 'wide(现成宽表)'
        if isinstance(spec, str):
            return spec
        keys = list(spec.keys()) if isinstance(spec, dict) else list(spec)
        tag = 'agg' if mode == 'unit01' else 'rank'
        return f'{tag}({",".join(keys)})'[:80]

    # ---------------- 核心执行 ---------------- #
    def run(self, spec, mode: str = 'rank', name: str = None, gate=None,
            engine_params: EngineParams = None) -> BacktestResult:
        """跑一个配置, 返回 BacktestResult.

        spec: 信号名 str / 权重 dict / 信号名 list(等权) / 现成分数宽表 DataFrame
        mode: 'rank'(默认, score_backtest 口径) 或 'unit01'(汇总信号口径)
        gate: None=kit 默认 / 'none'=常开 / panel 列名 / 布尔 DataFrame
        """
        comp = self._composite(spec, mode)
        comp = comp.reindex(index=self.open_wide.index, columns=self.open_wide.columns)
        g = self._resolve_gate(gate, comp)
        p = engine_params or self.engine_params
        payload = run_config(name or self._auto_name(spec, mode),
                             self.open_wide, self.close_wide, comp, g, p,
                             atr_wide=self.atr_wide)
        return BacktestResult(payload, comp=comp, gate=g, p=p)

    def run_many(self, specs: list, mode: str = 'rank', gate=None,
                 engine_params: EngineParams = None) -> list:
        """批量: specs 为 spec 列表(或 (name, spec) 元组), 返回 BacktestResult 列表。"""
        out = []
        for item in specs:
            if isinstance(item, tuple) and len(item) == 2 and not isinstance(item[0], (int, float)):
                nm, sp = item
            else:
                nm, sp = None, item
            out.append(self.run(sp, mode=mode, name=nm, gate=gate,
                                engine_params=engine_params))
        return out

    # ---------------- 内置对照(同一引擎, 公平核算) ---------------- #
    def shuffle(self, base, seed: int = 42, name: str = 'shuffle对照',
                engine_params: EngineParams = None) -> BacktestResult:
        """时序置换对照: 接收 run 的结果或分数宽表, 破坏分数-时间对齐(保留截面分布)。"""
        comp = base._comp if isinstance(base, BacktestResult) else base
        comp_shuf = shuffled_composite(comp, seed)
        g = base._gate if isinstance(base, BacktestResult) else self._resolve_gate(None, comp)
        p = engine_params or (base._p if isinstance(base, BacktestResult) else self.engine_params)
        payload = run_config(name, self.open_wide, self.close_wide,
                             comp_shuf.reindex(index=self.open_wide.index,
                                               columns=self.open_wide.columns), g, p,
                             atr_wide=self.atr_wide)
        return BacktestResult(payload, comp=comp, gate=g, p=p)

    def _const_comp(self) -> pd.DataFrame:
        return pd.DataFrame(0.5, index=self.open_wide.index, columns=self.open_wide.columns)

    def buyhold(self, name: str = 'buyhold_pool(等权持有)',
                engine_params: EngineParams = None) -> BacktestResult:
        """全池等权买入持有: 常数分数 + 常开门 + 大K 等权。"""
        comp = self._const_comp()
        g = pd.DataFrame(True, index=comp.index, columns=comp.columns)
        p_ge = (engine_params or self.engine_params).copy(top_k=9999, exit_rank=99999, sizing='equal')
        payload = run_config(name, self.open_wide, self.close_wide, comp, g, p_ge,
                             atr_wide=self.atr_wide)
        return BacktestResult(payload, comp=comp, gate=g, p=p_ge)

    def gate_equal(self, name: str = 'gate_equal(门内等权)',
                   engine_params: EngineParams = None) -> BacktestResult:
        """门内全等权(近似无分数行为): 常数分数 + kit 默认门 + 大K 等权。"""
        comp = self._const_comp()
        g = self._resolve_gate(None, comp)
        p_ge = (engine_params or self.engine_params).copy(top_k=9999, exit_rank=99999, sizing='equal')
        payload = run_config(name, self.open_wide, self.close_wide, comp, g, p_ge,
                             atr_wide=self.atr_wide)
        return BacktestResult(payload, comp=comp, gate=g, p=p_ge)

    def buyhold_symbol(self, sym: str, name: str = None,
                       engine_params: EngineParams = None) -> BacktestResult:
        """单标的买入持有(如 TQQQ/SPY 基准)。"""
        if sym not in self.open_wide.columns:
            raise KeyError(f'{sym} 不在池 {self.pool} 中')
        ob, cb = self.open_wide[[sym]], self.close_wide[[sym]]
        comp = self._const_comp()[[sym]]
        g = pd.DataFrame(True, index=comp.index, columns=[sym])
        p1 = (engine_params or self.engine_params).copy(top_k=1, exit_rank=2,
                                                        sizing='equal', per_symbol_cap=1.0)
        aw = self.atr_wide[[sym]] if (self.atr_wide is not None
                                      and sym in self.atr_wide.columns) else None
        payload = run_config(name or f'buyhold_{sym}', ob, cb, comp, g, p1,
                             atr_wide=aw)
        return BacktestResult(payload, comp=comp, gate=g, p=p1)

    # ---------------- 汇总 ---------------- #
    def compare(self, results: list, print_table: bool = True) -> pd.DataFrame:
        """对照汇总表(BacktestResult 与 run_config dict 混合均可)。"""
        t = summary_table(results)
        if print_table:
            print(t.to_string(index=False))
        return t

    def register_defaults(self):
        """注册常用信号源(等价于显式 register 三个 builder)。"""
        from quant.bc_factor_search import build_alpha_cands
        from quant.bc_factor_search import build_derived
        try:
            from quant.bc_factor_search import build_mined
        except Exception:
            build_mined = None
        self.register(build_alpha_cands, label='alpha候选')
        self.register(build_derived, label='衍生信号')
        if build_mined is not None:
            self.register(build_mined, label='挖矿信号')
        return self


# ==========================================================================
# ==== 源自 signal_timeseries.py ====  
# ==== 改名: main->sts_main
# ==========================================================================
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_SIGNALS = ['H_ichimoku_alpha', 'H_trendmag_alpha', 'C_tmqmom', 'F_er20']
A_POOL_PRESETS = {
    'hs300':     {'weights': {'N_range20': -1.0, 'D_ma20_dist': -1.0, 'C_tmqmom': 0.5}, 'top_k': 5},
    'a_etf_all': {'weights': {'N_range20': 1.0, 'G_oviv20': 0.5}, 'top_k': 5},
}

STATE_CN = {'buy': '买入', 'hold': '持有', 'sell': '卖出'}
STATE_COLOR = {'buy': 'green', 'hold': 'orange', 'sell': 'red'}

def parse_weights(text: str) -> dict:
    """
    解析 "sig:w,sig:w" 组合权重(与 signal_bridge.parse_weights 同格式, 负权如 N_range20:-1).
    """
    out = {}
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        name, _, w = tok.rpartition(':')
        if not name:
            raise SystemExit(f'权重格式错误: "{tok}"(期望 "sig:w", 如 N_range20:-1,C_tmqmom:0.5)')
        out[name.strip()] = float(w)
    return out

def build_symbol_view(pool: str, pkl: str, signals: list, weights: dict = None):
    """
    返回: open 宽表, close 宽表, {信号: 宽表}, 汇总信号宽表, 截面排名宽表, 权重 dict.

    weights 非空 → A 股池口径(build_a_combo_cands + 桥的 rank_pct 加权合成, 与
    signal_bridge/compute_composite 逐位一致); 否则美股口径(等权均值)。
    """
    from quant.bc_combo_search import build_a_combo_cands
    _, panel = load_panel(pkl, 'day')
    open_w = panel['Open'].unstack('symbol').sort_index()
    close = panel['Close'].unstack('symbol').sort_index()

    if weights:
        # ---- A 股池: 照搬 signal_bridge 的合成逻辑(逐行一致, 保证图上复现桥信号) ----
        cands = build_a_combo_cands(panel)
        missing = [s for s in weights if s not in cands]
        if missing:
            raise SystemExit(f'候选中不存在信号: {missing}; 可用: {sorted(cands)}')
        total = float(sum(abs(w) for w in weights.values()))
        comp = None
        for s, w in weights.items():
            pct = cands[s].rank(axis=1, pct=True)  # 当日截面百分位, NaN 保留
            part = (w / total) * pct
            comp = part if comp is None else comp.add(part, fill_value=0.0)  # 部分缺失以 0 参与
        comps = {s: cands[s].rank(axis=1, pct=True) for s in weights}       # 分量图 = 截面百分位
        agg = comp.fillna(0.5)                    # 全成分缺失 -> 中性 0.5(桥口径)
        rank = agg.rank(axis=1, ascending=False, method='first')            # 1 = 最强
        return open_w, close, comps, agg, rank, weights

    cands = build_alpha_cands(panel)                # 全历史构建(口径与挖掘一致)
    missing = [s for s in signals if s not in cands]
    if missing:
        raise SystemExit(f'候选中不存在信号: {missing}\n可用: {sorted(cands)}')

    comps = {s: to_unit01(s, cands[s]) for s in signals}
    comp_df = pd.concat(comps, axis=1)              # 列为 (signal, symbol) 两级 MultiIndex
    agg = comp_df.T.groupby(level=1).mean().T       # 汇总: 逐 symbol 对分量等权均值 → date×symbol
    agg = agg.clip(0, 1)
    rank = agg.rank(axis=1, ascending=False)        # 1 = 最强
    return open_w, close, comps, agg, rank, None

def classify_state(rank_row: pd.Series, top_k: int, exit_rank: int) -> pd.Series:
    """
    三态: rank<=top_k 买入 / rank<=exit_rank 持有 / 其余 卖出.
    """
    s = pd.Series('sell', index=rank_row.index, dtype=object)
    s[rank_row.notna() & (rank_row <= top_k)] = 'buy'
    s[rank_row.notna() & (rank_row > top_k) & (rank_row <= exit_rank)] = 'hold'
    s[rank_row.isna()] = np.nan
    return s

def plot_symbol(sym: str, pool: str, out_png: str, close: pd.DataFrame, comps: dict, agg: pd.DataFrame, rank: pd.DataFrame, top_k: int, exit_rank: int, box_size: int, weights: dict = None,  trades: pd.DataFrame = None):
    
    # 显示窗口取整个截取后 close 的首尾(而非逐标的 dropna): 数据滞后标的的价格止于中途,
    # 但其信号/排名(含桥的 0.5 中性幻影)仍持续到窗口末, 必须在同一窗口下才能显形
    start, end = close.index[0], close.index[-1]
    agg_s = agg[sym].loc[start:end]
    rank_s = rank[sym].loc[start:end]
    close_s = close[sym].loc[start:end]              # 原始: NaN 处价格线断开(诚实显示数据断更)
    close_ff = close_s.ffill()                       # 方框/标注 y 基准: 断更日沿用最近有效价
    state = classify_state(rank_s, top_k, exit_rank)

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    ax0, ax1, ax2 = axes
    fig.subplots_adjust(hspace=0.06, right=0.94, top=0.92, bottom=0.06)

    # ---- 行1: 价格 + 三态方框(边框=状态, 填充=汇总信号) ----
    ax0.plot(close_s.index, close_s.values, color='#333333', lw=1.0, alpha=0.9, label='Close')
    for st, color in STATE_COLOR.items():
        idx = state.index[state == st]
        if len(idx) == 0:
            continue
        a = agg_s.reindex(idx).fillna(0).values     # 填充 alpha = 归一化汇总信号
        base = to_rgba(color)
        fc = np.zeros((len(idx), 4))
        fc[:, :3] = base[:3]
        fc[:, 3] = np.clip(a, 0, 1)
        ax0.scatter(idx, close_ff.reindex(idx).values, marker='s', s=box_size,
                    facecolor=fc, edgecolor=(base[0], base[1], base[2], 0.9),
                    linewidths=1.3, zorder=5)

    # 右侧现状标注(参考 plot_signal 的写法)
    if not pd.isna(state.iloc[-1]) and not pd.isna(agg_s.iloc[-1]):
        st, av, rv = state.iloc[-1], agg_s.iloc[-1], rank_s.iloc[-1]
        color = STATE_COLOR[st]
        x_txt = end + pd.Timedelta(days=3)
        y_txt = close_ff.iloc[-1] if not pd.isna(close_ff.iloc[-1]) else close_s.dropna().iloc[-1]
        ax0.annotate(f'{STATE_CN[st]} {av:.2f} | 排名{rv:.0f}',
                     xy=(x_txt, y_txt), xytext=(x_txt, y_txt),
                     fontsize=10, color=color, va='center', ha='left')

    handles = [Patch(facecolor='none', edgecolor=c, label=STATE_CN[st])
               for st, c in STATE_COLOR.items()]
    handles.append(Patch(facecolor='grey', edgecolor='grey', alpha=0.4, label='填充亮度=汇总信号'))

    # ---- 行1 附加: 回测交易进出场标注(--trades-csv) ----
    n_trades_sym = 0
    if trades is not None and len(trades):
        for _, tr in trades.iterrows():
            e_dt = pd.Timestamp(tr['entry_exec'])
            if e_dt > end:                       # 整段在显示窗口之后
                continue
            x_dt = pd.Timestamp(tr['exit_exec']) if pd.notna(tr['exit_exec']) else None
            ret_v = float(tr['ret']) if pd.notna(tr['ret']) else np.nan
            is_open = (str(tr.get('exit_reason', '')) == 'open') or (not bool(tr.get('completed', True)))
            e_px = float(tr['entry_px'])
            x_px = float(tr['exit_px']) if pd.notna(tr.get('exit_px')) else None
            x_end = x_dt if x_dt is not None else end    # 未平仓段画到窗口末
            # 持有区间底色: 盈绿亏红, 未平仓灰
            span_c = 'grey' if is_open else ('green' if ret_v > 0 else 'red')
            ax0.axvspan(max(e_dt, start), min(x_end, end), color=span_c, alpha=0.10, zorder=1)
            n_trades_sym += 1
            if start <= e_dt <= end:              # 进场 ▲(标注日期)
                ax0.scatter([e_dt], [e_px], marker='^', s=90, color='limegreen',
                            edgecolor='darkgreen', linewidths=0.8, zorder=8)
                ax0.annotate(f'{e_dt:%m-%d}进', xy=(e_dt, e_px),
                             xytext=(0, 7), textcoords='offset points',
                             fontsize=7, color='green', ha='center', va='bottom', zorder=8)
            if x_dt is not None and start <= x_dt <= end and x_px is not None:
                if is_open:                      # 期末未平仓: 橙色▼ 盯市值
                    ax0.scatter([x_dt], [x_px], marker='v', s=90, facecolor='none',
                                edgecolor='darkorange', linewidths=1.6, zorder=8)
                    ax0.annotate(f'{x_dt:%m-%d}持有中(盯市{ret_v:+.1%})', xy=(x_dt, x_px),
                                 xytext=(0, -10), textcoords='offset points',
                                 fontsize=7, color='darkorange', ha='center', va='top', zorder=8)
                else:                             # 已平仓: 红色▼ 标注收益
                    ax0.scatter([x_dt], [x_px], marker='v', s=90, color='red',
                                edgecolor='darkred', linewidths=0.8, zorder=8)
                    ax0.annotate(f'{x_dt:%m-%d}出 {ret_v:+.1%}', xy=(x_dt, x_px),
                                 xytext=(0, -10), textcoords='offset points',
                                 fontsize=7, color='red', ha='center', va='top', zorder=8)
        if n_trades_sym:
            handles += [Line2D([], [], marker='^', color='limegreen', ls='none',
                               markersize=7, label='进场(开盘)'),
                        Line2D([], [], marker='v', color='red', ls='none',
                               markersize=7, label='出场(开盘,标注收益)'),
                        Line2D([], [], marker='v', color='darkorange', markerfacecolor='none',
                               ls='none', markersize=7, label='持有中(期末盯市)'),
                        Patch(facecolor='green', alpha=0.15, label='持有区间:盈利'),
                        Patch(facecolor='red', alpha=0.15, label='持有区间:亏损')]

    ax0.legend(handles=handles, loc='upper left', fontsize=9, ncol=4, framealpha=0.5)
    if weights:
        wdesc = ','.join(f'{k}:{v:g}' for k, v in sorted(weights.items(), key=lambda kv: -abs(kv[1])))
        ax0.set_title(f'{sym} ({pool}) 桥口径信号时序与三态 | 权重 {wdesc} | 买入≤{top_k} / '
                      f'持有≤{exit_rank} / 卖出>{exit_rank}(截面排名, 与 signal_bridge 一致; '
                      f'价格断更日方框沿用最近有效价)', fontsize=12)
    else:
        ax0.set_title(f'{sym} ({pool}) 推荐信号时序与三态 | 买入≤{top_k} / 持有≤{exit_rank} / 卖出>{exit_rank} '
                      f'(截面排名, 与 score_backtest 口径一致)', fontsize=12)
    ax0.set_ylabel('Close')
    ax0.grid(True, alpha=0.25, lw=0.4)

    # ---- 行2: 分量信号 + 汇总 ----
    for name, comp in comps.items():
        col = comp[sym].loc[start:end]
        label = f'{name}({weights[name]:g})' if weights and name in weights else name
        ax1.plot(col.index, col.values, lw=0.9, alpha=0.55, label=label)
    ax1.plot(agg_s.index, agg_s.values, color='black', lw=1.8, label='汇总信号')
    if weights:                                     # 负权组合的 agg 可为负, 动态范围
        lo = min(agg_s.min(), -0.05) if len(agg_s) else -0.05
        hi = max(agg_s.max(), 1.0) if len(agg_s) else 1.0
        ax1.set_ylim(lo - 0.05, hi + 0.05)
        ax1.set_ylabel('信号值(截面百分位)')
    else:
        ax1.set_ylim(-0.03, 1.03)
        ax1.set_ylabel('信号值 [0,1]')
    ax1.legend(loc='upper left', fontsize=8, ncol=5, framealpha=0.5)
    ax1.grid(True, alpha=0.25, lw=0.4)

    # ---- 行3: 截面排名(1=最强, 倒置) ----
    rank_max = rank.max(axis=1).max()
    if not np.isfinite(rank_max):
        rank_max = exit_rank + 10
    ax2.axhspan(0.5, top_k + 0.5, color='green', alpha=0.10)
    ax2.axhspan(top_k + 0.5, exit_rank + 0.5, color='orange', alpha=0.10)
    ax2.axhspan(exit_rank + 0.5, rank_max + 2, color='red', alpha=0.08)
    ax2.plot(rank_s.index, rank_s.values, color='#1f77b4', lw=1.0, marker='.',
             markersize=3, alpha=0.8)
    ax2.axhline(top_k, color='green', lw=0.8, ls='--', alpha=0.6)
    ax2.axhline(exit_rank, color='orange', lw=0.8, ls='--', alpha=0.6)
    ax2.invert_yaxis()
    ax2.set_ylabel('池内排名(倒置)')
    ax2.grid(True, alpha=0.25, lw=0.4)
    ax2.set_xlabel('日期')

    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return state, agg_s, rank_s

def sts_main():
    ap = argparse.ArgumentParser(description='单标的信号时序+汇总信号可视化(只读)')
    ap.add_argument('--pool', default='etf_3x', help='池名(美股池或 A 股池 hs300/a_etf_all)')
    ap.add_argument('--symbols', default=None, help='逗号分隔标的列表(默认: 全池所有标的)')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl(默认取 alpha_mining.POOLS; A 股池建议指生产版)')
    ap.add_argument('--signals', default=','.join(DEFAULT_SIGNALS),
                    help='逗号分隔的分量信号(A 股池忽略此参数, 默认取池预设权重)')
    ap.add_argument('--weights', default=None,
                    help='A 股池组合权重 "sig:w,sig:w"(默认取池预设; 如 N_range20:-1,C_tmqmom:0.5)')
    ap.add_argument('--start', default=None, help='显示窗口起点(默认最近252个交易日)')
    ap.add_argument('--end', default=None, help='显示窗口终点')
    ap.add_argument('--top-k', type=int, default=None, help='买入排名阈值(默认: A 股池取预设, 其余 5)')
    ap.add_argument('--exit-rank', type=int, default=12, help='卖出排名阈值(持有带上沿)')
    ap.add_argument('--box-size', type=int, default=34, help='方框大小(scatter s)')
    ap.add_argument('--trades-csv', default=None,
                    help='回测 trades 明细 CSV: 在价格图标注每笔进出场(不给 --symbols 时默认只画交易过的标的)')
    a = ap.parse_args()

    # ---- A 股池: 权重/top_k 默认取池预设(--weights/--top-k 显式覆盖) ----
    is_a_pool = a.pool in A_POOL_PRESETS
    preset = A_POOL_PRESETS.get(a.pool, {})
    if is_a_pool:
        weights = parse_weights(a.weights) if a.weights else dict(preset.get('weights', {}))
        if not weights:
            raise SystemExit(f'A 股池 {a.pool} 无预设权重, 请用 --weights 显式指定(如 N_range20:-1,C_tmqmom:0.5)')
    else:
        weights = None
    top_k = a.top_k if a.top_k is not None else preset.get('top_k', 5)

    pkl = a.pkl_path or POOLS.get(a.pool)
    if not pkl or not os.path.exists(pkl):
        raise SystemExit(f'pkl 不存在: {pkl}(pool={a.pool}); 用 --pkl-path 指定, '
                         f'建议 research 版: {os.path.join(os.path.expanduser("~"), "quant", a.pool + "_day_ta_data_research.pkl")}')
    signals = [s.strip() for s in a.signals.split(',') if s.strip()]

    print(f'加载数据: {pkl}')
    open_all, close_all, comps, agg, rank, weights = build_symbol_view(a.pool, pkl, signals, weights)
    if weights:
        print(f'A 股池口径: {weights} (rank_pct 加权 + 全缺失 fillna 0.5, 与 signal_bridge 一致)')

    # ---- 回测交易标注: 读 trades CSV, 按 symbol 分组 ----
    trades_by_sym = None
    if a.trades_csv:
        if not os.path.exists(a.trades_csv):
            raise SystemExit(f'trades CSV 不存在: {a.trades_csv}')
        tf = pd.read_csv(a.trades_csv, parse_dates=['entry_exec', 'exit_exec'],
                         dtype={'symbol': str})   # 防 000338 被推断成 int 338
        if tf.empty:
            raise SystemExit(f'trades CSV 为空: {a.trades_csv}')
        trades_by_sym = {sym: g.sort_values('entry_exec') for sym, g in tf.groupby('symbol')}
        print(f'交易标注: {a.trades_csv} ({len(tf)} 笔 / {len(trades_by_sym)} 只标的)')

    # 标的列表: 指定 --symbols 用之, 否则全池; 给了 --trades-csv 且未指定时默认只画交易过的
    if a.symbols:
        symbols = [s.strip().upper() for s in a.symbols.split(',') if s.strip()]
    elif trades_by_sym is not None:
        symbols = list(trades_by_sym.keys())
    else:
        symbols = list(close_all.columns)
    print(f'标的数: {len(symbols)}' + ('(全池)' if not a.symbols else ''))

    # 显示窗口: 默认最近 252 个交易日
    if a.end:
        close_all = close_all.loc[:a.end]
    if a.start:
        close_all = close_all.loc[a.start:]
    else:
        close_all = close_all.iloc[-252:]

    out_dir = os.path.join(HERE, 'output', f'signal_ts_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'输出目录: {out_dir}\n')

    for sym in symbols:
        if sym not in close_all.columns:
            raise SystemExit(f'{sym} 不在池 {a.pool} 中。可用标的: {sorted(close_all.columns)}')
        out_png = os.path.join(out_dir, f'{a.pool}_{sym}.png')
        sym_trades = trades_by_sym.get(sym) if trades_by_sym is not None else None
        state, agg_s, rank_s = plot_symbol(sym, a.pool, out_png, close_all, comps, agg, rank,
                                           top_k, a.exit_rank, a.box_size, weights,
                                           trades=sym_trades)

        # 控制台摘要: 取显示窗口最后一日(而非逐标的最后有效日)——A 股池数据滞后标的的
        # 信号(桥的 0.5 中性幻影)正是出现在窗口末, 需按窗口末日取状态才可见
        last_dt = close_all.index[-1]
        st = state.loc[last_dt] if last_dt in state.index else np.nan
        av = agg_s.loc[last_dt] if last_dt in agg_s.index else np.nan
        rv = rank_s.loc[last_dt] if last_dt in rank_s.index else np.nan
        print(f'== {sym} @ {last_dt.date()} ==')
        if sym_trades is not None and len(sym_trades):
            done = sym_trades[sym_trades['completed']] if 'completed' in sym_trades.columns else sym_trades
            tot = float(sym_trades['ret'].sum()) if 'ret' in sym_trades.columns else np.nan
            wr = float((done['ret'] > 0).mean()) if len(done) else np.nan
            print(f'  回测交易: {len(sym_trades)} 笔(平仓 {len(done)}, 胜率 {wr:.0%}),'
                  f' 逐笔收益合计 {tot:+.1%}(单利口径)')
        if pd.isna(st):
            print('  状态: 数据不足(NaN)')
        else:
            print(f'  状态: {STATE_CN[st]}  汇总信号: {av:.2f}  池内排名: {rv:.0f}')
            for name, comp in comps.items():
                v = comp[sym].loc[last_dt] if last_dt in comp.index else np.nan
                w_desc = f'({weights[name]:g})' if weights and name in weights else ''
                print(f'    {name:<20s} {v:6.2f} {w_desc}')
            # 最近状态切换
            chg = state[state.ne(state.shift()) & state.notna()]
            recent = chg.tail(6)
            if len(recent) > 1:
                print('  最近状态切换: ' + ', '.join(
                    f'{d.date()} {STATE_CN[s]}' for d, s in recent.items()))
        print(f'  图: {out_png}\n')


# ==========================================================================
# ==== 源自 signal_backtest.py ====  改名: main->sbt_main
# ==========================================================================
def sbt_main():
    ap = argparse.ArgumentParser(description='汇总信号组合回测(与 signal_timeseries 可视化同口径): 交易明细+回报率')
    ap.add_argument('--pool', default='etf_3x', help='池名, 默认 etf_3x')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl(默认取 alpha_mining.POOLS)')
    ap.add_argument('--signals', default=','.join(DEFAULT_SIGNALS), help='逗号分隔的分量信号')
    ap.add_argument('--start', default='2026-01-01', help='交易窗口起点(信号仍全历史构建)')
    ap.add_argument('--end', default=None, help='交易窗口终点')
    ap.add_argument('--top-k', type=int, default=5, help='买入排名阈值')
    ap.add_argument('--exit-rank', type=int, default=12, help='退出名次阈值(>该名次退出)')
    ap.add_argument('--sizing', default='equal', choices=['tier', 'linear', 'equal'],
                    help='已选持仓内仓位方式(默认等权)')
    ap.add_argument('--max-exposure', type=float, default=1.0)
    ap.add_argument('--per-symbol-cap', type=float, default=0.30)
    ap.add_argument('--cost-bps', type=float, default=10.0, help='单边成本(bps)')
    ap.add_argument('--band', type=float, default=0.05, help='重平衡带宽')
    ap.add_argument('--skip-baselines', action='store_true', help='不跑 shuffle/buyhold 对照')
    a = ap.parse_args()

    pkl = a.pkl_path or POOLS.get(a.pool)
    if not pkl or not os.path.exists(pkl):
        raise SystemExit(f'pkl 不存在: {pkl}(pool={a.pool})')
    signals = [s.strip() for s in a.signals.split(',') if s.strip()]

    print(f'加载数据: {pkl}')
    p = EngineParams(a.top_k, a.exit_rank, a.sizing, a.max_exposure,
                     a.per_symbol_cap, a.cost_bps, a.band)
    try:
        kit = BacktestKit(pool=a.pool, pkl_path=pkl, start=a.start, end=a.end,
                          engine_params=p)
    except ValueError as e:      # 窗口不足等
        raise SystemExit(str(e))
    kit.register(build_alpha_cands, label='alpha候选')

    print(f'== 回测 == {a.pool}: {kit.open_wide.index[0].date()}~{kit.open_wide.index[-1].date()}, '
          f'{kit.open_wide.shape[1]} 标的, 门=常开, {p.brief()}')
    print(f'== 汇总信号 == 分量: {signals}(归一化后等权均值)')

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output', f'signal_bt_{a.pool}_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    try:
        res = kit.run(signals, mode='unit01', name='agg_signal(汇总信号)')
    except KeyError as e:       # 分量信号未注册
        raise SystemExit(str(e))
    results = [res]
    if not a.skip_baselines:
        results.append(kit.shuffle(res))
        results.append(kit.buyhold())

    print('\n== 绩效汇总 ==')
    kit.compare(results)

    yr, mr = res.yearly(), res.monthly()
    print('\n[分年收益] agg_signal:\n' + yr.to_string())
    print('\n[月度收益] agg_signal:\n' + mr.to_string())

    # 交易明细摘要
    td = res.trades
    done = td[td['completed']] if (len(td) and 'completed' in td.columns) else td
    top5, bot5 = res.top_trades(5), res.bottom_trades(5)
    if len(done):
        print(f'\n[交易明细] 共 {len(td)} 笔(已完成 {len(done)}), 全量见 trades.csv')
        print('盈利最大 5 笔:\n' + top5.to_string(index=False))
        print('亏损最大 5 笔:\n' + bot5.to_string(index=False))

    # 落盘
    res.save(out_dir)

    lines = [f'signal_backtest 报告 | pool={a.pool} | run={run_id}',
             f'pkl: {pkl}',
             f'交易窗口: {kit.open_wide.index[0].date()}~{kit.open_wide.index[-1].date()}, '
             f'{kit.open_wide.shape[1]} 标的',
             f'汇总信号分量: {signals}(归一化后等权均值)',
             '门: 常开(与 signal_timeseries 三态口径一致, 无门)',
             f'引擎: {p.brief()}',
             '', '== 绩效汇总 ==', kit.compare(results, print_table=False).to_string(index=False),
             '', '== agg_signal 分年收益 ==', yr.to_string(),
             '', '== agg_signal 月度收益 ==', mr.to_string()]
    if len(done):
        lines += ['', f'== 交易明细 == n={len(td)}, 已完成 {len(done)}, '
                  f'胜率 {float((done["ret"] > 0).mean()):.1%}, '
                  f'平均收益 {float(done["ret"].mean()):.2%}, '
                  f'平均持仓 {float(done["days"].mean()):.1f} 日']
        lines += ['', '盈利最大 5 笔:', top5.to_string(index=False),
                  '', '亏损最大 5 笔:', bot5.to_string(index=False),
                  '', '全量明细: trades.csv']
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')
    print('  ' + ', '.join(sorted(os.listdir(out_dir))))


# ==========================================================================
# ==== 源自 stop_loss_eval.py ====  
# ==== 改名: main->sle_main
# ==========================================================================
SPEC = {
    'H_ichimoku_alpha': 0.25, 
    'H_trendmag_alpha': 0.25, 
    'C_tmqmom': 0.25, 
    'F_er20': 0.25
}

VARIANTS = [
    ('baseline', {}),
    ('SL-5%', {'stop_loss': 0.05}),
    ('SL-8%', {'stop_loss': 0.08}),
    ('SL-10%', {'stop_loss': 0.10}),
    ('TP-15%', {'take_profit': 0.15}),
    ('SL8+TP15', {'stop_loss': 0.08, 'take_profit': 0.15}),
    ('SL5+TP10', {'stop_loss': 0.05, 'take_profit': 0.10}),
    ('TRAIL-15%', {'trail_stop': 0.15}),
    ('TRAIL-20%', {'trail_stop': 0.20}),
    ('TRAIL-25%', {'trail_stop': 0.25}),
    ('ATR-SL1.5', {'stop_atr': 1.5}),
    ('ATR-SL2.0', {'stop_atr': 2.0}),
    ('ATR-SL2.5', {'stop_atr': 2.5}),
    ('CH-3', {'trail_atr': 3.0}),
    ('CH-4', {'trail_atr': 4.0}),
    ('TP15+CH3', {'take_profit': 0.15, 'trail_atr': 3.0}),
    ('TP15+ATR2', {'take_profit': 0.15, 'stop_atr': 2.0}),
    ('TP15+ATR1.5', {'take_profit': 0.15, 'stop_atr': 1.5}),
]

def sle_main():
    ap = argparse.ArgumentParser(description='止盈止损消融实验(口径与 signal_bridge 一致)')
    ap.add_argument('--pool', default='company_300', help='池名, 默认 company_300')
    ap.add_argument('--start', default='2021-01-01', help='回测起始(交易窗口), 默认 2021-01-01')
    ap.add_argument('--end', default=None, help='回测结束日, 默认最新')
    ap.add_argument('--cost-bps', type=float, default=10.0, help='单边成本(bps), 默认 10')
    args = ap.parse_args()

    kit = BacktestKit(pool=args.pool, start=args.start, end=args.end)
    kit.register(build_alpha_cands)
    base_p = EngineParams(cost_bps=args.cost_bps)

    results = [kit.run(SPEC, mode='unit01', name=name,
                       engine_params=base_p.copy(**kw))
               for name, kw in VARIANTS]

    print(f'\n=== 止盈止损消融: {args.pool} @ {args.start}~{args.end or "最新"} ===')
    print(f'参数: {base_p.brief()}  信号: 四信号等权 unit01(agg)')
    t = kit.compare(results + [kit.buyhold()], print_table=False)
    print(t.to_string(index=False))

    # 出场原因分布: 各配置下止损/止盈/移动止损出场的笔数与平均收益
    rows = []
    for r in results:
        td = r.trades
        if not len(td) or 'exit_reason' not in td.columns:
            continue
        done = td[td['completed']]
        for reason, sub in done.groupby('exit_reason'):
            rows.append({'config': r.name, 'reason': reason, 'n': len(sub),
                         'avg_ret': round(float(sub['ret'].mean()), 4),
                         'avg_days': round(float(sub['days'].mean()), 1),
                         'worst': round(float(sub['ret'].min()), 4)})
    if rows:
        print('\n--- 出场原因分布(已完成交易) ---')
        print(pd.DataFrame(rows).to_string(index=False))

    # 年度收益对比(看止盈止损在牛熊年份的差异化贡献)
    yr = pd.DataFrame({r.name: r.yearly() for r in results})
    if len(yr):
        print('\n--- 年度收益对比 ---')
        print(yr.round(3).to_string())

    out_dir = os.path.join(HERE,
                           'output', 'stop_loss_eval')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'summary_{args.pool}_{args.start[:4]}.csv')
    t.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'\n汇总表已存: {out_csv}')


# ==========================================================================
# ==== 源自 exec_price_ab_test.py ====  
# ==== 改名: main->ept_main, PoolRunner->cs_PoolRunner, POOLS->ept_POOLS
# ==========================================================================
ept_POOLS = [
    ('company_300', {'C_tmqmom': 1.0, 'F_er20': 1.0}, 12),
    ('etf_3x', {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2, 'F_obv20': 0.1}, 5),
    ('company_1000', {'G_vr20': 1.0, 'H_ichimoku_alpha': 1.0}, 8),
]

WINDOWS = [
    ('full', '2021-01-01', None),
    ('is', '2021-01-01', '2024-12-31'),
    ('oos', '2025-01-01', None)
]

STAT_KEYS = [
    'total_ret', 'cagr', 'sharpe', 'max_dd', 'ann_turnover',
    'n_trades', 'win_rate', 'avg_days', 'n_days'
]

def wdesc(weights: dict) -> str:
    return ','.join(f'{k}:{v:g}' for k, v in weights.items())

def gap_stats(vals: list) -> dict:
    if not vals:
        return {}
    a = np.array(vals, dtype=float)
    return {'n': len(a), 'mean': a.mean(), 'median': np.median(a),
            'q25': np.quantile(a, 0.25), 'q75': np.quantile(a, 0.75),
            'std': a.std(), 'pct_pos': float((a > 0).mean())}

def fmt_gap(g: dict) -> str:
    if not g:
        return 'n=0'
    return (f"n={g['n']:4d}  mean {g['mean']*100:+.3f}%  median {g['median']*100:+.3f}%  "
            f"q25 {g['q25']*100:+.3f}%  q75 {g['q75']*100:+.3f}%  "
            f"std {g['std']*100:.3f}%  P>0 {g['pct_pos']*100:.0f}%")

def ept_main():
    from quant.bc_combo_search import build_combo_cands, cs_PoolRunner, _slice
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(BASE, 'output', f'exec_ab_{run_id}')
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    t0 = time.time()

    for pool, w, top_k in ept_POOLS:
        print(f'\n{"=" * 78}\n===== pool={pool}  w={wdesc(w)}  top_k={top_k} =====', flush=True)
        kit = BacktestKit(pool=pool, start=None)
        kit.register(build_combo_cands, label='B3候选')
        runner = cs_PoolRunner(kit)
        comp = runner.fast_composite(w)
        p = kit.engine_params.copy(top_k=top_k)
        ow_all, cw_all = kit.open_wide, kit.close_wide
        assert ow_all.index.equals(cw_all.index), 'open/close index 不一致'
        print(f'  数据 {ow_all.index.min():%Y-%m-%d} ~ {ow_all.index.max():%Y-%m-%d}, '
              f'{ow_all.shape[1]} 标的', flush=True)

        # ---- 背景统计: 全池隔夜 vs 日内(逐日截面均值) ----
        on_ret = (ow_all / cw_all.shift(1) - 1.0)
        id_ret = (cw_all / ow_all - 1.0)
        m_on = on_ret.loc['2021-01-01':].mean(axis=1).mean()
        m_id = id_ret.loc['2021-01-01':].mean(axis=1).mean()
        print(f'  [背景] 全池逐日均值(2021~): 隔夜(close s -> open d) {m_on*100:+.4f}%/日   '
              f'日内(open d -> close d) {m_id*100:+.4f}%/日', flush=True)

        # ---- A/B 三窗口 ----
        for wname, start, end in WINDOWS:
            ow = _slice(ow_all, start, end)
            cw = _slice(cw_all, start, end)
            cols = ow.columns
            c = comp.reindex(index=ow.index, columns=cols)
            g = pd.DataFrame(True, index=ow.index, columns=cols)
            aw = (_slice(kit.atr_wide, start, end)
                  if kit.atr_wide is not None else None)
            resA = run_config(f'{pool}|{wname}|A_open', ow, cw, c, g, p, atr_wide=aw)
            resB = run_config(f'{pool}|{wname}|B_close', cw, ow, c, g, p, atr_wide=aw)
            for tag, r in (('A_open_exec', resA), ('B_close_exec', resB)):
                row = {'pool': pool, 'window': wname, 'exec': tag}
                row.update({k: r['stats'].get(k) for k in STAT_KEYS})
                all_rows.append(row)
            sA, sB = resA['stats'], resB['stats']
            hdr = (f"  [{wname} {start}~{end or 'end'}]  n_days={sA.get('n_days')}\n"
                   f"    {'':14s}{'total_ret':>10s}{'cagr':>8s}{'sharpe':>8s}"
                   f"{'max_dd':>8s}{'ann_to':>8s}{'n_tr':>6s}{'win':>7s}\n"
                   f"    A_open_exec  {sA['total_ret']:>10.3f}{sA['cagr']:>8.3f}"
                   f"{sA['sharpe']:>8.2f}{sA['max_dd']:>8.3f}"
                   f"{sA['ann_turnover']:>8.1f}{sA['n_trades']:>6d}{sA['win_rate']:>7.3f}\n"
                   f"    B_close_exec {sB['total_ret']:>10.3f}{sB['cagr']:>8.3f}"
                   f"{sB['sharpe']:>8.2f}{sB['max_dd']:>8.3f}"
                   f"{sB['ann_turnover']:>8.1f}{sB['n_trades']:>6d}{sB['win_rate']:>7.3f}\n"
                   f"    Δ(B-A)       {sB['total_ret']-sA['total_ret']:>+10.3f}"
                   f"{sB['cagr']-sA['cagr']:>+8.3f}{sB['sharpe']-sA['sharpe']:>+8.2f}"
                   f"{sB['max_dd']-sA['max_dd']:>+8.3f}")
            print(hdr, flush=True)

        # ---- 换仓执行日日内位移(A 版成交事件: B 与 A 成交价差 = close(d)/open(d)-1) ----
        trA = resA['trades']
        if 'completed' in trA.columns:
            trA = trA[trA['completed'] == True]  # noqa: E712
        buy_gap, sell_gap = [], []
        for _, t in trA.iterrows():
            for d, bucket in ((t['entry_exec'], buy_gap), (t['exit_exec'], sell_gap)):
                sym = t['symbol']
                if (d in ow_all.index and sym in ow_all.columns
                        and sym in cw_all.columns):
                    o, c_ = ow_all.at[d, sym], cw_all.at[d, sym]
                    if pd.notna(o) and pd.notna(c_) and o > 0:
                        bucket.append(float(c_) / float(o) - 1.0)
        print(f'  [执行日日内位移 close/open-1]  (= B 成交价相对 A 成交价)\n'
              f'    买入: {fmt_gap(gap_stats(buy_gap))}\n'
              f'    卖出: {fmt_gap(gap_stats(sell_gap))}', flush=True)
        pd.DataFrame({'buy': pd.Series(buy_gap), 'sell': pd.Series(sell_gap)}
                     ).to_csv(os.path.join(out_dir, f'gaps_{pool}.csv'), index=False)

    summ = pd.DataFrame(all_rows)
    summ.to_csv(os.path.join(out_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    print(f'\n[done] {time.time()-t0:.1f}s -> {out_dir}', flush=True)


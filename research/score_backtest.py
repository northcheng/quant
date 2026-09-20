# -*- coding: utf-8 -*-
"""
score_backtest.py — 独立只读研究模块: 组合分数 + 分数排序动态仓位 组合回测

设计原则(与 factor_research.py 一致):
  1. 只读: 仅读取 {pkl_dir}/{pool}_{interval}_ta_data.pkl, 不写任何现有路径
  2. 独立: 不 import quant 包; 仅复用同目录 factor_research 的数据加载函数
  3. 隔离: 全部输出写入 research/output/{pool}_bt_{run_id}/

策略逻辑(权重依据: factor_research 于 etf_3x 2021-2026 长历史的 IC/事件IC结论):
  - 截面排序: position_score(IC h20/h60 = +0.049/+0.053, 门内ICIR 0.13) + rsi(+0.039)
  - 入场质量: pattern_score_alpha(事件IC +0.050)
  - 反向因子(事件IC为负 → 取负权): trigger_score(-0.046) / boundary_score(-0.046)
    / candle_position_score(-0.047)
  - 组合分数 = Σ w_i * 成分当日截面百分位排名, 分数高 = 截面预期更好

交易引擎(逐日事件驱动, 防前视):
  - 信号日 s 收盘: 计算组合分数并截面排名; 滞回进出
    (进入: rank<=top_k 且门开; 退出: rank>exit_rank 或门关)
  - 执行日 d=s+1 开盘: 结算上一期(open s -> open d), 漂移权重, 换仓至目标, 收单边成本
  - 仓位: 已选持仓内按分数分档(tier 1.5/1.0/0.5 或 linear 0.5~1.5 或 equal),
    归一到 max_exposure, 单票上限 per_symbol_cap; band 带内不做微调(降换手)

内置对照(全部走同一引擎, 公平核算):
  - shuffle: 组合分数按交易日整体置换(破坏时序因果, 保留截面分布) -> 信号有效性对照
  - gate_equal: 门内全部等权(近似现系统"无分数"行为)
  - buyhold_pool / buyhold_tqqq: 等权买入持有基线

本文件是引擎与统计的底层实现(run_engine/EngineParams/perf_stats/compute_composite 等),
供 bt_core 复用; main 已收编为 bt_core.BacktestKit 的薄预设 —— CLI 参数与输出格式
不变, 程序化使用请直接用 bt_core.BacktestKit(见 bt_core.py 模块 docstring)。

用法:
  python score_backtest.py --pool etf_3x --pkl-dir .../research/data --start 2021-01-01
  python score_backtest.py ... --sensitivity    # 附消融/敏感性矩阵
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import normalize_causal  # alpha 复现用(其 main 不会在 import 时触发)
from signal_search import build_derived  # 复用衍生信号定义, 保证筛查与回测口径一致

warnings.filterwarnings('ignore')
try:  # Windows 控制台避免 UnicodeEncodeError
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 组合分数权重预设(绝对值和为 1; 负号 = 该因子事件级 IC 为负, 做反向使用)
WEIGHT_PRESETS = {
    'default': {'position_score': 0.25, 'rsi': 0.20, 'pattern_score_alpha': 0.20,
                'trigger_score': -0.15, 'boundary_score': -0.10, 'candle_position_score': -0.10},
    'rank_only': {'position_score': 0.45, 'rsi': 0.35, 'pattern_score_alpha': 0.20},
    'flip_only': {'pattern_score_alpha': 0.20, 'trigger_score': -0.30,
                  'boundary_score': -0.25, 'candle_position_score': -0.25},
}

# 经 signal_search 真伪检验(dyn_minus_static>0 且 n_eff_symbols>=0.7, 非静态身份效应)通过的
# 动态信号候选组合. 使用前需 --derived 注入衍生列(D_mom*/D_ma*_dist/D_sharpe*).
MOMENTUM_SETS = {
    'M1_mom250': {'D_mom250': 1.0},
    'M2_mom120': {'D_mom120': 1.0},
    'M3_ma200dist': {'D_ma200_dist': 1.0},
    'M4_adxstrength': {'adx_strength': 1.0},
    'M5_mom250+ma200': {'D_mom250': 0.6, 'D_ma200_dist': 0.4},
    'M6_mix': {'D_mom250': 0.4, 'D_mom120': 0.2, 'D_ma200_dist': 0.2,
               'D_sharpe120': 0.1, 'adx_strength': 0.1},
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

# 挖矿因子黄金组合(factor_signal.py WEIGHT_PRESETS 同源, 需 --mined 注入 F_* 列):
# 作为技术指标组合(M16/M17/M18)的能力对照基准
GOLD_SETS = {
    'gold4': {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2, 'F_obv20': 0.1},
    'gold3': {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2},
}
WEIGHT_PRESETS.update(GOLD_SETS)


# ================================================================ alpha 复现 ================================================================ #
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


# ================================================================ 组合分数 ================================================================ #
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


# ================================================================ 回测引擎 ================================================================ #
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


# ================================================================ 绩效统计 ================================================================ #
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


# ================================================================ 配置运行 ================================================================ #
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


# ================================================================ 主流程 ================================================================ #
def main():
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
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                           f'{args.pool}_bt_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    # ---- 套件构建(bt_core 薄预设; 引擎/统计仍为本文件底层实现, 口径逐位一致) ----
    # 延迟导入: bt_core 顶层 import 本模块引擎层, 函数内导入避免循环依赖
    from bt_core import BacktestKit
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
        from factor_mining import build_mined  # 延迟导入: 未用 --mined 时不依赖该模块
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


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
factor.backtest — 回测层(第5步)
================================
职责: 把第4步的组合分数(宽表)变成"一条真正可交易的净值曲线", 并给出绩效.

--- 这一层与前几层的根本区别 ---
  第1~4步都在回答"信号有没有预测力"(截面相关 / IC);
  本层回答"照这个信号交易, 到底赚不赚钱、承担多大风险、扣完成本还剩多少".
  两者不能互相替代: RankIC 0.04 的信号若换手极高, 扣成本后可能归零.

--- 防前视: 时间线只有一条 ---
  信号日 s 收盘: 读 composite.loc[s] 与当时持仓 -> 决定"明天开盘买卖什么"
  执行日 d 开盘: 先按 open(s) -> open(d) 结算, 再换仓, 最后收成本
  => 决策只用 s 及之前的信息, 成交发生在 d, 收益从 d 开盘起算. 全程不碰未来.
  (这与第3步"标签必须用未来"并不矛盾: 那条管评估口径, 这条管真实交易流程.)

--- rank -> 权重 ---
  1. 截面 rank(1 = 分数最高);
  2. 入场: rank <= top_k; 退出: 已持仓且 rank > exit_rank(或信号/价格缺失);
     top_k 与 exit_rank 之差即"滞回缓冲": 刚买进的票掉到第 6~12 名不会被立刻卖出,
     避免在门槛附近反复摩擦. top_k == exit_rank 时缓冲为 0, 换手最高.
  3. 已选持仓内按 sizing 分权重: equal(等权) / tier(三档 1.5/1.0/0.5) /
     linear(0.5~1.5 线性); 归一化到 max_exposure, 再受 per_symbol_cap 单票上限约束.
  4. band: 无进出场且目标与当前权重最大偏差 < band 时, 维持漂移权重不交易(省成本).

--- 成本 ---
  turnover = Σ|target_w - 当前_w|; 净值按 (1 - turnover * cost_rate) 计提,
  cost_rate = COST_BPS / 10000(单边). 一次换仓的买卖两侧都记成本.

--- 绩效 + IS/OOS ---
  perf_stats: total_ret / cagr / vol / sharpe / max_dd / calmar / ann_turnover /
              n_trades / win_rate / avg_ret / avg_days.
  权重只能在 IS 期估计(第4步红线), 再在 IS / OOS 两窗各自推进引擎对照 ——
  这才是"策略在样本外能否复现"的真正检验.

自检:
  cd ~/git && python -m quant.factor.backtest --pool etf_3x
  cd ~/git && python -m quant.factor.backtest --pool etf_3x --check
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import combine as cmb
from quant.factor import evaluate as ev
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import Dataset, load_pool

SCHEMES = ('equal', 'ic', 'greedy')
SIZINGS = ('equal', 'tier', 'linear')
STAT_KEYS = ['n_days', 'total_ret', 'cagr', 'vol', 'sharpe', 'max_dd', 'calmar',
             'ann_turnover', 'n_trades', 'win_rate', 'avg_ret', 'avg_days']
ANNUAL = 252


# ============================ 小工具 ============================

def _win(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
  """按交易日截取宽表的行(只选时间窗, 不改内容)."""
  idx = df.index
  if start is not None:
    idx = idx[idx >= pd.Timestamp(start)]
  if end is not None:
    idx = idx[idx <= pd.Timestamp(end)]
  return df.loc[idx]


# ============================ 1. 引擎参数 ============================

class EngineParams:
  """
  回测引擎参数(全是"交易规则", 与因子/权重无关).

  这些参数也属于"参数", 严格说只能在 IS 期调; 本层默认值直接取 config 的口径常量,
  与现有生产体系对齐, 便于互相印证.
  """

  def __init__(self, top_k: int = cfg.TOP_K, exit_rank: int = cfg.EXIT_RANK,
               sizing: str = 'equal', max_exposure: float = 1.0,
               per_symbol_cap: float = 0.30, cost_bps: float = cfg.COST_BPS,
               band: float = 0.05):
    """
    :param top_k: 入场 rank 门槛(持仓数上限)
    :param exit_rank: 退出 rank 门槛(应 >= top_k, 差额为滞回缓冲)
    :param sizing: 持仓内分权重方式 equal / tier / linear
    :param max_exposure: 总敞口上限(1.0 = 满仓)
    :param per_symbol_cap: 单票权重上限
    :param cost_bps: 单边交易成本(基点)
    :param band: 重平衡带宽(偏差小于该值不调仓)
    """
    if sizing not in SIZINGS:
      raise ValueError(f'未知 sizing: {sizing} (可选 {SIZINGS})')
    self.top_k = int(top_k)
    self.exit_rank = int(exit_rank)
    self.sizing = sizing
    self.max_exposure = float(max_exposure)
    self.per_symbol_cap = float(per_symbol_cap)
    self.cost_rate = float(cost_bps) / 10000.0            # 单边
    self.band = float(band)

  def copy(self, **kw) -> 'EngineParams':
    """复制并覆盖若干字段(敏感性对照用)."""
    p = EngineParams(self.top_k, self.exit_rank, self.sizing, self.max_exposure,
                     self.per_symbol_cap, self.cost_rate * 10000.0, self.band)
    for k, v in kw.items():
      if k == 'cost_bps':                              # 存储的是 rate, 需换算
        p.cost_rate = float(v) / 10000.0
      else:
        setattr(p, k, v)
    return p

  def brief(self) -> str:
    return (f'K={self.top_k}, exit_rank={self.exit_rank}, sizing={self.sizing}, '
            f'exposure={self.max_exposure}, cap={self.per_symbol_cap}, '
            f'cost={self.cost_rate * 10000:.0f}bps(单边), band={self.band}')


# ============================ 2. 仓位系数 ============================

def _sizing_coef(sub: pd.Series, sizing: str) -> pd.Series:
  """
  已选持仓内部的仓位系数: tier 三档 1.5/1.0/0.5, linear 0.5~1.5, equal 1.0.

  sub 是持仓的**组合分数**(高分 = 好); 返回与 sub 同索引的系数.
  系数最后会被归一化, 所以只看相对大小.

  :param sub: 持仓的组合分数
  :param sizing: equal / tier / linear
  :returns: 与 sub 同索引的仓位系数
  """
  n = len(sub)
  if n <= 1:
    return pd.Series(1.0, index=sub.index)
  frac = (n - sub.rank(ascending=False, method='first')) / (n - 1)   # 最好=1, 最差=0
  if sizing == 'equal':
    return pd.Series(1.0, index=sub.index)
  if sizing == 'linear':
    return 0.5 + frac
  tier = np.ceil(frac * 3.0)                                        # 0..3, 最好=3
  coef = np.where(tier >= 3, 1.5, np.where(tier >= 2, 1.0, 0.5))
  return pd.Series(coef, index=sub.index)


# ============================ 3. 逐日事件引擎 ============================

def run_engine(open_w: pd.DataFrame, close_w: pd.DataFrame, composite: pd.DataFrame,
               p: EngineParams) -> dict:
  """
  逐日事件引擎: 组合分数 -> 净值曲线 + 持仓 + 成交流水.

  时间线(防前视核心, 全程只有这一条):
    信号日 s = dates[i-1] 收盘: 取 composite.loc[s] 与当时持仓 -> 目标权重
    执行日 d = dates[i]   开盘: 结算(open(s) -> open(d)), 漂移权重, 换仓, 收成本
  分数只用 s 行; 目标只作用于 d 开盘之后; 收益只从成交开盘起算, 不碰未来.

  近似说明: 决策时用的"当前权重"是已漂移到 d 开盘的值(而非 s 收盘盯市),
  只影响 band 微调与权重小数, 不影响进出场决策(持仓集合是确定的).

  首日处理: 为让 total_ret 口径准确, 在 daily 首行补一条 dates[0] 的 equity=1.0
  初始行(参考实现缺这一行, 会把首日成本算进分母, 略偏).

  :param open_w: 开盘价宽表(date × symbol)
  :param close_w: 收盘价宽表
  :param composite: 组合分数宽表
  :param p: EngineParams
  :returns: {'daily','pos','trades','equity','ann_turnover','params'}
  """
  dates = open_w.index
  if len(dates) < 2:
    raise ValueError('交易日不足 2 天, 无法回测')
  rets = open_w.pct_change()                    # open(s) -> open(d) 的收益
  weights = {}                                  # symbol -> 执行后权重(以执行日开盘市值计)
  open_trades = {}                              # symbol -> 未平仓段信息
  trades, daily, pos_rows = [], [], []
  equity, turnover_sum, n_periods = 1.0, 0.0, 0

  for i in range(1, len(dates)):
    d, s = dates[i], dates[i - 1]
    n_periods += 1

    # --- A. 结算: weights 于 open(s) 建立, 持有至 open(d) ---
    gross = 0.0
    if weights:
      gross = float(sum(w * (rets.at[d, sym] if pd.notna(rets.at[d, sym]) else 0.0)
                        for sym, w in weights.items()))
    equity *= (1.0 + gross)
    if weights and gross > -1.0:                # 权重随价格自然漂移(仍和 = 1)
      weights = {sym: w * (1.0 + (rets.at[d, sym] if pd.notna(rets.at[d, sym]) else 0.0))
                 / (1.0 + gross) for sym, w in weights.items()}

    # --- B. 信号日 s 收盘: 决定"明天开盘买卖谁" ---
    comp = composite.loc[s] if s in composite.index else pd.Series(dtype=float)
    held = list(weights.keys())
    order = comp.rank(ascending=False, method='first') if len(comp) else pd.Series(dtype=float)
    exit_set = [sym for sym in held                  # 退出: 分数缺失 或 排名掉出 exit_rank
                if pd.isna(order.get(sym, np.nan)) or order.get(sym, np.nan) > p.exit_rank]
    for sym in exit_set:                             # 挂单卖出(价格缺失则留待下一日再试)
      px = open_w.at[d, sym] if sym in open_w.columns else np.nan
      t = open_trades.pop(sym, None)
      if t is None:
        continue
      if pd.notna(px):
        trades.append({'symbol': sym, 'entry_exec': t['entry_exec'], 'exit_exec': d,
                       'entry_px': t['entry_px'], 'exit_px': float(px),
                       'entry_w': t['entry_w'], 'peak_w': t['peak_w'],
                       'days': i - t['entry_i'], 'ret': float(px) / t['entry_px'] - 1.0,
                       'exit_reason': 'rank', 'completed': True})
      else:
        open_trades[sym] = t
    cand = []                                        # 入场候选: rank<=top_k & 未持有 & 有开盘价
    if len(order):
      for sym, r in order.items():
        if sym in weights or pd.isna(r) or r > p.top_k:
          continue
        if sym not in open_w.columns or not pd.notna(open_w.at[d, sym]):
          continue
        cand.append((float(r), sym))
      cand.sort()
    slots = max(0, p.top_k - (len(held) - len(exit_set)))
    entries = [sym for _, sym in cand[:slots]]
    new_held = [sym for sym in held if sym not in exit_set] + entries

    # --- C. 目标权重 ---
    target = {}
    if new_held:
      coef = _sizing_coef(comp.reindex(new_held), p.sizing)
      tot = float(coef.sum())
      if tot > 0:
        target = {sym: min(float(coef[sym] / tot * p.max_exposure), p.per_symbol_cap)
                  for sym in new_held}
    if not exit_set and not entries and weights:     # band: 无进出场且偏差小 -> 维持漂移
      maxdev = max((abs(target.get(sym, 0.0) - weights.get(sym, 0.0))
                    for sym in set(target) | set(weights)), default=0.0)
      if maxdev < p.band:
        target = dict(weights)

    # --- D. 换手与成本: 单边成本, 买卖两侧都记 ---
    turnover = float(sum(abs(target.get(sym, 0.0) - weights.get(sym, 0.0))
                         for sym in set(target) | set(weights)))
    cost_drag = turnover * p.cost_rate
    equity *= (1.0 - cost_drag)
    turnover_sum += turnover
    weights = {sym: w for sym, w in target.items() if w > 1e-9}
    for sym in entries:                              # 记录新开仓段
      if sym in weights:
        open_trades[sym] = {'entry_exec': d, 'entry_px': float(open_w.at[d, sym]),
                            'entry_w': weights[sym], 'peak_w': weights[sym], 'entry_i': i}
    for sym, t in open_trades.items():
      t['peak_w'] = max(t['peak_w'], weights.get(sym, 0.0))
    daily.append({'date': d, 'equity': equity, 'gross_ret': gross, 'cost_drag': cost_drag,
                  'turnover': turnover, 'n_pos': len(weights),
                  'exposure': float(sum(weights.values()))})
    pos_rows += [(d, sym, w) for sym, w in weights.items()]

  # --- 期末: 未平仓段以最后交易日收盘标记; 最后一日 open -> close 盯市 ---
  last_d = dates[-1]
  for sym in list(open_trades.keys()):
    t = open_trades.pop(sym)
    px = close_w.at[last_d, sym] if sym in close_w.columns else np.nan
    if pd.notna(px) and t['entry_px'] > 0:
      trades.append({'symbol': sym, 'entry_exec': t['entry_exec'], 'exit_exec': last_d,
                     'entry_px': t['entry_px'], 'exit_px': float(px),
                     'entry_w': t['entry_w'], 'peak_w': t['peak_w'],
                     'days': len(dates) - 1 - t['entry_i'],
                     'ret': float(px) / t['entry_px'] - 1.0,
                     'exit_reason': 'open', 'completed': False})
  if weights and daily:
    final = sum(w * (close_w.at[last_d, sym] / open_w.at[last_d, sym] - 1.0)
                for sym, w in weights.items()
                if sym in close_w.columns and pd.notna(close_w.at[last_d, sym])
                and pd.notna(open_w.at[last_d, sym]) and open_w.at[last_d, sym] > 0)
    if final:
      daily[-1]['equity'] *= (1.0 + final)

  daily_df = pd.DataFrame(daily).set_index('date')
  base = pd.DataFrame([{'date': dates[0], 'equity': 1.0, 'gross_ret': 0.0, 'cost_drag': 0.0,
                        'turnover': 0.0, 'n_pos': 0, 'exposure': 0.0}]).set_index('date')
  daily_df = pd.concat([base, daily_df])
  pos_df = pd.DataFrame(pos_rows, columns=['date', 'symbol', 'weight'])
  trades_df = pd.DataFrame(trades)
  ann_turnover = turnover_sum / max(n_periods, 1) * ANNUAL
  return {'daily': daily_df, 'pos': pos_df, 'trades': trades_df,
          'equity': daily_df['equity'], 'ann_turnover': ann_turnover, 'params': p}


# ============================ 4. 绩效指标 ============================

def perf_stats(equity: pd.Series, ann_turnover: float = np.nan,
               trades_df: pd.DataFrame = None) -> dict:
  """
  净值 -> 绩效指标.

  total_ret / cagr / vol / sharpe / max_dd / calmar / ann_turnover,
  再由成交流水派生 n_trades / win_rate / avg_ret / avg_days(只统计已平仓段).

  :param equity: 净值曲线(首行为 1.0 初始行)
  :param ann_turnover: 年化单边换手
  :param trades_df: run_engine 的成交流水(None = 不算交易统计)
  :returns: 指标 dict
  """
  if len(equity) < 2:
    return {k: np.nan for k in STAT_KEYS}
  ret = equity.pct_change().dropna()
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
  total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
  cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 else -1.0
  vol = float(ret.std() * np.sqrt(ANNUAL)) if len(ret) > 1 else np.nan
  sharpe = (float(ret.mean() / ret.std() * np.sqrt(ANNUAL))
            if len(ret) > 1 and ret.std() > 0 else np.nan)
  dd = float((equity / equity.cummax() - 1.0).min())
  calmar = cagr / abs(dd) if dd < 0 else np.nan
  out = {'n_days': len(equity), 'total_ret': round(total, 3), 'cagr': round(cagr, 3),
         'vol': round(vol, 3), 'sharpe': round(sharpe, 2), 'max_dd': round(dd, 3),
         'calmar': round(calmar, 2) if pd.notna(calmar) else np.nan,
         'ann_turnover': round(ann_turnover, 1) if pd.notna(ann_turnover) else np.nan,
         'n_trades': 0, 'win_rate': np.nan, 'avg_ret': np.nan, 'avg_days': np.nan}
  if trades_df is not None and len(trades_df) and 'completed' in trades_df.columns:
    done = trades_df[trades_df['completed']]
    if len(done):
      out.update({'n_trades': int(len(done)),
                  'win_rate': round(float((done['ret'] > 0).mean()), 3),
                  'avg_ret': round(float(done['ret'].mean()), 4),
                  'avg_days': round(float(done['days'].mean()), 1)})
  return out


def yearly_returns(equity: pd.Series) -> pd.Series:
  """年度收益: 年末净值 / 年初净值 - 1."""
  if len(equity) < 2:
    return pd.Series(dtype=float)
  yr = equity.groupby(equity.index.year)
  return (yr.last() / yr.first() - 1.0).round(3)


def monthly_returns(equity: pd.Series) -> pd.Series:
  """月度收益: 月末净值 / 月初净值 - 1(索引 'YYYY-MM')."""
  if len(equity) < 2:
    return pd.Series(dtype=float)
  g = equity.groupby([equity.index.year, equity.index.month])
  out = (g.last() / g.first() - 1.0).round(4)
  out.index = [f'{y}-{m:02d}' for y, m in out.index]
  return out


# ============================ 5. 对照与基准 ============================

def shuffle_composite(comp: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
  """随机对照: 整张截面按交易日整体置换, 破坏"分数-时间"对齐(但保留截面结构)."""
  rng = np.random.default_rng(seed)
  perm = rng.permutation(len(comp))
  return comp.iloc[perm].set_axis(comp.index)


def buyhold(open_w: pd.DataFrame, close_w: pd.DataFrame) -> dict:
  """
  全池等权买入持有(基准): 用**同一个引擎**核算, 保证与策略净值可比.

  做法: 常数分数(全 0.5) + top_k = 全池 + exit_rank = 全池+1(永不退出) +
       sizing=equal + 单票上限 1.0 + 零成本 + band=1.0(永不重平衡).
  """
  comp = pd.DataFrame(0.5, index=open_w.index, columns=open_w.columns)
  n = len(open_w.columns)
  p = EngineParams(top_k=n, exit_rank=n + 1, sizing='equal', max_exposure=1.0,
                   per_symbol_cap=1.0, cost_bps=0.0, band=1.0)
  return run_engine(open_w, close_w, comp, p)


def folded_perf(open_w: pd.DataFrame, close_w: pd.DataFrame, comp: pd.DataFrame,
                p: 'EngineParams', folds: list, keys=('sharpe', 'total_ret')) -> pd.DataFrame:
  """
  逐子窗绩效: 对每个 (start, end) 子窗**独立**跑一遍引擎, 汇总指定指标.

  用途: 把"一段 IS"拆成若干连续子段分别评估 —— 单段偶然很强的方案, 在多段上均值
  会被拉低. 这是 make_fold_objective 的底座; 也可单独用于事后观察方案的分段稳定性.

  :param open_w / close_w / comp: **全样本**宽表(函数内部按子窗截取)
  :param p: EngineParams
  :param folds: split_folds 产出的 [(start, end), ...]
  :param keys: 需要汇总的 perf_stats 字段
  :returns: DataFrame[fold, start, end, n_days, *keys](子窗过短则指标为 NaN)
  """
  rows = []
  for i, (a, b) in enumerate(folds, start=1):
    ow, cw = _win(open_w, a, b), _win(close_w, a, b)
    cp = _win(comp, a, b)
    row = {'fold': i, 'start': a, 'end': b, 'n_days': int(len(ow))}
    if len(ow) < 2 or cp.dropna(how='all').empty:            # 子窗不可回测
      row.update({k: np.nan for k in keys})
    else:
      res = run_engine(ow, cw, cp, p)
      st = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      row.update({k: st.get(k, np.nan) for k in keys})
    rows.append(row)
  return pd.DataFrame(rows)


def make_fold_objective(open_w: pd.DataFrame, close_w: pd.DataFrame,
                        p: 'EngineParams', folds: list, metric: str = 'sharpe',
                        agg: str = 'mean'):
  """
  构造"折外目标函数" objective(weights, comp) -> float.

  语义: 把 comp 在各子窗分别回测, 取 metric 的聚合值(默认均值)作为接受判据.
  全部子窗都无效/NaN 时返回 -inf, 保证不会被误选. 与 combine.greedy_weight_search /
  refine_weights 组合, 即可把权重搜索的目标从"IS RankIC"换成"折外更稳".

  注意: 只有"训练子窗"才应作为 folds 传进来 —— 若塞入 OOS 段即构成前视.

  :param open_w / close_w: 全样本宽表(函数内部按子窗截取)
  :param p: EngineParams
  :param folds: 训练期子窗列表
  :param metric: 取 perf_stats 的哪个字段(如 'sharpe' / 'total_ret' / 'calmar')
  :param agg: 'mean'(默认) 或 'min' —— min 更保守(要求每折都不差)
  :returns: objective(weights, comp) -> float(越大越好; 无有效折返回 -inf)
  """
  def objective(weights, comp):
    fp = folded_perf(open_w, close_w, comp, p, folds, keys=(metric,))
    v = fp[metric].dropna()
    if v.empty:
      return -np.inf
    return float(v.mean() if agg == 'mean' else v.min())
  return objective


def verify_backtest(ds: Dataset, method: str = prep.DEFAULT_PREP, h: int = 1,
                    top_k: int = cfg.TOP_K, min_cs: int = cfg.MIN_CS,
                    seed: int = 42) -> tuple:
  """
  引擎标定: 用"已知答案"的输入检验引擎有没有按预期工作.

  三档 sanity(同一窗口, 默认参数):
    oracle  = 把**未来收益**当分数喂进去 -> 应大幅盈利(挑的就是事后涨最多的);
    shuffle = 把 oracle 分数按交易日整体打乱 -> 应≈随机, 远逊于 oracle;
    buyhold = 全池等权持有 -> 市场基准.
  若 oracle 不赚 / shuffle 也大赚, 说明引擎或口径有 bug.

  前视代价: 定权用 IS(诚实) vs 用全样本(前视), 各自跑 OOS 段净值对照 ——
  前视版本会在 OOS 上虚高, 差额即"偷看未来"能虚增多少.

  :returns: (cases_df, leak_df)
  """
  open_w, close_w = ds.wide('open'), ds.wide('close')
  fwd = ev.forward_return(ds, h)
  start, end = ev.window_bounds('full')
  ow, cw, fw = _win(open_w, start, end), _win(close_w, start, end), _win(fwd, start, end)
  p = EngineParams(top_k=top_k)

  cases = [('oracle(=fwd)', run_engine(ow, cw, fw, p)),
           ('shuffle(oracle)', run_engine(ow, cw, shuffle_composite(fw, seed), p)),
           ('buyhold', buyhold(ow, cw))]
  rows = []
  for label, res in cases:
    s = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    rows.append({'case': label, 'n_fac': len(ow.columns),
                 **{k: s.get(k) for k in STAT_KEYS}})
  cases_df = pd.DataFrame(rows)

  sigs = cmb.build_signals(ds, method=method, min_cs=min_cs)
  is_s, is_e = ev.window_bounds('is')
  oos_s, oos_e = ev.window_bounds('oos')
  w_honest = cmb.ic_weights(sigs, _win(fwd, is_s, is_e), min_cs=min_cs)   # 只用 IS
  w_leak = cmb.ic_weights(sigs, fwd, min_cs=min_cs)                      # 偷看全样本
  rows = []
  for label, w in (('honest(IS定权)', w_honest), ('lookahead(全样本定权)', w_leak)):
    comp = cmb.combine_signals(sigs, w, min_cs=min_cs)
    res = run_engine(_win(open_w, oos_s, oos_e), _win(close_w, oos_s, oos_e),
                     _win(comp, oos_s, oos_e), p)
    s = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    rows.append({'case': label, 'n_fac': len(w), **{k: s.get(k) for k in STAT_KEYS}})
  leak_df = pd.DataFrame(rows)
  return cases_df, leak_df


# ============================ 6. CLI ============================

def main():
  ap = argparse.ArgumentParser(description='第5步 回测层: 组合分数 -> 净值 + 绩效')
  ap.add_argument('--pool', default=cfg.DEFAULT_POOL)
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  ap.add_argument('--method', default=prep.DEFAULT_PREP, choices=prep.PREP_METHODS)
  ap.add_argument('--h', type=int, default=1, help='前向收益持有期(仅影响标签评估)')
  ap.add_argument('--top-k', type=int, default=cfg.TOP_K, help='入场 rank 门槛(持仓数上限)')
  ap.add_argument('--exit-rank', type=int, default=cfg.EXIT_RANK, help='退出 rank 门槛')
  ap.add_argument('--sizing', default='equal', choices=SIZINGS)
  ap.add_argument('--max-exposure', type=float, default=1.0)
  ap.add_argument('--per-symbol-cap', type=float, default=0.30)
  ap.add_argument('--cost-bps', type=float, default=cfg.COST_BPS, help='单边成本(基点)')
  ap.add_argument('--band', type=float, default=0.05, help='重平衡带宽')
  ap.add_argument('--min-cs', type=int, default=cfg.MIN_CS)
  ap.add_argument('--group', default=None, help='只用某类因子')
  ap.add_argument('--scheme', default='all', choices=('all',) + SCHEMES)
  ap.add_argument('--thresh', type=float, default=cmb.DEDUPE_RHO)
  ap.add_argument('--max-n', type=int, default=8)
  ap.add_argument('--min-gain', type=float, default=cmb.MIN_GAIN)
  ap.add_argument('--min-ic', type=float, default=0.0)
  ap.add_argument('--check', action='store_true', help='只跑引擎标定自检')
  args = ap.parse_args()

  log = cfg.get_logger('backtest')
  ds = load_pool(args.pool, args.interval)
  log.info(ds.summary())

  if args.check:
    cases_df, leak_df = verify_backtest(ds, args.method, args.h, args.top_k, args.min_cs)
    print('\n[引擎标定] 同窗口, 默认参数(oracle 应大赚, shuffle 应≈随机):')
    print(cases_df.to_string(index=False))
    print('\n[前视代价] OOS 段净值对照(honest 应 <= lookahead):')
    print(leak_df.to_string(index=False))
    return

  open_w, close_w = ds.wide('open'), ds.wide('close')
  fwd = ev.forward_return(ds, args.h)
  is_s, is_e = ev.window_bounds('is')
  oos_s, oos_e = ev.window_bounds('oos')
  fwd_is = _win(fwd, is_s, is_e)

  sigs = cmb.build_signals(ds, method=args.method, names=fct.list_factors(args.group),
                           min_cs=args.min_cs)

  # --- 定权: 全部只用 IS 期(第4步红线) ---
  sc = cmb.rank_ic_scores(sigs, fwd_is, min_cs=args.min_cs)
  kept, dropped = cmb.dedupe(sigs, sc, thresh=args.thresh, min_cs=args.min_cs)
  log.info(f'入池 {len(sigs)} -> 去重后 {len(kept)}; 剔除 {[d["factor"] for d in dropped]}')

  weights = {}
  if args.scheme in ('all', 'equal'):
    weights['equal'] = cmb.equal_weights(kept)
  if args.scheme in ('all', 'ic'):
    weights['ic'] = cmb.ic_weights(sigs, fwd_is, names=kept, min_cs=args.min_cs,
                                   min_ic=args.min_ic)
  if args.scheme in ('all', 'greedy'):
    gnames, gain_df = cmb.greedy_forward(sigs, fwd_is, names=kept, max_n=args.max_n,
                                        min_gain=args.min_gain, min_cs=args.min_cs)
    if gnames:
      weights['greedy'] = cmb.equal_weights(gnames)
      if len(gain_df):
        log.info('贪心路径:\n' + gain_df.to_string(index=False))
    else:
      log.warning('贪心未选出任何因子, 跳过 greedy 方案')

  p = EngineParams(args.top_k, args.exit_rank, args.sizing, args.max_exposure,
                   args.per_symbol_cap, args.cost_bps, args.band)
  log.info('引擎: ' + p.brief())

  best = max(sc, key=lambda n: abs(sc[n]) if pd.notna(sc[n]) else -1.0)
  combos = {k: cmb.combine_signals(sigs, w, min_cs=args.min_cs) for k, w in weights.items()}
  combos[f'single:{best}'] = cmb.combine_signals(sigs, {best: 1.0}, min_cs=args.min_cs)
  if 'equal' in weights:
    combos['shuffle(equal)'] = shuffle_composite(combos['equal'], seed=42)

  windows = (('IS', is_s, is_e), ('OOS', oos_s, oos_e))
  rows = []
  for kind, comp in combos.items():
    if kind in weights:
      nf = len(weights[kind])
    elif kind.startswith('shuffle'):
      nf = len(kept)
    else:
      nf = 1
    for win, ws, we in windows:
      res = run_engine(_win(open_w, ws, we), _win(close_w, ws, we), _win(comp, ws, we), p)
      s = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      rows.append({'kind': kind, 'window': win, 'n_fac': nf,
                   **{k: s.get(k) for k in STAT_KEYS}})
  for win, ws, we in windows:
    res = buyhold(_win(open_w, ws, we), _win(close_w, ws, we))
    s = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    rows.append({'kind': 'buyhold', 'window': win, 'n_fac': len(open_w.columns),
                 **{k: s.get(k) for k in STAT_KEYS}})
  table = pd.DataFrame(rows)
  print('\n[回测汇总] 权重全部只在 IS 期估计; IS/OOS 各窗独立推进引擎')
  print(table.to_string(index=False))

  main_kind = 'equal' if 'equal' in weights else next(iter(weights))
  res_full = run_engine(_win(open_w, cfg.TRADE_START, None),
                        _win(close_w, cfg.TRADE_START, None),
                        _win(combos[main_kind], cfg.TRADE_START, None), p)
  print(f'\n[主方案 {main_kind} 全交易窗({cfg.TRADE_START} 起)分年收益]')
  print(yearly_returns(res_full['equity']).to_string())


if __name__ == '__main__':
  main()

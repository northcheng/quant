# -*- coding: utf-8 -*-
"""
factor.factor — 因子定义层(第1步)
==================================
职责: 定义"因子是什么".

  一个因子 = 一个把 Dataset 映射成 (date × symbol) 宽表的函数.

因子契约(所有因子必须遵守):
  1. 因果: 只读 t 及之前的数据, 只用 rolling / shift 这类因果算子,
     绝不用全样本统计量(mean/std/quantile/rank 不带窗口). 由 verify_causal 把关;
  2. 形态: 输入 Dataset, 输出 DataFrame(index=交易日, columns=标的);
  3. NaN 原样保留(停牌 / 未上市 / 暖机期), 不填充, 不置零;
  4. 不做截面处理 —— 去极值 / 标准化 / 排名是第2步 prepare.py 的事.

  第 4 条是刻意的分工: 因子层只回答"这个数字是多少", 不回答"它该排在什么位置".
  这样同一个因子可以被不同的预处理方案复用, 也便于单独做因果性检验.

  三类刻意的截面例外(都只读 t 时刻的截面, 因此仍然因果, 截断一致性检验照样通过):
    a) rel_mom_60(跨品种相对动量): "当日截面去均值" 才能构造出"相对强弱". 它刻意保留下来
       当"冗余因子"的教学样本 —— 减去当日截面常数后, 在 rank/zscore 预处理下与 mom_60
       完全等价(见该因子注释);
    b) 池等权截面 pr = mean_over_pool(ret_1, axis=1): R_/G_/K_ 族用它度量"相对全池"的
       残差动量 / 下行 beta / 池波动状态. 只读 t 时刻的截面均值, t 之后的收益不参与,
       与 a) 的区别仅在"减/除的是均值还是别的截面统计量";
    c) C_tmqmom: "rank_pct(mom_12_1) + rank_pct(er_20)" —— 逐日截面 pct 排名后相加.
       这是旧系统 bc_combo_search 的 C_tmqmom 口径(排名只读 t 时刻截面, 故仍因果),
       补进注册表是为让 cd_ic 导出的权重(含该因子名)在生产端可直接消费.

暖机处理: 基础因子的 rolling 都显式写 min_periods=窗口, 窗口不满 -> NaN;
  新因子矿里部分沿用参考实现的 min_periods=窗口//2 等口径(为与文献/研究报告对齐, 已在各因子处注明).
  宁可缺值, 也不用"只凑了3天就算的 MA20" —— 那种值数量级都不稳, 混进截面排序
  会污染 rank. 各因子的实际暖机天数由 overview() 实测给出, 不靠人工估算.

本模块五件事:
  1. 因子注册表: register 装饰器 + REGISTRY(名 -> FactorSpec);
  2. 基础因子 27 个: 动量 / 反转 / 波动 / 量能 / 趋势, 每个附一行公式;
  3. 新因子矿 29 个(见各段注释, 均附文献出处与"是否与 mom_* 冗余"的判断):
       alpha化(21)   —— normalize_causal(|TA 指标|, 252, 60): 把 adx/atr/rsi/kama/ichimoku
                       等"水平型"TA 指标转成"自身历史分位", 真正用上 pkl 里 200+ 列 TA 面板;
       残差动量(1)   —— R_resmom121, 对池等权收益回归后的残差动量(与 mom_* 正交);
       信息离散(1)   —— I_id60, Da-Gurun-Warachka(2014) 信息离散度;
       微观结构(4)   —— G_parkcc20 / G_oviv20 / G_vr20 / G_dnbeta60, 只需 OHLCV;
       门控(2)       —— K_mom_er / K_mom_poolvolq, 条件动量(Daniel-Moskowitz 动量崩溃防御);
  4. 旧预设兼容 3 个: C_tmqmom / N_range20 / N_idiovol60 —— 旧系统 signal_bridge 预设权重
       直接消费的因子, 公式逐字对齐 repro_old_presets.py, 保证权重口径一致;
  5. 因果性防线: verify_causal —— 截断一致性检验.

关于 direction 字段: 它只是"先验方向", 记下经济含义上的预期(值越大越看涨为 +1),
  便于第3步 IC 符号与先验不一致时立刻发现"这个因子可能反了".
  它不参与任何计算 —— 真正的符号以第3步实测 IC 为准.

自检:
  cd ~/git && python -m quant.factor.factor --pool etf_3x
  cd ~/git && python -m quant.factor.factor --pool etf_3x --check
"""
import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor.data import Dataset, load_pool

ANNUAL = 252                                    # 年化交易日数

FactorFn = Callable[[Dataset], pd.DataFrame]    # 因子函数签名


@dataclass(frozen=True)
class FactorSpec:
  """一个因子的元信息 + 计算函数.

  :param name: 因子名(唯一)
  :param group: 类别(动量/反转/波动/量能/趋势/alpha化/残差动量/信息离散/微观结构/门控)
  :param formula: 一行公式说明(t 表示当日, t-n 表示 n 个交易日前)
  :param window: 名义回看天数(公式里显式出现的窗口, 供人读)
  :param direction: 先验方向, +1 = 值越大越看涨, -1 = 反之
  :param fn: 计算函数 Dataset -> 宽表
  """
  name: str
  group: str
  formula: str
  window: int
  direction: int
  fn: FactorFn

  def compute(self, ds: Dataset) -> pd.DataFrame:
    """算因子 -> (date × symbol) 宽表."""
    df = self.fn(ds)
    if not isinstance(df, pd.DataFrame):
      raise TypeError(f'因子 {self.name} 未返回 DataFrame: {type(df)}')
    return df


REGISTRY: dict = {}                             # 因子名 -> FactorSpec


def register(name: str, group: str, formula: str, window: int,
             direction: int = 1):
  """
  因子注册装饰器.

  :param name: 因子名(唯一)
  :param group: 类别
  :param formula: 一行公式说明
  :param window: 名义回看天数
  :param direction: 先验方向(+1 / -1)
  """
  def deco(fn: FactorFn) -> FactorFn:
    if name in REGISTRY:
      raise ValueError(f'因子名重复: {name}')
    REGISTRY[name] = FactorSpec(name, group, formula, window, direction, fn)
    return fn
  return deco


# ============================ 因子计算小工具 ============================
# 全部是因果算子(只看 t 及之前), 且窗口不满即 NaN.

def _ret(close: pd.DataFrame, n: int = 1) -> pd.DataFrame:
  """n 日简单收益: close[t] / close[t-n] - 1."""
  return close / close.shift(n) - 1


def _ma(close: pd.DataFrame, n: int) -> pd.DataFrame:
  """n 日均线; 不足 n 天为 NaN."""
  return close.rolling(n, min_periods=n).mean()


def _std(x: pd.DataFrame, n: int) -> pd.DataFrame:
  """n 日滚动标准差(ddof=1); 不足 n 天为 NaN."""
  return x.rolling(n, min_periods=n).std()


def normalize_causal(x, window: int = cfg.NORM_WINDOW,
                     min_periods: int = cfg.NORM_MINP):
  """
  因果归一化到 [0, 1] —— alpha 化算子.

  把"水平型"指标(adx/atr/rsi/kama/... 数值大小随标的与时代漂移)转成"自身历史分位":
    lo/hi = 过去 window 天的最小/最大值(不足 window 天时用 expanding 补齐 warmup, 序列开头
    也能出值);  span = hi - lo;  输出 = (x - lo) / span.
  1) 只用 t 及之前的数据, 因此因果; 2) 值域天然 [0,1], 跨标的可比;
  3) span = 0(窗口内无波动, 如长期停牌)记 0; 4) 末尾 fillna(0.0) 兜底(与 bc_technical_analysis
     的 normalize_causal 同口径, 出处: research_2026-09/factor_research.py normalize_causal).

  注意: fillna(0.0) 会把"NaN" 也变成 0(最低分位), 所以对含暖机/停牌 NaN 的输入不要直接用,
  请用 _alphaize —— 它先把原 NaN 记下来, 算完再掩回去.

  :param x: Series 或 DataFrame
  :param window: 滚动窗口(默认 cfg.NORM_WINDOW = 252)
  :param min_periods: 滚动最少样本数(默认 cfg.NORM_MINP = 60)
  :returns: 与 x 同形的 [0,1] 归一化结果
  """
  s = x.astype(float)
  lo = s.rolling(window=window, min_periods=min_periods).min()
  hi = s.rolling(window=window, min_periods=min_periods).max()
  lo = lo.combine_first(s.expanding().min())
  hi = hi.combine_first(s.expanding().max())
  span = hi - lo
  return ((s - lo) / span.replace(0, np.nan)).fillna(0.0)


def _alphaize(x, window: int = cfg.NORM_WINDOW,
              min_periods: int = cfg.NORM_MINP) -> pd.DataFrame:
  """normalize_causal(|x|) 且 NaN 不被 fillna(0) 污染: 先记录有效位, 事后 mask 回 NaN."""
  ok = x.notna()
  v = normalize_causal(x.abs(), window=window, min_periods=min_periods)
  return v.where(ok)


def _pool_ret(close: pd.DataFrame) -> pd.Series:
  """当日池等权收益 pr[t] = mean_over_pool(ret_1)[t].

  这是文件头契约的"截面例外 b)": 只读 t 时刻的截面均值, t 之后的收益不参与, 故仍然因果.
  R_/G_/K_ 族用它做"相对全池"的残差动量 / 下行 beta / 池波动状态. NaN 自动跳过.
  """
  return _ret(close, 1).mean(axis=1)


# ============================ 动量组 ============================
# 逻辑: 强者继续强(趋势延续). 3x 杠杆 ETF 的动量效应通常很显著.

@register('mom_20', '动量', 'close[t] / close[t-20] - 1', window=20)
def _mom_20(ds: Dataset) -> pd.DataFrame:
  return _ret(ds.wide('close'), 20)


@register('mom_60', '动量', 'close[t] / close[t-60] - 1', window=60)
def _mom_60(ds: Dataset) -> pd.DataFrame:
  return _ret(ds.wide('close'), 60)


@register('mom_120', '动量', 'close[t] / close[t-120] - 1', window=120)
def _mom_120(ds: Dataset) -> pd.DataFrame:
  return _ret(ds.wide('close'), 120)


@register('mom_12_1', '动量', 'close[t-20] / close[t-252] - 1', window=252)
def _mom_12_1(ds: Dataset) -> pd.DataFrame:
  # 经典 "12-1 动量": 取最近一年收益, 但跳过最近一个月(避免与短期反转打架)
  close = ds.wide('close')
  return close.shift(20) / close.shift(252) - 1


@register('mom_sharpe_120', '动量',
          '(close[t]/close[t-120] - 1) / (std(ret_1, 120) * sqrt(252))', window=120)
def _mom_sharpe_120(ds: Dataset) -> pd.DataFrame:
  # 夏普动量(风险调整动量): 单位波动换来的趋势强度.
  # 与裸动量 mom_120 的关键差别: 分母 vol 在截面上有独立排序, 所以这不是 mom_120 的
  # 单调变换 —— 它把"高波动的暴涨"折算掉, 追的是"性价比高的强者". 业界真实在用.
  close = ds.wide('close')
  return _ret(close, 120) / (_std(_ret(close, 1), 120) * np.sqrt(ANNUAL))


@register('rel_mom_60', '动量', 'mom_60 - mean_over_pool(mom_60)', window=60)
def _rel_mom_60(ds: Dataset) -> pd.DataFrame:
  # 跨品种相对动量(相对强弱): 减去当日全池等权动量, 只留"跑赢同侪"的部分.
  # 这是本文件唯一一次截面操作(见文件头"唯一例外"), 只读 t 时刻截面, 因果性仍通过.
  # 教学用途: 减的是"当日截面常数", 而 rank/zscore 预处理也是逐日截面变换,
  #   二者叠加后与 mom_60 完全等价(连 Pearson IC 都不变) -> 第4步去重时会现出原形.
  mom = _ret(ds.wide('close'), 60)
  return mom.sub(mom.mean(axis=1), axis=0)


# ============================ 反转组 ============================
# 逻辑: 短期跌多了会反弹. 因子值已取负 -> 值越大 = 前期跌得越多 = 越看涨.

@register('rev_5', '反转', '-(close[t] / close[t-5] - 1)', window=5)
def _rev_5(ds: Dataset) -> pd.DataFrame:
  return -_ret(ds.wide('close'), 5)


@register('rev_10', '反转', '-(close[t] / close[t-10] - 1)', window=10)
def _rev_10(ds: Dataset) -> pd.DataFrame:
  return -_ret(ds.wide('close'), 10)


@register('rsi_dev_14', '反转', 'RSI(14)[t] - 50', window=14)
def _rsi_dev_14(ds: Dataset) -> pd.DataFrame:
  # RSI 相对 50 的偏离(去掉中性常数 50). 出处: research_2026-09/alpha_mining.py
  #   H_rsidev_alpha, 报告 etf_3x h=20 C弱(nw_t≈2.0~2.4).
  # Wilder 平滑: ewm(alpha=1/n, adjust=False) 是因果的前向递推(只随固定起点向右), 满足契约1.
  close = ds.wide('close')
  n = 14
  delta = close.diff()
  up = delta.clip(lower=0.0)
  dn = (-delta).clip(lower=0.0)
  roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
  roll_dn = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
  rsi = 100.0 - 100.0 / (1.0 + roll_up / roll_dn)
  return rsi - 50.0


# ============================ 波动组 ============================
# 逻辑: 低波动异象 —— 波动率低的标的(风险调整后)未来表现更好. 故 direction = -1.

@register('vol_20', '波动', 'std(ret_1, 20) * sqrt(252)', window=20, direction=-1)
def _vol_20(ds: Dataset) -> pd.DataFrame:
  return _std(_ret(ds.wide('close'), 1), 20) * np.sqrt(ANNUAL)


@register('vol_60', '波动', 'std(ret_1, 60) * sqrt(252)', window=60, direction=-1)
def _vol_60(ds: Dataset) -> pd.DataFrame:
  return _std(_ret(ds.wide('close'), 1), 60) * np.sqrt(ANNUAL)


@register('downside_vol_60', '波动', 'std(min(ret_1, 0), 60) * sqrt(252)',
          window=60, direction=-1)
def _downside_vol_60(ds: Dataset) -> pd.DataFrame:
  # 下行波动(半方差): 只统计下跌日的波动, 上涨日按 0 计入 min(ret, 0).
  # 与对称的 vol_60 差别: 两个标的可能总波动相近, 但一个靠"暴涨暴跌"、一个靠"阴跌",
  #   半方差把它们区分开 -> 排序不同, 有独立信息. 低下行波动 = 更优, 故 direction = -1.
  # 用 clip(upper=0) 而非 where(r<0, 0): clip 保留 NaN, where 会把暖机/停牌的 NaN 填成 0.
  r = _ret(ds.wide('close'), 1)
  return _std(r.clip(upper=0.0), 60) * np.sqrt(ANNUAL)


@register('bbw_20', '波动', 'std(close, 20) / MA20', window=20, direction=-1)
def _bbw_20(ds: Dataset) -> pd.DataFrame:
  # 布林带宽度(以均线归一). 低波动异象 -> 带宽小更优, direction = -1.
  # 出处: research_2026-09/alpha_mining.py H_bbw_alpha; etf_3x h=20 C弱(exc≈0.0055).
  close = ds.wide('close')
  return _std(close, 20) / _ma(close, 20)


@register('max_ret_20', '波动', 'max(ret_1, 20)', window=20, direction=-1)
def _max_ret_20(ds: Dataset) -> pd.DataFrame:
  # 20 日最大单日涨幅(MAX 彩票效应, Bali-Cakici-Whitelaw 2011): 越大越看跌, direction = -1.
  # 出处: research_2026-09/alpha_mining.py X_maxret_neg20(其取负版); etf_3x h=60 C弱(exc≈0.0148).
  return _ret(ds.wide('close'), 1).rolling(20, min_periods=20).max()


@register('kurt_60', '波动', 'kurt(ret_1, 60)', window=60, direction=-1)
def _kurt_60(ds: Dataset) -> pd.DataFrame:
  # 收益峰度(尾部肥厚): 高峰度 = 极端涨跌更频繁 -> direction = -1.
  # 出处: research_2026-09/factor_mining.py F_kurt60 / alpha_mining.py N_kurt60; 报告偏弱.
  return _ret(ds.wide('close'), 1).rolling(60, min_periods=60).kurt()


# ============================ 量能组 ============================
# 逻辑: 放量往往伴随趋势确认. 这是弱先验, 最终看 IC.

@register('vratio_20', '量能', 'volume[t] / mean(volume[t-19..t])', window=20)
def _vratio_20(ds: Dataset) -> pd.DataFrame:
  vol = ds.wide('volume')
  return vol / vol.rolling(20, min_periods=20).mean()


@register('vratio_60', '量能', 'volume[t] / mean(volume[t-59..t])', window=60)
def _vratio_60(ds: Dataset) -> pd.DataFrame:
  vol = ds.wide('volume')
  return vol / vol.rolling(60, min_periods=60).mean()


@register('obv_20', '量能',
          'sum(sign(ret_1) * volume, 20) / sum(volume, 20)', window=20)
def _obv_20(ds: Dataset) -> pd.DataFrame:
  # 归一化 OBV 变化: 20 日内"有向成交量"占总量之比, ∈ [-1, 1].
  #   > 0 = 成交更多发生在上涨日(资金吸筹) -> 看涨.
  # 出处: research_2026-09/factor_mining.py F_obv20; etf_3x h=5/20 C弱.
  # 注: 与"上涨日成交量占比"单调等价, 故只登记一个, 不重复造冗余因子.
  close, vol = ds.wide('close'), ds.wide('volume')
  flow = np.sign(_ret(close, 1)) * vol
  return (flow.rolling(20, min_periods=20).sum()
          / vol.rolling(20, min_periods=20).sum())


# ============================ 趋势组 ============================

@register('bias_20', '趋势', 'close[t] / MA20[t] - 1', window=20)
def _bias_20(ds: Dataset) -> pd.DataFrame:
  # 乖离率: 价格偏离均线的程度
  close = ds.wide('close')
  return close / _ma(close, 20) - 1


@register('bias_60', '趋势', 'close[t] / MA60[t] - 1', window=60)
def _bias_60(ds: Dataset) -> pd.DataFrame:
  close = ds.wide('close')
  return close / _ma(close, 60) - 1


@register('slope_20', '趋势', 'MA20[t] / MA20[t-20] - 1', window=40)
def _slope_20(ds: Dataset) -> pd.DataFrame:
  # 均线斜率: 趋势的方向与陡峭程度(不含当日价格, 比 bias 更"纯"的趋势度量)
  ma = _ma(ds.wide('close'), 20)
  return ma / ma.shift(20) - 1


@register('dd_60', '趋势', 'close[t] / max(close[t-59..t]) - 1', window=60)
def _dd_60(ds: Dataset) -> pd.DataFrame:
  # 距 60 日高点的回撤(<= 0); 越接近 0 = 越接近新高 = 越强势
  close = ds.wide('close')
  return close / close.rolling(60, min_periods=60).max() - 1


@register('hi_252', '趋势', 'close[t] / max(close[t-251..t]) - 1', window=252)
def _hi_252(ds: Dataset) -> pd.DataFrame:
  # 52 周高点接近度(George & Hwang, 2004): 越接近一年新高 = 越强势.
  # 与 dd_60 同构(都是 close / 区间最高 - 1, <= 0), 只是窗口 60 -> 252;
  #   第4步去重会检验它与 dd_60 是否冗余.
  close = ds.wide('close')
  return close / close.rolling(252, min_periods=252).max() - 1


@register('er_20', '趋势',
          '|close[t] - close[t-20]| / sum(|close[k] - close[k-1]|, 20)', window=20)
def _er_20(ds: Dataset) -> pd.DataFrame:
  # 趋势效率比(Kaufman ER): 20 日净位移 / 路径总长, ∈ [0, 1]; 越接近 1 = 越顺滑单边.
  # 出处: research_2026-09/factor_mining.py F_er20 / alpha_mining.py H_er_alpha;
  #   etf_3x h=20 C弱(exc≈0.0086).
  close = ds.wide('close')
  net = (close - close.shift(20)).abs()
  path = close.diff().abs().rolling(20, min_periods=20).sum()
  return net / path


@register('hihits_60', '趋势', 'count(close 创 60 日新高, 近 60 日)', window=120)
def _hihits_60(ds: Dataset) -> pd.DataFrame:
  # 60 日内创 60 日新高的次数(52 周高点的新高频率版).
  # 出处: research_2026-09/a_stat_mining.py W_hihits60; etf_3x 各 h 均为正.
  close = ds.wide('close')
  hh = close.rolling(60, min_periods=60).max()
  # where(hh.notna()) 让暖机期的 0.0 重新变回 NaN, 否则前 60 天会被当成"未创新高"计入窗口.
  is_new_high = (close >= hh).where(hh.notna()).astype(float)
  return is_new_high.rolling(60, min_periods=60).sum()


@register('martin_60', '趋势',
          'mean(ret_1, 60) / sqrt(mean(dd_60^2, 60))', window=60)
def _martin_60(ds: Dataset) -> pd.DataFrame:
  # Martin 比率(回撤版夏普): 单位"回撤痛感"换来的平均收益. 出处: research_2026-09/a_stat_mining.py W_martin60.
  close = ds.wide('close')
  r = _ret(close, 1)
  dd = close / close.rolling(60, min_periods=60).max() - 1
  ulcer = (dd ** 2).rolling(60, min_periods=60).mean() ** 0.5
  return r.rolling(60, min_periods=60).mean() / ulcer


@register('streak', '趋势', 'signed_run_length(sign(ret_1))', window=1)
def _streak(ds: Dataset) -> pd.DataFrame:
  # 连续同向天数(带符号): 连涨 k 天 = +k, 连跌 k 天 = -k, 翻转即重置.
  # 出处: research_2026-09/alpha_mining.py X_streak_alpha.
  # 逐列线性扫描(只读左侧), 因果; 用显式循环是为了让"翻转重置"一眼可读.
  close = ds.wide('close')
  sgn = np.sign(close / close.shift(1) - 1)

  def _one(col: pd.Series) -> pd.Series:
    v = col.to_numpy(dtype=float)
    out = np.full(v.shape, np.nan)
    run, prev = 0.0, np.nan
    for i, x in enumerate(v):
      if np.isnan(x):
        run, prev = 0.0, np.nan
        continue
      run = run + 1.0 if x == prev else 1.0
      prev = x
      out[i] = run * x
    return pd.Series(out, index=col.index)

  return sgn.apply(_one)


# ============================ alpha 化组(H_ 族) ============================
# 逻辑: pkl 面板里 adx / atr / rsi / kama / ichimoku 等 TA 指标都是"水平型" ——
#   数值大小随标的与时代漂移(2021 年的 RSI=60 与 2024 年的 RSI=60 不可比),
#   直接进截面排序没有意义. alpha 化 = 取 |x| 的"自身历史分位"(normalize_causal, 252/60):
#   既去掉量纲, 又保留"这只标的此刻处于自己历史的什么位置" -> 可跨标的比较.
# 出处: research_2026-09/alpha_mining.py 的 _alphaize + build_alpha_cands(H_ 族 21 个);
#   H_ichimoku_alpha 与该面板的 m_trend_score_alpha 同构, 是一致性锚点.
# direction 统一 +1: 分位本身没有方向先验(0 = 自身历史最低, 1 = 最高), 到底"高分位看涨"
#   还是"高分位看跌"由第3步实测 IC 决定 —— 硬填符号只会把 dir_ok 变成噪声.

def _alpha_reg(entries: list) -> None:
  """批量注册 H_ 族: entries = [(因子名, 公式, 原始水平量构造函数), ...]."""
  for name, formula, src in entries:
    def fn(ds: Dataset, _src=src) -> pd.DataFrame:
      return _alphaize(_src(ds))
    register(name, 'alpha化', formula, cfg.NORM_WINDOW)(fn)


_alpha_reg([
    ('H_ichimoku_alpha', 'normalize_causal(|ichimoku_distance|)',
     lambda ds: ds.wide('ichimoku_distance')),
    ('H_trendmag_alpha', 'normalize_causal(|trend_magnitude|)',
     lambda ds: ds.wide('trend_magnitude')),
    ('H_adxval_alpha', 'normalize_causal(|adx_value|)',
     lambda ds: ds.wide('adx_value')),
    ('H_adxstr_alpha', 'normalize_causal(|adx_strength|)',
     lambda ds: ds.wide('adx_strength')),
    ('H_adxdist_alpha', 'normalize_causal(|adx_distance|)',
     lambda ds: ds.wide('adx_distance')),
    ('H_adxpow_alpha', 'normalize_causal(|adx_power|)',
     lambda ds: ds.wide('adx_power')),
    ('H_atr_alpha', 'normalize_causal(|atr|)',
     lambda ds: ds.wide('atr')),
    ('H_trpct_alpha', 'normalize_causal(|tr / close|)',
     lambda ds: ds.wide('tr') / ds.wide('close')),
    ('H_bbw_alpha', 'normalize_causal(|(bb_high_band - bb_low_band) / mavg|)',
     lambda ds: ((ds.wide('bb_high_band') - ds.wide('bb_low_band'))
                 / ds.wide('mavg').replace(0, np.nan))),
    ('H_rsidev_alpha', 'normalize_causal(|rsi - 50|)',
     lambda ds: ds.wide('rsi') - 50.0),
    ('H_kamadist_alpha', 'normalize_causal(|kama_distance|)',
     lambda ds: ds.wide('kama_distance')),
    ('H_gapdist_alpha', 'normalize_causal(|candle_gap_distance|)',
     lambda ds: ds.wide('candle_gap_distance')),
    ('H_body_alpha', 'normalize_causal(|candle_entity_pct|)',
     lambda ds: ds.wide('candle_entity_pct')),
    ('H_shadow_alpha', 'normalize_causal(|candle_upper_shadow_pct + candle_lower_shadow_pct|)',
     lambda ds: (ds.wide('candle_upper_shadow_pct')
                 + ds.wide('candle_lower_shadow_pct'))),
    ('H_volratio_alpha', 'normalize_causal(|volume / MA20(volume)|)',
     lambda ds: (ds.wide('volume')
                 / ds.wide('volume').rolling(20, min_periods=20).mean().replace(0, np.nan))),
    ('H_volchg_alpha', 'normalize_causal(|volume_change|)',
     lambda ds: ds.wide('volume_change')),
    ('H_entitydiff_alpha', 'normalize_causal(|entity_diff|)',
     lambda ds: ds.wide('entity_diff')),
    ('H_pos_alpha', 'normalize_causal(|candle_position_score|)',
     lambda ds: ds.wide('candle_position_score')),
    ('H_trigger_alpha', 'normalize_causal(|trigger_score|)',
     lambda ds: ds.wide('trigger_score')),
    ('H_pattern_alpha', 'normalize_causal(|pattern_score|)',
     lambda ds: ds.wide('pattern_score')),
    ('H_er_alpha', 'normalize_causal(|er_20|)',
     lambda ds: _er_20(ds)),
])


# ============================ 残差动量组 ============================
# 逻辑: 把个股收益里"跟着全池一起动"的部分(beta × 池收益)剔掉, 只留个体特异的残差,
#   再对残差求动量. Blitz-Huij-Martens(2011) Residual Momentum: 残差动量比裸动量
#   更稳(不含市场成分, 不受市场整体涨跌反转拖累), 与现成的 mom_* 族正交.

@register('R_resmom121', '残差动量',
          'sum(resid[t-251..t-21], 231); resid = ret_1 - beta60 * pool_ret',
          window=252)
def _R_resmom121(ds: Dataset) -> pd.DataFrame:
  # 残差: 60 日滚动 beta 对"池等权收益"回归后的残差(ret_1 - mean - beta*(pr - mean_pr)).
  # 残差动量 = 跳过最近 21 天(避开短期反转)后, 对过去 231 天残差求和(经典 12-1 结构).
  # 出处: research_2026-09/a_stat_mining.py R_resmom121; 与 mom_12_1 的区别 = 剔掉市场成分.
  close = ds.wide('close')
  ret1 = _ret(close, 1)
  pr = _pool_ret(close)
  rb, mp = 60, 30
  pr_ma = pr.rolling(rb, min_periods=mp).mean()
  beta = (ret1.rolling(rb, min_periods=mp).cov(pr)
          .div(pr.rolling(rb, min_periods=mp).var(), axis=0))
  resid = (ret1.sub(ret1.rolling(rb, min_periods=mp).mean())
           .sub(beta.mul(pr.sub(pr_ma), axis=0)))
  return resid.shift(21).rolling(231, min_periods=120).sum()


# ============================ 信息离散组 ============================
# 逻辑: 信息离散度(ID, Da-Gurun-Warachka 2014) = 下跌日占比 - 上涨日占比, 再带上动量符号.
#   离散度高 = 收益分布被少数大涨日拉起来(信息到来是"跳跃式"的) -> 信息风险溢价,
#   预期收益更高. 只用 OHLCV, 与 mom_* 的信息维度不同.

@register('I_id60', '信息离散',
          'sign(close[t]/close[t-60] - 1) * (dn_share60 - up_share60)', window=60)
def _I_id60(ds: Dataset) -> pd.DataFrame:
  # up_share / dn_share = 过去 60 天里"上涨日/下跌日"的频率(平盘两边都不计).
  # 带 sign(mom60): 同幅度的离散, 上涨途中的离散与下跌途中的离散含义相反.
  # 出处: research_2026-09/a_stat_mining.py I_id60.
  close = ds.wide('close')
  ret1 = _ret(close, 1)
  up = (ret1 > 0).where(ret1.notna()).astype(float)
  dn = (ret1 < 0).where(ret1.notna()).astype(float)
  upsh = up.rolling(60, min_periods=30).mean()
  dnsh = dn.rolling(60, min_periods=30).mean()
  return np.sign(_ret(close, 60)) * (dnsh - upsh)


# ============================ 微观结构组(G_ 族) ============================
# 逻辑: 从 OHLCV 里拆出"隔夜/日内/极值/方差比/beta 不对称"这类结构量, 与 mom_* 正交.
# 出处: research_2026-09/factor_mining2.py G_ 族. 除 G_dnbeta60 外均无强方向先验, 登记 +1,
#   符号以实测 IC 为准(min_periods 沿用参考实现的 窗口//2).

@register('G_parkcc20', '微观结构',
          'sqrt(Parkinson_var20) / sqrt(cc_var20); Parkinson_var = mean(ln(H/L)^2) / (4 ln2)',
          window=20)
def _G_parkcc20(ds: Dataset) -> pd.DataFrame:
  # Parkinson(高低价)波动 / 收盘-收盘波动: 比值高 = 波动主要发生在日内(连续), 低 = 靠跳空.
  # 出处: research_2026-09/factor_mining2.py G_parkcc20(Alizadeh 等). 结构比值, 无方向先验.
  high, low, close = ds.wide('high'), ds.wide('low'), ds.wide('close')
  ln_hl = np.log((high / low).where(high > low))
  park_var = ln_hl.pow(2).rolling(20, min_periods=10).mean() / (4.0 * np.log(2.0))
  cc_var = _ret(close, 1).rolling(20, min_periods=10).var()
  return np.sqrt(park_var) / np.sqrt(cc_var.replace(0, np.nan))


@register('G_oviv20', '微观结构',
          'std(overnight, 20) / std(intraday, 20); overnight = Open/Close[t-1]-1',
          window=20)
def _G_oviv20(ds: Dataset) -> pd.DataFrame:
  # 隔夜波动 / 日内波动: 隔夜跳空占波动比重高 = 信息在非交易时段释放(风险结构不同).
  # 出处: research_2026-09/factor_mining2.py G_oviv20(Lou-Polk-Skouloudakis 式分解). 无方向先验.
  open_, close = ds.wide('open'), ds.wide('close')
  overnight = open_ / close.shift(1) - 1.0
  intraday = close / open_ - 1.0
  return (overnight.rolling(20, min_periods=10).std()
          / intraday.rolling(20, min_periods=10).std().replace(0, np.nan))


@register('G_vr20', '微观结构',
          'var(close[t]/close[t-20]-1, 120) / (20 * var(ret_1, 120))', window=140)
def _G_vr20(ds: Dataset) -> pd.DataFrame:
  # 方差比(Lo-MacKinlay): 20 日收益方差 / (20 × 1 日收益方差). > 1 = 趋势延续, < 1 = 均值回归.
  # 出处: research_2026-09/factor_mining2.py G_vr20. 先验 +1(>1 的趋势性更强 -> 动量方向).
  close = ds.wide('close')
  ret1 = _ret(close, 1)
  var_q = _ret(close, 20).rolling(120, min_periods=60).var()
  var_1 = ret1.rolling(120, min_periods=60).var()
  return var_q / (20.0 * var_1).replace(0, np.nan)


@register('G_dnbeta60', '微观结构',
          'beta60(ret_1 | pool_down) - beta60(ret_1 | pool_up)', window=60,
          direction=-1)
def _G_dnbeta60(ds: Dataset) -> pd.DataFrame:
  # 下行 beta - 上行 beta(Ang-Chen-Xing 2006 不对称 beta): 越负 = 池下跌时反而更抗跌.
  # 出处: research_2026-09/factor_mining2.py G_dnbeta60. 先验 -1(不对称风险高者在本池更差).
  close = ds.wide('close')
  ret1 = _ret(close, 1)
  pr = _pool_ret(close)
  # 用"池当日涨/跌"给截面分状态(only t 时刻截面), 再在两种状态下各算 60 日滚动 beta.
  r_dn = ret1.where(pr < 0, axis=0)
  r_up = ret1.where(pr > 0, axis=0)
  pr_dn, pr_up = pr.where(pr < 0), pr.where(pr > 0)

  def _beta(r, p):
    return (r.rolling(60, min_periods=15).cov(p)
            .div(p.rolling(60, min_periods=15).var().replace(0, np.nan), axis=0))

  return _beta(r_dn, pr_dn) - _beta(r_up, pr_up)


# ============================ 门控组(K_ 族) ============================
# 逻辑: 条件/门控因子 —— 把基础动量乘上一个"环境开关", 只在有利环境里放行动量.
# 出处: research_2026-09/conditional_mining.py K_ 族(Daniel-Moskowitz 1998 动量崩溃防御).

@register('K_mom_er', '门控', 'mom60 * er20', window=60)
def _K_mom_er(ds: Dataset) -> pd.DataFrame:
  # 动量 × 趋势效率: er20 高(单边顺滑)时动量才可信, er20 低(来回震荡)时把动量压小.
  # 出处: research_2026-09/conditional_mining.py K_mom_er. 先验 +1.
  return _ret(ds.wide('close'), 60) * _er_20(ds)


@register('K_mom_poolvolq', '门控',
          'mom60 * (1 - 2 * normalize_causal(std(pool_ret, 20), 252, 60))', window=252)
def _K_mom_poolvolq(ds: Dataset) -> pd.DataFrame:
  # 池波动状态门控(Daniel-Moskowitz): 池波动处于自身历史高位(崩溃后)时, 动量收益会反转,
  #   故按"池波动分位"把动量线性压到 [-1, 1] 倍(分位低 -> 1 倍, 分位高 -> -1 倍, 反手).
  # 出处: research_2026-09/conditional_mining.py K_mom_poolvolq. 先验 +1.
  close = ds.wide('close')
  pv = _pool_ret(close).rolling(20, min_periods=10).std()
  volq = normalize_causal(pv).where(pv.notna())   # 池波动在自身历史中的分位 ∈ [0,1]
  return _ret(close, 60).mul(1.0 - 2.0 * volq, axis=0)


# ============================ 旧预设兼容组(C_/N_ 族) ============================
# 逻辑: 旧系统 signal_bridge.py 的 5 组预设权重里, 有 3 个因子名不在注册表; 这里按
#   repro_old_presets.py 的定义逐字补进来, 让 cd_ic 导出的权重可直接被生产端消费.
# 出处: repro_old_presets.py native_signals / legacy_factors.

def _cs_rank_pct(v: pd.DataFrame) -> pd.DataFrame:
  """逐日截面 pct 排名; 默认 na_option='keep', 暖机 / 停牌 NaN 原样保留."""
  return v.rank(axis=1, pct=True)


@register('C_tmqmom', '动量',
          'rank_pct(mom_12_1) + rank_pct(er_20)', window=252)
def _C_tmqmom(ds: Dataset) -> pd.DataFrame:
  # 截面例外 c): 动量与趋势效率各自 pct 排名后相加 —— 名次可比, 尺度一致.
  # 与旧系统 C_tmqmom = rank_pct(F_mom121) + rank_pct(F_er20) 同构. 先验 +1.
  return _cs_rank_pct(_mom_12_1(ds)) + _cs_rank_pct(_er_20(ds))


@register('N_range20', '波动', '-(high - low).mean(20) / close', window=20)
def _N_range20(ds: Dataset) -> pd.DataFrame:
  # 负向 20 日平均振幅(以收盘价归一): 振幅越小(值越大)越看涨 —— 低波动异象. 先验 +1.
  # 出处: repro_old_presets.py _range20(旧 N_range20).
  high, low, close = ds.wide('high'), ds.wide('low'), ds.wide('close')
  return -(high - low).rolling(20, min_periods=20).mean() / close


@register('N_idiovol60', '波动',
          '-std(resid, 60); resid = ret_1 - beta60 * mean_over_pool(ret_1)',
          window=60)
def _N_idiovol60(ds: Dataset) -> pd.DataFrame:
  # 负向 60 日特质波动(对池等权收益回归取残差): 特质波动越低(值越大)越看涨. 先验 +1.
  # 出处: repro_old_presets.py _idiovol60(旧 N_idiovol60).
  close = ds.wide('close')
  ret1 = _ret(close, 1)
  pr = _pool_ret(close)
  var_pool = pr.rolling(60, min_periods=60).var().replace(0, np.nan)
  beta60 = ret1.rolling(60, min_periods=60).cov(pr).div(var_pool, axis=0)
  resid = ret1.sub(beta60.mul(pr, axis=0))
  return -resid.rolling(60, min_periods=60).std()


# ============================ 注册表访问 ============================

def list_factors(group: str = None) -> list:
  """
  列出因子名.

  :param group: 只列某类(None = 全部)
  :returns: 因子名列表(按注册顺序)
  """
  return [n for n, s in REGISTRY.items() if group is None or s.group == group]


def get_factor(name: str) -> FactorSpec:
  """按名取因子; 不存在则报错并给出可用名."""
  if name not in REGISTRY:
    raise KeyError(f'未知因子: {name}\n可用因子: {list_factors()}')
  return REGISTRY[name]


def compute_all(ds: Dataset, group: str = None) -> dict:
  """
  一次算多个因子.

  :param ds: Dataset
  :param group: 只算某类(None = 全部)
  :returns: {因子名: 宽表}
  """
  return {n: get_factor(n).compute(ds) for n in list_factors(group)}


def factor_table() -> pd.DataFrame:
  """因子字典表(名/类别/公式/名义窗口/先验方向), 供打印与文档."""
  return pd.DataFrame([{'factor': s.name, 'group': s.group, 'formula': s.formula,
                        'window': s.window, 'direction': s.direction}
                       for s in REGISTRY.values()])


# ============================ 自检: 总览 + 因果性 ============================

def overview(ds: Dataset, group: str = None) -> pd.DataFrame:
  """
  因子总览: 实测暖机天数 / 有效值占比 / 每日截面有效标的数中位数.

  暖机天数是实测的(第一个出现非 NaN 的交易日在该因子序列中的位置),
  而不是用 window 猜 —— 差一天都会让第3步的窗口对齐出错.

  :param ds: Dataset
  :param group: 只统计某类
  :returns: DataFrame, 每行一个因子
  """
  rows = []
  for name in list_factors(group):
    spec = REGISTRY[name]
    df = spec.compute(ds)
    valid = df.notna().any(axis=1)
    warmup = int(valid.values.argmax()) if valid.any() else len(df)
    rows.append({
        'factor': name,
        'group': spec.group,
        'window': spec.window,
        'direction': spec.direction,
        'warmup': warmup,                        # 实测: 前 warmup 个交易日全 NaN
        'coverage': round(float(df.notna().mean().mean()), 4),
        'cs_median': int(df.notna().sum(axis=1).median()),
        'formula': spec.formula,
    })
  return pd.DataFrame(rows)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
  """忽略 NaN 位置的最大绝对差; 全为 NaN 时返回 0.0."""
  d = np.abs(a - b)
  with np.errstate(invalid='ignore'):
    if np.isnan(d).all():
      return 0.0
    return float(np.nanmax(d))


def verify_causal(ds: Dataset, cut: str = '2024-06-28',
                  tol: float = 1e-9) -> pd.DataFrame:
  """
  截断一致性检验 —— 本体系防前视的核心防线.

  原理: 因果因子在 t 时刻的值只依赖 <= t 的数据, 所以"喂给它多少未来数据"
        不应改变 t 时刻的结果.
          A = 用全历史算因子, 取 <= cut 的部分
          B = 只把 <= cut 的数据喂进去再算
        若 A 与 B 逐值相同 -> 因果; 若不同 -> 该因子偷看了未来.

  为什么需要它: "全历史计算"本身不是问题(rolling 是全历史算 == 逐点算),
        真正会出问题的是夹带的全样本统计量(不带窗口的 mean/std/quantile/rank)、
        双向滤波等. 靠肉眼看代码容易漏, 靠这个检验能一次全查出来.

  :param ds: Dataset
  :param cut: 截断日(检验点)
  :param tol: 允许的最大绝对差
  :returns: DataFrame(factor, group, cells, max_diff, nan_same, causal)
  """
  part = ds.slice(end=cut)
  rows = []
  for name, spec in REGISTRY.items():
    full = spec.compute(ds).loc[:cut]            # A: 全历史算完再截
    trunc = spec.compute(part)                   # B: 只喂截断后的数据
    idx = full.index.intersection(trunc.index)
    col = full.columns.intersection(trunc.columns)
    a, b = full.loc[idx, col], trunc.loc[idx, col]
    shape_ok = a.shape == b.shape and a.size > 0
    if shape_ok:
      nan_same = not bool((np.isnan(a.values) != np.isnan(b.values)).any())
      max_diff = _max_abs_diff(a.values, b.values)
    else:
      nan_same, max_diff = False, float('nan')
    rows.append({
        'factor': name,
        'group': spec.group,
        'cells': int(a.size) if shape_ok else 0,
        'max_diff': max_diff,
        'nan_same': nan_same,
        'causal': bool(shape_ok and nan_same and max_diff <= tol),
    })
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser(description='factor 因子层自检: 总览 + 因果性检验')
  ap.add_argument('--pool', default=cfg.DEFAULT_POOL)
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  ap.add_argument('--group', default=None, help='只显示某类因子')
  ap.add_argument('--check', action='store_true', help='做截断一致性检验')
  ap.add_argument('--cut', default='2024-06-28', help='检验用的截断日')
  ap.add_argument('--show', default=None, help='打印该因子最近 5 行')
  a = ap.parse_args()

  log = cfg.get_logger('factor.factor')
  ds = load_pool(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  log.info(f'\n[因子总览] 共 {len(list_factors(a.group))} 个'
           + (f' (group={a.group})' if a.group else ''))
  log.info('\n' + overview(ds, a.group).to_string(index=False))

  if a.check:
    ck = verify_causal(ds, a.cut)
    log.info(f'\n[截断一致性检验] 全历史算 vs 只喂到 {a.cut}')
    log.info('\n' + ck.to_string(index=False))
    bad = ck.loc[~ck['causal'], 'factor'].tolist()
    log.info('[结论] 全部因子因果' if not bad else f'[结论] 疑似前视: {bad}')

  if a.show:
    df = get_factor(a.show).compute(ds)
    log.info(f'\n[{a.show}] {get_factor(a.show).formula}\n' + df.tail(5).to_string())


if __name__ == '__main__':
  main()

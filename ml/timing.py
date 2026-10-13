# -*- coding: utf-8 -*-
"""
ml.timing — 择时层(第5步, 方案B: 波动率目标)
=============================================
职责: 在"选股分数"之外, 再产出一条**逐日总敞口**序列 e_t ∈ [0, 1],
      交给 factor.backtest.run_engine(exposure=...) 控制组合总仓位.

--- 为什么单做一层, 而不是让选股模型直接预测仓位 ---
  现有标签 y 是**逐日截面 rank**([-1, 1], 尺度无关), 只表达"谁相对强", 不含任何
  "该不该满仓"的信息. 要让模型输出仓位, 必须换一条**带尺度**的预测目标. 于是把
  "选股(alpha)"与"择时(风险)"拆成两层: 选股层照旧出分数, 择时层只回答
  "今天整体该上多少仓".

--- 方案 B: 波动率目标(volatility targeting) ---
  核心不是预测涨跌方向(极难且易过拟合), 而是预测**未来一段时间的市场波动率**,
  再按"波动率越高、仓位越低"把总敞口压下来:
        e_t = clip( target_vol / pred_vol_t , floor, cap )
  用户红线: 不做空(floor=0) 且 不加杠杆(cap=1) -> e_t ∈ [0, 1].
  target_vol **只在 IS 上估计**(取 IS 市场实现波动率的中位数), OOS/recent 只做汇报,
  否则等于用未来信息定标(前视).

--- 本层输出(供 t3 训练 / t4 对比) ---
  1. build_market_features(ds): 池聚合的**市场级**时间序列特征(逐日, 全部因果);
  2. build_timing_label(ds):   未来 h 日市场实现波动率(标签, 必须吃未来);
  3. vol_target_exposure(...): 预测波动率 -> 逐日敞口的纯映射(纯函数, 无拟合);
  4. build_exposure(ds):       波动率目标臂(方案B, t3 主体) —— 连续 e_t = clip(tgt/pred,0,1);
  5. build_regime_exposure(ds): 状态分类臂(t4 第三臂) —— IS 分位二值化高/低波动状态 + 低容量
     线性概率模型, 再映射为**离散档位**敞口(低波动满仓 / 高波动降仓). 见第 7 节.

--- 因果性 ---
  市场特征全部由 ds 的价量宽表经 pct_change / rolling 现算, 只读 <= t 的观测,
  可用**截断一致性检验**证明(verify_market_features): "全历史算再截" 与 "先截再算"
  在被保留行上逐值相同 -> 因果. 这与 features.py 里 TA 列无法验证的情形不同:
  那些是 pkl 预计算值(截行不改值, 检验恒等失效); 这里是本包按需现算的.

--- 标签为何"必须吃未来" ---
  与选股标签同理: 信号日 t 决定 t+1 起的仓位, 而 e_t 要匹配"接下来真正发生的波动".
  故标签 = t+1..t+h 的实现波动率(用未来), 末尾 h 行必然 NaN. 这是**评估口径**,
  与"特征禁未来"并不矛盾.

自检:
  cd ~/git && python -m quant.ml.timing --pool etf_3x
  cd ~/git && python -m quant.ml.timing --pool etf_3x --check
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor import evaluate as ev
from quant.factor import factor as fct
from quant.factor.data import Dataset

from quant.ml import config as cfg
from quant.ml.data import load

ANNUAL = fct.ANNUAL                          # 年化交易日数(与 factor 同源)

# risk-on/off(风险偏好)参数 —— 见 risk_appetite_spread(标签) / risk_appetite_momentum(特征):
RISK_APPETITE_VOL_WINDOW = cfg.TIMING_VOL_SHORT   # 个股风险度(收益波动)的滚动窗口
RISK_APPETITE_GROUP_Q = 0.3                       # 高/低风险组的截面分位(顶部/底部各 30%)
RISK_APPETITE_MOM_WINDOW = cfg.TIMING_MOM_WINDOW  # 风险偏好**特征**的回看窗口(过去 window 日实现价差)

# 可选扩展特征: 默认不进 MKT_FEATURES, 由调用方显式开启(见 build_market_features)
EXT_FEATURES = ('riskon',)   # 风险偏好价差(过去 RISK_APPETITE_MOM_WINDOW 日, 因果)

# 市场特征的列名(顺序即输出列顺序) —— 全部为"t 时刻已知"的池聚合量
MKT_FEATURES = ('ret_1', 'vol_s', 'vol_l', 'vol_ratio', 'mom',
                'breadth', 'dispersion', 'dd', 'volume_z')


# ============================ 1. 市场级序列(单变量) ============================

def market_returns(ds: Dataset, price: str = 'close') -> pd.Series:
  """
  池等权日收益序列(逐日).

  口径: 每个交易日先算各标的日收益(price(t)/price(t-1)-1), 再在截面上等权平均
  (skipna). 只读 <= t 的价格 -> 因果.

  :param ds: Dataset
  :param price: 'close'(默认) 或 'open'
  :returns: Series(index=date), 值 = 当日市场等权收益
  """
  w = ds.wide(price)
  return w.pct_change().mean(axis=1, skipna=True)


def market_nav(mkt: pd.Series) -> pd.Series:
  """市场等权净值(1 起步, 按日收益累乘). 因果(只累乘到 t). NaN 收益按 0 处理."""
  return (1.0 + mkt.fillna(0.0)).cumprod()


# ============================ 2. 市场特征(逐日, 因果) ============================

def build_market_features(ds: Dataset, vol_short: int = None, vol_long: int = None,
                          mom_window: int = None, dd_window: int = None,
                          riskon_feat: bool = False, log=None) -> pd.DataFrame:
  """
  池聚合的**市场级**时间序列特征(逐日), 供择时模型吃.

  全部由价量宽表现算, 只用 <= t 的观测 -> 因果(见 verify_market_features):
    ret_1      当日市场等权收益
    vol_s      过去 vol_short 日市场收益的年化标准差(短期波动)
    vol_l      过去 vol_long  日市场收益的年化标准差(长期波动)
    vol_ratio  vol_s / vol_l(波动率期限比: >1 表示短期波动抬升)
    mom        过去 mom_window 日市场净值累计收益(市场动量)
    breadth    过去 mom_window 日"上涨家数占比"的均值(市场宽度)
    dispersion 过去 mom_window 日"截面收益离散度"的均值
    dd         市场净值相对过去 dd_window 日高点的回撤(<=0)
    volume_z   池成交量对数相对长期均值的 z 分数(量能偏离)
  riskon_feat=True 时**额外**输出(见 EXT_FEATURES / risk_appetite_momentum):
    riskon     过去 RISK_APPETITE_MOM_WINDOW 日"高波组 - 低波组"实现超额收益(风险偏好, 因果)

  :param ds: Dataset
  :param vol_short/vol_long/mom_window/dd_window: 各特征窗口(None = 用 config 默认)
  :param riskon_feat: 是否追加 riskon(风险偏好)扩展特征列
  :param log: 可选 logger
  :returns: DataFrame(index=date, columns=MKT_FEATURES[+EXT_FEATURES]), float
  """
  vol_short = cfg.TIMING_VOL_SHORT if vol_short is None else vol_short
  vol_long = cfg.TIMING_VOL_LONG if vol_long is None else vol_long
  mom_window = cfg.TIMING_MOM_WINDOW if mom_window is None else mom_window
  dd_window = cfg.TIMING_DD_WINDOW if dd_window is None else dd_window

  ret = ds.wide('close').pct_change()                  # 逐标的日收益(因果)
  mkt = ret.mean(axis=1, skipna=True)                  # 市场等权日收益
  nav = market_nav(mkt)                                # 市场等权净值

  n_valid = ret.notna().sum(axis=1)
  up_ratio = (ret > 0).sum(axis=1) / n_valid.replace(0, np.nan)   # 上涨家数占比
  disp = ret.std(axis=1, skipna=True)                             # 截面离散度

  vol_s = mkt.rolling(vol_short).std() * np.sqrt(ANNUAL)
  vol_l = mkt.rolling(vol_long).std() * np.sqrt(ANNUAL)

  volume = ds.wide('volume').sum(axis=1, skipna=True)             # 池成交量聚合
  lv = np.log(volume.replace(0.0, np.nan))
  vol_z = (lv - lv.rolling(vol_long).mean()) / lv.rolling(vol_long).std()

  data = {
      'ret_1': mkt,
      'vol_s': vol_s,
      'vol_l': vol_l,
      'vol_ratio': vol_s / vol_l,
      'mom': nav.pct_change(mom_window),
      'breadth': up_ratio.rolling(mom_window).mean(),
      'dispersion': disp.rolling(mom_window).mean(),
      'dd': nav / nav.rolling(dd_window).max() - 1.0,
      'volume_z': vol_z,
  }
  cols = list(MKT_FEATURES)
  if riskon_feat:
    data['riskon'] = risk_appetite_momentum(ds)      # 追加风险偏好(因果)特征
    cols += list(EXT_FEATURES)
  feat = pd.DataFrame(data)[cols]

  if log is not None:
    v = feat.to_numpy(dtype=float)
    log.info(f'[mkt-feat] {len(feat)} 天 × {feat.shape[1]} 特征, 有效值占比 '
             f'{np.isfinite(v).mean():.1%}; 窗口 vol {vol_short}/{vol_long}, '
             f'mom {mom_window}, dd {dd_window}'
             + (f', +riskon(窗口 {RISK_APPETITE_MOM_WINDOW})' if riskon_feat else ''))
  return feat


def rolling_zscore(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
  """
  逐列的**因果** z-score: 用"过去 window 期(含当期)"的均值/标准差标准化.

  为什么不用全局标准化: 在整段样本上 fit mean/std 会用到未来 -> 前视. 滚动标准化
  只读 <= t 的观测, 因果; 代价是开头 window 期内为 NaN(暖机).

  :param df: (date × feature) 宽表
  :param window: 滚动窗口(None = 用 TIMING_DD_WINDOW, 约一年)
  :returns: 同形状 DataFrame, 已标准化
  """
  window = cfg.TIMING_DD_WINDOW if window is None else window
  mp = max(2, window // 4)
  mu = df.rolling(window, min_periods=mp).mean()
  sd = df.rolling(window, min_periods=mp).std()
  return (df - mu) / sd


# ============================ 3. 标签(波动率目标, 用未来) ============================

def forward_vol(mkt: pd.Series, h: int, annual: int = ANNUAL) -> pd.Series:
  """
  未来 h 日市场实现波动率(年化) —— **标签**, 必须吃未来.

  口径: fwd_vol[t] = std( mkt[t+1 .. t+h] ) * sqrt(annual); 末尾 h 行无未来 -> NaN.
  与选股标签同构(信号日 t 决定 t+1 起的持仓, 敞口要匹配接下来真正发生的波动).

  :param mkt: 市场日收益序列(index=date)
  :param h: 未来窗口(交易日)
  :param annual: 年化交易日数
  :returns: Series(index=date), 值 = 未来 h 日年化实现波动率
  """
  # 向量化等价实现: 先朝前移一格(shift(-1)), 反转后做窗口 h 的滚动 std, 再反转回来 ——
  # 反转使滚动窗口恰好对齐到 mkt[t+1 .. t+h]. 用 replace(±inf -> NaN) + rolling(min_periods=2)
  # 复刻原循环里的 np.isfinite 过滤与"至少 2 个有限值"判据(ddof=1); 末尾 h 行无未来 -> 强制 NaN.
  s = mkt.replace([np.inf, -np.inf], np.nan)
  rv = s.shift(-1)[::-1].rolling(h, min_periods=min(2, h)).std(ddof=1)[::-1]
  rv.iloc[max(len(rv) - h, 0):] = np.nan
  return rv * np.sqrt(annual)


def forward_return(mkt: pd.Series, h: int) -> pd.Series:
  """
  未来 h 日市场累计收益(趋势/牛熊标签) —— **标签**, 必须吃未来.

  口径: fwd_ret[t] = (1+mkt[t+1..t+h]) 连乘 - 1; 末尾 h 行无未来 -> NaN.
  与 forward_vol 同构(信号日 t 决定 t+1 起的持仓, 敞口要匹配接下来真正发生的收益).

  :param mkt: 市场日收益序列(index=date)
  :param h: 未来窗口(交易日)
  :returns: Series(index=date), 值 = 未来 h 日市场累计收益
  """
  # 与 forward_vol 同款对齐: 先 shift(-1), 反转做窗口求和(对数收益可加), 再反转回来;
  # 窗口内至少 2 个有限值(min_periods), 末尾 h 行无未来 -> 强制 NaN.
  s = mkt.replace([np.inf, -np.inf], np.nan)
  lg = np.log1p(s)
  fwd = lg.shift(-1)[::-1].rolling(h, min_periods=max(2, h)).sum()[::-1]
  fwd.iloc[max(len(fwd) - h, 0):] = np.nan
  return np.expm1(fwd)


def risk_appetite_spread(ds: Dataset, h: int, vol_window: int = None,
                         group_q: float = None) -> pd.Series:
  """
  未来 h 日"高风险组 相对 低风险组"的超额收益(risk-on/risk-off 标签) —— **标签**, 必须吃未来.

  为什么这么构造: 4 个池内都没有债/黄金等避险资产, 无法做"风险资产 vs 避险资产"的跨资产价差.
  但风险偏好(risk appetite)在**股内截面**上有清晰印记 —— risk-on 时资金涌向高波动/高 beta
  的进攻性标的, risk-off 时转向低波动防御标的. 于是:

    spread[t] = 高波动组 未来 h 日平均收益 - 低波动组 未来 h 日平均收益
                (分组用 <= t 的个股收益波动, 因果; 未来收益吃未来)

  spread 高 => 高风险资产跑赢 => risk-on(风险偏好升温) -> 模型判"状态1=risk-on" -> 满仓.

  :param ds: Dataset
  :param h: 未来窗口(交易日)
  :param vol_window: 个股风险度(收益波动)的滚动窗口(None = RISK_APPETITE_VOL_WINDOW)
  :param group_q: 高/低风险组的截面分位(None = RISK_APPETITE_GROUP_Q, 各取顶部/底部 q)
  :returns: Series(index=date), 风险偏好价差(日频); 无未来/截面过薄处 NaN
  """
  vol_window = RISK_APPETITE_VOL_WINDOW if vol_window is None else int(vol_window)
  group_q = RISK_APPETITE_GROUP_Q if group_q is None else float(group_q)
  vol = ds.wide('close').pct_change().rolling(vol_window).std()   # 逐标的因果风险度
  fwd = ev.forward_return(ds, h).reindex(index=vol.index, columns=vol.columns)
  v = vol.to_numpy(dtype=float)
  f = fwd.to_numpy(dtype=float)
  out = np.full(v.shape[0], np.nan)
  min_n = max(fcfg.MIN_CS, 2)                    # 截面太薄则不分组(沿用 MIN_CS 口径)
  for i in range(v.shape[0]):
    ok = np.isfinite(v[i])
    if ok.sum() < min_n:
      continue
    vals = v[i][ok]
    hi = ok & (v[i] >= np.quantile(vals, 1.0 - group_q)) & np.isfinite(f[i])
    lo = ok & (v[i] <= np.quantile(vals, group_q)) & np.isfinite(f[i])
    if hi.any() and lo.any():
      out[i] = float(f[i][hi].mean() - f[i][lo].mean())
  return pd.Series(out, index=vol.index, dtype=float)


def risk_appetite_momentum(ds: Dataset, window: int = None, vol_window: int = None,
                           group_q: float = None) -> pd.Series:
  """
  过去 window 日"高风险组 相对 低风险组"的**实现**超额收益(risk-on/off 特征) —— 只吃 <= t, 因果.

  与 risk_appetite_spread(标签)镜像: 那里用"未来 h 日"收益(吃未来)当标签, 这里改用"过去 window 日"
  已实现收益(<= t)当特征. 分组仍用 <= t 的个股收益波动 -> 整条链因果, 可作择时模型的特征列.

    feature[t] = 高波动组 过去 window 日平均收益 - 低波动组 过去 window 日平均收益
                 (分组用 <= t 的个股波动; 收益用 <= t 的已实现收益)

  feature 高 => 高风险资产近期跑赢 => 风险偏好升温(risk-on) -> 供状态模型参考.

  :param ds: Dataset
  :param window: 回看窗口(交易日; None = RISK_APPETITE_MOM_WINDOW)
  :param vol_window: 个股风险度(收益波动)的滚动窗口(None = RISK_APPETITE_VOL_WINDOW)
  :param group_q: 高/低风险组的截面分位(None = RISK_APPETITE_GROUP_Q, 各取顶部/底部 q)
  :returns: Series(index=date), 风险偏好价差(日频, 因果); 暖机/截面过薄处 NaN
  """
  window = RISK_APPETITE_MOM_WINDOW if window is None else int(window)
  vol_window = RISK_APPETITE_VOL_WINDOW if vol_window is None else int(vol_window)
  group_q = RISK_APPETITE_GROUP_Q if group_q is None else float(group_q)
  close = ds.wide('close')
  vol = close.pct_change().rolling(vol_window).std()   # 逐标的因果风险度(<= t)
  past = close / close.shift(window) - 1.0             # 逐标的过去 window 日已实现收益(<= t)
  v = vol.to_numpy(dtype=float)
  p = past.to_numpy(dtype=float)
  out = np.full(v.shape[0], np.nan)
  min_n = max(fcfg.MIN_CS, 2)                    # 截面太薄则不分组(沿用 MIN_CS 口径)
  for i in range(v.shape[0]):
    ok = np.isfinite(v[i]) & np.isfinite(p[i])
    if ok.sum() < min_n:
      continue
    vals = v[i][ok]
    hi = ok & (v[i] >= np.quantile(vals, 1.0 - group_q))
    lo = ok & (v[i] <= np.quantile(vals, group_q))
    if hi.any() and lo.any():
      out[i] = float(p[i][hi].mean() - p[i][lo].mean())
  return pd.Series(out, index=close.index, dtype=float)


def build_timing_label(ds: Dataset, h: int = None, kind: str = 'vol',
                       log=None) -> pd.Series:
  """
  择时标签: 未来 h 日 市场实现波动率(年化) / 市场累计收益(趋势) / 风险偏好价差(risk-on/off).

  :param ds: Dataset
  :param h: 未来窗口(None = TIMING_LABEL_H)
  :param kind: 'vol'(默认, 未来波动率, 方案B) / 'ret'(未来累计收益, 趋势/牛熊) /
               'riskon'(风险偏好价差: 未来高波组 - 低波组超额收益)
  :param log: 可选 logger
  :returns: Series(index=date)
  """
  h = cfg.TIMING_LABEL_H if h is None else h
  if kind == 'ret':
    y = forward_return(market_returns(ds), h)
    unit = '未来累计收益中位数'
  elif kind == 'riskon':
    y = risk_appetite_spread(ds, h)
    unit = '风险偏好价差(高波组-低波组)中位数'
  else:
    y = forward_vol(market_returns(ds), h)
    unit = '年化波动率中位数'
  if log is not None:
    v = y.to_numpy(dtype=float)
    log.info(f'[timing-label] kind={kind}, h={h}: 有效值占比 {np.isfinite(v).mean():.1%}, '
             f'{unit} {np.nanmedian(v):.2%}')
  return y


# ============================ 4. 敞口映射(纯函数, 无拟合) ============================

def vol_target_exposure(pred_vol, target_vol: float, cap: float = None,
                        floor: float = None) -> pd.Series:
  """
  预测波动率 -> 逐日总敞口: e_t = clip(target_vol / pred_vol_t, floor, cap).

  纯函数(不拟合): 波动率越高 -> 敞口越低. 红线约束 floor=0(不做空) / cap=1(不加杠杆).
  pred_vol 为 0/NaN 或非有限值时该日敞口记 NaN, 由调用方决定填充(如回退满仓).

  :param pred_vol: 逐日预测波动率(Series, 年化)
  :param target_vol: 目标年化波动率(须只在 IS 估计)
  :param cap: 敞口上限(None = TIMING_EXPOSURE_CAP = 1.0)
  :param floor: 敞口下限(None = TIMING_EXPOSURE_FLOOR = 0.0)
  :returns: Series(index 同 pred_vol), e_t ∈ [floor, cap]
  """
  cap = cfg.TIMING_EXPOSURE_CAP if cap is None else cap
  floor = cfg.TIMING_EXPOSURE_FLOOR if floor is None else floor
  pv = pd.Series(pred_vol, dtype=float) if not isinstance(pred_vol, pd.Series) \
      else pred_vol.astype(float)
  e = (target_vol / pv).replace([np.inf, -np.inf], np.nan)
  return e.clip(lower=floor, upper=cap)


def estimate_target_vol(ds: Dataset, h: int = None, window: str = 'IS',
                        stat: str = 'median') -> float:
  """
  目标波动率: **只在 IS 上**估计(红线), 取 IS 市场实现波动率的中位数/均值.

  与标签同量纲(未来 h 日年化实现波动率), 使"预测值 ≈ IS 典型水平"时 e_t ≈ 1.

  :param ds: Dataset
  :param h: 未来窗口(None = TIMING_LABEL_H)
  :param window: 估计窗口(None/'full' 表示不限; 默认 'IS')
  :param stat: 'median'(默认, 抗极值) / 'mean'
  :returns: 目标年化波动率(float)
  """
  rv = forward_vol(market_returns(ds), cfg.TIMING_LABEL_H if h is None else h)
  if window is not None:
    start, end = cfg.window_range(window)
    if start is not None:
      rv = rv[rv.index >= pd.Timestamp(start)]
    if end is not None:
      rv = rv[rv.index <= pd.Timestamp(end)]
  return float(np.nanmedian(rv) if stat == 'median' else np.nanmean(rv))


# ============================ 5. 防前视检验 ============================

def verify_market_features(ds: Dataset, cut: str = fcfg.VERIFY_CUT,
                           tol: float = 1e-9, **kw) -> dict:
  """
  市场特征的截断一致性检验(证明"只用 <= t 的数据").

  原理(与 factor.prepare.truncation_check 同):
    A = 全历史算市场特征, 再截到 cut
    B = 只把 <= cut 的数据喂进构建函数
  A 与 B 在被保留行上逐值相同 -> 整条市场特征管线因果.

  :param ds: Dataset
  :param cut: 截断日
  :param tol: 允许的最大绝对差
  :param kw: 透传给 build_market_features 的窗口参数
  :returns: dict(ok, cut, max_diff, nan_same, per_col: DataFrame)
  """
  a = build_market_features(ds, **kw).loc[:cut]
  b = build_market_features(ds.slice(end=cut), **kw)
  idx = a.index.intersection(b.index)
  cols = a.columns.intersection(b.columns)
  A = a.loc[idx, cols].to_numpy(dtype=float)
  B = b.loc[idx, cols].to_numpy(dtype=float)

  nan_same = bool((np.isnan(A) == np.isnan(B)).all())
  both = np.isfinite(A) & np.isfinite(B)
  with np.errstate(invalid='ignore'):
    d = np.abs(A - B)
  max_diff = float(np.nanmax(d[both])) if both.any() else 0.0

  rows = []
  for j, c in enumerate(cols):
    m = np.isfinite(A[:, j]) & np.isfinite(B[:, j])
    rows.append({'feature': c, 'cells': int(m.sum()),
                 'max_diff': float(np.nanmax(np.abs(A[m, j] - B[m, j]))) if m.any() else 0.0,
                 'nan_same': bool((np.isnan(A[:, j]) == np.isnan(B[:, j])).all())})
  per_col = pd.DataFrame(rows).assign(
      causal=lambda t: t['nan_same'] & (t['max_diff'] <= tol))

  return {'ok': bool(nan_same and max_diff <= tol), 'cut': cut,
          'max_diff': max_diff, 'nan_same': nan_same, 'per_col': per_col}


def verify_timing_label_forward(ds: Dataset, h: int = None, tol: float = 1e-9) -> dict:
  """
  择时标签口径检验: 确认标签**确实**用了未来(否则等于预测"已经发生的波动").

  三条检查(全过才判定正确):
    1. 末尾 h 行必须为 NaN(未来窗口越界);
    2. 手工重算一个样本: fwd_vol(t) == std(mkt[t+1 .. t+h]) * sqrt(ANNUAL);
    3. 截断性: 把面板截到 cut 后, 边界样本应变 NaN(证明真依赖 t 之后的行情).

  :param ds: Dataset
  :param h: 未来窗口(None = TIMING_LABEL_H)
  :param tol: 逐值比对容差
  :returns: dict(ok, tail_nan_ok, manual_ok, truncation_ok, ...)
  """
  h = cfg.TIMING_LABEL_H if h is None else h
  mkt = market_returns(ds)
  y = forward_vol(mkt, h)
  n = len(mkt)

  # 1. 末尾 h 行应为 NaN
  tail_nan_ok = bool(np.isnan(y.tail(h).to_numpy(dtype=float)).all())

  # 2. 手工重算: 取倒数第 (h+2) 个交易日(其未来窗口完整可得)
  t_pos = n - (h + 2)
  manual_ok, manual_diff, probe = False, float('nan'), None
  if t_pos >= 0:
    t = y.index[t_pos]
    w = mkt.iloc[t_pos + 1:t_pos + 1 + h].to_numpy(dtype=float)
    w = w[np.isfinite(w)]
    if w.size >= 2:
      want = float(w.std(ddof=1) * np.sqrt(ANNUAL))
      got = float(y.iloc[t_pos])
      manual_diff = abs(got - want)
      manual_ok = bool(manual_diff <= 1e-9)
      probe = {'date': str(t.date()), 'label': round(got, 6), 'manual': round(want, 6)}

  # 3. 截断性: 截到 cut(留出至少 h 天)后, 边界样本应变 NaN
  cut = mkt.index[n - (h + 1)] if n > 2 * h else mkt.index[max(n // 2, 1)]
  part = ds.slice(end=cut)
  y_part = forward_vol(market_returns(part), h)
  t_edge = y_part.index[-1] if len(y_part.index) else None
  truncation_ok = False
  if t_edge is not None:
    truncation_ok = bool(np.isnan(float(y_part.loc[t_edge])) and np.isfinite(float(y.loc[t_edge])))

  return {'ok': bool(tail_nan_ok and manual_ok and truncation_ok), 'h': h,
          'tail_nan_ok': tail_nan_ok, 'manual_ok': manual_ok,
          'truncation_ok': truncation_ok, 'manual_diff': manual_diff,
          'probe': probe, 'cut': str(cut.date())}


# ============================ 6. 波动率预测器(时序 Ridge, 只在 IS 拟合) ============================

class VolRegressor:
  """
  时序波动率预测器(低容量 Ridge, 纯 numpy 闭式解).

  与 models.linear.RidgeModel 的根本区别:
    - 那边是**截面**模型(样本 = date×symbol, 输出分数宽表);
    - 这里是**时序**模型(样本 = 日, 输入 MKT_FEATURES, 输出单条 pred_vol 序列).
  "时序 + 少特征 + 强收缩" 是有意为之: 择时信噪比极低, 高容量模型(GBDT/DL)极易把
  噪声拟合进去, 故先立一个几乎不可过拟合的线性参照物.

  ⚠ 与截面 RidgeModel 不同, 这里**不做缺失填补**: fit 只用"特征与标签都齐"的行,
  predict 对特征不齐的行(如滚动标准化暖机期)直接记 NaN —— 宁可留白, 也不引入
  任何可能沾到未来信息的填补统计量. 下游 run_engine 对 NaN 敞口按前值/满仓因果兜底.

  :param alpha: L2 强度(None = RIDGE_ALPHA)
  """

  def __init__(self, alpha: float = None):
    self.alpha = float(cfg.RIDGE_ALPHA if alpha is None else alpha)
    self.features = None
    self.x_mean_ = None
    self.y_mean_ = None
    self.coef_ = None
    self.n_fit_ = 0

  # ---- fit: 只吃传入样本(约定 IS) ----
  def fit(self, X, y, log=None) -> 'VolRegressor':
    """在 (X, y) 上估计 列均值 / 系数. 只喂 IS 即不偷看未来."""
    if isinstance(X, pd.Series):
      X = X.to_frame()
    self.features = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    yv = pd.Series(y).reindex(X.index).to_numpy(dtype=float)
    ok = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)   # 只用"特征与标签都齐"的行
    Xv, yv = Xv[ok], yv[ok]
    if Xv.shape[0] <= Xv.shape[1]:
      raise ValueError('可拟合样本过少(特征/标签齐全的行不足), 检查窗口与 IS 段')
    self.x_mean_ = Xv.mean(axis=0)
    self.y_mean_ = float(yv.mean())
    Xc, yc = Xv - self.x_mean_, yv - self.y_mean_        # 中心化 -> 截距被吸收
    d = Xc.shape[1]
    self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d), Xc.T @ yc)
    self.n_fit_ = int(ok.sum())
    if log is not None:
      resid = yv - ((Xv - self.x_mean_) @ self.coef_ + self.y_mean_)
      ss_tot = float(((yv - yv.mean()) ** 2).sum())
      r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
      log.info(f'[vol-ridge] fit: n={self.n_fit_:,}(仅 IS 内特征/标签齐全的行), '
               f'F={len(self.features)}, alpha={self.alpha:g}, R²(IS 内)={r2:.3f}')
    return self

  # ---- predict: 用 fit 定下的参数打分, 返回单条 pred_vol 序列 ----
  def predict(self, X) -> pd.Series:
    """全历史预测: pred_vol[t] = 模型在 t 日市场特征上的输出; 特征不齐处记 NaN."""
    if self.coef_ is None:
      raise RuntimeError('请先 fit')
    if isinstance(X, pd.Series):
      X = X.to_frame()
    if list(X.columns) != self.features:
      raise ValueError('特征顺序与 fit 时不一致')
    Xv = X.to_numpy(dtype=float)
    ok = np.isfinite(Xv).all(axis=1)
    out = np.full(Xv.shape[0], np.nan)
    out[ok] = (Xv[ok] - self.x_mean_) @ self.coef_ + self.y_mean_
    return pd.Series(out, index=X.index)

  def coef_table(self) -> pd.DataFrame:
    """系数表(按绝对值降序), 用于看择时模型到底在依赖哪些市场特征."""
    return (pd.DataFrame({'feature': self.features, 'coef': self.coef_})
            .assign(abss=lambda d: d['coef'].abs())
            .sort_values('abss', ascending=False)
            .drop(columns='abss').reset_index(drop=True))


def timing_design(ds: Dataset, h: int = None, z_window: int = None,
                  kind: str = 'vol', riskon_feat: bool = False, log=None):
  """
  构造择时设计矩阵 (Z, y, is_mask) —— 波动率目标臂与状态分类臂**共用**.

    Z       因果滚动标准化的市场特征(只用 <= t 的分布)
    y       未来 h 日标签(吃未来; 回归目标 / 二值化来源): kind='vol' 波动率 / 'ret' 收益
    is_mask IS 段布尔掩码(所有拟合都只喂这一段 -> 红线)

  :param kind: 标签口径('vol' 默认 / 'ret')
  :param riskon_feat: 是否在市场特征里追加 riskon(风险偏好价差, 因果)扩展列
  :returns: (Z: DataFrame, y: Series, is_mask: Series[bool])
  """
  h = cfg.TIMING_LABEL_H if h is None else h
  feat = build_market_features(ds, riskon_feat=riskon_feat, log=log)
  Z = rolling_zscore(feat, z_window)                   # 因果标准化
  y = build_timing_label(ds, h, kind, log=log)
  start, end = cfg.window_range('IS')
  is_mask = (Z.index >= pd.Timestamp(start)) & (Z.index <= pd.Timestamp(end))
  return Z, y, is_mask


def fit_vol_regressor(ds: Dataset, h: int = None, alpha: float = None,
                      z_window: int = None, log=None):
  """
  在 IS 内拟合波动率预测器, 对全历史预测 -> (pred_vol, model).

  全流程因果(每一步都只读 <= t, 而拟合只用 IS 段):
    1. timing_design(ds)              市场特征 + 因果滚动标准化 + 标签
    2. VolRegressor.fit(IS 段)        **只在 IS 段**估系数(标签吃未来, 故严禁越界拟合)
    3. predict(全历史) -> pred_vol     暖机期(特征 NaN)记 NaN, 由引擎因果兜底

  :param ds: Dataset
  :param h: 波动率标签未来窗口(None = TIMING_LABEL_H)
  :param alpha: Ridge L2 强度(None = RIDGE_ALPHA)
  :param z_window: 滚动标准化窗口(None = TIMING_DD_WINDOW)
  :param log: 可选 logger
  :returns: (pred_vol: Series, model: VolRegressor)
  """
  Z, y, is_mask = timing_design(ds, h, z_window, log=log)
  model = VolRegressor(alpha=alpha).fit(Z[is_mask], y[is_mask], log=log)
  pred = model.predict(Z)
  if log is not None:
    log.info('\n[择时系数] 市场特征(按 |系数| 降序):\n'
             + model.coef_table().to_string(index=False))
    pair = pd.concat([pred.rename('pred'), y.rename('actual')], axis=1)
    is_pair = pair.loc[is_mask].dropna()
    if len(is_pair) > 2:
      log.info(f'[择时拟合] pred_vol 与未来实现波动率相关(仅汇报, 不定参): '
               f'IS {float(is_pair["pred"].corr(is_pair["actual"])):.3f}')
  return pred, model


def build_exposure(ds: Dataset, h: int = None, alpha: float = None,
                   z_window: int = None, target_vol: float = None,
                   window: str = 'IS', log=None) -> pd.Series:
  """
  一站式: 训练波动率预测器(仅 IS) -> 预测 -> 映射为逐日敞口 e_t ∈ [0, 1].

  e_t = clip(target_vol / pred_vol_t, 0, 1); target_vol 默认**在 IS 内**估计(红线).
  暖机/缺值处为 NaN, 由 run_engine 按前值填充再退回满仓(见 backtest.run_engine).

  :param ds: Dataset
  :param h/alpha/z_window: 透传给 fit_vol_regressor
  :param target_vol: 目标年化波动率(None = 在 window 段估计, 默认 IS)
  :param window: target_vol 的估计窗口(默认 'IS'; ⚠ 用 OOS/recent 即前视)
  :param log: 可选 logger
  :returns: Series(index=ds.dates), 逐日总敞口
  """
  pred, _ = fit_vol_regressor(ds, h, alpha, z_window, log=log)
  tgt = estimate_target_vol(ds, h, window) if target_vol is None else float(target_vol)
  e = vol_target_exposure(pred, tgt)
  if log is not None:
    v = e.to_numpy(dtype=float)
    log.info(f'[exposure] target_vol={tgt:.2%} '
             f'({"命令行指定" if target_vol is not None else f"{window} 中位数, 红线"}), '
             f'e_t 有效 {np.isfinite(v).mean():.1%}, 中位数 {np.nanmedian(v):.3f}, '
             f'均值 {np.nanmean(v):.3f}, 满仓 {float((v >= 1 - 1e-9).mean()):.1%}, '
             f'空仓 {float((v <= 1e-9).mean()):.1%}')
  return e


# ============================ 7. 状态分类臂(离散状态 + 固定档位) ============================

class RegimeClassifier:
  """
  时序"状态"分类器(低容量线性概率模型, 纯 numpy 闭式解).

  与 VolRegressor 共用特征与设计矩阵, 差别在**标签处理**:
    用 **IS 内分位**(红线: 只在 IS 取)把连续标签二值化成 状态1/状态0,
    再对 0/1 状态做 Ridge(即线性概率模型 LPM), 得到"处于状态 1"的概率.
  标签口径由 timing_design(kind=...) 决定: kind='vol' 未来波动率(状态1=高波动);
  kind='ret' 未来收益(状态1=上涨). 输出是**离散状态**(概率 > 0.5 为状态 1),
  与波动率目标的连续档位形成对照.

  ⚠ 同样**不做缺失填补**: fit 只用"特征与状态都齐"的行; predict 对特征不齐行记 NaN.

  :param alpha: L2 强度(None = RIDGE_ALPHA)
  :param q: 波动率二值化分位(None = TIMING_REGIME_THRESHOLD_Q; 0.5 = IS 中位数)
  """

  def __init__(self, alpha: float = None, q: float = None):
    self.alpha = float(cfg.RIDGE_ALPHA if alpha is None else alpha)
    self.q = float(cfg.TIMING_REGIME_THRESHOLD_Q if q is None else q)
    self.features = None
    self.threshold_ = None       # 波动率二值化阈值(IS 分位)
    self.base_rate_ = None       # IS 段高波动状态占比
    self.x_mean_ = None
    self.y_mean_ = None
    self.coef_ = None
    self.n_fit_ = 0

  def fit(self, X, y, log=None) -> 'RegimeClassifier':
    """在 (X, y) 上估阈值/系数. y 为未来波动率(连续), 内部按 IS 分位二值化. 只喂 IS 即不偷看未来."""
    if isinstance(X, pd.Series):
      X = X.to_frame()
    self.features = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    yv = pd.Series(y).reindex(X.index).to_numpy(dtype=float)
    ok = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)
    Xv, yv = Xv[ok], yv[ok]
    if Xv.shape[0] <= Xv.shape[1]:
      raise ValueError('可拟合样本过少(特征/标签齐全的行不足), 检查窗口与 IS 段')
    self.threshold_ = float(np.quantile(yv, self.q))       # ⚠ 只在传入的(IS)样本上取
    b = (yv > self.threshold_).astype(float)               # 高波动状态 = 1
    self.base_rate_ = float(b.mean())
    self.x_mean_ = Xv.mean(axis=0)
    self.y_mean_ = self.base_rate_
    Xc, bc = Xv - self.x_mean_, b - self.y_mean_
    d = Xc.shape[1]
    self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d), Xc.T @ bc)
    self.n_fit_ = int(ok.sum())
    if log is not None:
      log.info(f'[regime] fit: n={self.n_fit_:,}(仅 IS 内特征/标签齐全的行), '
               f'F={len(self.features)}, alpha={self.alpha:g}, '
               f'阈值={self.threshold_:.2%}(IS {self.q:.0%} 分位), 状态1占比={self.base_rate_:.1%}')
    return self

  def predict_proba(self, X) -> pd.Series:
    """全历史 P(状态 1) 的线性概率估计; 特征不齐处记 NaN."""
    if self.coef_ is None:
      raise RuntimeError('请先 fit')
    if isinstance(X, pd.Series):
      X = X.to_frame()
    if list(X.columns) != self.features:
      raise ValueError('特征顺序与 fit 时不一致')
    Xv = X.to_numpy(dtype=float)
    ok = np.isfinite(Xv).all(axis=1)
    out = np.full(Xv.shape[0], np.nan)
    out[ok] = (Xv[ok] - self.x_mean_) @ self.coef_ + self.y_mean_
    return pd.Series(out, index=X.index)

  def predict(self, X) -> pd.Series:
    """全历史状态: 1=状态1, 0=状态0(概率 > 0.5); 特征不齐处 NaN."""
    p = self.predict_proba(X)
    return (p > 0.5).astype(float).where(p.notna())

  def coef_table(self) -> pd.DataFrame:
    """系数表(按绝对值降序), 看状态分类器在依赖哪些市场特征."""
    return (pd.DataFrame({'feature': self.features, 'coef': self.coef_})
            .assign(abss=lambda d: d['coef'].abs())
            .sort_values('abss', ascending=False)
            .drop(columns='abss').reset_index(drop=True))


def regime_exposure(regime, on_exposure: float = None,
                    off_exposure: float = None, levels: dict = None) -> pd.Series:
  """
  离散状态 -> 逐日总敞口(纯映射, 无拟合).

    默认(levels=None, 波动率口径): 低波动状态(regime = 0) -> on_exposure (默认 1.0, 满仓)
      高波动状态(regime = 1) -> off_exposure(默认 0.3, 降仓)
    可传 levels 覆盖任意"状态 -> 敞口"映射(如趋势口径 {0: 降仓, 1: 满仓}).
  NaN 状态(暖机/缺值)保留为 NaN, 由 run_engine 因果兜底. 各档均 >= 0 -> 不做空/不加杠杆.

  :param regime: 状态序列(如 1=高波动, 0=低波动; NaN 保留)
  :param on_exposure: 状态 0 档敞口(None = TIMING_REGIME_ON_EXPOSURE)
  :param off_exposure: 状态 1 档敞口(None = TIMING_REGIME_OFF_EXPOSURE)
  :param levels: {状态值 -> 敞口} 字典; 给定则忽略 on/off
  :returns: Series(index 同 regime), 值取自 levels
  """
  r = regime.astype(float) if isinstance(regime, pd.Series) else pd.Series(regime, dtype=float)
  if levels is None:
    on_ = cfg.TIMING_REGIME_ON_EXPOSURE if on_exposure is None else float(on_exposure)
    off_ = cfg.TIMING_REGIME_OFF_EXPOSURE if off_exposure is None else float(off_exposure)
    levels = {0.0: on_, 1.0: off_}
  e = r.map(levels).astype(float)
  return e.where(r.notna())


def _regime_predict(ds: Dataset, h: int = None, alpha: float = None,
                    z_window: int = None, q: float = None, kind: str = 'vol',
                    riskon_feat: bool = False, log=None):
  """状态分类臂: 设计矩阵 -> 仅在 IS 拟合 -> 全历史 P(状态 1). 供 build/verify 复用.

  kind='vol': 状态 1 = 高波动; kind='ret': 状态 1 = 上涨(未来收益 > IS 分位).
  kind='riskon': 状态 1 = risk-on(风险偏好价差 > IS 分位).
  riskon_feat=True: 在市场特征里**追加** riskon(风险偏好价差, 因果)一列(与标签口径独立).
  """
  Z, y, is_mask = timing_design(ds, h, z_window, kind, riskon_feat=riskon_feat, log=log)
  clf = RegimeClassifier(alpha=alpha, q=q).fit(Z[is_mask], y[is_mask], log=log)
  return clf.predict_proba(Z), clf


def build_regime_exposure(ds: Dataset, h: int = None, alpha: float = None,
                          z_window: int = None, q: float = None,
                          on_exposure: float = None,
                          off_exposure: float = None, kind: str = 'vol',
                          levels: dict = None, riskon_feat: bool = False,
                          log=None) -> pd.Series:
  """
  一站式: 状态分类臂(t4 第三臂) -> 逐日**离散档位**敞口 e_t.

  步骤: 设计矩阵 -> IS 分位二值化标签 -> 低容量线性概率模型(仅 IS 拟合) -> 概率 > 0.5 判状态 1
        -> 映射固定档位. 与 build_exposure(连续档位)并列, 喂同一引擎做对比.
  暖机/缺值处为 NaN, 由 run_engine 按前值填充再退回满仓(见 backtest.run_engine).

  :param ds: Dataset
  :param h/alpha/z_window/q: 透传给 _regime_predict(alpha 为分类器 L2 强度)
  :param kind: 标签口径 'vol'(波动率状态, 状态1=高波动) / 'ret'(趋势状态, 状态1=上涨) /
               'riskon'(风险偏好状态, 状态1=risk-on)
  :param levels: 覆盖默认档位映射; 默认按 kind 自动定方向:
                 vol -> {0: 满仓, 1: 降仓}; ret/riskon -> {0: 降仓, 1: 满仓}
  :param on_exposure/off_exposure: 满/降仓档敞口(None = config 默认 1.0 / 0.3)
  :param riskon_feat: 是否在市场特征里**追加** riskon(风险偏好价差, 因果)扩展列
                      (把风险偏好维度降级为特征喂给该臂; 与 kind 标签口径独立)
  :param log: 可选 logger
  :returns: Series(index=ds.dates), 逐日总敞口
  """
  on_ = cfg.TIMING_REGIME_ON_EXPOSURE if on_exposure is None else float(on_exposure)
  off_ = cfg.TIMING_REGIME_OFF_EXPOSURE if off_exposure is None else float(off_exposure)
  if levels is None:
    # vol: 状态1=高波动(不利) -> 降仓; ret/riskon: 状态1=有利(上涨/risk-on) -> 满仓 (方向相反)
    levels = {0.0: off_, 1.0: on_} if kind in ('ret', 'riskon') else {0.0: on_, 1.0: off_}
  prob, clf = _regime_predict(ds, h, alpha, z_window, q, kind, riskon_feat=riskon_feat, log=log)
  regime = (prob > 0.5).astype(float).where(prob.notna())
  e = regime_exposure(regime, levels=levels)
  if log is not None:
    v = e.to_numpy(dtype=float)
    log.info(f'[regime-exposure] kind={kind}, 档位 {levels}, '
             f'e_t 有效 {np.isfinite(v).mean():.1%}, 均值 {np.nanmean(v):.3f}, '
             f'降仓占比 {float((v < max(levels.values()) - 1e-9).mean()):.1%}')
    log.info('\n[状态系数] 市场特征(按 |系数| 降序):\n'
             + clf.coef_table().to_string(index=False))
  return e


# ============================ 8. 管线前视检验(两臂共用) ============================

def verify_timing_pipeline(ds: Dataset, h: int = None, alpha: float = None,
                           z_window: int = None, cut: str = fcfg.VERIFY_CUT,
                           tol: float = 1e-9, kind: str = 'voltarget',
                           riskon_feat: bool = False) -> dict:
  """
  择时管线前视检验: 确认"整条链(特征 -> 标准化 -> 预测)因果, 且拟合只在 IS".

  做法: 用全历史跑完整管线得 pred_A; 把面板截到 cut(**>= IS 末**)后重跑得 pred_B.
    因截断不动 IS 段 -> 拟合集不变 -> 系数应逐值相同(证明"拟合只用 IS");
    因各步均因果   -> 公共行(<= cut)上 pred 应逐值相同(证明"无前视").
  仅当两条件同时成立才判 ok. 若 cut 落在 IS 内部, 拟合集改变属正常, 不能用此法.

  :param ds: Dataset
  :param h/alpha/z_window: 透传给对应臂的拟合函数
  :param cut: 截断日(须 >= IS 末, 默认 fcfg.VERIFY_CUT)
  :param tol: 逐值比对容差
  :param kind: 'voltarget'(默认, 比 pred_vol) / 'regime'(比 P(高波动)) /
               'regime_ret'(比 P(上涨)) / 'regime_riskon'(比 P(risk-on))
  :param riskon_feat: 是否在市场特征里追加 riskon(风险偏好价差)扩展列(与状态臂口径一致)
  :returns: dict(ok, coef_same, max_diff, nan_same, n_common, cut)
  """
  if kind.startswith('regime'):
    k = {'regime': 'vol', 'regime_ret': 'ret', 'regime_riskon': 'riskon'}[kind]
    pred_a, m_a = _regime_predict(ds, h, alpha, z_window, kind=k, riskon_feat=riskon_feat)
    pred_b, m_b = _regime_predict(ds.slice(end=cut), h, alpha, z_window, kind=k,
                                  riskon_feat=riskon_feat)
  else:
    pred_a, m_a = fit_vol_regressor(ds, h, alpha, z_window)
    pred_b, m_b = fit_vol_regressor(ds.slice(end=cut), h, alpha, z_window)

  idx = pred_a.index.intersection(pred_b.index)
  A = pred_a.loc[idx].to_numpy(dtype=float)
  B = pred_b.loc[idx].to_numpy(dtype=float)
  nan_same = bool((np.isnan(A) == np.isnan(B)).all())
  both = np.isfinite(A) & np.isfinite(B)
  with np.errstate(invalid='ignore'):
    max_diff = float(np.nanmax(np.abs(A[both] - B[both]))) if both.any() else 0.0
  coef_same = bool(np.allclose(m_a.coef_, m_b.coef_, rtol=0.0, atol=tol))
  return {'ok': bool(coef_same and nan_same and max_diff <= tol), 'cut': cut,
          'coef_same': coef_same, 'nan_same': nan_same, 'max_diff': max_diff,
          'n_common': int(len(idx)), 'alpha': m_a.alpha, 'n_fit': m_a.n_fit_}


# ============================ 9. 信号分数阈值臂(用选股分数本身定敞口) ============================

def score_breadth(score: pd.DataFrame, tau: float) -> pd.Series:
  """
  逐日"信号广度": 当日(截面)有效分数 > tau 的标的占比. 只用当日截面 -> 因果.

  :param score: (date × symbol) 分数宽表
  :param tau: 分数阈值(须在 IS 内取)
  :returns: Series(index=date), 当日无有效分数记 NaN
  """
  v = score.to_numpy(dtype=float)
  ok = np.isfinite(v)
  n = ok.sum(axis=1)
  gt = (ok & (v > tau)).sum(axis=1)
  with np.errstate(invalid='ignore'):
    b = np.where(n > 0, gt / np.maximum(n, 1), np.nan)
  return pd.Series(b, index=score.index, dtype=float)


def score_topk_mean(score: pd.DataFrame, top_k: int) -> pd.Series:
  """
  逐日"信号强度": 当日截面得分最高的 top_k 个标的的平均分(纯因果).

  :param score: (date × symbol) 分数宽表
  :param top_k: 取前 k 名平均(不足 k 则用全部有效值)
  :returns: Series(index=date), 当日无有效分数记 NaN
  """
  v = score.to_numpy(dtype=float)
  out = np.full(v.shape[0], np.nan)
  for i in range(v.shape[0]):
    row = v[i][np.isfinite(v[i])]
    if row.size:
      k = min(int(top_k), row.size)
      out[i] = np.sort(row)[-k:].mean()
  return pd.Series(out, index=score.index, dtype=float)


def _window_quantile(x: pd.Series, q: float, window: str = 'IS') -> float:
  """序列在指定窗口(默认 IS)上的分位(忽略 NaN). 只在 IS 取 -> 不偷看未来(红线)."""
  if window is not None and window != 'full':
    start, end = cfg.window_range(window)
    if start is not None:
      x = x[x.index >= pd.Timestamp(start)]
    if end is not None:
      x = x[x.index <= pd.Timestamp(end)]
  return float(np.nanquantile(x.to_numpy(dtype=float), q))


def build_score_exposure(score: pd.DataFrame, stat: str = None, tau_q: float = None,
                         lo_q: float = None, hi_q: float = None, top_k: int = None,
                         window: str = 'IS', cap: float = None,
                         floor: float = None, log=None) -> pd.Series:
  """
  一站式: 用**选股分数本身**产出逐日总敞口 e_t ∈ [0, 1] —— 与 voltarget/regime 信息源不同.

  步骤(每一步都只读 <= t, 阈值/分位只在 window 段估):
    1. 逐日统计量 x_t(纯因果, 只用当日截面):
         stat='breadth'  -> x_t = 当日有效分数 > tau 的占比(tau = window 段 tau_q 分位);
         stat='topk_mean'-> x_t = 当日 TopK 平均分.
    2. 把 x_t 按 window 段的 [lo_q, hi_q] 分位线性映射到 [floor, cap]:
         e_t = clip( (x_t - lo) / (hi - lo), floor, cap )
       弱信号 -> 接近 floor(降仓), 强信号 -> 接近 cap(满仓).
  score 在该日无有效列时 x_t 为 NaN -> e_t 为 NaN, 由 run_engine 按前值/满仓因果兜底.
  红线: 不做空(floor=0)/不加杠杆(cap=1); tau 与 lo/hi 分位只在 IS 上取.

  :param score: (date × symbol) 分数宽表(与回测同网格)
  :param stat: 'breadth'(默认, config) / 'topk_mean'
  :param tau_q: breadth 的分数阈值分位(None = TIMING_SCORE_TAU_Q)
  :param lo_q/hi_q: 敞口线性映射的下/上限分位(None = TIMING_SCORE_LO_Q / HI_Q)
  :param top_k: topk_mean 的 k(None = TOP_K)
  :param window: 阈值/分位估计窗口(默认 'IS'; ⚠ 用 OOS/recent 即前视)
  :param cap/floor: 敞口上/下限(None = TIMING_EXPOSURE_CAP / FLOOR)
  :param log: 可选 logger
  :returns: Series(index=score.index), 逐日总敞口
  """
  stat = cfg.TIMING_SCORE_STAT if stat is None else stat
  tau_q = cfg.TIMING_SCORE_TAU_Q if tau_q is None else tau_q
  lo_q = cfg.TIMING_SCORE_LO_Q if lo_q is None else lo_q
  hi_q = cfg.TIMING_SCORE_HI_Q if hi_q is None else hi_q
  top_k = fcfg.TOP_K if top_k is None else int(top_k)
  cap = cfg.TIMING_EXPOSURE_CAP if cap is None else cap
  floor = cfg.TIMING_EXPOSURE_FLOOR if floor is None else floor

  flat = pd.Series(score.to_numpy(dtype=float).ravel(),
                   index=np.repeat(score.index.to_numpy(), score.shape[1]))
  if stat == 'breadth':
    tau = _window_quantile(flat, tau_q, window)          # ⚠ 只在 window(IS) 取
    x = score_breadth(score, tau)
  elif stat == 'topk_mean':
    tau = np.nan
    x = score_topk_mean(score, top_k)
  else:
    raise ValueError(f"未知 stat: {stat} (可选 'breadth' / 'topk_mean')")

  lo = _window_quantile(x, lo_q, window)                 # ⚠ 只在 window(IS) 取
  hi = _window_quantile(x, hi_q, window)
  denom = (hi - lo) if np.isfinite(hi - lo) and abs(hi - lo) > 1e-12 else np.nan
  e = ((x - lo) / denom).clip(lower=floor, upper=cap)
  e = e.where(x.notna())                                 # 无有效分数 -> 保留 NaN
  if log is not None:
    v = e.to_numpy(dtype=float)
    log.info(f'[score-exposure] stat={stat}, 窗口={window}, 阈值 tau/tau_q='
             f'{tau:.4f}/{tau_q} ({"仅 breadth" if stat == "breadth" else "n/a"}), '
             f'映射分位 lo/hi={lo:.4f}/{hi:.4f}(q={lo_q}/{hi_q}); '
             f'e_t 有效 {np.isfinite(v).mean():.1%}, 中位数 {np.nanmedian(v):.3f}, '
             f'均值 {np.nanmean(v):.3f}, 满仓 {float((v >= cap - 1e-9).mean()):.1%}, '
             f'空仓 {float((v <= floor + 1e-9).mean()):.1%}')
  return e


# ============================ CLI ============================

def _window_stats(y: pd.Series) -> pd.DataFrame:
  """把标签序列按 IS/OOS/recent 汇总(仅汇报, 不参与定参)."""
  rows = []
  for w in ('IS', 'OOS', 'recent'):
    start, end = cfg.window_range(w)
    x = y
    if start is not None:
      x = x[x.index >= pd.Timestamp(start)]
    if end is not None:
      x = x[x.index <= pd.Timestamp(end)]
    v = x.to_numpy(dtype=float)
    rows.append({'window': w, 'days': int(np.isfinite(v).sum()),
                 'vol_median': round(float(np.nanmedian(v)), 4) if v.size else np.nan,
                 'vol_mean': round(float(np.nanmean(v)), 4) if v.size else np.nan})
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser(description='ml 择时层自检: 市场特征 + 波动率标签 + 敞口映射')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.TIMING_LABEL_H, help='波动率标签未来窗口')
  ap.add_argument('--alpha', type=float, default=cfg.RIDGE_ALPHA,
                  help='波动率预测器的 Ridge L2 强度')
  ap.add_argument('--z-window', type=int, default=None,
                  help='市场特征滚动标准化窗口(None = TIMING_DD_WINDOW)')
  ap.add_argument('--target-vol', type=float, default=None,
                  help='目标年化波动率(None = 用 IS 中位数估计)')
  ap.add_argument('--regime-q', type=float, default=cfg.TIMING_REGIME_THRESHOLD_Q,
                  help='状态分类臂: 波动率二值化分位(IS 内取, 红线)')
  ap.add_argument('--regime-on', type=float, default=cfg.TIMING_REGIME_ON_EXPOSURE,
                  help='状态分类臂: 低波动档敞口')
  ap.add_argument('--regime-off', type=float, default=cfg.TIMING_REGIME_OFF_EXPOSURE,
                  help='状态分类臂: 高波动档敞口')
  ap.add_argument('--check', action='store_true', help='做因果/前视口径检验')
  a = ap.parse_args()

  log = cfg.get_logger('ml.timing')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  mf = build_market_features(ds, log=log)
  log.info(f'\n[市场特征] 最近 5 行:\n{mf.tail(5).round(4).to_string()}')

  y = build_timing_label(ds, a.h, log=log)
  log.info(f'\n[标签分窗] 未来 {a.h} 日年化实现波动率(仅汇报, 不定参):')
  log.info('\n' + _window_stats(y).to_string(index=False))

  tgt = a.target_vol if a.target_vol is not None else estimate_target_vol(ds, a.h)
  log.info(f'\n[目标波动率] target_vol = {tgt:.2%} '
           f'({"命令行指定" if a.target_vol is not None else "IS 中位数, 红线: 只用 IS"})')

  # 敞口映射演示: 用"滚动短期波动"当朴素的波动率预测(波动聚集基线), 展示映射行为
  naive = mf['vol_s']
  e = vol_target_exposure(naive, tgt)
  log.info(f'\n[敞口演示] 朴素基线(用 vol_s 当预测): e_t 有效 {e.notna().mean():.1%}, '
           f'中位数 {np.nanmedian(e):.3f}, 均值 {np.nanmean(e):.3f}, '
           f'满仓占比 {float((e >= 1.0 - 1e-9).mean()):.1%}, '
           f'空仓占比 {float((e <= 1e-9).mean()):.1%}')

  # t3: 真正训练波动率预测器(只在 IS fit), 输出全历史 exposure Series
  expo = build_exposure(ds, a.h, a.alpha, a.z_window, a.target_vol, log=log)
  log.info(f'\n[敞口分窗] 波动率目标臂 e_t(仅汇报):')
  log.info('\n' + _window_stats(expo).rename(
      columns={'vol_median': 'e_median', 'vol_mean': 'e_mean'}).to_string(index=False))

  # t4: 状态分类臂(离散档位) —— 与波动率目标同特征/同引擎, 只换标签处理与档位映射
  expo_r = build_regime_exposure(ds, a.h, a.alpha, a.z_window, a.regime_q,
                                 a.regime_on, a.regime_off, log=log)
  log.info(f'\n[敞口分窗] 状态分类臂 e_t ∈ {{on, off}}(仅汇报):')
  log.info('\n' + _window_stats(expo_r).rename(
      columns={'vol_median': 'e_median', 'vol_mean': 'e_mean'}).to_string(index=False))

  if a.check:
    vm = verify_market_features(ds)
    log.info(f'\n[市场特征因果] 截断一致性(cut={vm["cut"]}): ok={vm["ok"]}, '
             f'nan 一致={vm["nan_same"]}, max_diff={vm["max_diff"]:.2e}')
    log.info('\n' + vm['per_col'].to_string(index=False))
    log.info('[结论] 市场特征只读 <= t 的数据(因果)' if vm['ok']
             else '[结论] 疑似前视, 需排查')

    vl = verify_timing_label_forward(ds, a.h)
    log.info(f'\n[标签口径] h={a.h}')
    log.info(f'  末尾 h 行全 NaN : {vl["tail_nan_ok"]}')
    log.info(f'  手工重算一致      : {vl["manual_ok"]} (diff={vl["manual_diff"]:.2e})')
    if vl['probe']:
      log.info(f'    探针 {vl["probe"]}')
    log.info(f'  截断后边界样本失效: {vl["truncation_ok"]} (cut={vl["cut"]})')
    log.info('[结论] 标签口径正确(确实用了未来)' if vl['ok']
             else '[结论] 标签口径异常, 需排查')

    vp = verify_timing_pipeline(ds, a.h, a.alpha, a.z_window)
    log.info(f'\n[择时管线前视] 截断到 {vp["cut"]} 后重跑(公共 {vp["n_common"]} 行):')
    log.info(f'  拟合集不变(系数逐值相同): {vp["coef_same"]} (fit n={vp["n_fit"]:,})')
    log.info(f'  pred 公共行逐值相同      : {vp["nan_same"] and vp["max_diff"] <= 1e-9} '
             f'(max_diff={vp["max_diff"]:.2e})')
    log.info('[结论] 择时管线无前视(特征/标准化/预测皆因果, 拟合只在 IS)' if vp['ok']
             else '[结论] 择时管线疑似前视, 需排查')

    vp2 = verify_timing_pipeline(ds, a.h, a.alpha, a.z_window, kind='regime')
    log.info(f'\n[状态分类管线前视] 截断到 {vp2["cut"]} 后重跑(公共 {vp2["n_common"]} 行):')
    log.info(f'  拟合集不变(系数逐值相同): {vp2["coef_same"]} (fit n={vp2["n_fit"]:,})')
    log.info(f'  prob 公共行逐值相同      : {vp2["nan_same"] and vp2["max_diff"] <= 1e-9} '
             f'(max_diff={vp2["max_diff"]:.2e})')
    log.info('[结论] 状态分类管线无前视(特征/标准化/概率皆因果, 阈值与拟合只在 IS)' if vp2['ok']
             else '[结论] 状态分类管线疑似前视, 需排查')

  log.info('\n[下一步] t3/t4 就绪: predict.py 用 --timing(波动率目标) / --timing-regime(状态分类)'
           ' 把各臂 exposure 喂给同一个 run_engine(exposure=...) 与满仓基线同表对照'
           '(分窗 + 分年 + shuffle).')


if __name__ == '__main__':
  main()
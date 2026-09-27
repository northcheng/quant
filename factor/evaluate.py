# -*- coding: utf-8 -*-
"""
factor.evaluate — 评估层(第3步)
================================
职责: 回答"这个因子到底有没有预测力".

  前两步把因子做得"干净"(因果)且"可比"(截面); 本层给它打分.

--- 与第1/2步**相反**的因果方向: 因子禁未来, 标签必须用未来 ---
  IC 是"因子值(t 时刻已知)"与"未来收益(t -> t+h)"的相关. 这里的未来收益**必须**
  真的用未来数据 —— 它不是因子的一部分, 而是要被预测的答案. 所以:

    - 因子 : 只读 <= t 的数据(第 1/2 步已用截断一致性检验把关);
    - 标签 : t 日收盘决策 -> t+1 开盘入场 -> t+1+h 开盘出场(可交易口径).

  若标签也用"当天已经能拿到的价格", 就变成了预测"已经发生的事", 毫无意义.
  前两步怕"偷看未来", 这一步怕"不敢看未来", 恰好互为镜像.

--- 五件事 ---
  1. 可交易前向收益  forward_return(ds, h)      -- 评估的 ground truth
  2. IC / RankIC / ICIR  ic_series + ic_summary  -- 因子值与未来收益的截面相关
  3. 分层收益        quantile_returns            -- 分桶单调性 + 多空价差
  4. top-k 换手      topk_turnover               -- 信号稳定度(成本前哨)
  5. 信号自相关衰减   signal_ac                  -- 多快失效/多久调仓一次

  汇总 evaluate_factors -> 每行一个因子的评估表; window 支持 full / IS / OOS.

--- 评估诚实性: 三重防伪(别把"撞上的显著"和"静态身份"当动态 alpha) ---
  1) 截面相关 -> 先按日聚合: IC 是逐日算的, 统计量都基于"日级序列", 不放大截面样本量;
  2) 窗口重叠 -> Newey-West HAC: 重叠 h 日标签 + 信号惯性 -> nw_t(比 t_stat 保守),
     并给出对应双尾 p 值 nw_p;
  3) 多重检验 -> BH-FDR: 一次评估几十个因子, 不做校正就会把"撞上的显著"当真 alpha.
     nw_p 经 Benjamini-Hochberg 跨因子校正得到 fdr_q(fdr_sig = q < FDR_ALPHA).
  另有两条"身份"探针, 专门识别"看起来很美但没有动态信息"的因子:
  - sig_ac1: 信号自身前后日截面秩自相关均值. >= STATIC_AC1(0.99) -> is_static:
    典型如"永远选同几个标的"的静态选票, 换手极低但选股不随时间变化;
  - validate_one: 分半 + 静态退化对照. 用前半段"每标的平均分"当静态选票, 看它在后半段
    能否复制动态信号的后半段表现 -> dyn_minus_static. <= 0 = "动态选股没有超过静态身份".
    同表另给 top3_share / hhi / n_eff_symbols, 量化持仓集中度(集中在少数标的上时,
    逐日 IC 看着稳, 其实只是押注了几个名字).

口径说明:
  - IC      : 截面 Pearson 相关(对数值敏感, 易被极值带偏);
  - RankIC  : 截面 Spearman(秩)相关, 与第 2 步默认的 rank 预处理天然配套;
  - ICIR    : ic_mean / ic_std, 即"信噪比"; t_stat = ICIR * sqrt(n_days);
  - nw_t    : IC 序列往往自相关(重叠 h 日标签贡献 h-1 阶, 且因子信号本身有惯性,
              如 mom_20 的 ac1 ~ 0.9), 普通 t 检验会偏乐观, 故用 Newey-West
              (Bartlett)做 HAC 修正. lag 取 max(h-1, 1): 至少修正 1 阶, 避免
              h=1 时因"标签不重叠"就误判为无需修正. nw_t 通常比 t_stat 保守.

自检:
  cd ~/git && python -m quant.factor.evaluate --pool etf_3x
  cd ~/git && python -m quant.factor.evaluate --pool etf_3x --check
  cd ~/git && python -m quant.factor.evaluate --pool etf_3x --split
  cd ~/git && python -m quant.factor.evaluate --pool etf_3x --show mom_20
"""
import argparse
from math import erf, sqrt

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import Dataset, load_pool

STATIC_AC1 = 0.99                              # sig_ac1 >= 此值 -> 判"静态选票"
FDR_ALPHA = 0.10                               # BH-FDR 显著性水平(报告层用)

# 展示用列(拆成"预测力" / "组合表现" / "诚实性"三块, 便于阅读)
PRED_COLS = ['factor', 'group', 'dir', 'dir_ok', 'n_days', 'ic', 'rank_ic',
             'rank_icir', 't_stat', 'nw_t', 'pos_rate']
PERF_COLS = ['factor', 'excess', 'ls_spread', 'mono', 'turnover', 'ac1', 'ac5', 'ac20']
HONEST_COLS = ['factor', 'nw_p', 'fdr_q', 'fdr_sig', 'sig_ac1', 'is_static',
               'top3_share', 'n_eff_symbols', 'dyn_minus_static']


# ============================ 1. 可交易前向收益 ============================

def forward_return(ds: Dataset, h: int = 1) -> pd.DataFrame:
  """
  可交易前向收益(t -> t+1+h): 信号日 t 收盘决策, t+1 开盘入场, t+1+h 开盘出场.

  用开盘价而非收盘价, 是因为"t 日收盘后才知道因子值, 当天已无法成交";
  最早只能吃到 t+1 的开盘. 这是评估口径与回测口径必须一致的关键一步.

  注意: 这里**必须**用未来数据(shift(-1) / shift(-(1+h))) —— 它是标签, 不是因子.
  末尾 h+1 行因缺未来开盘价而为 NaN, 属正常.

  :param ds: Dataset
  :param h: 持有天数(开盘到开盘)
  :returns: (date × symbol) 宽表, 值 = 入场到出场的简单收益
  """
  open_ = ds.wide('open')
  entry = open_.shift(-1)                         # t+1 开盘
  exit_ = open_.shift(-(1 + h))                   # t+1+h 开盘
  return exit_ / entry - 1.0


# ============================ 2. IC / RankIC / ICIR ============================

def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame, method: str = 'spearman',
              min_cs: int = cfg.MIN_CS) -> pd.Series:
  """
  逐日截面 IC 序列(因子值 t 日 vs 前向收益).

  实现: 先对齐日期与标的, 再按行(截面)算相关, NaN 按"成对剔除"处理
  (只有因子和收益都有效的标的才参与), 有效标的数 < min_cs 的日子不产出 IC.

  :param factor: 因子宽表(date × symbol), 通常是第2步预处理后的值
  :param fwd: 前向收益宽表
  :param method: 'spearman'(=RankIC) 或 'pearson'(=IC)
  :param min_cs: 每日最少配对数
  :returns: pd.Series(index=交易日), 值 = 当日 IC
  """
  idx = factor.index.intersection(fwd.index)
  col = factor.columns.intersection(fwd.columns)
  f, r = factor.loc[idx, col], fwd.loc[idx, col]
  if method == 'spearman':                        # 秩相关: 先各自按行排名
    f, r = f.rank(axis=1), r.rank(axis=1)
  elif method != 'pearson':
    raise KeyError(f'未知相关方法: {method} (可选 spearman / pearson)')

  both = f.notna() & r.notna()                    # 成对有效
  n = both.sum(axis=1)
  fm, rm = f.where(both), r.where(both)           # 屏蔽非配对格, mean 自动跳过 NaN
  fd = fm.sub(fm.mean(axis=1), axis=0)
  rd = rm.sub(rm.mean(axis=1), axis=0)
  cov = (fd * rd).sum(axis=1)
  vf = np.sqrt((fd ** 2).sum(axis=1))
  vr = np.sqrt((rd ** 2).sum(axis=1))
  with np.errstate(divide='ignore', invalid='ignore'):
    ic = cov / (vf * vr)                          # 某日因子或收益无差异 -> NaN
  ic = ic.where(n >= min_cs).dropna()
  ic.name = 'ic'
  return ic


def _nw_tstat(x, lag: int) -> tuple:
  """
  日级序列均值的 Newey-West HAC t 检验(Bartlett 权重) -> (t, p, n).

  重叠 h 日标签使相邻 IC 重叠 h-1 天; 此外因子信号自身有惯性, 故调用方用
  max(h-1, 1) 保证至少修正 1 阶. stdlib 实现(不引入 scipy).
  n < 30 或长方差 <= 0 时 t/p 返回 NaN.

  返回的 p 是双尾正态近似(大样本下 t ~ N(0,1)); 它是 evaluate_factors 里
  BH-FDR 多重检验校正的输入 —— 单个 nw_t 只能说明"这个因子自己显著",
  不能说明"在几十个因子里它还算显著".

  :param x: 日级序列(可含 NaN)
  :param lag: HAC 最大滞后阶
  :returns: (t 值, 双尾 p 值, 有效样本数)
  """
  x = np.asarray(x, dtype=float)
  x = x[~np.isnan(x)]
  n = int(x.size)
  if n < 30:
    return np.nan, np.nan, n
  e = x - x.mean()
  s = float(e @ e) / n
  lag = max(int(lag), 0)
  for l in range(1, min(lag, n - 1) + 1):
    w = 1.0 - l / (lag + 1)
    s += 2.0 * w * float(e[l:] @ e[:-l]) / n
  if s <= 0:
    return np.nan, np.nan, n
  t = float(x.mean() / sqrt(s / n))
  p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))   # 双尾
  return t, p, n


def ic_summary(ic: pd.Series, nw_lag: int = 0) -> dict:
  """
  IC 序列汇总: 均值 / 标准差 / ICIR / t 值 / NW-HAC t 值 + p 值 / 胜率.

  ICIR = ic_mean / ic_std(信噪比); t_stat = ICIR * sqrt(n_days)(假设日间独立).

  :param ic: ic_series 的输出
  :param nw_lag: 若 > 0, 额外给出 Newey-West 修正 t 值与对应双尾 p 值
  :returns: dict(n_days, ic_mean, ic_std, ic_ir, t_stat, nw_t, nw_p, pos_rate)
  """
  x = ic.dropna()
  n = len(x)
  if n == 0:
    return {'n_days': 0, 'ic_mean': np.nan, 'ic_std': np.nan, 'ic_ir': np.nan,
            't_stat': np.nan, 'nw_t': np.nan, 'nw_p': np.nan, 'pos_rate': np.nan}
  mean = float(x.mean())
  std = float(x.std(ddof=1))
  ir = mean / std if std > 0 else np.nan
  t = ir * sqrt(n) if std > 0 else np.nan
  nw_t, nw_p = (np.nan, np.nan)
  if nw_lag > 0:
    nw_t, nw_p = _nw_tstat(x.to_numpy(), nw_lag)[:2]
  return {'n_days': n,
          'ic_mean': round(mean, 4),
          'ic_std': round(std, 4),
          'ic_ir': round(ir, 3) if pd.notna(ir) else np.nan,
          't_stat': round(t, 2) if pd.notna(t) else np.nan,
          'nw_t': round(nw_t, 2) if pd.notna(nw_t) else np.nan,
          'nw_p': round(nw_p, 4) if pd.notna(nw_p) else np.nan,
          'pos_rate': round(float((x > 0).mean()), 3)}


# ============================ 3. 分层收益 ============================

def quantile_returns(factor: pd.DataFrame, fwd: pd.DataFrame, q: int = 5,
                     min_cs: int = cfg.MIN_CS) -> pd.DataFrame:
  """
  按日截面分桶: Q1(因子最小)...Qq(因子最大) 各桶的前向收益均值, 及多空差.

  做法: 每天对有效标的按因子值取 pct 秩 -> 升序切 q 桶 -> 每桶算当日均值,
        再对所有交易日取平均(日等权, 避免有效标的多的日子权重过大).

  :param factor: 因子宽表
  :param fwd: 前向收益宽表
  :param q: 分桶数
  :param min_cs: 每日最少有效标的数
  :returns: DataFrame(bucket, mean_fwd_ret, n); bucket=Q1..Qq 及末行 LS(Qq-Q1)
  """
  idx = factor.index.intersection(fwd.index)
  col = factor.columns.intersection(fwd.columns)
  f, r = factor.loc[idx, col], fwd.loc[idx, col]
  both = f.notna() & r.notna()
  keep = both.sum(axis=1) >= min_cs
  f, r, both = f[keep], r[keep], both[keep]
  if f.empty:
    return pd.DataFrame(columns=['bucket', 'mean_fwd_ret', 'n'])

  pct = f.where(both).rank(axis=1, pct=True)      # (0,1] 的截面分位
  bucket = np.ceil(pct * q).clip(upper=q)         # 1..q
  rows = []
  for b in range(1, q + 1):
    m = bucket.eq(b)
    day_mean = r.where(m).mean(axis=1)            # 每日该桶均值(跳过 NaN)
    rows.append({'bucket': f'Q{b}',
                 'mean_fwd_ret': round(float(day_mean.mean()), 5),
                 'n': int(m.to_numpy().sum())})
  ls = rows[-1]['mean_fwd_ret'] - rows[0]['mean_fwd_ret']
  rows.append({'bucket': f'LS(Q{q}-Q1)', 'mean_fwd_ret': round(float(ls), 5), 'n': np.nan})
  return pd.DataFrame(rows)


def _spearman(a, b) -> float:
  """两条序列的 Spearman 秩相关(= 秩上的 Pearson); 常量序列返回 NaN."""
  ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
  if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
    return np.nan
  return float(np.corrcoef(ra, rb)[0, 1])


# ============================ 4. top-k 换手 ============================

def topk_turnover(factor: pd.DataFrame, top_k: int = cfg.TOP_K,
                  min_cs: int = cfg.MIN_CS) -> float:
  """
  top-k 成员的日度更替率(信号稳定度 / 成本前哨).

  每天取因子值最大的 top_k 个标的为持仓, 与前一有效交易日比较:
    换手 = 1 - |今 ∩ 昨| / k.
  有效标的数 < min_cs 的日子直接跳过(不产生信号); 只在相邻的"有效日对"上统计.

  :param factor: 因子宽表
  :param top_k: 持仓数
  :param min_cs: 每日最少有效标的数
  :returns: 平均日换手(0 = 从不换仓, 1 = 每天全换)
  """
  cnt = factor.notna().sum(axis=1)
  ok = cnt >= max(min_cs, top_k)
  rank = factor.where(ok, other=np.nan).rank(axis=1, ascending=False)   # 1 = 最大
  memb = (rank <= top_k)
  cols = np.asarray(factor.columns)
  turns, prev = [], None
  for dt in factor.index:
    if not bool(ok.loc[dt]):
      continue
    cur = set(cols[memb.loc[dt].to_numpy()])
    if prev is not None and cur:
      turns.append(1.0 - len(cur & prev) / len(cur))
    prev = cur
  return round(float(np.mean(turns)), 3) if turns else np.nan


# ============================ 5. 信号自相关衰减 ============================

def signal_ac(factor: pd.DataFrame, lags: tuple = (1, 5, 20),
              min_cs: int = cfg.MIN_CS) -> dict:
  """
  信号自相关衰减: lag = L 时, factor[t] 与 factor[t-L] 的截面秩相关均值.

  含义: 值越接近 1 => 信号越"黏"(换手低, 可承载更长持有期);
        快速落到 0 => 信号一天就过期(只适合极短持有).
  直接复用 ic_series: 把 factor.shift(L) 当作"另一个因子"即可.

  :param factor: 因子宽表
  :param lags: 要看的滞后阶
  :param min_cs: 每日最少有效标的数
  :returns: {lag: 平均秩自相关}
  """
  out = {}
  for L in lags:
    ic = ic_series(factor, factor.shift(L), 'spearman', min_cs)
    out[L] = round(float(ic.mean()), 3) if len(ic) else np.nan
  return out


# ============================ 6. 多重检验与真伪检验 ============================

def bh_qvals(p) -> np.ndarray:
  """
  Benjamini-Hochberg FDR 校正: 一组 p 值 -> 一组 q 值(保序).

  q 值 = "把这组因子的全体声明里, 假阳性比例控制在 q 以下时, 该因子能通过的门槛".
  与单看 p 的区别: 一次评估几十个因子, 光看 p < 0.05 平均会撞上几个假显著;
  BH 把"这一族一起看"的代价算进去, 故 q 值 >= p 值(q 越大 = 门槛越严).

  实现: p 升序 -> 第 i 个的 q = p_i * n / i(等价于 i/n * alpha 的翻转) -> 从右往左
  取累积最小值(保证 q 单调不减 -> 单调不减的判据才是"拒绝的单调性"). 出处:
  research_2026-09/conditional_eval.py bh_qvals.

  :param p: p 值序列(可含 NaN; NaN 原样返回 NaN, 不参与校正)
  :returns: 与输入同形/同序的 q 值数组
  """
  p = np.asarray(p, dtype=float)
  q = np.full(p.shape, np.nan)
  ok = ~np.isnan(p)
  pp = p[ok]
  n = int(pp.size)
  if n == 0:
    return q
  order = np.argsort(pp, kind='stable')
  ranked = pp[order] * n / (np.arange(n) + 1)
  ranked = np.minimum.accumulate(ranked[::-1])[::-1]     # 从大 p 到小 p 取累积最小
  qq = np.empty(n)
  qq[order] = np.minimum(ranked, 1.0)
  q[ok] = qq
  return q


def sig_ac1(sig: pd.DataFrame, min_cs: int = cfg.MIN_CS) -> float:
  """
  信号自身前后日的截面秩自相关(平均) —— 静态度探针.

  值域 [-1, 1]: 越接近 1 = 信号越"黏"(换手越低), >= STATIC_AC1(0.99) 说明信号几乎
  逐日不变 -> 典型"静态选票"(永远选同几个标的), 看着 IC 稳, 其实没有动态信息.
  与 signal_ac(sig, lags=(1,)) 的区别: 这里不借 ic_series(避免二次预处理/口径漂移),
  直接用逐日秩向量的 Pearson, 且要求前后两天都达标, 是给"身份识别"专用的稳健版.
  出处: research_2026-09/indicator_eval.py sig_ac1.

  :param sig: 因子宽表(通常是预处理后的信号)
  :param min_cs: 每日最少有效标的数
  :returns: 平均截面秩自相关; 有效日不足 60 天时返回 NaN
  """
  s = sig.replace([np.inf, -np.inf], np.nan)
  r = s.rank(axis=1, pct=True)
  a = r.shift(1)
  ok = (s.notna().sum(axis=1) >= min_cs) & (a.notna().sum(axis=1) >= min_cs)
  if int(ok.sum()) < 60:
    return np.nan
  x, y = r.loc[ok].to_numpy(), a.loc[ok].to_numpy()
  m = ~np.isnan(x) & ~np.isnan(y)
  keep = m.sum(axis=1) >= min_cs
  if int(keep.sum()) < 60:
    return np.nan
  x, y, m = x[keep], y[keep], m[keep]
  xm = np.where(m, x, np.nan)
  ym = np.where(m, y, np.nan)
  dx = np.where(m, xm - np.nanmean(xm, axis=1, keepdims=True), 0.0)
  dy = np.where(m, ym - np.nanmean(ym, axis=1, keepdims=True), 0.0)
  cov = (dx * dy).sum(axis=1)
  vx = np.sqrt((dx ** 2).sum(axis=1))
  vy = np.sqrt((dy ** 2).sum(axis=1))
  with np.errstate(divide='ignore', invalid='ignore'):
    c = np.where((vx > 0) & (vy > 0), cov / (vx * vy), np.nan)
  c = c[~np.isnan(c)]
  return round(float(np.mean(c)), 4) if len(c) >= 60 else np.nan


def validate_one(sig: pd.DataFrame, fwd: pd.DataFrame, top_k: int = cfg.TOP_K,
                 min_cs: int = cfg.MIN_CS) -> dict:
  """
  真伪检验: 分半稳定 + 静态退化对照 + 持仓集中度.

  核心问题: "这个信号的超额, 是动态选股赚的, 还是只是长期押注某几个标的?"
  做法:
    1) 把样本按时间对半切, 前半段估计"每个标的的平均分"(= 静态选票), 冻结;
    2) 用这张静态选票在后半段选 top-k(等于"从今往后一直持有前半段选出的那几只"),
       看它能赚多少 -> static_sig_test_exc;
    3) 真正的动态信号在后半段赚了多少 -> dyn_test_exc;
    4) dyn_minus_static = dyn - static. <= 0 说明"动态选股没有超过静态身份",
       这个因子的超额很可能只是"押注了某个名字", 不是可复用的择时/选股能力.
  另给持仓集中度: top3_share(前三名被选中频率之和) / hhi(频率平方和) /
  n_eff_symbols = 1/hhi(有效持仓数). n_eff 远小于 top_k 就是"集中押注"的警报.

  出处: research_2026-09/signal_search.py validate_one.

  :param sig: 因子宽表(预处理后的信号)
  :param fwd: 前向收益宽表
  :param top_k: 持仓数
  :param min_cs: 每日最少有效标的数
  :returns: dict(k, exc_half1, exc_half2, dyn_test_exc, static_sig_test_exc,
                 static_oracle_test_exc, dyn_minus_static, top3_share, hhi,
                 n_eff_symbols); 样本不足时返回 {}
  """
  idx = sig.index.intersection(fwd.index)
  cols = sig.columns.intersection(fwd.columns)
  if len(idx) < 120 or len(cols) < 6:
    return {}
  s = sig.loc[idx, cols].to_numpy(dtype=float)
  r = fwd.loc[idx, cols].to_numpy(dtype=float)
  m = ~np.isnan(s) & ~np.isnan(r)
  ok = m.sum(axis=1) >= min_cs
  if int(ok.sum()) < 120:
    return {}
  sv = np.where(m & ok[:, None], s, np.nan)
  rv = np.where(m & ok[:, None], r, np.nan)
  k = max(1, min(top_k, int(m.sum(axis=1)[ok].min()) // 3))     # 别让持仓吃掉整个截面
  order = np.argsort(np.where(np.isnan(sv), -np.inf, sv), axis=1, kind='stable')[:, ::-1]
  rows = np.where(ok)[0]
  fvr = rv[rows]
  top = order[rows, :k]
  top_ret = np.nanmean(np.take_along_axis(fvr, top, axis=1), axis=1)
  pool = np.nanmean(fvr, axis=1)
  exc = top_ret - pool
  h = len(rows) // 2
  ex1, ex2 = float(np.nanmean(exc[:h])), float(np.nanmean(exc[h:]))
  freq = np.bincount(top.ravel(), minlength=len(cols)) / len(rows)   # 各标的被选中频率
  top3_share = float(np.sort(freq)[::-1][:3].sum())
  hhi = float((freq ** 2).sum())

  def static_excess(score_per_symbol: np.ndarray) -> float:
    """用"每标的固定分"在后半段选 top-k 的超额(静态选票的检验段表现)."""
    sel = np.argsort(-np.nan_to_num(score_per_symbol, nan=-np.inf))[:k]
    t2 = np.nanmean(fvr[h:][:, sel], axis=1)
    p2 = np.nanmean(fvr[h:], axis=1)
    return float(np.nanmean(t2 - p2))

  stat_sig = static_excess(np.nanmean(sv[rows[:h]], axis=0))       # 前半段冻结的静态选票
  stat_oracle = static_excess(np.nanmean(rv[rows[:h]], axis=0))    # 事后最优静态选票(上界)
  return {'k': k,
          'exc_half1': round(ex1, 5), 'exc_half2': round(ex2, 5),
          'dyn_test_exc': round(ex2, 5),
          'static_sig_test_exc': round(stat_sig, 5),
          'static_oracle_test_exc': round(stat_oracle, 5),
          'dyn_minus_static': round(ex2 - stat_sig, 5),
          'top3_share': round(top3_share, 3), 'hhi': round(hhi, 3),
          'n_eff_symbols': round(1.0 / hhi, 1) if hhi > 0 else np.nan}


# ============================ 汇总: 单因子评估 ============================

def topk_return(factor: pd.DataFrame, fwd: pd.DataFrame, top_k: int = cfg.TOP_K,
                min_cs: int = cfg.MIN_CS) -> tuple:
  """
  top-k 等权组合的前向收益 与 池等权基准收益(日等权平均).

  :param factor: 因子宽表
  :param fwd: 前向收益宽表
  :param top_k: 持仓数
  :param min_cs: 每日最少有效标的数
  :returns: (topk_ret, pool_ret)
  """
  idx = factor.index.intersection(fwd.index)
  col = factor.columns.intersection(fwd.columns)
  f, r = factor.loc[idx, col], fwd.loc[idx, col]
  both = f.notna() & r.notna()
  ok = both.sum(axis=1) >= max(min_cs, top_k)          # 逐日布尔(Series, 行对齐)
  order = f.where(ok).rank(axis=1, ascending=False)    # 有效日才排名, 否则 NaN
  top_ret = r.where(order <= top_k).mean(axis=1).mean()  # 每日 top-k 均值, 再日等权
  mask = both.to_numpy() & ok.to_numpy()[:, None]      # 逐格: 成对有效 且 当天达标
  pool_ret = r.where(mask).mean(axis=1).mean()
  return float(top_ret), float(pool_ret)


def evaluate_one(spec: fct.FactorSpec, sig: pd.DataFrame, fwd: pd.DataFrame,
                 top_k: int = cfg.TOP_K, q: int = 5, min_cs: int = cfg.MIN_CS,
                 nw_lag: int = 0, ac_lags: tuple = (1, 5, 20)) -> dict:
  """
  评估单个因子 -> 一行指标.

  :param spec: 因子元信息(取 group / direction)
  :param sig: 预处理后的因子宽表
  :param fwd: 前向收益宽表
  :param top_k: top-k 持仓数
  :param q: 分层桶数
  :param min_cs: 每日最少有效标的数
  :param nw_lag: NW-HAC 滞后阶(重叠标签场景, 调用方传 max(h-1, 1))
  :param ac_lags: 自相关滞后阶
  :returns: dict(评估指标, 含 nw_p / sig_ac1 / is_static / 集中度 / dyn_minus_static
                 等诚实性字段)
  """
  ic_p = ic_summary(ic_series(sig, fwd, 'pearson', min_cs), nw_lag)     # IC
  ic_s = ic_summary(ic_series(sig, fwd, 'spearman', min_cs), nw_lag)    # RankIC
  top_ret, pool_ret = topk_return(sig, fwd, top_k, min_cs)

  qr = quantile_returns(sig, fwd, q, min_cs)
  means = qr['mean_fwd_ret'].iloc[:q].to_numpy(dtype=float)             # Q1..Qq
  ls_spread = float(means[-1] - means[0])
  mono = round(_spearman(np.arange(q), means), 3) if q >= 2 else np.nan

  ac = signal_ac(sig, ac_lags, min_cs)
  rank_ic = ic_s['ic_mean']
  dir_ok = bool(spec.direction * rank_ic >= 0) if pd.notna(rank_ic) else False

  # 诚实性: 静态度探针 + 真伪检验(样本不足时 validate_one 返回 {}, 各字段取 NaN)
  sac1 = sig_ac1(sig, min_cs)
  vd = validate_one(sig, fwd, top_k, min_cs)

  rec = {
      'factor': spec.name,
      'group': spec.group,
      'dir': spec.direction,
      'dir_ok': dir_ok,
      # 预测力
      'n_days': ic_s['n_days'],
      'ic': ic_p['ic_mean'],
      'rank_ic': rank_ic,
      'rank_icir': ic_s['ic_ir'],
      't_stat': ic_s['t_stat'],
      'nw_t': ic_s['nw_t'],
      'pos_rate': ic_s['pos_rate'],
      # 组合表现
      'topk_ret': round(top_ret, 5),
      'pool_ret': round(pool_ret, 5),
      'excess': round(top_ret - pool_ret, 5),
      'ls_spread': round(ls_spread, 5),
      'mono': mono,
      'turnover': topk_turnover(sig, top_k, min_cs),
      # 诚实性: 多重检验输入(p) / 真伪检验 / 静态度
      'nw_p': ic_s['nw_p'],
      'sig_ac1': sac1,
      'is_static': bool(sac1 >= STATIC_AC1) if pd.notna(sac1) else False,
      'exc_half1': vd.get('exc_half1', np.nan),
      'exc_half2': vd.get('exc_half2', np.nan),
      'dyn_minus_static': vd.get('dyn_minus_static', np.nan),
      'static_sig_test_exc': vd.get('static_sig_test_exc', np.nan),
      'top3_share': vd.get('top3_share', np.nan),
      'hhi': vd.get('hhi', np.nan),
      'n_eff_symbols': vd.get('n_eff_symbols', np.nan),
  }
  rec.update({f'ac{L}': ac.get(L, np.nan) for L in ac_lags})
  return rec


# ============================ 汇总: 批量 + 窗口 ============================

def window_bounds(window: str = 'full', start: str = None, end: str = None) -> tuple:
  """
  研究窗口 -> (start, end). 显式传入的 start/end 优先.

  :param window: 'full'(全交易窗) / 'is'(样本内) / 'oos'(样本外)
  :returns: (start, end); None 表示不限
  """
  if start is not None or end is not None:
    return start, end
  if window == 'is':
    return cfg.TRADE_START, cfg.IS_END
  if window == 'oos':
    return cfg.OOS_START, None
  return cfg.TRADE_START, None


def _window(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
  """按交易日截取宽表的行(不改内容, 只选时间窗)."""
  idx = df.index
  if start is not None:
    idx = idx[idx >= pd.Timestamp(start)]
  if end is not None:
    idx = idx[idx <= pd.Timestamp(end)]
  return df.loc[idx]


def evaluate_factors(ds: Dataset, names: list = None, method: str = prep.DEFAULT_PREP,
                     h: int = 1, top_k: int = cfg.TOP_K, q: int = 5,
                     min_cs: int = cfg.MIN_CS, window: str = 'full',
                     start: str = None, end: str = None,
                     ac_lags: tuple = (1, 5, 20)) -> pd.DataFrame:
  """
  批量评估一批因子 -> 每行一个因子.

  关键: 前向收益在**全历史**上算, 之后才截评估窗口. 因为标签需要未来 h+1 天的
  开盘价, 若先截断再算 fwd, 窗口末尾 h+1 天的标签会全丢. 截窗口只是筛选"哪些
  交易日的 IC 进入统计", 不影响因子因果性(因子仍只看 <= t 的数据).

  :param ds: Dataset
  :param names: 因子名列表(None = 全部)
  :param method: 预处理方法(见 prepare.PREP_METHODS)
  :param h: 前向持有天数
  :param top_k: top-k 持仓数
  :param q: 分层桶数
  :param min_cs: 每日最少有效标的数
  :param window: 'full' / 'is' / 'oos'
  :param start/end: 显式窗口(覆盖 window)
  :param ac_lags: 自相关滞后阶
  :returns: DataFrame(评估表), 按 rank_ic 降序; 另含 fdr_q / fdr_sig(BH-FDR 校正)
  """
  start, end = window_bounds(window, start, end)
  fwd_full = forward_return(ds, h)
  prep_fn = prep._make_prep(method, min_cs)
  rows = []
  for name in (names or fct.list_factors()):
    spec = fct.get_factor(name)
    sig = prep_fn(spec.compute(ds))
    rows.append(evaluate_one(spec, _window(sig, start, end),
                             _window(fwd_full, start, end),
                             top_k, q, min_cs, nw_lag=max(h - 1, 1), ac_lags=ac_lags))
  df = pd.DataFrame(rows)
  # BH-FDR: 必须在排序之前算(校正依赖"这一族全体", 与展示顺序无关), 再保序赋回
  df['fdr_q'] = bh_qvals(df['nw_p'].to_numpy())
  df['fdr_sig'] = df['fdr_q'] < FDR_ALPHA
  return df.sort_values('rank_ic', ascending=False).reset_index(drop=True)


# ============================ 自检: 引擎标定 ============================

def verify_eval(ds: Dataset, h: int = 1, method: str = prep.DEFAULT_PREP,
                min_cs: int = cfg.MIN_CS, seed: int = 0,
                top_k: int = cfg.TOP_K) -> pd.DataFrame:
  """
  评估引擎标定(正负两个极端 + 一个"静态身份"探针, 看指标是否落在该在的地方).

    1) oracle : 直接拿"前向收益本身"当因子 -> RankIC 应为 1.0;
    2) random : 随机数因子(对齐真实因子的 NaN 格局) -> RankIC 应约等于 0;
    3) static : "前半段每标的平均分"铺满全期(时间上恒定, 典型静态选票)
                -> sig_ac1 应 >= STATIC_AC1, 用于标定静态检测器确实会报警;
                   dyn_minus_static 应 <= 0(它没有超过自己前半段的静态身份).

  三项都符合预期, 才说明 IC 计算、NaN 配对、截面方向都没写错, 且"静态 vs 动态"
  的判别探针真的能把静态身份挑出来.

  :param ds: Dataset
  :param h: 前向持有天数
  :param method: 预处理方法
  :param min_cs: 每日最少有效标的数
  :param seed: 随机种子
  :param top_k: top-k 持仓数(供 validate_one 用; 放在 seed 之后以兼容旧调用位置)
  :returns: DataFrame(case, n_days, rank_ic, sig_ac1, dyn_minus_static)
  """
  fwd = forward_return(ds, h)
  prep_fn = prep._make_prep(method, min_cs)
  base = fct.get_factor('mom_20').compute(ds)                  # 借 NaN 格局
  rows = []

  def _row(case: str, raw: pd.DataFrame) -> dict:
    sig = prep_fn(raw)
    s = ic_summary(ic_series(sig, fwd, 'spearman', min_cs), 0)
    return {'case': case, 'n_days': s['n_days'], 'rank_ic': s['ic_mean'],
            'sig_ac1': sig_ac1(sig, min_cs),
            'dyn_minus_static': validate_one(sig, fwd, top_k, min_cs).get(
                'dyn_minus_static', np.nan)}

  rows.append(_row('oracle(=fwd)', fwd))                       # 完美预知

  rng = np.random.default_rng(seed)
  rows.append(_row('random(对齐NaN)', pd.DataFrame(
      rng.standard_normal(base.shape), index=base.index, columns=base.columns
  ).where(base.notna())))

  half = base.index[:len(base.index) // 2]                     # 前半段 -> 每标的均分
  const = base.loc[half].mean(axis=0)
  stat = pd.DataFrame(np.tile(const.to_numpy(), (len(base.index), 1)),
                      index=base.index, columns=base.columns).where(base.notna())
  rows.append(_row('static(常量选票)', stat))
  return pd.DataFrame(rows)


# ============================ CLI ============================

def _show_one(ds: Dataset, name: str, method: str, h: int, q: int,
              min_cs: int, start, end, ac_lags: tuple) -> None:
  """打印单因子的分层收益表 + 自相关衰减(教学视角)."""
  log = cfg.get_logger('factor.evaluate')
  spec = fct.get_factor(name)
  prep_fn = prep._make_prep(method, min_cs)
  sig = prep_fn(spec.compute(ds))
  fwd = forward_return(ds, h)
  log.info(f'\n[{name}] {spec.group} | {spec.formula} | 先验方向 {spec.direction:+d}')
  log.info(f'[分层收益 q={q}] h={h}, 窗口 {start or "-"}~{end or "-"}\n'
           + quantile_returns(_window(sig, start, end), _window(fwd, start, end),
                              q, min_cs).to_string(index=False))
  ac = signal_ac(_window(sig, start, end), ac_lags, min_cs)
  log.info('[自相关衰减] ' + ', '.join(f'ac{L}={ac[L]}' for L in ac_lags))


def main():
  ap = argparse.ArgumentParser(description='factor 评估层: 前向收益 / IC / 分层 / 换手')
  ap.add_argument('--pool', default=cfg.DEFAULT_POOL)
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  ap.add_argument('--method', default=prep.DEFAULT_PREP, choices=prep.PREP_METHODS)
  ap.add_argument('--h', type=int, default=1, help='前向持有天数(开盘到开盘)')
  ap.add_argument('--top-k', type=int, default=cfg.TOP_K)
  ap.add_argument('--q', type=int, default=5, help='分层桶数')
  ap.add_argument('--min-cs', type=int, default=cfg.MIN_CS)
  ap.add_argument('--window', default='full', choices=['full', 'is', 'oos'])
  ap.add_argument('--start', default=None)
  ap.add_argument('--end', default=None)
  ap.add_argument('--split', action='store_true', help='同时打印 IS 与 OOS 两窗对照')
  ap.add_argument('--group', default=None, help='只看某类因子')
  ap.add_argument('--check', action='store_true', help='运行引擎标定(oracle/random)')
  ap.add_argument('--show', default=None, help='打印某因子的分层收益与自相关衰减')
  a = ap.parse_args()

  log = cfg.get_logger('factor.evaluate')
  ds = load_pool(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')
  log.info(f'口径: method={a.method}, h={a.h}, top_k={a.top_k}, q={a.q}, '
           f'min_cs={a.min_cs}')

  names = fct.list_factors(a.group)
  windows = [('is', None, None), ('oos', None, None)] if a.split \
      else [(a.window, a.start, a.end)]
  for wname, ws, we in windows:
    df = evaluate_factors(ds, names, a.method, a.h, a.top_k, a.q, a.min_cs,
                          wname, ws, we)
    lo, hi = window_bounds(wname, ws, we)
    log.info(f'\n===== 窗口 {wname}  ({lo or "数据起点"} ~ {hi or "至今"}), '
             f'按 RankIC 降序, {len(df)} 个因子 =====')
    log.info('\n' + df[PRED_COLS].to_string(index=False))
    log.info('\n' + df[PERF_COLS].to_string(index=False))
    log.info('\n' + df[HONEST_COLS].to_string(index=False))
    n_sig = int(df['fdr_sig'].sum())
    st = df.loc[df['is_static'], 'factor'].tolist()
    log.info(f'[诚实性] BH-FDR alpha={FDR_ALPHA}: 显著 {n_sig}/{len(df)} 个'
             f'(fdr_q < {FDR_ALPHA}); 静态选票(sig_ac1>={STATIC_AC1}): {st or "无"}')
    bad = df.loc[~df['dir_ok'], 'factor'].tolist()
    if bad:
      log.info(f'[方向提醒] 实测 RankIC 与先验方向不符: {bad}')

  if a.check:
    log.info(f'\n[引擎标定] h={a.h}: oracle 应 ~1.0, random 应 ~0.0, '
             f'static 的 sig_ac1 应 >={STATIC_AC1} 且 dyn_minus_static <= 0')
    log.info('\n' + verify_eval(ds, a.h, a.method, a.min_cs, top_k=a.top_k)
             .to_string(index=False))

  if a.show:
    start, end = window_bounds(a.window, a.start, a.end)
    _show_one(ds, a.show, a.method, a.h, a.q, a.min_cs, start, end, (1, 5, 20))


if __name__ == '__main__':
  main()

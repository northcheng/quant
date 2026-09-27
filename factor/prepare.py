# -*- coding: utf-8 -*-
"""
factor.prepare — 预处理层(第2步)
==================================
职责: 把因子层的原始宽表, 变成"同一天里可以跨标的比较"的分数.

  因子层只回答"这个数字是多少"; 本层回答"它该排在什么位置".

本层做三件事:
  1. 缺失处理: NaN 原样屏蔽(不填 0), 且截面有效标的数 < MIN_CS 的交易日整行作废;
  2. 去极值  : 逐日截面 MAD / 分位数 截尾;
  3. 标准化  : 逐日截面 zscore 或 秩(rank) -> [-1, 1].

--- 唯一的红线: 统计量必须"逐日截面"算 ---
  正确的: df.mean(axis=1)  每天单独算, 当天收盘后就确定 -> 因果;
  错误的: df.mean()        跨日算, 用到了尚未发生的交易日 -> 前视.

  两者在代码上只差一个 axis 参数, 结论却完全相反. 因为太容易写错, 本模块
  所有函数一律不接受"全局统计量", 并用 truncation_check() 兜底:
    A = 对全历史因子算完 prep, 再截到 cut
    B = 先把因子截到 cut, 再做 prep
  逐值比对 A / B, 相同才放行.

--- 秩变换与去极值的关系(决定默认方法) ---
  rank 本身对极值免疫(最大值只是"最大的那个名次"), 所以"去极值 + rank"是冗余的;
  去极值只在 zscore 之前才有意义. 因此默认方法取 rank —— 本池每天只有 28 个标的的
  截面太薄, 一个异常值足以把 mean 和 std 同时带偏, 而 rank 完全不受影响.

自检:
  cd ~/git && python -m quant.factor.prepare --pool etf_3x
  cd ~/git && python -m quant.factor.prepare --pool etf_3x --check
  cd ~/git && python -m quant.factor.prepare --pool etf_3x --check --demo
  cd ~/git && python -m quant.factor.prepare --pool etf_3x --show mom_20
"""
import argparse
from typing import Callable

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import factor as fct
from quant.factor.data import Dataset, load_pool

MAD_K = 1.4826                                  # 正态下 MAD -> sigma 的换算系数

PrepFn = Callable[[pd.DataFrame], pd.DataFrame]  # 预处理函数: 宽表 -> 宽表


# ============================ 1. 缺失处理 ============================

def guard_rows(df: pd.DataFrame, min_cs: int = cfg.MIN_CS) -> pd.DataFrame:
  """
  截面太薄的交易日整行作废.

  有效标的数 < min_cs 时, 当天排名缺乏统计意义(尤其本池只有 28 个标的),
  整行置 NaN, 交给下游当"无信号"处理.

  :param df: (date × symbol) 宽表
  :param min_cs: 每日最少有效标的数
  :returns: 新宽表(NaN 保持 NaN, 不填 0)
  """
  n = df.notna().sum(axis=1)
  out = df.copy()
  out.loc[n < min_cs, :] = np.nan
  return out


# ============================ 2. 去极值 ============================

def winsorize(df: pd.DataFrame, method: str = 'mad', n: float = 3.0,
              q: float = 0.01) -> pd.DataFrame:
  """
  逐日截面去极值(截尾): 把超出边界的值拉到边界上(不是删掉).

  - method='mad'      : 中位数 ± n * 1.4826 * MAD. 极值不参与估计, 比 mean/std 稳;
                        某日 MAD = 0(值大面积相同)时退回用该日 std, 仍无效则不截.
  - method='quantile' : 该日 [q, 1-q] 分位数为边界.

  注意: 边界是"当天的"截面统计量 -> 当天收盘即可算出 -> 因果.

  :param df: (date × symbol) 宽表
  :param method: 'mad' 或 'quantile'
  :param n: MAD 倍数(3.0 大致对应正态下的 3 sigma)
  :param q: 分位数截尾比例(0.01 -> 掐掉两头各 1%)
  :returns: 截尾后的新宽表
  """
  if method == 'mad':
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1) * MAD_K
    sd = df.std(axis=1)
    scale = mad.where(mad > 0).fillna(sd.where(sd > 0))    # MAD 失效时退回 std
    ok = scale > 0
    lo = (med - n * scale).where(ok, -np.inf)              # 无法估计尺度 -> 不截
    hi = (med + n * scale).where(ok, np.inf)
  elif method == 'quantile':
    lo = df.quantile(q, axis=1).fillna(-np.inf)
    hi = df.quantile(1 - q, axis=1).fillna(np.inf)
  else:
    raise KeyError(f'未知去极值方法: {method} (可选 mad / quantile)')
  return df.clip(lower=lo, upper=hi, axis=0)


# ============================ 3. 标准化 ============================

def zscore(df: pd.DataFrame) -> pd.DataFrame:
  """
  逐日截面 zscore: (x - 当日均值) / 当日标准差.

  某日标准差为 0/NaN(值全相同或只有一个有效值)时, 该日有效值记为 0(中性),
  而不是留下 NaN —— 否则下游会因为分母问题整行丢失.

  :param df: (date × symbol) 宽表
  :returns: 新宽表(NaN 保持 NaN)
  """
  mu = df.mean(axis=1)
  sd = df.std(axis=1)
  out = df.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
  return out.fillna(0.0).where(df.notna())


def rank_norm(df: pd.DataFrame) -> pd.DataFrame:
  """
  逐日截面秩变换 -> [-1, 1](最差 = -1, 最好 = +1, 对称).

  不是简单的 percentile: 用 (rank - (n+1)/2) / ((n-1)/2) 做对称映射,
  这样 n=1 时自然得到 NaN(没有可比对象), 而不是凭空给个 +1.

  对极值免疫, 且不假设分布形状 —— 本池截面薄, 这是默认选择.

  :param df: (date × symbol) 宽表
  :returns: 新宽表, 取值 [-1, 1](NaN 保持 NaN)
  """
  n = df.notna().sum(axis=1)
  r = df.rank(axis=1, method='average', na_option='keep')
  half = (n - 1) / 2
  return r.sub((n + 1) / 2, axis=0).div(half.where(half > 0), axis=0)


# ============================ 方法注册 ============================

PREP_METHODS = ('raw', 'z', 'rank', 'mad_z', 'q_z')
DEFAULT_PREP = 'rank'


def _make_prep(method: str, min_cs: int) -> PrepFn:
  """
  方法名 -> 预处理函数(已绑定 min_cs).

  :param method: PREP_METHODS 之一
  :param min_cs: 每日最少有效标的数
  :returns: PrepFn(宽表 -> 宽表)
  """
  steps = {
      'raw':    [],                                        # 对照基线: 不动因子值
      'z':      [zscore],
      'rank':   [rank_norm],
      'mad_z':  [lambda d: winsorize(d, 'mad'), zscore],
      'q_z':    [lambda d: winsorize(d, 'quantile'), zscore],
  }
  if method not in steps:
    raise KeyError(f'未知预处理方法: {method} (可选 {PREP_METHODS})')

  def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    for fn in steps[method]:
      out = fn(out)
    return guard_rows(out, min_cs)                       # 最后统一屏蔽薄截面

  return prep


def prepare(df: pd.DataFrame, method: str = DEFAULT_PREP,
            min_cs: int = cfg.MIN_CS) -> pd.DataFrame:
  """
  对单个因子宽表做预处理.

  :param df: 因子宽表(date × symbol)
  :param method: PREP_METHODS 之一
  :param min_cs: 每日最少有效标的数
  :returns: 预处理后的宽表
  """
  return _make_prep(method, min_cs)(df)


def prepare_all(ds: Dataset, method: str = DEFAULT_PREP, group: str = None,
                min_cs: int = cfg.MIN_CS) -> dict:
  """
  对一批因子做预处理(因子层算完后逐个过一遍).

  :param ds: Dataset
  :param method: PREP_METHODS 之一
  :param group: 只处理某类因子(None = 全部)
  :param min_cs: 每日最少有效标的数
  :returns: {因子名: 预处理后宽表}
  """
  prep = _make_prep(method, min_cs)
  return {n: prep(fct.get_factor(n).compute(ds)) for n in fct.list_factors(group)}


# ============================ 自检: 效果对照 + 截断一致性 ============================

def effect_table(ds: Dataset, methods: list = None, group: str = None) -> pd.DataFrame:
  """
  去极值/标准化效果对照: 各方法下, 因子截面值的最大绝对值.

  看点: raw 是原始尺度; z 会被极值放大(可能到十几); mad_z 把尾巴压到 ~n;
        rank 恒等于 1 —— 这就是"rank 天然免疫极值"的直接证据.

  :param ds: Dataset
  :param methods: 要对照的方法(None = ['raw', 'z', 'mad_z', 'rank'])
  :param group: 只统计某类因子
  :returns: DataFrame(factor, <method>...)
  """
  methods = methods or ['raw', 'z', 'mad_z', 'rank']
  rows = []
  for name in fct.list_factors(group):
    raw = fct.get_factor(name).compute(ds)
    row = {'factor': name, 'group': fct.get_factor(name).group}
    for m in methods:
      v = prepare(raw, m).to_numpy(dtype=float)
      row[m] = round(float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else np.nan, 3)
    rows.append(row)
  return pd.DataFrame(rows)


def _compare(a: pd.DataFrame, b: pd.DataFrame):
  """逐值比对两张宽表 -> (cells, max_diff, nan_same). 形状不一致直接判失败."""
  idx, col = a.index.intersection(b.index), a.columns.intersection(b.columns)
  a, b = a.loc[idx, col], b.loc[idx, col]
  if a.shape != b.shape or a.size == 0:
    return 0, float('nan'), False
  nan_same = not bool((np.isnan(a.to_numpy()) != np.isnan(b.to_numpy())).any())
  with np.errstate(invalid='ignore'):
    d = np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
    max_diff = 0.0 if np.isnan(d).all() else float(np.nanmax(d))
  return int(a.size), max_diff, nan_same


def truncation_check(ds: Dataset, prep_fn: PrepFn, cut: str = '2024-06-28',
                     tol: float = 1e-9, group: str = None) -> pd.DataFrame:
  """
  预处理层的截断一致性检验(与第1步 factor.verify_causal 同一原理, 但检验的是 prep).

  因子本身第1步已证明因果, 所以这里把因子输出当作共同输入, 只隔离 prep 的影响:
    A = prep(全历史因子) 再截到 cut
    B = 先截到 cut, 再 prep
  A 与 B 逐值相同 -> prep 没有偷看未来.

  :param ds: Dataset
  :param prep_fn: 预处理函数
  :param cut: 截断日
  :param tol: 允许的最大绝对差
  :param group: 只检验某类因子
  :returns: DataFrame(factor, group, cells, max_diff, nan_same, causal)
  """
  part = ds.slice(end=cut)
  rows = []
  for name in fct.list_factors(group):
    raw = fct.get_factor(name).compute(ds)               # 第1步已证因果, 作共同输入
    a = prep_fn(raw).loc[:cut]
    b = prep_fn(raw.loc[:cut])
    cells, max_diff, nan_same = _compare(a, b)
    rows.append({
        'factor': name,
        'group': fct.get_factor(name).group,
        'cells': cells,
        'max_diff': max_diff,
        'nan_same': nan_same,
        'causal': bool(cells > 0 and nan_same and max_diff <= tol),
    })
  return pd.DataFrame(rows)


def verify_prepare(ds: Dataset, method: str = DEFAULT_PREP, cut: str = '2024-06-28',
                   tol: float = 1e-9, min_cs: int = cfg.MIN_CS,
                   group: str = None) -> pd.DataFrame:
  """对某个预处理方法做截断一致性检验(truncation_check 的便捷封装)."""
  return truncation_check(ds, _make_prep(method, min_cs), cut, tol, group)


def _leak_fullsample_z(df: pd.DataFrame) -> pd.DataFrame:
  """反例(故意前视): 用整个样本的均值/标准差标准化 —— 检验必须能抓出它."""
  v = df.to_numpy(dtype=float)
  return (df - np.nanmean(v)) / np.nanstd(v)


def main():
  ap = argparse.ArgumentParser(description='factor 预处理层自检: 效果对照 + 因果性检验')
  ap.add_argument('--pool', default=cfg.DEFAULT_POOL)
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  ap.add_argument('--group', default=None, help='只看某类因子')
  ap.add_argument('--method', default=DEFAULT_PREP, choices=PREP_METHODS)
  ap.add_argument('--min-cs', type=int, default=cfg.MIN_CS)
  ap.add_argument('--check', action='store_true', help='做截断一致性检验')
  ap.add_argument('--demo', action='store_true', help='注入前视反例, 验证检验有效')
  ap.add_argument('--cut', default='2024-06-28')
  ap.add_argument('--show', default=None, help='打印某因子预处理后最近 5 行')
  a = ap.parse_args()

  log = cfg.get_logger('factor.prepare')
  ds = load_pool(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  log.info(f'\n[去极值/标准化效果] 各方法下因子截面值 max|value|  (min_cs={a.min_cs})')
  log.info('\n' + effect_table(ds, group=a.group).to_string(index=False))

  raw = fct.get_factor(a.show or fct.list_factors(a.group)[0]).compute(ds)
  pr = prepare(raw, a.method, a.min_cs)
  log.info(f'\n[默认方法 {a.method}] 单因子示例: '
           f'有效值占比 {raw.notna().mean().mean():.1%} -> {pr.notna().mean().mean():.1%}, '
           f'取值范围 [{np.nanmin(pr.to_numpy()):.3f}, {np.nanmax(pr.to_numpy()):.3f}]')

  if a.check:
    ck = verify_prepare(ds, a.method, a.cut, min_cs=a.min_cs, group=a.group)
    log.info(f'\n[截断一致性检验] method={a.method}, 全历史 prep vs 只喂到 {a.cut}')
    log.info('\n' + ck.to_string(index=False))
    bad = ck.loc[~ck['causal'], 'factor'].tolist()
    log.info('[结论] 预处理因果' if not bad else f'[结论] 疑似前视: {bad}')

    if a.demo:
      dm = truncation_check(ds, _leak_fullsample_z, a.cut, group=a.group)
      n_bad = int((~dm['causal']).sum())
      log.info(f'\n[反向验证] 注入"全样本 zscore"反例 -> '
               f'{n_bad}/{len(dm)} 个被抓出 (max_diff 示例 '
               f'{dm["max_diff"].max():.3f})')
      log.info('[结论] 检验有效' if n_bad == len(dm) else '[结论] 检验失效, 需排查')

  if a.show:
    log.info(f'\n[{a.show}] 预处理后最近 5 行:\n{pr.tail(5).to_string()}')


if __name__ == '__main__':
  main()

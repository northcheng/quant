# -*- coding: utf-8 -*-
"""
factor.trend — 分析师预期修正(analyst estimate trend)因子层(自包含)
====================================================================
职责: 把 EODHD fundamentals JSON(~ /quant/fundamental_data/fundamentals/*.json) 里的
      `Earnings.Trend` 板块转成 **PIT(Point-in-Time, 时点)可比**的 (date × symbol) 宽表,
      并注册为因子.

为什么是 PIT:
  `Earnings.Trend` 每条 = 某预测期(period ∈ {0q,+1q,0y,+1y}) 的一次快照, 快照时点 ≈ 该期公告日.
  该板块**无显式 as-of 字段**, 故用 `Earnings.History` 按报告期(date) join 出 reportDate
  作为**可用日(avail_date)** —— 保守, 全部向后看(信息在公告日之后才可用).

数据口径:
  * 只保留 `0q`(普通季)+`0y`(财年末季): 二者合起来 = 完整季度覆盖;
    排除未来指引 +1q/+1y(无锚定公告日, 无法 PIT).
  * 早期快照未做拆股调整 -> 只构造**比率型/增长率型**因子(量纲无关).
  * 同 (symbol, avail_date) 去重取最后.

因子(全部为比率型, 量纲无关, 免受早期快照未做拆股调整影响):
  T_eps_growth    = growth                        预期 EPS 增速(顶层 growth 字段)
  R_rev_growth    = revenueEstimateGrowth         预期营收增速
  T_eps_rev_1w    = (cur - 7d)  / |7d|            近 1 周 EPS 预期修正幅度
  T_eps_rev_1m    = (cur - 30d) / |30d|           近 1 月 EPS 预期修正幅度
  T_eps_rev_3m    = (cur - 90d) / |90d|           近 3 月 EPS 预期修正幅度
  T_eps_rev_accel = 近 1 月修正 - 上一 1 月修正      修正加速度(二阶动量)
  T_rev_breadth   = (up30 - down30)/(up30+down30) 净上调广度(分析师上调占比)
  T_eps_disp      = (high - low) / |avg|          预期分歧度(不确定性)
  T_coverage      = NumberOfAnalysts              分析师覆盖数(注意力/关注度)
  —— 均为原始值, 不做截面处理(去极值/标准化/排名交给 prepare.py).

月度路径(提高更新频率; 见 _symbol_trend_monthly):
  原快照每条只在"公告日"产生 1 个观测 -> 日频序列近乎分段常数(ac1 ~ 0.99), 静态性高.
  但每条快照内含 A / A-7 / A-30 / A-60 / A-90 五个**冻结**共识点, 可拆出 3 个间隔 30 天的
  月度观测, 把"每季 1 个观测"提升到"每月 1 个观测":
  T_eps_rev_m     = 当前月度 30 日修正 (v_cur - v_{30d})/|v_{30d}|, 按各自真实 asof 落点
  T_eps_rev_m3    = 近 3 个月度修正之和(≈ 90 日累计修正, 但按月更新)
  —— 二者仍是比率型; 供"降低静态性 / 提高 IC 显著性"的方向验证.

因果性:
  事件只读 <= t 已公告的快照; 对齐到交易日网格后 ffill 严格向后看,
  与喂给 Dataset 的长度无关(截断一致性成立) -> 通过 verify_causal.

产物:
  * 事件长表缓存: output/trend/trend_events.pkl(带 JSON 目录签名, 命中即复用);
  * 因子: group='analyst_trend' 下 11 个(T_eps_growth / R_rev_growth / T_eps_rev_1w /
    T_eps_rev_1m / T_eps_rev_3m / T_eps_rev_accel / T_rev_breadth / T_eps_disp / T_coverage /
    T_eps_rev_m / T_eps_rev_m3); 后 2 个来自月度路径表(②).

自检:
  cd ~/git && python -m quant.factor.trend --pool us_500
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import factor as _fct
from quant.factor.data import Dataset, load_pool

# 以 `python -m quant.factor.trend` 直接运行时, factor.py 底部导入会先生成 canonical
# 副本(quant.factor.trend)并完成注册, 再轮到本 __main__ 副本执行装饰器 -> 会重复注册.
# 此处令 __main__ 副本复用 no-op 装饰器, 只跑自检, 不再注册.
_RERUN = __name__ == '__main__' and 'quant.factor.trend' in sys.modules
_register = _fct.register if not _RERUN else (lambda *a, **k: (lambda fn: fn))

FUND_JSON_DIR = cfg.HOME / 'quant' / 'fundamental_data' / 'fundamentals'
CACHE_DIR = cfg.OUT_DIR / 'trend'
EVENTS_PKL = CACHE_DIR / 'trend_events.pkl'
MONTHLY_PKL = CACHE_DIR / 'trend_monthly.pkl'   # 月度共识修正路径(②: 降静态性)
PERIODS = ('0q', '0y')                          # 仅保留"当季/当年"快照(完整季度覆盖)
# 事件长表列(预期修正/离散度/覆盖度所需的原始原料)
EVENT_COLS = ('T_eps_growth', 'R_rev_growth', 'n_analysts', 'eps_avg', 'eps_high',
              'eps_low', 'v0', 'v7', 'v30', 'v60', 'v90', 'up30', 'down30')
# 本层注册的因子列(全部比率型)
TREND_COLS = ('T_eps_growth', 'R_rev_growth', 'T_eps_rev_1w', 'T_eps_rev_1m',
              'T_eps_rev_3m', 'T_eps_rev_accel', 'T_rev_breadth', 'T_eps_disp',
              'T_coverage')
# 月度路径因子列(源自独立月度事件表, 见 build_monthly)
MONTHLY_COLS = ('T_eps_rev_m', 'T_eps_rev_m3')
_SCHEMA = 2                                     # 缓存结构版本: 口径变更时 +1, 自动失效旧缓存

# ---- 内存缓存 ----
_EVENTS_CACHE: pd.DataFrame = None              # 长表(symbol, avail_date, T_eps_growth, R_rev_growth)
_WIDE_CACHE: dict = {}                          # 列名 -> 以 avail_date 为索引的宽表
_MONTHLY_CACHE: dict = {}                       # 月度列名 -> 以 avail_date 为索引的宽表


def _num(v):
  """JSON 数值 -> float; 缺失/非法 -> NaN."""
  if v is None:
    return np.nan
  try:
    return float(v)
  except (TypeError, ValueError):
    return np.nan


def _rel(a, b) -> float:
  """(a - b) / |b|: 相对修正幅度; 任一缺失或 b=0 -> NaN."""
  if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
    return np.nan
  return (a - b) / abs(b)


def _over(a, b) -> float:
  """a / |b|; 任一缺失或 b=0 -> NaN."""
  if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
    return np.nan
  return a / abs(b)


def _breadth(up, dn) -> float:
  """(up - dn)/(up + dn): 净上调广度; 两者皆缺失 -> NaN, 分母 0 -> 0."""
  fu, fd = np.isfinite(up), np.isfinite(dn)
  if not fu and not fd:
    return np.nan
  nu = up if fu else 0.0
  nd = dn if fd else 0.0
  t = nu + nd
  return (nu - nd) / t if t > 0 else 0.0


def _hist_rd(hist: dict) -> dict:
  """Earnings.History -> {报告期 date: reportDate}(公告日锚定)."""
  rd = {}
  for k, r in (hist or {}).items():
    if not isinstance(r, dict):
      continue
    d = r.get('date') or k
    rr = r.get('reportDate')
    if d and rr:
      rd[d] = rr
  return rd


def _symbol_trend(sym: str, raw: dict) -> list:
  """单个标的: Earnings.Trend -> 每期一条(avail_date = 该期 reportDate)."""
  e = raw.get('Earnings') or {}
  rd = _hist_rd(e.get('History'))
  rows = []
  for k, r in (e.get('Trend') or {}).items():
    if not isinstance(r, dict):
      continue
    if r.get('period') not in PERIODS:
      continue
    av = rd.get(r.get('date') or k)              # 无锚定公告日(未来指引) -> 跳过
    if not av:
      continue
    v0 = _num(r.get('epsTrendCurrent'))
    v7 = _num(r.get('epsTrend7daysAgo'))
    v30 = _num(r.get('epsTrend30daysAgo'))
    v60 = _num(r.get('epsTrend60daysAgo'))
    v90 = _num(r.get('epsTrend90daysAgo'))
    up30 = _num(r.get('epsRevisionsUpLast30days'))
    dn30 = _num(r.get('epsRevisionsDownLast30days'))
    avg = _num(r.get('earningsEstimateAvg'))
    high = _num(r.get('earningsEstimateHigh'))
    low = _num(r.get('earningsEstimateLow'))
    n_an = _num(r.get('earningsEstimateNumberOfAnalysts'))
    rows.append({
        'symbol': sym, 'avail_date': pd.to_datetime(av, errors='coerce'),
        # --- 原始原料(事件长表列) ---
        'T_eps_growth': _num(r.get('growth')),
        'R_rev_growth': _num(r.get('revenueEstimateGrowth')),
        'n_analysts': n_an, 'eps_avg': avg, 'eps_high': high, 'eps_low': low,
        'v0': v0, 'v7': v7, 'v30': v30, 'v60': v60, 'v90': v90,
        'up30': up30, 'down30': dn30,
        # --- 派生因子(比率型原值, 不做截面处理) ---
        'T_eps_rev_1w': _rel(v0, v7),
        'T_eps_rev_1m': _rel(v0, v30),
        'T_eps_rev_3m': _rel(v0, v90),
        'T_eps_rev_accel': _over((v0 - v30) - (v30 - v60), v60),
        'T_rev_breadth': _breadth(up30, dn30),
        'T_eps_disp': _over(high - low, avg),
        'T_coverage': n_an,
    })
  return rows


def _symbol_trend_monthly(sym: str, raw: dict) -> list:
  """
  单个标的: 把每条 Trend 快照内含的 5 个**冻结**共识点(A/A-30/A-60/A-90)拆成
  3 个间隔 30 天的月度观测(②降低静态性):
    asof = A       : T_eps_rev_m = (v0  - v30)/|v30|
    asof = A - 30d : T_eps_rev_m = (v30 - v60)/|v60|
    asof = A - 60d : T_eps_rev_m = (v60 - v90)/|v90|
  这些值均为"当时可观测的历史共识"(PIT 重建), 非前视; 仅保留 finite 观测.
  """
  e = raw.get('Earnings') or {}
  rd = _hist_rd(e.get('History'))
  rows = []
  for k, r in (e.get('Trend') or {}).items():
    if not isinstance(r, dict) or r.get('period') not in PERIODS:
      continue
    av = rd.get(r.get('date') or k)
    if not av:
      continue
    a0 = pd.to_datetime(av, errors='coerce')
    if pd.isna(a0):
      continue
    v0 = _num(r.get('epsTrendCurrent'))
    v30 = _num(r.get('epsTrend30daysAgo'))
    v60 = _num(r.get('epsTrend60daysAgo'))
    v90 = _num(r.get('epsTrend90daysAgo'))
    for back, cur, base in ((0, v0, v30), (30, v30, v60), (60, v60, v90)):
      rev = _rel(cur, base)
      if np.isfinite(rev):
        rows.append({'symbol': sym,
                     'avail_date': a0 - pd.Timedelta(days=back),
                     'T_eps_rev_m': rev})
  return rows


def _signature() -> tuple:
  """JSON 目录签名(文件数 + 最大 mtime + 结构版本), 用于判断缓存是否失效."""
  files = sorted(FUND_JSON_DIR.glob('*.json'))
  mtime = max((p.stat().st_mtime for p in files), default=0.0)
  return (len(files), mtime, _SCHEMA)


def build_events(force: bool = False) -> pd.DataFrame:
  """
  扫描 fundamentals JSON 的 Earnings.Trend -> PIT 预期修正长表
  (symbol, avail_date, T_eps_growth, R_rev_growth). 带磁盘缓存.

  :param force: True 强制重建
  :returns: 长表(按 symbol, avail_date 排序, 同 (symbol,avail_date) 去重取最后)
  """
  global _EVENTS_CACHE
  if _EVENTS_CACHE is not None and not force:
    return _EVENTS_CACHE

  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  sig = _signature()
  if not force and EVENTS_PKL.exists():
    try:
      cached = pd.read_pickle(EVENTS_PKL)
      if isinstance(cached, pd.DataFrame) and cached.attrs.get('sig') == sig:
        _EVENTS_CACHE = cached
        return cached
    except Exception:
      pass

  recs = []
  for p in sorted(FUND_JSON_DIR.glob('*.json')):
    try:
      raw = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
      continue
    recs.extend(_symbol_trend(p.stem, raw))

  ev = pd.DataFrame(recs)
  if ev.empty:
    raise RuntimeError(f'未解析出任何 Earnings.Trend 事件: {FUND_JSON_DIR}')
  ev = ev.dropna(subset=['avail_date'])
  ev = (ev.sort_values(['symbol', 'avail_date'])
          .drop_duplicates(['symbol', 'avail_date'], keep='last')
          .reset_index(drop=True))
  ev.attrs['sig'] = sig
  ev.to_pickle(EVENTS_PKL)
  _EVENTS_CACHE = ev
  return ev


def _trend_wide(col: str) -> pd.DataFrame:
  """长表某列 -> (avail_date × symbol) 宽表(带内存缓存)."""
  if col in _WIDE_CACHE:
    return _WIDE_CACHE[col]
  ev = build_events()
  w = (ev.pivot_table(index='avail_date', columns='symbol', values=col,
                      aggfunc='last', dropna=False)
         .sort_index())
  _WIDE_CACHE[col] = w
  return w


def _ewide(ds: Dataset, col: str) -> pd.DataFrame:
  """某列对齐到 ds 交易日网格: 每个交易日 t 取"<= t 最近一次快照"的值(ffill)."""
  w = _trend_wide(col)
  idx = w.index.union(ds.dates)
  aligned = w.reindex(idx).ffill().reindex(ds.dates)
  return aligned.reindex(columns=ds.symbols)


def build_monthly(force: bool = False) -> pd.DataFrame:
  """
  月度共识修正长表(symbol, avail_date, T_eps_rev_m, T_eps_rev_m3). 带磁盘缓存.
  在事件快照基础上每条拆出 3 个月度观测, 去重后按 symbol 计算"近 3 月度修正之和".

  :param force: True 强制重建
  :returns: 长表(按 symbol, avail_date 排序; 同 (symbol,avail_date) 去重取最后)
  """
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  sig = _signature() + ('m',)
  if not force and MONTHLY_PKL.exists():
    try:
      cached = pd.read_pickle(MONTHLY_PKL)
      if isinstance(cached, pd.DataFrame) and cached.attrs.get('sig') == sig:
        return cached
    except Exception:
      pass

  recs = []
  for p in sorted(FUND_JSON_DIR.glob('*.json')):
    try:
      raw = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
      continue
    recs.extend(_symbol_trend_monthly(p.stem, raw))

  mo = pd.DataFrame(recs)
  if mo.empty:
    raise RuntimeError(f'未解析出任何月度共识修正事件: {FUND_JSON_DIR}')
  mo = (mo.sort_values(['symbol', 'avail_date'])
          .drop_duplicates(['symbol', 'avail_date'], keep='last')
          .reset_index(drop=True))
  mo['T_eps_rev_m3'] = (mo.groupby('symbol')['T_eps_rev_m']
                          .rolling(3, min_periods=2).sum()
                          .reset_index(level=0, drop=True))
  mo = mo.sort_values(['symbol', 'avail_date']).reset_index(drop=True)
  mo.attrs['sig'] = sig
  mo.to_pickle(MONTHLY_PKL)
  return mo


def _monthly_wide(col: str) -> pd.DataFrame:
  """月度长表某列 -> (avail_date × symbol) 宽表(带内存缓存)."""
  if col in _MONTHLY_CACHE:
    return _MONTHLY_CACHE[col]
  mo = build_monthly()
  w = (mo.pivot_table(index='avail_date', columns='symbol', values=col,
                      aggfunc='last', dropna=False)
         .sort_index())
  _MONTHLY_CACHE[col] = w
  return w


def _mewide(ds: Dataset, col: str) -> pd.DataFrame:
  """月度某列对齐到 ds 交易日网格: 每个交易日 t 取"<= t 最近一次观测"(ffill)."""
  w = _monthly_wide(col)
  idx = w.index.union(ds.dates)
  return w.reindex(idx).ffill().reindex(ds.dates).reindex(columns=ds.symbols)


# ---- 因子注册 ----
@_register('T_eps_growth', 'analyst_trend',
               'analyst expected EPS growth: Earnings.Trend.growth (0q/0y, PIT)', window=1)
def _f_t_eps_growth(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'T_eps_growth')


@_register('R_rev_growth', 'analyst_trend',
               'analyst expected revenue growth: Earnings.Trend.revenueEstimateGrowth (0q/0y, PIT)',
               window=1)
def _f_r_rev_growth(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'R_rev_growth')


@_register('T_eps_rev_1w', 'analyst_trend',
               'EPS estimate revision last 1w: (cur-7d)/|7d| (0q/0y, PIT)', window=1)
def _f_t_eps_rev_1w(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'T_eps_rev_1w')


@_register('T_eps_rev_1m', 'analyst_trend',
               'EPS estimate revision last 1m: (cur-30d)/|30d| (0q/0y, PIT)', window=1)
def _f_t_eps_rev_1m(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'T_eps_rev_1m')


@_register('T_eps_rev_3m', 'analyst_trend',
               'EPS estimate revision last 3m: (cur-90d)/|90d| (0q/0y, PIT)', window=1)
def _f_t_eps_rev_3m(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'T_eps_rev_3m')


@_register('T_eps_rev_accel', 'analyst_trend',
               'EPS revision acceleration: ((cur-30d)-(30d-60d))/|60d| (0q/0y, PIT)', window=1)
def _f_t_eps_rev_accel(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'T_eps_rev_accel')


@_register('T_rev_breadth', 'analyst_trend',
               'net upward revision breadth last 30d: (up-dn)/(up+dn) (0q/0y, PIT)', window=1)
def _f_t_rev_breadth(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'T_rev_breadth')


@_register('T_eps_disp', 'analyst_trend',
               'EPS estimate dispersion: (high-low)/|avg| (0q/0y, PIT)',
               window=1, direction=-1)
def _f_t_eps_disp(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'T_eps_disp')


@_register('T_coverage', 'analyst_trend',
               'analyst coverage: earningsEstimateNumberOfAnalysts (0q/0y, PIT)', window=1)
def _f_t_coverage(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'T_coverage')


@_register('T_eps_rev_m', 'analyst_trend',
               'monthly EPS revision last 30d (from frozen consensus chain, PIT)', window=1)
def _f_t_eps_rev_m(ds: Dataset) -> pd.DataFrame:
  return _mewide(ds, 'T_eps_rev_m')


@_register('T_eps_rev_m3', 'analyst_trend',
               'sum of last 3 monthly EPS revisions (~90d cumulative, monthly stepping, PIT)',
               window=1)
def _f_t_eps_rev_m3(ds: Dataset) -> pd.DataFrame:
  return _mewide(ds, 'T_eps_rev_m3')


def main():
  import argparse

  ap = argparse.ArgumentParser(description='trend 因子层自检')
  ap.add_argument('--pool', default='us_500')
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  a = ap.parse_args()

  log = cfg.get_logger('factor.trend')
  ev = build_events()
  log.info(f'[事件] {len(ev):,} 条预期修正快照, {ev["symbol"].nunique()} 只, '
           f'{ev["avail_date"].min():%Y-%m-%d} ~ {ev["avail_date"].max():%Y-%m-%d}')

  ds = load_pool(a.pool, a.interval)
  log.info(f'[池] {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日')
  for name in TREND_COLS:
    df = _ewide(ds, name)
    cov = float(df.notna().to_numpy().mean())
    log.info(f'[{name}] 覆盖 {cov:.4f}, 最近一日有效 {int(df.iloc[-1].notna().sum())} 只')

  mo = build_monthly()
  log.info(f'[月度] {len(mo):,} 条月度共识修正观测, {mo["symbol"].nunique()} 只, '
           f'{mo["avail_date"].min():%Y-%m-%d} ~ {mo["avail_date"].max():%Y-%m-%d}')
  for name in MONTHLY_COLS:
    df = _mewide(ds, name)
    cov = float(df.notna().to_numpy().mean())
    log.info(f'[{name}] 覆盖 {cov:.4f}, 最近一日有效 {int(df.iloc[-1].notna().sum())} 只')


if __name__ == '__main__':
  main()
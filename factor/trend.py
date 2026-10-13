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

因子:
  T_eps_growth = growth                预期 EPS 增速(顶层 growth 字段)
  R_rev_growth = revenueEstimateGrowth 预期营收增速
  —— 均为原始值, 不做截面处理(去极值/标准化/排名交给 prepare.py).

因果性:
  事件只读 <= t 已公告的快照; 对齐到交易日网格后 ffill 严格向后看,
  与喂给 Dataset 的长度无关(截断一致性成立) -> 通过 verify_causal.

产物:
  * 事件长表缓存: output/trend/trend_events.pkl(带 JSON 目录签名, 命中即复用);
  * 因子: T_eps_growth / R_rev_growth (group='analyst_trend').

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
PERIODS = ('0q', '0y')                          # 仅保留"当季/当年"快照(完整季度覆盖)
TREND_COLS = ('T_eps_growth', 'R_rev_growth')   # 本层注册的因子列
_SCHEMA = 1                                     # 缓存结构版本: 口径变更时 +1, 自动失效旧缓存

# ---- 内存缓存 ----
_EVENTS_CACHE: pd.DataFrame = None              # 长表(symbol, avail_date, T_eps_growth, R_rev_growth)
_WIDE_CACHE: dict = {}                          # 列名 -> 以 avail_date 为索引的宽表


def _num(v):
  """JSON 数值 -> float; 缺失/非法 -> NaN."""
  if v is None:
    return np.nan
  try:
    return float(v)
  except (TypeError, ValueError):
    return np.nan


def _symbol_trend(sym: str, raw: dict) -> list:
  """单个标的: Earnings.Trend -> 每期一条(avail_date = 该期 reportDate)."""
  e = raw.get('Earnings') or {}
  hist = e.get('History') or {}
  rd = {}
  for k, r in hist.items():
    if not isinstance(r, dict):
      continue
    d = r.get('date') or k
    rr = r.get('reportDate')
    if d and rr:
      rd[d] = rr
  rows = []
  for k, r in (e.get('Trend') or {}).items():
    if not isinstance(r, dict):
      continue
    if r.get('period') not in PERIODS:
      continue
    av = rd.get(r.get('date') or k)              # 无锚定公告日(未来指引) -> 跳过
    if not av:
      continue
    rows.append({'symbol': sym, 'avail_date': pd.to_datetime(av, errors='coerce'),
                 'T_eps_growth': _num(r.get('growth')),
                 'R_rev_growth': _num(r.get('revenueEstimateGrowth'))})
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


if __name__ == '__main__':
  main()
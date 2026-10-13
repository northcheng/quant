# -*- coding: utf-8 -*-
"""
factor.pead — 财报公告后漂移(PEAD)因子层(自包含)
================================================
职责: 把 newdata/earnings_calendar.pkl(~ /quant/newdata/earnings_calendar.pkl) 的
      分析师盈余意外转成 **PIT(Point-in-Time, 时点)可比**的 (date × symbol) 宽表, 注册为因子.

为什么是 PEAD(Post-Earnings Announcement Drift, 财报公告后漂移):
  盈余意外(actual vs analyst estimate)公布后, 股价往往在随后数周~数月内沿意外方向**持续漂移**.
  故信号 = 盈余意外, 并在公告后保持 L 个交易日(ffill limit=L).

数据口径(earnings_calendar.pkl, 列 code/report_date/date/.../actual/estimate):
  * symbol = code 去掉 '.US' 后缀;
  * 盈余意外 surprise = (actual - estimate) / |estimate|, 裁剪到 [-1, 1];
  * PIT: 以**公告日 report_date** 为可用日; 映射到">= 公告日的最近交易日"(非交易日顺延);
          公告日晚于数据末端的事件直接丢弃(避免用未来信息污染截断检验).
  * 漂移: 因子的公告信号在该交易日后 keep L 个交易日内保持(ffill limit=L).

因果性:
  事件只读 <= t 已公告的记录; 映射到交易日网格后 ffill(limit=L) 严格向后看,
  与喂给 Dataset 的长度无关(截断一致性成立) -> 通过 verify_causal.

产物:
  * 事件长表缓存: output/pead/earn_surprise.pkl(带源文件签名, 命中即复用);
  * 因子: PEAD_20 / PEAD_60 (group='pead', L=20/60).

自检:
  cd ~/git && python -m quant.factor.pead --pool us_500
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import factor as _fct
from quant.factor.data import Dataset, load_pool

# 以 `python -m quant.factor.pead` 直接运行时, factor.py 底部导入会先生成 canonical
# 副本(quant.factor.pead)并完成注册, 再轮到本 __main__ 副本执行装饰器 -> 会重复注册.
# 此处令 __main__ 副本复用 no-op 装饰器, 只跑自检, 不再注册.
_RERUN = __name__ == '__main__' and 'quant.factor.pead' in sys.modules
_register = _fct.register if not _RERUN else (lambda *a, **k: (lambda fn: fn))

EARN_PKL = cfg.HOME / 'quant' / 'newdata' / 'earnings_calendar.pkl'
CACHE_DIR = cfg.OUT_DIR / 'pead'
EVENTS_PKL = CACHE_DIR / 'earn_surprise.pkl'
SURP_CLIP = 1.0                                 # 盈余意外上下限(避免极端值主导截面)
SCALES = (20, 60)                               # PEAD 漂移窗口(交易日): PEAD_20 / PEAD_60
_SCHEMA = 1                                     # 缓存结构版本: 口径变更时 +1, 自动失效旧缓存

# ---- 内存缓存 ----
_EVENTS_CACHE: pd.DataFrame = None              # 事件长表(symbol, report_date, surprise)
_BASE_CACHE: dict = {}                          # grid 指纹 -> (announce_day × symbol) 宽表


def _signature() -> tuple:
  """源文件签名(文件大小 + mtime + 结构版本), 用于判断缓存是否失效."""
  if not EARN_PKL.exists():
    return (0, 0.0, _SCHEMA)
  st = EARN_PKL.stat()
  return (st.st_size, st.st_mtime, _SCHEMA)


def build_events(force: bool = False) -> pd.DataFrame:
  """
  读取 earnings_calendar.pkl -> PIT 盈余意外长表(symbol, report_date, surprise). 带磁盘缓存.

  :param force: True 强制重建
  :returns: 长表(按 report_date 排序)
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

  if not EARN_PKL.exists():
    raise RuntimeError(f'未找到盈余日历: {EARN_PKL}')
  d = pd.read_pickle(EARN_PKL)
  d = d[['code', 'report_date', 'actual', 'estimate']].copy()
  d['symbol'] = d['code'].astype(str).str.replace(r'\.US$', '', regex=True)
  d['report_date'] = pd.to_datetime(d['report_date'], errors='coerce')
  d = d.dropna(subset=['report_date', 'actual', 'estimate'])
  d = d[d['estimate'].abs() > 1e-9]
  d['surprise'] = ((d['actual'] - d['estimate']) / d['estimate'].abs()).clip(-SURP_CLIP, SURP_CLIP)
  ev = (d[['symbol', 'report_date', 'surprise']]
        .sort_values(['symbol', 'report_date'])
        .drop_duplicates(['symbol', 'report_date'], keep='last')
        .reset_index(drop=True))
  if ev.empty:
    raise RuntimeError(f'未解析出任何盈余意外事件: {EARN_PKL}')
  ev.attrs['sig'] = sig
  ev.to_pickle(EVENTS_PKL)
  _EVENTS_CACHE = ev
  return ev


def _map_to_grid(report_dates: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
  """
  公告日 -> ">= 公告日的最近交易日"(非交易日顺延到下一交易日).
  公告日晚于网格末端(未来事件) -> NaT(丢弃), 保证截断一致性.
  """
  g = grid.values
  pos = np.searchsorted(g, pd.DatetimeIndex(report_dates).values, side='left')
  out = np.full(len(pos), np.datetime64('NaT'), dtype='datetime64[ns]')
  ok = pos < len(g)
  out[ok] = g[pos[ok]]
  return pd.Series(out, index=report_dates.index)


def _base_on_grid(grid: pd.DatetimeIndex) -> pd.DataFrame:
  """(公告交易日 × symbol) 的盈余意外宽表(带内存缓存, 按网格指纹复用)."""
  key = (len(grid), grid[0].value, grid[-1].value) if len(grid) else (0, 0, 0)
  if key in _BASE_CACHE:
    return _BASE_CACHE[key]
  ev = build_events()
  ev = ev.assign(_grid=_map_to_grid(ev['report_date'], grid)).dropna(subset=['_grid'])
  base = (ev.groupby(['_grid', 'symbol'])['surprise'].last().unstack()
            .reindex(index=grid))
  _BASE_CACHE.clear()                          # 只保留最近一次网格(评估通常单一网格)
  _BASE_CACHE[key] = base
  return base


def _pead_wide(ds: Dataset, L: int) -> pd.DataFrame:
  """PEAD 因子: 盈余意外映射到交易日网格后, 向后保持 L 个交易日(ffill limit=L)."""
  base = _base_on_grid(pd.DatetimeIndex(ds.dates))
  w = base.reindex(columns=ds.symbols).ffill(limit=L)
  return w


# ---- 因子注册 ----
@_register('PEAD_20', 'pead',
               'post-earnings drift: (actual-estimate)/|estimate|, held L=20 trading days',
               window=20)
def _f_pead20(ds: Dataset) -> pd.DataFrame:
  return _pead_wide(ds, 20)


@_register('PEAD_60', 'pead',
               'post-earnings drift: (actual-estimate)/|estimate|, held L=60 trading days',
               window=60)
def _f_pead60(ds: Dataset) -> pd.DataFrame:
  return _pead_wide(ds, 60)


def main():
  ap = argparse.ArgumentParser(description='pead 因子层自检')
  ap.add_argument('--pool', default='us_500')
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  a = ap.parse_args()

  log = cfg.get_logger('factor.pead')
  ev = build_events()
  log.info(f'[事件] {len(ev):,} 条盈余意外, {ev["symbol"].nunique()} 只, '
           f'{ev["report_date"].min():%Y-%m-%d} ~ {ev["report_date"].max():%Y-%m-%d}')

  ds = load_pool(a.pool, a.interval)
  log.info(f'[池] {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日')
  for name in ('PEAD_20', 'PEAD_60'):
    spec = _fct.get_factor(name) if not _RERUN else None
    df = _pead_wide(ds, 20 if name == 'PEAD_20' else 60)
    cov = float(df.notna().to_numpy().mean())
    log.info(f'[{name}] 覆盖 {cov:.4f}, 最近一日有效 {int(df.iloc[-1].notna().sum())} 只')


if __name__ == '__main__':
  main()
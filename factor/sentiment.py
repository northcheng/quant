# -*- coding: utf-8 -*-
"""
factor.sentiment — 情绪因子层(自包含)
=====================================
职责: 把 EODHD /sentiments 回填的新闻情绪序列
      (~ /quant/sentiment_data/us_500_sentiment.pkl) 转成 **PIT(Point-in-Time, 时点)**
      因果的 (date × symbol) 宽表, 并注册为因子.

定位: 作为**风控 / 中性化控制变量**并入因子库 —— 与价量对照(CONTROLS)一起, 用于
      检验其它因子(尤其 F_ 基本面族)的 alpha 是否只是"新闻情绪的代理".
      本身**不作为独立 alpha 来源**: 前序评估(见 lab_sent_eval_us_500.csv)显示
      2022+ 全窗 |nw_t| < 1.3, 无稳健独立 alpha; S_news_attn 更是静态大盘暴露
      (sig_ac1 ≈ 0.996, 换手极低)的伪信号. 故在此只以"控制变量"身份入库.

数据: ~/quant/sentiment_data/us_500_sentiment.pkl
      {'long': DataFrame[symbol, date, count, normalized], 'meta': {...}}
      * count      = 当日该股新闻条数;
      * normalized = EODHD 归一化情绪分, 一般落在 [-1, 1](正 = 看涨).
      ⚠ 覆盖: 数据自 2022-01 起, 2022 之前一律暖机 NaN; 池内 501 只.

因子(全部 PIT 因果: 只读 <= t 的新闻):
  S_sent_lvl   情绪水平   = 近 21 日 normalized 均值                    dir +1
  S_news_attn  新闻关注度 = log1p(近 21 日 count 之和)                  dir -1
                (关注度 dir = -1: 高关注度常伴随过度反应 -> 未来收益偏弱,
                 与"低关注度溢价"的经验一致)

对齐: 情绪是**日历日**(含周末/节假日)事件序列. 先在日历日网格上做时间窗滚动
      (rolling('21D'), 左闭右开, 严格只看 <= t), 再 reindex 到 ds 的交易日网格与标的.
      -> 值只依赖 <= t 的信息, 与喂给 ds 的数据长度无关(截断一致性成立), 无前视.

产物: output/sentiment/sent_wide.pkl(情绪宽表缓存, 带文件签名, 命中即复用).

自检:
  cd ~/git && python -m quant.factor.sentiment
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import factor as _fct
from quant.factor.data import Dataset

# 以 `python -m quant.factor.sentiment` 直接运行时, factor.py 底部导入会先生成
# canonical 副本(quant.factor.sentiment)并完成注册, 再轮到本 __main__ 副本执行装饰器
# -> 会重复注册. 此处令 __main__ 副本复用 no-op 装饰器, 只跑自检, 不再注册.
_RERUN = __name__ == '__main__' and 'quant.factor.sentiment' in sys.modules
_register = _fct.register if not _RERUN else (lambda *a, **k: (lambda fn: fn))

SENT_DIR = cfg.HOME / 'quant' / 'sentiment_data'
SENT_PKL = SENT_DIR / 'us_500_sentiment.pkl'
CACHE_DIR = cfg.OUT_DIR / 'sentiment'
WIDE_PKL = CACHE_DIR / 'sent_wide.pkl'
LOOKBACK_DAYS = 21                              # 情绪回看窗('21D' 自然日, 约 1 个月)
MIN_PERIODS = 5                                 # 滚动均值最小样本数(不满不成值)
_SCHEMA = 1                                     # 缓存结构版本: 口径变更时 +1, 自动失效旧缓存

# 注册的因子名(与其他因子区分; S_ 前缀 = Sentiment)
SENT_FACTORS = ['S_sent_lvl', 'S_news_attn']

# ---- 内存缓存 ----
_LONG_CACHE: pd.DataFrame = None
_WIDE_CACHE: dict = None


def _signature() -> tuple:
  """情绪 pkl 文件签名(大小 + mtime + 结构版本), 用于判断缓存是否失效."""
  if not SENT_PKL.exists():
    return (0, 0.0, _SCHEMA)
  st = SENT_PKL.stat()
  return (st.st_size, st.st_mtime, _SCHEMA)


def build_long(force: bool = False) -> pd.DataFrame:
  """
  读情绪 pkl -> 长表(symbol, date, count, normalized).

  :param force: True 强制重读
  :returns: DataFrame(已 drop 掉 date 为空的记录)
  :raises FileNotFoundError: 情绪数据缺失
  """
  global _LONG_CACHE
  if _LONG_CACHE is not None and not force:
    return _LONG_CACHE
  if not SENT_PKL.exists():
    raise FileNotFoundError(f'情绪数据缺失: {SENT_PKL}')
  raw = pd.read_pickle(SENT_PKL)
  long = raw['long'] if isinstance(raw, dict) and 'long' in raw else raw
  long = long.dropna(subset=['date']).copy()
  long['date'] = pd.to_datetime(long['date'])
  _LONG_CACHE = long.reset_index(drop=True)
  return _LONG_CACHE


def build_wides(force: bool = False) -> dict:
  """
  长表 -> 情绪因子宽表(以**日历日**为索引, 时间窗滚动, 严格 PIT). 带磁盘缓存.

  :param force: True 强制重建
  :returns: {'S_sent_lvl': 宽表, 'S_news_attn': 宽表, '__sig': tuple}
  """
  global _WIDE_CACHE
  if _WIDE_CACHE is not None and not force:
    return _WIDE_CACHE

  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  sig = _signature()
  if not force and WIDE_PKL.exists():
    try:
      cached = pd.read_pickle(WIDE_PKL)
      if isinstance(cached, dict) and cached.get('__sig') == sig:
        _WIDE_CACHE = cached
        return cached
    except Exception:
      pass

  long = build_long(force)
  norm = long.pivot(index='date', columns='symbol', values='normalized')
  cnt = long.pivot(index='date', columns='symbol', values='count')

  # 补全日历日网格: 无新闻日 = NaN(pandas 时间窗 rolling 自动跳过), 保证滚动窗按"天"推进.
  daily = pd.date_range(norm.index.min(), norm.index.max(), freq='D')
  cols = norm.columns
  norm = norm.reindex(daily).reindex(columns=cols)
  cnt = cnt.reindex(daily).reindex(columns=cols)

  win = f'{LOOKBACK_DAYS}D'
  lvl = norm.rolling(win, min_periods=MIN_PERIODS).mean()          # 近 21 日情绪均值
  attn = np.log1p(cnt.rolling(win, min_periods=1).sum()).fillna(0.0)  # 近 21 日新闻条数

  out = {'S_sent_lvl': lvl, 'S_news_attn': attn, '__sig': sig}
  pd.to_pickle(out, WIDE_PKL)
  _WIDE_CACHE = out
  return out


def _swide(ds: Dataset, name: str) -> pd.DataFrame:
  """
  情绪某因子对齐到 ds 交易日网格/标的.

  纯行(reindex 到 ds.dates)/列(reindex 到 ds.symbols)子集, 不改变任何值 ->
  因滚动窗只看历史, 值只依赖 <= t, 截断一致性成立. 2022 之前 / 池外标的 -> NaN.
  """
  w = build_wides()[name]
  return w.reindex(ds.dates).reindex(columns=ds.symbols)


# ============================ 注册因子 ============================
# 命名前缀 S_(sentiment). group = sentiment. 以"控制变量"身份入库.

@_register('S_sent_lvl', 'sentiment',
               'mean(新闻 normalized, 近 21 日)', window=LOOKBACK_DAYS, direction=1)
def _s_sent_lvl(ds: Dataset) -> pd.DataFrame:
  return _swide(ds, 'S_sent_lvl')


@_register('S_news_attn', 'sentiment',
               'log1p(sum(新闻 count, 近 21 日))', window=LOOKBACK_DAYS, direction=-1)
def _s_news_attn(ds: Dataset) -> pd.DataFrame:
  return _swide(ds, 'S_news_attn')


# ============================ 自检 ============================
def main():
  log = cfg.get_logger('factor.sentiment')
  long = build_long()
  log.info(f'[情绪长表] {len(long):,} 行, {long["symbol"].nunique()} 标的, '
           f'{long["date"].min():%Y-%m-%d} ~ {long["date"].max():%Y-%m-%d}')
  w = build_wides()
  for n in SENT_FACTORS:
    df = w[n]
    log.info(f'[{n}] 宽表 {df.shape} (日历日 × 标的), '
             f'非空占比={df.notna().mean().mean():.1%}')
  for s in ['AAPL', 'MSFT', 'NVDA']:
    cols = [c for c in w['S_sent_lvl'].columns if c == s]
    if cols:
      sub = pd.DataFrame({n: w[n][s].dropna().tail(3) for n in SENT_FACTORS})
      log.info(f'\n[{s}] 最近 3 个有值日历日:\n' + sub.to_string())


if __name__ == '__main__':
  main()
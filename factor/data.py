# -*- coding: utf-8 -*-
"""
factor.data — 数据层(自包含)
=============================
职责: 把 ~/quant 下的研究 pkl 读成统一的内存结构 Dataset, 供上层(因子/评估/回测)使用.

pkl 结构:  {f'{symbol}_{interval}': DataFrame}, 每个 DataFrame 是某个标的的 TA 面板,
           index 为交易日(DatetimeIndex), 列为 OHLCV + 各类技术指标(200+ 列).

本模块自己做三件事(不依赖任何 bc_* 脚本):
  1. load_panel  : dict -> MultiIndex(symbol, date) 长表(过滤空 df / object 列数值化)
  2. get_w       : 长表 -> (date × symbol) 宽表
  3. Dataset     : 统一封装 + 窗口截取(防 warmup 污染) + 缓存

Dataset 提供两种视角(这是后面所有步骤的数据基础):
  - panel : MultiIndex(symbol, date) 长表  -- 少数场景(按标的取时间序列)
  - wide  : (date × symbol) 宽表           -- 主战场(列=标的, 行=交易日)

  因子层输出宽表、组合层加权宽表、回测层按行(交易日)推进 -- 全部围绕宽表.
  宽表的一个必然产物是 NaN: 某标的某日停牌/未上市 -> 该格 NaN; 上层按"截面"处理时
  用 NaN 屏蔽该标的, 而不是填 0(填 0 会伪造出一个"中性值"混进排序).

自检(第0步验证入口):
  cd ~/git && python -m quant.factor.data --pool etf_3x
"""
import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from quant.factor import config as cfg

# 统一小写列名 -> pkl 里的真实列名(宽表访问用; 其余列按原样直查)
OHLCV_COLS = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}


def load_panel(pkl_path: str, interval: str = 'day') -> pd.DataFrame:
  """
  读研究 pkl -> MultiIndex(symbol, date) 长表.

  步骤: 逐标的取出 DataFrame -> 丢弃空 df -> object 列数值化 -> 打上 symbol 列 ->
  纵向 concat -> 把 symbol 并入索引并让 symbol 成为最外层 -> 排序.

  :param pkl_path: pkl 路径
  :param interval: 频率, 用于从键名 {symbol}_{interval} 里剥出 symbol
  :returns: DataFrame, index=MultiIndex['symbol','date'], 列=该标的全部 TA 列
  :raises FileNotFoundError: pkl 不存在
  :raises ValueError: pkl 内容不是 dict
  """
  with open(pkl_path, 'rb') as f:
    raw = pickle.load(f)
  if not isinstance(raw, dict):
    raise ValueError(f'pkl 内容不是 dict: {type(raw)}')

  tail = f'_{interval}'
  frames = []
  for key, df in raw.items():
    if df is None or len(df) == 0:                 # 个别标的可能没数据
      continue
    symbol = key[:-len(tail)] if key.endswith(tail) else key
    tmp = df.copy()
    for c in cfg.OBJECT_COLS:                      # 这几列在 pkl 里是 object, 需转数值
      if c in tmp.columns:
        tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
    tmp['symbol'] = symbol
    frames.append(tmp)

  if not frames:
    raise ValueError(f'pkl 内无有效数据: {pkl_path}')

  panel = (pd.concat(frames)
             .set_index('symbol', append=True)
             .swaplevel(0, 1)                      # (date, symbol) -> (symbol, date)
             .sort_index())
  panel.index.names = ['symbol', 'date']
  return panel


def get_w(panel: pd.DataFrame, col: str) -> pd.DataFrame:
  """
  长表取一列 -> (date × symbol) 宽表.

  :param panel: MultiIndex(symbol, date) 长表
  :param col: 列名(如 'Close')
  :returns: DataFrame, index=date, columns=symbol
  """
  return panel[col].unstack('symbol').sort_index()


@dataclass
class Dataset:
  """一个池的面板数据. slice() 返回新对象, 不改动自身(避免隐式副作用)."""

  pool: str
  interval: str
  path: str
  panel: pd.DataFrame                                   # MultiIndex(symbol, date)
  _wide_cache: dict = field(default_factory=dict, repr=False)

  # ---- 基本属性 ----
  @property
  def symbols(self) -> list:
    return sorted(self.panel.index.get_level_values('symbol').unique())

  @property
  def dates(self) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(self.panel.index.get_level_values('date').unique()))

  # ---- 宽表访问(核心) ----
  def wide(self, col: str) -> pd.DataFrame:
    """
    取任意列 -> (date × symbol) 宽表(带缓存). col 可写 'Close' 或 'close'.

    :param col: 列名(pkl 中的原始列名; OHLCV 五列可用小写)
    :returns: DataFrame, index=date, columns=symbol
    """
    real = OHLCV_COLS.get(col.lower(), col)
    if real not in self._wide_cache:
      self._wide_cache[real] = get_w(self.panel, real)
    return self._wide_cache[real]

  def ohlcv(self) -> dict:
    """一次取齐五个基础价量宽表: {'open'/'high'/'low'/'close'/'volume': 宽表}."""
    return {k: self.wide(v) for k, v in OHLCV_COLS.items()}

  # ---- 窗口截取 ----
  def slice(self, start=None, end=None) -> 'Dataset':
    """
    按交易日截取, 只截评估窗口(不重算指标, 故无 warmup 污染).

    :param start/end: 'YYYY-MM-DD' 或 Timestamp, None 表示不限
    :returns: 新的 Dataset
    """
    d = self.dates
    if start is not None:
      d = d[d >= pd.Timestamp(start)]
    if end is not None:
      d = d[d <= pd.Timestamp(end)]
    mask = self.panel.index.get_level_values('date').isin(d)
    return Dataset(self.pool, self.interval, self.path, self.panel.loc[mask])

  def summary(self) -> str:
    """多行概况字符串(自检/日志用)."""
    d = self.dates
    return (f'Dataset(pool={self.pool}, interval={self.interval})\n'
            f'  pkl    : {self.path}\n'
            f'  标的数 : {len(self.symbols)}\n'
            f'  交易日 : {len(d)} ({d.min():%Y-%m-%d} ~ {d.max():%Y-%m-%d})\n'
            f'  列数   : {self.panel.shape[1]}\n'
            f'  面板   : {self.panel.shape[0]:,} 行 (symbol × date)')

def load_pool(pool: str = cfg.DEFAULT_POOL, interval: str = cfg.DEFAULT_INTERVAL,
              pkl_path: str = None, start: str = None, end: str = None,
              exclude: list = None, research: bool = True) -> Dataset:
  """
  加载一个池的数据.

  :param pool: 池名(见 config.available_pools(); 如 etf_3x/company_300/hs300)
  :param interval: 频率('day'/'week'/'month')
  :param pkl_path: 显式 pkl 路径(默认按池名解析)
  :param start/end: 只保留该区间的交易日(评估窗口)
  :param exclude: 需剔除的标的列表(如 ['DJT'])
  :param research: True 取研究版全史 pkl; False 取生产版(不存在则回退研究版)
  :returns: Dataset
  """
  if pkl_path:
    path = Path(pkl_path)                         # pkl_path 允许传 str 或 Path
  else:
    path = cfg.pool_pkl(pool, interval, research=research)
    if not path.exists() and research:            # 研究版缺失则回退生产版
      path = cfg.pool_pkl(pool, interval, research=False)
  if not path.exists():
    raise SystemExit(f'pkl 不存在: {path}\n'
                     f'当前可用池: {cfg.available_pools(interval)}')
  return _load(str(path), pool, interval, start, end, exclude)

def _load(path: str, pool: str, interval: str, start, end, exclude) -> Dataset:
  panel = load_panel(path, interval)              # raw dict 不保留(内存翻倍且用不到)

  if exclude:
    have = set(panel.index.get_level_values('symbol'))
    drop = [s for s in exclude if s in have]
    if drop:
      panel = panel.drop(index=drop, level='symbol')

  ds = Dataset(pool=pool, interval=interval, path=path, panel=panel)
  if start is not None or end is not None:
    ds = ds.slice(start, end)
  return ds

def main():
  ap = argparse.ArgumentParser(description='factor 数据层自检: 加载池并打印概况')
  ap.add_argument('--pool', default=cfg.DEFAULT_POOL)
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  ap.add_argument('--pkl-path', default=None)
  ap.add_argument('--start', default=None)
  ap.add_argument('--end', default=None)
  a = ap.parse_args()

  log = cfg.get_logger('factor.data', prefix='factor')
  ds = load_pool(a.pool, a.interval, pkl_path=a.pkl_path, start=a.start, end=a.end)
  log.info('\n' + ds.summary())
  log.info(f'[pools]: 可用池 {cfg.available_pools(a.interval)}')

  close = ds.wide('close')
  log.info(f'[wide]: close 宽表 shape={close.shape} (date × symbol), '
           f'NaN 占比={close.isna().mean().mean():.1%}')
  log.info(f'[wide]: index.name={close.index.name!r}, columns.name={close.columns.name!r}')
  log.info(f'[wide]: 最后一个交易日截面(前5):\n{close.iloc[-1].dropna().head().to_string()}')
  log.info(f'[cols]: 面板可用列(前25): {list(ds.panel.columns)[:25]}')


if __name__ == '__main__':
  main()

# -*- coding: utf-8 -*-
"""
ml.data — 数据接入层(第0步)
============================
职责: 把 `quant.factor.data` 的数据能力接进来, 并补两个 ML 侧需要的小工具.

本层**不重写**任何读盘逻辑 —— pkl -> Dataset 的活全部由 factor.data 负责,
这里只做三件事:
  1. load(): 统一入口(带日志), 包一层 factor.data.load_pool;
     池名若命中 cfg.POOL_ALIASES, 则走 _load_merged: 逐成员池读入 -> 按 symbol 去重 -> 取并集;
  2. numeric_columns() / feature_candidates(): 列出可当特征用的数值列
     (面板里 200+ 的 TA 指标), 供第1步 features.py 选用;
  3. 自检入口: 打印池概况 + 可用特征列清单, 确认数据链路通.

为什么复用而不是重写: 两边必须是**同一份数据口径**(同一 pkl、同一 MultiIndex 结构、
同一 NaN 语义), 否则 ml 的结论无法与 factor 对照.

自检:
  cd ~/git && python -m quant.ml.data --pool etf_3x
  cd ~/git && python -m quant.ml.data --pool us_merged      (合并池别名, 应报 1077 标的)
"""
import argparse

import pandas as pd

from quant.factor.data import OHLCV_COLS, Dataset, load_pool

from quant.ml import config as cfg

# 基础价量列(pkl 里的真实列名) —— 通常已被 factor 的因子覆盖, 一般不直接当特征,
# 但列出来方便确认面板结构.
BASE_PRICE_COLS = list(OHLCV_COLS.values())          # ['Open','High','Low','Close','Volume']


def load(pool: str = 'etf_3x', interval: str = 'day', start: str = None,
         end: str = None, exclude: list = None, research: bool = True,
         log=None) -> Dataset:
  """
  加载一个池的数据(封装 factor.data.load_pool, 带日志).

  池名可以是 config.POOL_ALIASES 里的**合并池别名**(如 'us_merged'): 此时逐成员池读入、
  按 symbol 去重取并集, 返回合成 Dataset(池名 = 别名).

  :param pool: 池名(见 factor.config.available_pools()), 或合并池别名(见 cfg.POOL_ALIASES)
  :param interval: 频率('day'/'week'/'month')
  :param start/end: 只保留该区间的交易日
  :param exclude: 需剔除的标的
  :param research: True 取研究版全史 pkl
  :param log: 可选 logger; None 则不打印
  :returns: factor.data.Dataset
  """
  if pool in cfg.POOL_ALIASES:
    ds = _load_merged(pool, cfg.POOL_ALIASES[pool], interval, start=start, end=end,
                      exclude=exclude, research=research, log=log)
  else:
    ds = load_pool(pool, interval, start=start, end=end, exclude=exclude,
                   research=research)
  # 把"当前池"记到进程级: 窗口真源(cfg.window_range)据此取该池自己的 IS/OOS
  # (factor_config.json 的 meta), 而不是全局默认 —— 否则 ML 的样本外与因子研究/
  # 生产冻结权重可能不是同一套窗口.
  cfg.set_current_pool(ds.pool)
  if log is not None:
    log.info(f'[load] {ds.summary()}')
  return ds


def _load_merged(alias: str, members: tuple, interval: str = 'day', start: str = None,
                 end: str = None, exclude: list = None, research: bool = True,
                 log=None) -> Dataset:
  """
  合并多个池为一个合成 Dataset: 逐成员读面板 -> 按 symbol 去重(保留首现) -> concat.

  去重按 symbol(不是按行): 同一 symbol 若在多个池出现(如 company_300 ⊂ company_1000),
  只保留最先遇到的那份; 不同池若列集不一致, concat 会做列并集(缺失格填 NaN).
  特征/标签/拟合都将在该大宇宙上计算(截面 rank 在合并池内做), 保证训练分布自洽.

  :param alias: 合并池名(写进 Dataset.pool; 通常 = cfg.POOL_ALIASES 的键)
  :param members: 成员池名序列
  :param interval/start/end/exclude/research: 同 load()
  :param log: 可选 logger
  :returns: 合成 Dataset
  """
  parts, seen, paths = [], set(), []
  for m in members:
    d = load_pool(m, interval, research=research)
    panel = d.panel
    uniq = list(panel.index.get_level_values('symbol').unique())
    new = [s for s in uniq if s not in seen]
    if not new:
      if log is not None:
        log.info(f'[merge] {m}: {len(uniq)} 标的全部与已有池重叠 -> 跳过')
      continue
    if len(new) < len(uniq):
      panel = panel.loc[panel.index.get_level_values('symbol').isin(new)]
      if log is not None:
        log.info(f'[merge] {m}: {len(uniq)} 标的 -> 新增 {len(new)}, '
                 f'去重 {len(uniq) - len(new)}')
    elif log is not None:
      log.info(f'[merge] {m}: {len(uniq)} 标的(全部新增)')
    seen.update(new)
    parts.append(panel)
    paths.append(str(d.path))

  if not parts:
    raise SystemExit(f'合并池 {alias} 无可用数据: members={members}')
  panel = pd.concat(parts).sort_index()
  panel.index.names = ['symbol', 'date']

  ds = Dataset(pool=alias, interval=interval, path=' + '.join(paths), panel=panel)
  if exclude:
    have = set(panel.index.get_level_values('symbol'))
    drop = [s for s in exclude if s in have]
    if drop:
      ds = Dataset(alias, interval, ds.path, panel.drop(index=drop, level='symbol'))
  if start is not None or end is not None:
    ds = ds.slice(start, end)
  if log is not None:
    log.info(f'[merge] {alias} = {" ∪ ".join(members)} -> {len(ds.symbols)} 标的')
  return ds


def numeric_columns(ds: Dataset) -> list:
  """
  面板里可当特征用的**数值**列(排除基础价量与已知 object 列).

  判定: dtype 属于数值; 排除 OHLCV 五列(价格本身尺度随标的漂移, 应由因子表达);
  object 列(trend_score 等)在 factor.data.load_panel 里已数值化, 若转成功则保留.

  :param ds: Dataset
  :returns: 列名列表(保持面板原顺序)
  """
  cols = []
  for c in ds.panel.columns:
    if c in BASE_PRICE_COLS:
      continue
    if pd.api.types.is_numeric_dtype(ds.panel[c]):
      cols.append(c)
  return cols


def feature_candidates(ds: Dataset) -> pd.DataFrame:
  """
  可用特征列清单(名 + dtype + 有效值占比), 供人工挑选 / 文档.

  :param ds: Dataset
  :returns: DataFrame(column, dtype, valid_ratio), 按有效值占比降序
  """
  n = len(ds.panel)
  rows = []
  for c in numeric_columns(ds):
    rows.append({'column': c, 'dtype': str(ds.panel[c].dtype),
                 'valid_ratio': round(float(ds.panel[c].notna().mean()), 3)})
  return pd.DataFrame(rows).sort_values('valid_ratio', ascending=False) \
                           .reset_index(drop=True)


def main():
  ap = argparse.ArgumentParser(description='ml 数据接入层自检: 加载池 + 列出可用特征列')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--start', default=None)
  ap.add_argument('--end', default=None)
  ap.add_argument('--top', type=int, default=30, help='打印有效值占比最高的前 N 列')
  a = ap.parse_args()

  log = cfg.get_logger('ml.data')
  ds = load(a.pool, a.interval, a.start, a.end, log=log)

  close = ds.wide('close')
  log.info(f'[wide] close 宽表 shape={close.shape} (date × symbol), '
           f'NaN 占比={close.isna().mean().mean():.1%}')

  cols = numeric_columns(ds)
  log.info(f'[cols] 面板共 {ds.panel.shape[1]} 列, 其中可用数值特征列 {len(cols)} 个')
  log.info(f'[cols] 可用特征列(前 {a.top} 个): {cols[:a.top]}')

  tab = feature_candidates(ds).head(a.top)
  log.info(f'\n[feature_candidates] 有效值占比最高的前 {a.top} 列:\n'
           f'{tab.to_string(index=False)}')
  log.info(f'[note] 第1步 features.py 可用 --source factor|ta|both 选择特征来源; '
           f'TA 列(--source ta/both)会先做退化过滤, 但其因果性无法验证(见 features).')


if __name__ == '__main__':
  main()
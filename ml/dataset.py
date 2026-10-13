# -*- coding: utf-8 -*-
"""
ml.dataset — 样本组装层(第3步)
===============================
职责: 把"特征张量(第1步)"与"标签(第2步)"对齐成模型能训练/预测的样本集,
      并按统一口径做时序切分(IS / OOS / recent).

--- 一条样本 = 一个 (交易日 t, 标的 s) ---
  X 行 = t 日该标的的 F 个特征值(第1步; 截面标准化, 只用 <= t 的数据)
  y    = t 日该标的的截面 rank 标签(第2步; 用 t+1..t+1+h 的未来收益)

  两边的构造都已在各自模块里通过因果/前视检验; 本层只负责"对齐 + 丢弃无效样本".

--- 丢弃规则(宁缺毋滥) ---
  1. 标签为 NaN 的样本一律丢: 没有答案就无法训练(末尾 h+1 日、薄截面日都属此类);
  2. 特征全为 NaN 的样本丢(该标的该日没有任何可用信息);
  3. 部分特征为 NaN 的样本**保留** —— 树模型原生处理缺失; 线性模型在模型层做
     缺失填补(注意: 填补统计量只能在 IS 内估计, 属模型层红线).

--- 时序切分(红线: 只在 IS 上定参) ---
  IS / OOS / recent 三段与 factor 体系完全一致(config.WINDOWS).
  切分只按**日期**做掩码, 不做随机划分 —— 随机划分会让未来样本混进训练集(前视).

自检:
  cd ~/git && python -m quant.ml.dataset --pool etf_3x
"""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor.data import Dataset

from quant.ml import config as cfg
from quant.ml.data import load
from quant.ml.features import FeaturePanel, build_features
from quant.ml.labels import build_labels


class _SampleSetMixin:
  """
  SampleSet / SequenceSampleSet 共用的最小接口.

  两者只在 X 的维度(扁平 (n, F) vs 序列 (n, L, F))与 select 的返回类型上不同;
  这里收拢与维度无关、逐字相同的那部分(样本数 / 信号日集合 / 窗口掩码 / 切窗),
  避免两份实现各自漂移(见 seq.SequenceSampleSet).
  """

  @property
  def n_samples(self) -> int:
    return int(self.X.shape[0])

  @property
  def dates(self) -> pd.DatetimeIndex:
    """样本覆盖的交易日集合(序列样本取窗口末端信号日). 供 holdout_split 按日切分."""
    return pd.DatetimeIndex(sorted(self.meta['date'].unique()))

  def window_mask(self, window: str) -> np.ndarray:
    """按窗口名(IS/OOS/recent/full)生成样本掩码."""
    start, end = cfg.window_range(window)
    d = self.meta['date']
    m = np.ones(len(d), dtype=bool)
    if start is not None:
      m &= (d >= pd.Timestamp(start)).to_numpy()
    if end is not None:
      m &= (d <= pd.Timestamp(end)).to_numpy()
    return m

  def split(self, window: str):
    """取某窗口的子样本集(新对象, 不改自身)."""
    return self.select(self.window_mask(window))


@dataclass
class SampleSet(_SampleSetMixin):
  """
  对齐后的样本集.

  :param X: (n_samples × n_features) float32; NaN 表示该特征在此样本缺失
  :param y: (n_samples,) float32; 模型目标(截面 rank/z/raw)
  :param meta: DataFrame(n_samples × 2), 列 = date / symbol
  :param features: 特征名列表(与 X 列顺序一致)
  :param horizon: 预测跨度 h
  :param label_method: 标签方法('rank'/'z'/'raw')
  """

  X: np.ndarray
  y: np.ndarray
  meta: pd.DataFrame
  features: list
  horizon: int
  label_method: str

  @property
  def n_features(self) -> int:
    return int(self.X.shape[1])

  def select(self, mask) -> 'SampleSet':
    """按布尔掩码取子样本集(新对象, 不改自身; 行序保持原相对顺序)."""
    m = np.asarray(mask, dtype=bool)
    return SampleSet(self.X[m], self.y[m], self.meta.loc[m].reset_index(drop=True),
                     self.features, self.horizon, self.label_method)

  def summary(self) -> str:
    x = self.X
    return (f'SampleSet(n={self.n_samples:,} × F={self.n_features}), '
            f'h={self.horizon}, label={self.label_method}\n'
            f'  区间      : {self.meta["date"].min():%Y-%m-%d} ~ '
            f'{self.meta["date"].max():%Y-%m-%d}\n'
            f'  交易日/标的: {len(self.dates)} 天 / {self.meta["symbol"].nunique()} 个\n'
            f'  X 缺失占比 : {float(np.isnan(x).mean()):.1%}\n'
            f'  y 范围     : [{np.nanmin(self.y):.3f}, {np.nanmax(self.y):.3f}]')


def assemble(ds: Dataset, feature_names: list = None, h: int = cfg.DEFAULT_HORIZON,
             label_method: str = cfg.DEFAULT_LABEL,
             feature_prep: str = cfg.FEATURE_PREP, min_cs: int = fcfg.MIN_CS,
             feature_source: str = cfg.DEFAULT_FEATURE_SOURCE,
             window: str = None, start: str = None, end: str = None,
             dropna: str = 'all', log=None) -> SampleSet:
  """
  组装样本集.

  步骤:
    1. 在全历史上构建特征张量(不截断 -> 不引入 warmup 污染);
    2. 在全历史上构建标签(标签要用未来, 必须全历史);
    3. 按 window / start / end 取特征窗口, 展平成长表;
    4. 对齐标签, 按 dropna 规则丢弃无效样本.

  :param ds: Dataset
  :param feature_names: 特征名(None = 按 feature_source 自动取)
  :param h: 预测跨度
  :param label_method: 标签方法('rank'/'z')
  :param feature_prep: 特征预处理方法(factor.prepare 的方法名)
  :param min_cs: 每日截面最小样本数
  :param feature_source: 特征来源('factor'/'ta'/'both'); TA 列因果性未知, 见 features
  :param window: 取某个窗口('IS'/'OOS'/'recent'/'full'); None = 用 start/end
  :param start/end: 显式日期边界(优先级低于 window)
  :param dropna: 'all'(仅丢特征全 NaN) / 'any'(丢任一特征缺失) / 'none'(不丢)
  :param log: 可选 logger
  :returns: SampleSet
  """
  if window is not None:
    start, end = cfg.window_range(window)

  fp = build_features(ds, feature_names, feature_prep, min_cs,
                      source=feature_source, log=log)               # 全历史(不截)
  lab = build_labels(ds, h, label_method, min_cs)                   # 全历史(用未来)

  sub = fp.subset(start, end)
  # 直接按 numpy 展平(date-major / symbol-minor, 与 FeaturePanel.to_long 的 reshape 逐格同序),
  # 不再物化 (T*N, F) 的 float64 长表 / Series / isna 布尔表, 省下最大的一块内存峰值.
  t, n_, f = sub.data.shape
  X_all = sub.data.reshape(t * n_, f)                              # (T*N, F) 视图, 不复制
  y_all = (lab.reindex(index=sub.dates, columns=sub.symbols)
           .to_numpy(dtype=float).reshape(-1))                     # 与 X_all 逐行同序

  nan_x = np.isnan(X_all)
  if dropna == 'any':
    keep = ~nan_x.any(axis=1)
  elif dropna == 'all':
    keep = ~nan_x.all(axis=1)
  elif dropna == 'none':
    keep = np.ones(len(X_all), dtype=bool)
  else:
    raise KeyError(f'未知 dropna: {dropna} (可选 all/any/none)')
  keep &= ~np.isnan(y_all)                                         # 标签缺失必丢

  idx = pd.MultiIndex.from_product([sub.dates, sub.symbols], names=['date', 'symbol'])
  out = SampleSet(
      X=X_all[keep],
      y=y_all[keep].astype(np.float32),
      meta=idx.to_frame(index=False)[keep].reset_index(drop=True),
      features=list(sub.names), horizon=h, label_method=label_method)

  if log is not None:
    log.info(f'[assemble] 特征 {len(sub.names)} 个, 原始 {len(X_all):,} 格 -> '
             f'有效样本 {out.n_samples:,} ({out.n_samples / max(len(X_all), 1):.1%})')
    log.info(f'[assemble] {out.summary()}')
  return out


def holdout_split(s: SampleSet, valid_frac: float = 0.2,
                  embargo: int | None = None) -> tuple:
  """
  在样本集内做**时序**训练/验证切分(用于 IS 内部早停/调参).

  规则(红线: 时序, 且训练块末与验证块首之间留 embargo):
    1. 按交易日排序, 取时间尾部 valid_frac 比例的交易日作验证块;
    2. 训练块 = 验证块首日之前、且日期早于 (验证首日 - embargo 交易日) 的样本;
    3. embargo 交易日 = h+1, 防止训练样本的标签 span(t+1..t+1+h) 越过验证块首日
       -> 隐性前视.

  :param s: SampleSet
  :param valid_frac: 验证块占交易日比例(按天切, 不是按样本)
  :param embargo: 训练/验证之间空出的交易日数; None 时按 s.horizon + 1 自动取值
                  (与上文规则 3 一致, 换 h 时不必手改)
  :returns: (train SampleSet, valid SampleSet, info dict)
  """
  if embargo is None:
    embargo = int(s.horizon) + 1
  days = s.dates
  n_days = len(days)
  if n_days < 10:
    raise ValueError(f'交易日过少({n_days}), 无法时序切分')
  n_valid = max(int(round(n_days * valid_frac)), 1)
  v_start = days[n_days - n_valid]                       # 验证块首日
  # 训练块的日期上界: v_start 往前数 embargo 个交易日
  cut = max(n_days - n_valid - embargo, 1)
  t_end = days[cut - 1]                                  # 训练块末日
  d = s.meta['date']
  tr = s.select((d <= t_end).to_numpy())
  va = s.select((d >= v_start).to_numpy())
  info = {'n_days': n_days, 'n_valid_days': n_valid, 'embargo': embargo,
          'train_end': t_end, 'valid_start': v_start,
          'valid_end': days[-1], 'n_train': tr.n_samples, 'n_valid': va.n_samples}
  return tr, va, info


def split_table(s: SampleSet) -> pd.DataFrame:
  """各窗口样本量 / 标签均值 / 特征缺失占比一览(自检与报告用)."""
  rows = []
  total = s.n_samples
  for w in ('IS', 'OOS', 'recent'):
    sub = s.split(w)
    if sub.n_samples == 0:
      rows.append({'window': w, 'n': 0, 'share': 0.0, 'y_mean': np.nan,
                   'x_nan': np.nan, 'days': 0})
      continue
    rows.append({'window': w, 'n': sub.n_samples, 'share': round(sub.n_samples / total, 3),
                 'y_mean': round(float(np.nanmean(sub.y)), 4),
                 'x_nan': round(float(np.isnan(sub.X).mean()), 3),
                 'days': len(sub.dates)})
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser(description='ml 样本组装层自检: 对齐特征/标签 + 时序切分')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--prep', default=cfg.FEATURE_PREP, choices=('raw', 'z', 'rank', 'mad_z', 'q_z'))
  ap.add_argument('--source', default=cfg.DEFAULT_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES),
                  help='特征来源: factor / ta / both')
  ap.add_argument('--group', default=None, help='只用某类因子当特征(仅 factor 来源)')
  ap.add_argument('--dropna', default='all', choices=('all', 'any', 'none'))
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  a = ap.parse_args()

  from quant.factor import factor as fct

  log = cfg.get_logger('ml.dataset')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  if a.group:
    a.source, names = 'factor', fct.list_factors(a.group)
  else:
    names = None

  s = assemble(ds, names, a.h, a.method, a.prep, a.min_cs,
               feature_source=a.source, dropna=a.dropna, log=log)

  log.info('\n[时序切分] 各窗口样本量(红线: 只在 IS 定参)')
  log.info('\n' + split_table(s).to_string(index=False))

  log.info('\n[样本示例] 前 3 行 meta + y + 前 3 个特征值')
  head = s.meta.head(3).copy()
  head['y'] = s.y[:3].round(3)
  for j in range(min(3, s.n_features)):
    head[s.features[j]] = s.X[:3, j].round(3)
  log.info('\n' + head.to_string(index=False))

  log.info('\n[下一步] 第4步起进入模型层: 先线性基线(Ridge), 再 LightGBM, 后 DL. '
           '评分/回测将复用 factor.evaluate / factor.backtest.')


if __name__ == '__main__':
  main()
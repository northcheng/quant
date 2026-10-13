# -*- coding: utf-8 -*-
"""
ml.features — 特征工程层(第1步)
================================
职责: 把"原始数据"变成"模型能吃的特征矩阵".

--- 特征来源: factor 注册表因子 + pkl 面板的 TA 列 ---
  两类来源走同一条管线: 原始宽表 -> 逐日截面标准化(prepare) -> 堆叠成张量.
  source 参数:
    'factor'      : factor 注册表的 59 个因子 —— 每个都过了 verify_causal
                    (截断一致性, 只读 <= t 的数据), 可直接当"已验证特征";
    'ta'          : pkl 面板里的 163 个数值 TA 列(自动剔除退化列后约 155 个);
    'both'        : 两者并集(因子在前, TA 列在后) —— 默认.

  ⚠ TA 列的因果性**无法用截断一致性检验验证**(诚实声明, 不是遗漏):
    TA 列是 pkl 里预先算好的值, 不是本包按需重算的. ds.slice(end=cut) 只截掉行,
    被保留行的数值仍是"生成时用到了整段历史"的产物 -> A/B 逐值比对恒等, 检验
    形同虚设. 故 verify_features 对 TA 列一律标 verifiable=False, 而非给虚假的
    causal=True. 另一已知风险: pkl 的 TA 生成流程 bc_technical_analysis.preprocess
    里用了 df['Split'].shift(-1)(取未来一天), 存在潜在前视(详见 TA_CAVEAT).

  退化 TA 列过滤(取 TA 列表时默认执行): 剔除"逐日截面恒定"列(rank 后塌缩为 0,
  无信息)与"有效值过低"列(如全 NaN 的 support/resistant). 过滤统计量只在
  TA_FILTER_WINDOW(默认 IS)上算 —— 特征选择属全局决策, 用全样本会偷看 OOS(红线).

--- 关键一步: 逐日截面标准化 ---
  因子尺度千差万别(mom_20 是收益率、N_range20 是价格区间占比...), 直接喂给线性
  模型会被大量纲带偏. 所以每个特征都过一遍 factor.prepare(默认 rank):
    逐日(axis=1)截面秩变换 -> [-1, 1], 对极值免疫, 且天然因果(只用当天截面).

  这是全流程**唯一允许的"标准化"** —— 因为它逐日算, 不跨日、不拟合全局参数.
  任何"在整段样本上 fit 的 scaler"都会偷看未来, 本包一律禁止(红线).

--- 输出形态 ---
  FeaturePanel: 3D 张量 (T × N × F) + 名字/日期/标的索引
    T = 交易日, N = 标的, F = 特征数
  另提供 to_long(): 展平成 MultiIndex(date, symbol) 的长表, 供 sklearn/LightGBM 用.

自检:
  cd ~/git && python -m quant.ml.features --pool etf_3x
  cd ~/git && python -m quant.ml.features --pool etf_3x --check
  cd ~/git && python -m quant.ml.features --pool etf_3x --source ta
  cd ~/git && python -m quant.ml.features --pool etf_3x --source both --check --demo
"""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import Dataset

from quant.ml import config as cfg
from quant.ml.data import load, numeric_columns


# 特征来源标签(与 cfg.FEATURE_SOURCES 对应)
SRC_FACTOR, SRC_TA, SRC_BOTH = 'factor', 'ta', 'both'

# TA 列因果性的诚实声明(在构建/检验时反复提示, 避免被当成"已验证特征")
TA_CAVEAT = (
    'TA 列来自 pkl 预计算, 无独立生成器 -> 截断一致性检验对其恒等失效, 因果性无法'
    '验证; 且 bc_technical_analysis.preprocess 用了 Split.shift(-1)(取未来一天), '
    '存在潜在前视. 请勿把 TA 特征的 OOS 结论当作无偏证据.')


# ============================ 特征张量 ============================

@dataclass
class FeaturePanel:
  """
  一批特征组成的张量 (T × N × F), 附名字与索引.

  :param names: 特征名列表(长度 F, 顺序与 data 第 3 维一致)
  :param dates: 交易日索引(长度 T)
  :param symbols: 标的列表(长度 N)
  :param data: float32 数组, shape=(T, N, F); NaN = 该标的该日该特征无效
  :param prep_method: 生成时用的预处理方法(便于追溯)
  :param sources: 每个特征的来源('factor'/'ta'), 长度 F; None 视为全 'factor'
  """

  names: list
  dates: pd.DatetimeIndex
  symbols: list
  data: np.ndarray
  prep_method: str = cfg.FEATURE_PREP
  sources: list = None

  def __post_init__(self):
    if self.sources is None:
      self.sources = [SRC_FACTOR] * len(self.names)

  @property
  def n_dates(self) -> int:
    return len(self.dates)

  @property
  def n_symbols(self) -> int:
    return len(self.symbols)

  @property
  def n_features(self) -> int:
    return len(self.names)

  def index_of(self, name: str) -> int:
    """特征名 -> 第 3 维下标(不存在则报错)."""
    try:
      return self.names.index(name)
    except ValueError:
      raise KeyError(f'未知特征: {name}')

  def source_of(self, name: str) -> str:
    """特征名 -> 来源('factor'/'ta')."""
    return self.sources[self.index_of(name)]

  def wide(self, name: str) -> pd.DataFrame:
    """取单个特征 -> (date × symbol) 宽表."""
    j = self.index_of(name)
    return pd.DataFrame(self.data[:, :, j], index=self.dates, columns=self.symbols)

  def to_long(self) -> pd.DataFrame:
    """
    展平成 MultiIndex(date, symbol) 的长表: 每行一个"样本时点", 列 = 各特征.

    内存提示: shape = (T*N, F); 大池(如 company_1000)会占较多内存, 一般先
    用 subset() 截窗口再展平.
    """
    t, n, f = self.data.shape
    idx = pd.MultiIndex.from_product([self.dates, self.symbols],
                                     names=['date', 'symbol'])
    return pd.DataFrame(self.data.reshape(t * n, f), index=idx, columns=self.names)

  def subset(self, start=None, end=None) -> 'FeaturePanel':
    """按交易日截取(只切行, 不改数值) -> 新 FeaturePanel."""
    idx = self.dates
    if start is not None:
      idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
      idx = idx[idx <= pd.Timestamp(end)]
    pos = self.dates.get_indexer(idx)
    return FeaturePanel(self.names, self.dates[pos], self.symbols,
                        self.data[pos].copy(), self.prep_method, list(self.sources))

  def summary(self) -> str:
    v = self.data
    valid = float(np.isfinite(v).mean())
    n_f = sum(1 for s in self.sources if s == SRC_FACTOR)
    n_t = sum(1 for s in self.sources if s == SRC_TA)
    return (f'FeaturePanel(T={self.n_dates} × N={self.n_symbols} × F={self.n_features})\n'
            f'  特征名    : {self.names[:6]}{" ..." if self.n_features > 6 else ""}\n'
            f'  来源      : factor {n_f} + ta {n_t}\n'
            f'  预处理    : {self.prep_method}\n'
            f'  有效值占比: {valid:.1%}\n'
            f'  取值范围  : [{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]'
            if np.isfinite(v).any() else 'FeaturePanel(空)')


# ============================ 构建 ============================

def _align(df: pd.DataFrame, dates, symbols) -> pd.DataFrame:
  """把因子宽表对齐到统一的 (dates × symbols) 网格(缺的位置补 NaN)."""
  return df.reindex(index=dates, columns=symbols)


# ============================ 特征来源(因子 / TA 列) ============================

def ta_columns(ds: Dataset) -> list:
  """面板里可作特征的数值 TA 列(排除 OHLCV 基础列). 即 data.numeric_columns."""
  return numeric_columns(ds)


def filter_degenerate(ds: Dataset, cols: list, min_valid: float = cfg.TA_MIN_VALID,
                      max_flat: float = cfg.TA_MAX_FLAT, start=None, end=None,
                      log=None):
  """
  剔除退化的 TA 列(对逐日截面排序无信息).

  两类退化:
    - 有效值占比 < min_valid : 整列近乎全 NaN(如 support/resistant), 无料可用;
    - 逐日截面恒定占比 > max_flat: 当天所有标的取同一值, rank 后塌缩为同一个数
      (如 Dividend/Split/candle_gap_*), 对截面排序零贡献.

  统计量只在 [start, end] 上算 —— 特征选择属全局决策, 若用全样本会偷看 OOS(红线).

  :param ds: Dataset
  :param cols: 候选 TA 列
  :param min_valid: 有效值占比下限
  :param max_flat: 逐日截面恒定占比上限
  :param start/end: 统计区间(None 表示不限)
  :param log: 可选 logger
  :returns: (kept: list, dropped: DataFrame[column, reason])
  """
  sub = ds.slice(start, end) if (start is not None or end is not None) else ds
  kept, dropped = [], []
  for c in cols:
    w = sub.wide(c)
    valid = float(w.notna().to_numpy().mean())
    if valid < min_valid:
      dropped.append({'column': c, 'reason': f'有效值占比 {valid:.2f} < {min_valid}'})
      continue
    flat = float((w.std(axis=1, skipna=True).fillna(0.0) < 1e-12).mean())
    if flat > max_flat:
      dropped.append({'column': c, 'reason': f'截面恒定占比 {flat:.2f} > {max_flat}'})
      continue
    kept.append(c)
  dropped = pd.DataFrame(dropped, columns=['column', 'reason'])
  if log is not None:
    log.info(f'[ta-filter] 候选 {len(cols)} 列 -> 保留 {len(kept)}, 剔除 {len(dropped)}'
             f' (区间 {start}~{end}, min_valid={min_valid}, max_flat={max_flat})')
  return kept, dropped


def _resolve_features(ds: Dataset, names: list, source: str, log=None,
                      ta_filter: list = None):
  """
  决定"用哪些特征 + 各自来源".

  :param ds: Dataset
  :param names: 显式特征名(None = 按 source 自动取全量)
  :param source: 'factor' / 'ta' / 'both'
  :param log: 可选 logger
  :param ta_filter: 已算好的 TA 保留列(None = 自行 filter_degenerate);
                    调用方(如 main)若已跑过退化过滤, 传入以复用, 避免重复计算
  :returns: (names: list, sources: list) —— 两者等长且顺序一致
  """
  if source not in cfg.FEATURE_SOURCES:
    raise KeyError(f'未知特征来源: {source} (可选 {list(cfg.FEATURE_SOURCES)})')
  all_ta = set(ta_columns(ds))

  if names is not None:                              # 显式指定: 逐个判源
    src = []
    for n in names:
      if n in fct.REGISTRY:
        src.append(SRC_FACTOR)
      elif n in all_ta:
        src.append(SRC_TA)
      else:
        raise KeyError(f'未知特征(既非因子也非面板数值列): {n}')
    return list(names), src

  out_names, out_src = [], []
  if source in (SRC_FACTOR, SRC_BOTH):
    fs = fct.list_factors()
    out_names += fs
    out_src += [SRC_FACTOR] * len(fs)
  if source in (SRC_TA, SRC_BOTH):
    start, end = cfg.window_range(cfg.TA_FILTER_WINDOW)
    if ta_filter is not None:
      kept = list(ta_filter)                       # 复用调用方已算好的保留列(顺序一致)
    else:
      kept, _ = filter_degenerate(ds, sorted(set(ta_columns(ds))), cfg.TA_MIN_VALID,
                                  cfg.TA_MAX_FLAT, start, end, log=log)
    out_names += kept
    out_src += [SRC_TA] * len(kept)
  return out_names, out_src


def _raw_feature(ds: Dataset, name: str, source: str) -> pd.DataFrame:
  """特征名 -> 原始宽表(因子现算; TA 列直接取面板值)."""
  if source == SRC_FACTOR:
    return fct.get_factor(name).compute(ds)
  if source == SRC_TA:
    return ds.wide(name)
  raise KeyError(f'未知来源: {source}')


def build_features(ds: Dataset, names: list = None,
                   prep_method: str = cfg.FEATURE_PREP,
                   min_cs: int = fcfg.MIN_CS,
                   source: str = cfg.DEFAULT_FEATURE_SOURCE,
                   log=None, ta_filter: list = None) -> FeaturePanel:
  """
  构建特征张量(因子 / TA 列 / 两者).

  流程(每个特征): 取原始宽表 -> 逐日截面标准化(prepare) -> 对齐到统一网格 -> 堆叠.

  :param ds: Dataset
  :param names: 显式特征名(None = 按 source 自动取; 显式指定时逐个判源)
  :param prep_method: factor.prepare 的方法名('rank'/'z'/'raw'/'mad_z'/'q_z')
  :param min_cs: 每日截面最小样本数
  :param source: 'factor' / 'ta' / 'both'(默认); 取 TA 时自动做退化列过滤
  :param log: 可选 logger
  :param ta_filter: 已算好的 TA 保留列(None = 内部自行 filter_degenerate);
                    调用方若已过滤过, 传入以复用, 避免重复计算(顺序须与内部一致)
  :returns: FeaturePanel, data 为 float32 的 (T × N × F)
  """
  use_names, sources = _resolve_features(ds, names, source, log=log, ta_filter=ta_filter)
  if not use_names:
    raise ValueError('特征名列表为空')
  if SRC_TA in sources and log is not None:
    log.info(f'[caveat] {TA_CAVEAT}')

  dates, symbols = ds.dates, ds.symbols
  t, n_ = len(dates), len(symbols)
  data = np.full((t, n_, len(use_names)), np.nan, dtype=np.float32)

  for j, (name, src) in enumerate(zip(use_names, sources)):
    raw = _raw_feature(ds, name, src)
    pr = prep.prepare(raw, prep_method, min_cs)
    data[:, :, j] = _align(pr, dates, symbols).to_numpy(dtype=np.float32)

  fp = FeaturePanel(names=list(use_names), dates=dates, symbols=symbols, data=data,
                    prep_method=prep_method, sources=list(sources))
  if log is not None:
    log.info(f'[build] {fp.summary()}')
  return fp


# ============================ 防前视: 截断一致性检验 ============================

def _compare(a: np.ndarray, b: np.ndarray):
  """逐值比对两个数组 -> (cells, max_diff, nan_same)."""
  if a.shape != b.shape or a.size == 0:
    return 0, float('nan'), False
  nan_same = not bool((np.isnan(a) != np.isnan(b)).any())
  with np.errstate(invalid='ignore'):
    d = np.abs(a - b)
    max_diff = 0.0 if np.isnan(d).all() else float(np.nanmax(d))
  return int(a.size), max_diff, nan_same


def verify_features(ds: Dataset, names: list = None,
                    prep_method: str = cfg.FEATURE_PREP, cut: str = fcfg.VERIFY_CUT,
                    min_cs: int = fcfg.MIN_CS, tol: float = 1e-9,
                    source: str = cfg.DEFAULT_FEATURE_SOURCE) -> pd.DataFrame:
  """
  特征管线的截断一致性检验(特征 = 原始宽表 + 预处理 的复合).

  因子特征:
    原理(同 factor.verify_causal / prepare.truncation_check):
      A = 用全历史算"因子 -> 预处理", 再截到 cut
      B = 只把 <= cut 的数据喂进去, 算"因子 -> 预处理"
    A 与 B 逐值相同 -> 整条特征管线因果.

  TA 特征:
    **不可验证** —— TA 列是 pkl 预计算值, ds.slice 只截行不改值, A/B 恒等,
    检验形同虚设. 故直接标 verifiable=False / causal=None, 绝不给出虚假通过.

  :param ds: Dataset
  :param names: 要检验的特征(None = 按 source 自动取全量)
  :param prep_method: 预处理方法
  :param cut: 截断日
  :param min_cs: 每日截面最小样本数
  :param tol: 允许的最大绝对差
  :param source: 'factor' / 'ta' / 'both'
  :returns: DataFrame(feature, source, verifiable, cells, max_diff, nan_same, causal, note)
  """
  use_names, sources = _resolve_features(ds, names, source)
  part = ds.slice(end=cut)
  rows = []
  for name, src in zip(use_names, sources):
    if src == SRC_TA:
      rows.append({'feature': name, 'source': src, 'verifiable': False,
                   'cells': 0, 'max_diff': np.nan, 'nan_same': False, 'causal': None,
                   'note': 'pkl 预计算列: 截断检验恒等失效'})
      continue
    fn = fct.get_factor(name)
    a = prep.prepare(fn.compute(ds), prep_method, min_cs).loc[:cut]     # 全历史 -> 截
    b = prep.prepare(fn.compute(part), prep_method, min_cs)             # 先截 -> 再算
    idx = a.index.intersection(b.index)
    col = a.columns.intersection(b.columns)
    cells, max_diff, nan_same = _compare(a.loc[idx, col].to_numpy(dtype=float),
                                         b.loc[idx, col].to_numpy(dtype=float))
    rows.append({'feature': name, 'source': src, 'verifiable': True,
                 'cells': cells, 'max_diff': max_diff, 'nan_same': nan_same,
                 'causal': bool(cells > 0 and nan_same and max_diff <= tol), 'note': ''})
  return pd.DataFrame(rows)


def _leak_fullsample_rank(df: pd.DataFrame) -> pd.DataFrame:
  """反例(故意前视): 用整个样本排一次名次(跨日) —— 检验必须能抓出它."""
  return df.rank().rank(pct=True) * 2 - 1


def main():
  ap = argparse.ArgumentParser(description='ml 特征工程层自检: 构建特征张量 + 因果性检验')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--source', default=cfg.DEFAULT_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES),
                  help='特征来源: factor(因子) / ta(面板 TA 列) / both')
  ap.add_argument('--method', default=cfg.FEATURE_PREP, choices=prep.PREP_METHODS)
  ap.add_argument('--group', default=None, help='只用某类因子当特征(仅 factor 来源)')
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  ap.add_argument('--check', action='store_true', help='做截断一致性检验')
  ap.add_argument('--demo', action='store_true', help='注入前视反例, 验证检验有效')
  ap.add_argument('--cut', default=fcfg.VERIFY_CUT)
  ap.add_argument('--show', default=None, help='打印某特征最近 5 行')
  a = ap.parse_args()

  log = cfg.get_logger('ml.features')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  # --group 只作用于因子来源; 指定后强制 source=factor
  if a.group:
    if a.source != SRC_FACTOR:
      log.info(f'[note] --group 只作用于因子, source 由 {a.source} 改为 factor')
    a.source = SRC_FACTOR
    names = fct.list_factors(a.group)
  else:
    names = None

  # TA 来源: 先单独跑一遍过滤以便打印被剔除的退化列, 并把保留列传给 build_features 复用,
  # 避免在 _resolve_features 里对同一批 TA 列重复 filter_degenerate(顺序与内部 sorted(set()) 一致).
  ta_filter = None
  if a.source in (SRC_TA, SRC_BOTH):
    start, end = cfg.window_range(cfg.TA_FILTER_WINDOW)
    ta_filter, dropped = filter_degenerate(ds, sorted(set(ta_columns(ds))), cfg.TA_MIN_VALID,
                                           cfg.TA_MAX_FLAT, start, end, log=log)
    if len(dropped):
      log.info(f'\n[ta-filter] 剔除的退化 TA 列:\n{dropped.to_string(index=False)}')

  fp = build_features(ds, names, a.method, a.min_cs, source=a.source, log=log,
                      ta_filter=ta_filter)

  # 逐特征有效值占比(找出暖机长/覆盖低的特征)
  valid = np.isfinite(fp.data).mean(axis=(0, 1))
  tab = pd.DataFrame({'feature': fp.names, 'source': fp.sources,
                      'valid_ratio': valid.round(3)}) \
          .sort_values('valid_ratio', ascending=False)
  log.info(f'\n[coverage] 特征有效值占比(前10):\n{tab.head(10).to_string(index=False)}')
  log.info(f'[coverage] 有效值占比最低5个:\n{tab.tail(5).to_string(index=False)}')

  if a.check:
    ck = verify_features(ds, names, a.method, a.cut, a.min_cs, source=a.source)
    ver = ck.loc[ck['verifiable']]
    unver = ck.loc[~ck['verifiable']]
    ok = ver['causal'].astype(bool)                  # TA 行的 causal=None 使整列为 object
    bad = ver.loc[~ok, 'feature'].tolist()
    log.info(f'\n[截断一致性检验] method={a.method}, 全历史 pipeline vs 只喂到 {a.cut}')
    log.info(f'  可验证特征 {len(ver)} 个, 通过 {int(ok.sum())} 个, '
             f'max_diff 最大 = {ver["max_diff"].max():.2e}' if len(ver)
             else '  可验证特征 0 个')
    log.info(f'  不可验证特征 {len(unver)} 个(TA 列) -> 因果性未知, 不计入结论')
    if len(ver):
      log.info('[结论] 可验证的因子特征管线因果' if not bad
               else f'[结论] 疑似前视: {bad}')
    else:
      log.info('[结论] 无可验证特征(全是 TA 列), 本检验不适用')
    if len(unver):
      log.info(f'[caveat] {TA_CAVEAT}')

    if a.demo:
      demo = [n for n in (names if names is not None else fct.list_factors())
              if n in fct.REGISTRY][:5]
      part = ds.slice(end=a.cut)
      rows = []
      for name in demo:
        fn = fct.get_factor(name)
        aa = _leak_fullsample_rank(fn.compute(ds)).loc[:a.cut]
        bb = _leak_fullsample_rank(fn.compute(part))
        idx = aa.index.intersection(bb.index)
        col = aa.columns.intersection(bb.columns)
        _, md, ns = _compare(aa.loc[idx, col].to_numpy(dtype=float),
                             bb.loc[idx, col].to_numpy(dtype=float))
        rows.append({'feature': name, 'max_diff': md, 'nan_same': ns,
                     'causal': bool(ns and md <= 1e-9)})
      dm = pd.DataFrame(rows)
      n_bad = int((~dm['causal']).sum())
      log.info(f'\n[反向验证] 注入"全样本跨日 rank"反例 -> {n_bad}/{len(dm)} 个被抓出 '
               f'(max_diff 示例 {dm["max_diff"].max():.3f})')
      log.info('[结论] 检验有效' if n_bad == len(dm) else '[结论] 检验失效, 需排查')

  if a.show:
    log.info(f'\n[{a.show}] (来源 {fp.source_of(a.show)}) 最近 5 行:\n'
             f'{fp.wide(a.show).tail(5).to_string()}')


if __name__ == '__main__':
  main()
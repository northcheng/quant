# -*- coding: utf-8 -*-
"""
ml.seq — 序列样本构造层(第3步之补充)
=====================================
职责: 给序列模型(RNN/Transformer)造样本 —— 把第3步的"单日截面样本"换成
      "标的过去 L 天的特征窗口".

--- 一条序列样本 = 一个 (信号交易日 t, 标的 s) ---
  X = 标的 s 在 D_{t-L+1} .. D_t 上的 L × F 特征窗口(**窗口末端就是信号日 t**)
  y = t 日的 h 日前向 rank 标签(与 ridge/gbdt/mlp 用的是**同一份标签**)

  于是形状从 SampleSet 的 (n × F) 变成 (n × L × F), n 与 meta(date/symbol) 口径不变.
  换的只是"输入的横轴"(序列长度), 不是评估协议.

--- 关于"不用 IS/OOS/recent 行不行"(用户提问) ---
  两个必须同时满足的条件, 且彼此正交:
    1. **无前视(纵向)**: 单样本的输入不得含信号日 t 之后的信息
       -> 由滑窗构造保证(窗口末端 = t); 本模块用"截断一致性检验"逐值验证(见
          verify_no_lookahead);
    2. **样本外(横向)**: 模型拟合的时段不得与评估时段重叠
       -> IS/OOS/recent **就是**干这件事的; 换成"D_{t-n}~D_t 预测 D_{t+h}"只解决了 ①,
          并没有解决 ②. 若取消 IS/OOS, 报出的就是 in-sample 数字.
  结论: 本模块沿用 IS/OOS/recent 协议(与已算好的 ridge/gbdt/mlp 严格可比, 且守住
  "recent 不参与拟合"的红线); 滑窗只改样本构造, 不改协议.

--- 预处理红线(为什么这里默认只用 factor 列) ---
  X 的值来自 features.build_features, 已是逐日截面 rank(只用当天截面, 天然因果).
  本模块**不引入任何跨日统计量**, 填补/标准化一律留到模型层(只在 fit 样本上估).

  TA 列: 默认来源由 cfg.SEQ_FEATURE_SOURCE 决定(默认 'both', 与截面模型同口径, 便于对比).
  ⚠ TA 是 pkl 预计算列, 其生成流程用 Split.shift(-1)(取未来一天), 这处前视**无法用截断
  一致性检验验证**, 且在长度 L 的窗口下会被放大 L 倍 -> 一旦来源含 TA, build_sequences 会
  打印 TA_CAVEAT; 含 TA 的 OOS 结论仅作参考. 若需可验证的纯净口径, 用 source='factor'.

--- 与 dataset 的接口对齐 ---
  SequenceSampleSet 实现了 dates / meta / n_samples / select / split / window_mask,
  故可直接复用 dataset.holdout_split(按**交易日**切分 + embargo)—— 早停切分口径与
  ridge/gbdt/mlp 完全一致.

自检:
  cd ~/git && python -m quant.ml.seq --pool etf_3x
  cd ~/git && python -m quant.ml.seq --pool etf_3x --check
     (截断一致性检验: 把数据截到 t 后重算窗口, 应与全历史窗口逐值相同 -> 证明无前视;
      同时确认标签在截断后失效 -> 证明标签确实用了未来)
"""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor.data import Dataset

from quant.ml import config as cfg
from quant.ml.data import load
from quant.ml.dataset import _SampleSetMixin
from quant.ml.features import TA_CAVEAT, build_features
from quant.ml.labels import build_labels


# 无前视检验用的来源: 只有 factor 列能用"截断一致性"逐值验证(TA 恒等失效), 故固定 factor.
SEQ_SOURCE = 'factor'


@dataclass
class SequenceSampleSet(_SampleSetMixin):
  """
  滑窗后的序列样本集.

  :param X: (n_samples × window_len × n_features) float32; NaN = 该格特征缺失
  :param y: (n_samples,) float32; 模型目标(截面 rank/z), 与 SampleSet 同口径
  :param meta: DataFrame(n_samples × 2), 列 = date / symbol; **date = 信号日 t**(窗口末端)
  :param features: 特征名列表(与 X 第 3 维顺序一致)
  :param window_len: 窗口长度 L(交易日)
  :param horizon: 预测跨度 h
  :param label_method: 标签方法('rank'/'z')
  """

  X: np.ndarray
  y: np.ndarray
  meta: pd.DataFrame
  features: list
  window_len: int
  horizon: int
  label_method: str

  # n_samples / dates / window_mask / split 与 SampleSet 逐字相同 -> 由 _SampleSetMixin 统一提供
  # (split 内部走 self.select, 故此处返回的仍是 SequenceSampleSet). 本类只保留特有部分:
  # 序列维度 F 的取法, 以及返回 SequenceSampleSet 的 select / summary.

  @property
  def n_features(self) -> int:
    return int(self.X.shape[2])

  def select(self, mask) -> 'SequenceSampleSet':
    """按布尔掩码取子样本集(新对象, 不改自身; 行序保持原相对顺序)."""
    m = np.asarray(mask, dtype=bool)
    return SequenceSampleSet(self.X[m], self.y[m], self.meta.loc[m].reset_index(drop=True),
                             self.features, self.window_len, self.horizon, self.label_method)

  def summary(self) -> str:
    x = self.X
    return (f'SequenceSampleSet(n={self.n_samples:,} × L={self.window_len} '
            f'× F={self.n_features}), h={self.horizon}, label={self.label_method}\n'
            f'  信号日区间: {self.meta["date"].min():%Y-%m-%d} ~ '
            f'{self.meta["date"].max():%Y-%m-%d}\n'
            f'  交易日/标的: {len(self.dates)} 天 / {self.meta["symbol"].nunique()} 个\n'
            f'  X 缺失占比 : {float(np.isnan(x).mean()):.1%}\n'
            f'  y 范围     : [{np.nanmin(self.y):.3f}, {np.nanmax(self.y):.3f}]')


def build_sequences(ds: Dataset, feature_names: list = None,
                    h: int = cfg.DEFAULT_HORIZON, label_method: str = cfg.DEFAULT_LABEL,
                    feature_prep: str = cfg.FEATURE_PREP, min_cs: int = fcfg.MIN_CS,
                    window_len: int = cfg.SEQ_WINDOW, stride: int = cfg.SEQ_STRIDE,
                    source: str = cfg.SEQ_FEATURE_SOURCE,
                    log=None) -> SequenceSampleSet:
  """
  在全历史上造滑窗序列样本.

  步骤:
    1. 全历史构建特征张量 FeaturePanel(T × N × F, 逐日截面 rank) —— 不截断;
    2. 全历史构建标签(标签必须用未来);
    3. 对每个信号日 t(要求 t 之前够 L 天), 取窗口 [t-L+1, t] x 全部标的;
    4. 丢弃 y 缺失的样本(末尾 h+1 日)与"窗口内特征全 NaN"的样本.

  内存: 仅对**保留下来的样本**分配张量(先用计数掩码过滤, 再逐日填充), 不做全量复制.

  :param ds: Dataset
  :param feature_names: 特征名(None = 按 source 自动取全量; 显式指定时逐个判源)
  :param h: 预测跨度
  :param label_method: 标签方法('rank'/'z')
  :param feature_prep: 特征预处理方法(factor.prepare 的方法名)
  :param min_cs: 每日截面最小样本数
  :param window_len: 窗口长度 L(交易日)
  :param stride: 信号日起步(按交易日); >1 降采样窗口
  :param source: 特征来源 'factor' / 'ta' / 'both'(默认 cfg.SEQ_FEATURE_SOURCE);
                 含 TA 时会打印 TA_CAVEAT(其 1 日前视不可验证, 在长窗口下被放大)
  :param log: 可选 logger
  :returns: SequenceSampleSet
  """
  L = int(window_len)
  if L < 1:
    raise ValueError(f'window_len 必须 >= 1, 得到 {L}')
  if source not in cfg.FEATURE_SOURCES:
    raise KeyError(f'未知特征来源: {source} (可选 {list(cfg.FEATURE_SOURCES)})')
  stride = max(int(stride), 1)

  fp = build_features(ds, feature_names, feature_prep, min_cs, source=source, log=log)
  has_ta = any(s == 'ta' for s in fp.sources)
  if has_ta and log is not None:
    log.info(f'[seq][caveat] 来源含 TA(共 {sum(s == "ta" for s in fp.sources)} 列) -> '
             f'在长度 {L} 的窗口下 1 日前视被放大 {L} 倍. {TA_CAVEAT}')
  lab = build_labels(ds, h, label_method, min_cs)
  T, N, F = fp.data.shape
  if T < L:
    raise ValueError(f'历史交易日 {T} 少于窗口长度 {L}, 无法滑窗')

  # 信号日(窗口末端)的行下标: L-1, L-1+stride, ...
  sig_ends = np.arange(L - 1, T, stride)
  sig_dates = fp.dates[sig_ends]

  # 每个 (信号日, 标的) 窗口内的有效特征格数(用累积和算, 不物化窗口张量)
  cnt = np.isfinite(fp.data).sum(axis=2).astype(np.int64)          # (T, N)
  csum = np.cumsum(cnt, axis=0)
  starts = sig_ends - L + 1
  # starts 与 csum 均按日期主序; np.where 需把条件按列广播到 (n_sig, N)
  prev = np.where(starts.reshape(-1, 1) > 0, csum[np.maximum(starts - 1, 0)], 0)
  win_valid = csum[sig_ends] - prev                                # (n_sig, N)

  # 标签对齐到 (信号日 × 标的), 顺序 = 日期主序 / 标的次序(与填充循环一致)
  y_all = (lab.reindex(index=sig_dates, columns=fp.symbols)
           .to_numpy(dtype=np.float32).reshape(-1))
  keep_flat = np.isfinite(y_all) & (win_valid.reshape(-1) > 0)      # 标签缺失/窗口全空 -> 丢
  keep = keep_flat.reshape(len(sig_dates), N)
  n_keep = int(keep_flat.sum())

  # 只对保留样本分配张量, 逐日填充(避免对全量窗口张量做一次大复制)
  X = np.empty((n_keep, L, F), dtype=np.float32)
  pos = 0
  for k in range(len(sig_dates)):
    idx = np.nonzero(keep[k])[0]
    if len(idx) == 0:
      continue
    s0 = int(starts[k])
    # 只取保留标的的 (L, len(idx), F) 切片再转置, 等价于先造 (N, L, F) 再 win[idx],
    # 但省掉未被保留标的的全量复制.
    X[pos:pos + len(idx)] = fp.data[s0:s0 + L, idx, :].transpose(1, 0, 2)
    pos += len(idx)

  mi = pd.MultiIndex.from_product([sig_dates, fp.symbols], names=['date', 'symbol'])
  meta = mi.to_frame(index=False)[keep_flat].reset_index(drop=True)

  out = SequenceSampleSet(X=X, y=y_all[keep_flat], meta=meta, features=list(fp.names),
                          window_len=L, horizon=h, label_method=label_method)
  if log is not None:
    log.info(f'[seq] L={L}, stride={stride}, source={source}: '
             f'{len(sig_dates) * N:,} 窗口 -> 有效 {n_keep:,} '
             f'({n_keep / max(len(sig_dates) * N, 1):.1%})')
    log.info(f'[seq] {out.summary()}')
  return out


# ============================ 防前视: 截断一致性检验 ============================

def verify_no_lookahead(ds: Dataset, window_len: int = cfg.SEQ_WINDOW,
                        h: int = cfg.DEFAULT_HORIZON,
                        feature_prep: str = cfg.FEATURE_PREP,
                        min_cs: int = fcfg.MIN_CS, cut: str = None) -> dict:
  """
  序列构造的无前视检验(逐值比对, 不是"看起来对"):

    1. **窗口一致**: 把数据截到 cut 后重算因子特征, 其"以 cut 结尾的 L 天窗口"应与
       全历史版本逐值相同 -> 证明窗口只用 <= cut 的数据(无纵向前视);
    2. **标签确实用未来**: 同一截断下, cut 当日的标签应整体变 NaN(因为要 t+1..t+1+h
       的开盘价), 而全历史版本有效 -> 证明标签吃到了未来, 不是"预测已发生的事".

  :param ds: Dataset
  :param window_len: 窗口长度 L
  :param h: 持有天数
  :param feature_prep: 预处理方法
  :param min_cs: 每日截面最小样本数
  :param cut: 截断日(None = 取历史中点, 且保证两侧各有足够数据)
  :returns: dict(ok, win_equal, n_diff, label_forward_ok, cut)
  """
  dates = ds.dates
  if cut is None:
    cut = str(dates[len(dates) // 2].date())
  cut_ts = pd.Timestamp(cut)
  if cut_ts not in dates:
    cand = dates[dates <= cut_ts]
    if len(cand) == 0:
      raise ValueError(f'截断日 {cut} 之前没有交易日')
    cut_ts = cand[-1]
    cut = str(cut_ts.date())

  fp = build_features(ds, None, feature_prep, min_cs, source=SEQ_SOURCE)
  lab = build_labels(ds, h, cfg.DEFAULT_LABEL, min_cs)
  part = ds.slice(end=cut)
  fp2 = build_features(part, None, feature_prep, min_cs, source=SEQ_SOURCE)
  lab2 = build_labels(part, h, cfg.DEFAULT_LABEL, min_cs)

  i = int(fp.dates.get_indexer([cut_ts])[0])                       # cut 在 fp 中的行号
  win_equal, n_diff = False, -1
  if i >= window_len - 1 and len(fp2.dates) >= window_len:
    a = fp.data[i - window_len + 1:i + 1]
    b = fp2.data[-window_len:]
    same_nan = np.isnan(a) == np.isnan(b)
    neq = np.where(np.isfinite(a) & np.isfinite(b), a != b, False)
    n_diff = int((~same_nan | neq).sum())
    win_equal = bool(n_diff == 0)

  b_lab = lab2.reindex([cut_ts]).to_numpy(dtype=float)
  a_lab = lab.reindex([cut_ts]).to_numpy(dtype=float)
  label_forward_ok = bool(np.isnan(b_lab).all() and np.isfinite(a_lab).any())

  return {'ok': bool(win_equal and label_forward_ok), 'win_equal': win_equal,
          'n_diff': n_diff, 'label_forward_ok': label_forward_ok, 'cut': cut,
          'window_len': window_len, 'h': h}


# ============================ 自检 ============================

def seq_split_table(s: SequenceSampleSet) -> pd.DataFrame:
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
  ap = argparse.ArgumentParser(description='ml 序列样本层自检: 滑窗构造 + 无前视检验')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--prep', default=cfg.FEATURE_PREP,
                  choices=('raw', 'z', 'rank', 'mad_z', 'q_z'))
  ap.add_argument('--window-len', type=int, default=cfg.SEQ_WINDOW, help='窗口长度 L')
  ap.add_argument('--stride', type=int, default=cfg.SEQ_STRIDE, help='信号日起步')
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  ap.add_argument('--source', default=cfg.SEQ_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES),
                  help='序列特征来源: factor / ta / both(默认 cfg.SEQ_FEATURE_SOURCE)')
  ap.add_argument('--check', action='store_true', help='做截断一致性(无前视)检验')
  ap.add_argument('--cut', default=None, help='检验用的截断日(None = 历史中点)')
  a = ap.parse_args()

  log = cfg.get_logger('ml.seq')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')
  log.info(f'[口径] L={a.window_len}, stride={a.stride}, h={a.h}, label={a.method}, '
           f'prep={a.prep}, source={a.source}')

  s = build_sequences(ds, None, a.h, a.method, a.prep, a.min_cs,
                      window_len=a.window_len, stride=a.stride, source=a.source, log=log)

  log.info('\n[时序切分] 各窗口样本量(红线: 只在 IS 定参; 与 ridge/gbdt/mlp 同口径)')
  log.info('\n' + seq_split_table(s).to_string(index=False))

  log.info('\n[样本示例] 前 2 行 meta + y, 及其窗口末端(信号日)的 3 个特征值')
  for r in range(min(2, s.n_samples)):
    head = s.meta.iloc[r].to_dict()
    head['y'] = round(float(s.y[r]), 3)
    for j in range(min(3, s.n_features)):
      head[('t=' + s.features[j])] = round(float(s.X[r, -1, j]), 3)
    log.info(f'  {head}')

  if a.check:
    res = verify_no_lookahead(ds, a.window_len, a.h, a.prep, a.min_cs, a.cut)
    log.info(f'\n[无前视检验] 截断日 cut={res["cut"]}, L={res["window_len"]}, h={res["h"]}')
    log.info(f'  以 cut 结尾的窗口: 全历史 vs 截断后逐值相同 = {res["win_equal"]} '
             f'(不同格数 {res["n_diff"]})')
    log.info(f'  标签确实用未来(截断后 cut 当日标签全 NaN, 全历史版本有效) = '
             f'{res["label_forward_ok"]}')
    log.info('[结论] 特征无前视 + 标签用未来, 序列构造正确' if res['ok']
             else '[结论] 异常, 需排查')

  log.info('\n[下一步] predict.py --model gru|lstm: 用 SequenceSampleSet 拟合序列模型 -> 分数宽表 '
           '-> factor.backtest, 与 ridge/gbdt/mlp 同引擎对比.')


if __name__ == '__main__':
  main()
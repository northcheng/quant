# -*- coding: utf-8 -*-
"""
ml.models.linear — 线性基线(Ridge, 第4步之一)
===============================================
职责: 用最"笨"的模型(线性 + L2)给整条 ml 管线立一个参照物.

--- 为什么先做 Ridge, 而不是直接上 GBDT/DL ---
  模型越复杂, 越容易把"数据/标签/切分"里的毛病掩盖成"模型很强". Ridge 几乎没有可调
  空间(只一个 alpha); 若它连基线都跑不出正 RankIC, 那问题多半在前三步(特征 / 标签 /
  对齐)而不是模型. 先有线性参照物, 后面对比 GBDT/DL 的增量才有意义.

--- 它吃什么 ---
  输入是第3步的 SampleSet(X 已在特征层做过逐日截面 rank 标准化, 取值 [-1,1]).
  X 里仍可能有个别 NaN(该标的当日该特征缺失): 树模型能原生吃缺失, 线性模型不行,
  必须先填补. **填补统计量(每列中位数)只在本模型被 fit 的那段样本上估计** —— 约定
  fit 只喂 IS, 故中位数只从 IS 学, 不会偷看 OOS(红线).

--- 它吐什么 ---
  predict() 返回 (date × symbol) 的截面分数宽表 —— 与 factor 的 composite 同构,
  可直接丢进 factor.evaluate.ic_series / factor.backtest.run_engine.

--- 数学 ---
  带截距的岭回归闭式解(纯 numpy, 不引入 sklearn):
    填补后 Xf = X 按列中位数补缺; Xc = Xf - mean_fit; yc = y - y_mean_fit
    w = (Xc^T Xc + alpha * I)^-1 (Xc^T yc)      # 截距经中心化吸收, 不参与惩罚
    score = (Xf - mean_fit) @ w + y_mean_fit

自检:
  cd ~/git && python -m quant.ml.models.linear --pool etf_3x
  cd ~/git && python -m quant.ml.models.linear --pool etf_3x --check
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor import evaluate as ev
from quant.factor.data import window as _win

from quant.ml import config as cfg
from quant.ml.data import load
from quant.ml.dataset import SampleSet, assemble, holdout_split


# ============================ 缺失填补(统计量只在 fit 样本上算) ============================

def _col_fill(X: np.ndarray, strategy: str) -> np.ndarray:
  """按列算填补值(median/mean; 全 NaN 的列填 0). 只应由 fit 样本调用."""
  if strategy == 'zero':
    return np.zeros(X.shape[1], dtype=float)
  fn = np.nanmedian if strategy == 'median' else np.nanmean
  with np.errstate(invalid='ignore'):
    v = fn(X.astype(float), axis=0)
  return np.nan_to_num(v, nan=0.0)


def _apply_fill(X: np.ndarray, fill: np.ndarray) -> np.ndarray:
  """用已定的 fill 值补缺(不重新估计统计量) -> 新数组(不改原数组)."""
  out = np.array(X, dtype=float, copy=True)
  bad = ~np.isfinite(out)
  if bad.any():
    out[bad] = np.take(fill, np.nonzero(bad)[1])
  return out


def scores_to_wide(sample: SampleSet, scores: np.ndarray) -> pd.DataFrame:
  """一维分数 -> (date × symbol) 宽表(借助 sample.meta 的两列索引)."""
  idx = pd.MultiIndex.from_frame(sample.meta[['date', 'symbol']])
  return pd.Series(np.asarray(scores, dtype=float), index=idx).unstack('symbol')


# ============================ Ridge 基线 ============================

class RidgeModel:
  """
  带截距的岭回归基线(闭式解, 纯 numpy).

  :param alpha: L2 强度(越大越收缩)
  :param fill: 缺失填补策略('median'/'mean'/'zero')
  """

  def __init__(self, alpha: float = cfg.RIDGE_ALPHA, fill: str = cfg.FILL_STRATEGY):
    self.alpha = float(alpha)
    self.fill = fill
    self.features = None
    self.fill_ = None
    self.x_mean_ = None
    self.y_mean_ = None
    self.coef_ = None
    self.n_fit_ = 0

  # ---- fit: 只吃传入样本(约定 IS) ----
  def fit(self, sample: SampleSet, valid: SampleSet = None, log=None,
          valid_fwd: pd.DataFrame = None, min_cs: int = fcfg.MIN_CS,
          criterion: str = 'rank_ic') -> 'RidgeModel':
    """
    在样本上估计 填补值 / 列均值 / 系数. 只喂 IS 即不偷看未来.

    valid / valid_fwd / min_cs / criterion 仅为与 gbdt/neural 统一 fit 签名而保留;
    Ridge 是闭式解、无早停, 这些参数一律忽略.
    """
    X, y = sample.X, sample.y
    self.features = list(sample.features)
    self.fill_ = _col_fill(X, self.fill)
    Xf = _apply_fill(X, self.fill_)
    self.x_mean_ = Xf.mean(axis=0)
    self.y_mean_ = float(y.mean())
    Xc = Xf - self.x_mean_                     # 中心化 -> 截距被吸收
    yc = y - self.y_mean_
    d = Xc.shape[1]
    A = Xc.T @ Xc + self.alpha * np.eye(d)
    self.coef_ = np.linalg.solve(A, Xc.T @ yc)
    self.n_fit_ = int(X.shape[0])
    if log is not None:
      log.info(f'[ridge] fit: n={self.n_fit_:,}, F={d}, alpha={self.alpha:g}, '
               f'fill={self.fill}, 非零系数 {int((np.abs(self.coef_) > 1e-12).sum())}')
    return self

  # ---- predict: 用 fit 定下的参数打分, 返回截面分数宽表 ----
  def predict(self, sample: SampleSet) -> pd.DataFrame:
    if self.coef_ is None:
      raise RuntimeError('请先 fit')
    if list(sample.features) != self.features:
      raise ValueError('特征顺序与 fit 时不一致')
    Xf = _apply_fill(sample.X, self.fill_)
    scores = (Xf - self.x_mean_) @ self.coef_ + self.y_mean_
    return scores_to_wide(sample, scores)

  def coef_table(self) -> pd.DataFrame:
    """系数表(按绝对值降序), 用于看模型到底在依赖哪些特征."""
    return (pd.DataFrame({'feature': self.features, 'coef': self.coef_})
            .assign(abss=lambda d: d['coef'].abs())
            .sort_values('abss', ascending=False)
            .drop(columns='abss').reset_index(drop=True))


# ============================ 评估小工具(复用 factor.evaluate) ============================

def eval_scores(score_w: pd.DataFrame, fwd: pd.DataFrame, window: str,
                h: int, min_cs: int = fcfg.MIN_CS) -> dict:
  """
  分数宽表在指定窗口的 IC / RankIC(复用 factor.evaluate 口径).

  :param score_w: 截面分数宽表(date × symbol)
  :param fwd: 可交易前向收益宽表
  :param window: 'IS' / 'OOS' / 'recent' / 'full'
  :param h: 持有天数(算 NW-HAC lag)
  :param min_cs: 每日最少配对数
  :returns: dict(window, n_days, rank_ic, ic, rank_icir, nw_t, pos_rate)
  """
  start, end = cfg.window_range(window)
  s, f = _win(score_w, start, end), _win(fwd, start, end)
  ic_s = ev.ic_summary(ev.ic_series(s, f, 'spearman', min_cs), nw_lag=max(h - 1, 1))
  ic_p = ev.ic_summary(ev.ic_series(s, f, 'pearson', min_cs), nw_lag=max(h - 1, 1))
  return {'window': window, 'n_days': ic_s['n_days'], 'rank_ic': ic_s['ic_mean'],
          'ic': ic_p['ic_mean'], 'rank_icir': ic_s['ic_ir'],
          'nw_t': ic_s['nw_t'], 'pos_rate': ic_s['pos_rate']}


def main():
  ap = argparse.ArgumentParser(description='ml 模型层自检: Ridge 基线(fit IS -> 评 OOS)')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--prep', default=cfg.FEATURE_PREP,
                  choices=('raw', 'z', 'rank', 'mad_z', 'q_z'))
  ap.add_argument('--source', default=cfg.DEFAULT_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES),
                  help='特征来源: factor / ta / both')
  ap.add_argument('--alpha', type=float, default=cfg.RIDGE_ALPHA)
  ap.add_argument('--fill', default=cfg.FILL_STRATEGY,
                  choices=('median', 'mean', 'zero'))
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  ap.add_argument('--valid-frac', type=float, default=cfg.GBDT_VALID_FRAC,
                  help='与其余模型同一时序切分: ridge 只在训练块 tr 上拟合(无早停)')
  ap.add_argument('--top', type=int, default=12, help='打印系数绝对值最大的前 N 个特征')
  ap.add_argument('--check', action='store_true', help='加做 oracle/random/shuffle 对照')
  a = ap.parse_args()

  log = cfg.get_logger('ml.models.linear')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  s = assemble(ds, None, a.h, a.method, a.prep, a.min_cs,
               feature_source=a.source, log=log)
  fwd = ev.forward_return(ds, a.h)                 # 全历史(标签必须用未来)
  log.info(f'[口径] h={a.h}, label={a.method}, prep={a.prep}, source={a.source}, '
           f'alpha={a.alpha:g}, fill={a.fill}')

  is_s, oos_s = s.split('IS'), s.split('OOS')
  # 与 gbdt/mlp/gru/lstm 同一时序切分(embargo=h+1): ridge 无早停, 但只在训练块 tr 上拟合,
  # 使各模型训练样本量/时段一致 -> 净值可比(P0-1).
  tr, va, info = holdout_split(is_s, a.valid_frac, embargo=a.h + 1)
  log.info(f'[拟合块] train <= {info["train_end"]:%Y-%m-%d} (n={info["n_train"]:,}), '
           f'embargo={info["embargo"]} 天 (ridge 无早停, 只用 tr 拟合)')
  model = RidgeModel(a.alpha, a.fill).fit(tr, log=log)

  pred = model.predict(s)
  tab = pd.DataFrame([eval_scores(pred, fwd, w, a.h, a.min_cs)
                      for w in ('IS', 'OOS', 'recent')])
  log.info('\n[基线表现] Ridge 分数 vs 可交易前向收益(复用 factor.evaluate 的 IC 口径)')
  log.info('\n' + tab.to_string(index=False))
  log.info('[说明] IS 是 fit 段, 其 RankIC 必然偏乐观; OOS/recent 未参与任何拟合, '
           '才是可对外汇报的诚实数字.')

  ct = model.coef_table()
  log.info(f'\n[系数] 绝对值最大的前 {a.top} 个特征(模型在依赖谁):\n'
           f'{ct.head(a.top).to_string(index=False)}')
  n_ta = int(sum(1 for f in ct.head(a.top)['feature'] if f not in _factor_names()))
  if n_ta:
    log.info(f'[提示] 前 {a.top} 大系数里含 {n_ta} 个非 factor 特征(TA 列) —— '
             f'TA 因果性未验证, 若它主导打分, OOS 结论需格外谨慎.')

  if a.check:
    # 1) oracle: 拿前向收益本身当分数 -> RankIC 应 ~1.0(校准评估口径)
    # 2) random: 随机分数(对齐 pred 的 NaN 格局) -> 应 ~0.0
    # 3) shuffle: 打乱 IS 标签再 fit -> OOS RankIC 应塌回 ~0(排除"harness 自带泄漏")
    rng = np.random.default_rng(0)
    rows = [dict(case='oracle(=fwd)', **eval_scores(fwd, fwd, 'IS', a.h, a.min_cs)),
            dict(case='random', **eval_scores(
                _random_like(pred, rng), fwd, 'IS', a.h, a.min_cs))]
    sh = RidgeModel(a.alpha, a.fill)
    sh.features = list(tr.features)
    sh.fill_ = _col_fill(tr.X, a.fill)
    Xf = _apply_fill(tr.X, sh.fill_)
    sh.x_mean_, sh.y_mean_ = Xf.mean(axis=0), float(tr.y.mean())
    y_sh = rng.permutation(tr.y)
    Xc = Xf - sh.x_mean_
    sh.coef_ = np.linalg.solve(Xc.T @ Xc + a.alpha * np.eye(Xc.shape[1]),
                               Xc.T @ (y_sh - sh.y_mean_))
    rows.append(dict(case='shuffle-label', **eval_scores(sh.predict(s), fwd, 'OOS',
                                                         a.h, a.min_cs)))
    log.info(f'\n[对照] 校准评估口径(IS) + 打乱标签的 OOS')
    log.info('\n' + pd.DataFrame(rows).to_string(index=False))
    orc, rnd = rows[0]['rank_ic'], rows[1]['rank_ic']
    shf = rows[2]['rank_ic']
    log.info(f'[结论] oracle≈1 且 random≈0 判评估口径正确; '
             f'shuffle 的 OOS RankIC={shf} 应≈0(否则 harness 有泄漏) -> '
             + ('通过' if abs(rnd) < 0.1 and abs(shf) < 0.1 else '异常, 需排查'))

  log.info('\n[下一步] train.py 做 IS 内时序 walk-forward 选 alpha; 随后接 predict.py '
           '出分数宽表 -> factor.backtest 出净值.')


def _factor_names() -> set:
  """factor 注册表里的因子名集合(用于区分 TA 列)."""
  from quant.factor import factor as fct
  return set(fct.REGISTRY)


def _random_like(pred: pd.DataFrame, rng) -> pd.DataFrame:
  """生成与 pred 同形、同 NaN 格局的随机分数(随机对照)."""
  return pd.DataFrame(rng.standard_normal(pred.shape), index=pred.index,
                      columns=pred.columns).where(pred.notna())


if __name__ == '__main__':
  main()
# -*- coding: utf-8 -*-
"""
ml.models.gbdt — 梯度提升树(LightGBM, 第4步之二)
================================================
职责: 在 Ridge 线性基线之上, 引入一个**非线性 + 原生吃缺失**的模型做对比.
      两模型的下游完全共用(同特征 / 同标签 / 同切分 / 同回测引擎), 差异只在模型本身.

--- 与 Ridge 的关键差异 ---
  1. 非线性: 树能表达特征间交互 / 阈值切分, 线性模型只能给加权和;
  2. 原生吃 NaN: LightGBM 自动把缺失值分到增益更大的一侧 -> **不需要缺失填补**,
     也就不存在"填补统计量偷看未来"的红线问题(这是 tree 相对 linear 的便利之处);
  3. 可解释: 给出 split/增益重要性, 但不如线性系数直白.

--- 它吃什么 / 吐什么 ---
  与 Ridge 完全一致: 吃第3步 SampleSet(X 已逐日截面 rank), 吐 (date × symbol) 分数宽表,
  可直接丢进 factor.evaluate.ic_series / factor.backtest.run_engine.

--- 红线 ---
  1. fit 只喂 IS(约定), 早停验证块也在 IS 内部用**时序**切分(见 dataset.holdout_split,
     带 embargo = h+1), 绝不用 OOS 选轮数/调参;
  2. 下面是**未经 IS 内寻优的默认超参**(config.GBDT_PARAMS), 仅用于跑通骨架与做基线对比,
     报告里必须标注"未调参", 否则与调过参的模型对比会失真. 后续需在 IS 内 walk-forward 寻优.

自检:
  cd ~/git && python -m quant.ml.models.gbdt --pool etf_3x
  cd ~/git && python -m quant.ml.models.gbdt --pool etf_3x --check
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor import evaluate as ev

from quant.ml import config as cfg
from quant.ml.data import load
from quant.ml.dataset import SampleSet, assemble, holdout_split
from quant.ml.models.linear import eval_scores, _random_like


# ============================ GBDT 模型 ============================

def _rank_ic_feval(valid: SampleSet, valid_fwd: pd.DataFrame, min_cs: int):
  """
  构造 LightGBM 自定义评估函数: 在验证块上按**日**算 Spearman -> RankIC 均值(越大越好).

  ⚠ 走的是 sklearn API 的 eval_metric 口径(func(y_true, y_pred) -> (name, value,
  is_higher_better)), 与原生 lgb.train 的 feval(preds, dataset) 不同 —— 后者不是
  LGBMRegressor.fit 的合法参数. 这里忽略 y_true(样本 y 是截面 rank 标签),
  只用预对齐的**前向收益** row_fwd 算 IC.

  行序约定 (date-major, symbol-minor)(dataset.assemble 保证): 连续同日的行构成一个截面.
  低信噪比下 MSE 最小点不对应排序能力最强点, 故早停判据用 RankIC(与回测选股目标一致).
  """
  d = valid.meta['date'].to_numpy()
  syms = valid.meta['symbol'].to_numpy()
  # 逐行对齐的前向收益(标签必须用未来; 只在验证块上, 不越 OOS)
  dates_u = list(valid.meta['date'].drop_duplicates())
  syms_u = list(valid.meta['symbol'].drop_duplicates())
  fw = valid_fwd.reindex(index=dates_u, columns=syms_u).to_numpy(dtype=float)
  di = {x: i for i, x in enumerate(dates_u)}
  si = {x: i for i, x in enumerate(syms_u)}
  row_fwd = fw[[di[x] for x in d], [si[x] for x in syms]]
  # 连续同日块边界(不依赖日期是否全局有序)
  bnd = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])

  def _metric(y_true, y_pred):
    p = np.asarray(y_pred, dtype=float)
    ics = []
    for a_, b_ in zip(bnd[:-1], bnd[1:]):
      y = row_fwd[a_:b_]
      ok = np.isfinite(p[a_:b_]) & np.isfinite(y)
      if int(ok.sum()) < min_cs:
        continue
      pr = pd.Series(p[a_:b_][ok]).rank().to_numpy()
      yr = pd.Series(y[ok]).rank().to_numpy()
      if pr.std() < 1e-12 or yr.std() < 1e-12:
        continue
      ics.append(float(np.corrcoef(pr, yr)[0, 1]))
    return 'rank_ic', (float(np.mean(ics)) if ics else 0.0), True

  return _metric


class GBDTModel:
  """
  LightGBM 回归器(截面 rank 标签 -> 分数).

  :param params: LightGBM 参数字典(None = config.GBDT_PARAMS)
  :param early_stop: 早停轮数(需 fit 时给 valid 才生效)
  """

  def __init__(self, params: dict = None, early_stop: int = cfg.GBDT_EARLY_STOP):
    self.params = dict(cfg.GBDT_PARAMS if params is None else params)
    self.early_stop = int(early_stop)
    self.features = None
    self.model_ = None
    self.n_fit_ = 0
    self.best_iteration_ = None
    self.criterion_ = None

  # ---- fit: 只吃传入样本(约定 IS); valid 也来自 IS 内部时序切分 ----
  def fit(self, sample: SampleSet, valid: SampleSet = None, log=None,
          valid_fwd: pd.DataFrame = None, min_cs: int = fcfg.MIN_CS,
          criterion: str = 'rank_ic') -> 'GBDTModel':
    """
    在样本上训练. 若给 valid(IS 内时序验证块), 则用早停定树数.

    :param sample: 训练样本(约定 IS)
    :param valid: 早停验证样本(IS 内部时序切分; None = 用固定 n_estimators)
    :param log: 可选 logger
    :param valid_fwd: 可交易前向收益宽表(criterion='rank_ic' 时必给)
    :param min_cs: 截面最少标的数(算 RankIC 用)
    :param criterion: 早停判据 'rank_ic'(默认) / 'l2'; 'rank_ic' 需 valid_fwd, 缺失时自动回退 l2
    """
    try:
      import lightgbm as lgb
    except ImportError as e:                                  # 环境缺库时给出清晰指引
      raise ImportError('需要 lightgbm: pip install lightgbm') from e

    self.features = list(sample.features)
    p = dict(self.params)
    n_est = int(p.pop('n_estimators', 500))                   # 轮数单独提出来(早停会覆盖)
    # 早停判据: 默认 rank_ic(需 valid + valid_fwd); valid_fwd 缺失时回退 l2.
    use_ric = (valid is not None and criterion == 'rank_ic' and valid_fwd is not None)
    if use_ric:
      p['metric'] = 'None'          # 关掉内置 metric, 只让自定义 rank_ic 参与早停/日志
    self.model_ = lgb.LGBMRegressor(n_estimators=n_est, **p)

    if valid is not None:
      if list(valid.features) != self.features:
        raise ValueError('valid 与 train 的特征顺序不一致')
      if use_ric:
        feval = _rank_ic_feval(valid, valid_fwd, int(min_cs))
        self.model_.fit(
            sample.X, sample.y,
            eval_set=[(valid.X, valid.y)], eval_metric=feval,
            callbacks=[lgb.early_stopping(self.early_stop, verbose=False,
                                          first_metric_only=True),
                       lgb.log_evaluation(0)])
      else:
        self.model_.fit(
            sample.X, sample.y,
            eval_set=[(valid.X, valid.y)], eval_metric='l2',
            callbacks=[lgb.early_stopping(self.early_stop, verbose=False),
                       lgb.log_evaluation(0)])
      self.criterion_ = 'rank_ic' if use_ric else 'l2'
      self.best_iteration_ = int(getattr(self.model_, 'best_iteration_', n_est) or n_est)
    else:
      self.model_.fit(sample.X, sample.y,
                      callbacks=[lgb.log_evaluation(0)])
      self.best_iteration_ = n_est
      self.criterion_ = 'off'

    self.n_fit_ = int(sample.X.shape[0])
    if log is not None:
      log.info(f'[gbdt] fit: n={self.n_fit_:,}, F={len(self.features)}, '
               f'best_iter={self.best_iteration_}, '
               f'early={"on" if valid is not None else "off"}, '
               f'criterion={self.criterion_}, '
               f'lr={p.get("learning_rate")}, leaves={p.get("num_leaves")}')
    return self

  # ---- predict: 用训练好的树打分, 返回截面分数宽表 ----
  def predict(self, sample: SampleSet) -> pd.DataFrame:
    if self.model_ is None:
      raise RuntimeError('请先 fit')
    if list(sample.features) != self.features:
      raise ValueError('特征顺序与 fit 时不一致')
    from quant.ml.models.linear import scores_to_wide
    scores = self.model_.predict(sample.X)
    return scores_to_wide(sample, scores)

  def feature_importance(self, top: int = None) -> pd.DataFrame:
    """按 split 次数的重要性表(降序)."""
    imp = self.model_.feature_importances_                   # LightGBM 默认 = split 次数
    df = (pd.DataFrame({'feature': self.features, 'importance': imp})
          .sort_values('importance', ascending=False).reset_index(drop=True))
    return df if top is None else df.head(top)


# ============================ 自检 ============================

def main():
  ap = argparse.ArgumentParser(description='ml 模型层自检: LightGBM(fit IS -> 评 OOS)')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--prep', default=cfg.FEATURE_PREP,
                  choices=('raw', 'z', 'rank', 'mad_z', 'q_z'))
  ap.add_argument('--source', default=cfg.DEFAULT_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES),
                  help='特征来源: factor / ta / both')
  ap.add_argument('--valid-frac', type=float, default=cfg.GBDT_VALID_FRAC,
                  help='IS 内部时序早停验证块占比')
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  ap.add_argument('--top', type=int, default=12, help='打印重要性前 N 个特征')
  ap.add_argument('--check', action='store_true', help='加做 oracle/random/shuffle 对照')
  a = ap.parse_args()

  log = cfg.get_logger('ml.models.gbdt')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  s = assemble(ds, None, a.h, a.method, a.prep, a.min_cs,
               feature_source=a.source, log=log)
  fwd = ev.forward_return(ds, a.h)                 # 全历史(标签必须用未来)
  log.info(f'[口径] h={a.h}, label={a.method}, prep={a.prep}, source={a.source}, '
           f'early_stop={cfg.GBDT_EARLY_STOP}, valid_frac={a.valid_frac}')
  log.info('[提示] 当前用的是 config.GBDT_PARAMS 的**默认超参(未在 IS 内寻优)**, '
           '仅作骨架跑通与基线对比.')

  is_s, oos_s = s.split('IS'), s.split('OOS')
  tr, va, info = holdout_split(is_s, a.valid_frac, embargo=a.h + 1)   # IS 内时序早停块
  log.info(f'[早停] IS 内时序切分: train <= {info["train_end"]:%Y-%m-%d} '
           f'(n={info["n_train"]:,}), embargo={info["embargo"]} 天, '
           f'valid {info["valid_start"]:%Y-%m-%d}~{info["valid_end"]:%Y-%m-%d} '
           f'(n={info["n_valid"]:,})')

  model = GBDTModel().fit(tr, valid=va, log=log, valid_fwd=fwd,
                          min_cs=a.min_cs, criterion='rank_ic')

  pred = model.predict(s)
  tab = pd.DataFrame([eval_scores(pred, fwd, w, a.h, a.min_cs)
                      for w in ('IS', 'OOS', 'recent')])
  log.info('\n[基线表现] GBDT 分数 vs 可交易前向收益(复用 factor.evaluate 的 IC 口径)')
  log.info('\n' + tab.to_string(index=False))
  log.info('[说明] IS 含 fit 段, 其 RankIC 偏乐观; OOS/recent 未参与任何拟合, '
           '才是可对外汇报的诚实数字.')

  imp = model.feature_importance(a.top)
  log.info(f'\n[重要性] split 次数前 {a.top} 个特征(模型在依赖谁):\n'
           f'{imp.to_string(index=False)}')
  n_ta = int(sum(1 for f in imp['feature'] if f not in _factor_names()))
  if n_ta:
    log.info(f'[提示] 前 {a.top} 大重要性里含 {n_ta} 个非 factor 特征(TA 列) —— '
             f'TA 因果性未验证, 若它主导打分, OOS 结论需格外谨慎.')

  if a.check:
    # 1) oracle: 前向收益本身当分数 -> RankIC ~1(校准评估口径)
    # 2) random: 随机分数(对齐 pred 的 NaN 格局) -> ~0
    # 3) shuffle: 打乱 IS 标签再训 -> OOS RankIC 应塌回 ~0(排除 harness 自带泄漏)
    rng = np.random.default_rng(0)
    rows = [dict(case='oracle(=fwd)', **eval_scores(fwd, fwd, 'IS', a.h, a.min_cs)),
            dict(case='random', **eval_scores(
                _random_like(pred, rng), fwd, 'IS', a.h, a.min_cs))]
    sh = GBDTModel()
    sh.features = list(tr.features)
    sh_y = rng.permutation(tr.y)                    # 打乱标签(破坏 X->y 关系)
    import lightgbm as lgb
    sh.model_ = lgb.LGBMRegressor(n_estimators=model.best_iteration_,
                                  **{k: v for k, v in cfg.GBDT_PARAMS.items()
                                     if k != 'n_estimators'})
    sh.model_.fit(tr.X, sh_y, callbacks=[lgb.log_evaluation(0)])
    rows.append(dict(case='shuffle-label', **eval_scores(sh.predict(s), fwd, 'OOS',
                                                         a.h, a.min_cs)))
    log.info(f'\n[对照] 校准评估口径(IS) + 打乱标签的 OOS')
    log.info('\n' + pd.DataFrame(rows).to_string(index=False))
    rnd, shf = rows[1]['rank_ic'], rows[2]['rank_ic']
    log.info(f'[结论] oracle≈1 且 random≈0 判评估口径正确; '
             f'shuffle 的 OOS RankIC={shf} 应≈0(否则 harness 有泄漏) -> '
             + ('通过' if abs(rnd) < 0.1 and abs(shf) < 0.1 else '异常, 需排查'))

  log.info('\n[下一步] predict.py --model gbdt 出分数宽表 -> factor.backtest 出净值, '
           '与 ridge 在同一引擎下对比.')


def _factor_names() -> set:
  """factor 注册表里的因子名集合(用于区分 TA 列)."""
  from quant.factor import factor as fct
  return set(fct.REGISTRY)


if __name__ == '__main__':
  main()
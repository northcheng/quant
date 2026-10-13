# -*- coding: utf-8 -*-
"""
ml.train — 定参层: IS 内时序 walk-forward 选 alpha(第4步之二)
============================================================
职责: 只在 IS 内, 用**时序 walk-forward**的验证块挑 Ridge 的 L2 强度 alpha.
      OOS / recent 完全不参与挑选, 只在最后拿定好的 alpha 做一次诚实汇报.

--- 红线: 为什么必须"时序"+"embargo" ---
  1. 只在 IS 上定参: OOS/recent 一行都不许进验证(否则 OOS 不再是样本外).
  2. 验证块必须晚于训练块(扩张窗口 walk-forward), 不能随机抽取.
  3. 训练块的标签 span t+1..t+1+h(未来), 会与紧邻验证块的样本重叠 —— 必须留
     embargo = h+1 个交易日: 训练块最后一天 <= 验证块首日 - (h+1). 否则验证集
     偷偷含了与训练重叠的未来收益, 验证 RankIC 会被抬高(隐性前视).

--- 选参规则 ---
  对每个 alpha 算各折验证块的日均 RankIC, 取折均值; 再用 1-SE 规则挑选:
  在"折均值 >= 最优均值 - 1×标准误"的候选里, 选 **最大 alpha**(最保守/最不易过拟合).
  避免为了 0.001 的 RankIC 增量去选几乎不可辨识的小 alpha.

  注: 本脚本是 Ridge 专属的 alpha 网格寻优. GBDT 的超参(leaves/lr/... )寻优是后续工作,
      当前 GBDT 用的是 config.GBDT_PARAMS 的默认值(未寻优), 仅靠 IS 内时序早停定树数.

自检:
  cd ~/git && python -m quant.ml.train --pool etf_3x
  cd ~/git && python -m quant.ml.train --pool etf_3x --source factor
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor import evaluate as ev

from quant.ml import config as cfg
from quant.ml.data import load
from quant.ml.dataset import SampleSet, assemble
from quant.ml.models.linear import RidgeModel, eval_scores


def walk_forward_folds(is_s: SampleSet, n_folds: int, embargo: int,
                       min_train_frac: float = 0.3):
  """
  在 IS 内生成扩张窗口的 (train_mask, valid_mask) 折列表.

  :param is_s: IS 样本集
  :param n_folds: 折数(验证块个数)
  :param embargo: 训练/验证之间的隔离交易日数(应 = h+1)
  :param min_train_frac: 首折训练块至少占 IS 的比例
  :returns: list[dict(fold, train_end, valid_start, valid_end, n_train, n_valid)]
  """
  dates = is_s.dates
  n = len(dates)
  pos = is_s.meta['date'].map({d: i for i, d in enumerate(dates)}).to_numpy()
  first = max(int(n * min_train_frac), embargo + 30)
  block = max((n - first) // n_folds, 1)

  folds = []
  for i in range(n_folds):
    v0 = first + i * block
    v1 = n if i == n_folds - 1 else v0 + block
    t_end = v0 - embargo                       # 训练块截止(独占), 留出 embargo
    if t_end <= 0 or v1 <= v0:
      continue
    tr = pos < t_end
    va = (pos >= v0) & (pos < v1)
    if tr.sum() == 0 or va.sum() == 0:
      continue
    folds.append(dict(fold=i + 1, train_end=dates[t_end - 1], valid_start=dates[v0],
                      valid_end=dates[v1 - 1], n_train=int(tr.sum()),
                      n_valid=int(va.sum()), _tr=tr, _va=va))
  return folds


def fold_rank_ic(tr: SampleSet, va: SampleSet, fwd: pd.DataFrame, alpha: float,
                 fill: str, min_cs: int) -> tuple:
  """
  单折: 用 tr 拟合, 在 va 上算日均 RankIC.

  :returns: (valid_rank_ic, train_rank_ic) —— 均为日均 spearman IC
  """
  m = RidgeModel(alpha, fill).fit(tr)
  va_pred = m.predict(va)
  tr_pred = m.predict(tr)
  ic_va = ev.ic_series(va_pred, fwd.reindex(index=va_pred.index, columns=va_pred.columns),
                       'spearman', min_cs)
  ic_tr = ev.ic_series(tr_pred, fwd.reindex(index=tr_pred.index, columns=tr_pred.columns),
                       'spearman', min_cs)
  return float(np.nanmean(ic_va)), float(np.nanmean(ic_tr))


def select_alpha(is_s: SampleSet, fwd: pd.DataFrame, alphas: list, n_folds: int,
                 embargo: int, fill: str, min_cs: int, log=None) -> dict:
  """
  IS 内 walk-forward 网格搜索 alpha(1-SE 规则).

  :returns: dict(best, table, folds)
  """
  folds = walk_forward_folds(is_s, n_folds, embargo)
  if not folds:
    raise RuntimeError('IS 太短, 无法切出 walk-forward 折; 请减小 --folds')
  rows = []
  for al in alphas:
    va_ics, tr_ics = [], []
    for f in folds:
      v, t = fold_rank_ic(is_s.select(f['_tr']), is_s.select(f['_va']), fwd, al,
                          fill, min_cs)
      va_ics.append(v)
      tr_ics.append(t)
    va = np.array(va_ics, dtype=float)
    rows.append(dict(alpha=al, valid_rank_ic=float(np.nanmean(va)),
                     valid_std=float(np.nanstd(va, ddof=1)) if len(va) > 1 else 0.0,
                     valid_min=float(np.nanmin(va)), train_rank_ic=float(np.nanmean(tr_ics))))
  tab = pd.DataFrame(rows)

  best = float(tab['valid_rank_ic'].max())
  se = float(tab.loc[tab['valid_rank_ic'].idxmax(), 'valid_std']) / np.sqrt(len(folds))
  cand = tab[tab['valid_rank_ic'] >= best - se]                 # 1-SE 候选带
  pick = float(cand['alpha'].max())                             # 带内取最大 alpha(最保守)

  if log is not None:
    f_tab = pd.DataFrame([{k: f[k] for k in
                           ('fold', 'train_end', 'valid_start', 'valid_end',
                            'n_train', 'n_valid')} for f in folds])
    log.info(f'[wf] {len(folds)} 折(扩张窗口, embargo={embargo} 交易日):')
    log.info('\n' + f_tab.to_string(index=False))
    log.info(f'\n[wf] 各 alpha 的折均值 RankIC(valid=样本内验证块, train=训练块):')
    log.info('\n' + tab.to_string(index=False))
    log.info(f'[wf] 最优均值 {best:.4f}, 1-SE 带内取最大 alpha -> 选中 alpha={pick:g}')
  return dict(best=pick, table=tab, folds=folds)


def main():
  ap = argparse.ArgumentParser(description='ml 定参层: IS 内 walk-forward 选 Ridge alpha')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--prep', default=cfg.FEATURE_PREP,
                  choices=('raw', 'z', 'rank', 'mad_z', 'q_z'))
  ap.add_argument('--source', default=cfg.DEFAULT_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES), help='特征来源: factor / ta / both')
  ap.add_argument('--fill', default=cfg.FILL_STRATEGY, choices=('median', 'mean', 'zero'))
  ap.add_argument('--folds', type=int, default=4, help='walk-forward 折数')
  ap.add_argument('--min-train-frac', type=float, default=0.3, help='首折训练块最小占比')
  ap.add_argument('--alphas', default='0.01,0.03,0.1,0.3,1,3,10,30,100',
                  help='逗号分隔的 alpha 网格')
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  a = ap.parse_args()

  log = cfg.get_logger('ml.train')
  alphas = [float(x) for x in a.alphas.split(',')]
  embargo = a.h + 1                                   # 训练/验证隔离交易日数

  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日')
  s = assemble(ds, None, a.h, a.method, a.prep, a.min_cs,
               feature_source=a.source, log=log)
  fwd = ev.forward_return(ds, a.h)
  log.info(f'[口径] h={a.h}, label={a.method}, prep={a.prep}, source={a.source}, '
           f'fill={a.fill}, F={s.n_features}, embargo={embargo}')

  is_s, oos_s = s.split('IS'), s.split('OOS')
  log.info(f'[窗口] IS 样本 {is_s.n_samples:,}({len(is_s.dates)} 天); '
           f'OOS {oos_s.n_samples:,}({len(oos_s.dates)} 天). 定参只看 IS.')

  sel = select_alpha(is_s, fwd, alphas, a.folds, embargo, a.fill, a.min_cs, log=log)
  best = sel['best']

  # ---- 用定好的 alpha 全 IS 重拟合, 做一次诚实汇报(OOS/recent 未参与任何挑选) ----
  log.info('\n[定参结果 vs 默认] 全 IS 拟合后在 IS/OOS/recent 的表现')
  rows = []
  for tag, al in ((f'选定 alpha={best:g}', best), (f'默认 alpha={cfg.RIDGE_ALPHA:g}',
                                                   cfg.RIDGE_ALPHA)):
    m = RidgeModel(al, a.fill).fit(is_s)
    pred = m.predict(s)
    for w in ('IS', 'OOS', 'recent'):
      r = eval_scores(pred, fwd, w, a.h, a.min_cs)
      rows.append(dict(方案=tag, 窗口=r['window'], n_days=r['n_days'],
                       RankIC=round(r['rank_ic'], 4), IC=round(r['ic'], 4),
                       NW_t=round(r['nw_t'], 2), 胜率=round(r['pos_rate'], 3)))
  log.info('\n' + pd.DataFrame(rows).to_string(index=False))
  log.info('[说明] IS 为 fit 段(偏乐观); OOS/recent 未参与定参, 是诚实数字.')

  log.info(f'\n[记录] 本次选定 alpha={best:g}(source={a.source}, h={a.h}, fill={a.fill}). '
           f'下一步 predict.py 用它出全历史分数宽表 -> factor.backtest 出净值.')


if __name__ == '__main__':
  main()
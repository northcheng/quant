# -*- coding: utf-8 -*-
"""
ml.models.neural — 多层感知机(PyTorch MLP, 第4步之三)
======================================================
职责: 在 Ridge(线性) / GBDT(树) 之外, 加一个 **神经网络** 截面模型, 回答同一个问题:
      "在完全相同的特征 / 标签 / 切分 / 回测口径下, DL 相比 GBDT 有没有增量?"

--- 为什么起步选"MLP 截面"而不是 RNN/Transformer ---
  当前 SampleSet 是扁平的 (date, symbol) 样本: 每行是一天的截面特征, 行间无显式时序结构.
  序列模型(按标的切窗)需要另建一套数据管道, 工程量大且结论难与现有两模型严格可比.
  MLP 直接吃同一张表, 改动最小, 能第一时间给出"DL 有没有增量"的可比数字.

--- 诚实的先验 ---
  这类"低信噪比截面表格回归"问题, 业界经验是 **DL 通常打不过 GBDT**(样本量小、特征已是
  人工因子、标签噪声大). 因此本模型定位是**对照臂**, 而非"必然更好". 若 OOS 不及 GBDT,
  那就是一个有价值的负结论, 也应如实汇报.

--- 它吃什么 / 吐什么 ---
  与 Ridge 完全一致: 吃第3步 SampleSet(X 已逐日截面 rank, 取值 [-1,1]), 吐 (date × symbol)
  分数宽表, 可直接丢进 factor.evaluate.ic_series / factor.backtest.run_engine.

--- 预处理(红线) ---
  MLP 不原生吃 NaN -> 先按列中位数填补; 再做列标准化(减均值除标准差); 标签也标准化.
  **全部统计量(填补值 / 均值 / 标准差)只在本模型被 fit 的那段样本上估计** —— 约定 fit 只喂
  IS, 故不偷看 OOS. 具体做法与 RidgeModel 相同(复用 _col_fill / _apply_fill).

--- 小模型 + 早停(为什么要克制) ---
  1. 容量刻意压小(默认 2 隐层 64→32 + dropout): 214 特征 / ~6 万样本, 大网络会秒过拟合;
  2. 早停: 在 IS **内部**用 dataset.holdout_split 做**时序**切分(带 embargo=h+1),
     绝不用 OOS 选轮数(红线); 单 seed, 不做集成, 先拿一个可比数字;
  3. 超参是 config.MLP_* 的**默认值(未在 IS 内寻优)**, 报告须标注"未调参".

自检:
  cd ~/git && python -m quant.ml.models.neural --pool etf_3x
  cd ~/git && python -m quant.ml.models.neural --pool etf_3x --check
  cd ~/git && python -m quant.ml.models.neural --pool etf_3x --search
     (在 IS 内部的早停验证块上做 lr×wd 小网格选参, 判据 = 验证块 RankIC; OOS 不参与)
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor import evaluate as ev

from quant.ml import config as cfg
from quant.ml.data import load
from quant.ml.dataset import SampleSet, assemble, holdout_split
from quant.ml.models.linear import (_apply_fill, _col_fill, _random_like,
                                    eval_scores, scores_to_wide)


# ============================ 网络结构 ============================

def _make_net(n_features: int, hidden, dropout: float):
  """构造 MLP: [Linear->ReLU->Dropout] * len(hidden) -> Linear(1). 延迟 import torch."""
  import torch.nn as nn

  layers = []
  d = n_features
  for h in hidden:
    layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(float(dropout))]
    d = int(h)
  layers += [nn.Linear(d, 1)]
  return nn.Sequential(*layers)


# ============================ MLP 模型 ============================

class MLPModel:
  """
  截面 MLP 回归器(截面 rank 标签 -> 分数), 仅 CPU.

  :param hidden: 隐层宽度元组(如 (64, 32))
  :param dropout: 每层 dropout 概率
  :param lr: Adam 学习率
  :param weight_decay: Adam 权重衰减(L2)
  :param epochs: 最多训练轮数(有 valid 时被早停截断)
  :param batch_size: mini-batch 大小
  :param patience: 早停耐心(连续多少轮验证损失不改善即停)
  :param seed: 随机种子(单 seed)
  :param fill: 缺失填补策略('median'/'mean'/'zero')
  """

  def __init__(self, hidden=cfg.MLP_HIDDEN, dropout: float = cfg.MLP_DROPOUT,
               lr: float = cfg.MLP_LR, weight_decay: float = cfg.MLP_WEIGHT_DECAY,
               epochs: int = cfg.MLP_EPOCHS, batch_size: int = cfg.MLP_BATCH_SIZE,
               patience: int = cfg.MLP_PATIENCE, seed: int = cfg.MLP_SEED,
               fill: str = cfg.FILL_STRATEGY):
    self.hidden = tuple(hidden)
    self.dropout = float(dropout)
    self.lr = float(lr)
    self.weight_decay = float(weight_decay)
    self.epochs = int(epochs)
    self.batch_size = int(batch_size)
    self.patience = int(patience)
    self.seed = int(seed)
    self.fill = fill
    self.features = None
    self.fill_ = None
    self.x_mean_ = None
    self.x_std_ = None
    self.y_mean_ = 0.0
    self.y_std_ = 1.0
    self.net_ = None
    self.n_fit_ = 0
    self.best_epoch_ = None
    self.best_val_ = None
    self.criterion_ = None

  # ---- 预处理: 统计量只在 fit 样本上估(红线) ----
  def _prepare(self, sample: SampleSet) -> np.ndarray:
    """按 fit 时定下的 fill/mean/std 把 X 变成网络输入(不重新估计统计量)."""
    Xf = _apply_fill(sample.X, self.fill_)
    return ((Xf - self.x_mean_) / self.x_std_).astype(np.float32)

  # ---- fit: 只吃传入样本(约定 IS); valid 也来自 IS 内部时序切分 ----
  def fit(self, sample: SampleSet, valid: SampleSet = None, log=None,
          valid_fwd: pd.DataFrame = None, min_cs: int = fcfg.MIN_CS,
          criterion: str = 'rank_ic') -> 'MLPModel':
    """
    在样本上训练. 若给 valid(IS 内时序验证块), 则按其判据做早停并回滚到最优轮.

    :param sample: 训练样本(约定 IS)
    :param valid: 早停验证样本(IS 内部时序切分; None = 训练固定 epochs)
    :param log: 可选 logger
    :param valid_fwd: 可交易前向收益宽表(criterion='rank_ic' 时必给)
    :param min_cs: 截面最少标的数(算 RankIC 用)
    :param criterion: 早停判据 'rank_ic'(默认, 验证 RankIC 最大) / 'mse'(验证 MSE 最小);
                      'rank_ic' 需 valid_fwd, 缺失时自动回退 mse
    """
    try:
      import torch
      import torch.nn as nn
    except ImportError as e:                                  # 环境缺库时给出清晰指引
      raise ImportError('需要 torch: pip install torch --index-url '
                        'https://download.pytorch.org/whl/cpu') from e

    self.features = list(sample.features)
    F = len(self.features)

    # 1) 估预处理统计量(只在 fit 样本上) + 标签标准化
    self.fill_ = _col_fill(sample.X, self.fill)
    Xf = _apply_fill(sample.X, self.fill_)
    self.x_mean_ = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    self.x_std_ = np.where(std < 1e-8, 1.0, std)              # 常数列避免除零
    Xs = self._prepare(sample)
    self.y_mean_ = float(sample.y.mean())
    ys_std = float(sample.y.std())
    self.y_std_ = ys_std if ys_std > 1e-8 else 1.0
    ys = ((sample.y - self.y_mean_) / self.y_std_).astype(np.float32)

    # 2) 验证块预处理(用训练统计量, 不重估)
    if valid is not None:
      if list(valid.features) != self.features:
        raise ValueError('valid 与 train 的特征顺序不一致')
      Xv = self._prepare(valid)
      yv = ((valid.y - self.y_mean_) / self.y_std_).astype(np.float32)

    # 3) 建模 + 训练(单 seed, 确定性)
    torch.manual_seed(self.seed)
    net = _make_net(F, self.hidden, self.dropout)
    opt = torch.optim.Adam(net.parameters(), lr=self.lr,
                           weight_decay=self.weight_decay)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(ys)
    n = Xt.shape[0]
    gen = torch.Generator().manual_seed(self.seed)

    # 早停判据(统一为"越大越好"的标量): rank_ic=验证块 RankIC; mse=验证 MSE 取负
    use_ric = (valid is not None and criterion == 'rank_ic' and valid_fwd is not None)
    self.criterion_ = ('rank_ic' if use_ric else 'mse') if valid is not None else 'off'
    best_state, best_score, best_epoch, bad = None, -np.inf, 0, 0
    ran = 0
    for ep in range(1, self.epochs + 1):
      ran = ep
      net.train()
      perm = torch.randperm(n, generator=gen)
      for i in range(0, n, self.batch_size):
        idx = perm[i:i + self.batch_size]
        opt.zero_grad()
        loss = lossf(net(Xt[idx]).squeeze(-1), yt[idx])
        loss.backward()
        opt.step()
      if valid is None:
        continue
      net.eval()
      with torch.no_grad():                                   # 验证块前向
        out = net(torch.from_numpy(Xv)).squeeze(-1)
      if use_ric:
        # 判据 = 验证块 RankIC(还原到标签尺度 -> 按日 Spearman); 越大越好
        score_w = scores_to_wide(valid, out.numpy() * self.y_std_ + self.y_mean_)
        score = _rank_ic_on(valid, score_w, valid_fwd, int(valid.horizon), min_cs)
      else:
        score = -float(lossf(out, torch.from_numpy(yv)).item())     # MSE 取负 -> 越大越好
      if score > best_score + 1e-6:                           # 改善 -> 存最优状态
        best_score, best_epoch, bad = score, ep, 0
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
      else:                                                   # 不改善 -> 累计耐心
        bad += 1
        if bad >= self.patience:
          break

    if valid is not None and best_state is not None:          # 回滚到验证最优轮
      net.load_state_dict(best_state)
    self.net_, self.best_epoch_, self.best_val_ = net, best_epoch, best_score
    self.net_.eval()
    self.n_fit_ = int(n)
    if log is not None:
      log.info(f'[mlp] fit: n={self.n_fit_:,}, F={F}, hidden={self.hidden}, '
               f'dropout={self.dropout:g}, lr={self.lr:g}, wd={self.weight_decay:g}, '
               f'epochs={ran}/{self.epochs}, '
               f'early={"on" if valid is not None else "off"}, criterion={self.criterion_}'
               + (f', best_epoch={best_epoch}, best_{self.criterion_}={best_score:.4f}'
                  if valid is not None else ''))
    return self

  # ---- predict: 用训练好的网络打分, 返回截面分数宽表 ----
  def predict(self, sample: SampleSet) -> pd.DataFrame:
    if self.net_ is None:
      raise RuntimeError('请先 fit')
    if list(sample.features) != self.features:
      raise ValueError('特征顺序与 fit 时不一致')
    import torch
    Xs = self._prepare(sample)
    with torch.no_grad():
      out = self.net_(torch.from_numpy(Xs)).squeeze(-1).numpy()
    scores = out * self.y_std_ + self.y_mean_                 # 还原到标签尺度
    return scores_to_wide(sample, scores)

  def input_importance(self, top: int = None) -> pd.DataFrame:
    """
    首层权重的列 L1 范数作为"输入特征重要性"的**粗略代理**(非严格归因).

    仅用于快速看模型在依赖谁; MLP 无 GBDT 那样的 split 重要性, 代理有局限, 仅供参考.
    """
    w = self.net_[0].weight.detach().numpy()                  # (hidden0, F)
    imp = np.abs(w).sum(axis=0)
    df = (pd.DataFrame({'feature': self.features, 'importance': imp})
          .sort_values('importance', ascending=False).reset_index(drop=True))
    return df if top is None else df.head(top)


# ============================ 自检 ============================

def _rank_ic_on(sample: SampleSet, score_w: pd.DataFrame, fwd: pd.DataFrame,
                h: int, min_cs: int) -> float:
  """
  在给定样本集覆盖的交易日上算 RankIC(不按 IS/OOS 窗切, 直接用样本自身日期).

  用于 IS **内部**验证块选参: 判据与回测目标(RankIC 驱动的选股)一致, 且只在 IS 内取.
  """
  dates = sample.dates
  s = score_w.reindex(index=dates)
  f = fwd.reindex(index=dates)
  ic = ev.ic_series(s, f, 'spearman', min_cs)
  return float(ev.ic_summary(ic, nw_lag=max(h - 1, 1))['ic_mean'])


def main():
  ap = argparse.ArgumentParser(description='ml 模型层自检: MLP(fit IS -> 评 OOS)')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--prep', default=cfg.FEATURE_PREP,
                  choices=('raw', 'z', 'rank', 'mad_z', 'q_z'))
  ap.add_argument('--source', default=cfg.DEFAULT_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES),
                  help='特征来源: factor / ta / both')
  ap.add_argument('--hidden', default=','.join(str(x) for x in cfg.MLP_HIDDEN),
                  help='隐层宽度, 逗号分隔(如 64,32)')
  ap.add_argument('--dropout', type=float, default=cfg.MLP_DROPOUT)
  ap.add_argument('--lr', type=float, default=cfg.MLP_LR)
  ap.add_argument('--wd', type=float, default=cfg.MLP_WEIGHT_DECAY)
  ap.add_argument('--epochs', type=int, default=cfg.MLP_EPOCHS)
  ap.add_argument('--batch', type=int, default=cfg.MLP_BATCH_SIZE)
  ap.add_argument('--patience', type=int, default=cfg.MLP_PATIENCE)
  ap.add_argument('--seed', type=int, default=cfg.MLP_SEED)
  ap.add_argument('--valid-frac', type=float, default=cfg.MLP_VALID_FRAC,
                  help='IS 内部时序早停验证块占比')
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  ap.add_argument('--top', type=int, default=12, help='打印重要性前 N 个特征(粗略代理)')
  ap.add_argument('--check', action='store_true', help='加做 oracle/random/shuffle 对照')
  ap.add_argument('--search', action='store_true',
                  help='在 IS 内部的早停验证块上做 lr×wd 小网格选参(判据=验证 RankIC)')
  ap.add_argument('--lrs', default='0.001,0.0003,0.0001',
                  help='搜参的 lr 列表(逗号分隔)')
  ap.add_argument('--wds', default='0.0001,0.01',
                  help='搜参的 weight_decay 列表(逗号分隔)')
  a = ap.parse_args()

  hidden = tuple(int(x) for x in str(a.hidden).split(',') if x.strip())

  log = cfg.get_logger('ml.models.neural')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  s = assemble(ds, None, a.h, a.method, a.prep, a.min_cs,
               feature_source=a.source, log=log)
  fwd = ev.forward_return(ds, a.h)                 # 全历史(标签必须用未来)
  log.info(f'[口径] h={a.h}, label={a.method}, prep={a.prep}, source={a.source}, '
           f'hidden={hidden}, dropout={a.dropout:g}, lr={a.lr:g}, wd={a.wd:g}, '
           f'seed={a.seed}, valid_frac={a.valid_frac}')
  if a.search:
    log.info('[提示] --search 在 IS **内部**选 lr×wd; 其余超参(hidden/dropout/epochs)'
             '仍为默认, 未寻优.')
  else:
    log.info('[提示] 当前用的是 config.MLP_* 的**默认超参(未在 IS 内寻优)**, '
             '仅作骨架跑通与基线对比; 定位是"DL 相对 GBDT 有没有增量"的对照臂.')

  is_s, oos_s = s.split('IS'), s.split('OOS')
  tr, va, info = holdout_split(is_s, a.valid_frac, embargo=a.h + 1)   # IS 内时序早停块
  log.info(f'[早停] IS 内时序切分: train <= {info["train_end"]:%Y-%m-%d} '
           f'(n={info["n_train"]:,}), embargo={info["embargo"]} 天, '
           f'valid {info["valid_start"]:%Y-%m-%d}~{info["valid_end"]:%Y-%m-%d} '
           f'(n={info["n_valid"]:,})')

  if a.search:
    # 红线: 选参只看 IS **内部**的早停验证块(绝不看 OOS). 判据用验证块 RankIC ——
    # 与回测目标(按分数截面选股)一致, 比 MSE 更贴; 但同一块既选参又早停, 略有双重使用.
    lrs = [float(x) for x in str(a.lrs).split(',') if x.strip()]
    wds = [float(x) for x in str(a.wds).split(',') if x.strip()]
    log.info(f'\n[搜参] IS 内选 lr×wd: lrs={lrs}, wds={wds}, hidden={hidden}, '
             f'dropout={a.dropout:g} (判据 = 早停验证块 RankIC)')
    srows, best = [], None
    for lr in lrs:
      for wd in wds:
        m = MLPModel(hidden=hidden, dropout=a.dropout, lr=lr, weight_decay=wd,
                     epochs=a.epochs, batch_size=a.batch, patience=a.patience,
                     seed=a.seed).fit(tr, valid=va, valid_fwd=fwd,
                                      min_cs=a.min_cs, criterion='rank_ic')
        ric = _rank_ic_on(va, m.predict(va), fwd, a.h, a.min_cs)
        srows.append({'lr': lr, 'wd': wd, 'best_epoch': m.best_epoch_,
                      'best_score': round(float(m.best_val_), 4),
                      'valid_rank_ic': round(ric, 4)})
        if best is None or ric > best[0]:
          best = (ric, m, lr, wd)
    log.info('\n[搜参结果] 按 IS 内验证 RankIC 降序(与 OOS 完全无关):\n'
             + pd.DataFrame(srows).sort_values('valid_rank_ic',
                                               ascending=False).to_string(index=False))
    _, model, a.lr, a.wd = best
    log.info(f'[搜参] 选中 lr={a.lr:g}, wd={a.wd:g} '
             f'(验证 RankIC={best[0]:.4f}, best_epoch={model.best_epoch_})')
  else:
    model = MLPModel(hidden=hidden, dropout=a.dropout, lr=a.lr, weight_decay=a.wd,
                     epochs=a.epochs, batch_size=a.batch, patience=a.patience,
                     seed=a.seed).fit(tr, valid=va, log=log, valid_fwd=fwd,
                                      min_cs=a.min_cs, criterion='rank_ic')

  pred = model.predict(s)
  tab = pd.DataFrame([eval_scores(pred, fwd, w, a.h, a.min_cs)
                      for w in ('IS', 'OOS', 'recent')])
  log.info('\n[基线表现] MLP 分数 vs 可交易前向收益(复用 factor.evaluate 的 IC 口径)')
  log.info('\n' + tab.to_string(index=False))
  log.info('[说明] IS 含 fit 段, 其 RankIC 偏乐观; OOS/recent 未参与任何拟合, '
           '才是可对外汇报的诚实数字.')

  imp = model.input_importance(a.top)
  log.info(f'\n[重要性] 首层权重列 L1 前 {a.top} 个特征(**粗略代理**, 非严格归因):\n'
           f'{imp.to_string(index=False)}')
  n_ta = int(sum(1 for f in imp['feature'] if f not in _factor_names()))
  if n_ta:
    log.info(f'[提示] 前 {a.top} 里含 {n_ta} 个非 factor 特征(TA 列) —— '
             f'TA 因果性未验证, 若它主导打分, OOS 结论需格外谨慎.')

  if a.check:
    # 1) oracle: 前向收益本身当分数 -> RankIC ~1(校准评估口径)
    # 2) random: 随机分数(对齐 pred 的 NaN 格局) -> ~0
    # 3) shuffle: 打乱 IS 标签再训 -> OOS RankIC 应塌回 ~0(排除 harness 自带泄漏)
    rng = np.random.default_rng(0)
    rows = [dict(case='oracle(=fwd)', **eval_scores(fwd, fwd, 'IS', a.h, a.min_cs)),
            dict(case='random', **eval_scores(
                _random_like(pred, rng), fwd, 'IS', a.h, a.min_cs))]
    perm = rng.permutation(is_s.y).astype(np.float32)
    sh_s = SampleSet(is_s.X, perm, is_s.meta, is_s.features,
                     is_s.horizon, is_s.label_method)
    sh_tr, sh_va, _ = holdout_split(sh_s, a.valid_frac, embargo=a.h + 1)
    sh = MLPModel(hidden=hidden, dropout=a.dropout, lr=a.lr, weight_decay=a.wd,
                  epochs=a.epochs, batch_size=a.batch, patience=a.patience,
                  seed=a.seed).fit(sh_tr, valid=sh_va, valid_fwd=fwd,
                                   min_cs=a.min_cs, criterion='rank_ic')
    rows.append(dict(case='shuffle-label', **eval_scores(sh.predict(s), fwd, 'OOS',
                                                         a.h, a.min_cs)))
    log.info(f'\n[对照] 校准评估口径(IS) + 打乱标签的 OOS')
    log.info('\n' + pd.DataFrame(rows).to_string(index=False))
    rnd, shf = rows[1]['rank_ic'], rows[2]['rank_ic']
    log.info(f'[结论] oracle≈1 且 random≈0 判评估口径正确; '
             f'shuffle 的 OOS RankIC={shf} 应≈0(否则 harness 有泄漏) -> '
             + ('通过' if abs(rnd) < 0.1 and abs(shf) < 0.1 else '异常, 需排查'))

  log.info('\n[下一步] predict.py --model mlp 出分数宽表 -> factor.backtest 出净值, '
           '与 ridge / gbdt 在同一引擎下逐值对比.')


def _factor_names() -> set:
  """factor 注册表里的因子名集合(用于区分 TA 列)."""
  from quant.factor import factor as fct
  return set(fct.REGISTRY)


if __name__ == '__main__':
  main()
# -*- coding: utf-8 -*-
"""
ml.models.sequence — 序列神经网络(GRU / LSTM, 第4步之四)
==================================================
职责: 在 Ridge(线性) / GBDT(树) / MLP(截面) 之外, 加一个**吃时间窗口**的模型, 回答:
      "把 D_{t-L+1}..D_t 的窗口喂给 RNN, 相比只看当天截面的 GBDT/MLP 有没有增量?"

--- 它吃什么 / 吐什么 ---
  吃 ml.seq.SequenceSampleSet: X = (n × L × F) 窗口(**窗口末端 = 信号日 t**), y = t 日的
  h 日前向 rank 标签(与 ridge/gbdt/mlp 同一份标签). 吐 (date × symbol) 分数宽表,
  可直接丢进 factor.evaluate.ic_series / factor.backtest.run_engine.

--- 为什么是单层小 RNN, 不是 Transformer ---
  样本量小 + 信噪比低 + 仅 CPU -> 单层小 RNN 就够, 先拿到"时序结构有没有增量"的可比数字.
  大模型/Transformer 在这个数据规模上几乎必然过拟合, 留待确认有信号后再谈.

--- GRU 与 LSTM 的关系(对照臂) ---
  两者共用同一套管道/超参/早停协议, 只换循环单元(GRUModel / LSTMModel), 便于回答
  "GRU 的门控结构相对 LSTM 有没有增量". 故共用一个基类 SequenceNetModel, 各自只覆写
  TAG 与 _build_net —— 目标是"只换循环单元, 其余全同"的严格同口径对照.

--- 无前视的两条纪律 ---
  1. 纵向: 输入窗口只到信号日 t(由 ml.seq 的滑窗保证, 并过截断一致性检验);
  2. 横向: fit 只喂 IS; 早停在 IS **内部**用 dataset.holdout_split 做时序切分(embargo=h+1),
     绝不拿 OOS 选轮数(红线).

--- 预处理(红线) ---
  RNN(GRU/LSTM) 不原生吃 NaN -> 先按列中位数填补; 再做逐列标准化; 标签也标准化. 上述统计量
  (填补值 / 均值 / 标准差)**只在本模型被 fit 的那段样本上估计** —— 统计范围 = 该段样本的
  全部窗口格(n×L 个格), 不跨段. 复用 ml.models.linear 的 _col_fill / _apply_fill(把
  (n,L,F) 展平成 (n*L, F) 后逐列估计).

自检:
  cd ~/git && python -m quant.ml.models.sequence --pool etf_3x
  cd ~/git && python -m quant.ml.models.sequence --pool etf_3x --check
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor import evaluate as ev

from quant.ml import config as cfg
from quant.ml.data import load
from quant.ml.dataset import holdout_split
from quant.ml.models.linear import (_apply_fill, _col_fill, _random_like,
                                    eval_scores, scores_to_wide)
from quant.ml.models.neural import _rank_ic_on
from quant.ml.seq import SequenceSampleSet, build_sequences


# ============================ 网络结构 ============================

def _make_gru(n_features: int, hidden: int, num_layers: int, dropout: float):
  """构造单/多层 GRU + 输出头. 取窗口末端(t 日)隐状态打分. 延迟 import torch."""
  import torch.nn as nn

  class GRUNet(nn.Module):
    def __init__(self):
      super().__init__()
      # PyTorch 限制: num_layers==1 时 RNN dropout 必须为 0
      self.gru = nn.GRU(n_features, int(hidden), num_layers=int(num_layers),
                        batch_first=True,
                        dropout=(float(dropout) if int(num_layers) > 1 else 0.0))
      self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(hidden), 1))

    def forward(self, x):                                     # x: (B, L, F)
      out, _ = self.gru(x)
      return self.head(out[:, -1, :]).squeeze(-1)             # 只看窗口末端(t 日)

  return GRUNet()


def _make_lstm(n_features: int, hidden: int, num_layers: int, dropout: float):
  """构造单/多层 LSTM + 输出头. 取窗口末端(t 日)隐状态打分. 延迟 import torch.

  与 _make_gru 严格同构, 只把 nn.GRU 换成 nn.LSTM(门控结构: 遗忘/输入/输出门 vs 重置/更新门),
  以便做"只换循环单元"的同口径对照.
  """
  import torch.nn as nn

  class LSTMNet(nn.Module):
    def __init__(self):
      super().__init__()
      # PyTorch 限制: num_layers==1 时 RNN dropout 必须为 0
      self.lstm = nn.LSTM(n_features, int(hidden), num_layers=int(num_layers),
                          batch_first=True,
                          dropout=(float(dropout) if int(num_layers) > 1 else 0.0))
      self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(hidden), 1))

    def forward(self, x):                                     # x: (B, L, F)
      out, _ = self.lstm(x)                                   # 丢弃 (h_n, c_n)
      return self.head(out[:, -1, :]).squeeze(-1)             # 只看窗口末端(t 日)

  return LSTMNet()


# ======================== 序列模型(公共基类) ========================

class SequenceNetModel:
  """
  序列回归器公共基类(窗口 -> 分数), 仅 CPU. 只换循环单元, 其余全同.

  子类只需覆写 TAG 与 _build_net(n_features); 预处理/训练/早停/打分全在此实现.
  :param hidden: 循环单元隐层宽度
  :param num_layers: 循环单元层数
  :param dropout: 输出头 dropout
  :param lr: Adam 学习率
  :param weight_decay: Adam 权重衰减(L2)
  :param epochs: 最多训练轮数(有 valid 时被早停截断)
  :param batch_size: mini-batch 大小
  :param patience: 早停耐心(连续多少轮验证损失不改善即停)
  :param grad_clip: 梯度裁剪阈值
  :param seed: 随机种子(单 seed)
  :param fill: 缺失填补策略('median'/'mean'/'zero')
  """

  TAG = 'seq'                     # 子类覆写: 日志/文案用的模型名

  def __init__(self, hidden: int = cfg.SEQ_HIDDEN, num_layers: int = cfg.SEQ_LAYERS,
               dropout: float = cfg.SEQ_DROPOUT, lr: float = cfg.SEQ_LR,
               weight_decay: float = cfg.SEQ_WEIGHT_DECAY, epochs: int = cfg.SEQ_EPOCHS,
               batch_size: int = cfg.SEQ_BATCH_SIZE, patience: int = cfg.SEQ_PATIENCE,
               grad_clip: float = cfg.SEQ_GRAD_CLIP, seed: int = cfg.SEQ_SEED,
               fill: str = cfg.FILL_STRATEGY):
    self.hidden = int(hidden)
    self.num_layers = int(num_layers)
    self.dropout = float(dropout)
    self.lr = float(lr)
    self.weight_decay = float(weight_decay)
    self.epochs = int(epochs)
    self.batch_size = int(batch_size)
    self.patience = int(patience)
    self.grad_clip = float(grad_clip)
    self.seed = int(seed)
    self.fill = fill
    self.features = None
    self.window_len = None
    self.fill_ = None
    self.x_mean_ = None
    self.x_std_ = None
    self.y_mean_ = 0.0
    self.y_std_ = 1.0
    self.net_ = None
    self.n_fit_ = 0
    self.best_epoch_ = None
    self.best_val_ = None
    self.best_crit_ = None
    self.min_cs = int(fcfg.MIN_CS)

  # ---- 网络构造钩子: 子类覆写(换循环单元) ----
  def _build_net(self, n_features: int):
    raise NotImplementedError('子类须实现 _build_net')

  # ---- 预处理: 统计量只在 fit 样本上估(红线) ----
  @staticmethod
  def _flat(X: np.ndarray) -> np.ndarray:
    """(n, L, F) -> (n*L, F)(逐列统计用; 不改值)."""
    n, L, F = X.shape
    return X.reshape(n * L, F)

  def _prepare(self, sample: SequenceSampleSet) -> np.ndarray:
    """按 fit 时定下的 fill/mean/std 把窗口张量标准化(不重新估计统计量)."""
    return self._prepare_chunk(sample.X)

  def _prepare_chunk(self, X: np.ndarray) -> np.ndarray:
    """(b, L, F) 窗口张量标准化 -> (b, L, F) float32(不重估统计量).

    单独拎出来是为了让打分能**按样本维分块**调用: 合并池样本量达数百万窗口时,
    一次性 _prepare 整个张量会产生 flat 的 float64 副本 + 标准化副本, 额外占几十 GB.
    """
    n, L, F = X.shape
    Xf = _apply_fill(self._flat(X), self.fill_)
    Xs = ((Xf - self.x_mean_) / self.x_std_).astype(np.float32)
    return Xs.reshape(n, L, F)

  # ---- fit: 只吃传入样本(约定 IS); valid 也来自 IS 内部时序切分 ----
  def fit(self, sample: SequenceSampleSet, valid: SequenceSampleSet = None,
          log=None, valid_fwd: pd.DataFrame = None, min_cs: int = fcfg.MIN_CS,
          criterion: str = 'rank_ic') -> 'SequenceNetModel':
    """
    在序列样本上训练. 若给 valid(IS 内时序验证块), 则做早停并回滚到最优轮.

    早停判据 criterion:
      - 'rank_ic'(默认, 需给 valid_fwd): 用验证块 RankIC, 取最大 —— 与回测目标(按分数
        截面选股)同构. 实践发现按 MSE 早停会停在第 1 轮(近乎常值预测), 因为低信噪比下
        MSE 最小点并不对应排序能力最强点; 改用 RankIC 与 MLP 的 --search 判据一致.
      - 'mse': 用验证块标准化 MSE, 取最小(旧行为).
    ⚠ valid 与 valid_fwd 必须来自 IS **内部**, 绝不能用 OOS/recent 选轮数(红线).

    :param sample: 训练样本(约定 IS)
    :param valid: 早停验证样本(IS 内部时序切分; None = 训练固定 epochs)
    :param log: 可选 logger
    :param valid_fwd: 可交易前向收益宽表(算验证 RankIC 用; 须覆盖 valid 的交易日)
    :param min_cs: 每日截面最小样本数(RankIC 口径)
    :param criterion: 早停判据 'rank_ic' / 'mse'
    """
    try:
      import torch
      import torch.nn as nn
    except ImportError as e:                                  # 环境缺库时给出清晰指引
      raise ImportError('需要 torch: pip install torch --index-url '
                        'https://download.pytorch.org/whl/cpu') from e

    self.features = list(sample.features)
    self.window_len = int(sample.window_len)
    F = len(self.features)

    # 1) 估预处理统计量(只在 fit 样本的全部窗口格上) + 标签标准化
    flat = self._flat(sample.X)
    self.fill_ = _col_fill(flat, self.fill)
    Xf = _apply_fill(flat, self.fill_)
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
      if list(valid.features) != self.features or valid.window_len != self.window_len:
        raise ValueError('valid 与 train 的特征/窗口长度不一致')
      Xv = self._prepare(valid)
      yv = ((valid.y - self.y_mean_) / self.y_std_).astype(np.float32)

    # 3) 建模 + 训练(单 seed, 确定性)
    torch.manual_seed(self.seed)
    net = self._build_net(F)
    opt = torch.optim.Adam(net.parameters(), lr=self.lr,
                           weight_decay=self.weight_decay)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(ys)
    n = Xt.shape[0]
    gen = torch.Generator().manual_seed(self.seed)

    best_state, best_val, best_epoch, bad = None, np.inf, 0, 0
    # 判据 = RankIC 时为"越大越好", 初值取 -inf; 判据 = MSE 时为"越小越好", 初值取 +inf
    use_ric = (valid is not None and valid_fwd is not None and criterion == 'rank_ic')
    self.min_cs = int(min_cs)
    if use_ric:
      best_val = -np.inf
    ran = 0
    for ep in range(1, self.epochs + 1):
      ran = ep
      net.train()
      perm = torch.randperm(n, generator=gen)
      for i in range(0, n, self.batch_size):
        idx = perm[i:i + self.batch_size]
        opt.zero_grad()
        loss = lossf(net(Xt[idx]), yt[idx])
        loss.backward()
        if self.grad_clip > 0:                                # RNN 训练稳定性
          torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
        opt.step()
      if valid is None:
        continue
      net.eval()
      with torch.no_grad():                                   # 验证 MSE(标准化尺度), 始终记录
        vl = float(lossf(net(torch.from_numpy(Xv)),
                         torch.from_numpy(yv)).item())
      if use_ric:                                             # 判据: 验证块 RankIC(越大越好)
        key = _rank_ic_on(valid, self._score_wide(valid, net=net), valid_fwd,
                          valid.horizon, self.min_cs)
        improved = key > best_val + 1e-6
      else:                                                   # 判据: 验证 MSE(越小越好)
        key, improved = vl, vl < best_val - 1e-6
      if improved:                                            # 改善 -> 存最优状态
        best_val, best_epoch, bad = key, ep, 0
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
      else:                                                   # 不改善 -> 累计耐心
        bad += 1
        if bad >= self.patience:
          break

    if valid is not None and best_state is not None:          # 回滚到验证最优轮
      net.load_state_dict(best_state)
    self.net_, self.best_epoch_, self.best_val_ = net, best_epoch, best_val
    self.best_crit_ = ('rank_ic' if use_ric else 'mse')
    self.net_.eval()
    self.n_fit_ = int(n)
    if log is not None:
      crit = ('early_stop=rank_ic' if use_ric
              else ('early_stop=mse' if valid is not None else 'early_stop=off'))
      log.info(f'[{self.TAG}] fit: n={self.n_fit_:,}, L={self.window_len}, F={F}, '
               f'hidden={self.hidden}, layers={self.num_layers}, '
               f'dropout={self.dropout:g}, lr={self.lr:g}, wd={self.weight_decay:g}, '
               f'epochs={ran}/{self.epochs}, {crit}'
               + (f', best_epoch={best_epoch}, best_val={best_val:.4f}'
                  if valid is not None else ''))
    return self

  # ---- 打分: 用当前网络把序列样本打成截面分数宽表(供 predict / 早停共用) ----
  def _score_wide(self, sample: SequenceSampleSet, net=None) -> pd.DataFrame:
    """用给定网络(默认 self.net_)打分, 不校验特征/窗口(供 fit 内部对 valid 打分用).

    按样本维分块 prepare + 前向: 合并池(数百万窗口)下若一次性 _prepare 整个张量,
    会额外占几十 GB(见 _prepare_chunk) -> 逐块预处理, 峰值内存与块大小同阶.
    """
    import torch
    net = self.net_ if net is None else net
    scores = np.empty(sample.n_samples, dtype=np.float64)
    step = 8192
    with torch.no_grad():
      for i in range(0, sample.n_samples, step):              # 分批前向, 控内存
        j = min(i + step, sample.n_samples)
        chunk = torch.from_numpy(self._prepare_chunk(sample.X[i:j]))
        scores[i:j] = net(chunk).numpy()
    scores = scores * self.y_std_ + self.y_mean_              # 还原到标签尺度
    return scores_to_wide(sample, scores)

  # ---- predict: 用训练好的网络打分, 返回截面分数宽表 ----
  def predict(self, sample: SequenceSampleSet) -> pd.DataFrame:
    if self.net_ is None:
      raise RuntimeError('请先 fit')
    if list(sample.features) != self.features:
      raise ValueError('特征顺序与 fit 时不一致')
    if int(sample.window_len) != int(self.window_len):
      raise ValueError('窗口长度与 fit 时不一致')
    return self._score_wide(sample)


# ============================ 具体模型 ============================

class GRUModel(SequenceNetModel):
  """序列 GRU 回归器(窗口 -> 分数), 仅 CPU. 覆写 _build_net 换循环单元为 GRU."""
  TAG = 'gru'

  def _build_net(self, n_features: int):
    return _make_gru(n_features, self.hidden, self.num_layers, self.dropout)


class LSTMModel(SequenceNetModel):
  """序列 LSTM 回归器(窗口 -> 分数), 仅 CPU. 覆写 _build_net 换循环单元为 LSTM.

  与 GRUModel 共用同一组 SEQ_* 超参/管道/早停协议(严格同口径对照).
  """
  TAG = 'lstm'

  def _build_net(self, n_features: int):
    return _make_lstm(n_features, self.hidden, self.num_layers, self.dropout)


# ============================ 自检 ============================

def main():
  ap = argparse.ArgumentParser(description='ml 模型层自检: 序列模型 gru/lstm(fit IS -> 评 OOS)')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--model', default='gru', choices=list(cfg.SEQ_MODELS),
                  help='循环单元: gru / lstm(同一套超参/管道, 严格同口径对照)')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--prep', default=cfg.FEATURE_PREP,
                  choices=('raw', 'z', 'rank', 'mad_z', 'q_z'))
  ap.add_argument('--window-len', type=int, default=cfg.SEQ_WINDOW, help='窗口长度 L')
  ap.add_argument('--stride', type=int, default=cfg.SEQ_STRIDE, help='信号日起步')
  ap.add_argument('--source', default=cfg.SEQ_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES),
                  help='序列特征来源: factor / ta / both(默认 cfg.SEQ_FEATURE_SOURCE; '
                       '含 TA 时会打印 TA_CAVEAT, 其 1 日前视在长窗口下被放大)')
  ap.add_argument('--hidden', type=int, default=cfg.SEQ_HIDDEN)
  ap.add_argument('--layers', type=int, default=cfg.SEQ_LAYERS)
  ap.add_argument('--dropout', type=float, default=cfg.SEQ_DROPOUT)
  ap.add_argument('--lr', type=float, default=cfg.SEQ_LR)
  ap.add_argument('--wd', type=float, default=cfg.SEQ_WEIGHT_DECAY)
  ap.add_argument('--epochs', type=int, default=cfg.SEQ_EPOCHS)
  ap.add_argument('--batch', type=int, default=cfg.SEQ_BATCH_SIZE)
  ap.add_argument('--patience', type=int, default=cfg.SEQ_PATIENCE)
  ap.add_argument('--seed', type=int, default=cfg.SEQ_SEED)
  ap.add_argument('--valid-frac', type=float, default=cfg.SEQ_VALID_FRAC,
                  help='IS 内部时序早停验证块占比')
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  ap.add_argument('--check', action='store_true', help='加做 oracle/random/shuffle 对照')
  ap.add_argument('--search', action='store_true',
                  help='在 IS 内部做 L×lr×wd 网格选参(判据=IS 内验证块 RankIC; OOS 不参与)')
  ap.add_argument('--lens', default=str(cfg.SEQ_WINDOW),
                  help='搜参的窗口长度 L 列表(逗号分隔); L 越大内存/算力越高')
  ap.add_argument('--lrs', default='0.001,0.0003', help='搜参的 lr 列表(逗号分隔)')
  ap.add_argument('--wds', default='0.01,0.0001', help='搜参的 weight_decay 列表(逗号分隔)')
  ap.add_argument('--criterion', default='rank_ic', choices=('rank_ic', 'mse'),
                  help='早停/选参判据; rank_ic 与回测目标同构(推荐)')
  a = ap.parse_args()
  cls = {'gru': GRUModel, 'lstm': LSTMModel}[a.model]

  log = cfg.get_logger('ml.models.sequence')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')
  log.info(f'[口径] model={a.model}, L={a.window_len}, stride={a.stride}, h={a.h}, '
           f'label={a.method}, prep={a.prep}, source={a.source}, hidden={a.hidden}, '
           f'layers={a.layers}, dropout={a.dropout:g}, lr={a.lr:g}, wd={a.wd:g}, '
           f'seed={a.seed}, valid_frac={a.valid_frac}')

  s = build_sequences(ds, None, a.h, a.method, a.prep, a.min_cs,
                      window_len=a.window_len, stride=a.stride, source=a.source, log=log)
  fwd = ev.forward_return(ds, a.h)                 # 全历史(标签必须用未来)
  if a.search:
    log.info('[提示] --search 在 IS **内部**按 L×lr×wd 网格选参(判据 = IS 内验证块 '
             'RankIC); OOS/recent 完全不参与挑选.')
  else:
    log.info('[提示] 当前用的是 config.SEQ_*(L/lr/wd 已做 IS 内 --search 寻优, 见 config '
             '注释), 定位是"序列结构相对截面模型有没有增量"的对照臂.')

  is_s = s.split('IS')
  tr, va, info = holdout_split(is_s, a.valid_frac, embargo=a.h + 1)   # IS 内时序早停块
  log.info(f'[早停] IS 内时序切分: train <= {info["train_end"]:%Y-%m-%d} '
           f'(n={info["n_train"]:,}), embargo={info["embargo"]} 天, '
           f'valid {info["valid_start"]:%Y-%m-%d}~{info["valid_end"]:%Y-%m-%d} '
           f'(n={info["n_valid"]:,})')

  if a.search:
    # 红线: 选参只看 IS **内部**的验证块(绝不看 OOS/recent). 判据 = 验证块 RankIC, 与回测
    # 目标(按分数截面选股)同构。同一块既早停又选参, 略有双重使用(与 MLP --search 同).
    lens = [int(x) for x in str(a.lens).split(',') if x.strip()]
    lrs = [float(x) for x in str(a.lrs).split(',') if x.strip()]
    wds = [float(x) for x in str(a.wds).split(',') if x.strip()]
    seqs = {a.window_len: s}                                  # 各 L 的序列样本(含缓存)
    log.info(f'\n[搜参] IS 内选 L×lr×wd: lens={lens}, lrs={lrs}, wds={wds}, '
             f'hidden={a.hidden}, layers={a.layers}, dropout={a.dropout:g}, '
             f'epochs={a.epochs} (判据 = {a.criterion})')
    rows, best = [], None
    for L in lens:
      if L not in seqs:                                       # L 变了要重建窗口张量
        seqs[L] = build_sequences(ds, None, a.h, a.method, a.prep, a.min_cs,
                                  window_len=L, stride=a.stride, source=a.source, log=log)
      isl = seqs[L].split('IS')
      trl, val, _ = holdout_split(isl, a.valid_frac, embargo=a.h + 1)
      for lr in lrs:
        for wd in wds:
          m = cls(hidden=a.hidden, num_layers=a.layers, dropout=a.dropout, lr=lr,
                  weight_decay=wd, epochs=a.epochs, batch_size=a.batch,
                  patience=a.patience, seed=a.seed).fit(
              trl, valid=val, valid_fwd=fwd, min_cs=a.min_cs, criterion=a.criterion)
          ric = _rank_ic_on(val, m.predict(val), fwd, a.h, a.min_cs)
          rows.append({'L': L, 'lr': lr, 'wd': wd, 'best_epoch': m.best_epoch_,
                       'best_val': round(float(m.best_val_), 4),
                       'valid_rank_ic': round(float(ric), 4)})
          if best is None or ric > best[0]:
            best = (ric, L, lr, wd)
    stab = pd.DataFrame(rows).sort_values('valid_rank_ic', ascending=False)
    log.info('\n[搜参结果] 按 IS 内验证块 RankIC 降序(全部只在 IS 内部, OOS 未参与):')
    log.info('\n' + stab.to_string(index=False))
    log.info(f'[搜参] 选中 L={best[1]}, lr={best[2]:g}, wd={best[3]:g} '
             f'(valid RankIC={best[0]:.4f})')

    # 用"选定"与"默认"两组各在全 IS 上重拟合, 做一次诚实汇报(OOS/recent 未参与定参)
    log.info('\n[定参结果 vs 默认] 全 IS 拟合后在 IS/OOS/recent 的表现')
    r2 = []
    for tag, L, lr, wd in (('选定', best[1], best[2], best[3]),
                           ('默认', a.window_len, a.lr, a.wd)):
      if L not in seqs:
        seqs[L] = build_sequences(ds, None, a.h, a.method, a.prep, a.min_cs,
                                  window_len=L, stride=a.stride, source=a.source, log=None)
      sl = seqs[L]
      trl, val, _ = holdout_split(sl.split('IS'), a.valid_frac, embargo=a.h + 1)
      m = cls(hidden=a.hidden, num_layers=a.layers, dropout=a.dropout, lr=lr,
              weight_decay=wd, epochs=a.epochs, batch_size=a.batch,
              patience=a.patience, seed=a.seed).fit(
          trl, valid=val, valid_fwd=fwd, min_cs=a.min_cs, criterion=a.criterion)
      p = m.predict(sl)
      for w in ('IS', 'OOS', 'recent'):
        r = eval_scores(p, fwd, w, a.h, a.min_cs)
        r2.append(dict(方案=f'{tag} L={L},lr={lr:g},wd={wd:g}', 窗口=r['window'],
                       n_days=r['n_days'], RankIC=round(r['rank_ic'], 4),
                       IC=round(r['ic'], 4), NW_t=round(r['nw_t'], 2),
                       胜率=round(r['pos_rate'], 3)))
    log.info('\n' + pd.DataFrame(r2).to_string(index=False))
    log.info('[说明] IS 为 fit 段(偏乐观); OOS/recent 未参与定参, 是诚实数字.')
    log.info(f'[记录] 本次选定 L={best[1]}, lr={best[2]:g}, wd={best[3]:g}; '
             '下一步把该组值写回 config.SEQ_WINDOW / SEQ_LR / SEQ_WEIGHT_DECAY.')
    return

  model = cls(hidden=a.hidden, num_layers=a.layers, dropout=a.dropout, lr=a.lr,
              weight_decay=a.wd, epochs=a.epochs, batch_size=a.batch,
              patience=a.patience, seed=a.seed).fit(
      tr, valid=va, log=log, valid_fwd=fwd, min_cs=a.min_cs, criterion=a.criterion)
  ric = _rank_ic_on(va, model.predict(va), fwd, a.h, a.min_cs)
  log.info(f'[早停验证块] RankIC={ric:.4f}(判据与回测目标同构, 仅供选参参考)')

  pred = model.predict(s)
  tab = pd.DataFrame([eval_scores(pred, fwd, w, a.h, a.min_cs)
                      for w in ('IS', 'OOS', 'recent')])
  log.info(f'\n[基线表现] {a.model} 分数 vs 可交易前向收益(复用 factor.evaluate 的 IC 口径)')
  log.info('\n' + tab.to_string(index=False))
  log.info('[说明] IS 含 fit 段, 其 RankIC 偏乐观; OOS/recent 未参与任何拟合, '
           '才是可对外汇报的诚实数字.')

  if a.check:
    # 1) oracle: 前向收益本身当分数 -> RankIC ~1(校准评估口径)
    # 2) random: 随机分数(对齐 pred 的 NaN 格局) -> ~0
    # 3) shuffle: 打乱 IS 标签再训 -> OOS RankIC 应塌回 ~0(排除 harness 自带泄漏)
    rng = np.random.default_rng(0)
    rows = [dict(case='oracle(=fwd)', **eval_scores(fwd, fwd, 'IS', a.h, a.min_cs)),
            dict(case='random', **eval_scores(
                _random_like(pred, rng), fwd, 'IS', a.h, a.min_cs))]
    perm = rng.permutation(is_s.y).astype(np.float32)
    sh_s = SequenceSampleSet(is_s.X, perm, is_s.meta, is_s.features,
                             is_s.window_len, is_s.horizon, is_s.label_method)
    sh_tr, sh_va, _ = holdout_split(sh_s, a.valid_frac, embargo=a.h + 1)
    sh = cls(hidden=a.hidden, num_layers=a.layers, dropout=a.dropout, lr=a.lr,
             weight_decay=a.wd, epochs=a.epochs, batch_size=a.batch,
             patience=a.patience, seed=a.seed).fit(
        sh_tr, valid=sh_va, valid_fwd=fwd, min_cs=a.min_cs, criterion=a.criterion)
    rows.append(dict(case='shuffle-label', **eval_scores(sh.predict(s), fwd, 'OOS',
                                                         a.h, a.min_cs)))
    log.info('\n[对照] 校准评估口径(IS) + 打乱标签的 OOS')
    log.info('\n' + pd.DataFrame(rows).to_string(index=False))
    rnd, shf = rows[1]['rank_ic'], rows[2]['rank_ic']
    log.info(f'[结论] oracle≈1 且 random≈0 判评估口径正确; '
             f'shuffle 的 OOS RankIC={shf} 应≈0(否则 harness 有泄漏) -> '
             + ('通过' if abs(rnd) < 0.1 and abs(shf) < 0.1 else '异常, 需排查'))

  log.info(f'\n[下一步] predict.py --model {a.model} 出分数宽表 -> factor.backtest 出净值, '
           '与 ridge / gbdt / mlp 及另一循环单元在同一引擎下逐值对比.')


if __name__ == '__main__':
  main()
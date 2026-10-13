# -*- coding: utf-8 -*-
"""
ml.models — 模型层(第4步)
==========================
放"从特征学分数"的模型本身. 每个模型遵守同一条最小约定, 便于 train/predict 通用:

  fit(sample, valid=None, log=None, valid_fwd=None, min_cs=..., criterion='rank_ic')
                                  # 只在传入样本上估计参数(约定: 只喂 IS)
  predict(sample: SampleSet) -> pd.DataFrame   # (date × symbol) 截面分数宽表

约定 predict 返回**宽表**(与 factor 的 composite 同构), 这样评分/回测可直接复用
factor.evaluate / factor.backtest, 无需为 ml 另写一套.

模型工厂 make_model 让 train/predict 用一个字符串切换模型, 两条路径共享全部管线.

现有:
  linear   Ridge 基线(参照物: 线性+L2 都跑不出正 RankIC, 问题多半在前三步)
  gbdt     LightGBM(非线性 + 原生吃缺失; 默认超参未在 IS 内寻优)
  neural   PyTorch MLP(截面神经网络; 小模型 + IS 内时序早停; 定位 DL 对照臂)
  sequence PyTorch GRU / LSTM(吃 D_{t-L+1}..D_t 的窗口; 需 SequenceSampleSet, 见 ml.seq)
"""
from quant.ml import config as cfg
from quant.ml.models.linear import RidgeModel
from quant.ml.models.gbdt import GBDTModel
from quant.ml.models.neural import MLPModel
from quant.ml.models.sequence import GRUModel, LSTMModel


def make_model(name: str = cfg.DEFAULT_MODEL, **kw):
  """
  按名字造模型(统一入口).

  :param name: 'ridge' / 'gbdt' / 'mlp' / 'gru' / 'lstm'(见 config.MODELS)
  :param kw: 透传给对应模型构造函数的参数(如 alpha / fill / params / hidden)
  :returns: 未训练的模型实例
  """
  registry = {'ridge': RidgeModel, 'gbdt': GBDTModel, 'mlp': MLPModel,
              'gru': GRUModel, 'lstm': LSTMModel}
  try:
    cls = registry[name]
  except KeyError:
    raise KeyError(f'未知模型: {name} (可选 {list(cfg.MODELS)})') from None
  return cls(**kw)
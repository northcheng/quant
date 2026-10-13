# -*- coding: utf-8 -*-
"""
ml — 机器学习 / 深度学习研究体系
================================
与 `quant.factor` 平级的独立包, 复用 factor 的数据 / 标签口径 / 回测引擎作为底座,
只新建"特征工程 -> 模型 -> 训练 -> 预测"这一层, 用来探索"数据驱动"的研究路线.

--- 与 factor 体系的关系(不重复造轮子) ---
  factor 体系自下而上: config -> data -> prepare -> factor -> evaluate -> combine
                       -> backtest -> report, 回答的是"人写的规则有没有预测力".
  本包回答的是"让模型从同一批数据里自己学规律, 能不能更强". 因此:

    复用(底座, 不重写):
      - factor.data          数据层(Dataset: date×symbol 宽表)
      - factor.prepare       逐日截面标准化(rank / zscore) —— 天然因果
      - factor.factor        59 个已证因果的因子 = 现成的特征来源
      - factor.evaluate      可交易前向收益 forward_return + IC 口径
      - factor.backtest      回测引擎(run_engine / EngineParams / perf_stats)

    新建(本包):
      - features   特征工程: 因子 -> 逐日截面标准化 -> 特征张量 (T × N × F)
      - labels     标签:   可交易前向收益 -> 截面 rank/zscore (与 IC 口径同构)
      - dataset    样本组装: (X, y) 对齐 + 时序切分(IS/OOS/recent) + 防前视
      - models     模型层  (linear / gbdt / nn)
      - train      训练 + 时序 walk-forward CV
      - predict    预测 -> 截面分数宽表(与 factor 的 composite 同构)
      - report     报告与产物

--- 两条贯穿红线(与 factor 体系完全一致) ---
  1. 定参只用 IS 段(2011-01 ~ 2019-12); OOS / recent 只做汇报, 绝不回头调参;
  2. 防前视: 特征只读 <= t 的数据(逐日截面统计量, 禁全样本统计量);
     标签必须吃未来(t+1 开盘入场); 任何"全局拟合"的变换(如 scaler.fit)只能在 IS 内做.

分步结构(每步一个模块, 均可单独运行自检; cwd 为 ~/git):
  第0步 data.py     数据接入 -- 复用 factor.data.load_pool
  第1步 features.py 特征工程 -- 因子注册表 -> 逐日截面标准化特征
  第2步 labels.py   标签     -- 可交易前向收益 -> 截面 rank
  第3步 dataset.py  样本     -- (X, y) 对齐 + 时序切分
  第4步 models/ ... 模型 / 训练 / 预测(后续)

运行方式(用项目 venv, cwd 为 ~/git):
  C:\\Users\\northcheng\\.venv\\Scripts\\python.exe -m quant.ml.data --pool etf_3x
  C:\\Users\\northcheng\\.venv\\Scripts\\python.exe -m quant.ml.features --pool etf_3x --check
  C:\\Users\\northcheng\\.venv\\Scripts\\python.exe -m quant.ml.labels --pool etf_3x --check
  C:\\Users\\northcheng\\.venv\\Scripts\\python.exe -m quant.ml.dataset --pool etf_3x
"""
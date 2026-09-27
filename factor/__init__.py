# -*- coding: utf-8 -*-
"""
factor — 因子挖掘 / 验证 / 回测体系
====================================
一套**完全自包含**的研究体系: 只依赖 numpy / pandas / matplotlib 与标准库, 不 import
任何 bc_* 脚本(bc_factor_search / bc_combo_search / bc_backtest 后续可能被删除).
唯一的外部输入是 ~/quant 下的研究 pkl 数据文件.

设计原则:
  - 自包含: 所需实现(读 pkl / 因子 / 评估 / 回测)全部在本包内重写, 可读优先;
  - 防前视: 因子只用 t 及之前数据; 指标在全历史计算后才截评估窗口(warmup 不污染);
    信号日收盘决策 -> 次日开盘执行;
  - 宽表为主: (date × symbol) 宽表是各层之间的统一数据形态;
  - NaN 语义: 停牌/未上市为 NaN, 截面处理时屏蔽而非填 0.

分步结构(每步一个模块, 均可单独运行自检):
  第0步 data.py     数据层   -- pkl -> Dataset(长表 + date×symbol 宽表)
  第1步 factor.py   因子层   -- 因子注册表 + 基础因子(动量/反转/波动/量能/趋势)
  第2步 prepare.py  预处理   -- 去极值 / 标准化 / 缺失处理
  第3步 evaluate.py 评估层   -- 前向收益 / IC / 分层 / 换手
  第4步 combine.py  组合层   -- 等权 / IC 加权 / 贪心搜索
  第5步 backtest.py 回测层   -- rank -> 权重 -> 净值 + 绩效
  第6步 report.py   报告层   -- 汇总表 + 图 + 流水线

运行方式(用项目 venv, cwd 为 ~/git):
  C:\\Users\\northcheng\\.venv\\Scripts\\python.exe -m quant.factor.data --pool etf_3x
"""

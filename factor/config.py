# -*- coding: utf-8 -*-
"""
factor.config — 因子体系全局配置
=================================
本包**完全自包含**: 只依赖 numpy / pandas / matplotlib 与标准库, 不 import 任何 bc_*
模块(那三个研究脚本后续可能被删除). 需要用到的好实现一律在本包内重写.

集中管理:

  1. 路径: pkl 数据目录(~ 展开) 与 本系统输出目录;
  2. 池 -> pkl 路径解析(研究版全史 / 生产版两种命名);
  3. 时间与口径常量: 研究窗口 IS/OOS 切分、因果归一化窗口、回测默认参数;
  4. 日志: 统一 logger(handler 挂 root, 库层只发日志).

口径对齐: TRADE_START / IS_END / OOS_START / TOP_K / EXIT_RANK / COST_BPS 与现有
bc_combo_search、bc_backtest 的默认一致, 便于本系统结论与生产互相对照.
"""
import datetime
import logging
from pathlib import Path

# ---- 路径 ----
HOME = Path.home()
PKL_DIR = HOME / 'quant'                # 研究/生产 pkl 所在目录(资产数据, 只读)
FACTOR_DIR = Path(__file__).resolve().parent
OUT_DIR = FACTOR_DIR / 'output'         # 产物根目录; 报告层按池写到 OUT_DIR/{pool}/
LOG_DIR = OUT_DIR / 'logs'              # 日志(按日期命名, 与池无关, 仍留在根下)
FACTOR_CONFIG = FACTOR_DIR.parent / 'factor_config.json'   # 研究结论 -> 生产的接口(因子名单+权重)

# ---- 数据 ----
DEFAULT_POOL = 'etf_3x'
DEFAULT_INTERVAL = 'day'

# ---- 研究窗口(因子在全历史计算, 只在评估时截窗口 -> 无 warmup 污染) ----
TRADE_START = '2021-01-01'              # 交易窗口起点
IS_END = '2024-12-31'                   # 样本内(IS)终点: 选参
OOS_START = '2025-01-01'                # 样本外(OOS)起点: 复验

# ---- 口径常量 ----
NORM_WINDOW, NORM_MINP = 252, 60        # 因果归一化窗口(normalize_causal), 同现有体系
MIN_CS = 8                              # 每日截面最小样本数(ETF 池标的少, 取 8)
TOP_K, EXIT_RANK = 5, 12                # 入场 / 退出 rank 门槛(滞回缓冲)
COST_BPS = 10.0                         # 单边交易成本(基点)

# pkl 中需从 object 数值化的列(与生产 pkl 一致)
OBJECT_COLS = ['trend_score', 'trend_score_change']


def pool_pkl(pool: str, interval: str = DEFAULT_INTERVAL, research: bool = True) -> Path:
  """
  池名 -> pkl 路径.

  命名约定(与生产一致): {pool}_{interval}_ta_data_research.pkl 为研究版(全史),
  {pool}_{interval}_ta_data.pkl 为生产版(盘前更新, 通常更短).

  :param pool: 池名, 如 etf_3x
  :param interval: 频率 day/week/month
  :param research: True 取研究版全史; False 取生产版
  :returns: Path(可能不存在, 由调用方决定是否回退)
  """
  suffix = 'ta_data_research' if research else 'ta_data'
  return PKL_DIR / f'{pool}_{interval}_{suffix}.pkl'


def available_pools(interval: str = DEFAULT_INTERVAL) -> list:
  """扫描 pkl 目录, 列出当前可用的研究版池名(用于报错提示与 CLI)."""
  if not PKL_DIR.exists():
    return []
  tag = f'_{interval}_ta_data_research.pkl'
  return sorted(p.name[:-len(tag)] for p in PKL_DIR.glob(f'*{tag}'))


# ---- 日志 ----
_ROOT_DONE = False


def get_logger(name: str = 'factor', prefix: str = 'factor',
               level: int = logging.INFO) -> logging.Logger:
  """
  返回 logger. handler(文件 + 控制台)只在首次调用时挂到 root, 之后复用;
  因此各子模块 get_logger('factor.xxx') 只会输出一次, 且格式统一.

  :param name: logger 名(如 'factor.data')
  :param prefix: 日志文件名前缀(文件名 = {prefix}_{日期}.txt)
  :param level: 日志级别
  :returns: logging.Logger
  """
  global _ROOT_DONE
  root = logging.getLogger()
  if not _ROOT_DONE:
    root.setLevel(level)
    if not root.handlers:                      # 幂等: 已配过则不重复挂
      LOG_DIR.mkdir(parents=True, exist_ok=True)
      fh = logging.FileHandler(LOG_DIR / f'{prefix}_{datetime.date.today()}.txt',
                               encoding='utf-8')
      fh.setFormatter(logging.Formatter(
          '[%(asctime)s] - [%(levelname)s] - [%(name)s] - %(message)s'))
      ch = logging.StreamHandler()
      ch.setFormatter(logging.Formatter('[%(asctime)s] - %(message)s',
                                        '%Y-%m-%d %H:%M:%S'))
      root.addHandler(fh)
      root.addHandler(ch)
    _ROOT_DONE = True
  return logging.getLogger(name)

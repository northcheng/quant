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
import json
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
# ⚠ 下面是**默认/回退**窗口, 不是"唯一真源". 每个池的真实窗口写在
#   factor_config.json 的 pools[pool].meta.is_window / oos_window(由 report --write-config 导出).
#   历史问题(2026-09 审查发现): 代码一律读这里的全局常量, 于是
#     * etf_3x/hs300/a_etf_all 的 config 声称 IS=2021-2024 / OOS=2025+, 而流水线实际用 2011-2019/2020-2023;
#     * wf_strat_search.py 又自己硬编码了第三套 2021-2024/2025;
#     * ml/config.py 用第四套.
#   结果: "报告的样本外"对线上冻结权重可能其实是样本内.
#   现在统一走 resolve_windows(pool): **有该池的显式窗口就用它**, 没有才回退到这里.
#   (原来的长历史三段切分, 供长历史池 company_* 使用)
#   IS     2011-01 ~ 2019-12: 选参(定权/Top-K/去重) —— 红线: 权重只在这里估计
#   OOS    2020-01 ~ 2023-12: 样本外复验(独立窗口, 不含"近期")
#   recent 2024-01 ~ 至今    : 近期表现(仅事后汇报, 不参与任何定参)
TRADE_START = '2011-01-01'              # 交易窗口起点(= IS 起点)
IS_END = '2019-12-31'                   # 样本内(IS)终点: 选参
OOS_START = '2020-01-01'                # 样本外(OOS)起点: 复验
OOS_END = '2023-12-31'                  # 样本外(OOS)终点
RECENT_START = '2024-01-01'             # 近期窗起点(None = 不设近期待)

# ---- 逐池窗口解析(单一真源) ----
_WINDOW_CACHE: dict = {}


def _load_factor_config(path=None) -> dict:
  """读 factor_config.json(缺失/损坏时返回 {}, 不抛异常 —— 本函数是"尽力而为"的读取)."""
  p = Path(path or FACTOR_CONFIG)
  try:
    with open(p, 'r', encoding='utf-8') as f:
      d = json.load(f)
    return d if isinstance(d, dict) else {}
  except Exception:
    return {}


def resolve_windows(pool: str, path=None) -> dict:
  """
  解析某池的评估窗口 -> {'is': (start, end), 'oos': (start, end), 'recent': (start, None)}.

  优先级: factor_config.json 里该池的 `meta.is_window` / `meta.oos_window`
          > 本模块的全局默认(TRADE_START/IS_END/OOS_START/OOS_END/RECENT_START).

  这样"报告的窗口"与"生产冻结权重所用的窗口"来自同一处, 不会再出现
  "报告标 OOS、线上权重其实拟合在该窗内"的问题.

  :param pool: 池名; 传 None / '' 时直接返回全局默认
  :param path: factor_config.json 路径(None = cfg.FACTOR_CONFIG)
  :returns: {'is': (start, end), 'oos': (start, end), 'recent': (start, None)}
  :raises: none
  """
  global _WINDOW_CACHE
  key = (pool or '', str(path or FACTOR_CONFIG))
  if key in _WINDOW_CACHE:
    return _WINDOW_CACHE[key]

  # 全局默认(长历史三段)
  win = {'is': (TRADE_START, IS_END),
         'oos': (OOS_START, OOS_END),
         'recent': (RECENT_START, None),
         'source': 'global'}

  conf = ((_load_factor_config(path).get('pools') or {}).get(pool) or {}) if pool else {}
  meta = conf.get('meta') or {}
  is_w, oos_w = meta.get('is_window'), meta.get('oos_window')
  # 只在窗口形状合法(2 元素列表/元组, 且 is 两端都给出)时才采用配置值
  if (isinstance(is_w, (list, tuple)) and len(is_w) == 2 and is_w[0]):
    win['is'] = (is_w[0], is_w[1])
    if isinstance(oos_w, (list, tuple)) and len(oos_w) == 2 and oos_w[0]:
      win['oos'] = (oos_w[0], oos_w[1])
      # recent 从 **OOS 结束的次日** 起算(后闭区间语义: 直接拿 oos_end 当 recent 下界
      # 会把 OOS 最后一天重复计入). 用纯标准库做 +1 天.
      #
      # ⚠️ oos_end 为 None 表示"OOS 一直开到数据末尾"(如 etf_3x 的 oos=2025-01-01~None).
      #    此时**没有独立的 recent 窗口** —— OOS 已经覆盖到最新. 必须显式置 (None, None).
      #    历史 bug(2026-09-30 发现): 旧代码直接 strptime(None) 抛 TypeError, 被
      #    `except (ValueError, TypeError)` 吞掉后同样落到 (None, None) —— 值一样,
      #    但**语义是坏的**: window_bounds('recent', pool=...) 把 (None, None) 原样返回,
      #    调用方(backtest/evaluate --show/wf_strat_search)会把它当成"不限起点"= **全窗**,
      #    于是"recent 表现"实际算成了全历史, 且数字看起来完全正常、不会报错.
      #    现在改成显式判断, 并把语义写进 'recent_note' 供调用方区分。
      if oos_w[1] is None:
        win['recent'] = (None, None)
        win['recent_note'] = 'OOS 开放到数据末尾, 无独立 recent 窗'
      else:
        try:
          _d = datetime.datetime.strptime(str(oos_w[1]), '%Y-%m-%d').date()
          win['recent'] = ((_d + datetime.timedelta(days=1)).isoformat(), None)
          win['recent_note'] = f'从 OOS 末({oos_w[1]})次日开始'
        except (ValueError, TypeError):
          win['recent'] = (None, None)
          win['recent_note'] = f'OOS 末({oos_w[1]!r})无法解析, recent 置空'
    win['source'] = 'factor_config.json'

  _WINDOW_CACHE[key] = win
  return win

# ---- 口径常量 ----
NORM_WINDOW, NORM_MINP = 252, 60        # 因果归一化窗口(normalize_causal), 同现有体系
MIN_CS = 8                              # 每日截面最小样本数(ETF 池标的少, 取 8)
MIN_FACTORS = 1                         # 每格最少有效因子数(缺失感知归一化下, 太少会让单因子被顶到满量程产生伪信号)
TOP_K, EXIT_RANK = 5, 12                # 入场 / 退出 rank 门槛(滞回缓冲)
COST_BPS = 10.0                         # 单边交易成本(基点)
MAX_ABS_RET = 1.0                       # 单日|收益|上限: 超过视为坏数据(A 股日涨跌幅远小于 100%)
VERIFY_CUT = '2024-06-28'               # 因果/无前视检验的默认截断日(须 >= IS 末, 落在近期窗内)

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

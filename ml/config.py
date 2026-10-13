# -*- coding: utf-8 -*-
"""
ml.config — 机器学习体系全局配置
================================
集中管理本包自己的常量, 并把"与 factor 体系必须对齐的口径"直接桥接到
`quant.factor.config`, 避免两边各写一份导致窗口/成本/最小截面数漂移.

设计原则:
  - 口径(时间窗 / MIN_CS / 成本)一律引用 factor.config, 不复制数值;
  - 只新增 ML 特有的选择(预测跨度 h / 标签形式 / 特征预处理方法 / 输出目录);
  - 输出写到本包 output/, 不污染 factor 的产物.

时间窗(与 factor 完全一致, 长历史三段切分):
  IS     2011-01 ~ 2019-12  定参(训练/验证/调参) —— 红线区
  OOS    2020-01 ~ 2023-12  样本外复验(独立窗口)
  recent 2024-01 ~ 至今     近期表现(仅事后汇报)
"""
from pathlib import Path

from quant.factor import config as fcfg

# ---- 路径 ----
ML_DIR = Path(__file__).resolve().parent
OUT_DIR = ML_DIR / 'output'                 # 本包产物根(报告/模型/预测)

# ---- 数据池 ----
# 合并池别名: 值 = (成员池, ...); ml.data.load() 认出别名后逐池读入面板, 按 symbol 去重取并集,
# 返回一个池名 = 别名的合成 Dataset. 用途: 单池样本量少(etf_3x 仅 28 标的)时,
# 用公司池把训练宇宙撑大 -> 特征/标签/拟合都在大宇宙上做, 打分后再切回目标池交易.
#   us_merged = etf_3x(28) ∪ company_300(297) ∪ company_1000(1049) -> 去重后 1077 标的
#   (实测: 三池列完全相同; company_300 完全含于 company_1000; etf_3x 与公司池零重叠)
POOL_ALIASES = {
    'us_merged': ('etf_3x', 'company_300', 'company_1000'),
}

# ---- 口径(桥接 factor, 单一真源) ----
MIN_CS = fcfg.MIN_CS                        # 每日截面最小样本数
COST_BPS = fcfg.COST_BPS                    # 单边交易成本(基点)
TOP_K, EXIT_RANK = fcfg.TOP_K, fcfg.EXIT_RANK

# 三段窗口(统一口径); None 表示不设边界
#
# ⚠ 修正(2026-09): 原来是**模块级常量**, 全进程只用一套窗口 —— 而 factor_config.json 里
# 各池的 meta.is_window / oos_window 并不相同(长历史池 company_* 是 2011-2019/2020-2023,
# 而 etf_3x/hs300/a_etf_all 是 2021-2024/2025+). 于是 ML 的 IS/OOS 与因子研究/生产冻结
# 权重可能根本不是同一套窗口, 两边结论不可比.
# 现在统一走 factor.config.resolve_windows(pool): 该池 config 优先, 缺失才回退全局默认.
IS = (fcfg.TRADE_START, fcfg.IS_END)        # 仅作"未指定池"时的默认
OOS = (fcfg.OOS_START, fcfg.OOS_END)
RECENT = (fcfg.RECENT_START, None)
WINDOWS = {'IS': IS, 'OOS': OOS, 'recent': RECENT, 'full': (None, None)}


def windows_for(pool: str) -> dict:
  """
  取某池的窗口字典 -> {'IS': (s, e), 'OOS': (s, e), 'recent': (s, None), 'full': (None, None)}.

  单一真源: factor.config.resolve_windows(pool). 所有 ml 入口在切 IS/OOS 之前都应调它,
  不要再直接用模块级 WINDOWS(那是全局默认, 可能与池的实际口径不符).

  :param pool: 池名
  :returns: 窗口字典(键名与 WINDOWS 一致, 大写 IS/OOS)
  :raises: none
  """
  w = fcfg.resolve_windows(pool)
  return {'IS': w['is'], 'OOS': w['oos'], 'recent': w['recent'], 'full': (None, None)}


# 进程级"当前池": 由各入口(CLI main)在加载数据后设置一次, 这样深处只拿得到
# SampleSet/FeaturePanel(不带 pool 信息)的函数也能切到该池的正确窗口.
# 未设置时回退全局默认, 行为与改动前一致(安全回退).
_CURRENT_POOL: str = None


def set_current_pool(pool: str) -> None:
  """
  设置本进程的当前池(供 window_range/window_bounds 在未显式传 pool 时使用).

  :param pool: 池名; None 表示回到全局默认
  :returns: none
  :raises: none
  """
  global _CURRENT_POOL
  _CURRENT_POOL = pool


def current_pool() -> str:
  """:returns: 本进程当前的池名(None = 未设置, 将使用全局默认窗口)."""
  return _CURRENT_POOL


def window_bounds(name: str, pool: str = None) -> tuple:
  """
  窗口名 -> (start, end).

  解析顺序: 显式 `pool` > `set_current_pool()` 设过的进程级池 > 全局默认(cfg.WINDOWS).
  因此 CLI 入口只要在加载数据后调一次 set_current_pool(pool), 深层的切片就自动
  用上该池的真实窗口(来自 factor_config.json), 不需要逐个函数透传 pool.

  :param name: 'IS' / 'OOS' / 'recent' / 'full'
  :param pool: 池名; None = 用进程级池 / 全局默认
  :returns: (start, end); None 表示不设边界
  :raises: KeyError: name 未知
  """
  effective = pool or _CURRENT_POOL
  table = windows_for(effective) if effective else WINDOWS
  if name not in table:
    raise KeyError(f'未知窗口: {name} (可选 {list(table)})')
  return table[name]

# ---- ML 特有选择 ----
HORIZONS = (1, 5, 10)                       # 支持的预测跨度(开盘到开盘, 交易日)
DEFAULT_HORIZON = 5                         # 默认跨度: 兼顾信噪比与换手

LABEL_METHODS = ('rank', 'z', 'raw')        # 标签形式: 截面秩(默认) / 截面 zscore / 原始收益(截尾)
DEFAULT_LABEL = 'rank'                      # 与 IC(RankIC) 口径同构, 可比性最强

# 特征预处理: 直接复用 factor.prepare 的方法名(raw/z/rank/mad_z/q_z).
# 默认 rank —— 因子尺度各异, 逐日截面秩变换后天然可比, 且对极值免疫.
FEATURE_PREP = 'rank'

# 特征来源: factor(注册表因子, 已过因果检验) / ta(pkl 面板的数值 TA 列) / both.
# ⚠ TA 列是 pkl 预计算值, 无独立生成器 -> 因果性无法用截断检验验证(见 features.py).
# 默认 'both'(因子 + TA 并集): TA 列已纳入特征, 但其因果性未验证, 下游结论需谨慎解读.
FEATURE_SOURCES = ('factor', 'ta', 'both')
DEFAULT_FEATURE_SOURCE = 'both'

# TA 列退化过滤(剔除对逐日截面排序无意义的列).
# 统计量只在 TA_FILTER_WINDOW 上算 —— 特征选择属全局决策, 用全样本会偷看 OOS(红线).
TA_MIN_VALID = 0.5          # 有效值占比下限(低于此判为近乎全 NaN)
TA_MAX_FLAT = 0.9           # 逐日截面恒定占比上限(超过此判为退化: rank 后塌缩为 0)
TA_FILTER_WINDOW = 'IS'     # 退化统计的取值窗口

# ---- 模型层(第4步) ----
# Ridge 基线: L2 强度. 特征已在特征层做过逐日截面 rank(取值 [-1,1]), 尺度可比,
# 故 alpha 无需按列缩放; 具体取值靠 IS 内的时间序列验证挑选(见 train.py), 这里只给默认.
RIDGE_ALPHA = 1.0
# 线性模型的缺失填补统计量(每列中位数)只在 fit 的样本上估计 -> fit 只喂 IS 即不偷看未来.
FILL_STRATEGY = 'median'    # 'median' / 'mean' / 'zero'

# GBDT 默认超参(LightGBM). ⚠ 下面是**未经 IS 内寻优的默认值**, 仅作骨架跑通与基线,
# 后续需在 IS 内用 walk-forward 挑选; 报告里必须标注"未调参", 否则对比失真.
# LightGBM 原生吃 NaN -> 无需像 Ridge 那样做缺失填补(这是两条路径的关键差异).
GBDT_PARAMS = dict(
    objective='regression', learning_rate=0.05, num_leaves=31,
    min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_lambda=5.0, n_estimators=500,
    n_jobs=4, seed=42, deterministic=True, verbose=-1)
GBDT_EARLY_STOP = 50        # 早停轮数(在 IS 内部时序验证块上)
GBDT_VALID_FRAC = 0.2       # 早停验证块占 IS 交易日比例

# MLP 默认超参(PyTorch, 仅 CPU). ⚠ 同 GBDT, 下面是**未经 IS 内寻优的默认值**, 仅作骨架
# 跑通与基线对比, 报告里必须标注"未调参". 设计取向: 数据低信噪比 + 样本量小 -> 刻意压小容量.
# 与 Ridge 一样, MLP 不原生吃 NaN -> 先按列中位数填补(统计量只在 fit 样本上算, 红线).
MLP_HIDDEN = (64, 32)       # 隐层宽度(2 层, 刻意小: 214 特征 + ~6 万样本, 容量过大会秒过拟合)
MLP_DROPOUT = 0.2           # 每层 dropout(正则)
MLP_LR = 1e-3               # Adam 学习率
MLP_WEIGHT_DECAY = 1e-2     # Adam 权重衰减(L2). ⚠ 经 IS 内 --search(lr×wd 网格, 判据=验证块
                            # RankIC)选出; 结论: 该网格内验证 RankIC 几乎持平(0.067~0.073),
                            # 降 lr 能让模型多训几轮(best_epoch 1->12), 但对 OOS 增量微弱.
MLP_EPOCHS = 200            # 最多训练轮数(有 valid 时会被早停提前截断)
MLP_BATCH_SIZE = 2048       # mini-batch 大小
MLP_PATIENCE = 20           # 早停耐心(连续多少轮验证损失不改善就停)
MLP_SEED = 42               # 单 seed(先拿可比数字, 不做多 seed 集成)
MLP_VALID_FRAC = 0.2        # 早停验证块占 IS 交易日比例(同 GBDT)

# 序列模型默认超参(PyTorch GRU / LSTM, 仅 CPU). gru 已做 IS 内 --search(L×lr×wd, 判据=IS 内
# 验证块 RankIC); 下方 L/lr/wd 为选定值, 其余仍为默认. 全 IS 重拟合后与"默认"对比见 SEQ_LR 注释.
# lstm 复用**同一组 SEQ_* 超参**与同一份窗口/标签/早停协议, 以便与 gru 严格同口径对照
# (只换循环单元, 其余全同) —— 定位是"GRU 的门控结构相对 LSTM 有没有增量"的对照臂.
# 样本构造(见 ml/seq.py): 第 t 日某标的的输入 = 该标的 D_{t-L+1}..D_t 的 L 天截面特征窗口
# (**窗口末端 = 信号日 t, 只含 <= t 的历史**); 标签 = t 日的 h 日前向 rank 标签 —— 与
# ridge/gbdt/mlp 用的是**同一份标签**, 故性能可比. 特征来源见下方 SEQ_FEATURE_SOURCE.
# 设计取向同 MLP: 低信噪比 + 样本量不大 -> 容量刻意压小(单层, hidden 32).
SEQ_WINDOW = 20             # 输入窗口长度 L(交易日): D_{t-19}..D_t
SEQ_STRIDE = 1              # 信号日起步(按交易日); >1 降采样窗口, 省内存/算力
SEQ_HIDDEN = 32             # 循环单元隐层宽度(刻意小)
SEQ_LAYERS = 1              # 循环单元层数
SEQ_DROPOUT = 0.2           # 输出头 dropout(PyTorch 的 RNN dropout 仅层数 > 1 时生效)
SEQ_LR = 3e-4               # Adam 学习率. ⚠ 经 IS 内 --search(L×lr×wd 网格, 判据=验证块
                            # RankIC)选出: 最优组 L=20/lr=3e-4/wd=1e-2(valid RankIC 0.0403).
                            # 结论: 降 lr 让模型多训几轮(best_epoch 2->5), 全 IS 重拟合后 OOS
                            # RankIC 0.0308->0.0361, recent 0.0196->0.0257; L=20 略优于 L=10.
SEQ_WEIGHT_DECAY = 1e-2     # Adam 权重衰减(L2). ⚠ 同 SEQ_LR 由 IS 内 --search 选出(1e-2 略优于 1e-4)
SEQ_EPOCHS = 60             # 最多训练轮数(有 valid 时被早停截断)
SEQ_BATCH_SIZE = 1024       # mini-batch 大小
SEQ_PATIENCE = 8            # 早停耐心(连续多少轮验证损失不改善即停)
SEQ_GRAD_CLIP = 1.0         # 梯度裁剪阈值(GRU/LSTM 训练稳定性)
SEQ_SEED = 42               # 单 seed(先拿可比数字, 不做多 seed 集成)
SEQ_VALID_FRAC = 0.2        # 早停验证块占 IS 交易日比例(同 GBDT/MLP)

# 序列模型特征来源: 与截面模型的 DEFAULT_FEATURE_SOURCE 对齐(默认 'both' = 因子 59 + TA 并集),
# 使 gru 与 ridge/gbdt/mlp 在同一特征口径下可比. ⚠ TA 列疑似含 Split.shift(-1) 的 1 日前视,
# 不可用截断一致性检验验证; 放进长度 L 的窗口后该前视会被放大 L 倍 -> 构建时会打印 TA_CAVEAT,
# 含 TA 的 OOS 结论仅作参考(见 features.TA_CAVEAT / seq.SEQ_SOURCE).
SEQ_FEATURE_SOURCE = DEFAULT_FEATURE_SOURCE

# 支持的模型名(工厂见 models/__init__.py)
MODELS = ('ridge', 'gbdt', 'mlp', 'gru', 'lstm')
SEQ_MODELS = ('gru', 'lstm')  # 需要"序列样本"(SequenceSampleSet)的模型; 其余吃扁平 SampleSet
DEFAULT_MODEL = 'ridge'

# ---- 择时层(第5步, 方案B: 波动率目标; 红线: 不做空/不加杠杆) ----
# 择时只回答"今天整体上多少仓", 与选股层(分数)解耦; 详见 ml/timing.py.
TIMING_TARGET_VOL = 0.15      # 目标年化波动率(⚠ 默认值; 实际须在 IS 内估计, 见 timing.py)
TIMING_VOL_SHORT = 20         # 短期波动窗口(交易日)
TIMING_VOL_LONG = 60          # 长期波动窗口(交易日)
TIMING_MOM_WINDOW = 20        # 动量 / 宽度 / 离散的滚动窗口(交易日)
TIMING_DD_WINDOW = 252        # 回撤回看窗口(约一年)
TIMING_LABEL_H = DEFAULT_HORIZON   # 波动率标签的未来窗口(默认与选股跨度一致)
TIMING_EXPOSURE_CAP = 1.0     # 敞口上限(恒 = 1, 禁止杠杆)
TIMING_EXPOSURE_FLOOR = 0.0   # 敞口下限(恒 = 0, 禁止做空)

# 状态分类臂(t4 第三臂): 把"未来波动率"按 IS 分位二值化为 高/低波动状态, 再用低容量线性
# 概率模型判状态; 低波动 -> 满仓, 高波动 -> 降仓. 与波动率目标的差别是**离散状态 + 固定档位**.
TIMING_REGIME_THRESHOLD_Q = 0.5    # 波动率二值化分位(⚠ 只在 IS 内取, 红线)
TIMING_REGIME_ON_EXPOSURE = 1.0    # 低波动状态敞口(风险偏好)
TIMING_REGIME_OFF_EXPOSURE = 0.3   # 高波动状态敞口(风险规避; 仍 >= 0, 不做空/不加杠杆)

# 信号分数阈值臂(t5 第四臂): 用**选股分数本身**构造逐日总敞口 —— 与 voltarget/regime
# 的信息源不同(后者看市场波动率, 这里只看"今天模型给出的信号有多强/多广"). 详见 timing.py 第 9 节.
TIMING_SCORE_STAT = 'breadth'   # 'breadth'(超阈值占比) / 'topk_mean'(TopK 平均分)
TIMING_SCORE_TAU_Q = 0.8        # breadth 的分数阈值分位(⚠ IS 内取; 0.8 = 只数"模型强烈看好"的标的)
TIMING_SCORE_LO_Q = 0.1         # 敞口线性映射下限分位(⚠ IS 内取)
TIMING_SCORE_HI_Q = 0.9         # 敞口线性映射上限分位(⚠ IS 内取)


def window_range(name: str, pool: str = None):
  """
  窗口名 -> (start, end) 元组(供切片 / 掩码使用).

  :param name: 'IS'/'OOS'/'recent'/'full'
  :param pool: 池名; 给了就按该池口径解析(见 windows_for), 否则用全局默认
  :returns: (start, end); None 表示该侧不限
  :raises: KeyError: name 未知
  """
  return window_bounds(name, pool=pool)


def get_logger(name: str = 'ml'):
  """
  本包 logger. 复用 factor.config 的 handler 装配(日志格式/落盘一致),
  文件名前缀用 'ml', 与 factor 日志分离.

  :param name: logger 名(如 'ml.features')
  :returns: logging.Logger
  """
  return fcfg.get_logger(name, prefix='ml')
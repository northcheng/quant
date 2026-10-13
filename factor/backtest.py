# -*- coding: utf-8 -*-
"""
factor.backtest — 回测层(第5步)
================================
职责: 把第4步的组合分数(宽表)变成"一条真正可交易的净值曲线", 并给出绩效.

--- 这一层与前几层的根本区别 ---
  第1~4步都在回答"信号有没有预测力"(截面相关 / IC);
  本层回答"照这个信号交易, 到底赚不赚钱、承担多大风险、扣完成本还剩多少".
  两者不能互相替代: RankIC 0.04 的信号若换手极高, 扣成本后可能归零.

--- 防前视: 时间线只有一条 ---
  信号日 s 收盘: 读 composite.loc[s] 与当时持仓 -> 决定"明天开盘买卖什么"
  执行日 d 开盘: 先按 open(s) -> open(d) 结算, 再换仓, 最后收成本
  => 决策只用 s 及之前的信息, 成交发生在 d, 收益从 d 开盘起算. 全程不碰未来.
  (这与第3步"标签必须用未来"并不矛盾: 那条管评估口径, 这条管真实交易流程.)

--- rank -> 权重 ---
  1. 截面 rank(1 = 分数最高);
  2. 入场: rank <= top_k; 退出: 已持仓且 rank > exit_rank(或信号/价格缺失);
     top_k 与 exit_rank 之差即"滞回缓冲": 刚买进的票掉到第 6~12 名不会被立刻卖出,
     避免在门槛附近反复摩擦. top_k == exit_rank 时缓冲为 0, 换手最高.
  3. 已选持仓内按 sizing 分权重: equal(等权) / tier(三档 1.5/1.0/0.5) /
     linear(0.5~1.5 线性) / invvol(逆波动 1/σ, 需传 vol 面板); 归一化到
     max_exposure(可被 exposure 逐日覆盖), 再受 per_symbol_cap 单票上限约束.
  4. band: 无进出场且目标与当前权重最大偏差 < band 时, 维持漂移权重不交易(省成本).

--- 成本 ---
  turnover = Σ|target_w - 当前_w|; 净值按 (1 - turnover * cost_rate) 计提,
  cost_rate = COST_BPS / 10000(单边). 一次换仓的买卖两侧都记成本.

--- 绩效 + IS/OOS ---
  perf_stats: total_ret / cagr / vol / sharpe / max_dd / calmar / ann_turnover /
              n_trades / win_rate / avg_ret / avg_days.
  权重只能在 IS 期估计(第4步红线), 再在 IS / OOS 两窗各自推进引擎对照 ——
  这才是"策略在样本外能否复现"的真正检验.

自检:
  cd ~/git && python -m quant.factor.backtest --pool etf_3x
  cd ~/git && python -m quant.factor.backtest --pool etf_3x --check
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import combine as cmb
from quant.factor import evaluate as ev
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import Dataset, load_pool, window as _win

SCHEMES = ('equal', 'ic', 'greedy')
SIZINGS = ('equal', 'tier', 'linear', 'invvol', 'strength')
STAT_KEYS = ['n_days', 'total_ret', 'cagr', 'vol', 'sharpe', 'max_dd', 'calmar',
             'ann_turnover', 'n_trades', 'win_rate', 'avg_ret', 'avg_days']
ANNUAL = 252
_BAD_PX_SEEN = set()                          # 去重: 键=(坏点数量, 列集合), 同池只告警一次


# ============================ 1. 引擎参数 ============================

class EngineParams:
  """
  回测引擎参数(全是"交易规则", 与因子/权重无关).

  这些参数也属于"参数", 严格说只能在 IS 期调; 本层默认值直接取 config 的口径常量,
  与现有生产体系对齐, 便于互相印证.
  """

  def __init__(self, top_k: int = cfg.TOP_K, exit_rank: int = cfg.EXIT_RANK,
               sizing: str = 'equal', max_exposure: float = 1.0,
               per_symbol_cap: float = 0.30, cost_bps: float = cfg.COST_BPS,
               band: float = 0.05, strength_threshold: float = None,
               capital: float = None, commission_per_order: float = 0.0):
    """
    :param top_k: 入场 rank 门槛(持仓数上限)
    :param exit_rank: 退出 rank 门槛(应 >= top_k, 差额为滞回缓冲)
    :param sizing: 持仓内分权重方式 equal / tier / linear / invvol / strength
    :param max_exposure: 总敞口上限(1.0 = 满仓)
    :param per_symbol_cap: 单票权重上限
    :param cost_bps: 单边交易成本(基点) —— 比例成本(与成交额成正比)
    :param band: 重平衡带宽(偏差小于该值不调仓)
    :param strength_threshold: 路线A(权重与标的个数解耦)的**入场强度门槛 θ**(截面 z 口径).
      None(默认) = 关闭, 行为与旧引擎逐字一致; 设值后: 新入场券需满足"rank<=top_k 且 z>=θ",
      因此**持仓只数可 < top_k**(达标标的少则集中), 空缺不再回填.
    :param capital: 账户本金(USD). None(默认) = 关闭资金量建模, 行为与旧引擎逐字一致;
      设值后额外叠加**按笔固定佣金**(见下), 用于小资金实盘可行性 / 手续费拖累评估.
    :param commission_per_order: **每笔固定佣金**(USD, 含券商最低收费与平台费).
      当日固定成本(USD) = 成交笔数 * commission_per_order; 折算成当日收益拖累时,
      以**当日账户市值**(= capital * 当前 equity)为分母, 故本金越薄、持仓越散, 拖累越大.
      成交笔数 = 当日目标权重发生变化的标的数(买入 / 卖出 / 调仓各记一笔).
      仅在 capital 非空且本值 > 0 时生效; 否则成本仍只有 turnover * cost_bps 一项.
    """
    if sizing not in SIZINGS:
      raise ValueError(f'未知 sizing: {sizing} (可选 {SIZINGS})')
    self.top_k = int(top_k)
    self.exit_rank = int(exit_rank)
    self.sizing = sizing
    self.max_exposure = float(max_exposure)
    self.per_symbol_cap = float(per_symbol_cap)
    self.cost_rate = float(cost_bps) / 10000.0            # 单边
    self.band = float(band)
    self.strength_threshold = (None if strength_threshold is None
                               else float(strength_threshold))
    self.capital = None if capital is None else float(capital)
    self.commission_per_order = float(commission_per_order)

  def copy(self, **kw) -> 'EngineParams':
    """复制并覆盖若干字段(敏感性对照用)."""
    p = EngineParams(self.top_k, self.exit_rank, self.sizing, self.max_exposure,
                     self.per_symbol_cap, self.cost_rate * 10000.0, self.band,
                     self.strength_threshold, self.capital, self.commission_per_order)
    for k, v in kw.items():
      if k == 'cost_bps':                              # 存储的是 rate, 需换算
        p.cost_rate = float(v) / 10000.0
      else:
        setattr(p, k, v)
    return p

  def brief(self) -> str:
    s = (f'K={self.top_k}, exit_rank={self.exit_rank}, sizing={self.sizing}, '
         f'exposure={self.max_exposure}, cap={self.per_symbol_cap}, '
         f'cost={self.cost_rate * 10000:.0f}bps(单边), band={self.band}')
    if self.strength_threshold is not None:
      s += f', strength_thr={self.strength_threshold:g}(路线A: 截面z入场门槛)'
    if self.capital is not None:
      s += f', capital={self.capital:,.0f}USD'
      if self.commission_per_order:
        s += f', 佣金={self.commission_per_order:g}USD/笔'
    return s


# ============================ 2. 仓位系数 ============================

def _sizing_coef(sub: pd.Series, sizing: str) -> pd.Series:
  """
  已选持仓内部的仓位系数: tier 三档 1.5/1.0/0.5, linear 0.5~1.5, equal 1.0.

  sub 是持仓的**组合分数**(高分 = 好); 返回与 sub 同索引的系数.
  系数最后会被归一化, 所以只看相对大小.

  :param sub: 持仓的组合分数
  :param sizing: equal / tier / linear
  :returns: 与 sub 同索引的仓位系数
  """
  n = len(sub)
  if n <= 1:
    return pd.Series(1.0, index=sub.index)
  frac = (n - sub.rank(ascending=False, method='first')) / (n - 1)   # 最好=1, 最差=0
  if sizing == 'equal':
    return pd.Series(1.0, index=sub.index)
  if sizing == 'linear':
    return 0.5 + frac
  tier = np.ceil(frac * 3.0)                                        # 0..3, 最好=3
  coef = np.where(tier >= 3, 1.5, np.where(tier >= 2, 1.0, 0.5))
  return pd.Series(coef, index=sub.index)


def _invvol_coef(symbols: list, vol_row: np.ndarray, ccpos: dict) -> pd.Series:
  """
  持仓内**逆波动**系数: 1/σ(σ = 该标的信号日滚动波动率).

  风险平价(路径B)的最小实现: 波动大的标的小配, 波动小的大配 —— 在不改变持仓集合、
  不新增任何"拟合"自由度的前提下, 让组合风险更均衡(σ 窗口属结构性选择, 非拟合参数).
  σ 缺失/为 0 的标的回退到当日截面中位数; 全缺则退化为等权.

  :param symbols: 当前持仓标的
  :param vol_row: 信号日的横截面波动(按 ccols 排列)
  :param ccpos: 标的 -> 列号
  :returns: 与 symbols 同序的系数(仅相对大小有意义, 之后会归一化)
  """
  vals = {}
  for sym in symbols:
    j = ccpos.get(sym)
    v = vol_row[j] if j is not None else np.nan
    vals[sym] = (1.0 / v) if (v == v and v > 0.0) else np.nan
  s = pd.Series(vals)
  if s.notna().sum() == 0:
    return pd.Series(1.0, index=s.index)
  fill = float(s.median())
  return s.fillna(fill if fill == fill else 1.0)


def _z_panel(a: np.ndarray) -> np.ndarray:
  """
  逐行**截面 z-score** 面板: z = (x - 行均值) / 行标准差(只用非 NaN 格).

  手算 nansum/count 规避 np.nanmean 对全 NaN 行的告警; 全 NaN 行 -> 全 NaN.
  组合分数是逐日 rank 标准化的线性组合, 量纲不定且随池规模变化, 故一律用截面 z 比较.
  """
  cnt = np.sum(~np.isnan(a), axis=1, keepdims=True)
  ssum = np.nansum(a, axis=1, keepdims=True)
  with np.errstate(invalid='ignore', divide='ignore'):
    mu = ssum / cnt
    var = np.nansum((a - mu) ** 2, axis=1, keepdims=True) / cnt
    return (a - mu) / np.sqrt(var)


def _strength_coef(symbols: list, z_row: np.ndarray, ccpos: dict, theta: float) -> pd.Series:
  """
  路线A(权重与标的个数解耦)的**持仓内强度系数**: 0.5 + max(z - θ, 0).

  z = 信号日组合分数的**截面 z-score**(标准化), θ = 入场门槛. 强度越高(相对截面越突出)
  系数越大 -> 单票可有 >1 份的权重; z<=θ 的标的退到基准 0.5 份(仍保留, 因为已在持仓内).
  与 _invvol_coef 一样, 仅相对大小有意义(之后集合内归一化 + per_symbol_cap 截断).

  :param symbols: 当前持仓标的
  :param z_row: 信号日的横截面 z(按 ccols 排列)
  :param ccpos: 标的 -> 列号
  :param theta: 入场强度门槛 θ(截面 z 口径)
  :returns: 与 symbols 同序的强度系数
  """
  vals = {}
  for sym in symbols:
    j = ccpos.get(sym)
    z = z_row[j] if j is not None else np.nan
    vals[sym] = 0.5 + (z - theta) if (z == z and z > theta) else 0.5
  return pd.Series(vals)


def topk_z_baseline(comp: pd.DataFrame, top_k: int, window=None,
                    stat: str = 'mean') -> float:
  """
  路线A2(池无关门槛): 用 IS 估计"Top-K 的**平均截面 z**"作入场基准 baseline.

  对 window(应传 IS 窗; None=全部)内每个交易日: 取 rank<=top_k 的标的, 求其截面 z 的均值,
  再对这些逐日值取 stat('mean'/'median'). 关键性质: **池越大, Top-K 天然落在分布更深尾部,
  其平均 z 越高** -> baseline 自动抬高, 于是把入场门槛写成 "z >= baseline - margin" 后,
  同一 margin 在不同规模池上对应**相同分位强度**(池无关), 替代路线A 的固定 θ(池相关).

  只用 window 内数据拟合 -> 无前视(look-ahead); 与 run_engine 共用 _z_panel, 口径一致.

  :param comp: 组合分数面板(日期 x 标的)
  :param top_k: 持仓上限 K(与引擎一致)
  :param window: (start, end) 字符串/None; 应传 IS 窗
  :param stat: 'mean'(默认) 或 'median'
  :returns: baseline(截面 z 口径); 无有效日 -> nan
  """
  c = comp if window is None else _win(comp, window[0], window[1])
  a = c.to_numpy(dtype=float)
  if not a.shape[0]:
    return float('nan')
  z = _z_panel(a)
  rank = c.rank(axis=1, ascending=False, method='first').to_numpy(dtype=float)
  daily = []
  for i in range(a.shape[0]):
    zz = z[i][rank[i] <= top_k]
    zz = zz[~np.isnan(zz)]
    if zz.size:
      daily.append(float(zz.mean()))
  if not daily:
    return float('nan')
  return float(np.mean(daily)) if stat == 'mean' else float(np.median(daily))


# ============================ 3. 逐日事件引擎 ============================

def _sanitize_prices(open_w: pd.DataFrame, close_w: pd.DataFrame,
                     max_abs_ret: float = cfg.MAX_ABS_RET):
  """
  价格卫生守卫: 剔除数据源事故造成的脏价格, 防止单日极端"收益"打爆净值.

  背景: 研究 pkl 里偶发脏行(如某日多只标的 Open=High=Low=Close 被写成同一个垃圾常数,
  真实价 ~4.5 却被写成 ~9e5). 引擎直接 pct_change() 会得到上万倍的假收益,
  使 total_ret/vol 爆炸(dir IS total_ret 8169 / vol 2308 即此因).

  判据只用"单日收益合理性": A 股(主板/双创/新股)正常单日涨跌幅远小于 100%,
  故 |r| > max_abs_ret 或 r <= -0.9 必为坏数据.

  做法: 对开/收盘各自 pct_change, 标出极端收益的**当日与其相邻日** —— 垃圾价常成段以
  常数出现, 只有"进入/离开"那天收益才极端, 中间日收益为 0 却仍是假价(会被当成成交价
  污染 avg_ret). 因此把 bad[t] 与其后一日 bad[t+1] 对应格都视为可疑, 一并置 NaN.
  置 NaN 后: 假收益不计入净值(NaN -> 0), 假价格不会被用作 entry_px/exit_px.

  :param open_w / close_w: 原始价格宽表(date × symbol)
  :param max_abs_ret: 单日收益绝对值上限(超过视为坏数据)
  :returns: (open_clean, close_clean, n_bad) —— n_bad = 被置 NaN 的格数(open + close)
  """
  def _clean(w: pd.DataFrame):
    r = w.pct_change()
    bad = (r > max_abs_ret) | (r <= -0.9)                # 进入本日的收益是否极端
    suspect = bad | bad.shift(-1, fill_value=False)      # 本日坏 或 次日坏(本日=离开日)
    return w.mask(suspect), int(suspect.to_numpy().sum())

  open_clean, n_o = _clean(open_w)
  close_clean, n_c = _clean(close_w)
  return open_clean, close_clean, n_o + n_c


def run_engine(open_w: pd.DataFrame, close_w: pd.DataFrame, composite: pd.DataFrame,
               p: EngineParams, exposure: pd.Series = None,
               vol: pd.DataFrame = None) -> dict:
  """
  逐日事件引擎: 组合分数 -> 净值曲线 + 持仓 + 成交流水.

  时间线(防前视核心, 全程只有这一条):
    信号日 s = dates[i-1] 收盘: 取 composite.loc[s] 与当时持仓 -> 目标权重
    执行日 d = dates[i]   开盘: 结算(open(s) -> open(d)), 漂移权重, 换仓, 收成本
  分数只用 s 行; 目标只作用于 d 开盘之后; 收益只从成交开盘起算, 不碰未来.

  近似说明: 决策时用的"当前权重"是已漂移到 d 开盘的值(而非 s 收盘盯市),
  只影响 band 微调与权重小数, 不影响进出场决策(持仓集合是确定的).

  首日处理: 为让 total_ret 口径准确, 在 daily 首行补一条 dates[0] 的 equity=1.0
  初始行(参考实现缺这一行, 会把首日成本算进分母, 略偏).

  :param open_w: 开盘价宽表(date × symbol)
  :param close_w: 收盘价宽表
  :param composite: 组合分数宽表
  :param p: EngineParams
  :param exposure: 可选的逐日总敞口 Series(索引 = 交易日). 取**信号日 s** 的值作为总仓位
      上限(与 composite.loc[s] 同一时点, 不构成前视). 缺值先按前值填充(仅用过去), 再退回
      p.max_exposure; 负值按 0 处理. None(默认) = 恒用 p.max_exposure.
      语义: 敞口 < 1 时其余为现金(0 收益); 是否前视取决于调用方给的序列是否因果.
  :param vol: 可选的逐日逐标的波动率宽表(date × symbol, 用于 sizing='invvol'). 取
      **信号日 s** 的横截面作为逆波动系数(与 composite.loc[s] 同一时点, 不构成前视).
      未对齐的格视为缺失(回退等权). None(默认) = 不支持 invvol(退化为等权).
  :returns: {'daily','pos','trades','equity','ann_turnover','params','n_bad_px'}
  """
  open_w, close_w, n_bad_px = _sanitize_prices(open_w, close_w)
  # 去重键带池标识(列集合): 只按坏点数量去重会让不同池共用同一"数量"而互相吞掉告警;
  # 带上列集合后, 语义回到"同一池同一次引擎运行只告警一次".
  px_sig = (n_bad_px, tuple(open_w.columns))
  if n_bad_px and px_sig not in _BAD_PX_SEEN:
    _BAD_PX_SEEN.add(px_sig)
    cfg.get_logger('factor.backtest').warning(
        '价格卫生守卫: 剔除 %d 个脏价格格(单日|收益|>%.2f 或 <= -0.9)', n_bad_px, cfg.MAX_ABS_RET)
  dates = open_w.index
  if len(dates) < 2:
    raise ValueError('交易日不足 2 天, 无法回测')
  rets = open_w.pct_change()                    # open(s) -> open(d) 的收益

  # --- numpy 预取: 把逐日的 .at 标量访问 / 逐日 rank 换成整块运算 ---
  # 数值等价: 取到的仍是同一批浮点值, 只是访问方式从 pandas 标量改成数组下标;
  # 且 Series.rank(ascending=False, method='first') 与该面板 rank(axis=1, ...) 逐行等价.
  cols = list(open_w.columns)
  opos = {s: j for j, s in enumerate(cols)}
  open_np = open_w.to_numpy(dtype=float)
  close_np = close_w.reindex(index=open_w.index, columns=cols).to_numpy(dtype=float)
  rets_np = rets.to_numpy(dtype=float)
  ccols = list(composite.columns)
  ccpos = {s: j for j, s in enumerate(ccols)}
  comp_row = {ts: k for k, ts in enumerate(composite.index)}
  comp_np = composite.to_numpy(dtype=float)
  rank_np = composite.rank(axis=1, ascending=False, method='first').to_numpy(dtype=float)
  order_np = np.argsort(rank_np, axis=1)        # 每行按 rank 升序的列号(NaN 排最后)
  # 路线A: 组合分数的**截面 z-score** 面板(逐行去均值除标准差). 全 NaN 行 -> z 全 NaN
  # (自然被门槛过滤). 与 topk_z_baseline() 共用 _z_panel, 保证标定口径与执行口径一致.
  z_np = _z_panel(comp_np) if p.strength_threshold is not None else None
  if exposure is None:                          # 逐日敞口: 对齐 dates, 缺值前填(因果)再退回常数
    exp_np = None
  else:
    exp_np = (exposure.reindex(dates).ffill().fillna(p.max_exposure)
              .clip(lower=0.0).to_numpy(dtype=float))
  if vol is None:                               # 逆波动 sizing 用的逐日逐标的波动(对齐 composite)
    vol_np = None
  else:
    vol_np = vol.reindex(index=dates, columns=ccols).to_numpy(dtype=float)
  last_i = len(dates) - 1

  weights = {}                                  # symbol -> 执行后权重(以执行日开盘市值计)
  open_trades = {}                              # symbol -> 未平仓段信息
  trades, daily, pos_rows = [], [], []
  equity, turnover_sum, n_periods = 1.0, 0.0, 0

  for i in range(1, len(dates)):
    d, s = dates[i], dates[i - 1]
    n_periods += 1

    # --- A. 结算: weights 于 open(s) 建立, 持有至 open(d) ---
    rrow = rets_np[i]
    gross = 0.0
    if weights:
      acc = 0.0
      for sym, w in weights.items():
        r = rrow[opos[sym]]
        acc += w * (0.0 if r != r else r)       # NaN -> 0.0(与 pd.notna 等价)
      gross = float(acc)
    equity *= (1.0 + gross)
    if weights and gross > -1.0:                # 权重随价格自然漂移(仍和 = 1)
      nw = {}
      for sym, w in weights.items():
        r = rrow[opos[sym]]
        nw[sym] = w * (1.0 + (0.0 if r != r else r)) / (1.0 + gross)
      weights = nw

    # --- B. 信号日 s 收盘: 决定"明天开盘买卖谁" ---
    held = list(weights.keys())
    ci = comp_row.get(s)                             # s 不在 composite -> 该日 rank 全 NaN
    if ci is None:
      exit_set = list(held)                          # order 视为空 -> 所有持仓分数缺失
      cand = []
    else:
      rr, orow = rank_np[ci], order_np[ci]
      exit_set = []                                  # 退出: 分数缺失 或 排名掉出 exit_rank
      for sym in held:
        j = ccpos.get(sym)
        r = rr[j] if j is not None else np.nan
        if r != r or r > p.exit_rank:
          exit_set.append(sym)
      zr = z_np[ci] if z_np is not None else None   # 路线A: 该信号日的截面 z 行(θ 关闭则 None)
      cand = []                                      # 入场候选: rank<=top_k & 未持有 & 有开盘价
      for j in orow:                                 # 已按 rank 升序(NaN 在末尾)
        r = rr[j]
        if r != r or r > p.top_k:
          break
        sym = ccols[j]
        if sym in weights:
          continue
        k = opos.get(sym)
        if k is None or open_np[i, k] != open_np[i, k]:
          continue
        if zr is not None:                           # 路线A: 强度不足 -> 不发入场券(空缺不回填)
          z = zr[j]
          if not (z == z and z >= p.strength_threshold):
            continue
        cand.append((float(r), sym))
    for sym in exit_set:                             # 挂单卖出(价格缺失则留待下一日再试)
      k = opos.get(sym)
      px = open_np[i, k] if k is not None else np.nan
      t = open_trades.pop(sym, None)
      if t is None:
        continue
      if px == px:
        trades.append({'symbol': sym, 'entry_exec': t['entry_exec'], 'exit_exec': d,
                       'entry_px': t['entry_px'], 'exit_px': float(px),
                       'entry_w': t['entry_w'], 'peak_w': t['peak_w'],
                       'days': i - t['entry_i'], 'ret': float(px) / t['entry_px'] - 1.0,
                       'exit_reason': 'rank', 'completed': True})
      else:
        open_trades[sym] = t
    slots = max(0, p.top_k - (len(held) - len(exit_set)))
    entries = [sym for _, sym in cand[:slots]]
    new_held = [sym for sym in held if sym not in exit_set] + entries
    if not new_held and cand:                        # 保底: 达标候选非空却无持仓 -> 取最强一只
      entries = [cand[0][1]]
      new_held = list(entries)

    # --- C. 目标权重 ---
    target = {}
    if new_held:
      if ci is None:
        sub = pd.Series(np.nan, index=new_held)      # 等价于空 comp.reindex(new_held)
      else:
        sub = pd.Series([comp_np[ci, ccpos[sym]] if sym in ccpos else np.nan
                         for sym in new_held], index=new_held)
      if p.sizing == 'strength':                     # 路线A: 持仓内按截面 z 强度加权
        if z_np is not None and ci is not None:
          coef = _strength_coef(new_held, z_np[ci], ccpos, p.strength_threshold)
        else:
          coef = _sizing_coef(sub, 'equal')          # 无 z 面板 -> 退化为等权
      elif p.sizing == 'invvol':                     # 逆波动(路径B): 用信号日 σ 定仓
        if vol_np is not None and ci is not None:
          coef = _invvol_coef(new_held, vol_np[ci], ccpos)
        else:
          coef = _sizing_coef(sub, 'equal')          # 无 vol 面板 -> 退化为等权
      else:
        coef = _sizing_coef(sub, p.sizing)
      tot = float(coef.sum())
      e_s = p.max_exposure if exp_np is None else float(exp_np[i - 1])  # 信号日 s 的总敞口
      if tot > 0:
        target = {sym: min(float(coef[sym] / tot * e_s), p.per_symbol_cap)
                  for sym in new_held}
    if not exit_set and not entries and weights:     # band: 无进出场且偏差小 -> 维持漂移
      maxdev = max((abs(target.get(sym, 0.0) - weights.get(sym, 0.0))
                    for sym in set(target) | set(weights)), default=0.0)
      if maxdev < p.band:
        target = dict(weights)

    # --- D. 换手与成本: 单边比例成本(买卖两侧都记) + 可选按笔固定佣金 ---
    turnover, n_orders = 0.0, 0
    for sym in set(target) | set(weights):
      dw = abs(target.get(sym, 0.0) - weights.get(sym, 0.0))
      if dw > 1e-9:
        turnover += dw
        n_orders += 1
    prop_cost = turnover * p.cost_rate                 # 比例成本(与成交额成正比)
    commission_usd = 0.0
    fixed_cost = 0.0
    if p.capital and p.commission_per_order and equity > 1e-9:
      commission_usd = n_orders * p.commission_per_order
      fixed_cost = commission_usd / (p.capital * equity)   # 固定佣金占当前账户市值的比例
    cost_drag = prop_cost + fixed_cost
    equity *= (1.0 - cost_drag)
    turnover_sum += turnover
    weights = {sym: w for sym, w in target.items() if w > 1e-9}
    for sym in entries:                              # 记录新开仓段
      if sym in weights:
        open_trades[sym] = {'entry_exec': d, 'entry_px': float(open_np[i, opos[sym]]),
                            'entry_w': weights[sym], 'peak_w': weights[sym], 'entry_i': i}
    for sym, t in open_trades.items():
      t['peak_w'] = max(t['peak_w'], weights.get(sym, 0.0))
    daily.append({'date': d, 'equity': equity, 'gross_ret': gross, 'cost_drag': cost_drag,
                  'turnover': turnover, 'n_pos': len(weights), 'n_orders': n_orders,
                  'commission_usd': commission_usd,
                  'exposure': float(sum(weights.values()))})
    pos_rows += [(d, sym, w) for sym, w in weights.items()]

  # --- 期末: 未平仓段以最后交易日收盘标记; 最后一日 open -> close 盯市 ---
  last_d = dates[-1]
  for sym in list(open_trades.keys()):
    t = open_trades.pop(sym)
    k = opos.get(sym)
    px = close_np[last_i, k] if k is not None else np.nan
    if px == px and t['entry_px'] > 0:
      trades.append({'symbol': sym, 'entry_exec': t['entry_exec'], 'exit_exec': last_d,
                     'entry_px': t['entry_px'], 'exit_px': float(px),
                     'entry_w': t['entry_w'], 'peak_w': t['peak_w'],
                     'days': len(dates) - 1 - t['entry_i'],
                     'ret': float(px) / t['entry_px'] - 1.0,
                     'exit_reason': 'open', 'completed': False})
  if weights and daily:
    final = 0.0
    for sym, w in weights.items():
      k = opos.get(sym)
      if k is None:
        continue
      cv, ov = close_np[last_i, k], open_np[last_i, k]
      if cv == cv and ov == ov and ov > 0:
        final += w * (cv / ov - 1.0)
    if final:
      daily[-1]['equity'] *= (1.0 + final)

  daily_df = pd.DataFrame(daily).set_index('date')
  base = pd.DataFrame([{'date': dates[0], 'equity': 1.0, 'gross_ret': 0.0, 'cost_drag': 0.0,
                        'turnover': 0.0, 'n_pos': 0, 'n_orders': 0, 'commission_usd': 0.0,
                        'exposure': 0.0}]).set_index('date')
  daily_df = pd.concat([base, daily_df])
  pos_df = pd.DataFrame(pos_rows, columns=['date', 'symbol', 'weight'])
  trades_df = pd.DataFrame(trades)
  ann_turnover = turnover_sum / max(n_periods, 1) * ANNUAL
  return {'daily': daily_df, 'pos': pos_df, 'trades': trades_df,
          'equity': daily_df['equity'], 'ann_turnover': ann_turnover, 'params': p,
          'n_bad_px': n_bad_px}


# ============================ 4. 绩效指标 ============================

def perf_stats(equity: pd.Series, ann_turnover: float = np.nan,
               trades_df: pd.DataFrame = None) -> dict:
  """
  净值 -> 绩效指标.

  total_ret / cagr / vol / sharpe / max_dd / calmar / ann_turnover,
  再由成交流水派生 n_trades / win_rate / avg_ret / avg_days(只统计已平仓段).

  :param equity: 净值曲线(首行为 1.0 初始行)
  :param ann_turnover: 年化单边换手
  :param trades_df: run_engine 的成交流水(None = 不算交易统计)
  :returns: 指标 dict
  """
  if len(equity) < 2:
    return {k: np.nan for k in STAT_KEYS}
  ret = equity.pct_change().dropna()
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
  total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
  cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 else -1.0
  vol = float(ret.std() * np.sqrt(ANNUAL)) if len(ret) > 1 else np.nan
  sharpe = (float(ret.mean() / ret.std() * np.sqrt(ANNUAL))
            if len(ret) > 1 and ret.std() > 0 else np.nan)
  dd = float((equity / equity.cummax() - 1.0).min())
  calmar = cagr / abs(dd) if dd < 0 else np.nan
  out = {'n_days': len(equity), 'total_ret': round(total, 3), 'cagr': round(cagr, 3),
         'vol': round(vol, 3), 'sharpe': round(sharpe, 2), 'max_dd': round(dd, 3),
         'calmar': round(calmar, 2) if pd.notna(calmar) else np.nan,
         'ann_turnover': round(ann_turnover, 1) if pd.notna(ann_turnover) else np.nan,
         'n_trades': 0, 'win_rate': np.nan, 'avg_ret': np.nan, 'avg_days': np.nan}
  if trades_df is not None and len(trades_df) and 'completed' in trades_df.columns:
    done = trades_df[trades_df['completed']]
    if len(done):
      out.update({'n_trades': int(len(done)),
                  'win_rate': round(float((done['ret'] > 0).mean()), 3),
                  'avg_ret': round(float(done['ret'].mean()), 4),
                  'avg_days': round(float(done['days'].mean()), 1)})
  return out


def cost_summary(res: dict) -> dict:
  """
  成本归因摘要(资金量口径): 年化单边换手 / 累计成交笔数 / 累计佣金(USD) / 年化佣金占本金比.

  仅当 EngineParams 设了 capital 时才有佣金项(否则 commission 全为 0, 只报换手).
  用途: 小资金实盘评估时, 一眼看出"固定佣金到底吃掉多少年化".
  """
  daily, p = res['daily'], res['params']
  n_days = max(len(daily) - 1, 1)
  years = n_days / ANNUAL
  cum_comm = float(daily['commission_usd'].sum()) if 'commission_usd' in daily else 0.0
  cum_orders = int(daily['n_orders'].sum()) if 'n_orders' in daily else 0
  out = {'ann_turnover': round(res['ann_turnover'], 1),
         'n_orders_total': cum_orders,
         'commission_usd_total': round(cum_comm, 2)}
  if p.capital and years > 0:
    out['commission_pct_per_year'] = round(cum_comm / p.capital / years * 100, 2)
  return out


def yearly_returns(equity: pd.Series) -> pd.Series:
  """年度收益: 年末净值 / 年初净值 - 1."""
  if len(equity) < 2:
    return pd.Series(dtype=float)
  yr = equity.groupby(equity.index.year)
  return (yr.last() / yr.first() - 1.0).round(3)


def monthly_returns(equity: pd.Series) -> pd.Series:
  """月度收益: 月末净值 / 月初净值 - 1(索引 'YYYY-MM')."""
  if len(equity) < 2:
    return pd.Series(dtype=float)
  g = equity.groupby([equity.index.year, equity.index.month])
  out = (g.last() / g.first() - 1.0).round(4)
  out.index = [f'{y}-{m:02d}' for y, m in out.index]
  return out


# ============================ 5. 对照与基准 ============================

def shuffle_composite(comp: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
  """随机对照: 整张截面按交易日整体置换, 破坏"分数-时间"对齐(但保留截面结构)."""
  rng = np.random.default_rng(seed)
  perm = rng.permutation(len(comp))
  return comp.iloc[perm].set_axis(comp.index)


def buyhold(open_w: pd.DataFrame, close_w: pd.DataFrame) -> dict:
  """
  全池等权买入持有(基准): 用**同一个引擎**核算, 保证与策略净值可比.

  做法: 常数分数(全 0.5) + top_k = 全池 + exit_rank = 全池+1(永不退出) +
       sizing=equal + 单票上限 1.0 + 零成本 + band=1.0(永不重平衡).
  """
  comp = pd.DataFrame(0.5, index=open_w.index, columns=open_w.columns)
  n = len(open_w.columns)
  p = EngineParams(top_k=n, exit_rank=n + 1, sizing='equal', max_exposure=1.0,
                   per_symbol_cap=1.0, cost_bps=0.0, band=1.0)
  return run_engine(open_w, close_w, comp, p)


def folded_perf(open_w: pd.DataFrame, close_w: pd.DataFrame, comp: pd.DataFrame,
                p: 'EngineParams', folds: list, keys=('sharpe', 'total_ret')) -> pd.DataFrame:
  """
  逐子窗绩效: 对每个 (start, end) 子窗**独立**跑一遍引擎, 汇总指定指标.

  用途: 把"一段 IS"拆成若干连续子段分别评估 —— 单段偶然很强的方案, 在多段上均值
  会被拉低. 这是 make_fold_objective 的底座; 也可单独用于事后观察方案的分段稳定性.

  :param open_w / close_w / comp: **全样本**宽表(函数内部按子窗截取)
  :param p: EngineParams
  :param folds: split_folds 产出的 [(start, end), ...]
  :param keys: 需要汇总的 perf_stats 字段
  :returns: DataFrame[fold, start, end, n_days, *keys](子窗过短则指标为 NaN)
  """
  rows = []
  for i, (a, b) in enumerate(folds, start=1):
    ow, cw = _win(open_w, a, b), _win(close_w, a, b)
    cp = _win(comp, a, b)
    row = {'fold': i, 'start': a, 'end': b, 'n_days': int(len(ow))}
    if len(ow) < 2 or cp.dropna(how='all').empty:            # 子窗不可回测
      row.update({k: np.nan for k in keys})
    else:
      res = run_engine(ow, cw, cp, p)
      st = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      row.update({k: st.get(k, np.nan) for k in keys})
    rows.append(row)
  return pd.DataFrame(rows)


def make_fold_objective(open_w: pd.DataFrame, close_w: pd.DataFrame,
                        p: 'EngineParams', folds: list, metric: str = 'sharpe',
                        agg: str = 'mean'):
  """
  构造"折外目标函数" objective(weights, comp) -> float.

  语义: 把 comp 在各子窗分别回测, 取 metric 的聚合值(默认均值)作为接受判据.
  全部子窗都无效/NaN 时返回 -inf, 保证不会被误选. 与 combine.greedy_weight_search /
  refine_weights 组合, 即可把权重搜索的目标从"IS RankIC"换成"折外更稳".

  注意: 只有"训练子窗"才应作为 folds 传进来 —— 若塞入 OOS 段即构成前视.

  :param open_w / close_w: 全样本宽表(函数内部按子窗截取)
  :param p: EngineParams
  :param folds: 训练期子窗列表
  :param metric: 取 perf_stats 的哪个字段(如 'sharpe' / 'total_ret' / 'calmar')
  :param agg: 'mean'(默认) 或 'min' —— min 更保守(要求每折都不差)
  :returns: objective(weights, comp) -> float(越大越好; 无有效折返回 -inf)
  """
  def objective(weights, comp):
    fp = folded_perf(open_w, close_w, comp, p, folds, keys=(metric,))
    v = fp[metric].dropna()
    if v.empty:
      return -np.inf
    return float(v.mean() if agg == 'mean' else v.min())
  return objective


def cached_fold_objective(open_w: pd.DataFrame, close_w: pd.DataFrame,
                          p: 'EngineParams', folds: list, metric: str = 'sharpe',
                          agg: str = 'mean'):
  """
  make_fold_objective 的**带记忆**版本: 同一组权重(四舍五入到 6 位)只算一次折外目标.

  权重搜索结果会被"搜一遍"再"报一遍"(最终方案复评 / 导出), 逐折回测很贵, 故按权重记忆.
  记忆键只用 weights —— 组合 comp 由 (sigs, weights) 确定, 同一脚本内 sigs/min_cs 恒定,
  故 weights 相同即 comp 相同, 不会串味.

  :returns: objective(weights, comp) -> float(语义同 make_fold_objective)
  """
  base = make_fold_objective(open_w, close_w, p, folds, metric=metric, agg=agg)
  memo: dict = {}

  def objective(weights, comp):
    key = tuple(sorted((n, round(float(w), 6)) for n, w in (weights or {}).items()))
    if key not in memo:
      memo[key] = base(weights, comp)
    return memo[key]
  return objective


def verify_backtest(ds: Dataset, method: str = prep.DEFAULT_PREP, h: int = 1,
                    top_k: int = cfg.TOP_K, min_cs: int = cfg.MIN_CS,
                    min_factors: int = cfg.MIN_FACTORS, seed: int = 42) -> tuple:
  """
  引擎标定: 用"已知答案"的输入检验引擎有没有按预期工作.

  三档 sanity(同一窗口, 默认参数):
    oracle  = 把**未来收益**当分数喂进去 -> 应大幅盈利(挑的就是事后涨最多的);
    shuffle = 把 oracle 分数按交易日整体打乱 -> 应≈随机, 远逊于 oracle;
    buyhold = 全池等权持有 -> 市场基准.
  若 oracle 不赚 / shuffle 也大赚, 说明引擎或口径有 bug.

  前视防线(判 pass/fail, 两条不变量):
    1. 定权窗口不变量: 把 IS 窗外的 fwd 填成有限垃圾值(投毒)后, 用与 honest 完全
       相同的表达式重新定权, 权重应逐值不变 —— 若变了, 说明定权链路读到了 IS 窗
       外的标签(真前视), 必须排查. 这与 verify_causal 的截断一致性同一思想:
       "不该起作用的数据"改了, 输出就不能变;
    2. 对照生效: 全样本定权 != IS 定权 —— 若两份权重相同, 说明前视对照被接错
       线(退化成同一份权重), 对照失去意义.
  honest(IS 定权) vs lookahead(全样本定权) 的 OOS 段净值对照保留为**描述性**
  "前视代价"参考, 不再作方向性判据. 为什么: ic_weights 拟合的是 RankIC, 而引擎
  净值是非线性 top-k 选股 + 换手成本的函数, "偷看"只在 IC 口径占优, 不保证净值
  更高(实测 59/70 因子集均出现 honest > lookahead, 连"直接在 OOS 段拟合"的真
  作弊上界也输给 honest), 因此 honest <= lookahead 不是该构造的恒等式, 拿它当
  判据只会随机误报. 真正的泄露通道(定权数据窗)由上面两条不变量锁死.

  :returns: (cases_df, leak_df, wchk_df)
  """
  open_w, close_w = ds.wide('open'), ds.wide('close')
  fwd = ev.forward_return(ds, h)
  start, end = ev.window_bounds('full', pool=ds.pool)
  ow, cw, fw = _win(open_w, start, end), _win(close_w, start, end), _win(fwd, start, end)
  p = EngineParams(top_k=top_k)

  cases = [('oracle(=fwd)', run_engine(ow, cw, fw, p)),
           ('shuffle(oracle)', run_engine(ow, cw, shuffle_composite(fw, seed), p)),
           ('buyhold', buyhold(ow, cw))]
  rows = []
  for label, res in cases:
    s = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    rows.append({'case': label, 'n_fac': len(ow.columns),
                 **{k: s.get(k) for k in STAT_KEYS}})
  cases_df = pd.DataFrame(rows)

  sigs = cmb.build_signals(ds, method=method, min_cs=min_cs)
  is_s, is_e = ev.window_bounds('is', pool=ds.pool)
  oos_s, oos_e = ev.window_bounds('oos', pool=ds.pool)
  w_honest = cmb.ic_weights(sigs, _win(fwd, is_s, is_e), min_cs=min_cs)   # 只用 IS
  w_leak = cmb.ic_weights(sigs, fwd, min_cs=min_cs)                      # 偷看全样本

  # ---- 前视防线: 两条不变量 ----
  # 投毒: IS 窗外的 fwd 换成有限垃圾值(翻转*10). 用与 honest 相同的表达式重定权,
  # 若定权链路真的只读 [is_s, is_e], 垃圾不可见, 权重应与 w_honest 逐值相同.
  poison = fwd.copy()
  outside = (poison.index < is_s) | (poison.index > is_e)
  poison.loc[outside, :] = (-10.0 * fwd.loc[outside].fillna(1.0)).to_numpy()
  w_is_only = cmb.ic_weights(sigs, _win(poison, is_s, is_e), min_cs=min_cs)
  d_iso = max((abs(w_honest.get(k, 0.0) - w_is_only.get(k, 0.0))
               for k in set(w_honest) | set(w_is_only)), default=0.0)
  d_leak = max((abs(w_honest.get(k, 0.0) - w_leak.get(k, 0.0))
                for k in set(w_honest) | set(w_leak)), default=0.0)
  wchk_df = pd.DataFrame([
      {'check': '定权窗口不变量(IS 窗外投毒, honest 权重应逐值不变)',
       'detail': f'max|dw|={d_iso:.2e}', 'pass': bool(d_iso <= 1e-12)},
      {'check': '前视对照生效(全样本定权应不同于 IS 定权)',
       'detail': f'max|dw|={d_leak:.4f}', 'pass': bool(d_leak > 1e-12)},
  ])

  rows = []
  for label, w in (('honest(IS定权)', w_honest), ('lookahead(全样本定权)', w_leak)):
    comp = cmb.combine_signals(sigs, w, min_cs=min_cs, min_factors=min_factors)
    res = run_engine(_win(open_w, oos_s, oos_e), _win(close_w, oos_s, oos_e),
                     _win(comp, oos_s, oos_e), p)
    s = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    rows.append({'case': label, 'n_fac': len(w), **{k: s.get(k) for k in STAT_KEYS}})
  leak_df = pd.DataFrame(rows)
  return cases_df, leak_df, wchk_df


# ============================ 6. CLI ============================

def main():
  ap = argparse.ArgumentParser(description='第5步 回测层: 组合分数 -> 净值 + 绩效')
  ap.add_argument('--pool', default=cfg.DEFAULT_POOL)
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  ap.add_argument('--method', default=prep.DEFAULT_PREP, choices=prep.PREP_METHODS)
  ap.add_argument('--h', type=int, default=1, help='前向收益持有期(仅影响标签评估)')
  ap.add_argument('--top-k', type=int, default=cfg.TOP_K, help='入场 rank 门槛(持仓数上限)')
  ap.add_argument('--exit-rank', type=int, default=cfg.EXIT_RANK, help='退出 rank 门槛')
  ap.add_argument('--sizing', default='equal', choices=SIZINGS)
  ap.add_argument('--max-exposure', type=float, default=1.0)
  ap.add_argument('--per-symbol-cap', type=float, default=0.30)
  ap.add_argument('--strength-threshold', type=float, default=None,
                  help='路线A: 截面 z 入场门槛 θ(仅 sizing=strength 时生效; 省略=关闭)')
  ap.add_argument('--cost-bps', type=float, default=cfg.COST_BPS, help='单边成本(基点)')
  ap.add_argument('--band', type=float, default=0.05, help='重平衡带宽')
  ap.add_argument('--capital', type=float, default=None,
                  help='账户本金(USD); 设值后叠加**按笔固定佣金**(小资金实盘可行性评估用)')
  ap.add_argument('--commission-per-order', type=float, default=0.0,
                  help='每笔固定佣金(USD, 含券商最低收费/平台费); 需配合 --capital 生效')
  ap.add_argument('--min-cs', type=int, default=cfg.MIN_CS)
  ap.add_argument('--min-factors', type=int, default=cfg.MIN_FACTORS,
                  help='每格最少有效因子数(缺失感知归一化下, 太少会让单因子被顶到满量程产生伪信号)')
  ap.add_argument('--group', default=None, help='只用某类因子')
  ap.add_argument('--scheme', default='all', choices=('all',) + SCHEMES)
  ap.add_argument('--thresh', type=float, default=cmb.DEDUPE_RHO)
  ap.add_argument('--max-n', type=int, default=8)
  ap.add_argument('--min-gain', type=float, default=cmb.MIN_GAIN)
  ap.add_argument('--min-ic', type=float, default=0.0)
  ap.add_argument('--check', action='store_true', help='只跑引擎标定自检')
  args = ap.parse_args()

  log = cfg.get_logger('backtest')
  ds = load_pool(args.pool, args.interval)
  log.info(ds.summary())

  if args.check:
    cases_df, leak_df, wchk_df = verify_backtest(ds, args.method, args.h,
                                                 args.top_k, args.min_cs,
                                                 args.min_factors)
    print('\n[引擎标定] 同窗口, 默认参数(oracle 应大赚, shuffle 应≈随机):')
    print(cases_df.to_string(index=False))
    print('\n[前视防线] 定权窗口不变量(两项均应 pass):')
    print(wchk_df.to_string(index=False))
    print('\n[前视代价] OOS 段净值对照(描述性参考, 不作判据):')
    print(leak_df.to_string(index=False))
    return

  open_w, close_w = ds.wide('open'), ds.wide('close')
  fwd = ev.forward_return(ds, args.h)
  is_s, is_e = ev.window_bounds('is', pool=ds.pool)
  oos_s, oos_e = ev.window_bounds('oos', pool=ds.pool)
  fwd_is = _win(fwd, is_s, is_e)

  sigs = cmb.build_signals(ds, method=args.method, names=fct.list_factors(args.group),
                           min_cs=args.min_cs)

  # --- 定权: 全部只用 IS 期(第4步红线) ---
  sc = cmb.rank_ic_scores(sigs, fwd_is, min_cs=args.min_cs)
  kept, dropped = cmb.dedupe(sigs, sc, thresh=args.thresh, min_cs=args.min_cs)
  log.info(f'入池 {len(sigs)} -> 去重后 {len(kept)}; 剔除 {[d["factor"] for d in dropped]}')

  weights = {}
  if args.scheme in ('all', 'equal'):
    weights['equal'] = cmb.equal_weights(kept)
  if args.scheme in ('all', 'ic'):
    weights['ic'] = cmb.ic_weights(sigs, fwd_is, names=kept, min_cs=args.min_cs,
                                   min_ic=args.min_ic)
  if args.scheme in ('all', 'greedy'):
    gnames, gain_df = cmb.greedy_forward(sigs, fwd_is, names=kept, max_n=args.max_n,
                                        min_gain=args.min_gain, min_cs=args.min_cs,
                                        min_factors=args.min_factors)
    if gnames:
      weights['greedy'] = cmb.equal_weights(gnames)
      if len(gain_df):
        log.info('贪心路径:\n' + gain_df.to_string(index=False))
    else:
      log.warning('贪心未选出任何因子, 跳过 greedy 方案')

  p = EngineParams(args.top_k, args.exit_rank, args.sizing, args.max_exposure,
                   args.per_symbol_cap, args.cost_bps, args.band,
                   args.strength_threshold, args.capital, args.commission_per_order)
  log.info('引擎: ' + p.brief())

  best = max(sc, key=lambda n: abs(sc[n]) if pd.notna(sc[n]) else -1.0)
  combos = {k: cmb.combine_signals(sigs, w, min_cs=args.min_cs,
                                   min_factors=args.min_factors) for k, w in weights.items()}
  combos[f'single:{best}'] = cmb.combine_signals(sigs, {best: 1.0}, min_cs=args.min_cs,
                                                 min_factors=args.min_factors)
  if 'equal' in weights:
    combos['shuffle(equal)'] = shuffle_composite(combos['equal'], seed=42)

  windows = (('IS', is_s, is_e), ('OOS', oos_s, oos_e))
  rows = []
  for kind, comp in combos.items():
    if kind in weights:
      nf = len(weights[kind])
    elif kind.startswith('shuffle'):
      nf = len(kept)
    else:
      nf = 1
    for win, ws, we in windows:
      res = run_engine(_win(open_w, ws, we), _win(close_w, ws, we), _win(comp, ws, we), p)
      s = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      rows.append({'kind': kind, 'window': win, 'n_fac': nf,
                   **{k: s.get(k) for k in STAT_KEYS}})
  for win, ws, we in windows:
    res = buyhold(_win(open_w, ws, we), _win(close_w, ws, we))
    s = perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    rows.append({'kind': 'buyhold', 'window': win, 'n_fac': len(open_w.columns),
                 **{k: s.get(k) for k in STAT_KEYS}})
  table = pd.DataFrame(rows)
  print('\n[回测汇总] 权重全部只在 IS 期估计; IS/OOS 各窗独立推进引擎')
  print(table.to_string(index=False))

  main_kind = 'equal' if 'equal' in weights else next(iter(weights))
  trade_start = is_s                                  # 该池 IS 起点 = 全交易窗起点(窗口真源)
  res_full = run_engine(_win(open_w, trade_start, None),
                        _win(close_w, trade_start, None),
                        _win(combos[main_kind], trade_start, None), p)
  print(f'\n[主方案 {main_kind} 全交易窗({trade_start} 起)分年收益]')
  print(yearly_returns(res_full['equity']).to_string())
  print('\n[成本归因]', cost_summary(res_full))


if __name__ == '__main__':
  main()

# -*- coding: utf-8 -*-
"""
factor.combine — 组合层(第4步)
================================
职责: 把"一堆各自有效、但都很弱的因子"合成一个更强的信号.

--- 为什么需要组合 ---
  第3步的结论: 单因子 RankIC 大多落在 0.01~0.04. 0.04 在日频已算"有信息",
  但单看某一天的截面, 噪声远大于信号 —— 只押注单因子, 净值会被噪声主导.
  把 N 个"部分独立"的弱信号平均, 信噪比大致按 sqrt(N) 提升, 组合的 ICIR 往往
  明显高于任一单因子. 这就是组合层的全部动机.

--- "把因子加起来"远不是终点: 冗余 ---
  同族因子高度相关(mom_12_1 / mom_120 / slope_20 常 > 0.8), 等权相加等于把同一个
  信号重复押三遍: 既不增加信息, 又把组合的暴露集中到该信号上.
  所以顺序必须是: 先诊断相关 -> 再去重 -> 最后才谈加权.

--- 三层结构 ---
  1. 相关性诊断  factor_corr   -- 谁和谁在说同一件事
  2. 冗余去重    dedupe        -- 贪心: 按 |RankIC| 从强到弱纳入; 与已选 |rho| >= 阈值 的丢弃
  3. 加权方案:
       equal  : 去重后等权平均, 最稳的基线(不估计任何参数);
       dir    : 先按先验 direction 翻到"分高=看多"再等权 —— 修 equal 对 direction=-1 的
                因子(波动类)反向押注的缺陷;
       ic     : 权重 = 各因子 RankIC(带符号)归一化, 让数据决定"谁更准";
       inv_vol: 权重 ∝ 1/波动率 —— 用各因子"日频多空收益流"的波动定权, 波动大的少给;
       rp     : 风险平价(ERC) —— 让各因子的"风险贡献"相等, 进一步压制高波动因子;
       greedy : 每步挑"加入后组合 RankIC 提升最大"的因子, 提升 < min_gain 即停.

  ic / inv_vol / rp 的符号都取自 IS 实测 RankIC 符号(不迷信先验 direction).

  另有**通用离散权重搜索** greedy_weight_search / refine_weights 与**连续坐标下降**
  coord_descent_weights: 与 greedy 同骨架, 但**目标函数由调用方注入** —— 可以是 IS
  RankIC, 也可以是回测/折外(多段)指标. 这样就把"权重搜索骨架"与"接受判据"解耦:
  想换成"折外更稳才收"只需换 objective.

--- 前视红线(本层最容易踩的坑) ---
  权重是"参数", 参数只能用样本内(IS)数据估计. 若拿全样本(含 OOS)的 RankIC 定权,
  等于让权重提前知道了"OOS 期谁表现好" —— 回测出来的 OOS 表现是虚假的.
  本层所有定权函数都要求调用方显式传入**已截窗口的 fwd**(见 main 传入 fwd_is),
  verify_combine 的 leak 对照会把"全样本定权"的虚高实测出来.

  这与第3步"标签必须用未来"是同一问题的两面:
    第3步管"评估要在时间上诚实"; 本层管"选参也要在时间上诚实".

自检:
  cd ~/git && python -m quant.factor.combine --pool etf_3x
  cd ~/git && python -m quant.factor.combine --pool etf_3x --check
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import evaluate as ev
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import Dataset, load_pool

DEDUPE_RHO = 0.7                                # 冗余阈值: |rho| >= 该值视为"说同一件事"
MIN_GAIN = 0.001                                # 贪心增量下限: 低于此值视为噪声, 停止加入
IC_SHRINK = 0.25                                # ic 权重向"方向对齐等权"收缩的比例(0=纯 IC, 1=纯等权)
FOLD_OBJ_FOLDS = 6                              # 折外目标默认折数: 6 折等分均值比 4 折在缺口/全窗上同时更优
COMBO_SCHEMES = ('equal', 'dir', 'ic', 'inv_vol', 'rp', 'greedy')
WEIGHT_GRID = (0.5, 1.0)                        # 离散权重档位(贪心权重搜索用); signed=True 时自动补负档


# ============================ 小工具 ============================

def _rank_ic(sig: pd.DataFrame, other: pd.DataFrame, min_cs: int) -> float:
  """
  两张宽表的平均截面秩相关(按交易日取均值); 无有效交易日返回 NaN.

  :param sig: 因子宽表
  :param other: 另一张宽表(因子或前向收益)
  :param min_cs: 每日最少有效标的数
  :returns: 时间均值
  """
  ic = ev.ic_series(sig, other, 'spearman', min_cs)
  return float(ic.mean()) if len(ic) else np.nan


def _win(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
  """按交易日截取宽表的行(只选时间窗, 不改内容)."""
  idx = df.index
  if start is not None:
    idx = idx[idx >= pd.Timestamp(start)]
  if end is not None:
    idx = idx[idx <= pd.Timestamp(end)]
  return df.loc[idx]


def rank_ic_scores(sigs: dict, fwd: pd.DataFrame, names: list = None,
                   min_cs: int = cfg.MIN_CS) -> dict:
  """
  各因子相对前向收益的 RankIC 均值 -> {name: rank_ic}.

  这是去重的排序依据, 也是 IC 加权的原料. **调用方必须传入已截窗口的 fwd**:
  定权用 IS 期, 验证用 OOS 期, 不能混.

  :param sigs: {因子名: 预处理后宽表}
  :param fwd: (已截窗口的)前向收益宽表
  :param names: 参与计算的因子(None = 全部)
  :param min_cs: 每日最少有效标的数
  :returns: {因子名: rank_ic}
  """
  names = list(names or sigs.keys())
  return {n: _rank_ic(sigs[n], fwd, min_cs) for n in names}


def build_signals(ds: Dataset, method: str = prep.DEFAULT_PREP, names: list = None,
                  min_cs: int = cfg.MIN_CS) -> dict:
  """
  算并预处理一批因子 -> {因子名: 宽表}(第1步算, 第2步洗).

  :param ds: Dataset
  :param method: 预处理方法(见 prepare.PREP_METHODS)
  :param names: 因子名列表(None = 全部)
  :param min_cs: 每日最少有效标的数
  :returns: {因子名: 预处理后宽表}
  """
  pf = prep._make_prep(method, min_cs)
  return {n: pf(fct.get_factor(n).compute(ds)) for n in (names or fct.list_factors())}


# ============================ 1. 相关性诊断 ============================

def factor_corr(sigs: dict, names: list = None,
                min_cs: int = cfg.MIN_CS) -> pd.DataFrame:
  """
  因子两两之间的"平均截面秩相关"矩阵.

  做法: 对每一对 (a, b) 逐日算截面秩相关(直接复用 evaluate.ic_series, 把 b 当作
        另一个因子), 再对交易日取均值. 用秩相关而不是 Pearson, 与第2步默认的
        rank 预处理一致, 且不受极值影响.

  看点:
    - 对角线恒为 1;
    - 动量家族(mom_12_1 / mom_120 / slope_20)通常 > 0.8 -> 高度冗余, 只需留一个;
    - 反转(rev_5 / rev_10)与动量常为负相关 -> 信息互补.

  :param sigs: {因子名: 预处理后宽表}
  :param names: 参与计算的因子(None = 全部)
  :param min_cs: 每日最少有效标的数
  :returns: DataFrame(对称矩阵), index/columns = 因子名
  """
  names = list(names or sigs.keys())
  if not names:
    return pd.DataFrame()
  m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      r = _rank_ic(sigs[a], sigs[b], min_cs)
      m.loc[a, b] = m.loc[b, a] = round(r, 3) if pd.notna(r) else np.nan
  return m


# ============================ 2. 冗余去重 ============================

def dedupe(sigs: dict, scores: dict, thresh: float = DEDUPE_RHO,
           min_cs: int = cfg.MIN_CS, corr: pd.DataFrame = None) -> tuple:
  """
  冗余去重(贪心): 按 |score| 从强到弱依次纳入, 与已纳入因子 |rho| >= thresh 的丢弃.

  为什么取绝对值: 强负相关的两个因子同时等权入池会内部对冲(一个正一个负),
  组合信号相互抵消, 等于白算. 所以"冗余"看的是信息重复度 |rho|, 不是符号.

  :param sigs: {因子名: 预处理后宽表}
  :param scores: {因子名: 定序分数}, 通常是 IS 期 rank_ic_scores 的输出
  :param thresh: 冗余阈值
  :param min_cs: 每日最少有效标的数
  :param corr: 现成的相关矩阵(None 则内部计算)
  :returns: (kept 因子名列表, dropped 明细列表[{factor, rank_ic, conflict_with, rho}])
  """
  names = [n for n in scores if n in sigs]
  if not names:
    return [], []
  order = sorted(names,
                 key=lambda n: abs(scores[n]) if pd.notna(scores[n]) else -1.0,
                 reverse=True)                      # 强因子先入选
  if corr is None:
    corr = factor_corr(sigs, order, min_cs)
  kept, dropped = [], []
  for n in order:
    hit = None
    for k in kept:
      rho = corr.loc[k, n] if (k in corr.index and n in corr.columns) else np.nan
      if pd.notna(rho) and abs(rho) >= thresh:
        hit = (k, float(rho))
        break
    if hit:
      dropped.append({'factor': n,
                      'rank_ic': round(float(scores[n]), 4) if pd.notna(scores[n]) else np.nan,
                      'conflict_with': hit[0],
                      'rho': round(hit[1], 3)})
    else:
      kept.append(n)
  return kept, dropped


# ============================ 3. 加权方案 ============================

def combine_signals(sigs: dict, weights: dict, min_cs: int = cfg.MIN_CS,
                    require_all: bool = False) -> pd.DataFrame:
  """
  加权合成 -> 组合信号宽表.

  缺失感知: 分子 = sum(w_i * sig_i)(逐格跳过 NaN), 分母 = sum(|w_i|) 只累加该格
  **有效**因子的权重. 两者相除得到"可用因子的加权平均", 这样不同标的的有效因子
  数不同也不会造成尺度漂移; 权重带符号, 分母取 |w| 以保证结果仍在 [-1, 1].

  require_all=True 时改为"所有因子都有效才给值"(更严格, 但覆盖率会降).

  :param sigs: {因子名: 预处理后宽表}
  :param weights: {因子名: 权重}; 权重为 0 或不在 sigs 里的因子被跳过
  :param min_cs: 每日最少有效标的数(截面太薄的行整行作废)
  :param require_all: True = 要求所有入池因子在该格都有效
  :returns: 组合信号宽表(已 guard 薄截面)
  """
  names = [n for n, w in weights.items() if n in sigs and float(w) != 0.0]
  if not names:
    raise ValueError('权重为空(或全为 0)且无可用因子, 无法组合')
  base = sigs[names[0]]
  num = pd.DataFrame(0.0, index=base.index, columns=base.columns)
  den = pd.DataFrame(0.0, index=base.index, columns=base.columns)
  for n in names:
    s = sigs[n].reindex(index=base.index, columns=base.columns)   # 对齐到同一网格
    w = float(weights[n])
    num = num + s.fillna(0.0) * w
    den = den + s.notna().astype(float) * abs(w)
  combo = num.div(den.where(den > 0))                            # den=0 -> NaN
  if require_all:
    total = sum(abs(float(weights[n])) for n in names)
    combo = combo.where(den >= total - 1e-9)
  return prep.guard_rows(combo, min_cs)


def equal_weights(names: list) -> dict:
  """等权: 每个因子 1/N. 不估计任何参数, 是最稳的基线."""
  names = list(names)
  return {n: round(1.0 / len(names), 6) for n in names} if names else {}


def dir_equal_weights(names: list) -> dict:
  """
  方向对齐等权: 每个因子先按**先验 direction** 翻到"分数越高越看多", 再等权 1/N.

  与 equal 的唯一区别是符号: equal 直接用原始朝向信号, 对 direction=-1 的因子
  (波动类 vol/max_ret/kurt/bbw)等于反向押注 —— 组合里系统性地买高波动.
  本方案把方向对齐后再等权, 权重 = direction / N, 故 sum|w| = 1.
  用先验 direction 而非 IS 实测符号, 是为了不动用样本内信息、避免过拟合 IS.

  :param names: 因子名列表
  :returns: {因子名: 权重}(±1/N)
  """
  names = list(names)
  if not names:
    return {}
  return {n: round(float(fct.get_factor(n).direction) / len(names), 6) for n in names}


def ic_weights(sigs: dict, fwd: pd.DataFrame, names: list = None,
               min_cs: int = cfg.MIN_CS, min_ic: float = 0.0,
               shrink: float = IC_SHRINK, soft_thresh: float = 0.0,
               target: dict = None) -> dict:
  """
  IC 加权 + 收缩: 权重 ∝ 各因子 RankIC(带符号), 再向"方向对齐等权"收缩.

  **为什么要收缩**(直接针对 IS/OOS 一致性): |RankIC| 是"单期 IS 估计", 噪声很大 ——
  某期偶然很强的因子会吃掉大半权重, 到 OOS 往往失效. 收缩的做法是"半信半疑":

      w = (1 - shrink) * w_ic  +  shrink * w_dir_equal

  shrink=0 退化为纯 IC 加权(旧行为); shrink=1 退化为 dir 等权. 默认 0.5, 即"只在
  一半程度上相信 IS 实测的符号与幅度, 另一半退回先验等权" —— 单看 IS 会略弱, 但
  跨期更稳. 收缩目标取 dir 等权(而非 equal)是因为它已把先验 direction 对齐, 不会把
  IC 正确翻转的因子又拉回错误方向.

  带符号: 实测 RankIC 为负的因子自动被反向使用 —— 不迷信第1步的先验 direction
  (先验只是提示, 实测才是依据). 但注意: 收缩会部分抵消这个翻转(如果实测符号与先验
  相反), 这正是收缩的本意 —— 不把一次测量的翻转当成铁证.

  软阈值(soft_thresh): 先做 |RankIC| - soft_thresh 再保号(不足则置 0). 与硬门槛
  min_ic 的区别是"连续收缩"而非"一刀切", 对强度落在噪声带的因子更平滑. 默认 0(不生效).

  自定义收缩目标(target): 默认向"方向对齐等权"收缩; 若传入 target(={因子名: 权重},
  如 inv_vol / rp 的稳健权重), 则改为向它收缩 —— 让"半信半疑"退回到一个**风险更均衡**
  的先验, 而非简单等权. target 会被归一化到 sum|w|=1; 未覆盖的因子回退到方向对齐等权.

  :param sigs: {因子名: 预处理后宽表}
  :param fwd: **IS 期**的前向收益(绝不能传全样本)
  :param names: 参与加权的因子(None = 全部)
  :param min_cs: 每日最少有效标的数
  :param min_ic: 硬门槛, |RankIC| 低于此值的因子权重置 0
  :param shrink: 向收缩目标靠拢的比例, 0 = 纯 IC / 1 = 纯目标
  :param soft_thresh: 软阈值(作用在 |RankIC| 上), 0 = 不生效
  :param target: 自定义收缩目标 {因子名: 权重}(None = 方向对齐等权)
  :returns: {因子名: 权重}
  """
  sc = rank_ic_scores(sigs, fwd, names, min_cs)
  vals = {}
  for n, v in sc.items():
    if pd.notna(v):
      a = abs(float(v)) - max(float(soft_thresh), 0.0)      # 软阈值: 连续收缩, 不足则归 0
      vals[n] = 0.0 if a <= 0 else (a if v > 0 else -a)
    else:
      vals[n] = 0.0
  vals = {n: (v if abs(v) >= min_ic else 0.0) for n, v in vals.items()}
  tot = sum(abs(v) for v in vals.values())
  if tot <= 0:
    raise ValueError('所有因子 |RankIC| 都不足门槛, 无法 IC 加权')
  w_ic = {n: v / tot for n, v in vals.items() if v != 0.0}          # sum|w_ic| = 1
  s = float(min(max(shrink, 0.0), 1.0))
  if s > 0 and w_ic:                                                # 向收缩目标靠拢
    w_tgt = _shrink_target(target, list(w_ic))
    w_ic = {n: (1.0 - s) * w_ic[n] + s * w_tgt[n] for n in w_ic}
  return {n: round(v, 6) for n, v in w_ic.items()}


def _shrink_target(target: dict, names: list) -> dict:
  """构造收缩目标: 优先用调用方给的 target(归一化到 sum|w|=1), 未覆盖的因子回退到方向对齐等权."""
  if target:
    tgt = {n: float(target.get(n, 0.0)) for n in names}
    tot = sum(abs(v) for v in tgt.values())
    if tot > 0:
      return {n: v / tot for n, v in tgt.items()}
  return dir_equal_weights(names)


def factor_returns(sigs: dict, fwd: pd.DataFrame, names: list = None,
                   min_cs: int = cfg.MIN_CS) -> pd.DataFrame:
  """
  各因子的"日频多空收益流": 用每个因子自身的截面暴露做多空组合, 逐日算其收益.

  做法(逐因子): 先把当日暴露行内去均值(得到零和多空权重 z), 再以 z 加权次日收益,
  除以 sum|z| 归一 -> r_i(t) = Σ_s z_i(t,s)·fwd(t,s) / Σ_s |z_i(t,s)|.
  只在 sig 与 fwd 同时有效的格子上计算; 有效标的数 < min_cs 的日子置 NaN.

  这是 inv_vol / rp 两种"按风险定权"方案的原料: 波动大的因子收益流, 其日收益序列的
  标准差也大 —— 按 1/σ 定权即可压制它. **fwd 必须是 IS 期**(与定权红线一致).

  :param sigs: {因子名: 预处理后宽表}
  :param fwd: **IS 期**前向收益宽表
  :param names: 参与计算的因子(None = 全部)
  :param min_cs: 每日最少有效标的数
  :returns: DataFrame(index=交易日, columns=因子名), 值为各因子当日多空收益
  """
  names = [n for n in (names or sigs.keys()) if n in sigs]
  out = {}
  for n in names:
    a = sigs[n].reindex(index=fwd.index, columns=fwd.columns)
    valid = a.notna() & fwd.notna()
    z = a.sub(a.where(valid).mean(axis=1), axis=0)                 # 行内去均值 -> 零和多空暴露
    z = z.where(valid)
    num = (z * fwd).sum(axis=1, skipna=True)
    den = z.abs().sum(axis=1, skipna=True)
    r = num.div(den.where(den > 0))
    out[n] = r.where(valid.sum(axis=1) >= min_cs)
  return pd.DataFrame(out)


def _risk_sign(sigs: dict, fwd: pd.DataFrame, names: list, min_cs: int,
               sign_from: str) -> dict:
  """因子的方向符号: sign_from='ic' 用 IS 实测 RankIC 符号; 'dir' 用先验 direction."""
  if sign_from == 'dir':
    return {n: float(fct.get_factor(n).direction) for n in names}
  sc = rank_ic_scores(sigs, fwd, names, min_cs)
  return {n: (1.0 if (pd.notna(v) and v >= 0) else -1.0) for n, v in sc.items()}


def inv_vol_weights(sigs: dict, fwd: pd.DataFrame, names: list = None,
                    min_cs: int = cfg.MIN_CS, sign_from: str = 'ic') -> dict:
  """
  逆波动率加权: 权重 ∝ 1/σ_i, σ_i = 该因子 IS 期多空收益流的标准差(见 factor_returns).

  动机: RankIC 相同但"收益流更稳(波动小)"的因子, 单位风险带来的信息更多. 逆波动率让
  高波动因子少吃权重 —— 这是最朴素也最稳的"按风险定权", 不需估计相关矩阵, 不易过拟合.
  符号取自 IS 实测 RankIC 符号(或先验 direction), 结果归一化到 sum|w| = 1.

  :param sigs: {因子名: 预处理后宽表}
  :param fwd: **IS 期**前向收益
  :param names: 参与加权的因子(None = 全部)
  :param min_cs: 每日最少有效标的数
  :param sign_from: 'ic'(默认) 或 'dir'
  :returns: {因子名: 权重}(sum|w| = 1)
  """
  names = [n for n in (names or sigs.keys()) if n in sigs]
  rets = factor_returns(sigs, fwd, names, min_cs)
  vol = rets.std(ddof=1)
  sign = _risk_sign(sigs, fwd, names, min_cs, sign_from)
  raw = {}
  for n in names:
    sig_n = sign.get(n, 1.0)
    sd = float(vol.get(n, np.nan))
    raw[n] = sig_n / sd if (pd.notna(sd) and sd > 0) else 0.0
  tot = sum(abs(v) for v in raw.values())
  if tot <= 0:
    raise ValueError('所有因子收益流波动都无效, 无法逆波动率加权')
  return {n: round(v / tot, 6) for n, v in raw.items() if v != 0.0}


def risk_parity_weights(sigs: dict, fwd: pd.DataFrame, names: list = None,
                        min_cs: int = cfg.MIN_CS, sign_from: str = 'ic',
                        iters: int = 200, tol: float = 1e-8) -> dict:
  """
  风险平价(ERC, Equal Risk Contribution): 让各因子的"风险贡献"相等.

  与 inv_vol 的区别: inv_vol 只看各自波动(隐含"因子间不相关"); ERC 用因子收益流的
  协方差矩阵 Σ, 解出 w 使每个因子的风险贡献 w_i·(Σw)_i 相等 —— 对相关因子会少给权重,
  从而更彻底地避免"把风险集中到一族因子上". 用标准乘性迭代求解(long-only, 正权重),
  之后再乘 IS 实测 RankIC 符号, 归一化到 sum|w| = 1.

  :param sigs / fwd / names / min_cs / sign_from: 同 inv_vol_weights
  :param iters / tol: 迭代次数上限与收敛阈值
  :returns: {因子名: 权重}(sum|w| = 1)
  """
  names = [n for n in (names or sigs.keys()) if n in sigs]
  rets = factor_returns(sigs, fwd, names, min_cs).dropna(how='all')
  k = len(names)
  if k == 0 or len(rets) < k + 2:
    return inv_vol_weights(sigs, fwd, names, min_cs, sign_from)   # 样本不足以估协方差 -> 退化为逆波动
  cov = rets[names].cov().values
  cov = np.nan_to_num(cov, nan=0.0)
  np.fill_diagonal(cov, np.maximum(np.diag(cov), 1e-12))
  w = np.ones(k) / k
  for _ in range(int(iters)):
    mrc = cov @ w                                                 # 边际风险贡献
    rc = w * mrc                                                  # 风险贡献
    rc = np.maximum(rc, 1e-12)
    w_new = w * (rc.mean() / rc)                                  # 朝"风险贡献相等"缩放
    w_new = np.maximum(w_new, 0.0)
    if w_new.sum() <= 0:
      break
    w_new = w_new / w_new.sum()
    if np.max(np.abs(w_new - w)) < tol:
      w = w_new
      break
    w = w_new
  sign = _risk_sign(sigs, fwd, names, min_cs, sign_from)
  raw = {names[i]: float(sign.get(names[i], 1.0)) * float(w[i]) for i in range(k)}
  tot = sum(abs(v) for v in raw.values())
  if tot <= 0:
    raise ValueError('风险平价解为空, 无法加权')
  return {n: round(v / tot, 6) for n, v in raw.items() if v != 0.0}


def greedy_forward(sigs: dict, fwd: pd.DataFrame, names: list = None,
                   max_n: int = 8, min_gain: float = MIN_GAIN,
                   min_cs: int = cfg.MIN_CS) -> tuple:
  """
  贪心前向选择: 每步在剩余候选里挑"加入后组合 RankIC 提升最大"的一个.

  停止条件: 候选用尽 / 已达 max_n / 本轮最大增量 < min_gain(视为噪声).
  第一个因子必须让组合 RankIC > 0 才收 —— 否则说明这批因子整体无效, 应空手而归.
  入选后按等权合成(选择已经挑了"要谁", 不再额外估计权重, 避免二次过拟合).

  :param sigs: {因子名: 预处理后宽表}
  :param fwd: **IS 期**的前向收益(绝不能传全样本)
  :param names: 候选因子(None = 全部)
  :param max_n: 最多入选个数
  :param min_gain: 增量下限
  :param min_cs: 每日最少有效标的数
  :returns: (入选因子名列表, 逐步明细 DataFrame[step, add, combo_rank_ic, gain])
  """
  pool = [n for n in (names or sigs.keys()) if n in sigs]
  chosen, hist = [], []
  best_ic = 0.0
  while pool and len(chosen) < max_n:
    trial = None
    for n in pool:
      combo = combine_signals(sigs, {k: 1.0 for k in chosen + [n]}, min_cs)
      v = _rank_ic(combo, fwd, min_cs)
      v = v if pd.notna(v) else -np.inf
      if trial is None or v > trial[1]:
        trial = (n, v)
    n, v = trial
    if not chosen and v <= 0:                      # 连一个正 IC 因子都找不到 -> 空手
      break
    gain = v - best_ic
    if chosen and gain < min_gain:
      break
    chosen.append(n)
    pool.remove(n)
    best_ic = v
    hist.append({'step': len(chosen), 'add': n,
                 'combo_rank_ic': round(float(v), 4), 'gain': round(float(gain), 4)})
  return chosen, pd.DataFrame(hist)


def split_folds(index, start: str = None, end: str = None,
                n_folds: int = FOLD_OBJ_FOLDS) -> list:
  """
  把 [start, end) 区间**等分为 n_folds 个连续子窗**(不重叠, 后一段接前一段).

  用途: 作为"折外"评估的切分 —— 权重搜索只用其中若干段定参, 判据取**各段均值**,
  比单看一整段 IS 更抗"某一段偶然很强". 与 _wf_folds 的区别: _wf_folds 是"扩张窗
  训练 + 紧随其后的验证段"(用于走查一致性), 本函数只做最简单的等分(用于目标函数).

  :param index: DatetimeIndex(或可转为日期的序列)
  :param start / end: 窗口边界(字符串日期, None = 不限)
  :param n_folds: 折数(默认 FOLD_OBJ_FOLDS=6; <=1 或区间不足时返回单折 = 整个区间)
  :returns: [(seg_start, seg_end), ...], seg_end 为各段最后一个交易日(Timestamp)
  """
  idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
  sub = _win(pd.DataFrame(index=idx), start, end).index
  if n_folds <= 1 or len(sub) < n_folds:
    return [(sub[0], sub[-1])] if len(sub) else []
  edges = np.linspace(0, len(sub), n_folds + 1).astype(int)
  return [(sub[edges[i]], sub[edges[i + 1] - 1]) for i in range(n_folds)]


def _search_grid(weight_grid, signed: bool) -> list:
  """把正档位展开为搜索用档位列表; signed=True 时补上对应的负档(去重且保序)."""
  vals = []
  for w in weight_grid:
    vals.append(float(w))
    if signed:
      vals.append(-float(w))
  return list(dict.fromkeys(vals))


def greedy_weight_search(sigs: dict, candidates: list, objective,
                         min_cs: int = cfg.MIN_CS, weight_grid=WEIGHT_GRID,
                         max_n: int = 4, min_gain: float = 0.05,
                         signed: bool = False, first_min: float = 0.0) -> tuple:
  """
  贪心**离散权重**搜索: 每步在剩余候选里搜"因子 × 权重档位"的最佳加入.

  与 greedy_forward 同骨架(前向、增量阈值停止), 但有两点不同:
    (1) 入选时**同时决定权重**(从 weight_grid 里挑), 而非入选后一律等权;
    (2) 接受判据是**调用方注入的 objective**, 而非固定的 IS RankIC —— 因此可以把
        "折外更稳才收"作为判据(见 backtest.make_fold_objective).

  停止条件: 候选用尽 / 已达 max_n / 首步 objective <= first_min(空手) / 本轮增量 < min_gain.
  首步用 first_min 而非 0 是为了兼容"objective 可能为负"的指标(如折外均值收益).

  :param sigs: {因子名: 预处理后宽表}
  :param candidates: 候选因子名列表
  :param objective: 可调用 objective(weights_dict, comp_df) -> float(**越大越好**);
                    comp 已由本函数用 combine_signals(weights) 算好传入, 避免重复计算
  :param min_cs: 每日最少有效标的数
  :param weight_grid: 正档位元组
  :param max_n: 最多入选个数
  :param min_gain: 增量下限(objective 单位)
  :param signed: True = 允许负权重(补负档); False = 只用正档
  :param first_min: 第一步 objective 必须严格大于它才收(默认 0.0)
  :returns: (weights: {因子名: 权重}, hist: DataFrame[step, add, weight, obj, gain])
  """
  grid = _search_grid(weight_grid, signed)
  pool = [n for n in candidates if n in sigs]
  weights, hist = {}, []
  best_obj = -np.inf
  while pool and len(weights) < max_n:
    trial = None                                             # (name, weight, obj)
    for n in pool:
      for w in grid:
        cand = dict(weights)
        cand[n] = w
        try:
          comp = combine_signals(sigs, cand, min_cs)
        except ValueError:
          continue
        v = objective(cand, comp)
        v = float(v) if (v is not None and pd.notna(v)) else -np.inf
        if trial is None or v > trial[2]:
          trial = (n, w, v)
    if trial is None:                                        # 所有候选都不可用
      break
    n, w, v = trial
    if not weights and v <= first_min:                       # 首步不达标 -> 空手
      break
    gain = v - best_obj
    if weights and gain < min_gain:
      break
    weights[n] = w
    pool.remove(n)
    best_obj = v
    hist.append({'step': len(weights), 'add': n, 'weight': w,
                 'obj': round(best_obj, 4), 'gain': round(gain, 4)})
  return weights, pd.DataFrame(hist)


def refine_weights(sigs: dict, weights: dict, objective,
                   min_cs: int = cfg.MIN_CS, scales=(0.5, 2.0)) -> tuple:
  """
  权重精修一遍: 对每个成分依次试 "×scale / 剔除", 只要 objective 变好就采纳.

  scale<1 收缩(削弱该成分), scale>1 放大, 以及"直接剔除". 单轮扫描(不做迭代到收敛),
  与旧系统一致 —— 目的是微调已由贪心定好的量级, 而非重新搜索组合.

  :param sigs / objective / min_cs: 同 greedy_weight_search
  :param weights: 待精修的权重字典(通常是贪心输出)
  :param scales: 缩放档位元组(不含 1.0 与剔除, 二者由函数内部试)
  :returns: (refined_weights, hist: DataFrame[name, action, weight, obj])
  """
  cur = dict(weights)
  if not cur:
    return cur, pd.DataFrame([])

  def _obj(w):
    try:
      return objective(w, combine_signals(sigs, w, min_cs))
    except ValueError:
      return -np.inf

  best = _obj(cur)
  best = float(best) if pd.notna(best) else -np.inf
  hist = []
  for n in list(cur.keys()):
    base_w = cur[n]
    cands = [('drop', None)]
    for s in scales:
      cands.append((f'x{s:g}', base_w * float(s)))
    for action, new_w in cands:
      trial = dict(cur)
      if action == 'drop':
        trial.pop(n)
      else:
        trial[n] = new_w
      if not trial:                                          # 不许清空
        continue
      v = _obj(trial)
      v = float(v) if pd.notna(v) else -np.inf
      if v > best:
        best = v
        cur = trial
        hist.append({'name': n, 'action': action,
                     'weight': new_w if action != 'drop' else 0.0,
                     'obj': round(best, 4)})
  return cur, pd.DataFrame(hist)


def coord_descent_weights(sigs: dict, init_weights: dict, objective,
                          min_cs: int = cfg.MIN_CS,
                          scales=(0.5, 2.0), allow_sign: bool = True,
                          max_passes: int = 3, min_gain: float = 1e-4) -> tuple:
  """
  连续权重**坐标下降**: 轮流固定其余因子, 只对当前因子的权重试几种改动.

  与 greedy_weight_search / refine_weights 的关系: 三者都是"骨架", 目标函数由调用方
  注入. 区别在搜索方式 —— greedy 是"从空集前向加人", refine 是"对贪心结果扫一遍",
  本函数则是**从给定初值出发、反复轮扫到不再改善**, 可以对**连续权重**做局部精修
  (贪心/精修的档位是离散的). 适合"已有不错的初值(如 ic/inv_vol 权重), 想再优化一点点".

  每个因子的候选改动: {w×scale}(各 scale) ∪ {剔除} ∪ {符号翻转 w→-w}(allow_sign).
  一轮里所有因子各试一次, 采纳所有改善; 若整轮无改善(或改善 < min_gain)则停.
  不允许把所有权重清空.

  :param sigs: {因子名: 预处理后宽表}
  :param init_weights: 初值 {因子名: 权重}(通常来自 ic / inv_vol / rp)
  :param objective: 可调用 objective(weights_dict, comp_df) -> float(**越大越好**)
  :param min_cs: 每日最少有效标的数
  :param scales: 乘性档位(不含 1.0; 剔除与符号翻转由函数内部试)
  :param allow_sign: 是否允许符号翻转
  :param max_passes: 最多轮数
  :param min_gain: 单轮改善下限, 低于此值视为收敛
  :returns: (weights, hist: DataFrame[pass, name, action, weight, obj])
  """
  cur = {n: float(w) for n, w in init_weights.items() if n in sigs and float(w) != 0.0}
  if not cur:
    return {}, pd.DataFrame([])

  def _obj(w):
    try:
      return objective(w, combine_signals(sigs, w, min_cs))
    except ValueError:
      return -np.inf

  best = _obj(cur)
  best = float(best) if pd.notna(best) else -np.inf
  hist = []
  for p in range(1, int(max_passes) + 1):
    improved = False
    for n in list(cur.keys()):
      base_w = cur[n]
      cands = [('drop', None)]
      for s in scales:
        cands.append((f'x{s:g}', base_w * float(s)))
      if allow_sign:
        cands.append(('flip', -base_w))
      for action, new_w in cands:
        trial = dict(cur)
        if action == 'drop':
          trial.pop(n)
        else:
          trial[n] = new_w
        if not trial:                                        # 不许清空
          continue
        v = _obj(trial)
        v = float(v) if pd.notna(v) else -np.inf
        if v > best + float(min_gain):
          best, cur, improved = v, trial, True
          hist.append({'pass': p, 'name': n, 'action': action,
                       'weight': new_w if action != 'drop' else 0.0,
                       'obj': round(best, 4)})
          if action == 'drop':
            break                                            # n 已剔除, 本轮跳过其后续候选
    if not improved:
      break
  return cur, pd.DataFrame(hist)


# ============================ 组合评估 ============================

def evaluate_combo(sigs: dict, weights: dict, fwd: pd.DataFrame,
                   start: str = None, end: str = None, top_k: int = cfg.TOP_K,
                   q: int = 5, min_cs: int = cfg.MIN_CS, h: int = 1,
                   label: str = 'combo', ac_lags: tuple = (1, 5, 20)) -> dict:
  """
  评估一个组合 -> 一行指标(直接复用第3步的 evaluate_one, 口径完全一致).

  注意: fwd 传**全历史**(内部按 start/end 截), 因为标签在窗口边界需要未来价格;
  而权重必须已由 IS 期定好并固定传入 —— 本函数只负责"打分", 不参与定权.

  :param sigs: {因子名: 预处理后宽表}
  :param weights: 组合权重
  :param fwd: 前向收益宽表(全历史)
  :param start/end: 评估窗口
  :param top_k: top-k 持仓数
  :param q: 分层桶数
  :param min_cs: 每日最少有效标的数
  :param h: 前向持有天数(决定 NW 滞后阶)
  :param label: 组合名(仅用于展示)
  :param ac_lags: 自相关滞后阶
  :returns: dict(评估指标)
  """
  combo = combine_signals(sigs, weights, min_cs)
  spec = fct.FactorSpec(name=label, group='组合', formula='加权平均',
                        window=0, direction=1, fn=None)          # fn 不会被调用
  return ev.evaluate_one(spec, _win(combo, start, end), _win(fwd, start, end),
                         top_k, q, min_cs, nw_lag=max(h - 1, 1), ac_lags=ac_lags)


# ============================ IS 折内 walk-forward 一致性 ============================

def _wf_folds(index, start: str = None, end: str = None, n_folds: int = 4,
              min_train_days: int = 120) -> list:
  """
  把 [start, end] 内的交易日切成 n_folds+1 段 -> 折列表(expanding 训练 + 下一段测试).

  第 i 折(1-based): 用第 1..i 段训练, 在第 i+1 段测试. 这样每折训练集都不含测试集,
  且训练集随折扩张, 逼近"逐月上线的真实节奏".

  :param index: 交易日 DatetimeIndex
  :param start/end: 窗口(None = 不截)
  :param n_folds: 折数(= 测试段数 = 总段数 - 1)
  :param min_train_days: 每段最少交易日; 总量不足 (n_folds+1)*min_train_days 则返回 []
  :returns: [{fold, train_start, train_end, test_start, test_end}]
  """
  idx = pd.DatetimeIndex(index).sort_values()
  if start is not None:
    idx = idx[idx >= pd.Timestamp(start)]
  if end is not None:
    idx = idx[idx <= pd.Timestamp(end)]
  n_seg = int(n_folds) + 1
  if n_seg < 2 or len(idx) < n_seg * int(min_train_days):
    return []
  segs = [s for s in np.array_split(np.asarray(idx), n_seg) if len(s)]
  if len(segs) < n_seg:
    return []
  folds = []
  for i in range(1, len(segs)):
    tr = np.concatenate(segs[:i])
    te = segs[i]
    folds.append({'fold': i,
                  'train_start': pd.Timestamp(tr[0]),
                  'train_end': pd.Timestamp(tr[-1]),
                  'test_start': pd.Timestamp(te[0]),
                  'test_end': pd.Timestamp(te[-1])})
  return folds


def _weights_for(scheme: str, sigs: dict, fwd: pd.DataFrame, kept: list,
                 min_cs: int, min_ic: float, shrink: float, soft_thresh: float,
                 max_n: int, min_gain: float) -> dict:
  """按方案名拟合权重(只用传入的 fwd = 该折训练窗); 拟合失败返回 {}."""
  if scheme == 'equal':
    return equal_weights(kept)
  if scheme == 'dir':
    return dir_equal_weights(kept)
  if scheme == 'ic':
    try:
      return ic_weights(sigs, fwd, kept, min_cs, min_ic, shrink, soft_thresh)
    except ValueError:
      return {}
  if scheme == 'inv_vol':
    try:
      return inv_vol_weights(sigs, fwd, kept, min_cs)
    except ValueError:
      return {}
  if scheme == 'rp':
    try:
      return risk_parity_weights(sigs, fwd, kept, min_cs)
    except ValueError:
      return {}
  if scheme == 'greedy':
    chosen, _ = greedy_forward(sigs, fwd, kept, max_n, min_gain, min_cs)
    return equal_weights(chosen)
  raise ValueError(f'未知方案: {scheme} (可选 {COMBO_SCHEMES})')


def _wf_records(sigs: dict, fwd: pd.DataFrame, names: list, folds: list,
                schemes: tuple, thresh: float, min_cs: int, h: int,
                min_ic: float, shrink: float, soft_thresh: float,
                max_n: int, min_gain: float) -> pd.DataFrame:
  """
  逐折拟合 + 折外评估 -> 明细表[fold, scheme, n_fac, train_rank_ic, test_rank_ic, gap].

  每折只算一次 rank_ic_scores + dedupe(与 scheme 无关); 训练窗的 fwd 右端裁掉
  h+1 个交易日, 避免训练标签用到跨越折界的未来开盘价(否则 train_rank_ic 会偷看测试段开头).
  """
  names = list(names or sigs.keys())
  corr = factor_corr(sigs, names, min_cs)            # 与折无关, 只算一次
  idx = fwd.index
  rows = []
  for f in folds:
    tr_e_safe = f['train_end']
    pos = int(idx.searchsorted(tr_e_safe, side='right')) - 1
    safe_pos = pos - (h + 1)                         # 训练标签不越折界
    if safe_pos < 0:
      continue
    tr_e_safe = idx[safe_pos]
    fwd_tr = _win(fwd, f['train_start'], tr_e_safe)
    fwd_te = _win(fwd, f['test_start'], f['test_end'])
    sc_tr = rank_ic_scores(sigs, fwd_tr, names, min_cs)
    kept, _ = dedupe(sigs, sc_tr, thresh, min_cs, corr)
    for scheme in schemes:
      w = _weights_for(scheme, sigs, fwd_tr, kept, min_cs, min_ic, shrink,
                       soft_thresh, max_n, min_gain)
      if not w:
        rows.append({'fold': f['fold'], 'scheme': scheme, 'n_fac': 0,
                     'train_start': f['train_start'].date(),
                     'test_start': f['test_start'].date(),
                     'train_rank_ic': np.nan, 'test_rank_ic': np.nan})
        continue
      combo = combine_signals(sigs, w, min_cs)
      tr_ic = _rank_ic(combo, fwd_tr, min_cs)        # ic_series 自动对齐窗口
      te_ic = _rank_ic(combo, fwd_te, min_cs)
      rows.append({'fold': f['fold'], 'scheme': scheme, 'n_fac': len(w),
                   'train_start': f['train_start'].date(),
                   'test_start': f['test_start'].date(),
                   'train_rank_ic': round(float(tr_ic), 4) if pd.notna(tr_ic) else np.nan,
                   'test_rank_ic': round(float(te_ic), 4) if pd.notna(te_ic) else np.nan})
  return pd.DataFrame(rows)


def _wf_summary(per: pd.DataFrame) -> pd.DataFrame:
  """明细表 -> 每方案一行汇总: 折外均值/标准差/ICIR/正值占比/最小值 + 过拟合缺口."""
  if not len(per):
    return pd.DataFrame()
  out = []
  for scheme, g in per.groupby('scheme', sort=False):
    te = g['test_rank_ic'].dropna()
    tr = g['train_rank_ic'].dropna()
    mu = float(te.mean()) if len(te) else np.nan
    sd = float(te.std(ddof=1)) if len(te) > 1 else np.nan
    out.append({
        'scheme': scheme, 'n_folds': int(len(g)),
        'wf_ic_mean': round(mu, 4) if pd.notna(mu) else np.nan,
        'wf_ic_std': round(sd, 4) if pd.notna(sd) else np.nan,
        'wf_icir': round(mu / sd, 3) if (pd.notna(mu) and pd.notna(sd) and sd > 0) else np.nan,
        'wf_pos_frac': round(float((te > 0).mean()), 3) if len(te) else np.nan,
        'wf_ic_min': round(float(te.min()), 4) if len(te) else np.nan,
        'is_minus_wf': round(float(tr.mean() - mu), 4)
                       if (len(tr) and pd.notna(mu)) else np.nan,
    })
  return pd.DataFrame(out)


def walk_forward(sigs: dict, fwd: pd.DataFrame, names: list = None,
                 scheme: str = 'ic', start: str = None, end: str = None,
                 n_folds: int = 4, thresh: float = DEDUPE_RHO,
                 min_cs: int = cfg.MIN_CS, h: int = 1, min_ic: float = 0.0,
                 shrink: float = IC_SHRINK, soft_thresh: float = 0.0,
                 max_n: int = 8, min_gain: float = MIN_GAIN,
                 min_train_days: int = 120) -> tuple:
  """
  单方案的 IS 折内 walk-forward -> (逐折明细, 汇总结论).

  目的: 在**只用 IS 期**的前提下, 估计"这套定权流程换一段没见过的 IS 片段会怎样" ——
  它衡量的是流程本身的跨期稳定性, 而不是某个固定权重的表现. is_minus_wf(训练均值
  - 折外均值)就是过拟合缺口的直接估计: 越接近 0 越说明"IS 上看到的强度能延续".

  :param sigs: {因子名: 预处理后宽表}
  :param fwd: 前向收益宽表(全历史)
  :param names: 候选因子(None = 全部)
  :param scheme: 方案名
  :param start/end: 折切分窗口(默认调用方传 IS 边界)
  :param n_folds: 折数
  :param min_train_days: 每段最少交易日(不足则返回空)
  :returns: (per_fold DataFrame, summary DataFrame)
  """
  folds = _wf_folds(fwd.index, start, end, n_folds, min_train_days)
  per = _wf_records(sigs, fwd, names, folds, (scheme,), thresh, min_cs, h,
                    min_ic, shrink, soft_thresh, max_n, min_gain)
  return per, _wf_summary(per)


def consistency_table(sigs: dict, fwd: pd.DataFrame, names: list = None,
                      schemes: tuple = COMBO_SCHEMES, start: str = None,
                      end: str = None, n_folds: int = 4,
                      thresh: float = DEDUPE_RHO, min_cs: int = cfg.MIN_CS,
                      h: int = 1, min_ic: float = 0.0,
                      shrink: float = IC_SHRINK, soft_thresh: float = 0.0,
                      max_n: int = 8, min_gain: float = MIN_GAIN,
                      min_train_days: int = 120) -> tuple:
  """
  多方案 IS 折内 walk-forward 对照 -> (逐折明细, 每方案一行汇总).

  与单方案版共用同一批折, 便于横向比较"越拟合的方案是否折外越不稳".
  """
  folds = _wf_folds(fwd.index, start, end, n_folds, min_train_days)
  per = _wf_records(sigs, fwd, names, folds, tuple(schemes), thresh, min_cs, h,
                    min_ic, shrink, soft_thresh, max_n, min_gain)
  return per, _wf_summary(per)


# ============================ 自检: 标定 ============================

def verify_combine(ds: Dataset, method: str = prep.DEFAULT_PREP, h: int = 1,
                   min_cs: int = cfg.MIN_CS, thresh: float = DEDUPE_RHO,
                   top_k: int = cfg.TOP_K, q: int = 5) -> tuple:
  """
  组合层标定(三个对照, 正 / 负 / 前视各一):

    1) oracle      : 组合里只放 oracle(= 前向收益本身)  -> RankIC 应 = 1.0;
    2) dup_dedupe  : 给 mom_12_1 塞一个完全相同的副本    -> dedupe 应剔除副本;
    3) leak_weight : 全样本 IC 定权 vs 只用 IS IC 定权    -> 比较两者在 OOS 的表现,
                     前者通常虚高 —— 这就是"用未来定权"的代价.

  :param ds: Dataset
  :param method: 预处理方法
  :param h: 前向持有天数
  :param min_cs: 每日最少有效标的数
  :param thresh: 去重阈值
  :param top_k: top-k 持仓数
  :param q: 分层桶数
  :returns: (cases DataFrame[case, detail, value], leak DataFrame[scheme, is_ic, oos_ic])
  """
  fwd = ev.forward_return(ds, h)
  pf = prep._make_prep(method, min_cs)
  names = fct.list_factors()

  # --- 1) oracle ---
  sigs_or = {'oracle': pf(fwd)}
  v = _rank_ic(combine_signals(sigs_or, {'oracle': 1.0}, min_cs), fwd, min_cs)
  rows = [{'case': 'oracle 单因子组合', 'detail': '应 = 1.0', 'value': round(v, 4)}]

  # --- 2) 副本去重 ---
  sub = build_signals(ds, method, ['mom_12_1'], min_cs)
  sigs_dup = {'mom_12_1': sub['mom_12_1'], 'mom_12_1_dup': sub['mom_12_1'].copy()}
  sc = rank_ic_scores(sigs_dup, fwd, min_cs=min_cs)
  kept, dropped = dedupe(sigs_dup, sc, thresh, min_cs)
  dup_names = [d['factor'] for d in dropped]
  rows.append({'case': '副本去重(dup -> mom_12_1)',
               'detail': '应剔除 mom_12_1_dup',
               'value': f'kept={kept}, dropped={dup_names}'})

  # --- 3) 前视定权对照(IS 定权 vs 全样本定权, 都看 OOS 表现) ---
  sigs = build_signals(ds, method, names, min_cs)
  is_s, is_e = ev.window_bounds('is')
  oos_s, oos_e = ev.window_bounds('oos')
  fwd_is = _win(fwd, is_s, is_e)
  w_is = ic_weights(sigs, fwd_is, names, min_cs)          # 只用 IS 期 -> 诚实
  w_leak = ic_weights(sigs, fwd, names, min_cs)           # 用了全样本 -> 前视
  leak_rows = []
  for label, w in (('IS 定权(诚实)', w_is), ('全样本定权(前视)', w_leak)):
    rec = evaluate_combo(sigs, w, fwd, oos_s, oos_e, top_k, q, min_cs, h, label)
    leak_rows.append({'scheme': label, 'n_fac': len(w),
                      'is_rank_ic': round(_rank_ic(combine_signals(sigs, w, min_cs),
                                                   fwd_is, min_cs), 4),
                      'oos_rank_ic': rec['rank_ic'], 'oos_nw_t': rec['nw_t']})
  return pd.DataFrame(rows), pd.DataFrame(leak_rows)


# ============================ CLI ============================

def main():
  ap = argparse.ArgumentParser(description='factor 组合层: 相关诊断 / 冗余去重 / 多方案加权')
  ap.add_argument('--pool', default=cfg.DEFAULT_POOL)
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  ap.add_argument('--method', default=prep.DEFAULT_PREP, choices=prep.PREP_METHODS)
  ap.add_argument('--h', type=int, default=1, help='前向持有天数(与第3步口径一致)')
  ap.add_argument('--top-k', type=int, default=cfg.TOP_K)
  ap.add_argument('--q', type=int, default=5, help='分层桶数')
  ap.add_argument('--min-cs', type=int, default=cfg.MIN_CS)
  ap.add_argument('--group', default=None, help='只组合某类因子')
  ap.add_argument('--scheme', default='all',
                  help='equal / dir / ic / inv_vol / rp / greedy / all(逗号分隔可多选)')
  ap.add_argument('--thresh', type=float, default=DEDUPE_RHO, help='冗余阈值')
  ap.add_argument('--max-n', type=int, default=8, help='贪心最多入选个数')
  ap.add_argument('--min-gain', type=float, default=MIN_GAIN, help='贪心增量下限')
  ap.add_argument('--min-ic', type=float, default=0.0, help='IC 加权的 |RankIC| 门槛')
  ap.add_argument('--shrink', type=float, default=IC_SHRINK,
                  help='IC 权重向"方向对齐等权"收缩的比例(0=纯 IC, 1=纯等权)')
  ap.add_argument('--soft-thresh', type=float, default=0.0,
                  help='IC 权重软阈值(作用在 |RankIC| 上, 0=不生效)')
  ap.add_argument('--wf', action='store_true', help='跑 IS 折内 walk-forward 一致性表')
  ap.add_argument('--wf-folds', type=int, default=4, help='walk-forward 折数')
  ap.add_argument('--check', action='store_true', help='运行组合层标定')
  a = ap.parse_args()

  log = cfg.get_logger('factor.combine')
  ds = load_pool(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')
  log.info(f'口径: method={a.method}, h={a.h}, top_k={a.top_k}, q={a.q}, '
           f'min_cs={a.min_cs}, thresh={a.thresh}, shrink={a.shrink}, '
           f'soft_thresh={a.soft_thresh}')

  names = fct.list_factors(a.group)
  sigs = build_signals(ds, a.method, names, a.min_cs)
  fwd = ev.forward_return(ds, a.h)                        # 全历史标签
  is_s, is_e = ev.window_bounds('is')
  oos_s, oos_e = ev.window_bounds('oos')
  fwd_is, fwd_oos = _win(fwd, is_s, is_e), _win(fwd, oos_s, oos_e)

  # ---- 1. 相关性诊断 ----
  corr = factor_corr(sigs, names, a.min_cs)
  log.info(f'\n[因子相关矩阵] {len(names)} 个因子的平均截面秩相关(|rho| >= '
           f'{a.thresh} 视为冗余)')
  log.info('\n' + corr.to_string())
  pairs = [(x, y, corr.loc[x, y]) for i, x in enumerate(names) for y in names[i + 1:]]
  hi = sorted([p for p in pairs if pd.notna(p[2]) and abs(p[2]) >= a.thresh],
              key=lambda t: -abs(t[2]))
  if hi:
    log.info('[高冗余对] ' + '; '.join(f'{x}~{y}={r:+.3f}' for x, y, r in hi))

  # ---- 2. 冗余去重(定序用 IS 期 RankIC) ----
  sc_is = rank_ic_scores(sigs, fwd_is, names, a.min_cs)
  kept, dropped = dedupe(sigs, sc_is, a.thresh, a.min_cs, corr)
  log.info(f'\n[冗余去重] 按 IS 期 |RankIC| 定序, {len(names)} -> {len(kept)} 个')
  log.info(f'  保留: {kept}')
  for d in dropped:
    log.info(f'  剔除: {d["factor"]} (RankIC={d["rank_ic"]:+.4f}, '
             f'与 {d["conflict_with"]} rho={d["rho"]:+.3f})')
  if not dropped:
    log.info('  (无因子因冗余被剔除)')

  # ---- 3. 方案权重(全部只用 IS 期估计) ----
  want = list(COMBO_SCHEMES) if a.scheme == 'all' else \
      [s.strip() for s in a.scheme.split(',') if s.strip()]
  weights = {}
  if 'equal' in want:
    weights['equal'] = equal_weights(kept)
  if 'dir' in want:
    weights['dir'] = dir_equal_weights(kept)
  if 'ic' in want:
    weights['ic'] = ic_weights(sigs, fwd_is, kept, a.min_cs, a.min_ic,
                               a.shrink, a.soft_thresh)
  if 'greedy' in want:
    chosen, hist = greedy_forward(sigs, fwd_is, kept, a.max_n, a.min_gain, a.min_cs)
    weights['greedy'] = equal_weights(chosen)
    log.info(f'\n[贪心前向选择] 候选 {len(kept)} 个, 入选 {len(chosen)} 个: {chosen}')
    if len(hist):
      log.info('\n' + hist.to_string(index=False))

  log.info('\n[各方案权重] (均为 IS 期估计; 权重总和 = sum|w|)')
  for label, w in weights.items():
    body = ', '.join(f'{k}={v:+.3f}' for k, v in w.items())
    log.info(f'  {label:6s} (n={len(w)}): {body}')

  # ---- 4. IS 定权 -> IS / OOS 双窗评估 ----
  best_name = max(sc_is, key=lambda n: sc_is[n] if pd.notna(sc_is[n]) else -9)
  rows = []
  for label, w in weights.items():
    if not w:
      continue
    for wname, (s, e) in (('IS', (is_s, is_e)), ('OOS', (oos_s, oos_e))):
      rec = evaluate_combo(sigs, w, fwd, s, e, a.top_k, a.q, a.min_cs, a.h, label)
      rows.append({'scheme': label, 'window': wname, 'n_fac': len(w),
                   'ic': rec['ic'], 'rank_ic': rec['rank_ic'], 'nw_t': rec['nw_t'],
                   'pos_rate': rec['pos_rate'], 'excess': rec['excess'],
                   'ls_spread': rec['ls_spread'], 'mono': rec['mono'],
                   'turnover': rec['turnover']})
  df = pd.DataFrame(rows)
  log.info(f'\n[组合评估] 权重在 IS 期定, 分别看 IS / OOS (单因子最强基准: '
           f'{best_name} IS RankIC={sc_is[best_name]:+.4f})')
  log.info('\n' + df.to_string(index=False))

  # ---- 4b. IS 折内 walk-forward 一致性 ----
  if a.wf:
    per, cons = consistency_table(sigs, fwd, names, tuple(want), is_s, is_e,
                                  a.wf_folds, a.thresh, a.min_cs, a.h, a.min_ic,
                                  a.shrink, a.soft_thresh, a.max_n, a.min_gain)
    log.info(f'\n[IS 折内 walk-forward] 折数={a.wf_folds}, 逐折拟合(只用折内训练窗) '
             f'-> 折外测试; is_minus_wf = 训练均值 - 折外均值(过拟合缺口, 越接近 0 越好)')
    if len(per):
      log.info('\n' + per.to_string(index=False))
      log.info('\n' + cons.to_string(index=False))
    else:
      log.info('  (交易日不足, 无法切出折; 可减小 --wf-folds)')

  # ---- 5. 标定 ----
  if a.check:
    cases, leak = verify_combine(ds, a.method, a.h, a.min_cs, a.thresh, a.top_k, a.q)
    log.info('\n[标定 1/2] 组合引擎(正例 / 去重)')
    log.info('\n' + cases.to_string(index=False))
    log.info('\n[标定 2/2] 前视对照 —— 两种定权, 同样在 OOS 上评估')
    log.info('\n' + leak.to_string(index=False))
    log.info('[预期] IS 定权与全样本定权的 OOS 表现应当接近; 若全样本定权明显更高, '
             '说明"用未来定权"会凭空造出虚假收益, 必须坚持只用 IS.')


if __name__ == '__main__':
  main()

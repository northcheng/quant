# -*- coding: utf-8 -*-
"""
factor.report — 报告层(第6步)
==============================
职责: 把前五层的产物汇总成"一份能看的报告", 并用一条命令把全流程跑通.

--- 前五层各自产出什么 ---
  第0步 data     -> Dataset(宽表)
  第1步 factor   -> 56 个因子宽表(原始值)
  第2步 prepare  -> 预处理后的可比分数
  第3步 evaluate -> 单因子预测力表(IC/RankIC/分层/换手 + FDR/静态度诚实性)
  第4步 combine  -> 去重后的因子集 + 四套权重(equal/dir/ic/greedy)
  第5步 backtest -> 净值曲线 + 绩效 + IS/OOS
本层不发明任何新指标, 只把上面这些东西**并排摆出来**, 让人能一眼看清:
  单因子谁强谁弱 -> 组合后有没有变强 -> 落到净值上到底赚不赚钱 -> OOS 能不能复现.

--- 为什么要有"报告层" ---
  前五层每一步都能单独运行, 但它们分散在多个模块、多张表里. 真正做研究时,
  需要的是"一次跑完 + 产物落盘 + 图能直接看". 本层就是这个粘合层:
  一条 CLI 串起 data -> factor -> prepare -> evaluate -> combine -> backtest,
  把汇总表存成 CSV, 把关键关系画成 PNG, 全部落到 config.OUT_DIR/{pool}/
  (按池分子目录, 避免多池产物互相覆盖).

--- 三张图(为什么是这三张) ---
  1. IC 衰减      : 因子预测力随持有期 h 怎么掉 -> 决定该用多长的持有期;
  2. 分层收益     : Q1..Qq 是否单调 -> 说明信号是"真有区分度"还是只有 top 端有效;
  3. 净值曲线     : 各方案 + 全池基准(同一引擎核算), 并标出 IS/OOS 分隔线 ->
                    一眼看出"样本内选出来的东西, 样本外还灵不灵".

--- 本层必须显式声明的两件事(诚实性) ---
  1. 生存偏差: etf_3x 池 28 只标的都从 2020-01-02 起、零缺口 —— 意味着"只统计
     活到今天的标的", 回测天然偏乐观. 报告头用 data_quality() 实测并写明;
  2. 未复权: pkl 里 Adj Close == Close 且 Dividend/Split 全为常量 -> 数据未做
     分红/拆股调整, 除权日的跳空会被当成真实收益. 同样实测并写明.

--- 红线: 报告的每个数字都必须可复现 ---
  权重只在 IS 期估计(第4步红线); 图与表都直接调用前五层的函数, 不另写口径.
  verify_report() 端到端跑一遍精简流水线, 检查关键不变量(表完整 / 净值有限 /
  前视对照 direction 正确), 确保"能出报告"本身不是靠运气.

自检:
  cd ~/git && python -m quant.factor.report --pool etf_3x
  cd ~/git && python -m quant.factor.report --pool etf_3x --check
  cd ~/git && python -m quant.factor.report --pool etf_3x --no-fig
  cd ~/git && python -m quant.factor.report --pool etf_3x --write-config   # 导出因子配置给生产端
"""
import argparse
import datetime
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')                             # 无 GUI 后端: 服务器/终端也能存图
import matplotlib.pyplot as plt

from quant.factor import config as cfg
from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import evaluate as ev
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import Dataset, load_pool

# 汇总表展示列(预测力 + 组合表现 + 评估诚实性, 与第3步 evaluate 的三组列一致)
FACTOR_COLS = ['factor', 'group', 'dir', 'dir_ok', 'n_days', 'ic', 'rank_ic',
               'rank_icir', 'nw_t', 'pos_rate', 'excess', 'ls_spread', 'mono',
               'turnover', 'ac1', 'ac5', 'ac20', 'nw_p', 'fdr_q', 'fdr_sig',
               'sig_ac1', 'is_static', 'top3_share', 'n_eff_symbols',
               'dyn_minus_static']

# 摘要里"多重检验后仍显著"小表的展示列(精简, 只留判定相关的几列)
HONEST_SHOW = ['factor', 'rank_ic', 'nw_t', 'nw_p', 'fdr_q', 'sig_ac1',
               'dyn_minus_static']

# 图1/图2 默认只画 IS |RankIC| 前 top_n 名; 但冗余对的"被剔除方"往往掉出榜外,
# 没法并排比较. FIG_EXTRA 里的因子会被强制补进候选名单(仅影响出图, 不影响任何口径).
FIG_EXTRA = ['dd_60']                            # hi_252 的冗余对照


# ============================ 小工具 ============================

def _win(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
  """按交易日截取宽表的行(只选时间窗, 不改内容)."""
  idx = df.index
  if start is not None:
    idx = idx[idx >= pd.Timestamp(start)]
  if end is not None:
    idx = idx[idx <= pd.Timestamp(end)]
  return df.loc[idx]


def _setup_matplotlib() -> None:
  """matplotlib 中文字体 + 负号配置(避免图上中文变方框、负号变乱码)."""
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
  plt.rcParams['axes.unicode_minus'] = False


def _save_fig(fig, name: str, out_dir: Path = None) -> str:
  """存图到 out_dir/{name}.png(out_dir 缺省用 cfg.OUT_DIR)."""
  d = Path(out_dir) if out_dir else cfg.OUT_DIR
  d.mkdir(parents=True, exist_ok=True)
  path = d / f'{name}.png'
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)
  return str(path)


def save_table(df: pd.DataFrame, name: str, index: bool = False,
               out_dir: Path = None) -> str:
  """存表到 out_dir/{name}.csv(out_dir 缺省用 cfg.OUT_DIR; utf-8-sig 便于 Excel 打开中文)."""
  d = Path(out_dir) if out_dir else cfg.OUT_DIR
  d.mkdir(parents=True, exist_ok=True)
  path = d / f'{name}.csv'
  df.to_csv(path, index=index, encoding='utf-8-sig')
  return str(path)


# ============================ 1. 数据质量声明 ============================

def data_quality(ds: Dataset) -> pd.DataFrame:
  """
  实测数据质量 -> 报告头必须写明的免责事实.

  为什么放在报告层: 因子/评估/回测各层都假设"数据是干净的", 但它们都不负责说明
  数据**哪里不干净**. 把这些问题集中在这里显式露出, 避免读者把偏差当成 alpha.

  :param ds: Dataset
  :returns: DataFrame(check, value, note)
  """
  close = ds.wide('close')
  cols = ds.panel.columns
  rows = [{'check': '池 / 规模',
           'value': f'{ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日',
           'note': f'{ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}'},
          {'check': 'close NaN 占比', 'value': f'{close.isna().mean().mean():.2%}',
           'note': '停牌/未上市为 NaN, 上层按截面屏蔽(不填 0)'}]

  first = close.apply(lambda s: s.first_valid_index())
  n_first = int(first.nunique())
  rows.append({'check': '上市起点一致性', 'value': f'{n_first} 个不同起点',
               'note': ('全部同日上市 -> 存在生存偏差, 回测偏乐观'
                        if n_first == 1 else '起点分散 -> 生存偏差较小')})

  if 'Adj Close' in cols:
    a, c = ds.wide('Adj Close'), close
    d = float(np.nanmax(np.abs(a.to_numpy(dtype=float) - c.to_numpy(dtype=float))))
    rows.append({'check': '复权价一致性', 'value': f'max|AdjClose - Close| = {d:g}',
                 'note': ('相同 -> 数据未做分红/拆股调整'
                          if d < 1e-9 else '存在复权差异')})
  else:
    rows.append({'check': '复权价', 'value': '无 Adj Close 列', 'note': ''})

  for c in ('Dividend', 'Split'):
    if c not in cols:
      continue
    w = ds.wide(c).dropna()
    if c == 'Split':
      all_one = bool((w == 1.0).all().all())
      rows.append({'check': '拆股列 Split',
                   'value': f'非1占比 {float((w != 1.0).to_numpy().mean()):.2%}',
                   'note': '全为 1 -> 无拆股事件' if all_one else '含非 1 值 -> 有拆股事件'})
    else:
      all_zero = bool((w == 0.0).all().all())
      rows.append({'check': '分红列 Dividend',
                   'value': f'Σ|value| = {float(w.abs().to_numpy().sum()):g}',
                   'note': '全为 0 -> 无分红事件' if all_zero else '含非 0 值 -> 有分红'})

  for c in ('resistant', 'support'):
    if c in cols:
      r = float(ds.wide(c).isna().mean().mean())
      rows.append({'check': f'疑似空列 {c}', 'value': f'NaN 占比 {r:.1%}',
                   'note': '几乎全空 -> 垃圾列, 因子层已避开'})
  return pd.DataFrame(rows)


# ============================ 2. 三张图 ============================

def plot_ic_decay(ds: Dataset, sigs: dict, names: list,
                  hs: tuple = (1, 2, 3, 5, 10, 20), start: str = None,
                  end: str = None, min_cs: int = cfg.MIN_CS,
                  out_dir: Path = None) -> str:
  """
  图1: IC 衰减曲线 —— 每个因子一条线, x = 持有期 h, y = RankIC 均值.

  看点: 曲线越平 -> 信号越能承载长持有期(换手低); 快速跌破 0 -> 只适合极短持有.

  :param ds: Dataset
  :param sigs: {因子名: 预处理后宽表}
  :param names: 要画的因子
  :param hs: 持有期列表
  :param start/end: 评估窗口
  :param min_cs: 每日最少有效标的数
  :param out_dir: 输出目录(缺省 cfg.OUT_DIR)
  :returns: 图片路径
  """
  fig, ax = plt.subplots(figsize=(8.4, 4.8))
  for name in names:                              # fwd 按 h 预计算, 避免重复
    ys = []
    for h in hs:
      fwd_h = _win(ev.forward_return(ds, h), start, end)
      ic = ev.ic_series(_win(sigs[name], start, end), fwd_h, 'spearman', min_cs)
      ys.append(float(ic.mean()) if len(ic) else np.nan)
    ax.plot(list(hs), ys, marker='o', ms=4, lw=1.4, label=name)
  ax.axhline(0.0, color='#999', lw=0.8, ls='--')
  ax.set_xlabel('前向持有期 h (交易日)')
  ax.set_ylabel('RankIC 均值')
  ax.set_title('图1 IC 衰减: 预测力随持有期如何变化')
  ax.set_xticks(list(hs))
  ax.grid(alpha=0.25)
  ax.legend(fontsize=8, ncol=2)
  return _save_fig(fig, 'fig1_ic_decay', out_dir)


def plot_quantile(sigs: dict, fwd: pd.DataFrame, names: list, q: int = 5,
                  start: str = None, end: str = None,
                  min_cs: int = cfg.MIN_CS, out_dir: Path = None) -> str:
  """
  图2: 分层收益柱状图 —— 每个因子一格, Q1(因子最小)..Qq(因子最大) 的前向收益均值.

  看点: 若柱高大致单调递增, 说明信号在**整个截面**都有区分度;
        只有 Qq 一根高 -> 只在极端端有效. 标题附多空价差 LS(Qq-Q1).

  :param sigs: {因子名: 预处理后宽表}
  :param fwd: 前向收益宽表
  :param names: 要画的因子
  :param q: 分桶数
  :param start/end: 评估窗口
  :param min_cs: 每日最少有效标的数
  :param out_dir: 输出目录(缺省 cfg.OUT_DIR)
  :returns: 图片路径
  """
  n = len(names)
  ncols = 3 if n > 1 else 1
  nrows = ceil(n / ncols)
  fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
  for i, name in enumerate(names):
    ax = axes[i // ncols][i % ncols]
    qr = ev.quantile_returns(_win(sigs[name], start, end), _win(fwd, start, end), q, min_cs)
    if qr.empty:
      ax.set_title(f'{name} (无数据)')
      continue
    bars = qr.iloc[:q]
    colors = ['#c0392b' if v < 0 else '#27ae60' for v in bars['mean_fwd_ret']]
    ax.bar(bars['bucket'], bars['mean_fwd_ret'], color=colors, alpha=0.85)
    ls = float(qr.iloc[-1]['mean_fwd_ret'])
    ax.axhline(0.0, color='#666', lw=0.8)
    ax.set_title(f'{name}  LS={ls:+.4f}', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, axis='y')
  for j in range(n, nrows * ncols):               # 关掉多余的子图
    axes[j // ncols][j % ncols].axis('off')
  fig.suptitle('图2 分层收益: Q1(因子最小) -> Qq(因子最大)', fontsize=11)
  return _save_fig(fig, 'fig2_quantile', out_dir)


def plot_equity(curves: dict, split: str = cfg.IS_END, title: str = None,
                name: str = 'fig3_equity', out_dir: Path = None) -> str:
  """
  图3: 净值曲线对照 —— 各方案 + 全池基准, 并标出 IS/OOS 分隔线.

  所有曲线都用**同一个引擎**核算(open-to-open, 含成本), 因此高低可直接比较.
  分隔线左侧是选参用的样本内, 右侧才是真正的样本外检验.

  :param curves: {标签: 净值 Series}
  :param split: IS/OOS 分隔日(画竖线); None 则不画
  :param title: 图标题(None 用默认)
  :param name: 输出文件名(不含扩展名)
  :param out_dir: 输出目录(缺省 cfg.OUT_DIR)
  :returns: 图片路径
  """
  fig, ax = plt.subplots(figsize=(10.5, 5.2))
  for label, eq in curves.items():
    lw = 2.0 if label in ('equal', 'dir', 'ic', 'greedy') else 1.3
    ax.plot(eq.index, eq.to_numpy(), label=label, lw=lw, alpha=0.9)
  if split is not None:
    ax.axvline(pd.Timestamp(split), color='#888', ls='--', lw=1.0)
    ax.text(pd.Timestamp(split), ax.get_ylim()[0], ' IS | OOS', fontsize=8,
            color='#666', va='bottom')
  ax.set_title(title or f'图3 净值曲线(open-to-open, 含 {cfg.COST_BPS:.0f}bps 单边成本)')
  ax.set_ylabel('净值(起点 = 1.0)')
  ax.grid(alpha=0.25)
  ax.legend(fontsize=8, ncol=2)
  return _save_fig(fig, name, out_dir)


# ============================ 3. 稳健性附加分析 ============================

# 成本阶梯默认档位(单边 bps): 0 = 无摩擦理想值, 其余为不同费率假设.
COST_LADDER_BPS = (0.0, 10.0, 25.0, 50.0)


def _breakeven_bps(xs: list, ys: list) -> float:
  """线性插值求 ys(收益)=0 处的成本(bps); 整段同号(不跨越 0)则返回 nan."""
  for i in range(1, len(xs)):
    y0, y1 = ys[i - 1], ys[i]
    if pd.isna(y0) or pd.isna(y1):
      continue
    if y0 == 0:
      return float(xs[i - 1])
    if (y0 < 0 < y1) or (y0 > 0 > y1):
      return float(xs[i - 1] + (0.0 - y0) * (xs[i] - xs[i - 1]) / (y1 - y0))
  return float('nan')


def cost_ladder_scan(open_w: pd.DataFrame, close_w: pd.DataFrame, combos: dict,
                     p: bt.EngineParams, start: str = None, end: str = None,
                     levels: tuple = COST_LADDER_BPS,
                     min_turn: float = 5.0) -> pd.DataFrame:
  """
  成本阶梯: 固定窗口下, 对每个方案扫不同**单边成本**, 看绩效随成本如何衰减.

  换手是成本敏感度的决定因素(往返吃 2*turnover*cost): 阶梯越陡 = 越依赖低费率
  假设; 平缓 = 天然抗摩擦. 附 breakeven_bps(总收益跨 0 的成本档)供横向对照.

  :param open_w/close_w: 开 / 收盘宽表
  :param combos: {方案名: 合成信号}
  :param p: EngineParams(成本被逐档覆盖, 其余参数不变)
  :param start/end: 窗口
  :param levels: 成本档位(单边 bps)
  :param min_turn: 年化换手低于此值 -> 视为近乎免疫成本, breakeven 记 inf
  :returns: 长表(kind, cost_bps, sharpe, total_ret, excess_bh, cagr, max_dd,
            ann_turnover, breakeven_bps)
  """
  ow, cw = _win(open_w, start, end), _win(close_w, start, end)
  bh = bt.buyhold(ow, cw)
  bh_ret = bt.perf_stats(bh['equity'], bh['ann_turnover'], bh['trades'])['total_ret']
  rows = []
  for kind, comp in combos.items():
    wc = _win(comp, start, end)
    for c in levels:
      res = bt.run_engine(ow, cw, wc, p.copy(cost_bps=c))
      s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      ret = s.get('total_ret')
      rows.append({'kind': kind, 'cost_bps': float(c), 'sharpe': s.get('sharpe'),
                   'total_ret': ret,
                   'excess_bh': (ret - bh_ret if pd.notna(ret) and
                                 pd.notna(bh_ret) else np.nan),
                   'cagr': s.get('cagr'), 'max_dd': s.get('max_dd'),
                   'ann_turnover': s.get('ann_turnover')})
  df = pd.DataFrame(rows)
  be = {}
  for kind in combos:
    sub = df[df['kind'] == kind].sort_values('cost_bps')
    t0 = sub['ann_turnover'].iloc[0]
    be[kind] = (float('inf') if pd.notna(t0) and t0 < min_turn
                else _breakeven_bps(list(sub['cost_bps']), list(sub['total_ret'])))
  df['breakeven_bps'] = df['kind'].map(be)
  return df


# ============================ 3b. 流水线 ============================

def run_report(ds: Dataset, method: str = prep.DEFAULT_PREP, h: int = 1,
               top_k: int = cfg.TOP_K, q: int = 5, min_cs: int = cfg.MIN_CS,
               group: str = None, scheme: str = 'all',
               thresh: float = cmb.DEDUPE_RHO, max_n: int = 8,
               min_gain: float = cmb.MIN_GAIN, min_ic: float = 0.0,
               shrink: float = cmb.IC_SHRINK, soft_thresh: float = 0.0,
               wf_folds: int = 4,
               top_n: int = 6, p: bt.EngineParams = None,
               figures: bool = True, win_start: str = None, win_end: str = None,
               win_name: str = 'YTD', cost_ladder: bool = False,
               exclude_static: bool = False) -> dict:
  """
  全流程: data -> factor -> prepare -> evaluate -> combine -> backtest -> 出表出图.

  定权铁律: equal / dir / ic / greedy 四套权重**只用 IS 期** fwd 估计(第4步红线),
  再在 IS / OOS 两窗各自推进引擎 —— OOS 才是策略能否复现的判据.

  附加窗口(win_start/win_end 给出时): 再开一段, 如 2026-01-01 至今.
  注意它同样**从空仓、净值 1.0 独立推进**, 不能从 OOS 曲线上剪一段 —— 那一段
  带着 2025 年建起来的持仓, 不是"该段起点开始交易"的结果. 权重仍取 IS 冻结的那套.

  :param ds: Dataset
  :param method: 预处理方法(见 prepare.PREP_METHODS)
  :param h: 前向持有天数
  :param top_k: 入场 rank 门槛(持仓数上限)
  :param q: 分层桶数
  :param min_cs: 每日最少有效标的数
  :param group: 只用某类因子
  :param scheme: 'all' 或 equal/ic/greedy 逗号分隔
  :param thresh: 冗余去重阈值
  :param max_n: 贪心最多入选个数
  :param min_gain: 贪心增量下限
  :param min_ic: IC 加权的 |RankIC| 门槛
  :param shrink: IC 权重向"方向对齐等权"收缩的比例(0=纯 IC, 1=纯等权)
  :param soft_thresh: IC 权重软阈值(作用在 |RankIC| 上, 0=不生效)
  :param wf_folds: IS 折内 walk-forward 的一致性对照折数
  :param top_n: 画图取前 N 个因子(按 IS |RankIC|)
  :param p: EngineParams(None 用默认)
  :param figures: 是否出图
  :param win_start: 附加窗口起点(None 则不开附加窗)
  :param win_end: 附加窗口终点(None 表示到数据末尾)
  :param win_name: 附加窗口标签(用于汇总表与文件名)
  :param cost_ladder: 是否追加"成本阶梯"扫描(0/10/25/50bps, OOS 窗)
  :param exclude_static: 是否把"静态选票"(IS 窗 sig_ac1 >= STATIC_AC1)剔出组合 ——
                        只影响去重/定权/回测; 评估表仍保留全部因子以便审计
  :returns: dict(各汇总表 / 权重 / 路径 / 净值曲线)
  """
  p = p or bt.EngineParams(top_k=top_k)
  names = fct.list_factors(group)
  sigs = cmb.build_signals(ds, method, names, min_cs)
  fwd = ev.forward_return(ds, h)                  # 全历史标签(截窗只筛评估日)
  is_s, is_e = ev.window_bounds('is')
  oos_s, oos_e = ev.window_bounds('oos')
  fwd_is = _win(fwd, is_s, is_e)
  open_w, close_w = ds.wide('open'), ds.wide('close')

  # ---- 评估: 单因子 IS / OOS ----
  eval_is = ev.evaluate_factors(ds, names, method, h, top_k, q, min_cs, 'is')
  eval_oos = ev.evaluate_factors(ds, names, method, h, top_k, q, min_cs, 'oos')

  # 剔除静态选票(可选): sig_ac1 >= STATIC_AC1 说明信号几乎逐日不变(静态身份,
  # 而非动态 alpha). 只从"组合输入"里摘掉, eval 表仍保留全部因子以便审计.
  excluded_static = []
  if exclude_static:
    excluded_static = eval_is.loc[eval_is['is_static'], 'factor'].tolist()
    drop = set(excluded_static)
    names = [n for n in names if n not in drop]
    sigs = {k: v for k, v in sigs.items() if k not in drop}

  # ---- 组合: 相关诊断 -> 去重(IS 定序) -> 三方案(IS 定权) ----
  corr = cmb.factor_corr(sigs, names, min_cs)
  sc_is = cmb.rank_ic_scores(sigs, fwd_is, names, min_cs)
  kept, dropped = cmb.dedupe(sigs, sc_is, thresh, min_cs, corr)

  want = list(cmb.COMBO_SCHEMES) if scheme == 'all' else \
      [s.strip() for s in scheme.split(',') if s.strip()]
  weights = {}
  if 'equal' in want:
    weights['equal'] = cmb.equal_weights(kept)
  if 'dir' in want:
    weights['dir'] = cmb.dir_equal_weights(kept)
  if 'ic' in want:
    weights['ic'] = cmb.ic_weights(sigs, fwd_is, kept, min_cs, min_ic,
                                   shrink, soft_thresh)
  if 'greedy' in want:
    chosen, _ = cmb.greedy_forward(sigs, fwd_is, kept, max_n, min_gain, min_cs)
    if chosen:
      weights['greedy'] = cmb.equal_weights(chosen)

  best = max(sc_is, key=lambda n: abs(sc_is[n]) if pd.notna(sc_is[n]) else -1.0)
  combos = {k: cmb.combine_signals(sigs, w, min_cs) for k, w in weights.items()}
  combos[f'single:{best}'] = cmb.combine_signals(sigs, {best: 1.0}, min_cs)
  if 'equal' in weights:
    combos['shuffle(equal)'] = bt.shuffle_composite(combos['equal'], seed=42)

  # ---- IS 折内 walk-forward 一致性: 切 IS 期成折, 逐折只用折内训练窗定权 -> 折外测试 ----
  wf_per, consistency = cmb.consistency_table(
      sigs, fwd, names, tuple(want), is_s, is_e, wf_folds, thresh, min_cs, h,
      min_ic, shrink, soft_thresh, max_n, min_gain)

  # ---- 回测: 每方案 x 每窗独立推进引擎 ----
  windows = [('IS', is_s, is_e), ('OOS', oos_s, oos_e)]
  if win_start or win_end:
    windows.append((win_name, win_start, win_end))
  rows = []
  for kind, comp in combos.items():
    if kind in weights:
      nf = len(weights[kind])
    elif kind.startswith('shuffle'):
      nf = len(kept)
    else:
      nf = 1
    for win, ws, we in windows:
      res = bt.run_engine(_win(open_w, ws, we), _win(close_w, ws, we),
                          _win(comp, ws, we), p)
      s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      rows.append({'kind': kind, 'window': win, 'n_fac': nf,
                   **{k: s.get(k) for k in bt.STAT_KEYS}})
  for win, ws, we in windows:
    res = bt.buyhold(_win(open_w, ws, we), _win(close_w, ws, we))
    s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    rows.append({'kind': 'buyhold', 'window': win, 'n_fac': len(open_w.columns),
                 **{k: s.get(k) for k in bt.STAT_KEYS}})
  backtest = pd.DataFrame(rows)

  # ---- 稳健性附加分析(可选): 成本阶梯 ----
  ladder = cost_ladder_scan(open_w, close_w, combos, p, oos_s, oos_e) \
      if cost_ladder else None

  # ---- 全交易窗净值曲线(供绘图) + 分年收益 ----
  main_kind = 'equal' if 'equal' in weights else (next(iter(weights)) if weights
                                                  else f'single:{best}')
  plot_kinds = [k for k in ('equal', 'dir', 'ic', 'greedy', f'single:{best}',
                            'shuffle(equal)') if k in combos]
  curves = {}
  for kind in plot_kinds + ['buyhold']:
    if kind == 'buyhold':
      res = bt.buyhold(_win(open_w, cfg.TRADE_START, None),
                       _win(close_w, cfg.TRADE_START, None))
    else:
      res = bt.run_engine(_win(open_w, cfg.TRADE_START, None),
                          _win(close_w, cfg.TRADE_START, None),
                          _win(combos[kind], cfg.TRADE_START, None), p)
    curves[kind] = res['equity']
  yearly = bt.yearly_returns(curves[main_kind])

  # ---- 附加窗口独立净值(从空仓重启, 不是从长曲线剪一段) ----
  curves_win = None
  if win_start or win_end:
    curves_win = {}
    for kind in plot_kinds:
      res = bt.run_engine(_win(open_w, win_start, win_end),
                          _win(close_w, win_start, win_end),
                          _win(combos[kind], win_start, win_end), p)
      curves_win[kind] = res['equity']
    res = bt.buyhold(_win(open_w, win_start, win_end),
                     _win(close_w, win_start, win_end))
    curves_win['buyhold'] = res['equity']

  # ---- 出表(按池分子目录, 避免多个池互相覆盖) ----
  # 剔除静态票时写入 *_excl_static 子目录, 保留未剔除的基线产物以便对照
  out_dir = cfg.OUT_DIR / (f'{ds.pool}_excl_static' if exclude_static else ds.pool)
  quality = data_quality(ds)
  paths = {
      'quality': save_table(quality, 'data_quality', out_dir=out_dir),
      'eval_is': save_table(eval_is[FACTOR_COLS], 'factor_eval_IS', out_dir=out_dir),
      'eval_oos': save_table(eval_oos[FACTOR_COLS], 'factor_eval_OOS', out_dir=out_dir),
      'corr': save_table(corr, 'factor_corr', index=True, out_dir=out_dir),
      'backtest': save_table(backtest, 'backtest_summary', out_dir=out_dir),
      'yearly': save_table(yearly.rename('ret').reset_index()
                           .rename(columns={'index': 'year'}), 'yearly_returns',
                           out_dir=out_dir),
  }
  if dropped:
    paths['dropped'] = save_table(pd.DataFrame(dropped), 'dropped_factors',
                                  out_dir=out_dir)
  if curves_win is not None:
    paths[f'curve_{win_name}'] = save_table(pd.DataFrame(curves_win),
                                            f'curve_{win_name}', index=True,
                                            out_dir=out_dir)
  if ladder is not None:
    paths['cost_ladder'] = save_table(ladder, 'cost_ladder', out_dir=out_dir)
  if len(consistency):
    paths['walk_forward'] = save_table(consistency, 'walk_forward_consistency',
                                       out_dir=out_dir)
    paths['walk_forward_folds'] = save_table(wf_per, 'walk_forward_folds',
                                             out_dir=out_dir)

  # ---- 出图 ----
  if figures:
    _setup_matplotlib()
    top_names = sorted(names, key=lambda n: abs(sc_is.get(n, np.nan))
                       if pd.notna(sc_is.get(n, np.nan)) else -1.0,
                       reverse=True)[:top_n]
    for name in FIG_EXTRA:                       # 强制补进冗余对照因子(去重, 不重复画)
      if name in names and name not in top_names:
        top_names.append(name)
    paths['fig_ic_decay'] = plot_ic_decay(ds, sigs, top_names, start=is_s, end=is_e,
                                          min_cs=min_cs, out_dir=out_dir)
    paths['fig_quantile'] = plot_quantile(sigs, fwd, top_names, q, is_s, is_e, min_cs,
                                          out_dir=out_dir)
    paths['fig_equity'] = plot_equity(curves, split=cfg.IS_END, out_dir=out_dir)
    if curves_win is not None:
      paths[f'fig_equity_{win_name}'] = plot_equity(
          curves_win, split=None, name=f'fig4_equity_{win_name}',
          title=(f'图4 {win_name} 窗口独立净值(窗口起点空仓重启, 含 '
                 f'{cfg.COST_BPS:.0f}bps 单边成本)'), out_dir=out_dir)

  return {'quality': quality, 'eval_is': eval_is, 'eval_oos': eval_oos,
          'corr': corr, 'backtest': backtest, 'yearly': yearly,
          'kept': kept, 'dropped': dropped, 'scores_is': sc_is, 'weights': weights,
          'consistency': consistency,
          'curves': curves, 'curves_win': curves_win, 'windows': windows,
          'cost_ladder': ladder, 'excluded_static': excluded_static,
          'win_name': win_name, 'main_kind': main_kind, 'paths': paths, 'params': p,
          'out_dir': out_dir}


# ============================ 3b. 研究结论落盘(供生产脚本读取) ============================

def _jsonable(obj):
  """把 numpy 标量 / NaN 转成可 JSON 序列化的等价物(NaN -> None)."""
  if isinstance(obj, dict):
    return {k: _jsonable(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_jsonable(v) for v in obj]
  if isinstance(obj, (np.floating, float)):
    f = float(obj)
    return f if np.isfinite(f) else None
  if isinstance(obj, (np.integer,)):
    return int(obj)
  if isinstance(obj, (np.bool_,)):
    return bool(obj)
  return obj


def export_config(rep: dict, ds: Dataset, pool: str, interval: str, method: str,
                  min_cs: int, top_k: int, exit_rank: int,
                  default_scheme: str = 'greedy') -> dict:
  """
  把研究结论打包成 factor_config.json 的结构, 供生产端(signal_generator)直接读取.

  只导出**参数**(因子名单 + 各方案权重 + 口径), 不导出任何逐日序列: 生产端在最新
  数据上重算因子值, 用这里冻结的权重合成信号 —— 与"因子全历史算, IS 期定权"的口径一致,
  避免每次改因子/权重都去改生产脚本里的硬编码常量.

  导出的方案: equal / dir / ic / greedy(来自 run_report 的 IS 定权) + single(|RankIC| 最强的单因子).

  :param rep: run_report 的返回
  :param ds: Dataset(取数据起止日)
  :param pool/interval: 池与频率
  :param method: 预处理方法
  :param min_cs: 每日最少有效标的数
  :param top_k/exit_rank: 三态门槛
  :param default_scheme: 生产端默认使用的方案
  :returns: dict(meta / params / kept / dropped / scores_is / best_single / schemes)
  """
  sc = rep['scores_is']
  best = (max(sc, key=lambda n: abs(sc[n]) if pd.notna(sc[n]) else -1.0)
          if sc else None)
  schemes = {k: {n: float(w) for n, w in v.items()} for k, v in rep['weights'].items()}
  if best is not None:
    schemes['single'] = {best: 1.0}
  is_s, is_e = ev.window_bounds('is')
  oos_s, oos_e = ev.window_bounds('oos')
  return {
      'meta': {
          'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
          'source': 'quant.factor.report', 'pool': pool, 'interval': interval,
          'method': method, 'data_start': ds.dates.min().strftime('%Y-%m-%d'),
          'data_end': ds.dates.max().strftime('%Y-%m-%d'),
          'is_window': [is_s, is_e], 'oos_window': [oos_s, oos_e],
      },
      'params': {'prep': method, 'min_cs': int(min_cs), 'top_k': int(top_k),
                 'exit_rank': int(exit_rank), 'default_scheme': default_scheme},
      'kept': list(rep['kept']),
      'dropped': rep['dropped'],
      'scores_is': sc,
      'best_single': best,
      'schemes': schemes,
  }


def _load_config_doc(path: Path) -> dict:
  """
  读现有 factor_config.json -> {'default_pool', 'pools': {pool: conf}}.

  兼容旧版(单池扁平结构): 旧的 {meta/params/kept/schemes...} 会被收进
  pools[meta.pool], 因此升级到多池结构时不会丢结论.
  """
  doc = {}
  if path.exists():
    try:
      with open(path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    except Exception:
      doc = {}
  pools = doc.get('pools') if isinstance(doc, dict) else None
  if not isinstance(pools, dict):
    pools = {}
    if isinstance(doc, dict) and doc.get('schemes'):        # 旧版: 单池扁平
      old = (doc.get('meta') or {}).get('pool') or cfg.DEFAULT_POOL
      pools[old] = doc
  default_pool = doc.get('default_pool') if isinstance(doc, dict) else None
  return {'default_pool': default_pool, 'pools': pools}


def write_config(conf: dict, path=None) -> Path:
  """
  把 export_config 的产物写入 factor_config.json(多池: {default_pool, pools{pool:conf}}).

  **按池合并**: 只更新本次 conf.meta.pool 对应的一节, 其它池的既有条目原样保留,
  因此可以逐池跑 report --write-config, 彼此不覆盖. 旧版单池文件会自动迁移.

  :param conf: export_config 的返回(须含 meta.pool)
  :param path: 输出路径(None 用默认 cfg.FACTOR_CONFIG)
  :returns: 实际写入的 Path
  """
  path = Path(path or cfg.FACTOR_CONFIG)
  path.parent.mkdir(parents=True, exist_ok=True)
  pool = (conf.get('meta') or {}).get('pool')
  if not pool:
    raise ValueError('conf.meta.pool 缺失, 无法写入多池配置')
  doc = _load_config_doc(path)
  doc['pools'][pool] = conf
  doc['default_pool'] = doc.get('default_pool') or pool
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(_jsonable(doc), f, ensure_ascii=False, indent=2)
  return path


# ============================ 4. 报告文本 ============================

def _honest_section(tab: pd.DataFrame) -> list:
  """
  把 IS 评估表的"诚实性三件套"拼成报告文字段:
    - BH-FDR: 一次评估几十个因子, 不做多重检验校正就会把"撞上的显著"当真 alpha;
    - 静态选票: sig_ac1 >= STATIC_AC1(信号几乎逐日不变 = 静态身份, 非动态选股);
    - 动态增量: dyn_minus_static <= 0 表示动态选股没跑赢"前半段静态选票".

  :param tab: evaluate_factors 的输出(含 fdr_q/fdr_sig/sig_ac1/is_static/... 列)
  :returns: 报告文本行列表
  """
  n = len(tab)
  sig = tab.loc[tab['fdr_sig'], [c for c in HONEST_SHOW if c in tab.columns]]
  statics = tab.loc[tab['is_static'], 'factor'].tolist()
  return [
      '[评估诚实性(IS 窗)]',
      f'  BH-FDR(alpha={ev.FDR_ALPHA}): {len(sig)}/{n} 个因子通过多重检验校正 '
      f'(fdr_q < {ev.FDR_ALPHA}); 只看 nw_p 会高估显著个数',
      f'  静态选票(sig_ac1 >= {ev.STATIC_AC1}, 几乎不换仓 = 静态身份): '
      f'{statics or "无"}',
      '  dyn_minus_static <= 0 = 该因子的动态选股没有跑赢"前半段静态选票"',
      '  多重检验后仍显著(IS) 的因子:',
      sig.to_string(index=False) if len(sig) else '    (无)',
      '',
  ]


def build_summary(rep: dict, ds: Dataset, method: str, h: int) -> str:
  """把关键结论拼成一段可读文本(同时写入 OUT_DIR/{pool}/report_summary.txt)."""
  wins = rep.get('windows') or [('IS', cfg.TRADE_START, cfg.IS_END),
                                ('OOS', cfg.OOS_START, None)]
  win_desc = ' / '.join(f'{w} {s or "-"}~{e or "至今"}' for w, s, e in wins)
  tag = '  [已剔除静态票]' if rep.get('excluded_static') else ''
  lines = [
      '=' * 72,
      f'因子研究报告  |  池 {ds.pool}  |  预处理 {method}  |  前向 h={h}{tag}',
      f'窗口: {win_desc}  (权重只在 IS 期估计)',
      '=' * 72,
      '',
      '[数据质量声明]',
      rep['quality'].to_string(index=False),
      '',
      f'[冗余去重] {len(rep["scores_is"])} 个因子 -> 保留 {len(rep["kept"])} 个',
      f'  保留: {rep["kept"]}',
      f'  剔除: {[d["factor"] for d in rep["dropped"]]}',
      '',
  ]
  lines += _honest_section(rep['eval_is'])
  if rep.get('excluded_static'):
    lines += [f'[组合剔除] 静态选票 {rep["excluded_static"]} 已剔出组合'
              ' (仍列在评估表中以便审计)', '']
  lines += [
      '[各方案权重(IS 期估计, sum|w|=1)]',
  ]
  for label, w in rep['weights'].items():
    body = ', '.join(f'{k}={v:+.3f}' for k, v in w.items())
    lines.append(f'  {label:6s} (n={len(w)}): {body}')
  lines += [
      '',
      '[回测汇总] 同一引擎核算; 各窗独立推进(各自空仓起跑, 净值 1.0)',
      rep['backtest'].to_string(index=False),
      '',
      f'[主方案 {rep["main_kind"]} 全交易窗({cfg.TRADE_START} 起)分年收益]',
      rep['yearly'].to_string(),
      '',
  ]
  if rep.get('cost_ladder') is not None:
    lines += [
        f'[成本阶梯] OOS 窗扫单边成本 {list(COST_LADDER_BPS)} bps; '
        'breakeven_bps = 总收益跨 0 的成本(inf 表换手极低/近乎免疫)',
        rep['cost_ladder'].to_string(index=False),
        '',
    ]
  if rep.get('consistency') is not None and len(rep['consistency']):
    lines += [
        '[IS 折内 walk-forward 一致性] 把 IS 期切成折, 每折只用折内训练窗定权 -> 折外测试;',
        '  is_minus_wf = 训练均值 - 折外均值(过拟合缺口, 越接近 0 越说明 IS 强度能延续);',
        '  wf_pos_frac = 折外 RankIC 为正的折占比; wf_icir = 折外均值 / 折外标准差.',
        rep['consistency'].to_string(index=False),
        '',
    ]
  lines += [
      '[产物] ' + '; '.join(f'{k}={v}' for k, v in rep['paths'].items()),
      '=' * 72,
  ]
  if rep.get('curves_win'):
    n_days = len(next(iter(rep['curves_win'].values())))
    lines.insert(-2, f'[注意] {rep["win_name"]} 窗仅 {n_days} 个交易日, '
                     f'单段结果统计意义弱, 只看方向不看幅度')
    lines.insert(-2, '')
  return '\n'.join(lines)


# ============================ 5. 自检 ============================

def verify_report(ds: Dataset, method: str = prep.DEFAULT_PREP, h: int = 1,
                  top_k: int = cfg.TOP_K, min_cs: int = cfg.MIN_CS,
                  seed: int = 42) -> pd.DataFrame:
  """
  报告层标定: 端到端跑一遍精简流水线, 检查关键不变量(只在内存里, 不落盘).

  为什么这样查: 报告层的风险不是"算错某个指标"(那由前五层的自检覆盖), 而是
  "把不完整/不可信的东西当成结论展示出来". 因此检查五件事:
    1. 评估表完整: 覆盖全部因子, 展示列齐全(含诚实性列);
    2. 信号可用: 组合信号有效值占比足够高(不是一张几乎全 NaN 的表);
    3. 引擎标定: oracle RankIC ≈ 1, random ≈ 0(报告引用的口径没被改坏);
    4. 静态检测器: "常量选票"应被判静态(sig_ac1 >= STATIC_AC1 且 dyn_minus_static <= 0),
       确保"静态身份"不会被当成动态 alpha 展示;
    5. 前视对照: honest(IS 定权) OOS 收益 <= lookahead(全样本定权), 方向正确 ——
       若反过来, 说明"报告在偷看未来", 必须排查.

  :returns: DataFrame(check, detail, pass)
  """
  names = fct.list_factors()
  rows = []

  ev_tab = ev.evaluate_factors(ds, names, method, h, top_k, 5, min_cs, 'is')
  missing = [c for c in FACTOR_COLS if c not in ev_tab.columns]
  rows.append({'check': '评估表完整(行/列)',
               'detail': f'{len(ev_tab)} 行 / 缺列 {missing}',
               'pass': len(ev_tab) == len(names) and not missing})

  sigs = cmb.build_signals(ds, method, names, min_cs)
  comp = cmb.combine_signals(sigs, cmb.equal_weights(names), min_cs)
  cov = float(comp.notna().to_numpy().mean())
  rows.append({'check': '组合信号有效值占比', 'detail': f'{cov:.1%}', 'pass': cov > 0.3})

  ve = ev.verify_eval(ds, h, method, min_cs, seed, top_k)
  orc = float(ve.loc[ve['case'].str.startswith('oracle'), 'rank_ic'].iloc[0])
  rnd = float(ve.loc[ve['case'].str.startswith('random'), 'rank_ic'].iloc[0])
  rows.append({'check': 'oracle RankIC ≈ 1', 'detail': f'{orc:.4f}',
               'pass': abs(orc - 1.0) < 1e-6})
  rows.append({'check': 'random RankIC ≈ 0', 'detail': f'{rnd:+.4f}',
               'pass': abs(rnd) < 0.02})

  sta_row = ve[ve['case'].str.startswith('static')].iloc[0]
  sta = float(sta_row['sig_ac1'])
  dms = float(sta_row['dyn_minus_static'])
  rows.append({'check': '静态检测器判定 static',
               'detail': f'sig_ac1={sta:.4f} (需 >= {ev.STATIC_AC1}), '
                         f'dyn_minus_static={dms:+.5f} (需 <= 0)',
               'pass': bool(pd.notna(sta) and sta >= ev.STATIC_AC1
                            and pd.notna(dms) and dms <= 1e-9)})

  _, leak = bt.verify_backtest(ds, method, h, top_k, min_cs, seed)
  oos = dict(zip(leak['case'], leak['total_ret']))
  honest = float(oos.get('honest(IS定权)', np.nan))
  leaky = float(oos.get('lookahead(全样本定权)', np.nan))
  rows.append({'check': '前视对照方向(honest <= lookahead)',
               'detail': f'honest={honest:.3f}, lookahead={leaky:.3f}',
               'pass': bool(pd.notna(honest) and pd.notna(leaky) and honest <= leaky + 1e-9)})
  return pd.DataFrame(rows)


# ============================ 6. CLI ============================

def main():
  ap = argparse.ArgumentParser(description='第6步 报告层: 汇总表 + 图 + 全流程流水线')
  ap.add_argument('--pool', default=cfg.DEFAULT_POOL)
  ap.add_argument('--interval', default=cfg.DEFAULT_INTERVAL)
  ap.add_argument('--method', default=prep.DEFAULT_PREP, choices=prep.PREP_METHODS)
  ap.add_argument('--h', type=int, default=1, help='前向持有天数(评估口径)')
  ap.add_argument('--top-k', type=int, default=cfg.TOP_K)
  ap.add_argument('--exit-rank', type=int, default=cfg.EXIT_RANK)
  ap.add_argument('--sizing', default='equal', choices=bt.SIZINGS)
  ap.add_argument('--max-exposure', type=float, default=1.0)
  ap.add_argument('--per-symbol-cap', type=float, default=0.30)
  ap.add_argument('--cost-bps', type=float, default=cfg.COST_BPS)
  ap.add_argument('--band', type=float, default=0.05)
  ap.add_argument('--q', type=int, default=5, help='分层桶数')
  ap.add_argument('--min-cs', type=int, default=cfg.MIN_CS)
  ap.add_argument('--group', default=None, help='只用某类因子')
  ap.add_argument('--scheme', default='all', choices=('all',) + cmb.COMBO_SCHEMES)
  ap.add_argument('--thresh', type=float, default=cmb.DEDUPE_RHO)
  ap.add_argument('--max-n', type=int, default=8)
  ap.add_argument('--min-gain', type=float, default=cmb.MIN_GAIN)
  ap.add_argument('--min-ic', type=float, default=0.0)
  ap.add_argument('--shrink', type=float, default=cmb.IC_SHRINK,
                  help='IC 权重向"方向对齐等权"收缩的比例(0=纯 IC, 1=纯等权)')
  ap.add_argument('--soft-thresh', type=float, default=0.0,
                  help='IC 权重软阈值(作用在 |RankIC| 上, 0=不生效)')
  ap.add_argument('--wf-folds', type=int, default=4,
                  help='IS 折内 walk-forward 一致性对照的折数')
  ap.add_argument('--top-n', type=int, default=6, help='画图取前 N 个因子')
  ap.add_argument('--start', default=None, help='附加回测窗起点, 如 2026-01-01')
  ap.add_argument('--end', default=None, help='附加回测窗终点(None 表示到数据末)')
  ap.add_argument('--win-name', default='YTD', help='附加窗口标签(汇总表/文件名)')
  ap.add_argument('--no-fig', action='store_true', help='只出表不出图')
  ap.add_argument('--cost-ladder', action='store_true',
                  help='追加成本阶梯扫描(OOS 窗扫 0/10/25/50bps, 出 breakeven)')
  ap.add_argument('--check', action='store_true', help='端到端标定自检')
  ap.add_argument('--exclude-static', action='store_true',
                  help='把静态选票(IS 窗 sig_ac1 >= STATIC_AC1)剔出组合后重跑; '
                       '产物写入 {pool}_excl_static 子目录')
  ap.add_argument('--write-config', nargs='?', const=str(cfg.FACTOR_CONFIG),
                  default=None, metavar='PATH',
                  help='把因子名单+权重写成 JSON 供生产脚本(signal_generator)读取; '
                       '不带值则写默认 %s' % cfg.FACTOR_CONFIG)
  ap.add_argument('--default-scheme', default='greedy',
                  choices=('equal', 'dir', 'ic', 'greedy', 'single'),
                  help='写入 config 的默认方案(供 signal_generator 使用)')
  a = ap.parse_args()

  log = cfg.get_logger('factor.report')
  ds = load_pool(a.pool, a.interval)
  log.info(ds.summary())

  if a.check:
    ck = verify_report(ds, a.method, a.h, a.top_k, a.min_cs)
    print('\n[报告层标定] 端到端精简流水线, 检查关键不变量:')
    print(ck.to_string(index=False))
    print('[结论] 全部通过' if bool(ck['pass'].all())
          else '[结论] 存在未通过项, 需排查')
    return

  p = bt.EngineParams(a.top_k, a.exit_rank, a.sizing, a.max_exposure,
                      a.per_symbol_cap, a.cost_bps, a.band)
  log.info('引擎: ' + p.brief())
  rep = run_report(ds, a.method, a.h, a.top_k, a.q, a.min_cs, a.group, a.scheme,
                   a.thresh, a.max_n, a.min_gain, a.min_ic,
                   shrink=a.shrink, soft_thresh=a.soft_thresh,
                   wf_folds=a.wf_folds, top_n=a.top_n, p=p,
                   figures=not a.no_fig, win_start=a.start, win_end=a.end,
                   win_name=a.win_name, cost_ladder=a.cost_ladder,
                   exclude_static=a.exclude_static)

  text = build_summary(rep, ds, a.method, a.h)
  out_dir = rep['out_dir']
  out_dir.mkdir(parents=True, exist_ok=True)
  summary_path = out_dir / 'report_summary.txt'
  summary_path.write_text(text, encoding='utf-8')
  print('\n' + text)
  print(f'\n[落盘] 汇总文本: {summary_path}')

  if a.write_config:                              # 研究结论 -> 生产接口(因子名单+权重)
    conf = export_config(rep, ds, a.pool, a.interval, a.method, a.min_cs,
                         a.top_k, a.exit_rank, a.default_scheme)
    cpath = write_config(conf, a.write_config)
    log.info(f'[config] 因子配置已写出: {cpath} (kept={len(conf["kept"])}, '
             f'schemes={list(conf["schemes"])}, default={a.default_scheme})')
    print(f'[落盘] 因子配置: {cpath}')


if __name__ == '__main__':
  main()

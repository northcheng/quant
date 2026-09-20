# -*- coding: utf-8 -*-
"""
单标的信号时序 + 汇总信号可视化
================================
对指定池的每个标的(如 SOXL)计算推荐信号的时序序列, 合成汇总信号, 并在价格图上
用「绿/橙/红方框」标注 买入/持有/卖出 三态(参考 bc_technical_analysis.plot_signal
的 trend_magnitude 画法):

  - 方框边框颜色 = 当日状态: rank<=top_k 买入(绿) / rank<=exit_rank 持有(橙) / 其余 卖出(红)
    (rank = 该标的汇总信号在池内的当日截面排名, 与 score_backtest 的 top-k/exit-rank 口径一致)
  - 方框内部填充 = 汇总信号值(已归一化到 [0,1]), 值越大填充越实; 边框始终可见
  - 下方子图: 各分量信号(归一化) + 汇总信号 + 截面排名分区

信号构造直接 import alpha_mining.build_alpha_cands, 与挖掘评估口径完全一致(全历史
构建后再截取显示窗口, 避免 warmup 污染)。

用法:
  cd C:\\Users\\northcheng\\git\\quant\\research
  python signal_timeseries.py --pool etf_3x              # 全池所有标的各出一张图
  python signal_timeseries.py --pool etf_3x --symbols SOXL
  python signal_timeseries.py --pool etf_3x --symbols SOXL,TQQQ --start 2025-01-01
  python signal_timeseries.py --pool etf_3x --symbols SOXL --signals H_ichimoku_alpha,H_trendmag_alpha,C_tmqmom,F_er20
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel                             # noqa: E402
from alpha_mining import POOLS, build_alpha_cands                  # noqa: E402
from bt_core import to_unit01                                      # noqa: E402(实现已迁至 bt_core, 可视化与回测共用)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_SIGNALS = ['H_ichimoku_alpha', 'H_trendmag_alpha', 'C_tmqmom', 'F_er20']
STATE_CN = {'buy': '买入', 'hold': '持有', 'sell': '卖出'}
STATE_COLOR = {'buy': 'green', 'hold': 'orange', 'sell': 'red'}
HERE = os.path.dirname(os.path.abspath(__file__))


def build_symbol_view(pool: str, pkl: str, signals: list):
    """返回: open 宽表, close 宽表, {信号: [0,1] 宽表}, 汇总信号 [0,1] 宽表, 截面排名宽表."""
    _, panel = load_panel(pkl, 'day')
    open_w = panel['Open'].unstack('symbol').sort_index()
    close = panel['Close'].unstack('symbol').sort_index()
    cands = build_alpha_cands(panel)                # 全历史构建(口径与挖掘一致)

    missing = [s for s in signals if s not in cands]
    if missing:
        raise SystemExit(f'候选中不存在信号: {missing}\n可用: {sorted(cands)}')

    comps = {s: to_unit01(s, cands[s]) for s in signals}
    comp_df = pd.concat(comps, axis=1)              # 列为 (signal, symbol) 两级 MultiIndex
    agg = comp_df.T.groupby(level=1).mean().T       # 汇总: 逐 symbol 对分量等权均值 → date×symbol
    agg = agg.clip(0, 1)
    rank = agg.rank(axis=1, ascending=False)        # 1 = 最强
    return open_w, close, comps, agg, rank


def classify_state(rank_row: pd.Series, top_k: int, exit_rank: int) -> pd.Series:
    """三态: rank<=top_k 买入 / rank<=exit_rank 持有 / 其余 卖出."""
    s = pd.Series('sell', index=rank_row.index, dtype=object)
    s[rank_row.notna() & (rank_row <= top_k)] = 'buy'
    s[rank_row.notna() & (rank_row > top_k) & (rank_row <= exit_rank)] = 'hold'
    s[rank_row.isna()] = np.nan
    return s


def plot_symbol(sym: str, pool: str, out_png: str,
                close: pd.DataFrame, comps: dict, agg: pd.DataFrame, rank: pd.DataFrame,
                top_k: int, exit_rank: int, box_size: int):
    win_close = close[sym].dropna()
    start, end = win_close.index[0], win_close.index[-1]
    agg_s = agg[sym].loc[start:end]
    rank_s = rank[sym].loc[start:end]
    close_s = win_close.loc[start:end]
    state = classify_state(rank_s, top_k, exit_rank)

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    ax0, ax1, ax2 = axes
    fig.subplots_adjust(hspace=0.06, right=0.94, top=0.92, bottom=0.06)

    # ---- 行1: 价格 + 三态方框(边框=状态, 填充=汇总信号) ----
    ax0.plot(close_s.index, close_s.values, color='#333333', lw=1.0, alpha=0.9, label='Close')
    for st, color in STATE_COLOR.items():
        idx = state.index[state == st]
        if len(idx) == 0:
            continue
        a = agg_s.reindex(idx).fillna(0).values     # 填充 alpha = 归一化汇总信号
        base = to_rgba(color)
        fc = np.zeros((len(idx), 4))
        fc[:, :3] = base[:3]
        fc[:, 3] = np.clip(a, 0, 1)
        ax0.scatter(idx, close_s.reindex(idx).values, marker='s', s=box_size,
                    facecolor=fc, edgecolor=(base[0], base[1], base[2], 0.9),
                    linewidths=1.3, zorder=5)

    # 右侧现状标注(参考 plot_signal 的写法)
    if not pd.isna(state.iloc[-1]) and not pd.isna(agg_s.iloc[-1]):
        st, av, rv = state.iloc[-1], agg_s.iloc[-1], rank_s.iloc[-1]
        color = STATE_COLOR[st]
        x_txt = end + pd.Timedelta(days=3)
        ax0.annotate(f'{STATE_CN[st]} {av:.2f} | 排名{rv:.0f}',
                     xy=(x_txt, close_s.iloc[-1]), xytext=(x_txt, close_s.iloc[-1]),
                     fontsize=10, color=color, va='center', ha='left')

    handles = [Patch(facecolor='none', edgecolor=c, label=STATE_CN[st])
               for st, c in STATE_COLOR.items()]
    handles.append(Patch(facecolor='grey', edgecolor='grey', alpha=0.4, label='填充亮度=汇总信号'))
    ax0.legend(handles=handles, loc='upper left', fontsize=9, ncol=4, framealpha=0.5)
    ax0.set_title(f'{sym} ({pool}) 推荐信号时序与三态 | 买入≤{top_k} / 持有≤{exit_rank} / 卖出>{exit_rank} '
                  f'(截面排名, 与 score_backtest 口径一致)', fontsize=12)
    ax0.set_ylabel('Close')
    ax0.grid(True, alpha=0.25, lw=0.4)

    # ---- 行2: 分量信号(归一化) + 汇总 ----
    for name, comp in comps.items():
        col = comp[sym].loc[start:end]
        ax1.plot(col.index, col.values, lw=0.9, alpha=0.55, label=name)
    ax1.plot(agg_s.index, agg_s.values, color='black', lw=1.8, label='汇总信号')
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_ylabel('信号值 [0,1]')
    ax1.legend(loc='upper left', fontsize=8, ncol=5, framealpha=0.5)
    ax1.grid(True, alpha=0.25, lw=0.4)

    # ---- 行3: 截面排名(1=最强, 倒置) ----
    rank_max = rank.max(axis=1).max()
    if not np.isfinite(rank_max):
        rank_max = exit_rank + 10
    ax2.axhspan(0.5, top_k + 0.5, color='green', alpha=0.10)
    ax2.axhspan(top_k + 0.5, exit_rank + 0.5, color='orange', alpha=0.10)
    ax2.axhspan(exit_rank + 0.5, rank_max + 2, color='red', alpha=0.08)
    ax2.plot(rank_s.index, rank_s.values, color='#1f77b4', lw=1.0, marker='.',
             markersize=3, alpha=0.8)
    ax2.axhline(top_k, color='green', lw=0.8, ls='--', alpha=0.6)
    ax2.axhline(exit_rank, color='orange', lw=0.8, ls='--', alpha=0.6)
    ax2.invert_yaxis()
    ax2.set_ylabel('池内排名(倒置)')
    ax2.grid(True, alpha=0.25, lw=0.4)
    ax2.set_xlabel('日期')

    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return state, agg_s, rank_s


def main():
    ap = argparse.ArgumentParser(description='单标的信号时序+汇总信号可视化(只读)')
    ap.add_argument('--pool', default='etf_3x', help='池名')
    ap.add_argument('--symbols', default=None, help='逗号分隔标的列表(默认: 全池所有标的)')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl(默认取 alpha_mining.POOLS)')
    ap.add_argument('--signals', default=','.join(DEFAULT_SIGNALS), help='逗号分隔的分量信号')
    ap.add_argument('--start', default=None, help='显示窗口起点(默认最近252个交易日)')
    ap.add_argument('--end', default=None, help='显示窗口终点')
    ap.add_argument('--top-k', type=int, default=5, help='买入排名阈值')
    ap.add_argument('--exit-rank', type=int, default=12, help='卖出排名阈值(持有带上沿)')
    ap.add_argument('--box-size', type=int, default=34, help='方框大小(scatter s)')
    a = ap.parse_args()

    pkl = a.pkl_path or POOLS.get(a.pool)
    if not pkl or not os.path.exists(pkl):
        raise SystemExit(f'pkl 不存在: {pkl}(pool={a.pool}); 用 --pkl-path 指定, '
                         f'建议 research 版: C:\\Users\\northcheng\\quant\\{a.pool}_day_ta_data_research.pkl')
    signals = [s.strip() for s in a.signals.split(',') if s.strip()]

    print(f'加载数据: {pkl}')
    open_all, close_all, comps, agg, rank = build_symbol_view(a.pool, pkl, signals)  # open_all 供回测复用

    # 标的列表: 指定 --symbols 用之, 否则全池
    if a.symbols:
        symbols = [s.strip().upper() for s in a.symbols.split(',') if s.strip()]
    else:
        symbols = list(close_all.columns)
    print(f'标的数: {len(symbols)}' + ('(全池)' if not a.symbols else ''))

    # 显示窗口: 默认最近 252 个交易日
    if a.end:
        close_all = close_all.loc[:a.end]
    if a.start:
        close_all = close_all.loc[a.start:]
    else:
        close_all = close_all.iloc[-252:]

    out_dir = os.path.join(HERE, 'output', f'signal_ts_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'输出目录: {out_dir}\n')

    for sym in symbols:
        if sym not in close_all.columns:
            raise SystemExit(f'{sym} 不在池 {a.pool} 中。可用标的: {sorted(close_all.columns)}')
        out_png = os.path.join(out_dir, f'{a.pool}_{sym}.png')
        state, agg_s, rank_s = plot_symbol(sym, a.pool, out_png, close_all, comps, agg, rank,
                                           a.top_k, a.exit_rank, a.box_size)

        # 控制台摘要
        last_dt = close_all[sym].dropna().index[-1]
        st = state.loc[last_dt]
        av, rv = agg_s.loc[last_dt], rank_s.loc[last_dt]
        print(f'== {sym} @ {last_dt.date()} ==')
        if pd.isna(st):
            print('  状态: 数据不足(NaN)')
        else:
            print(f'  状态: {STATE_CN[st]}  汇总信号: {av:.2f}  池内排名: {rv:.0f}')
            for name, comp in comps.items():
                v = comp[sym].loc[last_dt] if last_dt in comp.index else np.nan
                print(f'    {name:<20s} {v:6.2f}')
            # 最近状态切换
            chg = state[state.ne(state.shift()) & state.notna()]
            recent = chg.tail(6)
            if len(recent) > 1:
                print('  最近状态切换: ' + ', '.join(
                    f'{d.date()} {STATE_CN[s]}' for d, s in recent.items()))
        print(f'  图: {out_png}\n')


if __name__ == '__main__':
    main()

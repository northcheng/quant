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

A 股池(hs300/a_etf_all): 候选走 build_a_combo_cands, 合成口径照搬 signal_bridge 桥
(逐成分 rank(axis=1, pct=True) 截面百分位 → |w| 归一加权累加 → 全缺失 fillna(0.5) →
rank(method='first')), 与桥/回测逐位一致, 因此能如实复现桥的信号(含数据滞后标的的
0.5 中性幻影)。权重默认取池预设, --weights 可覆盖:

  python signal_timeseries.py --pool hs300 --symbols 601059,600346 \\
      --pkl-path C:\\Users\\northcheng\\quant\\hs300_day_ta_data.pkl --end 2026-09-21

回测交易标注(--trades-csv, 配合 a_combo_search/回测脚本落盘的 trades 明细 CSV):
在价格子图叠加每笔交易: 进场▲(绿) / 出场▼(红, 标注收益) + 持有区间底色(盈绿亏红);
未平仓段(exit_reason='open')以橙色▼标注期末盯市值。不指定 --symbols 时默认只画
交易过的标的:

  python signal_timeseries.py --pool hs300 --trades-csv output\\xxx\\trades.csv \\
      --pkl-path C:\\Users\\northcheng\\quant\\hs300_day_ta_data.pkl --start 2026-01-01
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel                             # noqa: E402
from alpha_mining import POOLS, build_alpha_cands                  # noqa: E402
from bt_core import to_unit01                                      # noqa: E402(实现已迁至 bt_core, 可视化与回测共用)
from a_combo_search import build_a_combo_cands                     # noqa: E402(A 股候选, 与桥/回测同源)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_SIGNALS = ['H_ichimoku_alpha', 'H_trendmag_alpha', 'C_tmqmom', 'F_er20']
# A 股两池预设(照搬 signal_bridge.A_POOL_PRESETS; N_range20 两池方向相反:
# hs300 -1 买高振幅 / a_etf_all +1 买低振幅防御)
A_POOL_PRESETS = {
    'hs300':     {'weights': {'N_range20': -1.0, 'D_ma20_dist': -1.0,
                              'C_tmqmom': 0.5},                                'top_k': 5},
    'a_etf_all': {'weights': {'N_range20': 1.0, 'G_oviv20': 0.5},              'top_k': 5},
}
STATE_CN = {'buy': '买入', 'hold': '持有', 'sell': '卖出'}
STATE_COLOR = {'buy': 'green', 'hold': 'orange', 'sell': 'red'}
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_weights(text: str) -> dict:
    """解析 "sig:w,sig:w" 组合权重(与 signal_bridge.parse_weights 同格式, 负权如 N_range20:-1)."""
    out = {}
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        name, _, w = tok.rpartition(':')
        if not name:
            raise SystemExit(f'权重格式错误: "{tok}"(期望 "sig:w", 如 N_range20:-1,C_tmqmom:0.5)')
        out[name.strip()] = float(w)
    return out


def build_symbol_view(pool: str, pkl: str, signals: list, weights: dict = None):
    """返回: open 宽表, close 宽表, {信号: 宽表}, 汇总信号宽表, 截面排名宽表, 权重 dict.

    weights 非空 → A 股池口径(build_a_combo_cands + 桥的 rank_pct 加权合成, 与
    signal_bridge/compute_composite 逐位一致); 否则美股口径(等权均值)。
    """
    _, panel = load_panel(pkl, 'day')
    open_w = panel['Open'].unstack('symbol').sort_index()
    close = panel['Close'].unstack('symbol').sort_index()

    if weights:
        # ---- A 股池: 照搬 signal_bridge 的合成逻辑(逐行一致, 保证图上复现桥信号) ----
        cands = build_a_combo_cands(panel)
        missing = [s for s in weights if s not in cands]
        if missing:
            raise SystemExit(f'候选中不存在信号: {missing}; 可用: {sorted(cands)}')
        total = float(sum(abs(w) for w in weights.values()))
        comp = None
        for s, w in weights.items():
            pct = cands[s].rank(axis=1, pct=True)  # 当日截面百分位, NaN 保留
            part = (w / total) * pct
            comp = part if comp is None else comp.add(part, fill_value=0.0)  # 部分缺失以 0 参与
        comps = {s: cands[s].rank(axis=1, pct=True) for s in weights}       # 分量图 = 截面百分位
        agg = comp.fillna(0.5)                    # 全成分缺失 -> 中性 0.5(桥口径)
        rank = agg.rank(axis=1, ascending=False, method='first')            # 1 = 最强
        return open_w, close, comps, agg, rank, weights

    cands = build_alpha_cands(panel)                # 全历史构建(口径与挖掘一致)
    missing = [s for s in signals if s not in cands]
    if missing:
        raise SystemExit(f'候选中不存在信号: {missing}\n可用: {sorted(cands)}')

    comps = {s: to_unit01(s, cands[s]) for s in signals}
    comp_df = pd.concat(comps, axis=1)              # 列为 (signal, symbol) 两级 MultiIndex
    agg = comp_df.T.groupby(level=1).mean().T       # 汇总: 逐 symbol 对分量等权均值 → date×symbol
    agg = agg.clip(0, 1)
    rank = agg.rank(axis=1, ascending=False)        # 1 = 最强
    return open_w, close, comps, agg, rank, None


def classify_state(rank_row: pd.Series, top_k: int, exit_rank: int) -> pd.Series:
    """三态: rank<=top_k 买入 / rank<=exit_rank 持有 / 其余 卖出."""
    s = pd.Series('sell', index=rank_row.index, dtype=object)
    s[rank_row.notna() & (rank_row <= top_k)] = 'buy'
    s[rank_row.notna() & (rank_row > top_k) & (rank_row <= exit_rank)] = 'hold'
    s[rank_row.isna()] = np.nan
    return s


def plot_symbol(sym: str, pool: str, out_png: str,
                close: pd.DataFrame, comps: dict, agg: pd.DataFrame, rank: pd.DataFrame,
                top_k: int, exit_rank: int, box_size: int, weights: dict = None,
                trades: pd.DataFrame = None):
    # 显示窗口取整个截取后 close 的首尾(而非逐标的 dropna): 数据滞后标的的价格止于中途,
    # 但其信号/排名(含桥的 0.5 中性幻影)仍持续到窗口末, 必须在同一窗口下才能显形
    start, end = close.index[0], close.index[-1]
    agg_s = agg[sym].loc[start:end]
    rank_s = rank[sym].loc[start:end]
    close_s = close[sym].loc[start:end]              # 原始: NaN 处价格线断开(诚实显示数据断更)
    close_ff = close_s.ffill()                       # 方框/标注 y 基准: 断更日沿用最近有效价
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
        ax0.scatter(idx, close_ff.reindex(idx).values, marker='s', s=box_size,
                    facecolor=fc, edgecolor=(base[0], base[1], base[2], 0.9),
                    linewidths=1.3, zorder=5)

    # 右侧现状标注(参考 plot_signal 的写法)
    if not pd.isna(state.iloc[-1]) and not pd.isna(agg_s.iloc[-1]):
        st, av, rv = state.iloc[-1], agg_s.iloc[-1], rank_s.iloc[-1]
        color = STATE_COLOR[st]
        x_txt = end + pd.Timedelta(days=3)
        y_txt = close_ff.iloc[-1] if not pd.isna(close_ff.iloc[-1]) else close_s.dropna().iloc[-1]
        ax0.annotate(f'{STATE_CN[st]} {av:.2f} | 排名{rv:.0f}',
                     xy=(x_txt, y_txt), xytext=(x_txt, y_txt),
                     fontsize=10, color=color, va='center', ha='left')

    handles = [Patch(facecolor='none', edgecolor=c, label=STATE_CN[st])
               for st, c in STATE_COLOR.items()]
    handles.append(Patch(facecolor='grey', edgecolor='grey', alpha=0.4, label='填充亮度=汇总信号'))

    # ---- 行1 附加: 回测交易进出场标注(--trades-csv) ----
    n_trades_sym = 0
    if trades is not None and len(trades):
        for _, tr in trades.iterrows():
            e_dt = pd.Timestamp(tr['entry_exec'])
            if e_dt > end:                       # 整段在显示窗口之后
                continue
            x_dt = pd.Timestamp(tr['exit_exec']) if pd.notna(tr['exit_exec']) else None
            ret_v = float(tr['ret']) if pd.notna(tr['ret']) else np.nan
            is_open = (str(tr.get('exit_reason', '')) == 'open') or (not bool(tr.get('completed', True)))
            e_px = float(tr['entry_px'])
            x_px = float(tr['exit_px']) if pd.notna(tr.get('exit_px')) else None
            x_end = x_dt if x_dt is not None else end    # 未平仓段画到窗口末
            # 持有区间底色: 盈绿亏红, 未平仓灰
            span_c = 'grey' if is_open else ('green' if ret_v > 0 else 'red')
            ax0.axvspan(max(e_dt, start), min(x_end, end), color=span_c, alpha=0.10, zorder=1)
            n_trades_sym += 1
            if start <= e_dt <= end:              # 进场 ▲(标注日期)
                ax0.scatter([e_dt], [e_px], marker='^', s=90, color='limegreen',
                            edgecolor='darkgreen', linewidths=0.8, zorder=8)
                ax0.annotate(f'{e_dt:%m-%d}进', xy=(e_dt, e_px),
                             xytext=(0, 7), textcoords='offset points',
                             fontsize=7, color='green', ha='center', va='bottom', zorder=8)
            if x_dt is not None and start <= x_dt <= end and x_px is not None:
                if is_open:                      # 期末未平仓: 橙色▼ 盯市值
                    ax0.scatter([x_dt], [x_px], marker='v', s=90, facecolor='none',
                                edgecolor='darkorange', linewidths=1.6, zorder=8)
                    ax0.annotate(f'{x_dt:%m-%d}持有中(盯市{ret_v:+.1%})', xy=(x_dt, x_px),
                                 xytext=(0, -10), textcoords='offset points',
                                 fontsize=7, color='darkorange', ha='center', va='top', zorder=8)
                else:                             # 已平仓: 红色▼ 标注收益
                    ax0.scatter([x_dt], [x_px], marker='v', s=90, color='red',
                                edgecolor='darkred', linewidths=0.8, zorder=8)
                    ax0.annotate(f'{x_dt:%m-%d}出 {ret_v:+.1%}', xy=(x_dt, x_px),
                                 xytext=(0, -10), textcoords='offset points',
                                 fontsize=7, color='red', ha='center', va='top', zorder=8)
        if n_trades_sym:
            handles += [Line2D([], [], marker='^', color='limegreen', ls='none',
                               markersize=7, label='进场(开盘)'),
                        Line2D([], [], marker='v', color='red', ls='none',
                               markersize=7, label='出场(开盘,标注收益)'),
                        Line2D([], [], marker='v', color='darkorange', markerfacecolor='none',
                               ls='none', markersize=7, label='持有中(期末盯市)'),
                        Patch(facecolor='green', alpha=0.15, label='持有区间:盈利'),
                        Patch(facecolor='red', alpha=0.15, label='持有区间:亏损')]

    ax0.legend(handles=handles, loc='upper left', fontsize=9, ncol=4, framealpha=0.5)
    if weights:
        wdesc = ','.join(f'{k}:{v:g}' for k, v in sorted(weights.items(), key=lambda kv: -abs(kv[1])))
        ax0.set_title(f'{sym} ({pool}) 桥口径信号时序与三态 | 权重 {wdesc} | 买入≤{top_k} / '
                      f'持有≤{exit_rank} / 卖出>{exit_rank}(截面排名, 与 signal_bridge 一致; '
                      f'价格断更日方框沿用最近有效价)', fontsize=12)
    else:
        ax0.set_title(f'{sym} ({pool}) 推荐信号时序与三态 | 买入≤{top_k} / 持有≤{exit_rank} / 卖出>{exit_rank} '
                      f'(截面排名, 与 score_backtest 口径一致)', fontsize=12)
    ax0.set_ylabel('Close')
    ax0.grid(True, alpha=0.25, lw=0.4)

    # ---- 行2: 分量信号 + 汇总 ----
    for name, comp in comps.items():
        col = comp[sym].loc[start:end]
        label = f'{name}({weights[name]:g})' if weights and name in weights else name
        ax1.plot(col.index, col.values, lw=0.9, alpha=0.55, label=label)
    ax1.plot(agg_s.index, agg_s.values, color='black', lw=1.8, label='汇总信号')
    if weights:                                     # 负权组合的 agg 可为负, 动态范围
        lo = min(agg_s.min(), -0.05) if len(agg_s) else -0.05
        hi = max(agg_s.max(), 1.0) if len(agg_s) else 1.0
        ax1.set_ylim(lo - 0.05, hi + 0.05)
        ax1.set_ylabel('信号值(截面百分位)')
    else:
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
    ap.add_argument('--pool', default='etf_3x', help='池名(美股池或 A 股池 hs300/a_etf_all)')
    ap.add_argument('--symbols', default=None, help='逗号分隔标的列表(默认: 全池所有标的)')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl(默认取 alpha_mining.POOLS; A 股池建议指生产版)')
    ap.add_argument('--signals', default=','.join(DEFAULT_SIGNALS),
                    help='逗号分隔的分量信号(A 股池忽略此参数, 默认取池预设权重)')
    ap.add_argument('--weights', default=None,
                    help='A 股池组合权重 "sig:w,sig:w"(默认取池预设; 如 N_range20:-1,C_tmqmom:0.5)')
    ap.add_argument('--start', default=None, help='显示窗口起点(默认最近252个交易日)')
    ap.add_argument('--end', default=None, help='显示窗口终点')
    ap.add_argument('--top-k', type=int, default=None, help='买入排名阈值(默认: A 股池取预设, 其余 5)')
    ap.add_argument('--exit-rank', type=int, default=12, help='卖出排名阈值(持有带上沿)')
    ap.add_argument('--box-size', type=int, default=34, help='方框大小(scatter s)')
    ap.add_argument('--trades-csv', default=None,
                    help='回测 trades 明细 CSV: 在价格图标注每笔进出场(不给 --symbols 时默认只画交易过的标的)')
    a = ap.parse_args()

    # ---- A 股池: 权重/top_k 默认取池预设(--weights/--top-k 显式覆盖) ----
    is_a_pool = a.pool in A_POOL_PRESETS
    preset = A_POOL_PRESETS.get(a.pool, {})
    if is_a_pool:
        weights = parse_weights(a.weights) if a.weights else dict(preset.get('weights', {}))
        if not weights:
            raise SystemExit(f'A 股池 {a.pool} 无预设权重, 请用 --weights 显式指定(如 N_range20:-1,C_tmqmom:0.5)')
    else:
        weights = None
    top_k = a.top_k if a.top_k is not None else preset.get('top_k', 5)

    pkl = a.pkl_path or POOLS.get(a.pool)
    if not pkl or not os.path.exists(pkl):
        raise SystemExit(f'pkl 不存在: {pkl}(pool={a.pool}); 用 --pkl-path 指定, '
                         f'建议 research 版: {os.path.join(os.path.expanduser("~"), "quant", a.pool + "_day_ta_data_research.pkl")}')
    signals = [s.strip() for s in a.signals.split(',') if s.strip()]

    print(f'加载数据: {pkl}')
    open_all, close_all, comps, agg, rank, weights = build_symbol_view(a.pool, pkl, signals, weights)
    if weights:
        print(f'A 股池口径: {weights} (rank_pct 加权 + 全缺失 fillna 0.5, 与 signal_bridge 一致)')

    # ---- 回测交易标注: 读 trades CSV, 按 symbol 分组 ----
    trades_by_sym = None
    if a.trades_csv:
        if not os.path.exists(a.trades_csv):
            raise SystemExit(f'trades CSV 不存在: {a.trades_csv}')
        tf = pd.read_csv(a.trades_csv, parse_dates=['entry_exec', 'exit_exec'],
                         dtype={'symbol': str})   # 防 000338 被推断成 int 338
        if tf.empty:
            raise SystemExit(f'trades CSV 为空: {a.trades_csv}')
        trades_by_sym = {sym: g.sort_values('entry_exec') for sym, g in tf.groupby('symbol')}
        print(f'交易标注: {a.trades_csv} ({len(tf)} 笔 / {len(trades_by_sym)} 只标的)')

    # 标的列表: 指定 --symbols 用之, 否则全池; 给了 --trades-csv 且未指定时默认只画交易过的
    if a.symbols:
        symbols = [s.strip().upper() for s in a.symbols.split(',') if s.strip()]
    elif trades_by_sym is not None:
        symbols = list(trades_by_sym.keys())
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
        sym_trades = trades_by_sym.get(sym) if trades_by_sym is not None else None
        state, agg_s, rank_s = plot_symbol(sym, a.pool, out_png, close_all, comps, agg, rank,
                                           top_k, a.exit_rank, a.box_size, weights,
                                           trades=sym_trades)

        # 控制台摘要: 取显示窗口最后一日(而非逐标的最后有效日)——A 股池数据滞后标的的
        # 信号(桥的 0.5 中性幻影)正是出现在窗口末, 需按窗口末日取状态才可见
        last_dt = close_all.index[-1]
        st = state.loc[last_dt] if last_dt in state.index else np.nan
        av = agg_s.loc[last_dt] if last_dt in agg_s.index else np.nan
        rv = rank_s.loc[last_dt] if last_dt in rank_s.index else np.nan
        print(f'== {sym} @ {last_dt.date()} ==')
        if sym_trades is not None and len(sym_trades):
            done = sym_trades[sym_trades['completed']] if 'completed' in sym_trades.columns else sym_trades
            tot = float(sym_trades['ret'].sum()) if 'ret' in sym_trades.columns else np.nan
            wr = float((done['ret'] > 0).mean()) if len(done) else np.nan
            print(f'  回测交易: {len(sym_trades)} 笔(平仓 {len(done)}, 胜率 {wr:.0%}),'
                  f' 逐笔收益合计 {tot:+.1%}(单利口径)')
        if pd.isna(st):
            print('  状态: 数据不足(NaN)')
        else:
            print(f'  状态: {STATE_CN[st]}  汇总信号: {av:.2f}  池内排名: {rv:.0f}')
            for name, comp in comps.items():
                v = comp[sym].loc[last_dt] if last_dt in comp.index else np.nan
                w_desc = f'({weights[name]:g})' if weights and name in weights else ''
                print(f'    {name:<20s} {v:6.2f} {w_desc}')
            # 最近状态切换
            chg = state[state.ne(state.shift()) & state.notna()]
            recent = chg.tail(6)
            if len(recent) > 1:
                print('  最近状态切换: ' + ', '.join(
                    f'{d.date()} {STATE_CN[s]}' for d, s in recent.items()))
        print(f'  图: {out_png}\n')


if __name__ == '__main__':
    main()

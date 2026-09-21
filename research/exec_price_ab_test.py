"""A/B 成交口径对比实验: 回测(执行日开盘成交) vs 实盘(执行日收盘成交).

问题来源: run_engine 的时间线是 "信号日 s 收盘决策 -> 执行日 d=s+1 开盘成交,
open-to-open 结算"; 而 automatic_trader 实盘在 D 日 15:45 前市价成交(≈D 收盘价),
桥信号用 D-1 收盘截面. 两种口径的决策截面一致(都是执行日前一交易日收盘),
唯一差异是执行价: A 用 open(d), B 用 close(d).

设计:
  A = open_exec  : run_engine(ow, cw, ...)  原样 —— entry/exit 用执行日开盘, open-to-open
  B = close_exec : run_engine(cw, ow, ...)  交换传参 —— entry/exit 用执行日收盘,
                   close-to-close 结算, 即实盘 15:45 行为的模拟
  两版 composite / gate / 引擎参数逐位一致(列集合统一用 ow.columns, 排名可比).
  B3 引擎未开价格/ATR 止损(EngineParams 默认全 None), 交换传参不影响止损判定路径.

输出: 每池 full/IS/OOS 三窗的 A/B stats 对照 + 换仓执行日日内位移(close/open-1)
分布 + 全池隔夜/日内均值背景 + equity 端到端差异.
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from bt_core import BacktestKit                                  # noqa: E402
from score_backtest import run_config                            # noqa: E402
from combo_search import build_combo_cands, PoolRunner, _slice   # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# B3 三池胜出配置(见 combo_20260920_165604/summary.csv)
POOLS = [
    ('company_300', {'C_tmqmom': 1.0, 'F_er20': 1.0}, 12),
    ('etf_3x', {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2, 'F_obv20': 0.1}, 5),
    ('company_1000', {'G_vr20': 1.0, 'H_ichimoku_alpha': 1.0}, 8),
]
WINDOWS = [('full', '2021-01-01', None),
           ('is', '2021-01-01', '2024-12-31'),
           ('oos', '2025-01-01', None)]
STAT_KEYS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'ann_turnover',
             'n_trades', 'win_rate', 'avg_days', 'n_days']


def wdesc(weights: dict) -> str:
    return ','.join(f'{k}:{v:g}' for k, v in weights.items())


def gap_stats(vals: list) -> dict:
    if not vals:
        return {}
    a = np.array(vals, dtype=float)
    return {'n': len(a), 'mean': a.mean(), 'median': np.median(a),
            'q25': np.quantile(a, 0.25), 'q75': np.quantile(a, 0.75),
            'std': a.std(), 'pct_pos': float((a > 0).mean())}


def fmt_gap(g: dict) -> str:
    if not g:
        return 'n=0'
    return (f"n={g['n']:4d}  mean {g['mean']*100:+.3f}%  median {g['median']*100:+.3f}%  "
            f"q25 {g['q25']*100:+.3f}%  q75 {g['q75']*100:+.3f}%  "
            f"std {g['std']*100:.3f}%  P>0 {g['pct_pos']*100:.0f}%")


def main():
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(BASE, 'output', f'exec_ab_{run_id}')
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    t0 = time.time()

    for pool, w, top_k in POOLS:
        print(f'\n{"=" * 78}\n===== pool={pool}  w={wdesc(w)}  top_k={top_k} =====', flush=True)
        kit = BacktestKit(pool=pool, start=None)
        kit.register(build_combo_cands, label='B3候选')
        runner = PoolRunner(kit)
        comp = runner.fast_composite(w)
        p = kit.engine_params.copy(top_k=top_k)
        ow_all, cw_all = kit.open_wide, kit.close_wide
        assert ow_all.index.equals(cw_all.index), 'open/close index 不一致'
        print(f'  数据 {ow_all.index.min():%Y-%m-%d} ~ {ow_all.index.max():%Y-%m-%d}, '
              f'{ow_all.shape[1]} 标的', flush=True)

        # ---- 背景统计: 全池隔夜 vs 日内(逐日截面均值) ----
        on_ret = (ow_all / cw_all.shift(1) - 1.0)
        id_ret = (cw_all / ow_all - 1.0)
        m_on = on_ret.loc['2021-01-01':].mean(axis=1).mean()
        m_id = id_ret.loc['2021-01-01':].mean(axis=1).mean()
        print(f'  [背景] 全池逐日均值(2021~): 隔夜(close s -> open d) {m_on*100:+.4f}%/日   '
              f'日内(open d -> close d) {m_id*100:+.4f}%/日', flush=True)

        # ---- A/B 三窗口 ----
        for wname, start, end in WINDOWS:
            ow = _slice(ow_all, start, end)
            cw = _slice(cw_all, start, end)
            cols = ow.columns
            c = comp.reindex(index=ow.index, columns=cols)
            g = pd.DataFrame(True, index=ow.index, columns=cols)
            aw = (_slice(kit.atr_wide, start, end)
                  if kit.atr_wide is not None else None)
            resA = run_config(f'{pool}|{wname}|A_open', ow, cw, c, g, p, atr_wide=aw)
            resB = run_config(f'{pool}|{wname}|B_close', cw, ow, c, g, p, atr_wide=aw)
            for tag, r in (('A_open_exec', resA), ('B_close_exec', resB)):
                row = {'pool': pool, 'window': wname, 'exec': tag}
                row.update({k: r['stats'].get(k) for k in STAT_KEYS})
                all_rows.append(row)
            sA, sB = resA['stats'], resB['stats']
            hdr = (f"  [{wname} {start}~{end or 'end'}]  n_days={sA.get('n_days')}\n"
                   f"    {'':14s}{'total_ret':>10s}{'cagr':>8s}{'sharpe':>8s}"
                   f"{'max_dd':>8s}{'ann_to':>8s}{'n_tr':>6s}{'win':>7s}\n"
                   f"    A_open_exec  {sA['total_ret']:>10.3f}{sA['cagr']:>8.3f}"
                   f"{sA['sharpe']:>8.2f}{sA['max_dd']:>8.3f}"
                   f"{sA['ann_turnover']:>8.1f}{sA['n_trades']:>6d}{sA['win_rate']:>7.3f}\n"
                   f"    B_close_exec {sB['total_ret']:>10.3f}{sB['cagr']:>8.3f}"
                   f"{sB['sharpe']:>8.2f}{sB['max_dd']:>8.3f}"
                   f"{sB['ann_turnover']:>8.1f}{sB['n_trades']:>6d}{sB['win_rate']:>7.3f}\n"
                   f"    Δ(B-A)       {sB['total_ret']-sA['total_ret']:>+10.3f}"
                   f"{sB['cagr']-sA['cagr']:>+8.3f}{sB['sharpe']-sA['sharpe']:>+8.2f}"
                   f"{sB['max_dd']-sA['max_dd']:>+8.3f}")
            print(hdr, flush=True)

        # ---- 换仓执行日日内位移(A 版成交事件: B 与 A 成交价差 = close(d)/open(d)-1) ----
        trA = resA['trades']
        if 'completed' in trA.columns:
            trA = trA[trA['completed'] == True]  # noqa: E712
        buy_gap, sell_gap = [], []
        for _, t in trA.iterrows():
            for d, bucket in ((t['entry_exec'], buy_gap), (t['exit_exec'], sell_gap)):
                sym = t['symbol']
                if (d in ow_all.index and sym in ow_all.columns
                        and sym in cw_all.columns):
                    o, c_ = ow_all.at[d, sym], cw_all.at[d, sym]
                    if pd.notna(o) and pd.notna(c_) and o > 0:
                        bucket.append(float(c_) / float(o) - 1.0)
        print(f'  [执行日日内位移 close/open-1]  (= B 成交价相对 A 成交价)\n'
              f'    买入: {fmt_gap(gap_stats(buy_gap))}\n'
              f'    卖出: {fmt_gap(gap_stats(sell_gap))}', flush=True)
        pd.DataFrame({'buy': pd.Series(buy_gap), 'sell': pd.Series(sell_gap)}
                     ).to_csv(os.path.join(out_dir, f'gaps_{pool}.csv'), index=False)

    summ = pd.DataFrame(all_rows)
    summ.to_csv(os.path.join(out_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    print(f'\n[done] {time.time()-t0:.1f}s -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
signal_backtest.py — 汇总信号组合回测: 交易明细 + 回报率
========================================================
对 signal_timeseries.py 可视化的「汇总信号」(4 推荐信号归一化后等权均值)做组合回测。

本文件是 bt_core.BacktestKit 的薄 CLI 预设: 引擎/对照/统计/落盘全部来自 bt_core,
仅保留命令行解析与报告格式; 程序化使用请直接:
  kit = BacktestKit(pool, start=...); kit.register(build_alpha_cands)
  r = kit.run(['H_ichimoku_alpha', 'H_trendmag_alpha', 'C_tmqmom', 'F_er20'],
              mode='unit01', name='agg_signal')

口径(与 bt_core mode='unit01' 一致, 引擎复用 score_backtest.run_engine):
  - 信号构造与可视化/挖掘完全同口径: 全历史构建后再截取交易窗口, 无 warmup 污染
  - 截面排序: 汇总信号降序排名, 滞回进出(进入 rank<=top_k, 退出 rank>exit_rank), 无门(常开)
  - 执行: 信号日 s 收盘决策, 执行日 d=s+1 开盘成交; 持仓结算 open s -> open d; 单边成本
  - 仓位: 已选持仓等权(默认 equal, 可选 tier/linear), 归一到 max_exposure

对照(同一引擎, 公平核算):
  - shuffle: 汇总信号按交易日整体置换(破坏时序因果, 保留截面分布) -> 信号有效性对照
  - buyhold_pool: 全池等权买入持有基线

用法:
  cd C:\\Users\\northcheng\\git\\quant\\research
  python signal_backtest.py --pool company_300 --start 2026-01-01
  python signal_backtest.py --pool etf_3x --start 2026-01-01

输出: research/output/signal_bt_{pool}_{run_id}/
  trades.csv 交易明细 | equity_curve.csv 逐日净值 | positions.csv 持仓 | report.txt
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_mining import POOLS, build_alpha_cands                   # noqa: E402
from signal_timeseries import DEFAULT_SIGNALS                       # noqa: E402
from bt_core import (BacktestKit, EngineParams, monthly_returns,    # noqa: E402
                     yearly_returns)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(
        description='汇总信号组合回测(与 signal_timeseries 可视化同口径): 交易明细+回报率')
    ap.add_argument('--pool', default='etf_3x', help='池名, 默认 etf_3x')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl(默认取 alpha_mining.POOLS)')
    ap.add_argument('--signals', default=','.join(DEFAULT_SIGNALS), help='逗号分隔的分量信号')
    ap.add_argument('--start', default='2026-01-01', help='交易窗口起点(信号仍全历史构建)')
    ap.add_argument('--end', default=None, help='交易窗口终点')
    ap.add_argument('--top-k', type=int, default=5, help='买入排名阈值')
    ap.add_argument('--exit-rank', type=int, default=12, help='退出名次阈值(>该名次退出)')
    ap.add_argument('--sizing', default='equal', choices=['tier', 'linear', 'equal'],
                    help='已选持仓内仓位方式(默认等权)')
    ap.add_argument('--max-exposure', type=float, default=1.0)
    ap.add_argument('--per-symbol-cap', type=float, default=0.30)
    ap.add_argument('--cost-bps', type=float, default=10.0, help='单边成本(bps)')
    ap.add_argument('--band', type=float, default=0.05, help='重平衡带宽')
    ap.add_argument('--skip-baselines', action='store_true', help='不跑 shuffle/buyhold 对照')
    a = ap.parse_args()

    pkl = a.pkl_path or POOLS.get(a.pool)
    if not pkl or not os.path.exists(pkl):
        raise SystemExit(f'pkl 不存在: {pkl}(pool={a.pool})')
    signals = [s.strip() for s in a.signals.split(',') if s.strip()]

    print(f'加载数据: {pkl}')
    p = EngineParams(a.top_k, a.exit_rank, a.sizing, a.max_exposure,
                     a.per_symbol_cap, a.cost_bps, a.band)
    try:
        kit = BacktestKit(pool=a.pool, pkl_path=pkl, start=a.start, end=a.end,
                          engine_params=p)
    except ValueError as e:      # 窗口不足等
        raise SystemExit(str(e))
    kit.register(build_alpha_cands, label='alpha候选')

    print(f'== 回测 == {a.pool}: {kit.open_wide.index[0].date()}~{kit.open_wide.index[-1].date()}, '
          f'{kit.open_wide.shape[1]} 标的, 门=常开, {p.brief()}')
    print(f'== 汇总信号 == 分量: {signals}(归一化后等权均值)')

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output', f'signal_bt_{a.pool}_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    try:
        res = kit.run(signals, mode='unit01', name='agg_signal(汇总信号)')
    except KeyError as e:       # 分量信号未注册
        raise SystemExit(str(e))
    results = [res]
    if not a.skip_baselines:
        results.append(kit.shuffle(res))
        results.append(kit.buyhold())

    print('\n== 绩效汇总 ==')
    kit.compare(results)

    yr, mr = res.yearly(), res.monthly()
    print('\n[分年收益] agg_signal:\n' + yr.to_string())
    print('\n[月度收益] agg_signal:\n' + mr.to_string())

    # 交易明细摘要
    td = res.trades
    done = td[td['completed']] if (len(td) and 'completed' in td.columns) else td
    top5, bot5 = res.top_trades(5), res.bottom_trades(5)
    if len(done):
        print(f'\n[交易明细] 共 {len(td)} 笔(已完成 {len(done)}), 全量见 trades.csv')
        print('盈利最大 5 笔:\n' + top5.to_string(index=False))
        print('亏损最大 5 笔:\n' + bot5.to_string(index=False))

    # 落盘
    res.save(out_dir)

    lines = [f'signal_backtest 报告 | pool={a.pool} | run={run_id}',
             f'pkl: {pkl}',
             f'交易窗口: {kit.open_wide.index[0].date()}~{kit.open_wide.index[-1].date()}, '
             f'{kit.open_wide.shape[1]} 标的',
             f'汇总信号分量: {signals}(归一化后等权均值)',
             '门: 常开(与 signal_timeseries 三态口径一致, 无门)',
             f'引擎: {p.brief()}',
             '', '== 绩效汇总 ==', kit.compare(results, print_table=False).to_string(index=False),
             '', '== agg_signal 分年收益 ==', yr.to_string(),
             '', '== agg_signal 月度收益 ==', mr.to_string()]
    if len(done):
        lines += ['', f'== 交易明细 == n={len(td)}, 已完成 {len(done)}, '
                  f'胜率 {float((done["ret"] > 0).mean()):.1%}, '
                  f'平均收益 {float(done["ret"].mean()):.2%}, '
                  f'平均持仓 {float(done["days"].mean()):.1f} 日']
        lines += ['', '盈利最大 5 笔:', top5.to_string(index=False),
                  '', '亏损最大 5 笔:', bot5.to_string(index=False),
                  '', '全量明细: trades.csv']
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')
    print('  ' + ', '.join(sorted(os.listdir(out_dir))))


if __name__ == '__main__':
    main()

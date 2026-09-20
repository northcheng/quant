# -*- coding: utf-8 -*-
"""stop_loss_eval.py — 止盈止损消融实验

验证问题: 在既有 rank 进出场之上叠加价格止盈止损, 回测收益是否更好?
两轮变体: 固定百分比(SL/TP/TRAIL) + ATR 动态(SLA 固定 ATR 止损 / CH 吊灯移动止损, 波动自适应).
口径与生产(signal_bridge)完全一致:
  company_300 池 + 四信号等权 unit01 汇总 + gate 常开 +
  K=5 / exit_rank=12 / tier / cap 30% / 10bps 单边 / band 0.05.

触发时间线(防前视): 信号日 s 收盘价较 entry_px 判定触发 -> 执行日 d=s+1 开盘卖出,
与 rank 退出同一时间线; 止损后同一信号日不会重买(需等下一信号日 rank<=K 才可能重进).

用法:
  python stop_loss_eval.py                       # 默认矩阵 + 全部汇总
  python stop_loss_eval.py --start 2023-01-01    # 换窗口
  python stop_loss_eval.py --pool etf_3x         # 换池
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_core import BacktestKit, EngineParams          # noqa: E402
from alpha_mining import build_alpha_cands             # noqa: E402

SPEC = {'H_ichimoku_alpha': 0.25, 'H_trendmag_alpha': 0.25,
        'C_tmqmom': 0.25, 'F_er20': 0.25}

# (名称, EngineParams 覆盖项): baseline 为生产现状, 其余为价格止盈止损变体
# 第一轮: 固定百分比 SL/TP/TRAIL; 第二轮: ATR 动态(波动自适应) SLA/CH(吊灯)
VARIANTS = [
    ('baseline', {}),
    ('SL-5%', {'stop_loss': 0.05}),
    ('SL-8%', {'stop_loss': 0.08}),
    ('SL-10%', {'stop_loss': 0.10}),
    ('TP-15%', {'take_profit': 0.15}),
    ('SL8+TP15', {'stop_loss': 0.08, 'take_profit': 0.15}),
    ('SL5+TP10', {'stop_loss': 0.05, 'take_profit': 0.10}),
    ('TRAIL-15%', {'trail_stop': 0.15}),
    ('TRAIL-20%', {'trail_stop': 0.20}),
    ('TRAIL-25%', {'trail_stop': 0.25}),
    ('ATR-SL1.5', {'stop_atr': 1.5}),
    ('ATR-SL2.0', {'stop_atr': 2.0}),
    ('ATR-SL2.5', {'stop_atr': 2.5}),
    ('CH-3', {'trail_atr': 3.0}),
    ('CH-4', {'trail_atr': 4.0}),
    ('TP15+CH3', {'take_profit': 0.15, 'trail_atr': 3.0}),
    ('TP15+ATR2', {'take_profit': 0.15, 'stop_atr': 2.0}),
    ('TP15+ATR1.5', {'take_profit': 0.15, 'stop_atr': 1.5}),
]


def main():
    ap = argparse.ArgumentParser(description='止盈止损消融实验(口径与 signal_bridge 一致)')
    ap.add_argument('--pool', default='company_300', help='池名, 默认 company_300')
    ap.add_argument('--start', default='2021-01-01', help='回测起始(交易窗口), 默认 2021-01-01')
    ap.add_argument('--end', default=None, help='回测结束日, 默认最新')
    ap.add_argument('--cost-bps', type=float, default=10.0, help='单边成本(bps), 默认 10')
    args = ap.parse_args()

    kit = BacktestKit(pool=args.pool, start=args.start, end=args.end)
    kit.register(build_alpha_cands)
    base_p = EngineParams(cost_bps=args.cost_bps)

    results = [kit.run(SPEC, mode='unit01', name=name,
                       engine_params=base_p.copy(**kw))
               for name, kw in VARIANTS]

    print(f'\n=== 止盈止损消融: {args.pool} @ {args.start}~{args.end or "最新"} ===')
    print(f'参数: {base_p.brief()}  信号: 四信号等权 unit01(agg)')
    t = kit.compare(results + [kit.buyhold()], print_table=False)
    print(t.to_string(index=False))

    # 出场原因分布: 各配置下止损/止盈/移动止损出场的笔数与平均收益
    rows = []
    for r in results:
        td = r.trades
        if not len(td) or 'exit_reason' not in td.columns:
            continue
        done = td[td['completed']]
        for reason, sub in done.groupby('exit_reason'):
            rows.append({'config': r.name, 'reason': reason, 'n': len(sub),
                         'avg_ret': round(float(sub['ret'].mean()), 4),
                         'avg_days': round(float(sub['days'].mean()), 1),
                         'worst': round(float(sub['ret'].min()), 4)})
    if rows:
        print('\n--- 出场原因分布(已完成交易) ---')
        print(pd.DataFrame(rows).to_string(index=False))

    # 年度收益对比(看止盈止损在牛熊年份的差异化贡献)
    yr = pd.DataFrame({r.name: r.yearly() for r in results})
    if len(yr):
        print('\n--- 年度收益对比 ---')
        print(yr.round(3).to_string())

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'output', 'stop_loss_eval')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'summary_{args.pool}_{args.start[:4]}.csv')
    t.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'\n汇总表已存: {out_csv}')


if __name__ == '__main__':
    main()

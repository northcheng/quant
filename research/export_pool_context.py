# -*- coding: utf-8 -*-
"""
export_pool_context.py — 导出池上下文文件(供 bc_technical_analysis.add_factor_mining_features 读取)

定位: etf_3x 黄金组合因子(F_mom121 / F_er20 / F_idiovol60 / F_obv20 + composite)在
      calculate_ta_signal 的单标的环境下需要两类池信息:
        1. 等权池收益 pool_ret   — F_idiovol60 的 β 剥离(resid = ret − β60×pool_ret)
        2. 全池各标的当日因子值  — F_composite 的当日截面 rank pct(score_backtest.compute_composite 口径)
      本脚本预先从研究 pkl 计算并导出为 csv, 一次生成多处使用, 避免每个标的重算全池。

输出: research/data/{pool}_context.csv, 长表列:
  date, symbol, pool_ret, F_mom121, F_er20, F_obv20, F_idiovol60
  其中 pool_ret 为当日全池等权日收益(与 factor_mining.build_mined F 段同口径);
  F_idiovol60 已按池口径完成 β 剥离(单标的集成时用于截面 rank 对照)。

因子公式与 research/factor_mining.py build_mined 逐行一致(仅 t 及之前数据, 无前视)。

用法:
  python research/export_pool_context.py --pool etf_3x --pkl-path research/data/etf_3x_day_ta_data.pkl
  python research/export_pool_context.py --pool etf_3x   (走 research/data 默认路径)

数据更新后重跑一次即可(因子为滚动窗口计算, 历史值不变, 只追加新日期)。
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel
from factor_mining import build_mined

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 黄金组合涉及的 4 个因子(与 build_mined 输出键一致)
CONTEXT_FACTORS = ['F_mom121', 'F_er20', 'F_obv20', 'F_idiovol60']


def export_pool_context(pkl_path: str, interval: str = 'day', out_path: str = None) -> str:
    """从研究 pkl 计算全池 4 因子 + 等权池收益, 导出池上下文 csv.

    :param pkl_path: 研究 pkl 路径({symbol}_{interval} -> df)
    :param interval: 数据频率, 'day' 为默认
    :param out_path: 输出 csv 路径, 默认 research/data/{pool}_context.csv(按 pkl 名推断池名)
    :returns: 实际写出的 csv 路径
    """
    raw, panel = load_panel(pkl_path, interval)
    close = panel['Close'].unstack('symbol').sort_index()
    ret1 = close.pct_change()

    # 等权池收益: 与 build_mined F 段完全同口径(当日池内有效列等权平均)
    pool_ret = ret1.mean(axis=1)

    # 全池因子(池口径, F_idiovol60 含 β 剥离)
    mined = build_mined(panel)
    factors = {f: mined[f].reindex(index=close.index, columns=close.columns) for f in CONTEXT_FACTORS}

    # 堆成长表: date, symbol, pool_ret, F_xxx
    stacked = pd.DataFrame({f: factors[f].stack() for f in CONTEXT_FACTORS})
    stacked.index.names = ['date', 'symbol']
    long = stacked.reset_index()
    long['pool_ret'] = long['date'].map(pool_ret)
    long = long.sort_values(['date', 'symbol'])

    if out_path is None:
        base = os.path.splitext(os.path.basename(pkl_path))[0]
        pool = base.replace(f'_{interval}_ta_data', '')
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f'{pool}_context.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    long.to_csv(out_path, index=False, encoding='utf-8')

    dts = long['date']
    print(f'== 池上下文导出 == {out_path}')
    print(f'  标的 {long["symbol"].nunique()} 个, {dts.min()} ~ {dts.max()}, {len(long)} 行')
    print(f'  因子: {CONTEXT_FACTORS} + pool_ret')
    return out_path


def main():
    ap = argparse.ArgumentParser(description='导出池上下文 csv(全池 4 因子 + 等权池收益)')
    ap.add_argument('--pool', default='etf_3x', help='池名, 默认 etf_3x')
    ap.add_argument('--interval', default='day', help='数据频率, 默认 day')
    ap.add_argument('--pkl-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                    help='pkl 所在目录, 默认 research/data')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl 路径(优先)')
    ap.add_argument('--out-path', default=None, help='输出 csv 路径, 默认 research/data/{pool}_context.csv')
    args = ap.parse_args()

    pkl_path = args.pkl_path or os.path.join(args.pkl_dir, f'{args.pool}_{args.interval}_ta_data.pkl')
    if not os.path.exists(pkl_path):
        print(f'[ERROR] 找不到 pkl: {pkl_path}')
        sys.exit(1)
    export_pool_context(pkl_path, interval=args.interval, out_path=args.out_path)


if __name__ == '__main__':
    main()

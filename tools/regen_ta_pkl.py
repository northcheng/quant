# -*- coding: utf-8 -*-
"""
重算 etf_3x 的 TA 数据(用修复后的指标代码), 输出到**新文件**, 不碰原始 pkl.

设计要点(安全):
  * 数据源用本地 stock_data/*.csv(离线, 不联网 -> 不引入"今天的数据"这个混淆变量);
  * 每算完一个标的就落盘一次(断点可续), 中断不丢进度;
  * 输出到 <pool>_day_ta_data_regen.pkl, 由调用方比较后再决定是否替换。

用法:
  python quant/tools/regen_ta_pkl.py --pool etf_3x --interval day
"""
import argparse
import datetime
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant import bc_technical_analysis as ta
from quant import bc_util as util

HOME = Path.home()
QUANT = HOME / 'quant'
STOCK = QUANT / 'stock_data'


def load_pool_symbols(pool: str) -> list:
    import json
    with open(REPO / 'quant' / 'selected_sec_list.json', encoding='utf-8') as f:
        return json.load(f)[pool]


def read_csv_raw(symbol: str) -> pd.DataFrame:
    """读本地原始 CSV -> DatetimeIndex 的 OHLCV 表(不预处理的原始复权数据)."""
    p = STOCK / f'{symbol}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding='utf8')
    if 'Date' not in df.columns:
        return None
    return util.df_2_timeseries(df=df, time_col='Date')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='etf_3x')
    ap.add_argument('--interval', default='day')
    ap.add_argument('--start', default=None, help='计算起点(None=用全部本地历史)')
    ap.add_argument('--end', default=None)
    ap.add_argument('--out', default=None, help='输出 pkl 路径(None=quant/<pool>_<interval>_ta_data_regen.pkl)')
    ap.add_argument('--limit', type=int, default=0, help='只算前 N 个标的(0=全部, 用于试跑计时)')
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    symbols = load_pool_symbols(a.pool)
    if a.limit:
        symbols = symbols[:a.limit]
    out = Path(a.out) if a.out else QUANT / f'{a.pool}_{a.interval}_ta_data_regen.pkl'
    keys = [f'{s}_{a.interval}' for s in symbols]

    # 断点续算: 已存在的结果直接复用
    data = {}
    if out.exists():
        try:
            with open(out, 'rb') as f:
                data = pickle.load(f)
            print(f'[resume] 已加载 {len(data)} 个标的的既有结果')
        except Exception as e:
            print(f'[resume] 读取失败({e}), 从头开始')
            data = {}

    print(f'池={a.pool}  标的数={len(symbols)}  输出={out}')
    t_all = time.time()
    for i, (sym, key) in enumerate(zip(symbols, keys), 1):
        if key in data and data[key] is not None and len(data[key]) > 0:
            print(f'[{i:3d}/{len(symbols)}] {sym:8s} 跳过(已有)')
            continue
        t0 = time.time()
        try:
            raw = read_csv_raw(sym)
            if raw is None or len(raw) == 0:
                print(f'[{i:3d}/{len(symbols)}] {sym:8s} 无本地数据, 跳过')
                continue
            feat = ta.calculate_ta_feature(df=raw, symbol=sym,
                                           start_date=a.start, end_date=a.end)
            if feat is None or len(feat) == 0:
                print(f'[{i:3d}/{len(symbols)}] {sym:8s} TA 计算返回空')
                continue
            sig = ta.calculate_ta_signal(df=feat, interval=a.interval)
            data[key] = sig
            dt = time.time() - t0
            print(f'[{i:3d}/{len(symbols)}] {sym:8s} OK  行={len(sig):5d} 列={len(sig.columns):4d} '
                  f'用时={dt:6.1f}s  {sig.index.min():%Y-%m-%d}~{sig.index.max():%Y-%m-%d}')
        except Exception as e:
            print(f'[{i:3d}/{len(symbols)}] {sym:8s} EXC {type(e).__name__}: {e}')
            continue
        # 每算完一个就落盘(断点可续)
        tmp = out.with_suffix('.pkl.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out)

    print(f'\n完成: {len(data)} 个标的, 总用时 {time.time()-t_all:.1f}s -> {out}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
隔离实验: 只加 T_dawn_evt + T_band_evt(而非全部 8 个 T_), 看 OOS 是否仍退化。

做法: 打补丁把 fct.list_factors 的结果收窄到 [baseline + 指定新因子],
走**完全相同的** run_report 流水线。不写配置。

用法: python quant/tools/abo_factor_subset.py --pool etf_3x --add T_dawn_evt T_band_evt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant.factor import factor as fct
from quant.factor import report as rep
from quant.factor import backtest as bt
from quant.factor import config as cfg
from quant.factor.data import load_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='etf_3x')
    ap.add_argument('--pkl', default=None)
    ap.add_argument('--add', nargs='*', default=[], help='要额外加入的新因子名')
    ap.add_argument('--drop', nargs='*', default=[], help='要剔除的因子名')
    ap.add_argument('--no-fig', action='store_true', default=True)
    a = ap.parse_args()

    all_names = fct.list_factors()
    T_all = [n for n in all_names if n.startswith('T_')]
    # baseline = 除 T_ 族以外的一切(即"不含 T_"的原始 70 个)
    base = [n for n in all_names if not n.startswith('T_')]
    keep = [n for n in base if n not in set(a.drop)] + [n for n in a.add]
    keep = [n for n in keep if n in all_names]
    missing = [n for n in a.add if n not in all_names]
    if missing:
        raise SystemExit(f'未知因子: {missing}')

    print(f'池={a.pool}')
    print(f'  注册总数        : {len(all_names)}')
    print(f'  baseline(无T_)  : {len(base)}')
    print(f'  T_ 族全部       : {len(T_all)} {T_all}')
    print(f'  本次保留        : {len(keep)}  (baseline {len(base)} + 新增 {a.add} - 剔除 {a.drop})')
    assert not any(n.startswith("T_") for n in keep if n not in a.add), 'baseline 里混入了 T_'

    _orig = fct.list_factors

    def patched(group=None, _keep=tuple(keep)):
        if group is None:
            return [n for n in _orig(None) if n in set(_keep)]
        return _orig(group)

    fct.list_factors = patched
    rep.fct.list_factors = patched          # report 模块里可能是独立引用

    pkl = a.pkl or str(Path.home() / 'quant' / f'{a.pool}_day_ta_data_research.pkl')
    ds = load_pool(a.pool, 'day', pkl_path=pkl)
    print(ds.summary())

    p = bt.EngineParams(cfg.TOP_K, cfg.EXIT_RANK, 'equal', 1.0, 0.30, cfg.COST_BPS, 0.05)
    r = rep.run_report(ds, figures=not a.no_fig, cost_ladder=True, p=p)

    bs = r.get('backtest') if isinstance(r, dict) else None
    if bs is None:
        print('\n[可用键]', list(r.keys())); return
    print()
    print('=' * 108)
    print(f'回测结果 (保留 {len(keep)} 个因子)')
    print('=' * 108)
    print(bs.to_string(index=False))
    kept = r.get('kept') or []
    print()
    print(f'去重后 kept: {len(kept)} 个; 其中 T_ : {[k for k in kept if k.startswith("T_")]}')
    w = (r.get('weights') or {})
    for s in ('equal', 'dir', 'ic', 'greedy'):
        ws = w.get(s) or {}
        if ws:
            print(f'  方案 {s:7s}: {len(ws):3d} 个, T_ -> {[k for k in ws if k.startswith("T_")]}')
    cl = r.get('cost_ladder')
    if cl is not None:
        print()
        print('成本阶梯:'); print(cl.to_string(index=False))


if __name__ == '__main__':
    main()

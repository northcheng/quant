# -*- coding: utf-8 -*-
"""横向比较 factor_config.json 里各加权方案(scheme)的冻结权重回测表现。

逻辑复用 tools/bt_prod_weights.py: 只读 config 冻结权重, 同一引擎/同一窗口,
对每个池的每个 scheme(含 buyhold 基准)在 IS/OOS/recent 三窗回测。

产出: factor/output/scheme_bench.csv (长表) + 终端 sharpe/max_dd pivot。
用法: python quant/tools/scheme_bench.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import load_pool

CFG = REPO / 'quant' / 'factor_config.json'
OUT = REPO / 'quant' / 'factor' / 'output'
pd.set_option('display.width', 260)

POOLS = ['etf_3x', 'hs300', 'a_etf_all', 'company_300', 'company_1000']


def _win(df, s, e):
    out = df
    if s is not None:
        out = out.loc[out.index >= pd.Timestamp(s)]
    if e is not None:
        out = out.loc[out.index <= pd.Timestamp(e)]
    return out


def bench_pool(pool, conf):
    params = conf.get('params') or {}
    schemes = conf.get('schemes') or {}
    pkl = Path.home() / 'quant' / f'{pool}_day_ta_data_research.pkl'
    if not pkl.exists():
        print(f'[skip] {pool}: pkl 不存在 {pkl}')
        return []
    ds = load_pool(pool, 'day', pkl_path=str(pkl))
    need = sorted({n for w in schemes.values() for n in w})
    sigs = {}
    for n in need:
        if n not in fct.list_factors():
            print(f'  {pool}/{n}: 不在注册表, 跳过')
            continue
        try:
            sigs[n] = prep.prepare(fct.get_factor(n).fn(ds), method=params.get('prep', 'rank'))
        except Exception as e:
            print(f'  {pool}/{n}: 计算失败 {type(e).__name__}: {e}')

    ep = bt.EngineParams(int(params.get('top_k', cfg.TOP_K)),
                         int(params.get('exit_rank', cfg.EXIT_RANK)),
                         'equal', 1.0, 0.30, cfg.COST_BPS, 0.05)
    w = cfg.resolve_windows(pool)
    wins = [('IS', w['is'][0], w['is'][1]), ('OOS', w['oos'][0], w['oos'][1])]
    rs, re_ = w['recent']
    if rs not in (None, '9999-12-31'):
        wins.append(('recent', rs, re_))
    open_w, close_w = ds.wide('open'), ds.wide('close')

    rows = []
    for name, wt in sorted(schemes.items()):
        use = {k: float(v) for k, v in wt.items() if k in sigs}
        if not use:
            print(f'  {pool}/{name}: 无可用因子, 跳过')
            continue
        comp = cmb.combine_signals(sigs, use, min_cs=int(params.get('min_cs', cfg.MIN_CS)))
        for wn, ws, we in wins:
            res = bt.run_engine(_win(open_w, ws, we), _win(close_w, ws, we), _win(comp, ws, we), ep)
            st = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
            rows.append({'pool': pool, 'scheme': name, 'window': wn, 'n_fac': len(use),
                         **{k: st.get(k) for k in bt.STAT_KEYS}})
    for wn, ws, we in wins:
        res = bt.buyhold(_win(open_w, ws, we), _win(close_w, ws, we))
        st = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
        rows.append({'pool': pool, 'scheme': 'buyhold', 'window': wn, 'n_fac': len(open_w.columns),
                     **{k: st.get(k) for k in bt.STAT_KEYS}})
    return rows


def main():
    conf_all = json.load(open(CFG, encoding='utf-8'))['pools']
    all_rows = []
    for p in POOLS:
        if p not in conf_all:
            print(f'[skip] {p} 不在 config')
            continue
        ds_name = (conf_all[p].get('params') or {}).get('default_scheme')
        print(f'>>> {p}  default={ds_name}')
        all_rows += bench_pool(p, conf_all[p])

    df = pd.DataFrame(all_rows)
    out_csv = OUT / 'scheme_bench.csv'
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'\nWROTE {out_csv}  ({len(df)} 行)')

    for wn in ['IS', 'OOS', 'recent']:
        sub = df[df['window'] == wn]
        if sub.empty:
            continue
        print('\n' + '=' * 100)
        print(f'[{wn}] sharpe')
        print(sub.pivot_table(index='pool', columns='scheme', values='sharpe',
                              aggfunc='first').round(3).to_string())
        print(f'[{wn}] max_dd')
        print(sub.pivot_table(index='pool', columns='scheme', values='max_dd',
                              aggfunc='first').round(3).to_string())
        print(f'[{wn}] total_ret')
        print(sub.pivot_table(index='pool', columns='scheme', values='total_ret',
                              aggfunc='first').round(3).to_string())


if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
用 factor_config.json 里**冻结的生产权重**跑回测, 与 factor.output 里
"流水线现算权重"的同窗口结果对比。

关键: 生产用的 cd_ic 权重是历史上 wf_step4 导出的, **不在 run_report 的重算方案里**;
所以此前所有 backtest_summary 都没有"真实生产权重"这一行。本脚本补上。

用法: python quant/tools/bt_prod_weights.py --pool etf_3x
"""
import argparse
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
from quant.factor import evaluate as ev
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import load_pool

CFG = REPO / 'quant' / 'factor_config.json'
OUT = REPO / 'quant' / 'factor' / 'output'
pd.set_option('display.width', 220)


def win(df, s, e):
    out = df
    if s is not None:
        out = out.loc[out.index >= pd.Timestamp(s)]
    if e is not None:
        out = out.loc[out.index <= pd.Timestamp(e)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='etf_3x')
    ap.add_argument('--pkl', default=None)
    a = ap.parse_args()

    conf = json.load(open(CFG, encoding='utf-8'))['pools'][a.pool]
    params = conf.get('params') or {}
    ds_name = params.get('default_scheme')
    schemes = conf.get('schemes') or {}
    pkl = a.pkl or str(Path.home() / 'quant' / f'{a.pool}_day_ta_data_research.pkl')

    print(f'池={a.pool}  默认方案={ds_name}')
    print(f'  params       : {json.dumps(params, ensure_ascii=False, sort_keys=True)}')
    print(f'  方案因子数   : ' + ', '.join(f'{k}={len(v)}' for k, v in sorted(schemes.items())))
    ds = load_pool(a.pool, 'day', pkl_path=pkl)
    print(ds.summary())

    # 需要哪些因子 -> 只算这些(省时间)
    need = sorted({n for w in schemes.values() for n in w} | {'mom_12_1'})
    miss = [n for n in need if n not in fct.list_factors()]
    if miss:
        print(f'  !! 配置里有 {len(miss)} 个因子不在注册表中, 将被跳过: {miss[:8]}')
    sigs = {}
    for n in need:
        if n not in fct.list_factors():
            continue
        try:
            sigs[n] = prep.prepare(fct.get_factor(n).fn(ds), method=params.get('prep', 'rank'))
        except Exception as e:
            print(f'  {n}: 计算失败 {type(e).__name__}: {e}')
    print(f'  已准备 {len(sigs)}/{len(need)} 个因子的信号')

    # 引擎参数: 用配置里的 top_k/exit_rank, 其余取 config 默认(与 report 一致)
    ep = bt.EngineParams(int(params.get('top_k', cfg.TOP_K)),
                         int(params.get('exit_rank', cfg.EXIT_RANK)),
                         'equal', 1.0, 0.30, cfg.COST_BPS, 0.05)
    print(f'  引擎: {ep.brief() if hasattr(ep, "brief") else ep.__dict__}')

    w = cfg.resolve_windows(a.pool)
    wins = [('IS', w['is'][0], w['is'][1]), ('OOS', w['oos'][0], w['oos'][1])]
    rs, re_ = w['recent']
    if rs not in (None, '9999-12-31'):
        wins.append(('recent', rs, re_))

    open_w, close_w = ds.wide('open'), ds.wide('close')
    rows = []
    for name, wt in schemes.items():
        use = {k: float(v) for k, v in wt.items() if k in sigs}
        if not use:
            print(f'  方案 {name}: 无可用因子, 跳过'); continue
        comp = cmb.combine_signals(sigs, use, min_cs=int(params.get('min_cs', cfg.MIN_CS)))
        for wn, ws, we in wins:
            res = bt.run_engine(win(open_w, ws, we), win(close_w, ws, we), win(comp, ws, we), ep)
            s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
            rows.append({'kind': f'PROD:{name}', 'window': wn, 'n_fac': len(use),
                         **{k: s.get(k) for k in bt.STAT_KEYS}})
    for wn, ws, we in wins:
        res = bt.buyhold(win(open_w, ws, we), win(close_w, ws, we))
        s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
        rows.append({'kind': 'buyhold', 'window': wn, 'n_fac': len(open_w.columns),
                     **{k: s.get(k) for k in bt.STAT_KEYS}})
    mine = pd.DataFrame(rows)

    print()
    print('=' * 118)
    print('A) 用**生产冻结权重**回测(本脚本现算)')
    print('=' * 118)
    key = ['kind', 'window', 'n_fac', 'total_ret', 'sharpe', 'max_dd', 'ann_turnover', 'n_trades']
    print(mine[key].round(4).to_string(index=False))

    # 读流水线现算的结果做并排对比
    pth = OUT / a.pool / 'backtest_summary.csv'
    if pth.exists():
        live = pd.read_csv(pth)
        live = live[live['kind'].isin(['equal', 'dir', 'ic', 'greedy', 'buyhold',
                                       'shuffle(equal)'])]
        print()
        print('=' * 118)
        print('B) 并排对比: 生产冻结权重 vs 流水线现算(同一份数据/同一引擎/同一窗口)')
        print('=' * 118)
        piv = pd.concat([mine, live]).pivot_table(
            index='kind', columns='window', values='sharpe', aggfunc='first')
        print('[OOS sharpe]')
        print(piv.round(3).to_string())
        print()
        piv2 = pd.concat([mine, live]).pivot_table(
            index='kind', columns='window', values='total_ret', aggfunc='first')
        print('[OOS total_ret]')
        print(piv2.round(3).to_string())
        # 生产默认方案 vs 现算 ic / equal
        prod_kind = f'PROD:{ds_name}'
        print()
        print('=' * 118)
        print(f'C) 焦点: 生产默认方案 {prod_kind}  对比  现算 equal / ic / buyhold')
        print('=' * 118)
        focus = pd.concat([mine, live])
        focus = focus[focus['kind'].isin([prod_kind, 'equal', 'ic', 'buyhold'])]
        print(focus[['kind', 'window', 'n_fac', 'total_ret', 'sharpe', 'max_dd',
                     'ann_turnover']].round(4).to_string(index=False))
    else:
        print(f'\n(未找到 {pth}, 跳过对比)')


if __name__ == '__main__':
    main()

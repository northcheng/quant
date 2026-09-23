# -*- coding: utf-8 -*-
"""临时脚本: 等权持仓(sizing='equal') vs 现有分档持仓(sizing='tier') 对比.

除 sizing 外, 数据/信号/窗口/成本/其余引擎参数完全一致(与 _tmp_export_ts.py 同口径).
产出: research/output/pool_sizing_cmp.csv + 控制台表格
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from bt_core import BacktestKit                              # noqa: E402
from combo_search import _slice, build_combo_cands           # noqa: E402
from alpha_mining import build_alpha_cands                   # noqa: E402
from a_combo_search import build_a_combo_cands               # noqa: E402
from score_backtest import run_config                        # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

PRESETS = [
    ('company_300', {'C_tmqmom': 1.0, 'F_er20': 1.0}, 12, 'us'),
    ('company_1000', {'G_vr20': 1.0, 'H_ichimoku_alpha': 1.0}, 8, 'us'),
    ('etf_3x', {'F_mom121': 0.5, 'F_er20': 0.3, 'N_idiovol60': 0.2, 'F_obv20': 0.1}, 5, 'us'),
    ('hs300', {'N_range20': -1.0, 'D_ma20_dist': -1.0, 'C_tmqmom': 0.5}, 5, 'a'),
    ('a_etf_all', {'N_range20': 1.0, 'G_oviv20': 0.5}, 5, 'a'),
]
START, END = '2021-01-01', None
YTD_START = pd.Timestamp('2026-01-01')


def window_stats(equity: pd.Series, since: pd.Timestamp) -> dict:
    """自 since 起的窗口收益(净值起点归一)."""
    eq = equity[equity.index >= since]
    if len(eq) < 2:
        return {'ret': np.nan, 'sharpe': np.nan, 'max_dd': np.nan}
    r = eq.pct_change().dropna()
    dd = float((eq / eq.cummax() - 1.0).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    return {'ret': float(eq.iloc[-1] / eq.iloc[0] - 1.0), 'sharpe': sharpe, 'max_dd': dd}


rows = []
for pool, w, top_k, kind in PRESETS:
    print(f'--- {pool} ---', flush=True)
    kit = BacktestKit(pool=pool, start=None)
    if kind == 'us':
        kit.register(build_alpha_cands, label='alpha候选')
        kit.register(build_combo_cands, label='combo候选')
    else:
        kit.register(build_a_combo_cands, label='A候选')
    base = kit.engine_params.copy(top_k=top_k)
    comp = kit._composite(w, 'rank')
    ow, cw = _slice(kit.open_wide, START, END), _slice(kit.close_wide, START, END)
    c = comp.reindex(index=ow.index, columns=ow.columns)
    g = pd.DataFrame(True, index=ow.index, columns=ow.columns)
    atr = _slice(kit.atr_wide, START, END) if kit.atr_wide is not None else None

    res = {}
    for sizing in ('tier', 'equal'):
        p = base.copy(sizing=sizing)
        payload = run_config(f'{pool}|{sizing}', ow, cw, c, g, p, atr_wide=atr)
        s = payload['stats']
        ytd = window_stats(payload['equity'], YTD_START)
        res[sizing] = {'stats': s, 'ytd': ytd, 'payload': payload}
        print(f"  {sizing:5s}: CAGR={s['cagr']:+.1%} Sharpe={s['sharpe']:.2f} "
              f"maxDD={s['max_dd']:.1%} 年换手={s['ann_turnover']:.0f} "
              f"YTD26={ytd['ret']:+.1%}", flush=True)

    t, e = res['tier']['stats'], res['equal']['stats']
    ty, ey = res['tier']['ytd'], res['equal']['ytd']
    rows.append({
        'pool': pool, 'top_k': top_k,
        'tier_cagr': t['cagr'], 'equal_cagr': e['cagr'], 'd_cagr': round(e['cagr'] - t['cagr'], 4),
        'tier_sharpe': t['sharpe'], 'equal_sharpe': e['sharpe'], 'd_sharpe': round(e['sharpe'] - t['sharpe'], 3),
        'tier_maxdd': t['max_dd'], 'equal_maxdd': e['max_dd'], 'd_maxdd': round(e['max_dd'] - t['max_dd'], 4),
        'tier_calmar': t['calmar'], 'equal_calmar': e['calmar'],
        'tier_vol': t['vol'], 'equal_vol': e['vol'],
        'tier_turn': t['ann_turnover'], 'equal_turn': e['ann_turnover'],
        'tier_ytd26': round(ty['ret'], 4), 'equal_ytd26': round(ey['ret'], 4),
        'd_ytd26': round(ey['ret'] - ty['ret'], 4),
        'tier_nTrades': t.get('n_trades'), 'equal_nTrades': e.get('n_trades'),
        'tier_avgDays': t.get('avg_days'), 'equal_avgDays': e.get('avg_days'),
        'tier_equityEnd': round(res['tier']['payload']['equity'].iloc[-1], 3),
        'equal_equityEnd': round(res['equal']['payload']['equity'].iloc[-1], 3),
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'pool_sizing_cmp.csv'), index=False, encoding='utf-8-sig')

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)
print('\n===== 分档(tier) vs 等权(equal) =====')
show = df[['pool', 'tier_cagr', 'equal_cagr', 'd_cagr', 'tier_sharpe', 'equal_sharpe', 'd_sharpe',
           'tier_maxdd', 'equal_maxdd', 'tier_calmar', 'equal_calmar',
           'tier_turn', 'equal_turn', 'tier_ytd26', 'equal_ytd26']]
print(show.to_string(index=False))
print('\n[saved]', os.path.join(OUT, 'pool_sizing_cmp.csv'))

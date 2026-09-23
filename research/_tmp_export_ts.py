# -*- coding: utf-8 -*-
"""临时脚本 3: 按 signal_bridge 各池预设, 导出逐日/逐笔记录 + 时序可视化.

产出:
  research/output/pool_timeseries/<pool>/{equity_curve.csv,positions.csv,trades.csv}
  research/output/pool_timeseries/charts/<pool>_full.png
  research/output/pool_timeseries/charts/all_pools_equity.png
  research/output/pool_timeseries/charts/monthly_heatmap.png
  research/output/pool_timeseries/dashboard.html   (plotly 交互: 5 池可切换)
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from bt_core import BacktestKit, monthly_returns        # noqa: E402
from combo_search import _slice, build_combo_cands      # noqa: E402
from alpha_mining import build_alpha_cands              # noqa: E402
from a_combo_search import build_a_combo_cands          # noqa: E402
from score_backtest import run_config                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

OUT = os.path.join(BASE, 'output', 'pool_timeseries')
CHART = os.path.join(OUT, 'charts')
os.makedirs(CHART, exist_ok=True)

PRESETS = [
    ('company_300', {'C_tmqmom': 1.0, 'F_er20': 1.0}, 12, 'us'),
    ('company_1000', {'G_vr20': 1.0, 'H_ichimoku_alpha': 1.0}, 8, 'us'),
    ('etf_3x', {'F_mom121': 0.5, 'F_er20': 0.3, 'N_idiovol60': 0.2, 'F_obv20': 0.1}, 5, 'us'),
    ('hs300', {'N_range20': -1.0, 'D_ma20_dist': -1.0, 'C_tmqmom': 0.5}, 5, 'a'),
    ('a_etf_all', {'N_range20': 1.0, 'G_oviv20': 0.5}, 5, 'a'),
]
START, END = '2021-01-01', None

store = {}
for pool, w, top_k, kind in PRESETS:
    print(f'--- {pool} ---', flush=True)
    kit = BacktestKit(pool=pool, start=None)
    if kind == 'us':
        kit.register(build_alpha_cands, label='alpha候选')
        kit.register(build_combo_cands, label='combo候选')
    else:
        kit.register(build_a_combo_cands, label='A候选')
    p = kit.engine_params.copy(top_k=top_k)
    comp = kit._composite(w, 'rank')
    ow, cw = _slice(kit.open_wide, START, END), _slice(kit.close_wide, START, END)
    c = comp.reindex(index=ow.index, columns=ow.columns)
    g = pd.DataFrame(True, index=ow.index, columns=ow.columns)
    atr = _slice(kit.atr_wide, START, END) if kit.atr_wide is not None else None
    payload = run_config(f'{pool}|full', ow, cw, c, g, p, atr_wide=atr)

    d = os.path.join(OUT, pool)
    os.makedirs(d, exist_ok=True)
    payload['daily'].to_csv(os.path.join(d, 'equity_curve.csv'), encoding='utf-8-sig')
    payload['pos'].to_csv(os.path.join(d, 'positions.csv'), index=False, encoding='utf-8-sig')
    payload['trades'].to_csv(os.path.join(d, 'trades.csv'), index=False, encoding='utf-8-sig')

    store[pool] = payload
    s = payload['stats']
    print(f"  daily={len(payload['daily'])} pos={len(payload['pos'])} "
          f"trades={len(payload['trades'])} equity_end={payload['equity'].iloc[-1]:.3f} "
          f"sharpe={s['sharpe']:.2f}", flush=True)

# ---------------------------------------------------------------- 静态图 (matplotlib)
import matplotlib                                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 130

CMAP = {'company_300': '#d62728', 'company_1000': '#ff7f0e', 'etf_3x': '#9467bd',
        'hs300': '#1f77b4', 'a_etf_all': '#2ca02c'}

for pool, payload in store.items():
    eq = payload['equity']
    dd = eq / eq.cummax() - 1.0
    dl = payload['daily']
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.2, 1.2]})
    ax = axes[0]
    ax.plot(eq.index, eq.values, color=CMAP[pool], lw=1.5, label='策略净值')
    bh = dl['equity'].iloc[-1] * 0  # 占位
    ax.axvspan(pd.Timestamp('2026-01-01'), eq.index[-1], color='#ffd700', alpha=0.18,
               label='2026 年以来')
    ax.set_yscale('log')
    ax.set_ylabel('净值(对数轴, 起点=1)')
    ax.set_title(f'{pool}  signal_bridge 预设策略 · 全期资金曲线 '
                 f'(CAGR {payload["stats"]["cagr"]:.1%} / Sharpe {payload["stats"]["sharpe"]:.2f} '
                 f'/ maxDD {payload["stats"]["max_dd"]:.1%})')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

    ax = axes[1]
    ax.fill_between(dd.index, dd.values, 0, color='#c0392b', alpha=0.5)
    ax.set_ylabel('回撤')
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(dl.index, dl['turnover'].rolling(20).sum() / 20 * 252, color='#555', lw=1.2,
            label='20日滚动年化换手')
    ax2 = ax.twinx()
    ax2.plot(dl.index, dl['n_pos'], color='#2980b9', lw=1.0, alpha=0.8, label='持仓数')
    ax.set_ylabel('年化换手')
    ax2.set_ylabel('持仓数')
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    plt.tight_layout()
    fn = os.path.join(CHART, f'{pool}_full.png')
    fig.savefig(fn, bbox_inches='tight')
    plt.close(fig)
    print('[png]', fn, flush=True)

# 5 池对比
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1.2]})
ax = axes[0]
for pool, payload in store.items():
    eq = payload['equity']
    ax.plot(eq.index, eq.values, color=CMAP[pool], lw=1.5,
            label=f'{pool} (Sharpe {payload["stats"]["sharpe"]:.2f})')
ax.axvspan(pd.Timestamp('2026-01-01'), store['hs300']['equity'].index[-1],
           color='#ffd700', alpha=0.18)
ax.set_yscale('log')
ax.set_ylabel('净值(对数轴, 起点=1)')
ax.set_title('signal_bridge 五池策略净值对比(2021-01 ~ 2026-09, 扣单边 10bps)')
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=9)
ax = axes[1]
for pool, payload in store.items():
    eq = payload['equity']
    dd = eq / eq.cummax() - 1.0
    ax.plot(dd.index, dd.values, color=CMAP[pool], lw=1.1, label=pool)
ax.set_ylabel('回撤')
ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=8, ncol=5)
plt.tight_layout()
fn = os.path.join(CHART, 'all_pools_equity.png')
fig.savefig(fn, bbox_inches='tight')
plt.close(fig)
print('[png]', fn, flush=True)

# 月度收益热力图
mats = {}
for pool, payload in store.items():
    m = monthly_returns(payload['equity'])
    mr = m.to_frame('r')
    mr['y'] = [int(x[:4]) for x in mr.index]
    mr['m'] = [int(x[5:7]) for x in mr.index]
    mats[pool] = mr.pivot(index='y', columns='m', values='r')
fig, axes = plt.subplots(len(mats), 1, figsize=(12, 2.0 * len(mats)))
vmax = max(np.nanmax(np.abs(m.values)) for m in mats.values())
for ax, (pool, mat) in zip(np.atleast_1d(axes), mats.items()):
    im = ax.imshow(mat.values, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(mat.columns)), [f'{c}月' for c in mat.columns])
    ax.set_yticks(range(len(mat.index)), [str(i) for i in mat.index])
    ax.set_title(f'{pool} 月度收益', fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{v * 100:.0f}', ha='center', va='center', fontsize=7)
fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.6, location='right')
fn = os.path.join(CHART, 'monthly_heatmap.png')
fig.savefig(fn, bbox_inches='tight')
plt.close(fig)
print('[png]', fn, flush=True)

# ---------------------------------------------------------------- 交互 HTML (plotly)
import plotly.graph_objects as go                        # noqa: E402
from plotly.subplots import make_subplots                # noqa: E402

fig = make_subplots(
    rows=2, cols=2, specs=[[{'secondary_y': False}, {}], [{}, {}]],
    subplot_titles=('资金曲线(对数轴)', '回撤', '逐笔交易盈亏(按出场日, 颜色=收益)',
                    '日换手与持仓数'),
    vertical_spacing=0.13, horizontal_spacing=0.09)
first = True
for pool, payload in store.items():
    eq = payload['equity']
    dd = eq / eq.cummax() - 1.0
    dl = payload['daily']
    tr = payload['trades']
    vis = True
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=f'{pool} 净值',
                             legendgroup=pool, visible=vis, line=dict(color=CMAP[pool], width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=f'{pool} 回撤', fill='tozeroy',
                             legendgroup=pool, visible=vis,
                             line=dict(color=CMAP[pool], width=1)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=pd.to_datetime(tr['exit_exec']), y=tr['ret'],
                             mode='markers', name=f'{pool} 逐笔收益', legendgroup=pool,
                             visible=vis,
                             marker=dict(size=5, color=tr['ret'], colorscale='RdYlGn',
                                         cmid=0, showscale=False,
                                         line=dict(width=0)),
                             text=[f"{s} | {a:%Y-%m-%d}→{b:%Y-%m-%d} | {d}天 | {r:+.1%}"
                                   for s, a, b, d, r in zip(tr['symbol'], pd.to_datetime(tr['entry_exec']),
                                                            pd.to_datetime(tr['exit_exec']), tr['days'], tr['ret'])]),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dl.index, y=dl['turnover'].rolling(20).sum() / 20 * 252,
                             name=f'{pool} 年化换手', legendgroup=pool, visible=vis,
                             line=dict(color=CMAP[pool], width=1.2)),
                  row=2, col=2)
    first = False

# 下拉切换: 每个池一份 visible 掩码
n_pool = len(store)
n_trace = 4                                     # 每池 4 条 trace
buttons = []
for k, pool in enumerate(store):
    vis = []
    for t in range(n_pool * n_trace):
        vis.append((t // n_trace) == k)
    buttons.append(dict(label=pool, method='update',
                        args=[{'visible': vis},
                              {'title': f'{pool} · signal_bridge 预设策略时序诊断'}]))
fig.update_layout(
    updatemenus=[dict(buttons=buttons, x=0.0, y=1.14, xanchor='left', showactive=True)],
    title=f'{list(store)[0]} · signal_bridge 预设策略时序诊断',
    template='plotly_white', height=880, hovermode='closest',
    legend=dict(orientation='h', y=-0.13))
fig.update_yaxes(type='log', title_text='净值', row=1, col=1)
fig.update_yaxes(tickformat='.0%', title_text='回撤', row=1, col=2)
fig.update_yaxes(tickformat='.0%', title_text='单笔收益', row=2, col=1)
fig.update_yaxes(title_text='年化换手(20日滚动)', row=2, col=2)
fn = os.path.join(OUT, 'dashboard.html')
fig.write_html(fn, include_plotlyjs='inline')
print('[html]', fn, f'({os.path.getsize(fn) / 1e6:.1f} MB)', flush=True)

# 汇总索引
rows = []
for pool, payload in store.items():
    s, tr = payload['stats'], payload['trades']
    done = tr[tr['completed']] if 'completed' in tr.columns else tr
    rows.append({'pool': pool, 'days': len(payload['daily']), 'positions_rows': len(payload['pos']),
                 'trades': len(tr), 'completed': len(done),
                 'first_day': f"{payload['equity'].index[0]:%Y-%m-%d}",
                 'last_day': f"{payload['equity'].index[-1]:%Y-%m-%d}",
                 'sharpe': s['sharpe'], 'cagr': s['cagr'], 'max_dd': s['max_dd'],
                 'ann_turnover': s['ann_turnover']})
idx = pd.DataFrame(rows)
idx.to_csv(os.path.join(OUT, 'INDEX.csv'), index=False, encoding='utf-8-sig')
print('\n', idx.to_string(index=False))

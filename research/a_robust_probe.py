"""a_robust_probe.py — a_robust_combo.py 的配套探针(结论证据链).

两部分:
  A) 关键候选的日截面秩相关(冗余度): 判断新因子是否只是既有成分的变形
  B) 关键组合的窗口敏感性 + 反向控制: 判断结论是否被单点数值偶然支撑

用法: python a_robust_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a_combo_search import daily_spearman, MIN_CS            # noqa: E402
from a_robust_combo import Pool, WIN, BASELINE               # noqa: E402

PAIRS = [('N_range20', 'G_oviv20'), ('N_range20', 'X_ovnshare60'),
         ('G_oviv20', 'X_ovnshare60'), ('N_range20', 'D_ma20_dist'),
         ('N_range20', 'C_tmqmom'), ('D_ma20_dist', 'C_tmqmom'),
         ('N_range20', 'D_sharpe20'), ('N_range20', 'F_beta60')]
WINS = ('IS1', 'IS2', 'OOS', 'FULL')


def show(P, tag, weights, top_ks=(5,)):
    s = []
    for k in top_ks:
        vals = '  '.join(f'{w}={P.sh(weights, w, k):+.2f}' for w in WINS)
        s.append(f'top{k} {vals}')
    print(f'  {tag:<34} ' + ' | '.join(s))


def part_a(P):
    pct = {n: P.runner._pct[n] for n in sorted({n for p in PAIRS for n in p})}
    print('\n A) 日截面秩相关(冗余度)')
    for win in WINS:
        st, en = WIN[win]
        out = []
        for a, b in PAIRS:
            rho = daily_spearman(pct[a].loc[st:en], pct[b].loc[st:en], MIN_CS).dropna()
            out.append(f'{a}~{b}={rho.mean():+.3f}')
        print(f'    [{win:>4}] ' + '  '.join(out))


def part_b(P):
    print('\n B) 窗口敏感性 + 反向控制')
    if P.pool == 'a_etf_all':
        print('   G_oviv20 幅度扫描 (N_range20:1 固定)')
        for w in (0.0, 0.25, 0.5, 1.0):
            d = {'N_range20': 1.0}
            if w:
                d['G_oviv20'] = w
            show(P, f'N_range20:1,G_oviv20:{w}', d)
        print('   替代与反向')
        show(P, 'X_ovnshare60:1 单因子', {'X_ovnshare60': 1.0})
        show(P, 'N_range20:-1 (反向)', {'N_range20': -1.0})
        show(P, '基线', BASELINE[P.pool])
    else:
        print('   基线成分删除扫描')
        show(P, '基线 (N:-1,D:-1,C:0.5)', BASELINE[P.pool])
        show(P, '去 N_range20', {'D_ma20_dist': -1.0, 'C_tmqmom': 0.5})
        show(P, '去 C_tmqmom', {'N_range20': -1.0, 'D_ma20_dist': -1.0})
        show(P, '去 D_ma20_dist', {'N_range20': -1.0, 'C_tmqmom': 0.5})
        print('   反向控制')
        show(P, 'N_range20:+1 (反向)', {'N_range20': 1.0})
        show(P, 'F_beta60:+1 (反向)', {'F_beta60': 1.0})
        print('   候选单信号')
        for n in ('D_ma20_dist', 'N_range20', 'F_beta60', 'D_sharpe20', 'G_vr20', 'G_dnbeta60'):
            show(P, f'{n}:-1 单因子', {n: -1.0})


for pool in ('a_etf_all', 'hs300'):
    P = Pool(pool)
    print(f'\n===== {pool} =====')
    part_a(P)
    part_b(P)

# -*- coding: utf-8 -*-
"""
a_robust_combo.py — A 股组合层稳健性重估与新组合构建
==========================================================================
动机: a_combo_search 的贪心仅在 IS(2021~2024) 单窗上优化, 报告已指出
"建议加 --objective min(is,oos)"。本脚本把选参口径改成**IS 内部双半窗**
(IS1=2021~2022, IS2=2023~2024), 目标函数 = min(sharpe_IS1, sharpe_IS2),
即要求候选在两种截然不同的市场 regime 下都成立; OOS(2025~) 只做复验不参与选参。

阶段:
  0. 基线复现: signal_bridge A_POOL_PRESETS 两池预设 (FULL/IS1/IS2/OOS + shuffle/buyhold)
  1. 候选矩阵: 16 候选(14 既有 + I_id20 + X_ovnshare60) × 双向 × IS1/IS2/OOS/FULL
  2. 贪心构图: 目标 min(IS1,IS2) sharpe, 新增纳入需提升 >= margin; 坐标精修; top_k 网格
  3. 复验: 新组合 OOS/FULL + shuffle/buyhold + goldA4 + 单信号 OOS 对照

候选公式全部逐行复用 a_combo_search.build_a_combo_cands (权威口径), 附加探针:
  I_id20        = sign(pct_change20) * (跌天占比 - 涨天占比)      [a_stat_mining L117-121]
  X_ovnshare60  = var60(隔夜) / (var60(隔夜)+var60(日内))          [a_share_mining L140-141]

输出: research/output/a_robust_{run_id}/
用法: python a_robust_combo.py
"""
import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_core import BacktestKit                                  # noqa: E402
from a_combo_search import (                                     # noqa: E402
    build_a_combo_cands, PoolRunner, RED_NAMES, GOLD_A4, wdesc, log)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
WIN = {'FULL': ('2021-01-01', None),
       'IS1': ('2021-01-01', '2022-12-31'),
       'IS2': ('2023-01-01', '2024-12-31'),
       'OOS': ('2025-01-01', None),
       'OOSA': ('2025-01-01', '2025-06-30'),
       'OOSB': ('2025-07-01', None)}
EXTRA = ['I_id20', 'X_ovnshare60']
CANDS = RED_NAMES + EXTRA
BASELINE = {
    'hs300': {'N_range20': -1.0, 'D_ma20_dist': -1.0, 'C_tmqmom': 0.5},
    'a_etf_all': {'N_range20': 1.0, 'G_oviv20': 0.5},
}
TOPK_GRID = [5, 8, 12]
MARGIN = 0.10
MAX_ADD = 4
SHUFFLE_SEED = 42
STAT_KEYS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'vol',
             'ann_turnover', 'n_trades', 'win_rate', 'n_days']


def build_a_extra(panel: pd.DataFrame) -> dict:
    """两个补充探针(A 股专属 + 唯一 FDR 显著单信号), 与既有候选同口径."""
    close = panel['Close'].unstack('symbol').sort_index()
    open_ = panel['Open'].unstack('symbol').sort_index()
    ret1 = close.pct_change()
    out = {}
    # I_id20: 信息离散度 x 动量符号 (a_stat_mining)
    up = (ret1 > 0).where(ret1.notna()).astype(float)
    dn = (ret1 < 0).where(ret1.notna()).astype(float)
    out['I_id20'] = np.sign(close.pct_change(20)) * (
        dn.rolling(20, min_periods=10).mean() - up.rolling(20, min_periods=10).mean())
    # X_ovnshare60: 隔夜方差占比 (a_share_mining)
    ov = (open_ / close.shift(1) - 1.0).rolling(60, min_periods=30).var()
    iv = (close / open_ - 1.0).rolling(60, min_periods=30).var()
    out['X_ovnshare60'] = ov / (ov + iv).replace(0, np.nan)
    return out


def _sh(st: dict) -> float:
    v = st.get('sharpe')
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return -1e9
    return float(v)


class Runner(PoolRunner):
    """PoolRunner 的列集合可配版: 预缓存 CANDS(16) 而非默认的 14 列."""

    def __init__(self, kit, cols):
        self.kit = kit
        kit._get_cands()
        self._pct = {}
        for col in cols:
            wide = pd.to_numeric(kit.study[col], errors='coerce') \
                         .unstack('symbol').sort_index()
            self._pct[col] = wide.rank(axis=1, pct=True)
        probe = {cols[0]: 1.0, cols[3]: -0.7}
        ref = kit._composite(dict(probe), 'rank')
        fast = self.fast_composite(probe).reindex(index=ref.index, columns=ref.columns)
        if not np.allclose(ref.to_numpy(), fast.to_numpy(), equal_nan=True):
            raise RuntimeError('快速组合口径与 compute_composite 不一致, 禁用快速路径')


class Pool:
    """单池: 一次性建 kit + runner, 提供 ev/obj 缓存."""

    def __init__(self, pool: str):
        t0 = time.time()
        self.pool = pool
        self.kit = BacktestKit(pool=pool, start=None)
        self.kit.register(build_a_combo_cands, label='a_combo候选')
        self.kit.register(build_a_extra, label='补充探针')
        self.runner = Runner(self.kit, CANDS)
        self._cache = {}
        log(f'[{pool}] 就绪 {time.time() - t0:.0f}s  标的 {self.kit.open_wide.shape[1]} '
            f'交易日 {self.kit.open_wide.shape[0]}')

    # ---- 单次评估(带缓存) ----
    def ev(self, weights: dict, window: str, top_k: int = 5, shuffle_seed=None):
        key = (tuple(sorted(weights.items())), window, top_k, shuffle_seed)
        if key in self._cache:
            return self._cache[key]
        st, en = WIN[window]
        p = self.kit.engine_params if top_k == 5 else self.kit.engine_params.copy(top_k=top_k)
        payload = self.runner.run_window(weights, f'{window}', st, en,
                                         engine_params=p, shuffle_seed=shuffle_seed)
        self._cache[key] = payload
        return payload

    def sh(self, weights, window, top_k=5, shuffle_seed=None) -> float:
        return _sh(self.ev(weights, window, top_k, shuffle_seed)['stats'])

    def bh(self, window: str, top_k: int = 5) -> float:
        key = ('__bh__', window, top_k)
        if key not in self._cache:
            st, en = WIN[window]
            self._cache[key] = self.runner.buyhold_window('buyhold', st, en)
        return _sh(self._cache[key]['stats'])

    # ---- 目标函数: min(IS1, IS2) ----
    def obj(self, weights, top_k=5):
        s1, s2 = self.sh(weights, 'IS1', top_k), self.sh(weights, 'IS2', top_k)
        return min(s1, s2), s1, s2


def stage0_baseline(P: Pool, rows: list, rep: list):
    rep.append(f'\n===== [{P.pool}] 阶段0 基线复现 =====')
    base = BASELINE[P.pool]
    rep.append(f'signal_bridge 预设: {wdesc(base)}  top_k=5')
    for w in ('FULL', 'IS1', 'IS2', 'OOS'):
        for tag, wd in (('baseline', base), ('buyhold', None)):
            if wd is None:
                s = P.bh(w)
                rows.append({'pool': P.pool, 'stage': 'baseline', 'window': w,
                             'config': 'buyhold_pool', 'weights': '', 'sharpe': round(s, 3)})
                continue
            s = P.sh(wd, w)
            rows.append({'pool': P.pool, 'stage': 'baseline', 'window': w,
                         'config': 'baseline_preset', 'weights': wdesc(wd),
                         'sharpe': round(s, 3)})
        sh_c = P.sh(base, w, shuffle_seed=SHUFFLE_SEED)
        rows.append({'pool': P.pool, 'stage': 'baseline', 'window': w,
                     'config': 'baseline_shuffle', 'weights': wdesc(base),
                     'sharpe': round(sh_c, 3)})
        rep.append(f'  {w}: baseline={P.sh(base, w):.3f}  shuffle={sh_c:.3f}  '
                   f'buyhold={P.bh(w):.3f}')
    ob, s1, s2 = P.obj(base)
    rep.append(f'  目标 min(IS1,IS2) = {ob:.3f}  (IS1={s1:.3f}, IS2={s2:.3f})')
    g = P.sh(GOLD_A4, 'FULL', 5)
    rep.append(f'  goldA4(两池同号4因子) FULL={g:.3f}')
    for w in ('IS1', 'IS2', 'OOS', 'FULL'):
        rows.append({'pool': P.pool, 'stage': 'baseline', 'window': w,
                     'config': 'goldA4', 'weights': wdesc(GOLD_A4),
                     'sharpe': round(P.sh(GOLD_A4, w, 5), 3)})


def stage1_matrix(P: Pool, rows: list, rep: list) -> dict:
    """每候选 × 双向 → IS1/IS2/OOS/FULL; 选向 = 最大化 min(IS1,IS2)."""
    rep.append(f'\n===== [{P.pool}] 阶段1 候选稳健性矩阵 =====')
    table, chosen = [], {}
    for name in CANDS:
        best = None
        for d, tag in ((1.0, 'pos'), (-1.0, 'neg')):
            w = {name: d}
            o, s1, s2 = P.obj(w)
            so, sf = P.sh(w, 'OOS'), P.sh(w, 'FULL')
            rows.append({'pool': P.pool, 'stage': 'single', 'window': 'IS1', 'config': tag,
                         'weights': wdesc(w), 'sharpe': round(s1, 3)})
            row = {'signal': name, 'dir': tag, 'obj_min': round(o, 3),
                   'IS1': round(s1, 3), 'IS2': round(s2, 3),
                   'OOS': round(so, 3), 'FULL': round(sf, 3)}
            if best is None or o > best['obj_min']:
                best = row
        table.append(best)
        chosen[name] = 1.0 if best['dir'] == 'pos' else -1.0
    t = pd.DataFrame(table).sort_values('obj_min', ascending=False)
    rep.append(t.to_string(index=False))
    log(f'[{P.pool}] 阶段1 候选矩阵 obj 前5:\n' + t.head(5).to_string(index=False))
    return chosen, t


def stage2_greedy(P: Pool, chosen: dict, rows: list, rep: list):
    """贪心: 目标 min(IS1,IS2); 种子=最优单信号, 逐次新增(方向取单信号最优向)."""
    rep.append(f'\n===== [{P.pool}] 阶段2 贪心构图 (目标=min(IS1,IS2)) =====')
    log_lines = []
    # 种子
    singles = sorted(((P.obj({n: chosen[n]})[0], n) for n in CANDS), reverse=True)
    seed_obj, seed = singles[0]
    have = {seed: chosen[seed]}
    cur, s1, s2 = P.obj(have)
    log_lines.append({'step': 0, 'action': 'seed', 'have': wdesc(have),
                      'obj': round(cur, 3), 'IS1': round(s1, 3), 'IS2': round(s2, 3)})
    rep.append(f'  种子 {seed} → obj={cur:.3f} (IS1={s1:.3f}, IS2={s2:.3f})')
    log(f'[{P.pool}] 贪心种子 {seed} obj={cur:.3f}')

    for it in range(1, MAX_ADD + 1):
        best = (cur, None)
        for n in CANDS:
            if n in have:
                continue
            for mag in (1.0, 0.5):
                trial = dict(have)
                trial[n] = chosen[n] * mag
                o, _, _ = P.obj(trial)
                if o > best[0]:
                    best = (o, (n, chosen[n] * mag))
        if best[1] is None or best[0] - cur < MARGIN:
            log_lines.append({'step': it, 'action': 'stop',
                              'have': wdesc(have), 'obj': round(cur, 3),
                              'IS1': round(s1, 3), 'IS2': round(s2, 3)})
            rep.append(f'  第{it}轮: 无满足 margin>{MARGIN} 的新增 → 停止')
            break
        n, wv = best[1]
        have[n] = wv
        prev = cur
        cur, s1, s2 = P.obj(have)
        log_lines.append({'step': it, 'action': f'add {n}:{wv:g}', 'have': wdesc(have),
                          'obj': round(cur, 3), 'IS1': round(s1, 3), 'IS2': round(s2, 3)})
        rep.append(f'  第{it}轮: +{n}:{wv:g} → obj={cur:.3f} (IS1={s1:.3f}, '
                   f'IS2={s2:.3f}, +{cur - prev:.3f})')
        log(f'[{P.pool}] 贪心第{it}轮 +{n}:{wv:g} → obj={cur:.3f}')

    # 坐标精修(幅度网格, 方向锁定; 2 遍)
    for p in range(2):
        improved = False
        for n in list(have):
            cur_w = have[n]
            base_o = P.obj(have)[0]
            best = (base_o, cur_w)
            for mag in (0.25, 0.5, 1.0):
                have[n] = chosen[n] * mag
                o = P.obj(have)[0]
                if o > best[0] + 1e-9:
                    best = (o, chosen[n] * mag)
            have[n] = best[1]
            if abs(best[1] - cur_w) > 1e-9:
                improved = True
                log_lines.append({'step': f'refine{p}', 'action': f'{n} -> {best[1]:g}',
                                  'have': wdesc(have), 'obj': round(best[0], 3),
                                  'IS1': '', 'IS2': ''})
        if not improved:
            break
    cur, s1, s2 = P.obj(have)
    rep.append(f'  精修后 obj={cur:.3f} (IS1={s1:.3f}, IS2={s2:.3f})  {wdesc(have)}')

    # top_k 网格
    best_k = (cur, 5)
    for k in TOPK_GRID:
        o = P.obj(have, k)[0]
        if o > best_k[0]:
            best_k = (o, k)
    top_k = best_k[1]
    cur, s1, s2 = P.obj(have, top_k)
    rep.append(f'  top_k 网格 → top_k={top_k} obj={cur:.3f}')
    log_lines.append({'step': 'topk', 'action': f'top_k={top_k}', 'have': wdesc(have),
                      'obj': round(cur, 3), 'IS1': round(s1, 3), 'IS2': round(s2, 3)})
    return have, top_k, log_lines


def stage4_from_baseline(P: Pool, chosen: dict, rows: list, rep: list):
    """从基线权重出发做删除/加入/调幅(目标 min(IS1,IS2)), 直接回答'基线能否被改进'.
    删除动作是关键: 检验多因子基线是否被某个单因子支配(分散化是否值得保留)."""
    rep.append(f'\n===== [{P.pool}] 阶段4 从基线出发的改进 =====')
    have = dict(BASELINE[P.pool])
    cur, s1, s2 = P.obj(have)
    rep.append(f'  起点 obj={cur:.3f} (IS1={s1:.3f}, IS2={s2:.3f})  {wdesc(have)}')
    log(f'[{P.pool}] 阶段4 基线起点 obj={cur:.3f} {wdesc(have)}')
    M = 0.05
    # 1) 删除成分(可多轮, 每次删提升最大者)
    while True:
        best = (cur, None)
        for n in list(have):
            t = {k: v for k, v in have.items() if k != n}
            if not t:
                continue
            o = P.obj(t)[0]
            if o > best[0]:
                best = (o, n)
        if best[1] is None or best[0] - cur < M:
            break
        n = best[1]
        t = {k: v for k, v in have.items() if k != n}
        rep.append(f'  删除 {n}: obj {cur:.3f} → {best[0]:.3f}  {wdesc(t)}')
        log(f'[{P.pool}] 阶段4 删除 {n} → obj={best[0]:.3f}')
        have, cur = t, best[0]
    # 2) 加入候选(先按 ±1.0 试探, 幅度交给下方精修)
    for it in range(MAX_ADD):
        best = (cur, None)
        for n in CANDS:
            if n in have:
                continue
            t = dict(have)
            t[n] = chosen[n]
            o = P.obj(t)[0]
            if o > best[0]:
                best = (o, n)
        if best[1] is None or best[0] - cur < M:
            break
        n = best[1]
        have[n] = chosen[n]
        cur = P.obj(have)[0]
        rep.append(f'  加入 {n}:{chosen[n]:g} → obj={cur:.3f}  {wdesc(have)}')
        log(f'[{P.pool}] 阶段4 加入 {n}:{chosen[n]:g} → obj={cur:.3f}')
    # 3) 调幅精修(方向锁定)
    for _ in range(2):
        improved = False
        for n in list(have):
            w0 = have[n]
            best = (P.obj(have)[0], w0)
            for mag in (0.25, 0.5, 1.0):
                have[n] = chosen[n] * mag
                o = P.obj(have)[0]
                if o > best[0] + 1e-9:
                    best = (o, chosen[n] * mag)
            have[n] = best[1]
            improved |= abs(best[1] - w0) > 1e-9
        if not improved:
            break
    cur, s1, s2 = P.obj(have)
    rep.append(f'  精修后 obj={cur:.3f} (IS1={s1:.3f}, IS2={s2:.3f})  {wdesc(have)}')
    # 4) top_k 网格
    best_k = (cur, 5)
    for k in TOPK_GRID:
        o = P.obj(have, k)[0]
        if o > best_k[0]:
            best_k = (o, k)
    top_k = best_k[1]
    rep.append(f'  top_k 网格 → top_k={top_k}  {wdesc(have)}')
    log(f'[{P.pool}] 阶段4 完成 {wdesc(have)} top_k={top_k} obj={P.obj(have, top_k)[0]:.3f}')
    return have, top_k


def stage5_final(P: Pool, have: dict, top_k: int, rows: list, rep: list,
                 tag: str = 'new_combo') -> dict:
    """胜出配置终验: 六窗(含 OOS 子窗) + shuffle/buyhold + top_k 敏感性."""
    rep.append(f'\n===== [{P.pool}] 阶段5 终验 [{tag}]  {wdesc(have)} top_k={top_k} =====')
    out = {'pool': P.pool, 'tag': tag, 'weights': wdesc(have), 'top_k': top_k}
    for w in ('FULL', 'IS1', 'IS2', 'OOS', 'OOSA', 'OOSB'):
        s, bh = P.sh(have, w, top_k), P.bh(w, top_k)
        sc = P.sh(have, w, top_k, shuffle_seed=SHUFFLE_SEED)
        out[f'{w}_sharpe'], out[f'{w}_bh'], out[f'{w}_shuf'] = round(s, 3), round(bh, 3), round(sc, 3)
        rep.append(f'  {w:<5} sharpe={s:+.3f}  buyhold={bh:+.3f}  shuffle={sc:+.3f}  '
                   f'excess_bh={s - bh:+.3f}')
        rows.append({'pool': P.pool, 'stage': 'final', 'window': w, 'config': tag,
                     'weights': wdesc(have), 'sharpe': round(s, 3)})
    sens = []
    for k in TOPK_GRID:
        r = {f'{w}': round(P.sh(have, w, k), 3) for w in ('IS1', 'IS2', 'OOS', 'FULL')}
        sens.append({'top_k': k, **r, 'obj_min': round(min(r['IS1'], r['IS2']), 3)})
    st = pd.DataFrame(sens)
    rep.append('  top_k 敏感性:\n' + st.to_string(index=False))
    log(f'[{P.pool}] 阶段5 [{tag}] top_k 敏感性:\n' + st.to_string(index=False))
    return out


def run_pool(pool: str, out_dir: str):
    P = Pool(pool)
    rows = []
    rep = [f'pool={pool}  IS1={WIN["IS1"]} IS2={WIN["IS2"]} OOS={WIN["OOS"]}',
           f'目标函数 = min(sharpe_IS1, sharpe_IS2); margin={MARGIN}; '
           f'候选={len(CANDS)}(含补充探针 {EXTRA})']
    stage0_baseline(P, rows, rep)
    chosen, matrix = stage1_matrix(P, rows, rep)
    # A) 单信号种子贪心(旧路径, 易过拟合, 保留作对照)
    have_g, topk_g, glog = stage2_greedy(P, chosen, rows, rep)
    info_g = stage5_final(P, have_g, topk_g, rows, rep, tag='greedy_from_single')
    # B) 从基线出发的改进(主推荐路径: 删除/加入/调幅)
    have_b, topk_b = stage4_from_baseline(P, chosen, rows, rep)
    info_b = stage5_final(P, have_b, topk_b, rows, rep, tag='baseline_improved')
    # C) 基线自身(参照)
    info_ref = stage5_final(P, BASELINE[pool], 5, rows, rep, tag='baseline_ref')
    matrix.to_csv(os.path.join(out_dir, f'matrix_{pool}.csv'), index=False,
                  encoding='utf-8-sig')
    pd.DataFrame(glog).to_csv(os.path.join(out_dir, f'greedy_{pool}.csv'), index=False,
                              encoding='utf-8-sig')
    return rows, rep, [info_g, info_b, info_ref]


def main():
    ap = argparse.ArgumentParser(description='A 股组合层稳健性重估(IS 双半窗目标)')
    ap.add_argument('--pool', default=None, help='单池(默认 hs300,a_etf_all)')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    pools = [args.pool] if args.pool else ['hs300', 'a_etf_all']
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(BASE, 'output', f'a_robust_{run_id}' + (f'_{args.tag}' if args.tag else ''))
    os.makedirs(out_dir, exist_ok=True)
    log(f'A 股组合层稳健性重估 | pools={pools} | 目标=min(IS1,IS2) | 输出 {out_dir}')

    all_rows, report, finals, t0 = [], [], [], time.time()
    for pool in pools:
        try:
            rows, rep, infos = run_pool(pool, out_dir)
            all_rows.extend(rows)
            report.extend(rep + [''])
            finals.extend(infos)
        except Exception:
            msg = f'## pool={pool} 失败: {traceback.format_exc()}'
            log(msg)
            report.append(msg)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, 'summary.csv'),
                                      index=False, encoding='utf-8-sig')
    if finals:
        ft = pd.DataFrame(finals)
        log('\n===== 汇总(新组合) =====')
        log(ft.to_string(index=False))
        ft.to_csv(os.path.join(out_dir, 'finals.csv'), index=False, encoding='utf-8-sig')
        report.append('## 汇总(新组合)\n' + ft.to_string(index=False))
    report.append(f'\n总耗时 {time.time() - t0:.0f}s | 输出 {out_dir}')
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    log(f'\n完成: {out_dir}')


if __name__ == '__main__':
    main()

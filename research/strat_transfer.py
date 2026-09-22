# -*- coding: utf-8 -*-
"""strat_transfer.py — 方向三: 跨池迁移矩阵(5 池 × 全候选 × 双向).

回答一个问题: 单个候选因子的**方向与强度**能从池迁移到别的池吗? 以及哪些因子是跨池
普适的、哪些只是某个池的特产。

不引入任何新数据, 口径与生产一致(signal_bridge 权重/成本 + bc_backtest 引擎: 信号日收盘
决策 -> 次日开盘执行, 单边 10bps)。三点刻意的选择:

  1. 全候选而非生产预设。方向一/二的 bundle 只存生产预设的**一个**组合分数, 无法覆盖
     "全候选"; 于是本脚本的 bundle 额外落盘每个候选的**截面 rank_pct**(全历史, NaN 保留),
     单成分组合分数在跑之前由 rank_pct 现场还原 —— 因为 _rank_composite({c: w}) 对单成分
     就是 (w/|w|)·rank_pct, 归一化后 ±0.5 与 ±1.0 等价(排序规模完全抵消), 所以"双向"只需
     {+1, -1} 两个方向。此处的口径与 a_weight_grid("A 股版: 全部候选双向") 同源:
     方向未在组合层被确认, 故两个方向都测。

  2. 方向必须用 IS 选、OOS 评。每个候选在每个池上都有 ±1 两条 IS 曲线, 取 IS sharpe 更高的
     方向作为"该池该因子选中的方向", 再在**同一个池**的 OOS 上评价这条选中方向 —— 这就是
     可迁移性的无偏检验(方向选择不含未来信息)。若 OOS 的符号大面积翻转, 说明该因子的方向
     不可迁移(池特有 / 噪声)。

  3. 跨池口径统一。同一 FULL_START/IS_END/OOS_START 三窗、同一 exit_rank=12、gate 常开
     (方向一/二的结论: 体制门无跨池跨窗稳定加分)、同一 top_k 取该池生产预设。于是池间的
     sharpe 差异只来自标的池本身, 可直接横向比较。

并行与内存: 与方向一/二同一 bundle 架构, 但候选 bundle 更大 —— 价格宽表 float64 保真(执行
价不吃浮点损失), 只有 ~58 张 rank_pct 压成 float32(排名口径对精度不敏感)。父进程逐池构建
落盘(公司池数百 MB), 子进程只读; 默认 `--bundle-dir` 可直接复用已落盘的候选 bundle。

输出(out 目录):
  transfer_long.csv          每 (池, 候选, 方向, 窗) 一行(全部统计口径)
  transfer_pool_cand.csv     每 (池, 候选): ±方向的 IS/OOS sharpe + IS 选向后的诚实 OOS
  transfer_matrix_is.csv     池 × 候选 的 IS 最优 sharpe 矩阵(肉眼看池间强弱)
  transfer_matrix_oos.csv    池 × 候选 的"IS 选向后" OOS sharpe 矩阵(迁移性能)
  transfer_cand_summary.csv  每候选跨池汇总 + 标签(universal / mixed / pool_specific)
  transfer_pool_summary.csv  每池汇总(候选整体可迁移性 + 头部/尾部候选 + 等权基准)
  cand_bundle_{pool}.pkl     本池候选 bundle(仅缓存缺失时生成)

用法:
  python -u strat_transfer.py --jobs 12
  python -u strat_transfer.py --cand-dir output/transfer_20260922_XXXXXX   # 复用候选 bundle
  python -u strat_transfer.py --pools etf_3x --jobs 1                      # 单池快速验证
"""
import argparse
import datetime as dt
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strat_common import (PRESETS, base_params, composite, load_bundle, log,  # noqa: E402
                          make_kit, run_bundle, save_bundle, slice_w, stat_row)

FULL_START, IS_END, OOS_START = '2021-01-01', '2024-12-31', '2025-01-01'
WINDOWS = {'IS': (FULL_START, IS_END), 'OOS': (OOS_START, None)}
SIGNS = (1.0, -1.0)                 # 双向; 单成分下 ±0.5 与 ±1.0 等价, 故只留符号
PROD_EXIT = 12                      # 与生产一致
GATE = 'none'                       # 方向一/二: 体制门无稳定加分 -> 常开
Q = 0.3                             # "有意义的正 sharpe" 门槛(跨池计数用)
UNIVERSAL_N = 4                     # >=4/5 池达标 -> 普适
POOL_SPECIFIC_N = 1                 # <=1/5 池达标 -> 池特有

_BUNDLE_CACHE = {}                  # 候选 bundle 体积大, 每进程最多留 2 个池


def _bundle(path: str):
    if path not in _BUNDLE_CACHE:
        if len(_BUNDLE_CACHE) >= 2:
            _BUNDLE_CACHE.pop(next(iter(_BUNDLE_CACHE)))
        _BUNDLE_CACHE[path] = load_bundle(path)
    return _BUNDLE_CACHE[path]


def prod_params(pool: str):
    """生产口径引擎参数: 该池预设 top_k / exit_rank=12 / 其余引擎默认."""
    return base_params().copy(top_k=PRESETS[pool][1] or 5, exit_rank=PROD_EXIT)


def cand_pct(kit, name: str) -> pd.DataFrame:
    """单候选的截面 rank_pct(全历史宽表, NaN 保留).

    与 compute_composite / _rank_composite 对单成分的实现逐位一致:
    wide.rank(axis=1, pct=True); 组合分数 = ±pct.fillna(0.5)。
    """
    if name in kit.panel_full.columns:
        wide = pd.to_numeric(kit.panel_full[name], errors='coerce') \
                     .unstack('symbol').sort_index()
    else:
        cands = kit._get_cands()
        if name not in cands:
            raise KeyError(f'候选不存在(既非 panel 列也非注册信号): {name}')
        wide = cands[name].astype(float)
    return wide.rank(axis=1, pct=True)


def build_cand_bundle(pool: str, names: list = None, checks: int = 2) -> dict:
    """一个池的候选 bundle: 价格宽表(float64 保真) + 每候选 rank_pct(float32).

    前 checks 个候选做口径自检(还原的 ±组合分数 == kit._composite({c: ±1.0}, 'rank'))。
    """
    kit = make_kit(pool)
    cands = kit._get_cands()
    names = [n for n in (names or sorted(cands))]
    ow = kit.open_wide
    pct = {}
    for i, name in enumerate(names):
        raw = cand_pct(kit, name).reindex(index=ow.index, columns=ow.columns)
        if i < checks:
            ref = composite(kit, {name: 1.0})
            got = raw.fillna(0.5).to_numpy(dtype=float)
            if not np.allclose(got, ref.to_numpy(dtype=float), equal_nan=True, atol=1e-9):
                raise AssertionError(f'口径自检失败: {name}(rank_pct 还原 != 单成分组合分数)')
        pct[name] = raw.astype('float32')
    bundle = {'pool': pool, 'names': names,
              'open_wide': kit.open_wide, 'close_wide': kit.close_wide,
              'atr_wide': kit.atr_wide, 'pct': pct}
    del kit
    gc.collect()
    return bundle


def cand_comp(bundle: dict, name: str, sign: float) -> pd.DataFrame:
    """候选方向分数: +1 -> pct.fillna(0.5); -1 -> (-pct).fillna(0.5)(与 _rank_composite 同式)."""
    raw = bundle['pct'][name]
    return raw.fillna(0.5) if sign > 0 else (-raw).fillna(0.5)


def transfer_task(bundle_path: str, pool: str, cand: str) -> list:
    """一个 (池, 候选): ±方向 × IS/OOS 共 4 次回测 -> 4 行统计."""
    b = _bundle(bundle_path)
    p = prod_params(pool)
    rows = []
    for sign in SIGNS:
        comp = cand_comp(b, cand, sign)
        for win in ('IS', 'OOS'):
            ws, we = WINDOWS[win]
            pay = run_bundle(b, f'{pool}|{cand}|{sign:+.0f}|{win}', p, ws, we,
                             gate=GATE, comp=comp)
            rows.append(stat_row(pay, pool=pool, cand=cand, sign=sign, window=win))
    return rows


def bench_task(bundle_path: str, pool: str) -> dict:
    """参考基准: 池内等权收盘收益的 sharpe(判断候选的 OOS sharpe 是否只是池 beta)."""
    b = _bundle(bundle_path)
    cw = b['close_wide']
    out = {'pool': pool}
    for win, (ws, we) in WINDOWS.items():
        r = slice_w(cw, ws, we).pct_change().mean(axis=1).dropna()
        out[f'bh_{win.lower()}_sharpe'] = (float(r.mean() / r.std() * np.sqrt(252))
                                           if len(r) > 2 and r.std() > 0 else np.nan)
        out[f'bh_{win.lower()}_ret'] = float((1.0 + r).prod() - 1.0) if len(r) else np.nan
    return out


def pool_cand_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """每 (池, 候选): ±方向的 IS/OOS sharpe + IS 选向后的诚实 OOS(方向选择不用未来信息)."""
    idx = ['pool', 'cand']
    w = long_df.pivot_table(index=idx, columns=['window', 'sign'], values='sharpe')
    out = w.reset_index()
    out.columns = [f'{a}|{b:+.0f}' if b != '' else a for a, b in out.columns]
    ren = {'IS|+1': 'is_pos', 'IS|-1': 'is_neg', 'OOS|+1': 'oos_pos', 'OOS|-1': 'oos_neg'}
    out = out.rename(columns=ren)
    for c in ('is_pos', 'is_neg', 'oos_pos', 'oos_neg'):
        if c not in out.columns:
            out[c] = np.nan
    out['dir_is_best'] = np.where(out['is_pos'] >= out['is_neg'], 1.0, -1.0)
    out['is_best_sharpe'] = out[['is_pos', 'is_neg']].max(axis=1)
    out['oos_at_is_dir'] = np.where(out['dir_is_best'] > 0, out['oos_pos'], out['oos_neg'])
    out['delta_oos_is'] = out['oos_at_is_dir'] - out['is_best_sharpe']
    out['sign_transfer'] = (out['is_best_sharpe'] > 0) & (out['oos_at_is_dir'] > 0)
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    from scipy.stats import spearmanr
    r = spearmanr(a[m], b[m]).statistic
    return float(r) if np.isfinite(r) else np.nan


def cand_summary(pc: pd.DataFrame) -> pd.DataFrame:
    """每候选跨池汇总 + 标签: 方向是否普适、强度是否迁移、IS 选向是否站得住."""
    out = []
    for cand, g in pc.groupby('cand'):
        pos_n = int((g['oos_pos'] > Q).sum())
        neg_n = int((g['oos_neg'] > Q).sum())
        best_dir = '+' if pos_n >= neg_n else '-'
        best_n = max(pos_n, neg_n)
        if best_n >= UNIVERSAL_N:
            tag = f'universal({best_dir})'
        elif best_n <= POOL_SPECIFIC_N:
            tag = 'pool_specific'
        else:
            tag = 'mixed'
        out.append({
            'cand': cand, 'n_pools': len(g),
            'dir_best': best_dir, 'oos_pos_n': pos_n, 'oos_neg_n': neg_n, 'tag': tag,
            'is_best_mean': g['is_best_sharpe'].mean(),
            'oos_at_is_dir_mean': g['oos_at_is_dir'].mean(),
            'oos_at_is_dir_med': g['oos_at_is_dir'].median(),
            'is_best_std': g['is_best_sharpe'].std(),
            'delta_mean': g['delta_oos_is'].mean(),
            'sign_transfer_rate': g['sign_transfer'].mean(),
            'is_oos_spearman': _spearman(g['is_best_sharpe'], g['oos_at_is_dir']),
            'best_pool': g.loc[g['oos_at_is_dir'].idxmax(), 'pool'],
            'worst_pool': g.loc[g['oos_at_is_dir'].idxmin(), 'pool'],
            'is_dir_agree': float((np.sign(g['is_pos'] - g['is_neg'])
                                   == np.sign(g['oos_pos'] - g['oos_neg'])).mean()),
        })
    df = pd.DataFrame(out)
    return df.sort_values(['tag', 'oos_at_is_dir_mean'], ascending=[True, False]) \
             .reset_index(drop=True)


def pool_summary(pc: pd.DataFrame, bh: pd.DataFrame = None) -> pd.DataFrame:
    """每池汇总: 候选整体可迁移性 + 头部/尾部候选(按 IS 选向后的 OOS) + 等权基准."""
    out = []
    for pool, g in pc.groupby('pool'):
        gg = g.sort_values('oos_at_is_dir', ascending=False)
        head = ';'.join(f'{r.cand}:{r.oos_at_is_dir:.2f}' for r in gg.head(5).itertuples())
        tail = ';'.join(f'{r.cand}:{r.oos_at_is_dir:.2f}' for r in gg.tail(5).itertuples())
        out.append({
            'pool': pool, 'n_cands': len(g),
            'is_best_mean': g['is_best_sharpe'].mean(),
            'is_best_median': g['is_best_sharpe'].median(),
            'oos_at_is_dir_mean': g['oos_at_is_dir'].mean(),
            'oos_at_is_dir_median': g['oos_at_is_dir'].median(),
            'oos_pos_rate': float((g['oos_at_is_dir'] > 0).mean()),
            'sign_transfer_rate': float(g['sign_transfer'].mean()),
            'is_oos_spearman_cands': _spearman(g['is_best_sharpe'], g['oos_at_is_dir']),
            'dir_is_pos_rate': float((g['dir_is_best'] > 0).mean()),
            'head5': head, 'tail5': tail,
        })
    df = pd.DataFrame(out)
    if bh is not None and not bh.empty:
        cols = ['pool'] + [c for c in bh.columns if c.startswith('bh_')]
        df = df.merge(bh[cols], on='pool', how='left')
    return df


def _matrix(pc: pd.DataFrame, value: str, path: str):
    m = pc.pivot_table(index='cand', columns='pool', values=value)
    m.to_csv(path, encoding='utf-8-sig', float_format='%.3f')


def _parallel(fn, arglist: list, jobs: int, label: str) -> list:
    """并行执行(串行回退), 结果扁平化; 单任务失败只告警不中断其余."""
    rows, t0 = [], time.time()

    def _acc(r):
        if r is None:
            return
        rows.extend(r if isinstance(r, list) else [r])

    if jobs > 1 and len(arglist) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        log(f'{label}: {jobs} 进程 / {len(arglist)} 任务')
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(fn, *a): a for a in arglist}
            for fu in as_completed(futs):
                try:
                    _acc(fu.result())
                except Exception as exc:
                    log(f'  [{futs[fu]}] 任务失败: {exc!r}')
    else:
        log(f'{label}: 串行 / {len(arglist)} 任务')
        for a in arglist:
            try:
                _acc(fn(*a))
            except Exception as exc:
                log(f'  [{a}] 任务失败: {exc!r}')
    log(f'{label} 完成 ({time.time() - t0:.0f}s, {len(rows)} 行)')
    return rows


def main():
    ap = argparse.ArgumentParser(description='跨池迁移矩阵: 5 池 × 全候选 × 双向')
    ap.add_argument('--pools', default=None, help='逗号分隔池名(默认全部 5 池)')
    ap.add_argument('--jobs', type=int, default=12, help='并行进程数(1=串行)')
    ap.add_argument('--cands', default=None, help='逗号分隔候选名(默认全部)')
    ap.add_argument('--cand-dir', default=None, help='复用已落盘的候选 bundle 目录')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pools = [p.strip() for p in (args.pools.split(',') if args.pools else list(PRESETS))]
    pools = [p for p in pools if p in PRESETS]
    want = [c.strip() for c in args.cands.split(',')] if args.cands else None
    if args.jobs > 1:
        os.environ['STRAT_SILENT_INIT'] = '1'

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output',
        'transfer_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    log(f'跨池迁移矩阵 | pools={pools} | 双向 {SIGNS} | gate={GATE} '
        f'exit_rank={PROD_EXIT}\n  输出 {out_dir}')

    # 第 1 步: 候选 bundle(可复用)
    paths, cand_names = {}, {}
    for p in pools:
        dst = os.path.join(out_dir, f'cand_bundle_{p}.pkl')
        src = os.path.join(args.cand_dir, f'cand_bundle_{p}.pkl') if args.cand_dir else None
        if src and os.path.exists(src):
            paths[p] = src
            cand_names[p] = load_bundle(src)['names'] if want is None else want
            log(f'  [{p}] 复用候选 bundle {src} '
                f'({os.path.getsize(src) / 2**20:.0f}MB, {len(cand_names[p])} 候选)')
            continue
        t0 = time.time()
        b = build_cand_bundle(p, names=want)
        save_bundle(b, dst)
        paths[p] = dst
        cand_names[p] = b['names']
        log(f'  [{p}] 构建候选 bundle | 标的 {b["open_wide"].shape[1]} | 交易日 '
            f'{b["open_wide"].shape[0]} | top_k {prod_params(p).top_k} | '
            f'{len(b["names"])} 候选 | {os.path.getsize(dst) / 2**20:.0f}MB'
            f' ({time.time() - t0:.0f}s)')
        del b

    # 第 2 步: 迁移矩阵(池 × 候选 × 双向 × IS/OOS)
    log('\n迁移矩阵: 每候选 ±方向 × IS/OOS')
    arglist = [(paths[p], p, c) for p in pools for c in cand_names[p]]
    rows = _parallel(transfer_task, arglist, args.jobs, '迁移回测')
    if not rows:
        log('无有效结果, 退出.')
        return

    long_df = pd.DataFrame(rows)
    long_df.to_csv(os.path.join(out_dir, 'transfer_long.csv'), index=False,
                   encoding='utf-8-sig')
    pc = pool_cand_table(long_df).sort_values(['pool', 'cand']).reset_index(drop=True)
    pc.to_csv(os.path.join(out_dir, 'transfer_pool_cand.csv'), index=False,
              encoding='utf-8-sig')

    # 第 3 步: 矩阵与汇总
    _matrix(pc, 'is_best_sharpe', os.path.join(out_dir, 'transfer_matrix_is.csv'))
    _matrix(pc, 'oos_at_is_dir', os.path.join(out_dir, 'transfer_matrix_oos.csv'))
    cs = cand_summary(pc)
    cs.to_csv(os.path.join(out_dir, 'transfer_cand_summary.csv'), index=False,
              encoding='utf-8-sig')
    bh = pd.DataFrame(_parallel(bench_task, [(paths[p], p) for p in pools],
                                args.jobs, '等权基准'))
    ps = pool_summary(pc, bh)
    ps.to_csv(os.path.join(out_dir, 'transfer_pool_summary.csv'), index=False,
              encoding='utf-8-sig')

    all_r = _spearman(pc['is_best_sharpe'], pc['oos_at_is_dir'])
    n_st = int(pc['sign_transfer'].sum())
    log(f'\n整体: (池,候选) 对 {len(pc)} 个 | IS→OOS sharpe spearman={all_r:.3f} | '
        f'IS 选向后 OOS 同号 {n_st}/{len(pc)}')
    log('\n每池汇总:')
    log(ps.round(3).to_string(index=False))
    log('\n候选标签分布:')
    log(cs['tag'].value_counts().to_string())
    log('\n普适候选(tag=universal)与池特有候选(tag=pool_specific):')
    log(cs[cs['tag'].str.startswith(('universal', 'pool_specific'))]
        [['cand', 'tag', 'oos_pos_n', 'oos_neg_n', 'is_best_mean',
          'oos_at_is_dir_mean', 'sign_transfer_rate', 'is_oos_spearman']]
        .round(3).to_string(index=False))
    log(f'\n完成. 输出目录: {out_dir}')


if __name__ == '__main__':
    main()
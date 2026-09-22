# -*- coding: utf-8 -*-
"""strat_param_scan.py — 方向一: 组合层参数扫描(top_k × exit_rank × gate).

对 5 池的生产胜出权重, 在 IS/OOS 两窗上扫 top_k × exit_rank × 体制门 网格, 回答:
  1) 生产参数(预设 top_k, exit_rank=12, gate 常开)在邻域里是"高原"还是"孤峰";
  2) 换成体制门('breadth20' 截面宽度 / 'vr_trend' 方差比趋势)能否在 OOS 上稳定加分.

口径:
  - 零新数据; 组合分数(全历史 rank 口径)一次算好, 跨窗复用(截面 rank 逐日, 与窗口无关).
  - 时间线由 run_engine 保证(信号日收盘决策 -> 次日开盘执行); 门为因果滚动构造, 无前视.
  - 成本取 EngineParams 默认 10bps 单边, 与生产回测一致; 引擎参数只动 top_k/exit_rank.

并行: 任务粒度 = (池, 门, 窗). 父进程逐池构建"轻型 bundle"(open/close/atr 宽表 + 组合分数,
不含 BacktestKit —— kit 持有 panel_full 与全部候选列, 大池可达数 GB), 落盘后子进程只读 bundle,
故每个子进程内存约数十 MB 而非 GB 级, 并行度不受单池体积限制.

输出(out 目录):
  param_scan_long.csv          全部网格单元(含全指标)
  param_scan_best.csv          每 (pool, window, gate) 的最优参数
  param_scan_prod_vs_best.csv  生产参数 vs 网格最优(sharpe 差)
  param_scan_neighborhood.csv  生产参数邻域稳健性(均值/中位/最优/最差 sharpe)
  heat_{pool}_{window}.csv     sharpe 热力图(index=top_k, columns=(gate, exit_rank))

用法:
  python -u strat_param_scan.py                      # 全五池 IS+OOS
  python -u strat_param_scan.py --pools etf_3x --jobs 1    # 单池串行快速验证
"""
import argparse
import datetime as dt
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strat_common import (PRESETS, base_params, build_bundle, load_bundle,  # noqa: E402
                          log, run_bundle, save_bundle, stat_row, wdesc)

FULL_START, IS_END, OOS_START = '2021-01-01', '2024-12-31', '2025-01-01'
WINDOWS = {'IS': (FULL_START, IS_END), 'OOS': (OOS_START, None), 'FULL': (FULL_START, None)}
TOPK_GRID = [5, 8, 12, 20]
EXIT_GRID = [8, 12, 16, 20]
GATE_GRID = ['none', 'breadth20', 'vr_trend']
DEF_WINDOWS = 'IS,OOS'   # FULL = IS+OOS 交易的拼接, 代价近翻倍, 按需 --windows FULL

_BUNDLE_CACHE = {}       # bundle 路径 -> bundle; 每进程最多留 3 个池


def _parse_ints(text: str) -> list:
    return [int(t) for t in str(text).split(',') if t.strip()]


def _bundle(path: str):
    """子进程内缓存 bundle(每进程最多 3 个池, 超出丢最早的)."""
    if path not in _BUNDLE_CACHE:
        if len(_BUNDLE_CACHE) >= 3:
            _BUNDLE_CACHE.pop(next(iter(_BUNDLE_CACHE)))
        _BUNDLE_CACHE[path] = load_bundle(path)
    return _BUNDLE_CACHE[path]


def scan_task(bundle_path: str, pool: str, gate: str, window: str,
              topk_grid: list, exit_grid: list) -> list:
    """一个 (池, 门, 窗) 任务的全部 (top_k, exit_rank) 单元格."""
    b = _bundle(bundle_path)
    ws, we = WINDOWS[window]
    t0, rows = time.time(), []
    for k in topk_grid:
        for e in exit_grid:
            if e < k:
                continue
            p = base_params().copy(top_k=k, exit_rank=e)
            pay = run_bundle(b, f'{pool}|{gate}|{window}|k{k}e{e}', p, ws, we, gate=gate)
            rows.append(stat_row(pay, pool=pool, window=window, gate=gate,
                                 top_k=k, exit_rank=e))
    log(f'  [{pool}] {gate:>9s} {window:<4s} {len(rows):>3d} 格 ({time.time() - t0:.0f}s)')
    return rows


def heat_table(df: pd.DataFrame, pool: str, window: str, metric: str = 'sharpe') -> pd.DataFrame:
    """index=top_k, columns=(gate, exit_rank), values=metric."""
    sub = df[(df['pool'] == pool) & (df['window'] == window)]
    t = sub.pivot_table(index='top_k', columns=['gate', 'exit_rank'],
                        values=metric, aggfunc='mean')
    have_g = set(t.columns.get_level_values('gate'))
    gates = [g for g in GATE_GRID if g in have_g]
    exits = sorted(set(int(c) for c in t.columns.get_level_values('exit_rank')))
    return t.reindex(columns=pd.MultiIndex.from_product([gates, exits],
                                                      names=['gate', 'exit_rank']))


def best_table(df: pd.DataFrame) -> pd.DataFrame:
    """每 (pool, window, gate) 的最优 sharpe 行."""
    idx = df.groupby(['pool', 'window', 'gate'])['sharpe'].idxmax()
    cols = ['pool', 'window', 'gate', 'top_k', 'exit_rank', 'sharpe', 'total_ret',
            'max_dd', 'calmar', 'vol', 'ann_turnover', 'n_trades', 'win_rate']
    return df.loc[idx, cols].sort_values(['pool', 'window', 'gate']).reset_index(drop=True)


def prod_vs_best(df: pd.DataFrame) -> pd.DataFrame:
    """生产参数(预设 top_k / exit_rank=12 / gate=none) 与网格最优的对照."""
    best = best_table(df)
    out = []
    for (pool, window), grp in df.groupby(['pool', 'window']):
        prod = grp[(grp['gate'] == 'none') & (grp['exit_rank'] == 12) &
                   (grp['top_k'] == (PRESETS[pool][1] or 5))]
        if prod.empty:      # 预设 top_k 不在网格内时退化为同 exit_rank 下最优
            prod = grp[(grp['gate'] == 'none') & (grp['exit_rank'] == 12)]
        if prod.empty:
            continue
        prod = prod.loc[prod['sharpe'].idxmax()]
        cand = best[(best['pool'] == pool) & (best['window'] == window)]
        cand = cand.loc[cand['sharpe'].idxmax()]
        out.append({'pool': pool, 'window': window,
                    'prod_top_k': int(prod['top_k']), 'prod_sharpe': prod['sharpe'],
                    'prod_total_ret': prod['total_ret'],
                    'best_gate': cand['gate'], 'best_top_k': int(cand['top_k']),
                    'best_exit_rank': int(cand['exit_rank']),
                    'best_sharpe': cand['sharpe'], 'best_total_ret': cand['total_ret'],
                    'delta_sharpe': round(float(cand['sharpe'] - prod['sharpe']), 3)})
    return pd.DataFrame(out)


def neighborhood(df: pd.DataFrame, pool: str, prod_topk: int) -> list:
    """生产参数(top_k=prod, exit_rank=12, gate=none)邻域的 sharpe 分布: 孤峰 or 高原.
    按窗分别统计(IS 与 OOS 不可混算)."""
    out = []
    for window in sorted(df['window'].unique()):
        sub = df[(df['pool'] == pool) & (df['window'] == window) & (df['gate'] == 'none') &
                 (df['exit_rank'].isin([8, 12, 16, 20])) &
                 (df['top_k'].between(prod_topk - 3, prod_topk + 3))]
        if sub.empty:
            continue
        s = sub['sharpe'].dropna()
        core = sub[(sub['top_k'] == prod_topk) & (sub['exit_rank'] == 12)]['sharpe']
        out.append({'pool': pool, 'window': window, 'prod_top_k': prod_topk,
                    'n_cells': len(s),
                    'prod_sharpe': float(core.iloc[0]) if len(core) else float('nan'),
                    'nbr_mean': round(float(s.mean()), 3),
                    'nbr_median': round(float(s.median()), 3),
                    'nbr_min': round(float(s.min()), 3), 'nbr_max': round(float(s.max()), 3),
                    'nbr_sign_flip': int((s <= 0).sum())})
    return out


def main():
    ap = argparse.ArgumentParser(description='组合层参数扫描: top_k × exit_rank × gate')
    ap.add_argument('--pools', default=None, help='逗号分隔池名(默认全部 5 池)')
    ap.add_argument('--windows', default=DEF_WINDOWS, help='IS,OOS,FULL 的任意组合')
    ap.add_argument('--gates', default=','.join(GATE_GRID))
    ap.add_argument('--topk', default=','.join(map(str, TOPK_GRID)))
    ap.add_argument('--exit', default=','.join(map(str, EXIT_GRID)))
    ap.add_argument('--jobs', type=int, default=12, help='并行进程数(1=串行)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pools = [p.strip() for p in (args.pools.split(',') if args.pools else list(PRESETS))]
    pools = [p for p in pools if p in PRESETS]
    windows = [w.strip() for w in args.windows.split(',') if w.strip() in WINDOWS]
    gates = [g.strip() for g in args.gates.split(',')]
    topk_grid, exit_grid = _parse_ints(args.topk), _parse_ints(args.exit)

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output',
        'param_scan_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    log(f'组合层参数扫描 | pools={pools} | windows={windows} | gates={gates}\n'
        f'  top_k={topk_grid} exit_rank={exit_grid}\n  输出 {out_dir}')

    tasks = [(p, g, w) for p in pools for g in gates for w in windows]
    rows, t0 = [], time.time()

    log(f'\n第 1 步: 逐池构建轻型 bundle({len(pools)} 池, 串行 —— 父进程只留一份 kit)')
    paths = {}
    for p in pools:
        tp = time.time()
        b = build_bundle(p)
        path = os.path.join(out_dir, f'bundle_{p}.pkl')
        save_bundle(b, path)
        paths[p] = path
        log(f'  [{p}] 标的 {b["open_wide"].shape[1]} | 交易日 {b["open_wide"].shape[0]} | '
            f'权重 {wdesc(PRESETS[p][0])} | 落盘 {os.path.getsize(path) / 2**20:.0f}MB '
            f'({time.time() - tp:.0f}s)')
        del b

    jobs = max(1, min(args.jobs, len(tasks)))
    if jobs > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        os.environ['STRAT_SILENT_INIT'] = '1'   # 子进程不重复打印环境/预设日志
        log(f'\n第 2 步: 并行扫描 {jobs} 进程 / {len(tasks)} 任务(任务粒度 = 池×门×窗)')
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(scan_task, paths[p], p, g, w, topk_grid, exit_grid): (p, g, w)
                    for p, g, w in tasks}
            for fu in as_completed(futs):
                try:
                    rows.extend(fu.result())
                except Exception as exc:
                    log(f'[{futs[fu]}] 任务失败: {exc!r}')
    else:
        log(f'\n第 2 步: 串行扫描 {len(tasks)} 任务')
        for p, g, w in tasks:
            try:
                rows.extend(scan_task(paths[p], p, g, w, topk_grid, exit_grid))
            except Exception as exc:
                log(f'[{(p, g, w)}] 任务失败: {exc!r}')

    if not rows:
        log('无结果, 退出')
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'param_scan_long.csv'), index=False, encoding='utf-8-sig')
    log(f'\n长表已写: param_scan_long.csv ({len(df)} 行, 总耗时 {time.time() - t0:.0f}s)')

    best = best_table(df)
    best.to_csv(os.path.join(out_dir, 'param_scan_best.csv'), index=False, encoding='utf-8-sig')
    log('\n最优参数表:')
    log(best.to_string(index=False))

    pvb = prod_vs_best(df)
    pvb.to_csv(os.path.join(out_dir, 'param_scan_prod_vs_best.csv'),
               index=False, encoding='utf-8-sig')
    log('\n生产参数 vs 网格最优:')
    log(pvb.to_string(index=False))

    nbr = pd.DataFrame([r for p in pools if p in PRESETS
                        for r in neighborhood(df, p, PRESETS[p][1] or 5)])
    nbr.to_csv(os.path.join(out_dir, 'param_scan_neighborhood.csv'),
               index=False, encoding='utf-8-sig')
    log('\n生产参数邻域稳健性(|Δtop_k|<=3, exit_rank∈{8,12,16,20}, gate=none, 分窗):')
    log(nbr.to_string(index=False))

    for pool in pools:
        for w in windows:
            t = heat_table(df, pool, w)
            if t.dropna(how='all').empty:
                continue
            t.round(3).to_csv(os.path.join(out_dir, f'heat_{pool}_{w}.csv'), encoding='utf-8-sig')
            log(f'\n[{pool} / {w}] sharpe 热力图 (index=top_k, columns=gate×exit_rank):')
            log(t.round(2).to_string())

    log(f'\n完成. 输出目录: {out_dir}')


if __name__ == '__main__':
    main()
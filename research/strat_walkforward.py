# -*- coding: utf-8 -*-
"""strat_walkforward.py — 方向二: 稳健性治理(滚动 walk-forward + sign-flip + 多重检验).

不引入任何新数据, 口径与生产一致(signal_bridge 权重 + bc_backtest 引擎: 信号日收盘决策 ->
次日开盘执行, 单边成本 10bps)。三件事回答三个不同的问题:

  A. 滚动 walk-forward —— "选参数"这件事本身有效率吗?
     锚定起点、测试段逐段后移(默认训练 >=504 日, 测试段 126 日)。每个 fold 只在训练窗上
     网格选参(取 train sharpe 最优), 再在紧随其后的测试窗上评价; 同时用生产固定参数
     (预设 top_k, exit_rank=12, gate 常开)在**同一个测试窗**上跑对照。测试段全部为样本外,
     故"选中参数 vs 生产参数"的差就是参数选择的无偏增量(损失/增益)。

  B. sign-flip 置换对照 —— 分数真的携带时序信息吗?
     把组合分数整张截面按交易日置换(shuffled_composite), 保留截面内排序结构、破坏与日期的
     对齐, 从而构造"无预测力"零分布。生产 sharpe 在零分布中的分位/z 值 + 置换经验 p 值
     (单侧)即为检验结果。若生产 sharpe 落在零分布内, 说明业绩可由截面结构本身解释。

  C. 多重检验(DSR) —— 扫了多少格要扣多少分?
     Deflated Sharpe Ratio(Bailey & López de Prado): 用同窗网格全部试验的逐期 sharpe 方差
     Var(SR_n) 与试验基数 N, 算出"纯靠搜索能得到的最好 sharpe"的期望上界 SR0, 再把生产
     sharpe 相对 SR0 的显著性打成 DSR 概率。N 同时给池内(单池网格格数)与全局(跨池合计)
     两种口径, 后者更保守。

并行与内存: 与 strat_param_scan 同一 bundle 架构 —— 父进程逐池构建轻型 bundle
(open/close/atr 宽表 + 组合分数, 不含 BacktestKit)落盘, 子进程只读 bundle,
每进程内存数十 MB。默认 `--bundle-dir` 可复用方向一已落盘的 bundle, 省去重建。

输出(out 目录):
  wf_folds.csv         每个 (池, fold) 的训练窗选参 / 测试窗评价(选中 vs 生产)
  wf_summary.csv       每池跨 fold 汇总(选中参数是否稳定胜出、门/参数选择分布)
  signflip.csv         每 (池, 窗) 的置换零分布统计与 p 值
  dsr.csv              每 (池, 窗) 的 DSR(池内 N / 全局 N 两种口径)
  bundle_{pool}.pkl    本池轻型 bundle(仅在无 --bundle-dir 或缓存缺失时生成)

用法:
  python -u strat_walkforward.py --bundle-dir output/param_scan_20260921_234455
  python -u strat_walkforward.py --pools etf_3x --jobs 1 --seeds 5     # 单池快速验证
"""
import argparse
import datetime as dt
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strat_common import (PRESETS, base_params, build_bundle, daily_ret,  # noqa: E402
                          load_bundle, log, run_bundle, save_bundle,
                          sharpe_per, slice_w, wdesc)
from quant.bc_backtest import shuffled_composite  # noqa: E402

FULL_START, IS_END, OOS_START = '2021-01-01', '2024-12-31', '2025-01-01'
WINDOWS = {'IS': (FULL_START, IS_END), 'OOS': (OOS_START, None)}
TOPK_GRID = [5, 8, 12, 20]
EXIT_GRID = [8, 12, 16, 20]
GATE_GRID = ['none', 'breadth20', 'vr_trend']
PROD_EXIT = 12
GAMMA = 0.5772156649015329          # Euler-Mascheroni, DSR 的 SR0 期望项

_BUNDLE_CACHE = {}                  # bundle 路径 -> bundle; 每进程最多留 3 个池


def _parse_ints(text: str) -> list:
    return [int(t) for t in str(text).split(',') if t.strip()]


def _bundle(path: str):
    if path not in _BUNDLE_CACHE:
        if len(_BUNDLE_CACHE) >= 3:
            _BUNDLE_CACHE.pop(next(iter(_BUNDLE_CACHE)))
        _BUNDLE_CACHE[path] = load_bundle(path)
    return _BUNDLE_CACHE[path]


def grid_cells(topk_grid: list, exit_grid: list, gate_grid: list) -> list:
    """(gate, top_k, exit_rank) 组合; exit_rank < top_k 非法(滞回带不能倒挂)."""
    return [(g, k, e) for g in gate_grid for k in topk_grid for e in exit_grid if e >= k]


def prod_params(pool: str):
    """生产口径引擎参数: 预设 top_k / exit_rank=12 / gate 常开 / 其余引擎默认."""
    return base_params().copy(top_k=PRESETS[pool][1] or 5, exit_rank=PROD_EXIT)


def make_folds(index: pd.DatetimeIndex, full_start: str, min_train: int, step: int) -> list:
    """锚定式滚动折: train=[起点, t), test=[t, t+step), t 从 min_train 起按 step 后移."""
    d = index[index >= pd.Timestamp(full_start)]
    if len(d) <= min_train:
        raise ValueError(f'可用交易日不足: {len(d)} <= min_train {min_train}')
    folds, t, i = [], min_train, 0
    while t + step <= len(d):
        tr, te = d[:t], d[t:t + step]
        folds.append({'fold': i, 'train_start': str(tr[0].date()), 'train_end': str(tr[-1].date()),
                      'test_start': str(te[0].date()), 'test_end': str(te[-1].date()),
                      'n_train': len(tr), 'n_test': len(te)})
        t, i = t + step, i + 1
    return folds


def wf_task(bundle_path: str, pool: str, fold: dict, topk_grid: list,
            exit_grid: list, gate_grid: list) -> dict:
    """一个 (池, fold): 训练窗网格选参 -> 测试窗评价选中参数与生产参数."""
    b = _bundle(bundle_path)
    best = None
    for g, k, e in grid_cells(topk_grid, exit_grid, gate_grid):
        pay = run_bundle(b, f'{pool}|train', base_params().copy(top_k=k, exit_rank=e),
                         fold['train_start'], fold['train_end'], gate=g)
        s = pay['stats'].get('sharpe')
        if s is not None and (best is None or s > best[0]):
            best = (s, g, k, e)
    pp = prod_params(pool)
    prod = run_bundle(b, f'{pool}|prod', pp, fold['test_start'], fold['test_end'], gate='none')
    row = {**fold, 'pool': pool, 'prod_top_k': pp.top_k,
           'prod_test_sharpe': prod['stats'].get('sharpe'),
           'prod_test_ret': prod['stats'].get('total_ret'),
           'prod_test_dd': prod['stats'].get('max_dd')}
    if best is None:
        log(f'  [{pool}] fold{fold["fold"]} 训练窗无有效网格结果')
        return row
    s, g, k, e = best
    sel = run_bundle(b, f'{pool}|sel', base_params().copy(top_k=k, exit_rank=e),
                     fold['test_start'], fold['test_end'], gate=g)
    ss, ps = sel['stats'].get('sharpe'), row['prod_test_sharpe']
    row.update({'sel_gate': g, 'sel_top_k': k, 'sel_exit_rank': e,
                'sel_train_sharpe': s,
                'sel_test_sharpe': ss, 'sel_test_ret': sel['stats'].get('total_ret'),
                'sel_test_dd': sel['stats'].get('max_dd'),
                'sel_test_trades': sel['stats'].get('n_trades'),
                'delta_test_sharpe': (round(float(ss - ps), 3)
                                      if ss is not None and ps is not None else np.nan),
                'sel_wins': bool(ss is not None and ps is not None and ss > ps)})
    log(f'  [{pool}] fold{fold["fold"]:>2d} test {fold["test_start"]}~{fold["test_end"]}: '
        f'选{g}/k{k}/e{e} train{s:.2f} -> test{ss} | 生产 k{pp.top_k} test{ps}')
    return row


def wf_summary(df: pd.DataFrame, pools: list) -> pd.DataFrame:
    """每池汇总: 选中参数与生产参数在测试窗的表现分布 + 选择偏好.

    sel_nan_folds 为选中参数在测试窗算不出 sharpe 的折数(多为选中的门在该窗常关 -> 空仓),
    这些折不计入均值, 但计入 n_folds."""
    out = []
    for pool in pools:
        allf = df[df['pool'] == pool]
        sub = allf.dropna(subset=['sel_test_sharpe', 'prod_test_sharpe'])
        if sub.empty:
            continue
        out.append({
            'pool': pool, 'n_folds': len(allf), 'n_eval': len(sub),
            'sel_nan_folds': int(len(allf) - len(sub)),
            'sel_test_sharpe_mean': round(float(sub['sel_test_sharpe'].mean()), 3),
            'sel_test_sharpe_median': round(float(sub['sel_test_sharpe'].median()), 3),
            'prod_test_sharpe_mean': round(float(sub['prod_test_sharpe'].mean()), 3),
            'prod_test_sharpe_median': round(float(sub['prod_test_sharpe'].median()), 3),
            'delta_mean': round(float(sub['delta_test_sharpe'].mean()), 3),
            'sel_wins_rate': round(float(sub['sel_wins'].mean()), 3),
            'sel_gate_dist': ','.join(f'{g}:{n}' for g, n in
                                      sub['sel_gate'].value_counts().items()),
            'sel_top_k_dist': ','.join(f'{int(k)}:{n}' for k, n in
                                       sub['sel_top_k'].value_counts().items()),
            'sel_exit_dist': ','.join(f'{int(e)}:{n}' for e, n in
                                      sub['sel_exit_rank'].value_counts().items())})
    return pd.DataFrame(out)


def signflip_task(bundle_path: str, pool: str, window: str, seeds: list) -> dict:
    """一个 (池, 窗): 生产分数 sharpe vs 置换零分布(逐 seed 置换整张截面)."""
    b = _bundle(bundle_path)
    ws, we = WINDOWS[window]
    p = prod_params(pool)
    prod = run_bundle(b, f'{pool}|{window}|prod', p, ws, we, gate='none')
    sr = sharpe_per(prod)
    null = []
    for seed in seeds:
        pay = run_bundle(b, f'{pool}|{window}|null{seed}', p, ws, we, gate='none',
                         comp=shuffled_composite(b['composite'], seed))
        null.append(sharpe_per(pay))
    arr = np.array([x for x in null if np.isfinite(x)])
    if len(arr) < 2 or arr.std() == 0:
        return {'pool': pool, 'window': window, 'prod_sharpe': prod['stats'].get('sharpe'),
                'prod_sr_per': sr, 'n_seeds': len(arr)}
    ge = int((arr >= sr).sum())
    row = {'pool': pool, 'window': window, 'prod_sharpe': prod['stats'].get('sharpe'),
           'prod_sr_per': round(sr, 4), 'n_seeds': len(arr),
           'null_mean': round(float(arr.mean()), 4), 'null_std': round(float(arr.std(ddof=1)), 4),
           'null_min': round(float(arr.min()), 4), 'null_p95': round(float(np.percentile(arr, 95)), 4),
           'null_max': round(float(arr.max()), 4),
           'z': round(float((sr - arr.mean()) / arr.std(ddof=1)), 3),
           'pct_rank': round(float((arr < sr).mean()), 3),
           'p_value': round(float((ge + 1) / (len(arr) + 1)), 4)}
    log(f'  [{pool}/{window}] prod SR/T={sr:.4f} | null mean={arr.mean():.4f} '
        f'sd={arr.std(ddof=1):.4f} max={arr.max():.4f} | z={row["z"]} p={row["p_value"]}')
    return row


def dsr_task(bundle_path: str, pool: str, window: str, topk_grid: list,
             exit_grid: list, gate_grid: list) -> dict:
    """一个 (池, 窗): 收集网格全部试验的逐期 sharpe(供跨池方差)与生产配置矩量."""
    b = _bundle(bundle_path)
    ws, we = WINDOWS[window]
    trials = []
    for g, k, e in grid_cells(topk_grid, exit_grid, gate_grid):
        pay = run_bundle(b, f'{pool}|{window}|{g}{k}e{e}',
                         base_params().copy(top_k=k, exit_rank=e), ws, we, gate=g)
        trials.append({'gate': g, 'top_k': k, 'exit_rank': e,
                       'sharpe': pay['stats'].get('sharpe'), 'sr_per': sharpe_per(pay)})
    pp = prod_params(pool)
    prod = run_bundle(b, f'{pool}|{window}|prod', pp, ws, we, gate='none')
    r = daily_ret(prod)
    return {'pool': pool, 'window': window, 'prod_top_k': pp.top_k,
            'prod_exit_rank': pp.exit_rank,
            'prod_sharpe': prod['stats'].get('sharpe'), 'prod_sr_per': sharpe_per(prod),
            'T': int(len(r)),
            'skew': float(r.skew()) if len(r) > 2 else np.nan,
            'kurt': float(r.kurt() + 3.0) if len(r) > 3 else np.nan,
            'trials_sr': [t['sr_per'] for t in trials if np.isfinite(t['sr_per'])],
            'trials': trials}


def deflated_sharpe(sr: float, trials_sr: list, n_trials: int,
                    T: int, skew: float, kurt: float) -> dict:
    """DSR: 相对"搜索能得到的最好 sharpe 期望上界 SR0"的显著性(正态尾部概率)."""
    from scipy.stats import norm
    sd = float(np.std(trials_sr, ddof=1)) if len(trials_sr) > 1 else 0.0
    sr0 = sd * ((1.0 - GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
                + GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    var_term = max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12)
    z = (sr - sr0) * np.sqrt(max(T - 1, 1)) / np.sqrt(var_term)
    z_naive = sr * np.sqrt(max(T - 1, 1)) / np.sqrt(var_term)
    return {'n_trials': int(n_trials), 'sr_std': round(sd, 4), 'sr0': round(sr0, 4),
            'z': round(float(z), 3), 'dsr': round(float(norm.cdf(z)), 4),
            'p_naive': round(float(1.0 - norm.cdf(z_naive)), 4)}


def _parallel(fn, arglist: list, jobs: int, label: str) -> list:
    """按任务列表并行执行(串行回退), 失败只告警不中断其余任务."""
    rows, t0 = [], time.time()
    if jobs > 1 and len(arglist) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        log(f'{label}: {jobs} 进程 / {len(arglist)} 任务')
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(fn, *a): a for a in arglist}
            for fu in as_completed(futs):
                try:
                    rows.append(fu.result())
                except Exception as exc:
                    log(f'  [{futs[fu]}] 任务失败: {exc!r}')
    else:
        log(f'{label}: 串行 / {len(arglist)} 任务')
        for a in arglist:
            try:
                rows.append(fn(*a))
            except Exception as exc:
                log(f'  [{a}] 任务失败: {exc!r}')
    log(f'{label} 完成 ({time.time() - t0:.0f}s)')
    return [r for r in rows if r is not None]


def main():
    ap = argparse.ArgumentParser(description='稳健性治理: walk-forward + sign-flip + DSR')
    ap.add_argument('--pools', default=None, help='逗号分隔池名(默认全部 5 池)')
    ap.add_argument('--jobs', type=int, default=12, help='并行进程数(1=串行)')
    ap.add_argument('--seeds', type=int, default=30, help='置换对照的 seed 数(默认 30)')
    ap.add_argument('--min-train', type=int, default=504, help='walk-forward 首折训练日数')
    ap.add_argument('--step', type=int, default=126, help='walk-forward 测试段长度(交易日)')
    ap.add_argument('--bundle-dir', default=None, help='复用已有 bundle 目录(如方向一输出目录)')
    ap.add_argument('--skip', default='', help='跳过的阶段, 逗号分隔(wf,signflip,dsr)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pools = [p.strip() for p in (args.pools.split(',') if args.pools else list(PRESETS))]
    pools = [p for p in pools if p in PRESETS]
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}
    seeds = list(range(1, args.seeds + 1))
    if args.jobs > 1:
        os.environ['STRAT_SILENT_INIT'] = '1'   # 子进程不重复打印预设/环境日志

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output',
        'walkforward_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    log(f'稳健性治理 | pools={pools} | seeds={len(seeds)} | '
        f'min_train={args.min_train} step={args.step}\n  输出 {out_dir}')

    # 第 1 步: bundle(可复用方向一落盘产物)
    paths = {}
    for p in pools:
        dst = os.path.join(out_dir, f'bundle_{p}.pkl')
        src = os.path.join(args.bundle_dir, f'bundle_{p}.pkl') if args.bundle_dir else None
        if src and os.path.exists(src):
            paths[p] = src
            log(f'  [{p}] 复用 bundle {src} ({os.path.getsize(src) / 2**20:.0f}MB)')
            continue
        t0 = time.time()
        b = build_bundle(p)
        save_bundle(b, dst)
        paths[p] = dst
        log(f'  [{p}] 构建 bundle | 标的 {b["open_wide"].shape[1]} | 交易日 '
            f'{b["open_wide"].shape[0]} | 权重 {wdesc(PRESETS[p][0])} | '
            f'{os.path.getsize(dst) / 2**20:.0f}MB ({time.time() - t0:.0f}s)')
        del b

    df_wf = pd.DataFrame()
    if 'wf' not in skip:
        log('\n阶段 A: 滚动 walk-forward(训练窗选参 -> 测试窗样本外评价)')
        arglist = []
        for p in pools:
            idx = _bundle(paths[p])['open_wide'].index
            for fold in make_folds(idx, FULL_START, args.min_train, args.step):
                arglist.append((paths[p], p, fold, TOPK_GRID, EXIT_GRID, GATE_GRID))
        rows = _parallel(wf_task, arglist, args.jobs, 'walk-forward')
        if rows:
            df_wf = pd.DataFrame(rows).sort_values(['pool', 'fold']).reset_index(drop=True)
            df_wf.to_csv(os.path.join(out_dir, 'wf_folds.csv'), index=False,
                         encoding='utf-8-sig')
            sm = wf_summary(df_wf, pools)
            sm.to_csv(os.path.join(out_dir, 'wf_summary.csv'), index=False,
                      encoding='utf-8-sig')
            log('\nwalk-forward 汇总(测试窗 sharpe: 选中参数 vs 生产参数):')
            log(sm.to_string(index=False))

    df_sf = pd.DataFrame()
    if 'signflip' not in skip:
        log('\n阶段 B: sign-flip 置换对照(整张截面按交易日置换, 构造无预测力零分布)')
        arglist = [(paths[p], p, w, seeds) for p in pools for w in WINDOWS]
        rows = _parallel(signflip_task, arglist, args.jobs, 'sign-flip')
        if rows:
            df_sf = pd.DataFrame(rows).sort_values(['pool', 'window']).reset_index(drop=True)
            df_sf.to_csv(os.path.join(out_dir, 'signflip.csv'), index=False,
                         encoding='utf-8-sig')
            log('\nsign-flip 结果(prod = 生产配置, 其余为置换零分布):')
            log(df_sf.to_string(index=False))

    if 'dsr' not in skip:
        log('\n阶段 C: 多重检验(DSR) —— 收集同窗网格全部试验的逐期 sharpe')
        arglist = [(paths[p], p, w, TOPK_GRID, EXIT_GRID, GATE_GRID)
                   for p in pools for w in WINDOWS]
        rows = _parallel(dsr_task, arglist, args.jobs, 'DSR 采样')
        if rows:
            all_sr = [x for r in rows for x in r['trials_sr']]
            n_all = len(all_sr)
            out = []
            for r in rows:                       # 池内 N 口径
                d = deflated_sharpe(r['prod_sr_per'], r['trials_sr'], len(r['trials_sr']),
                                    r['T'], r['skew'], r['kurt'])
                d.update({'pool': r['pool'], 'window': r['window'],
                          'prod_top_k': r['prod_top_k'], 'prod_sharpe': r['prod_sharpe'],
                          'prod_sr_per': round(r['prod_sr_per'], 4), 'T': r['T'],
                          'skew': round(r['skew'], 3), 'kurt': round(r['kurt'], 3),
                          'scope': 'pool'})
                out.append(d)
            for r in rows:                       # 全局 N 口径(跨池搜索同样计入)
                d = deflated_sharpe(r['prod_sr_per'], all_sr, n_all,
                                    r['T'], r['skew'], r['kurt'])
                d.update({'pool': r['pool'], 'window': r['window'],
                          'prod_top_k': r['prod_top_k'], 'prod_sharpe': r['prod_sharpe'],
                          'prod_sr_per': round(r['prod_sr_per'], 4), 'T': r['T'],
                          'skew': round(r['skew'], 3), 'kurt': round(r['kurt'], 3),
                          'scope': 'global'})
                out.append(d)
            df_dsr = pd.DataFrame(out)[['pool', 'window', 'scope', 'prod_top_k',
                                        'prod_sharpe', 'prod_sr_per', 'T', 'skew', 'kurt',
                                        'n_trials', 'sr_std', 'sr0', 'z', 'dsr', 'p_naive']]
            df_dsr = df_dsr.sort_values(['pool', 'window', 'scope']).reset_index(drop=True)
            df_dsr.to_csv(os.path.join(out_dir, 'dsr.csv'), index=False, encoding='utf-8-sig')
            log(f'\nDSR(全局试验基数 N={n_all}; pool 口径 N=单池网格格数, global 口径 N={n_all}):')
            log(df_dsr.to_string(index=False))

    log(f'\n完成. 输出目录: {out_dir}')


if __name__ == '__main__':
    main()
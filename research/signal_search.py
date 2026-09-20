# -*- coding: utf-8 -*-
"""
signal_search.py — 独立只读研究模块: 全列信号筛查(面向可交易超额收益)

定位: 补 factor_research.py(仅测 18 个内置因子) 与 score_backtest.py(仅回测给定因子组合) 之间
的空白 —— 系统性筛查 panel 全部数值列 + 项目里不存在的经典信号(动量/波动/反转/量能),
按"可交易超额收益"排序, 为组合回测提供候选池.

关键口径(与 factor_research 的 close->close IC 不同, 这里用真实可执行口径):
  信号日 t 收盘已知信号值 -> t+1 开盘入场 -> 持有 h 日 -> t+1+h 开盘出场
  前瞻收益 = Open[t+1+h] / Open[t+1] - 1
  超额基准 = 同池等权持有(buyhold_pool) 的同期收益, 即 excess_k = topk_ret - pool_ret
  (long-only 策略的真实超额来源就是这个差值, 而非多空价差)

输出: research/output/{pool}_sig_{run_id}/screen_h{h}.csv + report.txt

用法:
  python signal_search.py --pool etf_3x --pkl-dir .../research/data --start 2021-01-01
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 疑似"标签/策略动作"类列: 可能由未来信息生成, 单独分组并显著标记, 不可直接当信号使用
SUSPECT_PAT = ('label', 'action', 'signal', 'pos_label', 'neg_label', '_day')


# ================================================================ 衍生信号 ================================================================ #
def build_derived(panel: pd.DataFrame) -> dict:
    """构造项目 panel 中不存在的经典截面信号(全部只用 t 及之前的数据, 无前视)."""
    close = panel['Close'].unstack('symbol').sort_index()
    volume = panel['Volume'].unstack('symbol').sort_index()
    high = panel['High'].unstack('symbol').sort_index()
    low = panel['Low'].unstack('symbol').sort_index()
    ret1 = close.pct_change()
    out = {}

    # 动量族: 不同回看期(3x杠杆ETF因杠杆重置有强路径依赖, 动量窗口敏感)
    for n in (5, 10, 20, 60, 120, 250):
        out[f'D_mom{n}'] = close.pct_change(n)
    # 反转族(短周期动量的反面)
    for n in (1, 3, 5):
        out[f'D_rev{n}'] = -close.pct_change(n)
    # 波动族(低波动异象; 3xETF中高波动=高衰减成本, 预期取负权)
    for n in (10, 20, 60):
        out[f'D_vol{n}'] = ret1.rolling(n).std()
    # 波动调整动量(动量/波动, 提升可比性)
    for n in (20, 60, 120):
        out[f'D_sharpe{n}'] = close.pct_change(n) / (ret1.rolling(n).std() * np.sqrt(n))
    # 均线距离 / 趋势位置
    for n in (20, 60, 200):
        out[f'D_ma{n}_dist'] = close / close.rolling(n).mean() - 1.0
    # 距一年高点(回撤深度; 反弹候选 vs 强者恒强候选)
    out['D_dd_250'] = close / close.rolling(250, min_periods=60).max() - 1.0
    # 量能: 20日量比与量能z-score
    out['D_volratio20'] = volume / volume.rolling(20).mean()
    v20 = volume.rolling(20)
    out['D_volz20'] = (volume - v20.mean()) / v20.std()
    # 日内波动占价格比(ATR 简化: 真实波幅均值/收盘)
    tr = pd.DataFrame(
        np.maximum.reduce([(high - low).values,
                           (high - close.shift(1)).abs().values,
                           (low - close.shift(1)).abs().values]),
        index=close.index, columns=close.columns)
    out['D_atr_pct20'] = tr.rolling(20).mean() / close
    # 偏度(收益分布形态)
    out['D_skew60'] = ret1.rolling(60).skew()
    return {k: v for k, v in out.items() if isinstance(v, pd.DataFrame)}


# ================================================================ 筛查核心 ================================================================ #
def tradable_fwd(open_wide: pd.DataFrame, h: int) -> pd.DataFrame:
    """可交易前瞻收益: 信号日 t 收盘 -> t+1 开盘入场 -> t+1+h 开盘出场."""
    entry = open_wide.shift(-1)
    exit_ = open_wide.shift(-(1 + h))
    return exit_ / entry - 1.0


def screen_one(sig: pd.DataFrame, fwd: pd.DataFrame, top_k: int = 5, min_cs: int = 10) -> dict:
    """单信号筛查(全向量化): 截面 Spearman IC / top-k 收益 / 超额(相对等权池) / 多空价差 / top-k 换手.

    与逐日循环等价, 但按整行做秩相关: IC_row = cov(rank_s, rank_f) / (std_s * std_f), NaN 成对剔除.
    """
    idx = sig.index.intersection(fwd.index)
    if len(idx) < 60:
        return {}
    cols = sig.columns.intersection(fwd.columns)
    s = sig.loc[idx, cols]
    r = fwd.loc[idx, cols]
    mask = s.notna().values & r.notna().values
    cnt = mask.sum(axis=1)
    ok = cnt >= min_cs
    if ok.sum() < 60:
        return {}
    sv = s.values.astype(float)
    rv = r.values.astype(float)
    m = mask & ok[:, None]
    sv_m = np.where(m, sv, np.nan)
    rv_m = np.where(m, rv, np.nan)
    # 行内秩(仅对被保留样本; 用 nan 感知的 double-argsort 秩)
    def row_rank(a):
        order = np.argsort(np.where(np.isnan(a), np.inf, a), axis=1, kind='stable')
        ranks = np.empty(a.shape, dtype=float)
        rows = np.arange(a.shape[0])[:, None]
        ranks[rows, order] = np.arange(a.shape[1])[None, :]
        return np.where(np.isnan(a), np.nan, ranks)
    rs, rr = row_rank(sv_m), row_rank(rv_m)
    sm = np.nanmean(rs, axis=1, keepdims=True)
    rm = np.nanmean(rr, axis=1, keepdims=True)
    d_s = np.where(m, rs - sm, 0.0)
    d_r = np.where(m, rr - rm, 0.0)
    cov = (d_s * d_r).sum(axis=1)
    vs = np.sqrt((d_s ** 2).sum(axis=1))
    vr = np.sqrt((d_r ** 2).sum(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        ic = np.where((vs > 0) & (vr > 0), cov / (vs * vr), np.nan)
    ic = ic[ok & ~np.isnan(ic)]
    if len(ic) < 60:
        return {}
    k = max(1, min(top_k, int(cnt[ok].min()) // 3))
    # top-k / bottom-k 的成员: 按信号值降序(信号最强在前); top-k = 信号值最大的 k 个
    sr_sorted = np.argsort(np.where(np.isnan(sv_m), -np.inf, sv_m),
                           axis=1, kind='stable')[:, ::-1]
    fv_all = np.where(np.isnan(rv), np.nan, rv)
    def gather(pos):
        return np.take_along_axis(fv_all, pos, axis=1)
    top_pos = sr_sorted[:, :k]
    bot_pos = sr_sorted[:, -k:]
    top_m = np.nanmean(gather(top_pos), axis=1)
    bot_m = np.nanmean(gather(bot_pos), axis=1)
    pool_m = np.nanmean(np.where(np.isnan(rv_m), np.nan, rv_m), axis=1)
    use = ok & ~np.isnan(top_m) & ~np.isnan(pool_m)
    top_m, pool_m = top_m[use], pool_m[use]
    bot_use = ok & ~np.isnan(bot_m)
    # top-k 换手(相邻交易日成员更替率)
    turn = []
    prev = None
    for i in np.where(use)[0]:
        cur = set(sr_sorted[i, :k])
        if prev is not None:
            turn.append(1.0 - len(cur & prev) / k)
        prev = cur
    ic_s = pd.Series(ic)
    std = ic_s.std(ddof=1)
    return {
        'n_days': len(ic_s),
        'ic_mean': round(float(ic_s.mean()), 4),
        'icir': round(float(ic_s.mean() / std), 3) if std and std > 0 else np.nan,
        't_stat': round(float(ic_s.mean() / std * np.sqrt(len(ic_s))), 2) if std and std > 0 else np.nan,
        'ic_pos_rate': round(float((ic_s > 0).mean()), 3),
        'topk_ret': round(float(top_m.mean()), 5),
        'pool_ret': round(float(pool_m.mean()), 5),
        'excess_k': round(float(top_m.mean() - pool_m.mean()), 5),
        'ls_spread': round(float(top_m.mean() - bot_m[bot_use].mean()), 5) if bot_use.any() else np.nan,
        'topk_turnover': round(float(np.mean(turn)), 3) if turn else np.nan,
    }


def validate_one(sig: pd.DataFrame, fwd: pd.DataFrame, top_k: int = 5, min_cs: int = 10) -> dict:
    """信号真伪检验: 排除"静态身份效应"(看似高IC, 实则永远选同几只标的).

    三项证据:
      1) 符号集中度: top-k 成员的持仓频率 top3 占比 与 HHI(越高越像固定选票, 越低越像动态轮动)
      2) 分半稳定: 前半段/后半段各自超额, 检验是否只在某个 regime 有效
      3) 静态倾斜对照: 用前半段信息构造"固定选 k 只持有"的静态组合, 在 后半段 的超额.
         - static_sig: 按前半段信号均值固定选票(信号自身的静态退化版)
         - static_oracle: 按前半段实际收益固定选票(事后最优的上界参考)
         若信号的 dyn_test 超额 <= static_sig, 说明信号价值约等于一个固定持仓, 无动态择时/轮动价值
    """
    idx = sig.index.intersection(fwd.index)
    cols = sig.columns.intersection(fwd.columns)
    if len(idx) < 120 or len(cols) < 6:
        return {}
    s = sig.loc[idx, cols].values.astype(float)
    r = fwd.loc[idx, cols].values.astype(float)
    m = ~np.isnan(s) & ~np.isnan(r)
    ok = m.sum(axis=1) >= min_cs
    if ok.sum() < 120:
        return {}
    sv = np.where(m & ok[:, None], s, np.nan)
    rv = np.where(m & ok[:, None], r, np.nan)
    k = max(1, min(top_k, int(m.sum(axis=1)[ok].min()) // 3))
    order = np.argsort(np.where(np.isnan(sv), -np.inf, sv), axis=1, kind='stable')[:, ::-1]
    rows = np.where(ok)[0]
    fvr = rv[rows]
    top = order[rows, :k]
    top_ret = np.nanmean(np.take_along_axis(fvr, top, axis=1), axis=1)
    pool = np.nanmean(fvr, axis=1)
    exc = top_ret - pool
    h = len(rows) // 2
    ex1, ex2 = float(np.nanmean(exc[:h])), float(np.nanmean(exc[h:]))
    freq = np.bincount(top.ravel(), minlength=len(cols)) / len(rows)
    top3_share = float(np.sort(freq)[::-1][:3].sum())
    hhi = float((freq ** 2).sum())
    # 静态倾斜: 前半段信息 -> 后半段固定持有同一批标的
    def static_excess(score_per_symbol):
        sel = np.argsort(-np.nan_to_num(score_per_symbol, nan=-np.inf))[:k]
        t2 = np.nanmean(fvr[h:][:, sel], axis=1)
        p2 = np.nanmean(fvr[h:], axis=1)
        return float(np.nanmean(t2 - p2))
    stat_sig = static_excess(np.nanmean(sv[rows[:h]], axis=0))
    stat_oracle = static_excess(np.nanmean(rv[rows[:h]], axis=0))
    return {
        'k': k,
        'exc_half1': round(ex1, 5),
        'exc_half2': round(ex2, 5),
        'dyn_test_exc': round(ex2, 5),
        'static_sig_test_exc': round(stat_sig, 5),
        'static_oracle_test_exc': round(stat_oracle, 5),
        'dyn_minus_static': round(ex2 - stat_sig, 5),
        'top3_share': round(top3_share, 3),
        'hhi': round(hhi, 3),
        'n_eff_symbols': round(1.0 / hhi, 1) if hhi > 0 else np.nan,
    }


# ================================================================ 主流程 ================================================================ #
def main():
    ap = argparse.ArgumentParser(description='独立只读研究: 全列信号筛查(可交易口径)')
    ap.add_argument('--pool', default='etf_3x')
    ap.add_argument('--interval', default='day')
    ap.add_argument('--pkl-dir', default=os.path.join(os.path.expanduser('~'), 'quant'))
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl 路径(优先, 用于读取带后缀的 pkl)')
    ap.add_argument('--start', default='2021-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--horizons', default='5,20,60')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--min-cs', type=int, default=10)
    ap.add_argument('--no-derived', action='store_true', help='不加入衍生信号')
    ap.add_argument('--validate-top', type=int, default=0,
                    help='对每个周期 excess_k 前 N 个信号做真伪检验(静态身份/分半/OOS静态对照)')
    ap.add_argument('--signals', default=None, help='逗号分隔, 只筛指定信号(便于交互式深挖)')
    args = ap.parse_args()

    pkl_path = args.pkl_path or os.path.join(args.pkl_dir, f'{args.pool}_{args.interval}_ta_data.pkl')
    if not os.path.exists(pkl_path):
        print(f'[ERROR] 找不到 pkl: {pkl_path}')
        sys.exit(1)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                           f'{args.pool}_sig_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    raw, panel = load_panel(pkl_path, args.interval)
    if args.start:
        panel = panel[panel.index.get_level_values('date') >= pd.Timestamp(args.start)]
    if args.end:
        panel = panel[panel.index.get_level_values('date') <= pd.Timestamp(args.end)]
    dts = panel.index.get_level_values('date')
    symbols = sorted(panel.index.get_level_values('symbol').unique())
    print(f'== 数据 == {args.pool}: {len(symbols)} 标的, {dts.min().date()}~{dts.max().date()}, '
          f'panel {len(panel)} 行')

    open_wide = panel['Open'].unstack('symbol').sort_index()

    # 候选信号集合: panel 数值列 + 衍生信号
    num_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    cands = {}
    for c in num_cols:
        try:
            w = panel[c].unstack('symbol').sort_index()
        except Exception:
            continue
        if isinstance(w, pd.DataFrame) and w.shape[1] >= 2:
            cands[c] = w
    n_raw = len(cands)
    if not args.no_derived:
        for k, v in build_derived(panel).items():
            v = v.reindex(index=open_wide.index, columns=open_wide.columns)
            if v.notna().sum().sum() > 0:
                cands[k] = v
    print(f'== 候选信号 == panel 数值列 {n_raw} + 衍生 {len(cands) - n_raw} = {len(cands)} 个')
    if args.signals:
        keep = [x.strip() for x in args.signals.split(',') if x.strip()]
        missing = [x for x in keep if x not in cands]
        cands = {k: v for k, v in cands.items() if k in keep}
        if missing:
            print(f'  [警告] 未找到: {missing}')
        print(f'== 仅筛指定信号({len(cands)}) == {list(cands)}')

    horizons = [int(x) for x in args.horizons.split(',')]
    all_rows = []
    all_val = []
    for h in horizons:
        fwd = tradable_fwd(open_wide, h)
        rows = []
        for name, sig in cands.items():
            try:
                rec = screen_one(sig, fwd, top_k=args.top_k, min_cs=args.min_cs)
            except Exception as e:
                print(f'  [SKIP] {name}: {e}')
                continue
            if rec:
                rec['signal'] = name
                rec['group'] = ('衍生' if name.startswith('D_') else
                                ('疑似标签' if any(p in name.lower() for p in SUSPECT_PAT) else 'panel列'))
                rows.append(rec)
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        cols = ['signal', 'group', 'excess_k', 'topk_ret', 'pool_ret', 'ls_spread',
                'ic_mean', 'icir', 't_stat', 'ic_pos_rate', 'topk_turnover', 'n_days']
        df = df[cols].sort_values('excess_k', ascending=False)
        df.to_csv(os.path.join(out_dir, f'screen_h{h}.csv'), index=False, encoding='utf-8-sig')
        all_rows.append((h, df))
        print(f'\n===== 前瞻 h={h} 日: 按超额收益(excess_k) 排序 Top 20 =====')
        print(df.head(20).to_string(index=False))
        print(f'--- 同周期 bottom 5(最差) ---')
        print(df.tail(5).to_string(index=False))

        if args.validate_top > 0:
            vrows = []
            for name in df.head(args.validate_top)['signal']:
                rec = validate_one(cands[name], fwd, top_k=args.top_k, min_cs=args.min_cs)
                if rec:
                    rec['signal'] = name
                    vrows.append(rec)
            if vrows:
                vdf = pd.DataFrame(vrows)[
                    ['signal', 'dyn_test_exc', 'static_sig_test_exc', 'static_oracle_test_exc',
                     'dyn_minus_static', 'exc_half1', 'exc_half2', 'top3_share',
                     'n_eff_symbols', 'hhi', 'k']]
                vdf.to_csv(os.path.join(out_dir, f'validate_h{h}.csv'), index=False,
                           encoding='utf-8-sig')
                print(f'\n----- h={h} 真伪检验(Top {args.validate_top}) -----')
                print('dyn_test_exc=后半段实际超额; static_sig=静态退化版后半段超额; '
                      'dyn_minus_static>0 才有动态增量; n_eff_symbols 越大越分散')
                print(vdf.to_string(index=False))
                all_val.append((h, vdf))

    # 报告: 汇总各周期都靠前的信号
    lines = [f'signal_search 报告 | pool={args.pool} | run={run_id}',
             f'窗口 {dts.min().date()}~{dts.max().date()}, {len(symbols)} 标的',
             f'口径: 信号日收盘 -> 次日开盘入场 -> 持有h日开盘出场; excess_k = top{args.top_k}均值 - 池等权均值',
             '']
    for h, df in all_rows:
        lines.append(f'== h={h} 全量排序 ==')
        lines.append(df.to_string(index=False))
        lines.append('')
    for h, vdf in all_val:
        lines.append(f'== h={h} 真伪检验(静态身份/分半/OOS静态对照) ==')
        lines.append(vdf.to_string(index=False))
        lines.append('')
    # 交叉周期稳健性: 各周期都进入前 15 的信号
    if len(all_rows) >= 2:
        sets = [set(df.head(15)['signal']) for _, df in all_rows]
        common = set.intersection(*sets)
        lines.append(f'== 各周期均进 Top15 的信号({len(common)}) ==')
        lines.append(', '.join(sorted(common)))
        print(f'\n===== 各周期均进 Top15 的信号({len(common)}) =====')
        for h, df in all_rows:
            sub = df[df['signal'].isin(common)][['signal', 'excess_k', 'icir', 'topk_turnover']]
            print(f'-- h={h} --')
            print(sub.to_string(index=False))
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')


if __name__ == '__main__':
    main()
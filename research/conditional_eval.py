# -*- coding: utf-8 -*-
"""conditional_eval.py — 条件层评估工具: 事件×状态 条件收益矩阵(只读)

三层验证漏斗的中间层:
  成分层  indicator_eval.py     — 单信号列的截面预测力(单元测试)
  条件层  conditional_eval.py   — "事件+状态"组合的条件收益统计(本工具)
  集成层  vectorbt              — 完整策略回测(成本/路径/仓位/回撤)

回答的问题: 组合条件(如 s_trend=='up' 且 m_trend=='up' 且当日存在正向触发)
在信号日 t 确认后, 未来 h 日可交易收益相对三类基线的条件超额是否显著:

  exc_pool  vs 触发日全池等权     — 绝对有效性(逐日对齐, 消除市场日间波动)
  exc_event vs 同事件不加状态过滤 — 状态过滤的边际贡献
  exc_state vs 同状态无事件       — 事件触发的边际贡献(timing 价值)

统计口径(与 signal_search/indicator_eval 完全一致):
  信号日 t 收盘确认 -> t+1 开盘入场 -> 持有 h 日 -> t+1+h 开盘出场
  fwd(t,s) = Open[t+1+h]/Open[t+1] - 1

t 检验(防伪三坑):
  1) 截面相关   -> 先按日聚合: d_t = 触发组当日均值 - 基线当日均值(仅触发日取值)
  2) 窗口重叠   -> 日级序列自相关 -> Newey-West HAC 标准误, lag = h-1
  3) 多重检验   -> 全表 Benjamini-Hochberg FDR q 值(族 = 本次运行全部组合×h)

结果解读(进集成层的建议门槛, 可按需调整):
  n_days >= MIN_DAYS 且 low_n==0 且 same_sign_half==1
  且 t_exc_pool >= 2 且 fdr_q <= 0.10 且 exc_event 或 exc_state 至少一个同号为正

挖掘友好:
  --grid s,m      自动展开 s_trend×m_trend 全状态网格(全组合+单维, 便于看边际分解)
  --events        用内置事件名(break_up/trig_up/s_flip_up/...), 或 --event-expr 自定义
  --event-expr "名字:表达式"  长表逐行条件(pandas eval), 须无前视
  --state-expr    同上, 定义状态
  --cooldown N    触发后 N 日同标的不重复计为触发(对齐真实交易, 默认 0 全统计口径)
  --pkl "池=路径"  覆盖数据源(etf_3x 生产 pkl 仅 2025+, 建议用 research pkl 补全史)

用法:
  python conditional_eval.py --pool etf_3x --grid s,m --horizons 5,20
  python conditional_eval.py --pool all --pkl "etf_3x=C:\\Users\\northcheng\\quant\\etf_3x_day_ta_data_research.pkl"
  python conditional_eval.py --pool hs300 --states-bull? -> --state-expr "bull:s_trend=='up' and m_trend=='up'"
"""
import argparse
import os
import sys
import warnings
from itertools import product
from math import erf, sqrt

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factor_research import load_panel                    # noqa: E402
from signal_search import tradable_fwd                    # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ================================================================ 参数 ================================================================ #
POOLS = {
    'etf_3x':      r'C:\Users\northcheng\quant\etf_3x_day_ta_data.pkl',
    'company_300': r'C:\Users\northcheng\quant\company_300_day_ta_data.pkl',
    'hs300':       r'C:\Users\northcheng\quant\hs300_day_ta_data.pkl',
    'a_etf_all':   r'C:\Users\northcheng\quant\a_etf_all_day_ta_data.pkl',
}
START = '2021-01-01'
HORIZONS = [5, 20]
MIN_DAYS = 60          # 触发日数低于此 => low_n=1, t 值解读需谨慎

# 事件注册表(表达式型): 逐行条件, 在长表上求值。
# 符号约定(按 bc_technical_analysis L815/816 字面): up 族 >0 为触发, down 族 <0 为触发。
EXPR_EVENTS = {
    'trig_up':      'trigger_up_score > 0',
    'trig_down':    'trigger_down_score < 0',
    'break_up':     'break_up_score > 0',
    'break_down':   'break_down_score < 0',
    'support':      'support_score > 0',
    'resistant':    'resistant_score < 0',
    'pattern_up':   'pattern_up_score > 0',
    'pattern_down': 'pattern_down_score < 0',
    'net_pos':      'trigger_net > 0',
    'net_neg':      'trigger_net < 0',
}

# 事件注册表(翻转型): 状态当日为目标值且前一日不是 => 状态进入事件(timing 灵感)
FLIP_EVENTS = {
    's_flip_up':   ('s_trend', 'up'),
    's_flip_down': ('s_trend', 'down'),
    'm_flip_up':   ('m_trend', 'up'),
    'm_flip_down': ('m_trend', 'down'),
}

DEFAULT_EVENTS = ['trig_up', 'trig_down', 'break_up', 'break_down',
                  'pattern_up', 'pattern_down',
                  's_flip_up', 's_flip_down', 'm_flip_up', 'm_flip_down']

# 状态维度(供 --grid 展开)
TREND_DIMS = {
    's': ('s_trend', ('up', 'down', 'wave')),
    'm': ('m_trend', ('up', 'down', 'wave')),
}


# ================================================================ 统计核心 ================================================================ #
def nw_tstat(x, lag):
    """日级序列均值的 Newey-West HAC t 检验(Bartlett 权重), 返回 (t, p, n)。

    lag 取 h-1: h 日 fwd 窗口相邻重叠 h-1 天, 是序列相关的最长阶数。
    n < 30 或长方差 <= 0 时返回 NaN。
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 30:
        return np.nan, np.nan, n
    m = float(x.mean())
    e = x - m
    s = float(e @ e) / n
    lag = int(lag)
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1)
        s += 2.0 * w * float(e[l:] @ e[:-l]) / n
    if s <= 0:
        return np.nan, np.nan, n
    se = sqrt(s / n)
    t = m / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return t, p, n


def bh_qvals(p):
    """Benjamini-Hochberg FDR q 值(NaN 剔除后回填)。"""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pp = p[ok]
    n = int(pp.size)
    if n == 0:
        return q
    order = np.argsort(pp, kind='stable')
    ranked = pp[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(n)
    qq[order] = np.minimum(ranked, 1.0)
    q[ok] = qq
    return q


def cond_stats(cond_w, event_w, state_w, fwd, h, min_days=MIN_DAYS):
    """单个 (事件×状态×h) 的条件收益统计。

    cond_w / event_w / state_w / fwd 均为对齐的 (date × symbol) 宽表。
    event_w/state_w 为 None 时对应基线不计算(事件自身行/状态自身行)。
    日级序列: d_t = 触发组当日均值 - 基线组当日均值, 仅在有触发的日子取值
    (基线组是触发组的超集, 触发日上基线均值必有定义)。
    """
    fv = fwd.values
    valid = ~np.isnan(fv)
    cm = cond_w.values & valid
    any_c = cm.any(axis=1)
    n_obs = int(cm.sum())
    n_days = int(any_c.sum())
    rec = {'n_obs': n_obs, 'n_days': n_days,
           'obs_per_day': round(n_obs / n_days, 2) if n_days else np.nan,
           'n_symbols': int((cm.any(axis=0)).sum()),
           'low_n': int(n_days < min_days)}
    if n_days == 0:
        return rec

    def rowmean(m):
        return np.nanmean(np.where(m, fv, np.nan), axis=1)   # 无触发日 -> NaN

    c_t = rowmean(cm)
    p_t = rowmean(valid)                                      # 触发日全池等权
    d_pool = c_t - p_t                                        # 逐日对齐
    t1, p1, _ = nw_tstat(d_pool[any_c], lag=h - 1)

    c_act, p_act, dd = c_t[any_c], p_t[any_c], d_pool[any_c]
    rec.update({
        'cond_ret': round(float(np.nanmean(c_act)), 5),
        'pool_ret': round(float(np.nanmean(p_act)), 5),
        'exc_pool': round(float(np.nanmean(dd)), 5),
        'win_rate': round(float((dd > 0).mean()), 3),
        'std_exc': round(float(np.nanstd(dd, ddof=1)), 5),
        'median_exc': round(float(np.nanmedian(dd)), 5),
        't_exc_pool': round(t1, 2) if np.isfinite(t1) else np.nan,
        'p_exc_pool': round(p1, 4) if np.isfinite(p1) else np.nan,
    })
    # 边际基线(仅真组合行): 同事件无状态 / 同状态无事件
    for key, base, tag in (('exc_event', event_w, 'event'),
                           ('exc_state', state_w, 'state')):
        if base is None:
            rec[f'exc_{tag}'] = np.nan
            rec[f't_exc_{tag}'] = np.nan
            continue
        db = (c_t - rowmean(base.values & valid))[any_c]
        tb, _, _ = nw_tstat(db, lag=h - 1)
        rec[f'exc_{tag}'] = round(float(np.nanmean(db)), 5)
        rec[f't_exc_{tag}'] = round(tb, 2) if np.isfinite(tb) else np.nan

    # 时间分半稳定性(触发日按时间先后对半)
    idx = np.where(any_c)[0]
    half = len(idx) // 2
    h1 = float(np.nanmean(d_pool[idx[:half]])) if half else np.nan
    h2 = float(np.nanmean(d_pool[idx[half:]]))
    rec['exc_half1'] = round(h1, 5)
    rec['exc_half2'] = round(h2, 5)
    rec['same_sign_half'] = (int(np.sign(h1) == np.sign(h2))
                             if np.isfinite(h1) and np.isfinite(h2) else np.nan)
    return rec


def apply_cooldown(mask, n):
    """事件触发后 n 日内同标的不重复计为触发(逐 symbol O(k))。"""
    if n <= 0:
        return mask
    arr = mask.values
    res = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        idx = np.where(arr[:, j])[0]
        if idx.size == 0:
            continue
        keep = [idx[0]]
        for k in idx[1:]:
            if k - keep[-1] > n:
                keep.append(k)
        res[keep, j] = True
    return pd.DataFrame(res, index=mask.index, columns=mask.columns)


# ================================================================ mask 构建 ================================================================ #
def _align(w, open_wide):
    return w.reindex(index=open_wide.index, columns=open_wide.columns).fillna(False).astype(bool)


def expr_mask(panel, open_wide, expr):
    try:
        s = panel.eval(expr, engine='python')
    except Exception as e:
        print(f'  [SKIP 表达式] {expr}: {e}')
        return None
    if not isinstance(s, pd.Series):
        return None
    return _align(s.unstack('symbol').sort_index(), open_wide)


def build_event_masks(panel, open_wide, names, customs, cooldown):
    masks = {}
    for name in names:
        if name in FLIP_EVENTS:
            col, target = FLIP_EVENTS[name]
            wide = panel[col].unstack('symbol').sort_index()
            w = _align((wide == target) & (wide.shift(1) != target), open_wide)
        elif name in EXPR_EVENTS:
            w = expr_mask(panel, open_wide, EXPR_EVENTS[name])
        else:
            print(f'  [WARN] 未知事件: {name} (内置: {sorted(list(EXPR_EVENTS) + list(FLIP_EVENTS))})')
            continue
        if w is None:
            continue
        if not w.any().any():
            print(f'  [WARN] 事件 {name} 在该池无触发')
            continue
        masks[name] = apply_cooldown(w, cooldown)
    for spec in customs:
        if ':' not in spec:
            print(f'  [WARN] 自定义事件须为 名字:表达式: {spec}')
            continue
        name, expr = spec.split(':', 1)
        w = expr_mask(panel, open_wide, expr.strip())
        if w is not None and w.any().any():
            masks[name] = apply_cooldown(w, cooldown)
    return masks


def build_state_masks(panel, open_wide, grid, customs):
    states = {'(none)': None}          # '(none)' = 不加状态过滤(事件自身基线行)
    dims = [d.strip() for d in (grid or '').split(',') if d.strip()]
    dim_wides = {}
    for d in dims:
        if d not in TREND_DIMS:
            print(f'  [WARN] 未知 grid 维度: {d} (可用: {list(TREND_DIMS)})')
            continue
        col, _ = TREND_DIMS[d]
        dim_wides[d] = panel[col].unstack('symbol').sort_index()
    dims = list(dim_wides)
    # 全组合(笛卡尔积) + 单维(便于看状态组合的边际分解)
    specs = []
    if dims:
        for combo in product(*[TREND_DIMS[d][1] for d in dims]):
            specs.append([(d, lv) for d, lv in zip(dims, combo)])
    for d in dims:
        for lv in TREND_DIMS[d][1]:
            specs.append([(d, lv)])
    for spec in specs:
        name = '_'.join(f'{d}{lv}' for d, lv in spec)
        m = None
        for d, lv in spec:
            w = (dim_wides[d] == lv)
            m = w if m is None else (m & w)
        states[name] = _align(m, open_wide)
    for specx in customs:
        if ':' not in specx:
            print(f'  [WARN] 自定义状态须为 名字:表达式: {specx}')
            continue
        name, expr = specx.split(':', 1)
        w = expr_mask(panel, open_wide, expr.strip())
        if w is not None:
            states[name] = w
    return states


# ================================================================ 主流程 ================================================================ #
def eval_pool(pool, pkl_path, args):
    _, panel = load_panel(pkl_path, 'day')
    panel = panel[panel.index.get_level_values('date') >= pd.Timestamp(args.start)]
    dts = panel.index.get_level_values('date')
    print(f'\n===== {pool}: {panel.index.get_level_values("symbol").nunique()} 标的, '
          f'{dts.min().date()}~{dts.max().date()}, {len(panel)} 行 =====')

    open_wide = panel['Open'].unstack('symbol').sort_index()
    fwds = {h: tradable_fwd(open_wide, h) for h in args.horizons}
    event_masks = build_event_masks(panel, open_wide, args.events, args.event_expr, args.cooldown)
    state_masks = build_state_masks(panel, open_wide, args.grid, args.state_expr)
    print(f'  事件 {len(event_masks)} 个 × 状态 {len(state_masks)} 个 × 周期 {len(args.horizons)} 个'
          f'  (cooldown={args.cooldown})')

    # 组合清单: 事件自身行(state=none) + 状态自身行(event=all) + 真组合行
    combos = [(e, em, '(none)', None) for e, em in event_masks.items()]
    combos += [('(all)', None, s, sm) for s, sm in state_masks.items() if s != '(none)']
    for e, em in event_masks.items():
        for s, sm in state_masks.items():
            if s == '(none)':
                continue
            combos.append((e, em, s, sm))

    rows = []
    for ename, emask, sname, smask in combos:
        if smask is None:                       # 事件自身行: 只对池基线
            cond, ev, st = emask, None, None
        elif emask is None:                     # 状态自身行: 只对池基线
            cond, ev, st = smask, None, None
        else:                                   # 真组合: 三基线齐全
            cond, ev, st = emask & smask, emask, smask
        for h in args.horizons:
            rec = cond_stats(cond, ev, st, fwds[h], h, args.min_days)
            rec.update({'pool': pool, 'event': ename, 'state': sname, 'horizon': h})
            rows.append(rec)

    df = pd.DataFrame(rows)
    df['fdr_q'] = np.round(bh_qvals(df['p_exc_pool'].values.astype(float)), 3)
    fp = _out(f'cond_eval_{pool}.csv')
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f'  已写出 {fp}  ({len(df)} 行)')

    # 控制台摘要: 各 h 按 t_exc_pool 排序 top15 + FDR 显著清单
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 60)
    cols = [c for c in ['event', 'state', 'n_obs', 'n_days', 'obs_per_day', 'cond_ret',
                        'exc_pool', 'exc_event', 'exc_state', 'win_rate', 't_exc_pool',
                        't_exc_event', 't_exc_state', 'same_sign_half', 'fdr_q', 'low_n']
            if c in df.columns]
    for h in args.horizons:
        sub = df[(df['horizon'] == h) & (df['n_days'] > 0)]
        if sub.empty:
            continue
        print(f'\n----- {pool} h={h}: 按 t_exc_pool 排序 Top 15 -----')
        print(sub.sort_values('t_exc_pool', ascending=False).head(15)[cols].to_string(index=False))
        sig = sub[(sub['fdr_q'] <= 0.10) & (sub['t_exc_pool'].abs() >= 2.0) & (sub['low_n'] == 0)]
        print(f'----- FDR q<=0.10 且 |t|>=2 且 n_days>={args.min_days}: {len(sig)} 行 -----')
        if not sig.empty:
            print(sig.sort_values('t_exc_pool', ascending=False)[cols].to_string(index=False))
    return df


def _out(fname: str) -> str:
    d = os.path.join(HERE, 'output')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def main():
    ap = argparse.ArgumentParser(description='条件层评估工具: 事件×状态 条件收益矩阵(只读)')
    ap.add_argument('--pool', default='etf_3x', help='池名, 逗号分隔多个, 或 all')
    ap.add_argument('--pkl', action='append', default=[],
                    help='覆盖数据源, 形如 "etf_3x=C:\\path\\to.pkl", 可多次')
    ap.add_argument('--start', default=START)
    ap.add_argument('--horizons', default=','.join(str(h) for h in HORIZONS))
    ap.add_argument('--events', default=','.join(DEFAULT_EVENTS),
                    help=f'逗号分隔内置事件名(可选: {sorted(list(EXPR_EVENTS) + list(FLIP_EVENTS))})')
    ap.add_argument('--event-expr', action='append', default=[],
                    help='自定义事件 "名字:表达式", 如 "big_break:break_up_score>2"')
    ap.add_argument('--grid', default='s,m', help='状态网格维度, 逗号分隔(可选: s,m)')
    ap.add_argument('--state-expr', action='append', default=[],
                    help='自定义状态 "名字:表达式", 如 "bull:s_trend==\'up\' and m_trend==\'up\'"')
    ap.add_argument('--cooldown', type=int, default=0,
                    help='触发后 N 日同标的不重复计数(对齐交易口径, 默认 0)')
    ap.add_argument('--min-days', type=int, default=MIN_DAYS, help='触发日数下限(低样本标记)')
    args = ap.parse_args()

    args.horizons = [int(x) for x in args.horizons.split(',')]
    args.events = [x.strip() for x in args.events.split(',') if x.strip()]
    overrides = {}
    for spec in args.pkl:
        if '=' in spec:
            k, v = spec.split('=', 1)
            overrides[k.strip()] = v.strip()

    pools = list(POOLS) if args.pool == 'all' else [p.strip() for p in args.pool.split(',')]
    for p in pools:
        path = overrides.get(p) or POOLS.get(p)
        if path is None:
            print(f'[ERROR] 未知池: {p}; 可选 {list(POOLS)}')
            continue
        if not os.path.exists(path):
            print(f'[ERROR] 找不到 pkl: {p} -> {path}  (该池本轮跳过)')
            continue
        eval_pool(p, path, args)


if __name__ == '__main__':
    main()

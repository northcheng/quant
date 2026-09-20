# -*- coding: utf-8 -*-
"""indicator_eval.py — 指标价值评估工具(只读)

固化自原先的过程脚本 _tmp_eval / _tmp_grade / _tmp_redund / _tmp_neigh,
统一入口为 4 个子命令。只读 pkl, 全部产物写入 research/output/。

评估维度(子命令 eval):
  1) 可用性   : coverage(非空率) / nuniq(取值个数) / is_const
  2) 预测力   : ic_mean / icir / t_stat / ic_pos_rate (截面 Spearman IC, 可交易口径)
  3) 可交易超额: excess_k = top-k 均值 - 同池等权; topk_ret / pool_ret
  4) 区分度   : ls_spread (top-k 减 bottom-k)
  5) 换手成本 : topk_turnover / sig_ac1 (信号前后日截面秩自相关, 越高越省成本)
  6) 时间稳定 : exc_half1 / exc_half2 / same_sign_half
  7) 动态增量 : dyn_minus_static (后半段实际超额 - 静态退化版后半段超额)
  8) 身份集中 : top3_share / n_eff_symbols (越低越分散, 越高越像固定选票)

用法:
  python indicator_eval.py eval   --pool etf_3x | --pool all [--derived]
  python indicator_eval.py grade
  python indicator_eval.py redund --pool etf_3x
  python indicator_eval.py neigh
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factor_research import load_panel, normalize_causal                    # noqa: E402
from signal_search import (tradable_fwd, screen_one, validate_one,           # noqa: E402
                           build_derived)

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
TOP_K = 5
MIN_CS = 10

# 本评估只把"明确可能是未来标签/动作"的列标为疑似; 不用 signal_search 的宽口径
# (那个口径把 *_day 状态列也算进来, 但 *_day 是 sda 回看状态, 是合法特征)
SUSPECT_PAT = ('pos_label', 'neg_label', 'label', 'action')

# 描述型/非信号列(不应作为截面信号)
SKIP_PAT = ('_description', 'description')
# 仅按列名精确跳过(避免子串误伤 pattern_up_score / pattern_down_score)
SKIP_EXACT = ('pattern_up', 'pattern_down')

# 归一化窗口(与 bc_technical_analysis.calculate_ta_signal 的 day 口径一致)
NORM_WINDOW, NORM_MINP = 252, 60

# 修复后的合成净分: 分量 -> 权重。与 bc_technical_analysis 中
# trigger_net / pattern_net 的 w 完全一致(见 calculate_ta_score / calculate_ta_signal)。
SYNTH_WEIGHTS = {
    'trigger_net': {'break_up_score': 1.0, 'break_down_score': 1.0,
                    'support_score': 0.5, 'resistant_score': 0.5},
    'pattern_net': {'pattern_up_score': 1.0, 'pattern_down_score': 1.0},
}

# 分级阈值(grade)
GRADE_POOLS = ['etf_3x', 'company_300', 'hs300', 'a_etf_all']
STATIC_AC1 = 0.99          # 信号截面排序几乎不随时间变化 => 事实上的固定选票
TURN_LOW = 0.05
THRESH = 0.90              # redund: |corr| 高共线阈值

# neigh: 重点指标
FOCUS = ['Low_to_kijun', 'High_to_kijun', 'ichimoku_distance_alpha', 'adx_power',
         'adx_strength_change', 'kama_slow_rate', 'candle_gap_distance',
         'ichimoku_distance_day', 'kijun_day', 'trend_magnitude',
         'trend_score_alpha', 'trend_magnitude_alpha', 'kama_rate',
         'Low_to_kama_slow', 'Low_to_tankan', 'kijun', 'Close',
         'trigger_score', 'trigger_net', 'pattern_net']


# ================================================================ 公共 ================================================================ #
def group_of(name: str) -> str:
    """按产出函数给指标分组, 便于报告分类。"""
    n = name.lower()
    if name.startswith('D_'):
        return '衍生基准'
    if any(p in n for p in ('label', 'action')):
        return '疑似标签'
    if name in ('Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Amount'):
        return '原始OHLCV'
    if n.endswith('_day') or n.endswith('_days'):
        return '状态持续(day)'
    if name.startswith(('trigger', 'break_up', 'break_down', 'support', 'resistant')):
        return 'ta_score触发'
    if name.startswith(('trend', 's_direction')) or name.endswith('_alpha'):
        return 'ta_signal趋势'
    if name.startswith('pattern') or name in (
            '超买超卖', '关键突破', '长线边界', '趋势转换', '趋势启动',
            '区间波动', '触顶触底', '短期转向', '中期转向'):
        return 'ta_signal形态'
    if any(k in n for k in ('ichimoku', 'kama', 'tankan', 'kijun', 'senkou', 'chikou',
                            'kumo', 'cloud')):
        return 'ta_basic/static云图'
    if any(k in n for k in ('adx', 'adi', 'rsi', 'macd', 'atr', 'boll', 'cci', 'kdj',
                            'obv', 'mfi', 'wr', 'roc', 'stoch')):
        return 'ta_basic指标'
    if any(k in n for k in ('candle', 'shadow', 'entity', '十字星', '平头', '长影线',
                            'candle_color', 'gap')):
        return 'ta_basic蜡烛'
    if any(k in n for k in ('distance', 'rate', 'change', 'direction', 'magnitude',
                            'score', 'position', 'support', 'resistant', 'break')):
        return 'ta_basic/static其他'
    return '其他'


def sig_ac1(sig: pd.DataFrame, min_cs: int = MIN_CS) -> float:
    """信号自身前后日截面秩自相关(平均): 越高 => 信号越稳 => 换手越低。"""
    s = sig.replace([np.inf, -np.inf], np.nan)
    r = s.rank(axis=1, pct=True)
    a = r.shift(1)
    ok = (s.notna().sum(axis=1) >= min_cs) & (a.notna().sum(axis=1) >= min_cs)
    if ok.sum() < 60:
        return np.nan
    x = r.loc[ok].values
    y = a.loc[ok].values
    m = ~np.isnan(x) & ~np.isnan(y)
    cnt = m.sum(axis=1)
    keep = cnt >= min_cs
    if keep.sum() < 60:
        return np.nan
    x, y, m = x[keep], y[keep], m[keep]
    xm = np.where(m, x, np.nan)
    ym = np.where(m, y, np.nan)
    dx = xm - np.nanmean(xm, axis=1, keepdims=True)
    dy = ym - np.nanmean(ym, axis=1, keepdims=True)
    dx = np.where(m, dx, 0.0)
    dy = np.where(m, dy, 0.0)
    cov = (dx * dy).sum(axis=1)
    vx = np.sqrt((dx ** 2).sum(axis=1))
    vy = np.sqrt((dy ** 2).sum(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.where((vx > 0) & (vy > 0), cov / (vx * vy), np.nan)
    c = c[~np.isnan(c)]
    return round(float(np.mean(c)), 4) if len(c) >= 60 else np.nan


def add_synth(cands: dict, window: int = NORM_WINDOW, min_periods: int = NORM_MINP):
    """注入修复后的合成净分 trigger_net / pattern_net 及其 *_alpha。

    口径与生产代码一致: 每个分量先做因果归一化(|x| -> [0,1])再带符号加权求和
    (bc_technical_analysis.calculate_ta_score / calculate_ta_signal)。
    pkl 中已存在该列(由修复后的生产代码生成)时, 直接沿用原值, 不覆盖。
    """
    for name, weights in SYNTH_WEIGHTS.items():
        if name in cands:
            net = cands[name]
        elif all(c in cands for c in weights):
            net = None
            for col, w in weights.items():
                alpha = normalize_causal(cands[col].abs(), window=window, min_periods=min_periods)
                term = w * alpha * np.sign(cands[col].fillna(0.0))
                net = term if net is None else net + term
            cands[name] = net
        else:
            continue
        if f'{name}_alpha' not in cands:
            cands[f'{name}_alpha'] = normalize_causal(net.abs(), window=window,
                                                      min_periods=min_periods)


def build_cands(panel: pd.DataFrame, with_derived: bool = False):
    """把 panel 的数值列转为 (date × symbol) 宽表, 并注入合成净分。"""
    open_wide = panel['Open'].unstack('symbol').sort_index()
    num_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    cands = {}
    for c in num_cols:
        if any(p in c.lower() for p in SKIP_PAT) or c in SKIP_EXACT:
            continue
        try:
            w = panel[c].unstack('symbol').sort_index()
        except Exception:
            continue
        if isinstance(w, pd.DataFrame) and w.shape[1] >= 2:
            cands[c] = w
    if with_derived:
        for k, v in build_derived(panel).items():
            cands[k] = v.reindex(index=open_wide.index, columns=open_wide.columns)
    add_synth(cands)
    return cands, num_cols, open_wide


# ================================================================ eval ================================================================ #
def eval_pool(pool: str, pkl: str, with_derived: bool = False) -> pd.DataFrame:
    _, panel = load_panel(pkl, 'day')
    panel = panel[panel.index.get_level_values('date') >= pd.Timestamp(START)]
    syms = panel.index.get_level_values('symbol').unique()
    dts = panel.index.get_level_values('date')
    print(f'\n===== {pool}: {len(syms)} 标的, {dts.min().date()}~{dts.max().date()}, '
          f'{len(panel)} 行 =====')
    cands, num_cols, open_wide = build_cands(panel, with_derived=with_derived)
    print(f'  候选指标数 = {len(cands)}  (原生 {len(num_cols)} 数值列)')

    fwds = {h: tradable_fwd(open_wide, h) for h in HORIZONS}
    n_days_total = panel.index.get_level_values('date').nunique()
    rows = []
    for i, (name, sig) in enumerate(cands.items(), 1):
        rec = {'signal': name, 'group': group_of(name),
               'coverage': round(float(sig.notna().mean().mean()), 3),
               'nuniq': int(pd.unique(sig.values[~pd.isna(sig.values)]).size)
               if sig.notna().any().any() else 0,
               'suspect': any(p in name.lower() for p in SUSPECT_PAT),
               'sig_ac1': sig_ac1(sig)}
        rec['is_const'] = int(rec['nuniq'] <= 1)
        for h in HORIZONS:
            fwd = fwds[h]
            try:
                s = screen_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception as e:
                print(f'  [SKIP screen {h}] {name}: {e}')
                s = {}
            for k2, v2 in (s or {}).items():
                rec[f'h{h}_{k2}'] = v2
            try:
                v = validate_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception as e:
                print(f'  [SKIP valid {h}] {name}: {e}')
                v = {}
            for k2, v2 in (v or {}).items():
                rec[f'h{h}_{k2}'] = v2
            e1 = rec.get(f'h{h}_exc_half1')
            e2 = rec.get(f'h{h}_exc_half2')
            rec[f'h{h}_same_sign_half'] = (int(np.sign(e1) == np.sign(e2))
                                           if e1 is not None and e2 is not None
                                           and not (pd.isna(e1) or pd.isna(e2)) else np.nan)
        rows.append(rec)
        if i % 40 == 0:
            print(f'  ... 已评估 {i}/{len(cands)}')

    df = pd.DataFrame(rows)
    fp = _out(f'eval_{pool}.csv')
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f'  已写出 {fp}  ({len(df)} 行 × {len(df.columns)} 列)')
    print(f'  [窗口] 交易日 {n_days_total} 天')
    return df


def cmd_eval(a):
    targets = list(POOLS) if a.pool == 'all' else [a.pool]
    missing = []
    for p in targets:
        if p not in POOLS:
            print(f'[ERROR] 未知池: {p}; 可选 {list(POOLS)}')
            continue
        if not os.path.exists(POOLS[p]):
            missing.append((p, POOLS[p]))
            continue
        eval_pool(p, POOLS[p], with_derived=a.derived)
    for p, fp in missing:
        print(f'[ERROR] 找不到 pkl: {p} -> {fp}  (该池本轮跳过)')


# ================================================================ grade ================================================================ #
def load_all():
    out = {}
    for p in GRADE_POOLS:
        fp = _out(f'eval_{p}.csv')
        if os.path.exists(fp):
            out[p] = pd.read_csv(fp).set_index('signal')
        else:
            print(f'[WARN] 缺少 {fp}')
    return out


def build_summary(base):
    all_sig = sorted(set().union(*[set(df.index) for df in base.values()]))
    rows = []
    for s in all_sig:
        avail = [p for p in base if s in base[p].index]
        r0 = base[avail[0]].loc[s]
        rec = {'signal': s, 'group': r0['group'],
               'suspect': int(any(p in s.lower() for p in SUSPECT_PAT)),
               'coverage': round(float(np.mean([base[p].loc[s, 'coverage'] for p in avail])), 3),
               'nuniq': int(np.max([base[p].loc[s, 'nuniq'] for p in avail])),
               'sig_ac1': round(float(np.nanmean([base[p].loc[s, 'sig_ac1'] for p in avail])), 3)}
        for p in avail:
            r = base[p].loc[s]
            rec[f'{p}_h5_exc'] = r.get('h5_excess_k')
            rec[f'{p}_h20_exc'] = r.get('h20_excess_k')
            rec[f'{p}_h20_icir'] = r.get('h20_icir')
        for h in HORIZONS:
            e = {}
            for p in avail:
                r = base[p].loc[s]
                v = r.get(f'h{h}_excess_k')
                if pd.notna(v):
                    e[p] = float(v)
            for k in ['icir', 'topk_turnover', 'dyn_minus_static', 'n_eff_symbols',
                      'top3_share']:
                vals = [float(base[p].loc[s, f'h{h}_{k}']) for p in avail
                        if pd.notna(base[p].loc[s, f'h{h}_{k}'])]
                rec[f'h{h}_{k}_mean'] = round(float(np.mean(vals)), 4) if vals else np.nan
            ss = [int(base[p].loc[s, f'h{h}_same_sign_half']) for p in avail
                  if pd.notna(base[p].loc[s, f'h{h}_same_sign_half'])]
            rec[f'h{h}_same_sign_n'] = int(np.sum(ss)) if ss else 0
            rec[f'h{h}_same_sign_of'] = len(ss)
            if e:
                vals = np.array(list(e.values()))
                # 方向对齐: 以池间多数符号为正 ; 若无多数, 用均值符号
                npos = int((vals > 0).sum())
                d = 1.0 if npos * 2 >= len(vals) else -1.0
                al = vals * d
                rec[f'h{h}_n_pool'] = len(vals)
                rec[f'h{h}_n_raw_pos'] = npos
                rec[f'h{h}_n_consist'] = int(max(npos, len(vals) - npos))
                rec[f'h{h}_dir'] = '正' if d > 0 else '负'
                rec[f'h{h}_exc_mean'] = round(float(al.mean()), 4)
                rec[f'h{h}_exc_min'] = round(float(al.min()), 4)
                rec[f'h{h}_exc_neg_pool'] = int((al <= 0).sum())
            else:
                rec[f'h{h}_n_pool'] = 0
        rows.append(rec)
    return pd.DataFrame(rows)


def classify(r):
    """规则分级:
       T0不可用    覆盖太低 / 取值太少
       L疑似标签  可能是未来标签/动作列, 不可当信号
       S静态身份  截面排序几乎不随时间变化 = 事实上的固定选票(非轮动信号)
       A强 / B中 / C弱 / D无效
    """
    if r['coverage'] < 0.5 or r['nuniq'] <= 2:
        return 'T0不可用', 0.0
    if r['suspect']:
        return 'L疑似标签', 0.0
    if pd.notna(r['sig_ac1']) and r['sig_ac1'] >= STATIC_AC1:
        return 'S静态身份', 0.0

    best = None
    for h in HORIZONS:
        if not r.get(f'h{h}_n_pool'):
            continue
        npool = r[f'h{h}_n_pool']
        cons = r[f'h{h}_n_consist'] / npool
        emin = r[f'h{h}_exc_min']
        emean = r[f'h{h}_exc_mean']
        if pd.isna(emin):
            continue
        key = (cons, emin, emean)
        if best is None or key > best[0]:
            best = (key, h, cons, emin, emean, npool)
    if best is None:
        return 'T0不可用', 0.0
    _, h, cons, emin, emean, npool = best
    ss_n, ss_of = r.get(f'h{h}_same_sign_n', 0), r.get(f'h{h}_same_sign_of', 0)
    ss = (ss_n / ss_of) if ss_of else 0.0
    dyn = r.get(f'h{h}_dyn_minus_static_mean', np.nan)
    score = (0.35 * cons + 0.30 * min(max(emin, 0) / 0.02, 1.0)
             + 0.20 * ss + 0.15 * (1.0 if pd.notna(dyn) and dyn > 0 else 0.0))
    score = round(float(score), 3)
    if cons >= 1.0 and emin >= 0.005 and emean >= 0.015:
        tier = 'A强'
    elif cons >= 1.0 and emin >= 0.001 and emean >= 0.005:
        tier = 'B中'
    elif cons >= 0.75 and emin >= 0.005 and emean >= 0.010:
        tier = 'B中'
    elif cons >= 0.75 and emean >= 0.002:
        tier = 'C弱'
    elif cons >= 0.5 and emean >= 0.005:
        tier = 'C弱'
    else:
        tier = 'D无效'
    if tier in ('A强', 'B中') and (pd.isna(dyn) or dyn <= 0):
        tier = tier + '|无动态增量'
    return tier, score


def cmd_grade(a):
    base = load_all()
    if not base:
        print('[ERROR] 无 eval_*.csv')
        sys.exit(1)
    summ = build_summary(base)
    res = summ.apply(classify, axis=1, result_type='expand')
    summ['tier'] = res[0]
    summ['score'] = res[1]
    summ = summ.sort_values(['score', 'h20_exc_mean'], ascending=False)
    fp = _out('eval_summary.csv')
    summ.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f'已写出 {fp} ({len(summ)} 行); 参与汇总的池: {list(base)}')

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 80)
    cols = ['signal', 'group', 'tier', 'score', 'coverage', 'sig_ac1',
            'h5_exc_mean', 'h5_exc_min', 'h5_n_raw_pos', 'h5_n_consist', 'h5_n_pool',
            'h20_exc_mean', 'h20_exc_min', 'h20_n_raw_pos', 'h20_n_consist', 'h20_n_pool',
            'h20_icir_mean', 'h20_topk_turnover_mean', 'h20_same_sign_n',
            'h20_same_sign_of', 'h20_dyn_minus_static_mean', 'h20_n_eff_symbols_mean']
    cols = [c for c in cols if c in summ.columns]
    print('\n===== 分级计数 =====')
    print(summ['tier'].value_counts().to_string())
    for t in ['A强', 'B中', 'C弱']:
        sub = summ[summ['tier'].str.startswith(t)]
        print(f'\n===== {t} ({len(sub)}) =====')
        print(sub[cols].head(25).to_string(index=False))
    print('\n===== S静态身份 (前 20) =====')
    print(summ[summ['tier'] == 'S静态身份'][cols].head(20).to_string(index=False))
    print('\n===== 各组最优 =====')
    idx = summ.groupby('group')['score'].idxmax()
    print(summ.loc[idx, cols].sort_values('score', ascending=False).to_string(index=False))


# ================================================================ redund ================================================================ #
def cmd_redund(a):
    pool = a.pool
    if pool not in POOLS or not os.path.exists(POOLS[pool]):
        print(f'[ERROR] 找不到池数据: {pool} -> {POOLS.get(pool)}')
        return
    _, panel = load_panel(POOLS[pool], 'day')
    panel = panel[panel.index.get_level_values('date') >= pd.Timestamp(START)]
    cands, num_cols, _ = build_cands(panel, with_derived=False)
    if not cands:
        print(f'[ERROR] 没有可用指标列; num_cols={len(num_cols)}')
        return
    ranks = {c: w.rank(axis=1, pct=True) for c, w in cands.items()}
    # 逐日截面秩 -> 展平为 (date, symbol) 长表, 列=指标
    R = pd.DataFrame({c: v.stack() for c, v in ranks.items()}, dtype='float32')
    R = R.loc[:, R.notna().mean() > 0.5]
    R = R.dropna(how='all')
    print(f'pool={pool}  参与相关性计算的指标 {R.shape[1]} 个, 展平后 {R.shape[0]} 行')
    C = R.corr(method='pearson')                  # 逐日截面秩 -> 拼平后相关

    pairs = []
    cols = list(C.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = C.iloc[i, j]
            if pd.notna(v):
                pairs.append({'a': cols[i], 'b': cols[j], 'corr': round(float(v), 3),
                              'abs_corr': round(abs(float(v)), 3),
                              'group_a': group_of(cols[i]), 'group_b': group_of(cols[j])})
    P = pd.DataFrame(pairs).sort_values('abs_corr', ascending=False)
    P.to_csv(_out(f'eval_redundancy_{pool}.csv'), index=False, encoding='utf-8-sig')
    print(f'\n===== 高度共线 (|corr| >= {THRESH}) 前 60 对 =====')
    print(P[P['abs_corr'] >= THRESH].head(60).to_string(index=False))
    print(f'\n|corr|>=0.90 的对数: {(P["abs_corr"] >= 0.90).sum()} / {len(P)}')
    print(f'|corr|>=0.80 的对数: {(P["abs_corr"] >= 0.80).sum()} / {len(P)}')

    print('\n===== 各指标"最像的邻居"(最强共线对象) =====')
    rows = []
    for c in cols:
        sub = P[(P['a'] == c) | (P['b'] == c)]
        if sub.empty:
            continue
        top = sub.iloc[0]
        other = top['b'] if top['a'] == c else top['a']
        rows.append({'signal': c, 'group': group_of(c), 'nearest': other,
                     'corr': top['corr'], 'abs_corr': top['abs_corr'],
                     'mean_abs_corr': round(float(sub['abs_corr'].mean()), 3)})
    N = pd.DataFrame(rows).sort_values('mean_abs_corr')
    N.to_csv(_out(f'eval_neighbors_{pool}.csv'), index=False, encoding='utf-8-sig')
    print(N.head(40).to_string(index=False))

    # 组内平均|相关|: 判断哪些家族内部信息重复最严重
    g = P.groupby('group_a')['abs_corr'].agg(['mean', 'count']).sort_values('mean')
    print('\n===== 各指标组内部平均 |corr| (越低 => 组内越互补) =====')
    print(g.to_string())


# ================================================================ neigh ================================================================ #
def cmd_neigh(a):
    pd.set_option('display.width', 240)
    pd.set_option('display.max_rows', 400)
    rows = []
    for p in GRADE_POOLS:
        fp = _out(f'eval_neighbors_{p}.csv')
        if not os.path.exists(fp):
            continue
        N = pd.read_csv(fp).set_index('signal')
        for s in FOCUS:
            if s in N.index:
                rows.append({'pool': p, 'signal': s, 'group': N.loc[s, 'group'],
                             'nearest': N.loc[s, 'nearest'],
                             'corr_to_nearest': N.loc[s, 'corr'],
                             'mean_abs_corr': N.loc[s, 'mean_abs_corr']})
    if not rows:
        print('[ERROR] 无 eval_neighbors_*.csv, 请先运行 redund 子命令')
        return
    D = pd.DataFrame(rows)
    print('===== 重点指标的冗余伴侣(跨池) =====')
    print(D.to_string(index=False))

    print('\n===== 同一指标在多池的平均 |corr| (越小越独立) =====')
    print(D.groupby('signal')['mean_abs_corr'].agg(['mean', 'min', 'max', 'count'])
          .sort_values('mean').to_string())

    print('\n===== 与重点指标 |corr| >= 0.85 的伙伴(各池) =====')
    for p in GRADE_POOLS:
        fp = _out(f'eval_redundancy_{p}.csv')
        if not os.path.exists(fp):
            continue
        P = pd.read_csv(fp)
        sub = P[((P['a'].isin(FOCUS)) | (P['b'].isin(FOCUS))) & (P['abs_corr'] >= 0.85)]
        print(f'\n--- {p} ({len(sub)} 对) ---')
        print(sub[['a', 'b', 'corr', 'group_a', 'group_b']].to_string(index=False))


# ================================================================ 入口 ================================================================ #
def _out(fname: str) -> str:
    d = os.path.join(HERE, 'output')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def main():
    ap = argparse.ArgumentParser(description='指标价值评估工具(只读)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('eval', help='对指定池全体指标做多角度评估')
    p1.add_argument('--pool', default='etf_3x', help='池名 or all')
    p1.add_argument('--derived', action='store_true', help='是否附加衍生基准信号')
    p1.set_defaults(func=cmd_eval)

    p2 = sub.add_parser('grade', help='汇总 eval_*.csv 并做价值分级')
    p2.set_defaults(func=cmd_grade)

    p3 = sub.add_parser('redund', help='指标两两截面秩相关/冗余簇')
    p3.add_argument('--pool', default='etf_3x', help='池名')
    p3.set_defaults(func=cmd_redund)

    p4 = sub.add_parser('neigh', help='重点指标的跨池冗余伴侣汇总')
    p4.set_defaults(func=cmd_neigh)

    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
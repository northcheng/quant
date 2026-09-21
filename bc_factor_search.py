# -*- coding: utf-8 -*-
"""bc_factor_search.py — 合并自 research 平铺模块(11 个: factor_research, signal_search, factor_mining, conditional_eval, indicator_eval, alpha_mining, factor_mining2, conditional_mining, a_stat_mining, factor_signal, export_pool_context).

生成: _dbg_build_bc.py 自动拼接 + AST 精确改名(同名冲突加模块前缀, 未经人工改动).
规则:
  - 值完全相同的重复常量仅保留首处定义;
  - 被跨模块 import 的符号保留原名, 私有冲突符号加模块前缀(见各段内改名注释);
  - 源模块保留于 research/ 目录未删除, 供新旧一致性对拍.
"""
from datetime import datetime
from itertools import product
from math import erf, sqrt
import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


# 原 research 模块目录(合并文件位于 git/quant/, 输出路径保持与源模块一致)
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research')


# ==========================================================================
# ==== 源自 factor_research.py ====  改名: main->fr_main
# ==========================================================================
ADX_N = 12              # add_adx_features(n=12, method='wilder')

RSI_N = 14              # add_rsi_features(n=14)

BB_N, BB_NDEV = 20, 2   # add_bb_features(n=20, ndev=2)

ICH_N = (9, 26, 52)     # add_ichimoku_features(n_short=9, n_medium=26, n_long=52, method='ta', is_shift=True)

KAMA_FAST = (10, 2, 30)  # cal_kama(n1=10, n2=2, n3=30)

KAMA_SLOW = (20, 4, 60)  # cal_kama(n1=20, n2=4, n3=60)

NORM_WINDOW, NORM_MINP = 252, 60  # normalize_causal, day interval

TREND_WEIGHTS = {'ichimoku_distance': 0.2, 'ichimoku_distance_change': 0.3,
                 'trend_score': 0.3, 'trend_score_change': 0.2}

PATTERN_FLAGS = ['超买超卖', '关键突破', '长线边界', '趋势转换', '趋势启动',
                 '区间波动', '触顶触底', '短期转向', '中期转向']

OBJECT_COLS = ['trend_score', 'trend_score_change']  # pkl 中为 object dtype, 需数值化

ROUND3_COLS = {'kama_fast', 'kama_slow', 'tankan', 'kijun', 'candle_gap_top', 'candle_gap_bottom'}

DEFAULT_FACTOR_COLS = [
    'trend_magnitude', 'trend_magnitude_alpha', 'trend_magnitude_change',
    'pattern_score', 'pattern_score_alpha', 'pattern_score_change',
    'trigger_score', 'position_score', 'boundary_score', 'break_score',
    'trend_score', 'trend_score_change',
    'candle_position_score', 'candle_pattern_score',
    'adx_value', 'adx_strength', 'rsi', 'rate',
]

def rma(series: pd.Series, n: int) -> pd.Series:
    """Wilder 平滑: SMA 种子于第 n 个样本, 之后 rma[i]=(rma[i-1]*(n-1)+v[i])/n; NaN 向前携带. (同 bc_technical_analysis.rma)"""
    arr = np.asarray(series, dtype=float)
    out = np.full(arr.shape, np.nan)
    if len(arr) >= n and n >= 1:
        out[n - 1] = np.nanmean(arr[:n])
        for i in range(n, len(arr)):
            out[i] = out[i - 1] if np.isnan(arr[i]) else (out[i - 1] * (n - 1) + arr[i]) / n
    return pd.Series(out, index=series.index)

def sda(series, zero_as=None) -> pd.Series:
    """同号累积: 符号翻转时重新计数, 0 值按 zero_as 延续当前符号的计数. (同 bc_technical_analysis.sda)"""
    s = pd.Series(series)
    lst = list(s.values)
    result = [lst[0]] if lst else []
    for i in range(1, len(lst)):
        cur_sign = np.sign(lst[i])
        cum_sign = np.sign(result[i - 1])
        if cur_sign == 0:
            if zero_as is None or result[i - 1] == 0:
                result.append(lst[i])
            else:
                result.append(result[i - 1] + zero_as * (1 if cum_sign == 1 else -1))
        elif cur_sign != cum_sign:
            result.append(lst[i])
        else:
            result.append(result[i - 1] + lst[i])
    return pd.Series(result, index=s.index)

def normalize_causal(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """因果归一化到 [0,1]: 滚动 min/max, warmup 用 expanding 补齐, span=0 记 0. (同 bc_technical_analysis.normalize_causal)"""
    s = series.astype(float)
    lo = s.rolling(window=window, min_periods=min_periods).min()
    hi = s.rolling(window=window, min_periods=min_periods).max()
    lo = lo.combine_first(s.expanding().min())
    hi = hi.combine_first(s.expanding().max())
    span = hi - lo
    return ((s - lo) / span.replace(0, np.nan)).fillna(0.0)

def calc_rate(df: pd.DataFrame) -> pd.Series:
    return df['Close'].pct_change(periods=1)

def calc_tr(df: pd.DataFrame) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    return pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)

def calc_rsi(df: pd.DataFrame, n: int = RSI_N) -> pd.Series:
    diff = df['Close'].diff(1)
    up = diff.clip(lower=0.0).fillna(0.0)
    down = (-diff).clip(lower=0.0).fillna(0.0)
    avg_up, avg_dn = rma(up, n), rma(down, n)
    denom = avg_up + avg_dn
    return (100 * avg_up / denom).where(denom != 0, 50.0)

def calc_adx(df: pd.DataFrame, n: int = ADX_N) -> dict:
    """wilder 路径. 注意: adx_value 并非教科书 ADX, 而是 EMA5(pdi-mdi); adx_strength 才是真 ADX."""
    h, l = df['High'], df['Low']
    tr = calc_tr(df)
    atr = rma(tr, n)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pdi = 100 * rma(pdm, n) / atr
    mdi = 100 * rma(mdm, n) / atr
    denom = pdi + mdi
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = pd.Series(np.where(denom > 0, 100 * (pdi - mdi).abs() / denom, np.nan), index=df.index)
    adx = rma(dx, n)
    adx_value = (pdi - mdi).ewm(span=5, min_periods=5).mean()
    return {'tr': tr, 'atr': atr, 'adx_strength': adx, 'adx_value': adx_value}

def calc_adi(df: pd.DataFrame) -> pd.Series:
    h, l, c, v = df['High'], df['Low'], df['Close'], df['Volume']
    clv = ((c - l) - (h - c)) / (h - l)
    return (clv.fillna(0.0) * v).cumsum()

def calc_bb(df: pd.DataFrame, n: int = BB_N, ndev: int = BB_NDEV) -> dict:
    close = df['Close']
    mavg = close.rolling(window=n, min_periods=n).mean()
    mstd = close.rolling(window=n, min_periods=n).std(ddof=0)
    return {'mavg': mavg, 'mstd': mstd,
            'bb_high_band': mavg + ndev * mstd, 'bb_low_band': mavg - ndev * mstd}

def calc_ichimoku(df: pd.DataFrame) -> dict:
    ns, nm, nl = ICH_N
    h, l = df['High'], df['Low']
    tankan = (h.rolling(ns, min_periods=0).max() + l.rolling(ns, min_periods=0).min()) / 2
    kijun = (h.rolling(nm, min_periods=0).max() + l.rolling(nm, min_periods=0).min()) / 2
    senkou_a = (tankan + kijun) / 2
    senkou_b = (h.rolling(nl, min_periods=0).max() + l.rolling(nl, min_periods=0).min()) / 2
    return {'tankan': tankan, 'kijun': kijun,
            'senkou_a': senkou_a.shift(nm), 'senkou_b': senkou_b.shift(nm)}

def calc_kama(df: pd.DataFrame, n1: int, n2: int, n3: int) -> pd.Series:
    close = df['Close']
    close_values = close.values
    vol = close.diff().abs()
    er_num = pd.Series(close_values, index=close.index).diff(n1).abs()
    er = (er_num / vol.rolling(n1).sum()).ffill()
    sc = ((er * (2.0 / (n2 + 1.0) - 2.0 / (n3 + 1.0)) + 2.0 / (n3 + 1.0)) ** 2.0).values
    kama = np.full(sc.size, np.nan)
    first_value = True
    for i in range(len(kama)):
        if np.isnan(sc[i]):
            continue
        if first_value:
            kama[i] = close_values[i]
            first_value = False
        else:
            kama[i] = kama[i - 1] + sc[i] * (close_values[i] - kama[i - 1])
    return pd.Series(kama, index=close.index)

def calc_trend_score(df: pd.DataFrame) -> pd.Series:
    adx_value = pd.to_numeric(df['adx_value'], errors='coerce')
    adx_strength = pd.to_numeric(df['adx_strength'], errors='coerce')
    return (adx_value.diff(1) + adx_strength.diff(1) * (adx_value > 0).replace({True: 1, False: -1})).round(2)

def calc_trend_magnitude(df: pd.DataFrame) -> pd.Series:
    tm = pd.Series(0.0, index=df.index)
    for col, w in TREND_WEIGHTS.items():
        v = pd.to_numeric(df[col], errors='coerce')
        alpha = normalize_causal(v.abs(), NORM_WINDOW, NORM_MINP)
        tm = tm + w * alpha * np.nan_to_num(np.sign(v))
    return tm

def calc_trigger_score(df: pd.DataFrame) -> pd.Series:
    return (df['break_up_score'] + df['support_score'] * 0.5
            + df['break_down_score'] + df['resistant_score'] * 0.5).round(2)

def calc_pattern_score(df: pd.DataFrame) -> pd.Series:
    flags = [pd.to_numeric(df[c], errors='coerce').fillna(0.0) for c in PATTERN_FLAGS if c in df.columns]
    return pd.concat(flags, axis=1).sum(axis=1).round(2) if flags else pd.Series(np.nan, index=df.index)

def compare_series(mine: pd.Series, theirs: pd.Series, tol_rel: float = 1e-6) -> dict:
    """逐行比对两条序列: 双 NaN 视为一致; 单边 NaN 视为不一致; 数值按相对容差判定."""
    mine = pd.to_numeric(mine, errors='coerce')
    theirs = pd.to_numeric(theirs, errors='coerce')
    mine, theirs = mine.align(theirs, join='inner')
    both_nan = mine.isna() & theirs.isna()
    comparable = ~both_nan
    a, b = mine[comparable], theirs[comparable]
    nan_mismatch = a.isna() ^ b.isna()
    ok = ~nan_mismatch
    if ok.any():
        diff = (a[ok] - b[ok]).abs()
        scale = np.maximum(1.0, np.maximum(a[ok].abs(), b[ok].abs()))
        ok.loc[ok] = (diff <= tol_rel * scale).values
    n_match = int(ok.sum()) + int(both_nan.sum())
    n_total = int(comparable.sum() + both_nan.sum())
    mismatch_idx = mine.index[comparable][~ok]
    first_diff = str(mismatch_idx[0].date()) if len(mismatch_idx) > 0 else ''
    valid_pair = a.notna() & b.notna()
    corr = float(a[valid_pair].corr(b[valid_pair])) if valid_pair.sum() >= 3 else np.nan
    max_diff = float((a[ok] - b[ok]).abs().max()) if ok.sum() > 0 else 0.0
    return {'n_rows': n_total, 'n_match': n_match,
            'match_pct': round(100.0 * n_match / n_total, 3) if n_total else np.nan,
            'max_abs_diff': max_diff, 'corr': corr, 'first_diff': first_diff}

def verdict_of(rec: dict) -> str:
    mp = rec.get('match_pct', np.nan)
    corr = rec.get('corr', np.nan)
    if pd.isna(mp):
        return 'N/A'
    if mp >= 99.9 and (pd.isna(corr) or corr >= 0.999):
        return 'PASS'
    if mp >= 97.0 and (pd.isna(corr) or corr >= 0.98):
        return 'WARN'
    return 'FAIL'

def audit_one(df: pd.DataFrame, target_col: str, mine: pd.Series, note: str = '') -> dict:
    if target_col not in df.columns:
        return {'target_col': target_col, 'note': '列不存在', 'verdict': 'MISSING'}
    if target_col in ROUND3_COLS:
        mine = pd.to_numeric(pd.Series(mine), errors='coerce').round(3)
        note = f'{note} [pkl侧round(3), 以round后值比对]'
    rec = compare_series(mine, df[target_col])
    rec.update({'target_col': target_col, 'recompute': note or target_col, 'verdict': verdict_of(rec)})
    return rec

def audit_raw_indicators(df: pd.DataFrame) -> list:
    recs = []
    adx = calc_adx(df)
    recs.append(audit_one(df, 'rate', calc_rate(df), 'Close.pct_change(1)'))
    recs.append(audit_one(df, 'tr', adx['tr'], 'max(H-L,|H-PC|,|L-PC|)'))
    recs.append(audit_one(df, 'atr', adx['atr'], f'Wilder RMA(tr, n={ADX_N}) [adx路径, 非n=14]'))
    recs.append(audit_one(df, 'rsi', calc_rsi(df), f'Wilder RSI(n={RSI_N})'))
    if 'rsi_signal' in df.columns:
        rsi = calc_rsi(df)
        sig = pd.Series('n', index=df.index)
        sig[rsi > 70] = 's'
        sig[rsi < 35] = 'b'  # 项目阈值: 上界70, 下界35(boundary=[35,70]的min为下界? 见备注)
        recs.append(audit_one(df, 'rsi_signal', sig, '边界信号 s/b/n'))
    recs.append(audit_one(df, 'adx_strength', adx['adx_strength'], f'Wilder ADX(n={ADX_N})'))
    recs.append(audit_one(df, 'adx_value', adx['adx_value'], 'EMA5(pdi-mdi) [自定义, 非教科书ADX]'))
    if 'adi' in df.columns:
        recs.append(audit_one(df, 'adi', calc_adi(df), 'CLV*Volume 累计和'))
    bb = calc_bb(df)
    for col, note in [('mavg', f'SMA({BB_N})'), ('mstd', f'rolling std ddof=0 (n={BB_N})'),
                      ('bb_high_band', f'mavg+{BB_NDEV}*mstd'), ('bb_low_band', f'mavg-{BB_NDEV}*mstd')]:
        recs.append(audit_one(df, col, bb[col], note))
    ich = calc_ichimoku(df)
    for col, note in [('tankan', f'({ICH_N[0]}周期高+低)/2'), ('kijun', f'({ICH_N[1]}周期高+低)/2'),
                      ('senkou_a', '(tankan+kijun)/2 前移26'), ('senkou_b', f'({ICH_N[2]}周期高+低)/2 前移26')]:
        recs.append(audit_one(df, col, ich[col], note))
    recs.append(audit_one(df, 'kama_fast', calc_kama(df, *KAMA_FAST), f'KAMA{KAMA_FAST}'))
    recs.append(audit_one(df, 'kama_slow', calc_kama(df, *KAMA_SLOW), f'KAMA{KAMA_SLOW}'))
    return recs

def audit_score_aggregation(df: pd.DataFrame) -> list:
    recs = []
    # trend_score 及其成分
    recs.append(audit_one(df, 'trend_score', calc_trend_score(df),
                          'adx_value_diff + adx_strength_diff*sign(adx_value>0), round2'))
    if 'trend_score_change' in df.columns:
        recs.append(audit_one(df, 'trend_score_change',
                              pd.to_numeric(df['trend_score'], errors='coerce').diff(1).round(2), 'trend_score.diff, round2'))
    # 四个成分的 *_alpha 归一化
    for col in TREND_WEIGHTS:
        if col in df.columns and f'{col}_alpha' in df.columns:
            v = pd.to_numeric(df[col], errors='coerce')
            recs.append(audit_one(df, f'{col}_alpha', normalize_causal(v.abs(), NORM_WINDOW, NORM_MINP),
                                  f'normalize_causal(|{col}|, {NORM_WINDOW}/{NORM_MINP})'))
    # trend_magnitude 链
    recs.append(audit_one(df, 'trend_magnitude', calc_trend_magnitude(df),
                          'Σ w*alpha*sign(成分) [w=0.2/0.3/0.3/0.2]'))
    if 'trend_magnitude_day' in df.columns:
        tm = pd.to_numeric(df['trend_magnitude'], errors='coerce')
        # 显式携带 DatetimeIndex: np.nan_to_num 返回 ndarray, 若直接进 sda 会被 RangeIndex 包装导致对齐失败
        tm_sign = pd.Series(np.nan_to_num(np.sign(tm.values)), index=tm.index)
        recs.append(audit_one(df, 'trend_magnitude_day', sda(tm_sign, zero_as=1),
                              'sda(sign(trend_magnitude), zero_as=1) [同号计数器]'))
    # trigger / pattern
    for col in ['break_up_score', 'support_score', 'break_down_score', 'resistant_score']:
        if col not in df.columns:
            return recs
    recs.append(audit_one(df, 'trigger_score', calc_trigger_score(df),
                          'break_up + 0.5*support + break_down + 0.5*resistant, round2'))
    recs.append(audit_one(df, 'pattern_score', calc_pattern_score(df), 'Σ 9个pattern旗标(±权重), round2'))
    if 'pattern_score' in df.columns:
        recs.append(audit_one(df, 'pattern_score_alpha',
                              normalize_causal(pd.to_numeric(df['pattern_score'], errors='coerce').abs(), NORM_WINDOW, NORM_MINP),
                              f'normalize_causal(|pattern_score|, {NORM_WINDOW}/{NORM_MINP})'))
    return recs

def daily_spearman(factor_wide: pd.DataFrame, fwd_wide: pd.DataFrame, min_cs: int) -> pd.Series:
    """按日截面 Spearman 秩相关 (因子值 t 日, 前瞻收益 t->t+h)."""
    idx = factor_wide.index.intersection(fwd_wide.index)
    f, r = factor_wide.loc[idx], fwd_wide.loc[idx]
    out = {}
    for dt in idx:
        fd, rd = f.loc[dt], r.loc[dt]
        m = fd.notna() & rd.notna()
        if int(m.sum()) >= min_cs:
            ic = fd[m].rank().corr(rd[m].rank())
            if pd.notna(ic):
                out[dt] = float(ic)
    return pd.Series(out, dtype=float)

def ic_summary(ic_series: pd.Series) -> dict:
    n = len(ic_series)
    if n == 0:
        return {'n_days': 0, 'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan,
                't_stat': np.nan, 'pos_rate': np.nan}
    mean, std = ic_series.mean(), ic_series.std(ddof=1)
    icir = mean / std if std and std > 0 else np.nan
    t = mean / std * np.sqrt(n) if std and std > 0 else np.nan
    return {'n_days': n, 'ic_mean': round(mean, 4), 'ic_std': round(std, 4),
            'icir': round(icir, 3) if pd.notna(icir) else np.nan,
            't_stat': round(t, 2) if pd.notna(t) else np.nan,
            'pos_rate': round((ic_series > 0).mean(), 3)}

def quantile_returns(factor_wide: pd.DataFrame, fwd_wide: pd.DataFrame, q: int, min_cs: int) -> pd.DataFrame:
    """按日截面分位分桶(1=最低, q=最高), 各桶的前瞻收益均值与样本数, 及多空差."""
    idx = factor_wide.index.intersection(fwd_wide.index)
    f, r = factor_wide.loc[idx], fwd_wide.loc[idx]
    data = f.stack().to_frame('f').join(r.stack().to_frame('r'), how='inner')
    if data.empty:
        return pd.DataFrame()
    data = data.dropna(subset=['f', 'r'])
    # 按日 rank(pct) -> 桶
    data['rank_pct'] = data.groupby(level=0)['f'].rank(pct=True)
    data['bucket'] = np.minimum((data['rank_pct'] * q).apply(np.ceil).astype(int), q)
    grp = data.groupby('bucket')['r']
    out = pd.DataFrame({'mean_fwd_ret': grp.mean().round(5), 'n': grp.size()})
    if len(out) >= 2:
        out.loc['LS(Q%d-Q1)' % q] = [round(out['mean_fwd_ret'].iloc[-1] - out['mean_fwd_ret'].iloc[0], 5), 0]
    return out

def extract_trades(df: pd.DataFrame, symbol: str, gate: pd.Series, score_cols: list) -> list:
    """状态机重放: gate(布尔状态)连续为 True 的段落 -> 合成交易. 信号日收盘确认, 次日开盘成交."""
    on = gate.fillna(False).astype(bool)
    if len(on) == 0:
        return []
    grp = (on != on.shift()).cumsum()
    trades = []
    for _, seg in df.groupby(grp):
        if not bool(on.loc[seg.index[0]]):
            continue
        e, x = seg.index[0], seg.index[-1]
        pos_e, pos_x = df.index.get_loc(e), df.index.get_loc(x)
        if e == x:
            continue  # 单日门段: 入场执行日=出场执行日, 因果上不会成交(ret 恒 0 的退化交易), 跳过
        entry_pos = pos_e + 1
        if entry_pos >= len(df):
            continue  # 信号日为最后一日, 无法次日开盘入场
        exit_pos = pos_x + 1
        completed = exit_pos < len(df)
        entry_px = df['Open'].iloc[entry_pos]
        if completed:
            exit_px = df['Open'].iloc[exit_pos]
            path = df.iloc[entry_pos:exit_pos]
        else:
            exit_px = df['Close'].iloc[-1]
            path = df.iloc[entry_pos:]
        ret = exit_px / entry_px - 1
        mfe = path['High'].max() / entry_px - 1 if len(path) > 0 else 0.0
        mae = path['Low'].min() / entry_px - 1 if len(path) > 0 else 0.0
        row = {'symbol': symbol, 'entry_signal': e, 'exit_signal': x,
               'entry_exec': df.index[entry_pos],
               'exit_exec': df.index[exit_pos] if completed else df.index[-1],
               'completed': completed, 'days': len(path),
               'ret': ret, 'mfe': mfe, 'mae': mae}
        for c in score_cols:
            row[f'score_{c}'] = df[c].iloc[pos_e] if c in df.columns else np.nan
        trades.append(row)
    return trades

def event_stats(trades: pd.DataFrame) -> dict:
    done = trades[trades['completed']] if 'completed' in trades.columns else trades
    if len(done) == 0:
        return {'n_trades': 0}
    wins, losses = done[done['ret'] > 0], done[done['ret'] <= 0]
    avg_w = wins['ret'].mean() if len(wins) else np.nan
    avg_l = losses['ret'].mean() if len(losses) else np.nan
    payoff = (avg_w / abs(avg_l)) if (len(wins) and len(losses) and avg_l != 0) else np.nan
    return {
        'n_trades': int(len(done)),
        'n_open': int((~trades['completed']).sum()) if 'completed' in trades.columns else 0,
        'win_rate': round(len(wins) / len(done), 3),
        'avg_ret': round(done['ret'].mean(), 4),
        'median_ret': round(done['ret'].median(), 4),
        'avg_win': round(avg_w, 4) if pd.notna(avg_w) else np.nan,
        'avg_loss': round(avg_l, 4) if pd.notna(avg_l) else np.nan,
        'payoff_ratio': round(payoff, 2) if pd.notna(payoff) else np.nan,
        'avg_days': round(done['days'].mean(), 1),
        'median_days': float(done['days'].median()),
        'avg_mfe': round(done['mfe'].mean(), 3),
        'avg_mae': round(done['mae'].mean(), 3),
    }

def event_ic(trades: pd.DataFrame, score_col: str, min_events: int = 5) -> dict:
    """事件级IC: 同一入场信号日的入场分数与交易最终收益的截面秩相关."""
    done = trades[trades['completed']] if 'completed' in trades.columns else trades
    col = f'score_{score_col}'
    if col not in done.columns or len(done) < min_events:
        return {'n_groups': 0}
    ics = {}
    for dt, g in done.groupby('entry_signal'):
        gg = g[[col, 'ret']].dropna()
        if len(gg) >= min_events:
            ic = gg[col].rank().corr(gg['ret'].rank())
            if pd.notna(ic):
                ics[dt] = float(ic)
    s = pd.Series(ics, dtype=float)
    if len(s) == 0:
        return {'n_groups': 0}
    return {'n_groups': len(s), 'event_ic_mean': round(s.mean(), 4),
            'event_ic_ir': round(s.mean() / s.std(), 3) if s.std() > 0 else np.nan,
            'pos_rate': round((s > 0).mean(), 3)}

def load_panel(pkl_path: str, interval: str) -> tuple:
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f'pkl 内容不是 dict: {type(raw)}')
    frames = []
    for key, df in raw.items():
        if df is None or len(df) == 0:
            continue
        symbol = key[:-len(f'_{interval}')] if key.endswith(f'_{interval}') else key
        tmp = df.copy()
        for c in OBJECT_COLS:
            if c in tmp.columns:
                tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
        tmp['symbol'] = symbol
        frames.append(tmp)
    panel = pd.concat(frames).set_index('symbol', append=True).swaplevel(0, 1).sort_index()
    panel.index.names = ['symbol', 'date']
    return raw, panel

def fr_main():
    ap = argparse.ArgumentParser(description='独立只读研究: 指标审计 + 条件IC + 事件研究')
    ap.add_argument('--pool', default='etf_3x', help='池名, 默认 etf_3x')
    ap.add_argument('--interval', default='day', help='数据频率, 默认 day')
    ap.add_argument('--pkl-dir', default=os.path.join(os.path.expanduser('~'), 'quant'), help='pkl 所在目录')
    ap.add_argument('--start', default=None, help='分析起始日(仅截取研究窗口, 不影响审计)')
    ap.add_argument('--end', default=None, help='分析结束日')
    ap.add_argument('--gate-col', default='trend_magnitude_day', help='门/状态列, >0 视为开')
    ap.add_argument('--horizons', default='1,5,10,20,60', help='IC 前瞻周期列表')
    ap.add_argument('--min-cs', type=int, default=8, help='IC 每日最小截面样本数')
    ap.add_argument('--q', type=int, default=5, help='分位数分桶数')
    ap.add_argument('--q-horizon', type=int, default=20, help='分位数单调性使用的前瞻周期')
    ap.add_argument('--factors', default=None, help='逗号分隔的因子列(默认内置清单)')
    ap.add_argument('--skip-audit', action='store_true')
    ap.add_argument('--skip-ic', action='store_true')
    ap.add_argument('--skip-events', action='store_true')
    args = ap.parse_args()

    pkl_path = os.path.join(args.pkl_dir, f'{args.pool}_{args.interval}_ta_data.pkl')
    if not os.path.exists(pkl_path):
        print(f'[ERROR] 找不到 pkl: {pkl_path}')
        sys.exit(1)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output', f'{args.pool}_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    raw, panel = load_panel(pkl_path, args.interval)
    symbols = sorted(panel.index.get_level_values('symbol').unique())
    dates = panel.index.get_level_values('date')
    print(f'== 数据 == {args.pool}: {len(symbols)} 标的, {len(raw)} 键, '
          f'{dates.min().date()} ~ {dates.max().date()}, panel {len(panel)} 行')

    # 分析窗口(IC/事件用; 审计始终全窗, 以暴露 warmup 段问题)
    study = panel
    if args.start:
        study = study[study.index.get_level_values('date') >= pd.Timestamp(args.start)]
    if args.end:
        study = study[study.index.get_level_values('date') <= pd.Timestamp(args.end)]

    gate_wide = pd.to_numeric(study[args.gate_col], errors='coerce').unstack('symbol') > 0
    coverage = float(gate_wide.values.mean())
    print(f'== 门 == {args.gate_col} > 0 的样本占比: {coverage:.1%}')

    report_lines = []
    report_lines.append(f'factor_research 报告 | pool={args.pool} | run={run_id}')
    report_lines.append(f'数据: {len(symbols)} 标的, {dates.min().date()}~{dates.max().date()}, panel {len(panel)} 行; '
                        f'研究窗口 {study.index.get_level_values("date").min().date()}~{study.index.get_level_values("date").max().date()}')
    report_lines.append(f'门: {args.gate_col} > 0, 覆盖率 {coverage:.1%}')

    # ---------------- Layer 0: 审计 ----------------
    if not args.skip_audit:
        print('\n== Layer 0: 指标正确性审计(独立重算 vs pkl) ==')
        raw_recs, score_recs = [], []
        for key, df in raw.items():
            if df is None or len(df) == 0:
                continue
            symbol = key[:-len(f'_{args.interval}')] if key.endswith(f'_{args.interval}') else key
            work = df.copy()
            for c in OBJECT_COLS:
                if c in work.columns:
                    work[c] = pd.to_numeric(work[c], errors='coerce')
            for rec in audit_raw_indicators(work):
                rec['symbol'] = symbol
                raw_recs.append(rec)
            for rec in audit_score_aggregation(work):
                rec['symbol'] = symbol
                score_recs.append(rec)

        for name, recs in [('audit_raw', raw_recs), ('audit_scores', score_recs)]:
            t = pd.DataFrame(recs)
            if t.empty:
                continue
            t.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False, encoding='utf-8-sig')
            agg = t.groupby(['target_col', 'recompute']).agg(
                symbols=('symbol', 'count'), min_match=('match_pct', 'min'),
                mean_match=('match_pct', 'mean'), n_fail=('verdict', lambda s: int((s != 'PASS').sum()))).reset_index()
            agg = agg.sort_values('min_match')
            print(f'\n[{name}] 按指标聚合(最差优先):')
            print(agg.to_string(index=False))
            report_lines.append(f'\n== {name} ==\n{agg.to_string(index=False)}')
            fails = t[t['verdict'] == 'FAIL']
            if len(fails) > 0:
                print(f'\n[FAIL 明细] {len(fails)} 条:')
                cols = [c for c in ['symbol', 'target_col', 'match_pct', 'max_abs_diff', 'corr', 'first_diff'] if c in fails.columns]
                print(fails[cols].to_string(index=False))
                report_lines.append(f'\n[FAIL 明细]\n{fails[cols].to_string(index=False)}')

    # ---------------- Layer 1: 条件IC ----------------
    if not args.skip_ic:
        print('\n== Layer 1: 多周期IC(全样本 & 门内) ==')
        horizons = [int(x) for x in args.horizons.split(',')]
        factor_cols = [c.strip() for c in args.factors.split(',')] if args.factors else DEFAULT_FACTOR_COLS
        factor_cols = [c for c in factor_cols if c in study.columns]
        px = study['Close'].unstack('symbol').sort_index()
        rows_full, rows_gate, qtables = [], [], {}
        for h in horizons:
            fwd = (px.shift(-h) / px - 1).dropna(how='all')
            for col in factor_cols:
                fw = pd.to_numeric(study[col], errors='coerce').unstack('symbol').sort_index()
                s_full = daily_spearman(fw, fwd, args.min_cs)
                r = {'horizon': h, 'factor': col}
                r.update(ic_summary(s_full))
                rows_full.append(r)
                fg = fw.where(gate_wide.reindex_like(fw).fillna(False))
                s_gate = daily_spearman(fg, fwd, args.min_cs)
                r2 = {'horizon': h, 'factor': col}
                r2.update(ic_summary(s_gate))
                rows_gate.append(r2)
                if h == args.q_horizon:
                    qtables[col] = quantile_returns(fg, fwd, args.q, args.min_cs)
        t_full, t_gate = pd.DataFrame(rows_full), pd.DataFrame(rows_gate)
        t_full.to_csv(os.path.join(out_dir, 'ic_full.csv'), index=False, encoding='utf-8-sig')
        t_gate.to_csv(os.path.join(out_dir, 'ic_gate.csv'), index=False, encoding='utf-8-sig')
        for name, t in [('IC-全样本', t_full), ('IC-门内', t_gate)]:
            key = t[t['horizon'].isin([1, 10, 20, 60])] if len(t) else t
            piv = key.pivot(index='factor', columns='horizon', values='ic_mean')
            print(f'\n[{name}] ic_mean(horizon):')
            print(piv.round(4).to_string())
            piv_ir = key.pivot(index='factor', columns='horizon', values='icir')
            print(f'\n[{name}] ICIR:')
            print(piv_ir.round(2).to_string())
            report_lines.append(f'\n== {name} ==\nic_mean:\n{piv.round(4).to_string()}\nICIR:\n{piv_ir.round(2).to_string()}')
        with open(os.path.join(out_dir, 'quantile_tables.json'), 'w', encoding='utf-8') as f:
            json.dump({k: v.reset_index().to_dict(orient='records') for k, v in qtables.items() if len(v)},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f'\n[分位数单调性] horizon={args.q_horizon}, 门内样本, 前5个因子:')
        for col, qt in list(qtables.items())[:5]:
            if len(qt):
                print(f'  {col}: {qt["mean_fwd_ret"].to_dict()}')

    # ---------------- Layer 2: 事件研究 ----------------
    if not args.skip_events:
        print('\n== Layer 2: 事件研究(门状态重放 -> 合成交易) ==')
        score_cols = [c for c in DEFAULT_FACTOR_COLS if c in study.columns]
        all_trades = []
        for symbol in symbols:
            sub = study.xs(symbol, level='symbol').sort_index()
            gate = pd.to_numeric(sub[args.gate_col], errors='coerce') > 0
            all_trades += extract_trades(sub, symbol, gate, score_cols)
        trades = pd.DataFrame(all_trades)
        if len(trades) == 0:
            print(f'[WARN] 门 {args.gate_col} > 0 未重放出任何交易(短样本下门可能几乎常关或整段常开)')
            report_lines.append('\n== Layer 2: 事件研究 ==\n无合成交易')
        else:
            trades.to_csv(os.path.join(out_dir, 'trades.csv'), index=False, encoding='utf-8-sig')
            st = event_stats(trades)
            print(f'  事件统计(仅 completed 交易): {st}')
            report_lines.append(f'\n== Layer 2: 事件统计 ==\n{json.dumps(st, ensure_ascii=False, default=str)}')
            per_symbol = trades.groupby('symbol').agg(
                n=('ret', 'count'), win=('ret', lambda s: (s > 0).mean()), avg_ret=('ret', 'mean'))
            print('\n[分标的] 交易数/胜率/平均收益:')
            print(per_symbol.round(4).to_string())
            report_lines.append(f'\n== Layer 2: 分标的 ==\n{per_symbol.round(4).to_string()}')
            ic_rows = []
            for col in score_cols:
                r = {'factor': col}
                r.update(event_ic(trades, col))
                ic_rows.append(r)
            t_ev = pd.DataFrame(ic_rows)
            t_ev.to_csv(os.path.join(out_dir, 'event_ic.csv'), index=False, encoding='utf-8-sig')
            if len(t_ev) and 'n_groups' in t_ev.columns:
                print('\n[event_ic] 事件级 IC(同日入场分数 vs 交易最终收益的截面秩相关):')
                print(t_ev.to_string(index=False))
                report_lines.append(f'\n== event_ic ==\n{t_ev.to_string(index=False)}')

    # ---------------- 收尾 ----------------
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')
    print('  ' + ', '.join(sorted(os.listdir(out_dir))))


# ==========================================================================
# ==== 源自 signal_search.py ====  改名: main->ss_main, SUSPECT_PAT->ss_SUSPECT_PAT
# ==========================================================================
ss_SUSPECT_PAT = ('label', 'action', 'signal', 'pos_label', 'neg_label', '_day')

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

def ss_main():
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
    out_dir = os.path.join(HERE, 'output',
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
                                ('疑似标签' if any(p in name.lower() for p in ss_SUSPECT_PAT) else 'panel列'))
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


# ==========================================================================
# ==== 源自 factor_mining.py ====  改名: main->fm_main
# ==========================================================================
EPS = 1e-12

def _roll_reg_stats(log_close: pd.DataFrame, n: int) -> dict:
    """对 log(close) 做 n 日滚动线性回归(对时间轴), 返回 R² 与斜率的 t 统计量.

    窗口内 x 为连续整数, 可用滚动和精确展开:
      r = (n*Sxy - Sx*Sy) / sqrt((n*Sxx - Sx²)(n*Syy - Sy²))
      t_stat = r * sqrt(n-2) / sqrt(1-r²)   (符号与斜率方向一致)
    """
    T = len(log_close)
    tser = pd.Series(np.arange(T, dtype=float), index=log_close.index)
    y = log_close
    Sy = y.rolling(n).sum()
    Sxy = y.mul(tser, axis=0).rolling(n).sum()
    Sxx = tser.pow(2).rolling(n).sum()
    Syy = (y * y).rolling(n).sum()
    Sx = tser.rolling(n).sum()
    # 注意: Sx/Sxx 是按日期索引的 Series, 与 (日期×symbol) DataFrame 相乘必须 axis=0 按行广播,
    # 否则 Series 日期索引会去对齐 DataFrame 的 symbol 列名, 产出全 NaN 的并集列矩阵.
    num = n * Sxy - Sy.mul(Sx, axis=0)
    den = np.sqrt((n * Syy - Sy ** 2).mul(n * Sxx - Sx ** 2, axis=0))
    r = (num / den).clip(-1.0, 1.0)
    r2 = r ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat = r * np.sqrt(n - 2) / np.sqrt(np.clip(1.0 - r2, EPS, None))
    return {'r2': r2, 'tstat': tstat}

def build_mined(panel: pd.DataFrame) -> dict:
    """构造 38 个挖矿候选因子(全部只用 t 及之前数据, 无前视)."""
    close = panel['Close'].unstack('symbol').sort_index()
    open_ = panel['Open'].unstack('symbol').sort_index()
    high = panel['High'].unstack('symbol').sort_index()
    low = panel['Low'].unstack('symbol').sort_index()
    volume = panel['Volume'].unstack('symbol').sort_index()
    ret1 = close.pct_change()
    log_close = np.log(close.where(close > 0))
    out = {}

    # ---- A. 趋势效率/质量 ----
    for n in (10, 20, 60):
        path = close.diff().abs().rolling(n).sum()
        out[f'F_er{n}'] = (close - close.shift(n)).abs() / path.replace(0, np.nan)
    for n in (20, 60, 250):
        st = _roll_reg_stats(log_close, n)
        out[f'F_r2_{n}'] = st['r2']
        out[f'F_slopet{n}'] = st['tstat']

    # ---- B. 隔夜/日内结构 ----
    overnight = open_ / close.shift(1) - 1.0            # t 日跳空(前收 -> 今开)
    intraday = close / open_ - 1.0                      # t 日日内(今开 -> 今收)
    out['F_overn20'] = overnight.rolling(20).mean()
    out['F_overn60'] = overnight.rolling(60).mean()
    out['F_intrad20'] = intraday.rolling(20).mean()
    out['F_gapabs20'] = overnight.abs().rolling(20).mean()
    rng = (high - low).replace(0, np.nan)
    out['F_closepos20'] = ((close - low) / rng).rolling(20).mean()   # 收盘在日内区间位置(买压)

    # ---- C. 波动结构 ----
    out['F_volts'] = ret1.rolling(20).std() / ret1.rolling(60).std().replace(0, np.nan)  # 波动期限结构
    dn = ret1.where(ret1 < 0, 0.0)
    out['F_dnvol20'] = dn.rolling(20).std() / ret1.rolling(20).std().replace(0, np.nan)  # 下行波动占比
    out['F_kurt60'] = ret1.rolling(60).kurt()
    vol20 = ret1.rolling(20).std()
    out['F_vovol20'] = vol20.rolling(20).std() / vol20.rolling(20).mean().abs().replace(0, np.nan)
    dd60 = close / close.rolling(60, min_periods=20).max() - 1.0
    out['F_dd60'] = dd60
    out['F_ulcer60'] = (dd60 ** 2).rolling(60, min_periods=20).mean().pow(0.5)
    out['F_range20'] = (high - low).rolling(20).mean() / close    # 日内振幅/价格

    # ---- D. 流动性/量能 ----
    dv = close * volume                                     # 成交额(近似)
    out['F_amihud20'] = (ret1.abs() / dv.replace(0, np.nan)).rolling(20).mean() * 1e6
    out['F_cvcorr20'] = ret1.rolling(20).corr(volume.diff())      # 量价同向性
    obv_cs = (np.sign(ret1.fillna(0.0)) * volume.fillna(0.0)).cumsum()
    out['F_obv20'] = (obv_cs - obv_cs.shift(20)) / volume.rolling(20).sum().replace(0, np.nan)
    upv = volume.where(ret1 > 0, 0.0)
    dnv = volume.where(ret1 < 0, 0.0)
    out['F_updnvol20'] = upv.rolling(20).sum() / dnv.rolling(20).sum().replace(0, np.nan)  # 吸筹/派发
    v20m = volume.rolling(20).mean().abs()
    out['F_volstab20'] = -volume.rolling(20).std() / v20m.replace(0, np.nan)
    out['F_dvmom20'] = dv.rolling(20).mean() / dv.rolling(60).mean().replace(0, np.nan)    # 关注度趋势

    # ---- E. 价格路径形态 ----
    hh250 = close.rolling(250, min_periods=60).max()
    ll250 = close.rolling(250, min_periods=60).min()
    out['F_hl52pos'] = (close - ll250) / (hh250 - ll250).replace(0, np.nan)
    body = (close - open_).abs()
    out['F_body20'] = (body / rng).rolling(20).mean()
    upshad = high - pd.DataFrame(np.maximum(close.values, open_.values), index=close.index, columns=close.columns)
    dnshad = pd.DataFrame(np.minimum(close.values, open_.values), index=close.index, columns=close.columns) - low
    out['F_upshad20'] = (upshad / rng).rolling(20).mean()
    out['F_dnshad20'] = (dnshad / rng).rolling(20).mean()
    # 连续同向天数(带符号: 上行连涨为正, 下行连跌为负; 缺口/停牌处归零)
    sign = np.where(np.isnan(ret1.values), 0, np.where(ret1.values > 0, 1, -1)).astype(np.int8)
    sdf = pd.DataFrame(sign, index=close.index, columns=close.columns)
    newgrp = (sdf != sdf.shift(1)) | (sdf == 0)
    idxarr = np.repeat(np.arange(len(sdf), dtype=float)[:, None], sdf.shape[1], axis=1)
    first = pd.DataFrame(np.where(newgrp.values, idxarr, np.nan), index=sdf.index, columns=sdf.columns).ffill()
    streak = (idxarr - first.values) + 1.0
    out['F_streak'] = pd.DataFrame(sign * streak, index=close.index, columns=close.columns)

    # ---- F. 相对池强度(池等权指数, 数据仅含 t 及之前) ----
    pool_ret = ret1.mean(axis=1)
    alpha = ret1.sub(pool_ret, axis=0)
    out['F_alpha20'] = alpha.rolling(20).mean()
    out['F_alpha60'] = alpha.rolling(60).mean()
    var_pool = pool_ret.rolling(60).var().replace(0, np.nan)
    beta60 = ret1.rolling(60).cov(pool_ret).div(var_pool, axis=0)
    out['F_beta60'] = beta60
    resid = ret1.sub(beta60.mul(pool_ret, axis=0))
    out['F_idiovol60'] = resid.rolling(60).std()

    # ---- G. 动量变体 ----
    out['F_mom121'] = close.shift(21) / close.shift(252) - 1.0     # 12-1 动量(剥离近月反转)
    out['F_momaccel'] = close.pct_change(20) - close.pct_change(120)  # 动量加速
    dd120 = close / close.rolling(120, min_periods=40).max() - 1.0
    ulcer120 = (dd120 ** 2).rolling(120, min_periods=40).mean().pow(0.5)
    out['F_momdd120'] = close.pct_change(120) / (ulcer120 + 0.02)   # 回撤调整动量

    return {k: v for k, v in out.items() if isinstance(v, pd.DataFrame)}

def fm_main():
    ap = argparse.ArgumentParser(description='独立只读研究: 新因子挖矿(多 horizon 可交易口径筛查)')
    ap.add_argument('--pool', default='etf_3x')
    ap.add_argument('--interval', default='day')
    ap.add_argument('--pkl-dir', default=os.path.join(os.path.expanduser('~'), 'quant'))
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl 路径(优先)')
    ap.add_argument('--start', default='2021-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--horizons', default='5,10,20,60')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--min-cs', type=int, default=10)
    ap.add_argument('--validate-top', type=int, default=8,
                    help='对每个周期 excess_k 前 N 个因子做真伪检验(静态身份/分半/OOS静态对照)')
    ap.add_argument('--signals', default=None, help='逗号分隔, 只筛指定因子(交互式深挖)')
    args = ap.parse_args()

    pkl_path = args.pkl_path or os.path.join(args.pkl_dir, f'{args.pool}_{args.interval}_ta_data.pkl')
    if not os.path.exists(pkl_path):
        print(f'[ERROR] 找不到 pkl: {pkl_path}')
        sys.exit(1)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output',
                           f'{args.pool}_mine_{run_id}')
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

    cands = {k: v.reindex(index=open_wide.index, columns=open_wide.columns)
             for k, v in build_mined(panel).items()}
    cands = {k: v for k, v in cands.items() if v.notna().sum().sum() > 0}
    print(f'== 挖矿因子 == 共 {len(cands)} 个: {sorted(cands)}')
    if args.signals:
        keep = [x.strip() for x in args.signals.split(',') if x.strip()]
        missing = [x for x in keep if x not in cands]
        cands = {k: v for k, v in cands.items() if k in keep}
        if missing:
            print(f'  [警告] 未找到: {missing}')
        print(f'== 仅筛指定因子({len(cands)}) ==')

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
                rows.append(rec)
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        cols = ['signal', 'excess_k', 'topk_ret', 'pool_ret', 'ls_spread',
                'ic_mean', 'icir', 't_stat', 'ic_pos_rate', 'topk_turnover', 'n_days']
        df = df[cols].sort_values('excess_k', ascending=False)
        df.to_csv(os.path.join(out_dir, f'mine_h{h}.csv'), index=False, encoding='utf-8-sig')
        all_rows.append((h, df))
        print(f'\n===== h={h}: 按 excess_k 排序 全量 {len(df)} 因子 =====')
        print(df.to_string(index=False))

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
                print('dyn_test_exc=后半段实际超额; static_sig=静态退化版; dyn_minus_static>0 才有动态增量')
                print(vdf.to_string(index=False))
                all_val.append((h, vdf))

    # 报告
    lines = [f'factor_mining 报告 | pool={args.pool} | run={run_id}',
             f'窗口 {dts.min().date()}~{dts.max().date()}, {len(symbols)} 标的',
             f'口径: 信号日收盘 -> 次日开盘入场 -> 持有h日开盘出场; excess_k = top{args.top_k}均值 - 池等权均值',
             f'因子数: {len(cands)}',
             '']
    for h, df in all_rows:
        lines.append(f'== h={h} 全量排序 ==')
        lines.append(df.to_string(index=False))
        lines.append('')
    for h, vdf in all_val:
        lines.append(f'== h={h} 真伪检验 ==')
        lines.append(vdf.to_string(index=False))
        lines.append('')
    if len(all_rows) >= 2:
        sets = [set(df.head(15)['signal']) for _, df in all_rows]
        common = set.intersection(*sets)
        lines.append(f'== 各周期均进 Top15 的因子({len(common)}) ==')
        lines.append(', '.join(sorted(common)))
        print(f'\n===== 各周期均进 Top15 的因子({len(common)}) =====')
        for h, df in all_rows:
            sub = df[df['signal'].isin(common)][['signal', 'excess_k', 'icir', 'topk_turnover']]
            print(f'-- h={h} --')
            print(sub.to_string(index=False))
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')


# ==========================================================================
# ==== 源自 conditional_eval.py ====  改名: main->ce_main, eval_pool->ce_eval_pool, POOLS->ce_POOLS, HORIZONS->ce_HORIZONS
# ==========================================================================
_PKL_DIR = os.path.join(os.path.expanduser('~'), 'quant')   # 本地数据目录(跨机器: 家目录下 quant)

ce_POOLS = {
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data.pkl'),
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data.pkl'),
    'hs300':       os.path.join(_PKL_DIR, 'hs300_day_ta_data.pkl'),
    'a_etf_all':   os.path.join(_PKL_DIR, 'a_etf_all_day_ta_data.pkl'),
}

START = '2021-01-01'

ce_HORIZONS = [5, 20]

MIN_DAYS = 60          # 触发日数低于此 => low_n=1, t 值解读需谨慎

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

FLIP_EVENTS = {
    's_flip_up':   ('s_trend', 'up'),
    's_flip_down': ('s_trend', 'down'),
    'm_flip_up':   ('m_trend', 'up'),
    'm_flip_down': ('m_trend', 'down'),
}

DEFAULT_EVENTS = ['trig_up', 'trig_down', 'break_up', 'break_down',
                  'pattern_up', 'pattern_down',
                  's_flip_up', 's_flip_down', 'm_flip_up', 'm_flip_down']

TREND_DIMS = {
    's': ('s_trend', ('up', 'down', 'wave')),
    'm': ('m_trend', ('up', 'down', 'wave')),
}

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

def ce_eval_pool(pool, pkl_path, args):
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

def ce_main():
    ap = argparse.ArgumentParser(description='条件层评估工具: 事件×状态 条件收益矩阵(只读)')
    ap.add_argument('--pool', default='etf_3x', help='池名, 逗号分隔多个, 或 all')
    ap.add_argument('--pkl', action='append', default=[],
                    help='覆盖数据源, 形如 "etf_3x=C:\\path\\to.pkl", 可多次')
    ap.add_argument('--start', default=START)
    ap.add_argument('--horizons', default=','.join(str(h) for h in ce_HORIZONS))
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

    pools = list(ce_POOLS) if args.pool == 'all' else [p.strip() for p in args.pool.split(',')]
    for p in pools:
        path = overrides.get(p) or ce_POOLS.get(p)
        if path is None:
            print(f'[ERROR] 未知池: {p}; 可选 {list(ce_POOLS)}')
            continue
        if not os.path.exists(path):
            print(f'[ERROR] 找不到 pkl: {p} -> {path}  (该池本轮跳过)')
            continue
        ce_eval_pool(p, path, args)


# ==========================================================================
# ==== 源自 indicator_eval.py ====  改名: main->ie_main, eval_pool->ie_eval_pool, group_of->ie_group_of, build_summary->ie_build_summary, POOLS->ie_POOLS, HORIZONS->ie_HORIZONS, SUSPECT_PAT->ie_SUSPECT_PAT
# ==========================================================================
ie_POOLS = {
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data.pkl'),
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data.pkl'),
    'hs300':       os.path.join(_PKL_DIR, 'hs300_day_ta_data.pkl'),
    'a_etf_all':   os.path.join(_PKL_DIR, 'a_etf_all_day_ta_data.pkl'),
}

ie_HORIZONS = [5, 20]

TOP_K = 5

MIN_CS = 10

ie_SUSPECT_PAT = ('pos_label', 'neg_label', 'label', 'action')

SKIP_PAT = ('_description', 'description')

SKIP_EXACT = ('pattern_up', 'pattern_down')

SYNTH_WEIGHTS = {
    'trigger_net': {'break_up_score': 1.0, 'break_down_score': 1.0,
                    'support_score': 0.5, 'resistant_score': 0.5},
    'pattern_net': {'pattern_up_score': 1.0, 'pattern_down_score': 1.0},
}

GRADE_POOLS = ['etf_3x', 'company_300', 'hs300', 'a_etf_all']

STATIC_AC1 = 0.99          # 信号截面排序几乎不随时间变化 => 事实上的固定选票

TURN_LOW = 0.05

THRESH = 0.90              # redund: |corr| 高共线阈值

FOCUS = ['Low_to_kijun', 'High_to_kijun', 'ichimoku_distance_alpha', 'adx_power',
         'adx_strength_change', 'kama_slow_rate', 'candle_gap_distance',
         'ichimoku_distance_day', 'kijun_day', 'trend_magnitude',
         'trend_score_alpha', 'trend_magnitude_alpha', 'kama_rate',
         'Low_to_kama_slow', 'Low_to_tankan', 'kijun', 'Close',
         'trigger_score', 'trigger_net', 'pattern_net']

def ie_group_of(name: str) -> str:
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

def ie_eval_pool(pool: str, pkl: str, with_derived: bool = False) -> pd.DataFrame:
    _, panel = load_panel(pkl, 'day')
    panel = panel[panel.index.get_level_values('date') >= pd.Timestamp(START)]
    syms = panel.index.get_level_values('symbol').unique()
    dts = panel.index.get_level_values('date')
    print(f'\n===== {pool}: {len(syms)} 标的, {dts.min().date()}~{dts.max().date()}, '
          f'{len(panel)} 行 =====')
    cands, num_cols, open_wide = build_cands(panel, with_derived=with_derived)
    print(f'  候选指标数 = {len(cands)}  (原生 {len(num_cols)} 数值列)')

    fwds = {h: tradable_fwd(open_wide, h) for h in ie_HORIZONS}
    n_days_total = panel.index.get_level_values('date').nunique()
    rows = []
    for i, (name, sig) in enumerate(cands.items(), 1):
        rec = {'signal': name, 'group': ie_group_of(name),
               'coverage': round(float(sig.notna().mean().mean()), 3),
               'nuniq': int(pd.unique(sig.values[~pd.isna(sig.values)]).size)
               if sig.notna().any().any() else 0,
               'suspect': any(p in name.lower() for p in ie_SUSPECT_PAT),
               'sig_ac1': sig_ac1(sig)}
        rec['is_const'] = int(rec['nuniq'] <= 1)
        for h in ie_HORIZONS:
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
    targets = list(ie_POOLS) if a.pool == 'all' else [a.pool]
    missing = []
    for p in targets:
        if p not in ie_POOLS:
            print(f'[ERROR] 未知池: {p}; 可选 {list(ie_POOLS)}')
            continue
        if not os.path.exists(ie_POOLS[p]):
            missing.append((p, ie_POOLS[p]))
            continue
        ie_eval_pool(p, ie_POOLS[p], with_derived=a.derived)
    for p, fp in missing:
        print(f'[ERROR] 找不到 pkl: {p} -> {fp}  (该池本轮跳过)')

def load_all():
    out = {}
    for p in GRADE_POOLS:
        fp = _out(f'eval_{p}.csv')
        if os.path.exists(fp):
            out[p] = pd.read_csv(fp).set_index('signal')
        else:
            print(f'[WARN] 缺少 {fp}')
    return out

def ie_build_summary(base):
    all_sig = sorted(set().union(*[set(df.index) for df in base.values()]))
    rows = []
    for s in all_sig:
        avail = [p for p in base if s in base[p].index]
        r0 = base[avail[0]].loc[s]
        rec = {'signal': s, 'group': r0['group'],
               'suspect': int(any(p in s.lower() for p in ie_SUSPECT_PAT)),
               'coverage': round(float(np.mean([base[p].loc[s, 'coverage'] for p in avail])), 3),
               'nuniq': int(np.max([base[p].loc[s, 'nuniq'] for p in avail])),
               'sig_ac1': round(float(np.nanmean([base[p].loc[s, 'sig_ac1'] for p in avail])), 3)}
        for p in avail:
            r = base[p].loc[s]
            rec[f'{p}_h5_exc'] = r.get('h5_excess_k')
            rec[f'{p}_h20_exc'] = r.get('h20_excess_k')
            rec[f'{p}_h20_icir'] = r.get('h20_icir')
        for h in ie_HORIZONS:
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
    for h in ie_HORIZONS:
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
    summ = ie_build_summary(base)
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

def cmd_redund(a):
    pool = a.pool
    if pool not in ie_POOLS or not os.path.exists(ie_POOLS[pool]):
        print(f'[ERROR] 找不到池数据: {pool} -> {ie_POOLS.get(pool)}')
        return
    _, panel = load_panel(ie_POOLS[pool], 'day')
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
                              'group_a': ie_group_of(cols[i]), 'group_b': ie_group_of(cols[j])})
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
        rows.append({'signal': c, 'group': ie_group_of(c), 'nearest': other,
                     'corr': top['corr'], 'abs_corr': top['abs_corr'],
                     'mean_abs_corr': round(float(sub['abs_corr'].mean()), 3)})
    N = pd.DataFrame(rows).sort_values('mean_abs_corr')
    N.to_csv(_out(f'eval_neighbors_{pool}.csv'), index=False, encoding='utf-8-sig')
    print(N.head(40).to_string(index=False))

    # 组内平均|相关|: 判断哪些家族内部信息重复最严重
    g = P.groupby('group_a')['abs_corr'].agg(['mean', 'count']).sort_values('mean')
    print('\n===== 各指标组内部平均 |corr| (越低 => 组内越互补) =====')
    print(g.to_string())

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

def ie_main():
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


# ==========================================================================
# ==== 源自 alpha_mining.py ====  改名: main->am_main, eval_pool->am_eval_pool, group_of->am_group_of, HORIZONS->am_HORIZONS
# ==========================================================================
POOLS = {
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data_research.pkl'),  # 生产 pkl 截短 2025+, 用 research 全史
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data_research.pkl'),  # 全史; 生产版仅保留约近 2 年
    'company_1000': os.path.join(_PKL_DIR, 'company_1000_day_ta_data_research.pkl'),  # 2020+ 全史(无生产版)
    'hs300':       os.path.join(_PKL_DIR, 'hs300_day_ta_data_research.pkl'),  # 2020+ 全史
    'a_etf_all':   os.path.join(_PKL_DIR, 'a_etf_all_day_ta_data_research.pkl'),  # 全史
}

am_HORIZONS = [5, 20, 60]

F_ANCHORS = ['F_mom121', 'F_er20', 'F_updnvol20', 'F_volstab20',
             'F_obv20', 'F_alpha60', 'F_beta60']

F_NEG = ['F_idiovol60', 'F_cvcorr20', 'F_ulcer60', 'F_range20',
         'F_kurt60', 'F_dnvol20']

def _alphaize(x: pd.DataFrame, window: int = NORM_WINDOW, min_periods: int = NORM_MINP) -> pd.DataFrame:
    """normalize_causal 但 NaN 不被 fillna(0) 污染: 先记录有效位, 事后 mask 回 NaN."""
    ok = x.notna()
    v = normalize_causal(x.abs(), window=window, min_periods=min_periods)
    return v.where(ok)

def build_alpha_cands(panel: pd.DataFrame) -> dict:
    """构建全部候选(在全历史 panel 上计算, 因果安全; 评估窗口截取在主流程做)."""
    def w(col):
        return panel[col].unstack('symbol').sort_index()

    close = w('Close')
    open_ = w('Open')
    high = w('High')
    low = w('Low')
    volume = w('Volume')
    ret1 = close.pct_change()
    mined = build_mined(panel)          # F_ 族(全历史构建)
    out = {}

    # ---- H. alpha 化族(原型: m_trend_score_alpha) ----
    out['H_ichimoku_alpha'] = _alphaize(w('ichimoku_distance'))    # = m_trend_score_alpha 精确重建(锚)
    out['H_trendmag_alpha'] = _alphaize(w('trend_magnitude'))      # = trend_magnitude_alpha 重建(锚)
    out['H_adxval_alpha'] = _alphaize(w('adx_value'))
    out['H_adxstr_alpha'] = _alphaize(w('adx_strength'))           # 真实 ADX(非负, abs 无损)
    out['H_adxdist_alpha'] = _alphaize(w('adx_distance'))
    out['H_adxpow_alpha'] = _alphaize(w('adx_power'))
    out['H_atr_alpha'] = _alphaize(w('atr'))
    out['H_trpct_alpha'] = _alphaize(w('tr') / close)              # 真实波幅占比(短周期波动爆发度)
    mavg = w('mavg')
    bb_width = (w('bb_high_band') - w('bb_low_band')) / mavg.replace(0, np.nan)
    out['H_bbw_alpha'] = _alphaize(bb_width)                       # 布林带宽分位
    rsi = w('rsi')
    out['H_rsidev_alpha'] = _alphaize(rsi - 50.0)                  # RSI 偏离强度
    out['H_kamadist_alpha'] = _alphaize(w('kama_distance'))
    out['H_gapdist_alpha'] = _alphaize(w('candle_gap_distance'))   # 对照 eval B中 candle_gap_distance
    out['H_body_alpha'] = _alphaize(w('candle_entity_pct'))
    out['H_shadow_alpha'] = _alphaize(w('candle_upper_shadow_pct') + w('candle_lower_shadow_pct'))
    out['H_volratio_alpha'] = _alphaize(volume / volume.rolling(20).mean().replace(0, np.nan))
    out['H_volchg_alpha'] = _alphaize(w('volume_change'))
    out['H_entitydiff_alpha'] = _alphaize(w('entity_diff'))
    out['H_pos_alpha'] = _alphaize(w('candle_position_score'))
    out['H_trigger_alpha'] = _alphaize(w('trigger_score'))
    out['H_pattern_alpha'] = _alphaize(w('pattern_score'))
    out['H_er_alpha'] = _alphaize(mined['F_er20'])                 # 趋势质量处于自身历史分位

    # ---- S. 方向 score 族(sign * alpha, m_trend_score 模板) ----
    out['S_rsi'] = np.sign(rsi - 50.0) * out['H_rsidev_alpha']
    out['S_kama'] = np.sign(w('kama_distance')) * out['H_kamadist_alpha']

    # ---- X. 新异象族 ----
    out['X_maxret_neg20'] = -ret1.rolling(20).max()                # MAX 效应: 近期极端单日涨幅(彩票型)未来跑输
    up = (ret1 > 0).where(ret1.notna()).astype(float)
    out['X_upday20'] = up.rolling(20, min_periods=10).mean()       # 涨天数占比(路径动量质量)
    vol20 = ret1.rolling(20).std()
    out['X_lowvol_alpha'] = -_alphaize(vol20)                      # 低波动异象 alpha 版
    hh60 = close.rolling(60, min_periods=20).max()
    ll60 = close.rolling(60, min_periods=20).min()
    out['X_hlpos60'] = (close - ll60) / (hh60 - ll60).replace(0, np.nan)
    rng = (high - low).replace(0, np.nan)
    out['X_closepos60'] = ((close - low) / rng).rolling(60, min_periods=20).mean()
    out['X_streak_alpha'] = _alphaize(mined['F_streak'])           # 连涨/连跌强度分位
    out['X_mom_alpha60'] = _alphaize(close.pct_change(60))         # 动量处于自身历史分位(regime 感知)
    out['X_overn_alpha'] = _alphaize(mined['F_overn20'])           # 隔夜动量自身分位

    # ---- N. 上轮负向稳健因子取反 ----
    for f in F_NEG:
        out['N_' + f[2:]] = -mined[f]

    # ---- C. 日度截面秩组合 ----
    rk = lambda v: v.rank(axis=1, pct=True)
    out['C_qmom60'] = rk(mined['F_alpha60']) + rk(-mined['F_idiovol60'])   # 质量动量: 高相对强度+低特质波动
    out['C_tmqmom'] = rk(mined['F_mom121']) + rk(mined['F_er20'])          # 12-1 动量 + 趋势效率

    # ---- 锚: pkl 原列 alpha(交叉验证本工具复刻口径) + 上轮单池 Top 因子跨池复验 ----
    # 旧版列结构 research pkl 可能缺 *_alpha 锚列: 缺列时静默跳过(对照 build_alpha_synths 容错模式,
    # 不影响实验信号 H_/C_/F_ —— 它们依赖的基础 TA 列在两版 pkl 均存在).
    if 'trend_magnitude_alpha' in panel.columns:
        out['P_trend_magnitude_alpha'] = w('trend_magnitude_alpha')
    if 'pattern_net_alpha' in panel.columns:
        out['P_pattern_net_alpha'] = w('pattern_net_alpha')
    for f in F_ANCHORS:
        out[f] = mined[f]

    return {k: v for k, v in out.items()
            if isinstance(v, pd.DataFrame) and v.notna().sum().sum() > 0}

def am_group_of(name: str) -> str:
    if name.startswith('H_'):
        return 'H_alpha化'
    if name.startswith('S_'):
        return 'S_方向score'
    if name.startswith('X_'):
        return 'X_新异象'
    if name.startswith('N_'):
        return 'N_反向'
    if name.startswith('C_'):
        return 'C_组合'
    if name.startswith('P_'):
        return 'P_pkl原列锚'
    if name.startswith('F_'):
        return 'F_上轮复验'
    return '其他'

def daily_excess(sig: pd.DataFrame, fwd: pd.DataFrame, top_k: int = TOP_K, min_cs: int = MIN_CS):
    """日级超额序列(top-k 均值 - 池等权均值), 口径与 screen_one 完全一致, 供 NW-HAC 检验."""
    idx = sig.index.intersection(fwd.index)
    cols = sig.columns.intersection(fwd.columns)
    if len(idx) < 60:
        return pd.Series(dtype=float), None
    s = sig.loc[idx, cols].values.astype(float)
    r = fwd.loc[idx, cols].values.astype(float)
    mask = ~np.isnan(s) & ~np.isnan(r)
    cnt = mask.sum(axis=1)
    ok = cnt >= min_cs
    if ok.sum() < 60:
        return pd.Series(dtype=float), None
    sv = np.where(mask & ok[:, None], s, np.nan)
    rv = np.where(mask & ok[:, None], r, np.nan)
    k = max(1, min(top_k, int(cnt[ok].min()) // 3))
    order = np.argsort(np.where(np.isnan(sv), -np.inf, sv), axis=1, kind='stable')[:, ::-1]
    top = np.nanmean(np.take_along_axis(rv, order[:, :k], axis=1), axis=1)
    pool = np.nanmean(rv, axis=1)
    return pd.Series(top - pool, index=idx), k

def am_eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
    _, panel = load_panel(pkl, 'day')
    dts = panel.index.get_level_values('date')
    open_wide = panel['Open'].unstack('symbol').sort_index()
    cands = build_alpha_cands(panel)                 # 全历史构建
    start_ts = pd.Timestamp(start)
    cands = {k: v.loc[v.index >= start_ts] for k, v in cands.items()}
    n_sym = len(panel.index.get_level_values('symbol').unique())
    print(f'\n===== {pool}: {n_sym} 标的, {dts.min().date()}~{dts.max().date()} | '
          f'候选 {len(cands)} 个, 评估窗 {start}~ =====')

    fwds = {h: tradable_fwd(open_wide, h) for h in horizons}
    vrows, screen_rows = [], []
    for h in horizons:
        fwd = fwds[h]
        rows, pvals, names = [], [], []
        for name, sig in cands.items():
            try:
                rec = screen_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception as e:
                print(f'  [SKIP screen] {name}: {e}')
                continue
            if not rec:
                continue
            exc, k = daily_excess(sig, fwd)
            t, p, n = nw_tstat(exc.values, lag=max(h - 1, 1))
            rec.update({'signal': name, 'group': am_group_of(name),
                        'nw_t': round(t, 2) if pd.notna(t) else np.nan,
                        'nw_p': round(p, 4) if pd.notna(p) else np.nan,
                        'nw_n': n, 'exc_k_daily': k})
            rows.append(rec)
            names.append(name)
            pvals.append(p)
            try:
                v = validate_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception:
                v = {}
            if v:
                v['signal'] = name
                v['h'] = h
                vrows.append(v)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df['fdr_q'] = bh_qvals(df['nw_p'].values)
        cols = ['signal', 'group', 'excess_k', 'topk_ret', 'pool_ret', 'ls_spread',
                'ic_mean', 'icir', 't_stat', 'ic_pos_rate', 'topk_turnover',
                'nw_t', 'nw_p', 'fdr_q', 'nw_n', 'n_days']
        df = df[cols].sort_values('excess_k', ascending=False)
        fp = os.path.join(out_dir, f'{pool}_h{h}_screen.csv')
        df.to_csv(fp, index=False, encoding='utf-8-sig')
        screen_rows.append(df)
        print(f'-- h={h}: {len(df)} 候选, BH 族大小 {len(df)}; '
              f'q<=0.10 的 {int((df["fdr_q"] <= 0.10).sum())} 个; '
              f'q<=0.25 的 {int((df["fdr_q"] <= 0.25).sum())} 个')
        print(df.head(12).to_string(index=False))

    # 信号自相关(静态身份检验, 跨周期共用)
    ac1 = {}
    for name, sig in cands.items():
        try:
            ac1[name] = sig_ac1(sig)
        except Exception:
            ac1[name] = np.nan
    if vrows:
        vdf = pd.DataFrame(vrows)
        vdf['sig_ac1'] = vdf['signal'].map(ac1)
        vdf.to_csv(os.path.join(out_dir, f'{pool}_validate.csv'),
                   index=False, encoding='utf-8-sig')
    # 返回长表(含全 h 的 screen + validate 合并)供四池汇总
    long_rows = []
    for h, df in zip([h for h in horizons], screen_rows):
        vd = {r['signal']: r for r in vrows if r['h'] == h} if vrows else {}
        for _, r in df.iterrows():
            v = vd.get(r['signal'], {})
            long_rows.append({
                'pool': pool, 'h': h, 'signal': r['signal'], 'group': r['group'],
                'excess_k': r['excess_k'], 'topk_ret': r['topk_ret'], 'pool_ret': r['pool_ret'],
                'icir': r['icir'], 't_stat': r['t_stat'], 'topk_turnover': r['topk_turnover'],
                'nw_t': r['nw_t'], 'nw_p': r['nw_p'], 'fdr_q': r['fdr_q'], 'nw_n': r['nw_n'],
                'n_days': r['n_days'], 'sig_ac1': ac1.get(r['signal'], np.nan),
                'exc_half1': v.get('exc_half1', np.nan), 'exc_half2': v.get('exc_half2', np.nan),
                'dyn_minus_static': v.get('dyn_minus_static', np.nan),
                'top3_share': v.get('top3_share', np.nan),
                'n_eff_symbols': v.get('n_eff_symbols', np.nan),
            })
    return pd.DataFrame(long_rows)

def build_summary(longs: dict) -> pd.DataFrame:
    """四池汇总: 方向对齐(多数符号) -> 同号性 / exc_min / FDR q_max / 分半 / 分级."""
    all_df = pd.concat(longs.values(), ignore_index=True)
    rows = []
    for (sig, h), sub in all_df.groupby(['signal', 'h']):
        g0 = sub.iloc[0]['group']
        rec = {'signal': sig, 'h': h, 'group': g0, 'n_pool': len(sub)}
        for _, r in sub.iterrows():
            rec[f"{r['pool']}_exc"] = r['excess_k']
            rec[f"{r['pool']}_q"] = r['fdr_q']
            rec[f"{r['pool']}_nw_t"] = r['nw_t']
        # 方向对齐
        vals = sub['excess_k'].dropna().values
        if len(vals) == 0:
            rows.append(rec)
            continue
        npos = int((vals > 0).sum())
        d = 1.0 if npos * 2 >= len(vals) else -1.0
        al = vals * d
        rec['dir'] = '正' if d > 0 else '负'
        rec['n_consist'] = int(max(npos, len(vals) - npos))
        rec['cons_ratio'] = round(rec['n_consist'] / len(vals), 2)
        rec['exc_mean'] = round(float(al.mean()), 5)
        rec['exc_min'] = round(float(al.min()), 5)
        qvals = sub['fdr_q'].dropna()
        rec['q_max'] = round(float(qvals.max()), 3) if len(qvals) else np.nan
        tvals = (d * sub['nw_t']).dropna()
        rec['nw_t_min'] = round(float(tvals.min()), 2) if len(tvals) else np.nan
        ss = sub.apply(lambda r: np.sign(r['exc_half1']) == np.sign(r['exc_half2'])
                       if pd.notna(r['exc_half1']) and pd.notna(r['exc_half2']) else np.nan, axis=1)
        ssn = ss.dropna()
        rec['same_sign_half_n'] = int((ssn == True).sum()) if len(ssn) else 0
        rec['same_sign_of'] = len(ssn)
        dyn = sub['dyn_minus_static'].dropna()
        rec['dyn_mean'] = round(float(dyn.mean()), 5) if len(dyn) else np.nan
        rec['icir_mean'] = round(float(sub['icir'].abs().mean()), 3)
        rec['turnover_mean'] = round(float(sub['topk_turnover'].mean()), 3)
        rec['top3_share_max'] = round(float(sub['top3_share'].max()), 3) \
            if sub['top3_share'].notna().any() else np.nan
        rec['sig_ac1'] = round(float(sub['sig_ac1'].dropna().mean()), 4) \
            if sub['sig_ac1'].notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def classify_alpha(r):
    """分级(参照 indicator_eval.classify 阈值 + FDR q_max):
       S静态身份  sig_ac1 >= 0.99
       A强  四池全同号 且 exc_min>=0.005 且 exc_mean>=0.015 且 q_max<=0.10
            且分半全同号 且 dyn_mean>0
       B中  (全同号 且 exc_min>=0.001 且 exc_mean>=0.005) 或
            (cons>=0.75 且 exc_min>=0.005 且 exc_mean>=0.010), 且 q_max<=0.25
       C弱  cons>=0.5 且 exc_mean>=0.002
       D无效 其余
    """
    if pd.notna(r.get('sig_ac1')) and r['sig_ac1'] >= STATIC_AC1:
        return 'S静态身份', 0.0
    cons, emin, emean = r.get('cons_ratio', 0), r.get('exc_min', np.nan), r.get('exc_mean', np.nan)
    qmax = r.get('q_max', np.nan)
    ss = r['same_sign_half_n'] / r['same_sign_of'] if r.get('same_sign_of') else 0.0
    dyn = r.get('dyn_mean', np.nan)
    if pd.isna(emin) or pd.isna(emean):
        return 'D无效', 0.0
    score = (0.30 * cons + 0.25 * min(max(emin, 0) / 0.02, 1.0)
             + 0.15 * ss + 0.15 * (max(0.0, 1.0 - qmax / 0.25) if pd.notna(qmax) else 0.0)
             + 0.15 * (1.0 if pd.notna(dyn) and dyn > 0 else 0.0))
    score = round(float(score), 3)
    fdr_ok_a = pd.notna(qmax) and qmax <= 0.10
    fdr_ok_b = pd.notna(qmax) and qmax <= 0.25
    if (cons >= 1.0 and emin >= 0.005 and emean >= 0.015 and fdr_ok_a
            and r['same_sign_half_n'] == r['same_sign_of'] and r['same_sign_of'] > 0
            and pd.notna(dyn) and dyn > 0):
        tier = 'A强'
    elif fdr_ok_b and ((cons >= 1.0 and emin >= 0.001 and emean >= 0.005)
                       or (cons >= 0.75 and emin >= 0.005 and emean >= 0.010)):
        tier = 'B中'
        if pd.isna(dyn) or dyn <= 0:
            tier = 'B中|无动态增量'
    elif cons >= 0.5 and emean >= 0.002:
        tier = 'C弱'
    else:
        tier = 'D无效'
    return tier, score

def am_main():
    ap = argparse.ArgumentParser(description='独立只读研究: m_trend_score_alpha 式信号挖掘(四池+FDR+分级)')
    ap.add_argument('--pools', default='etf_3x,company_300,hs300,a_etf_all')
    ap.add_argument('--pkl', default=None, help='"池=路径" 覆盖默认数据源(可重复)')
    ap.add_argument('--start', default=START)
    ap.add_argument('--horizons', default=','.join(str(h) for h in am_HORIZONS))
    ap.add_argument('--signals', default=None, help='逗号分隔, 只评估指定候选')
    args = ap.parse_args()

    pools = [x.strip() for x in args.pools.split(',') if x.strip()]
    for kv in (args.pkl or '').split(';'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            POOLS[k.strip()] = v.strip()
    horizons = [int(x) for x in args.horizons.split(',')]

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output',
                           f'alpha_mine_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    longs = {}
    for pool in pools:
        if pool not in POOLS or not os.path.exists(POOLS[pool]):
            print(f'[ERROR] 池数据缺失: {pool} -> {POOLS.get(pool)}')
            continue
        df = am_eval_pool(pool, POOLS[pool], out_dir, horizons, args.start)
        if not df.empty:
            longs[pool] = df
    if not longs:
        print('[ERROR] 无任何池产出')
        sys.exit(1)

    summ = build_summary(longs)
    if args.signals:
        keep = [x.strip() for x in args.signals.split(',') if x.strip()]
        summ = summ[summ['signal'].isin(keep)]
    res = summ.apply(classify_alpha, axis=1, result_type='expand')
    summ['tier'] = res[0]
    summ['score'] = res[1]
    summ = summ.sort_values(['score', 'exc_mean'], ascending=False)
    fp = os.path.join(out_dir, 'summary.csv')
    summ.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f'\n== 汇总 == {fp} ({len(summ)} 行)')

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 60)
    show = ['signal', 'h', 'group', 'tier', 'score', 'dir', 'n_pool', 'n_consist',
            'cons_ratio', 'exc_mean', 'exc_min', 'q_max', 'nw_t_min',
            'same_sign_half_n', 'same_sign_of', 'dyn_mean', 'top3_share_max', 'sig_ac1']
    show = [c for c in show if c in summ.columns]
    print('\n===== 分级计数 =====')
    print(summ['tier'].value_counts().to_string())
    for t in ['A强', 'B中', 'B中|无动态增量', 'C弱']:
        sub = summ[summ['tier'] == t]
        if len(sub):
            print(f'\n===== {t} ({len(sub)}) =====')
            print(sub[show].head(30).to_string(index=False))

    # 报告
    lines = [f'alpha_mining 报告 | pools={list(longs)} | run={run_id}',
             f'评估窗 {args.start}~, horizons={horizons}, top_k={TOP_K}, min_cs={MIN_CS}',
             '口径: 信号日收盘 -> 次日开盘入场 -> 持有h日开盘出场; '
             'NW-HAC(lag=h-1) on 日级超额; BH-FDR 族=每池每h全部候选; 分级=四池同号+FDR+分半+动态增量',
             '',
             '== 分级计数 ==',
             summ['tier'].value_counts().to_string(), '']
    for t in ['A强', 'B中', 'B中|无动态增量', 'C弱', 'D无效', 'S静态身份']:
        sub = summ[summ['tier'] == t]
        if len(sub):
            lines.append(f'== {t} ({len(sub)}) ==')
            lines.append(sub[show].to_string(index=False))
            lines.append('')
    for pool, df in longs.items():
        for h in horizons:
            sub = df[df['h'] == h].sort_values('excess_k', ascending=False)
            if len(sub):
                lines.append(f'== {pool} h={h} 全量(excess_k 降序) ==')
                cols = ['signal', 'group', 'excess_k', 'icir', 'nw_t', 'fdr_q', 'n_days']
                lines.append(sub[cols].to_string(index=False))
                lines.append('')
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')


# ==========================================================================
# ==== 源自 factor_mining2.py ====  改名: main->fm2_main, GROUPS->fm2_GROUPS, POOLS->fm2_POOLS, HORIZONS->fm2_HORIZONS, eval_pool->fm2_eval_pool, group_of->fm2_group_of
# ==========================================================================
fm2_POOLS = {
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data_research.pkl'),
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data_research.pkl'),
    'company_1000': os.path.join(_PKL_DIR, 'company_1000_day_ta_data_research.pkl'),
}

fm2_HORIZONS = [5, 20, 60]

fm2_GROUPS = {  # 前缀 -> 分族
    'G_park': '波动微观结构', 'G_oviv': '波动微观结构', 'G_semi': '波动微观结构', 'G_ext': '波动微观结构',
    'G_ac1': '时序结构', 'G_vr': '时序结构', 'G_gap': '时序结构',
    'G_imp': '量能不对称', 'G_volb': '量能不对称', 'G_ami': '量能不对称', 'G_vwap': '量能不对称',
    'G_upv': '量能不对称', 'G_volc': '量能不对称',
    'G_days': '路径时间形态', 'G_dist': '路径时间形态', 'G_chop': '路径时间形态',
    'G_dn': '风险结构', 'G_cosk': '风险结构', 'G_pool': '风险结构',
    'G_seas': '季节性',
    'G_consist': '组合', 'G_mom_': '组合', 'G_low': '组合',
}

def fm2_group_of(name: str) -> str:
    for k, v in fm2_GROUPS.items():
        if name.startswith(k):
            return v
    return '其他'

def build_g_cands(panel: pd.DataFrame) -> dict:
    """构建全部 G_ 候选(全历史构建, 因果安全; 评估窗截取在主流程做)."""
    def w(col):
        return panel[col].unstack('symbol').sort_index()

    close = w('Close')
    open_ = w('Open')
    high = w('High')
    low = w('Low')
    volume = w('Volume')
    ret1 = close.pct_change()
    out = {}

    # ---- 波动微观结构 ----
    ln_hl = np.log((high / low).where(high > low))
    park_var = ln_hl.pow(2).rolling(20, min_periods=10).mean() / (4.0 * np.log(2.0))
    cc_var = ret1.rolling(20, min_periods=10).var()
    out['G_parkcc20'] = np.sqrt(park_var) / np.sqrt(cc_var.replace(0, np.nan))
    overnight = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    out['G_oviv20'] = overnight.rolling(20, min_periods=10).std() \
        / intraday.rolling(20, min_periods=10).std().replace(0, np.nan)
    dn2 = ret1.where(ret1 < 0).pow(2)
    out['G_semidn20'] = dn2.rolling(20, min_periods=10).mean() \
        / cc_var.replace(0, np.nan)
    out['G_extasym60'] = ret1.rolling(60, min_periods=30).max() + ret1.rolling(60, min_periods=30).min()

    # ---- 时序结构 ----
    ac1_60 = ret1.rolling(60, min_periods=30).corr(ret1.shift(1))
    out['G_ac1_60'] = ac1_60
    mom60 = close.pct_change(60)
    out['G_ac1mom'] = ac1_60 * np.sign(mom60)
    for q in (5, 20):
        rq = close.pct_change(q)
        var_q = rq.rolling(120, min_periods=60).var()
        var_1 = ret1.rolling(120, min_periods=60).var()
        out[f'G_vr{q}'] = var_q / (q * var_1).replace(0, np.nan)
    out['G_gaprev20'] = overnight.rolling(20, min_periods=10).corr(intraday)

    # ---- 量能不对称 ----
    up_m, dn_m = ret1 > 0, ret1 < 0
    up_vol = volume.where(up_m)
    up_abs = ret1.where(up_m).abs()
    dn_vol = volume.where(dn_m)
    dn_abs = ret1.where(dn_m).abs()
    uv = up_vol.rolling(20, min_periods=5).sum() / up_abs.rolling(20, min_periods=5).sum().replace(0, np.nan)
    dv = dn_vol.rolling(20, min_periods=5).sum() / dn_abs.rolling(20, min_periods=5).sum().replace(0, np.nan)
    out['G_impupdn20'] = uv / dv.replace(0, np.nan)
    lv = np.log(volume.where(volume > 0) + 1.0)
    pool_lv = lv.mean(axis=1)
    cov_lv = lv.rolling(60, min_periods=30).cov(pool_lv)
    out['G_volbeta60'] = cov_lv.div(pool_lv.rolling(60, min_periods=30).var().replace(0, np.nan), axis=0)
    dvusd = (close * volume).replace(0, np.nan)
    illiq = ret1.abs() / dvusd
    out['G_amiasym20'] = illiq.where(up_m).rolling(20, min_periods=5).mean() \
        - illiq.where(dn_m).rolling(20, min_periods=5).mean()
    tp = (high + low + close) / 3.0
    vwap20 = (tp * volume).rolling(20, min_periods=10).sum() \
        / volume.rolling(20, min_periods=10).sum().replace(0, np.nan)
    out['G_vwapdev20'] = close / vwap20 - 1.0
    out['G_upvshare20'] = up_vol.rolling(20, min_periods=10).sum() \
        / volume.rolling(20, min_periods=10).sum().replace(0, np.nan)
    out['G_volconf20'] = close.pct_change(20) * ret1.rolling(20, min_periods=10).corr(volume)

    # ---- 路径时间形态 ----
    hi250 = close.rolling(250, min_periods=60).max()
    n = len(close)
    rowno = pd.DataFrame(np.repeat(np.arange(n, dtype=float)[:, None], close.shape[1], axis=1),
                         index=close.index, columns=close.columns)
    lasthi = rowno.where(close.ge(hi250)).ffill()
    out['G_days_hi250'] = -(rowno - lasthi)
    out['G_dist_hi20'] = close / close.rolling(20, min_periods=10).max() - 1.0
    sgn = np.sign(ret1)
    flip = sgn.diff().abs().eq(2.0).astype(float)
    out['G_chop20'] = -flip.rolling(20, min_periods=10).sum()

    # ---- 风险结构 ----
    pr = ret1.mean(axis=1)                     # 池等权收益(仅用 t 及之前)
    m_dn = pd.DataFrame(np.repeat((pr.values < 0)[:, None], ret1.shape[1], axis=1),
                        index=ret1.index, columns=ret1.columns)
    m_up = pd.DataFrame(np.repeat((pr.values > 0)[:, None], ret1.shape[1], axis=1),
                        index=ret1.index, columns=ret1.columns)
    r_dn = ret1.where(m_dn)
    r_up = ret1.where(m_up)
    pr_dn = pr.where(pr < 0)
    pr_up = pr.where(pr > 0)
    beta_dn = r_dn.rolling(60, min_periods=15).cov(pr_dn) \
        .div(pr_dn.rolling(60, min_periods=15).var().replace(0, np.nan), axis=0)
    beta_up = r_up.rolling(60, min_periods=15).cov(pr_up) \
        .div(pr_up.rolling(60, min_periods=15).var().replace(0, np.nan), axis=0)
    out['G_dnbeta60'] = beta_dn - beta_up
    rm_c = pr - pr.rolling(60, min_periods=30).mean()
    r_c = ret1 - ret1.rolling(60, min_periods=30).mean()
    m3 = r_c.mul(rm_c.pow(2), axis=0).rolling(60, min_periods=30).mean()
    sr = ret1.rolling(60, min_periods=30).std()
    sm = pr.rolling(60, min_periods=30).std()
    out['G_coskew60'] = m3.div(sr.mul(sm.pow(2), axis=0).replace(0, np.nan))
    cov_rp = ret1.rolling(60, min_periods=30).cov(pr)
    out['G_poolcorr60'] = cov_rp.div(np.sqrt(pr.rolling(60, min_periods=30).var()), axis=0) \
        / ret1.rolling(60, min_periods=30).std().replace(0, np.nan)

    # ---- 季节性(Heston-Sadka 式) ----
    dow = ret1.index.dayofweek
    parts = [ret1[dow == d].rolling(48, min_periods=24).mean() for d in range(5)]
    out['G_seas_dow'] = pd.concat(parts).sort_index()
    mon = ret1.index.month
    parts = [ret1[mon == m].rolling(63, min_periods=21).mean() for m in range(1, 13)]
    out['G_seas_mon'] = pd.concat(parts).sort_index()

    # ---- 组合 ----
    up = (ret1 > 0).where(ret1.notna()).astype(float)
    upday60 = up.rolling(60, min_periods=30).mean()
    vol60 = ret1.rolling(60, min_periods=30).std()
    out['G_consist_sharpe'] = (mom60 / vol60.replace(0, np.nan)) * upday60
    mom121 = close.shift(21) / close.shift(252) - 1.0
    out['G_mom_cons'] = np.sign(mom121) * upday60
    mined = build_mined(panel)
    rk = lambda v: v.rank(axis=1, pct=True)     # noqa: E731
    out['G_lowbabi'] = rk(-mined['F_beta60']) + rk(mined['F_alpha60'])

    return {k: v for k, v in out.items()
            if isinstance(v, pd.DataFrame) and v.notna().sum().sum() > 0}

def fm2_eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
    _, panel = load_panel(pkl, 'day')
    dts = panel.index.get_level_values('date')
    open_wide = panel['Open'].unstack('symbol').sort_index()
    cands = build_g_cands(panel)
    start_ts = pd.Timestamp(start)
    cands = {k: v.loc[v.index >= start_ts] for k, v in cands.items()}
    n_sym = len(panel.index.get_level_values('symbol').unique())
    print(f'\n===== {pool}: {n_sym} 标的, {dts.min().date()}~{dts.max().date()} | '
          f'候选 {len(cands)} 个, 评估窗 {start}~ =====', flush=True)

    fwds = {h: tradable_fwd(open_wide, h) for h in horizons}
    vrows, screen_rows = [], []
    for h in horizons:
        fwd = fwds[h]
        rows, pvals = [], []
        for name, sig in cands.items():
            try:
                rec = screen_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception as e:
                print(f'  [SKIP screen] {name}: {e}', flush=True)
                continue
            if not rec:
                continue
            exc, k = daily_excess(sig, fwd)
            t, p, nn = nw_tstat(exc.values, lag=max(h - 1, 1))
            rec.update({'signal': name, 'group': fm2_group_of(name),
                        'nw_t': round(t, 2) if pd.notna(t) else np.nan,
                        'nw_p': round(p, 4) if pd.notna(p) else np.nan,
                        'nw_n': nn, 'exc_k_daily': k})
            rows.append(rec)
            pvals.append(p)
            try:
                v = validate_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception:
                v = {}
            if v:
                v['signal'] = name
                v['h'] = h
                vrows.append(v)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df['fdr_q'] = bh_qvals(df['nw_p'].values)
        cols = ['signal', 'group', 'excess_k', 'topk_ret', 'pool_ret', 'ls_spread',
                'ic_mean', 'icir', 't_stat', 'ic_pos_rate', 'topk_turnover',
                'nw_t', 'nw_p', 'fdr_q', 'nw_n', 'n_days']
        df = df[cols].sort_values('excess_k', ascending=False)
        df.to_csv(os.path.join(out_dir, f'{pool}_h{h}_screen.csv'),
                  index=False, encoding='utf-8-sig')
        screen_rows.append(df)
        print(f'-- h={h}: {len(df)} 候选; q<=0.10 的 {int((df["fdr_q"] <= 0.10).sum())} 个; '
              f'q<=0.25 的 {int((df["fdr_q"] <= 0.25).sum())} 个', flush=True)
        print(df.head(10).to_string(index=False), flush=True)

    ac1 = {}
    for name, sig in cands.items():
        try:
            ac1[name] = sig_ac1(sig)
        except Exception:
            ac1[name] = np.nan
    if vrows:
        vdf = pd.DataFrame(vrows)
        vdf['sig_ac1'] = vdf['signal'].map(ac1)
        vdf.to_csv(os.path.join(out_dir, f'{pool}_validate.csv'),
                   index=False, encoding='utf-8-sig')

    long_rows = []
    for h, df in zip(horizons, screen_rows):
        vd = {r['signal']: r for r in vrows if r['h'] == h} if vrows else {}
        for _, r in df.iterrows():
            v = vd.get(r['signal'], {})
            long_rows.append({
                'pool': pool, 'h': h, 'signal': r['signal'], 'group': r['group'],
                'excess_k': r['excess_k'], 'topk_ret': r['topk_ret'], 'pool_ret': r['pool_ret'],
                'icir': r['icir'], 't_stat': r['t_stat'], 'topk_turnover': r['topk_turnover'],
                'nw_t': r['nw_t'], 'nw_p': r['nw_p'], 'fdr_q': r['fdr_q'], 'nw_n': r['nw_n'],
                'n_days': r['n_days'], 'sig_ac1': ac1.get(r['signal'], np.nan),
                'exc_half1': v.get('exc_half1', np.nan), 'exc_half2': v.get('exc_half2', np.nan),
                'dyn_minus_static': v.get('dyn_minus_static', np.nan),
                'top3_share': v.get('top3_share', np.nan),
                'n_eff_symbols': v.get('n_eff_symbols', np.nan),
            })
    return pd.DataFrame(long_rows)

def fm2_main():
    ap = argparse.ArgumentParser(description='第二轮新因子挖矿(G_族, 美股三池+FDR+分级)')
    ap.add_argument('--pools', default='etf_3x,company_300,company_1000')
    ap.add_argument('--pkl', default=None, help='"池=路径" 覆盖默认数据源(可重复)')
    ap.add_argument('--start', default=START)
    ap.add_argument('--horizons', default=','.join(str(h) for h in fm2_HORIZONS))
    ap.add_argument('--signals', default=None, help='逗号分隔, 只评估指定候选')
    args = ap.parse_args()

    pools = [x.strip() for x in args.pools.split(',') if x.strip()]
    for kv in (args.pkl or '').split(';'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            fm2_POOLS[k.strip()] = v.strip()
    horizons = [int(x) for x in args.horizons.split(',')]

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output',
                           f'g_mine_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    longs = {}
    for pool in pools:
        if pool not in fm2_POOLS or not os.path.exists(fm2_POOLS[pool]):
            print(f'[ERROR] 池数据缺失: {pool} -> {fm2_POOLS.get(pool)}')
            continue
        df = fm2_eval_pool(pool, fm2_POOLS[pool], out_dir, horizons, args.start)
        if not df.empty:
            longs[pool] = df
    if not longs:
        print('[ERROR] 无任何池产出')
        sys.exit(1)

    summ = build_summary(longs)
    if args.signals:
        keep = [x.strip() for x in args.signals.split(',') if x.strip()]
        summ = summ[summ['signal'].isin(keep)]
    res = summ.apply(classify_alpha, axis=1, result_type='expand')
    summ['tier'] = res[0]
    summ['score'] = res[1]
    summ = summ.sort_values(['score', 'exc_mean'], ascending=False)
    fp = os.path.join(out_dir, 'summary.csv')
    summ.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f'\n== 汇总 == {fp} ({len(summ)} 行)')

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 60)
    show = ['signal', 'h', 'group', 'tier', 'score', 'dir', 'n_pool', 'n_consist',
            'cons_ratio', 'exc_mean', 'exc_min', 'q_max', 'nw_t_min',
            'same_sign_half_n', 'same_sign_of', 'dyn_mean', 'top3_share_max', 'sig_ac1']
    show = [c for c in show if c in summ.columns]
    print('\n===== 分级计数 =====')
    print(summ['tier'].value_counts().to_string())
    for t in ['A强', 'B中', 'B中|无动态增量', 'C弱']:
        sub = summ[summ['tier'] == t]
        if len(sub):
            print(f'\n===== {t} ({len(sub)}) =====')
            print(sub[show].head(30).to_string(index=False))

    # 报告
    lines = [f'factor_mining2 报告(G_族) | pools={list(longs)} | run={run_id}',
             f'评估窗 {args.start}~, horizons={horizons}, top_k={TOP_K}, min_cs={MIN_CS}',
             '口径: 信号日收盘 -> 次日开盘入场 -> 持有h日开盘出场; '
             'NW-HAC(lag=h-1) on 日级超额; BH-FDR 族=每池每h全部候选; 分级=三池同号+FDR+分半+动态增量',
             '',
             '== 分级计数 ==',
             summ['tier'].value_counts().to_string(), '']
    for t in ['A强', 'B中', 'B中|无动态增量', 'C弱', 'D无效', 'S静态身份']:
        sub = summ[summ['tier'] == t]
        if len(sub):
            lines.append(f'== {t} ({len(sub)}) ==')
            lines.append(sub[show].to_string(index=False))
            lines.append('')
    for pool, df in longs.items():
        for h in horizons:
            sub = df[df['h'] == h].sort_values('excess_k', ascending=False)
            if len(sub):
                lines.append(f'== {pool} h={h} 全量(excess_k 降序) ==')
                cols = ['signal', 'group', 'excess_k', 'icir', 'nw_t', 'fdr_q', 'n_days']
                lines.append(sub[cols].to_string(index=False))
                lines.append('')
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')


# ==========================================================================
# ==== 源自 conditional_mining.py ====  改名: main->cm_main, GROUPS->cm_GROUPS, POOLS->cm_POOLS, HORIZONS->cm_HORIZONS, eval_pool->cm_eval_pool, group_of->cm_group_of
# ==========================================================================
cm_POOLS = {
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data_research.pkl'),
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data_research.pkl'),
    'company_1000': os.path.join(_PKL_DIR, 'company_1000_day_ta_data_research.pkl'),
}

cm_HORIZONS = [5, 20, 60]

cm_GROUPS = {  # 前缀 -> 分族(注意 startswith 匹配顺序: 更长前缀在前)
    'K_mom_pool': '池regime门控', 'K_121_pool': '池regime门控',
    'K_mom_': '个股门控', 'K_121_': '个股门控',
    'K_reso': '多周期共振', 'K_tmq': '强度×质量交叉',
    'F_mom': '裸动量锚',
}

def cm_group_of(name: str) -> str:
    for k, v in cm_GROUPS.items():
        if name.startswith(k):
            return v
    return '其他'

def build_k_cands(panel: pd.DataFrame) -> dict:
    """构建全部 K_ 候选(全历史构建, 因果安全; 评估窗截取在主流程做)."""
    close = panel['Close'].unstack('symbol').sort_index()
    volume = panel['Volume'].unstack('symbol').sort_index()
    ret1 = close.pct_change()
    mined = build_mined(panel)      # F_ 族底料(复用公式, 保证与上轮可比)

    def cq(x: pd.DataFrame) -> pd.DataFrame:
        """因果 min-max 分位 [0,1](252/60), 输入 NaN 不被 fillna(0) 污染."""
        ok = x.notna()
        v = normalize_causal(x, window=252, min_periods=60)
        return v.where(ok)

    def zc(x: pd.DataFrame) -> pd.DataFrame:
        """因果时序 z-score(252/60)."""
        m = x.rolling(252, min_periods=60).mean()
        s = x.rolling(252, min_periods=60).std().replace(0, np.nan)
        return (x - m) / s

    def eqw(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        """sign 一致性权重: 同号 +1 / 异号 −1, 任一 NaN → NaN(缺失不算异号)."""
        ok = a.notna() & b.notna()
        v = (2.0 * (a == b) - 1.0).astype(float)
        return v.where(ok)

    out = {}

    # ---- 个股级门控变量 ----
    vol20 = ret1.rolling(20, min_periods=10).std()
    volr20 = volume / volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    ovq = cq(vol20)                # 个股波动历史分位
    vq = cq(volr20)                # 量比历史分位
    amiq = cq(mined['F_amihud20'])  # 非流动性历史分位(高=流动性差)
    er20 = mined['F_er20']
    mom60 = close.pct_change(60)
    mom20 = close.pct_change(20)
    mom121 = mined['F_mom121']

    # ---- 一、个股门控(8) ----
    out['K_mom_lowvol'] = mom60 * (1.0 - ovq)
    out['K_mom_highvol'] = mom60 * ovq
    out['K_mom_lquiet'] = mom60 * (1.0 - vq)
    out['K_mom_hquiet'] = mom60 * vq
    out['K_mom_er'] = mom60 * er20
    out['K_mom_liq'] = mom60 * (1.0 - amiq)
    out['K_mom_illiq'] = mom60 * amiq
    out['K_121_lowvol'] = mom121 * (1.0 - ovq)

    # ---- 二、池 regime 门控(4): ±1 时序 gate, 无 0/NaN 并列退化 ----
    pr = ret1.mean(axis=1)                       # 池等权日收益(仅用 t 及之前)
    pc = (1.0 + pr.fillna(0.0)).cumprod()        # 池等权指数
    def psgn(n):
        v = np.sign(pc.pct_change(n))
        return v.where(v != 0)                   # 0(含早期恒定段) → NaN 不计
    pool_mom20, pool_mom60 = psgn(20), psgn(60)
    pool_volq = normalize_causal(pr.rolling(20, min_periods=10).std(),
                                 window=252, min_periods=60)
    pool_volq = pool_volq.where(pr.rolling(20, min_periods=10).std().notna())
    pool_tr = np.sign(pc / pc.rolling(60, min_periods=30).mean() - 1.0)
    pool_tr = pool_tr.where(pool_tr != 0)

    out['K_mom_poolsgn'] = mom60.mul(pool_mom20, axis=0)
    out['K_mom_poolvolq'] = mom60.mul(1.0 - 2.0 * pool_volq, axis=0)
    out['K_mom_pooltr'] = mom60.mul(pool_tr, axis=0)
    out['K_121_poolsgn'] = mom121.mul(pool_mom60, axis=0)

    # ---- 三、多周期共振(5): 连续强度 × 一致性权重 ----
    s5 = np.sign(close.pct_change(5))
    s20 = np.sign(mom20)
    s60 = np.sign(mom60)
    s250 = np.sign(close.pct_change(250))
    c3 = eqw(s5, s20) + eqw(s20, s60) + eqw(s5, s60)      # ∈ {−3,−1,+1,+3}
    out['K_reso3'] = mom60 * (c3 / 3.0)
    out['K_reso2'] = mom60 * eqw(s20, s60)
    out['K_reso_l5'] = mom20 * eqw(s5, s250)
    out['K_reso_mag'] = mom60 * eqw(s60, s250)
    ok4 = s5.notna() & s20.notna() & s60.notna() & s250.notna()
    agree = ((s20 == s5).astype(float) + (s20 == s60).astype(float)
             + (s20 == s250).astype(float))               # 0~3
    out['K_reso_cnt'] = (mom60 * ((agree - 1.5) / 1.5)).where(ok4)

    # ---- 四、强度×质量交叉(2) ----
    out['K_tmq_z'] = zc(mom121) * zc(er20)
    out['K_tmqr'] = mom121.rank(axis=1, pct=True) * er20.rank(axis=1, pct=True)

    # ---- 裸动量锚(2): 同口径对照 ----
    out['F_mom60'] = mom60
    out['F_mom121'] = mom121

    return {k: v for k, v in out.items()
            if isinstance(v, pd.DataFrame) and v.notna().sum().sum() > 0}

def cm_eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
    _, panel = load_panel(pkl, 'day')
    dts = panel.index.get_level_values('date')
    open_wide = panel['Open'].unstack('symbol').sort_index()
    cands = build_k_cands(panel)
    start_ts = pd.Timestamp(start)
    cands = {k: v.loc[v.index >= start_ts] for k, v in cands.items()}
    n_sym = len(panel.index.get_level_values('symbol').unique())
    print(f'\n===== {pool}: {n_sym} 标的, {dts.min().date()}~{dts.max().date()} | '
          f'候选 {len(cands)} 个, 评估窗 {start}~ =====', flush=True)

    fwds = {h: tradable_fwd(open_wide, h) for h in horizons}
    vrows, screen_rows = [], []
    for h in horizons:
        fwd = fwds[h]
        rows, pvals = [], []
        for name, sig in cands.items():
            try:
                rec = screen_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception as e:
                print(f'  [SKIP screen] {name}: {e}', flush=True)
                continue
            if not rec:
                continue
            exc, k = daily_excess(sig, fwd)
            t, p, nn = nw_tstat(exc.values, lag=max(h - 1, 1))
            rec.update({'signal': name, 'group': cm_group_of(name),
                        'nw_t': round(t, 2) if pd.notna(t) else np.nan,
                        'nw_p': round(p, 4) if pd.notna(p) else np.nan,
                        'nw_n': nn, 'exc_k_daily': k})
            rows.append(rec)
            pvals.append(p)
            try:
                v = validate_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception:
                v = {}
            if v:
                v['signal'] = name
                v['h'] = h
                vrows.append(v)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df['fdr_q'] = bh_qvals(df['nw_p'].values)
        cols = ['signal', 'group', 'excess_k', 'topk_ret', 'pool_ret', 'ls_spread',
                'ic_mean', 'icir', 't_stat', 'ic_pos_rate', 'topk_turnover',
                'nw_t', 'nw_p', 'fdr_q', 'nw_n', 'n_days']
        df = df[cols].sort_values('excess_k', ascending=False)
        df.to_csv(os.path.join(out_dir, f'{pool}_h{h}_screen.csv'),
                  index=False, encoding='utf-8-sig')
        screen_rows.append(df)
        print(f'-- h={h}: {len(df)} 候选; q<=0.10 的 {int((df["fdr_q"] <= 0.10).sum())} 个; '
              f'q<=0.25 的 {int((df["fdr_q"] <= 0.25).sum())} 个', flush=True)
        print(df.head(10).to_string(index=False), flush=True)

    ac1 = {}
    for name, sig in cands.items():
        try:
            ac1[name] = sig_ac1(sig)
        except Exception:
            ac1[name] = np.nan
    if vrows:
        vdf = pd.DataFrame(vrows)
        vdf['sig_ac1'] = vdf['signal'].map(ac1)
        vdf.to_csv(os.path.join(out_dir, f'{pool}_validate.csv'),
                   index=False, encoding='utf-8-sig')

    long_rows = []
    for h, df in zip(horizons, screen_rows):
        vd = {r['signal']: r for r in vrows if r['h'] == h} if vrows else {}
        for _, r in df.iterrows():
            v = vd.get(r['signal'], {})
            long_rows.append({
                'pool': pool, 'h': h, 'signal': r['signal'], 'group': r['group'],
                'excess_k': r['excess_k'], 'topk_ret': r['topk_ret'], 'pool_ret': r['pool_ret'],
                'icir': r['icir'], 't_stat': r['t_stat'], 'topk_turnover': r['topk_turnover'],
                'nw_t': r['nw_t'], 'nw_p': r['nw_p'], 'fdr_q': r['fdr_q'], 'nw_n': r['nw_n'],
                'n_days': r['n_days'], 'sig_ac1': ac1.get(r['signal'], np.nan),
                'exc_half1': v.get('exc_half1', np.nan), 'exc_half2': v.get('exc_half2', np.nan),
                'dyn_minus_static': v.get('dyn_minus_static', np.nan),
                'top3_share': v.get('top3_share', np.nan),
                'n_eff_symbols': v.get('n_eff_symbols', np.nan),
            })
    return pd.DataFrame(long_rows)

def cm_main():
    ap = argparse.ArgumentParser(description='第五轮条件/门控层因子挖矿(K_族, 美股三池+FDR+分级)')
    ap.add_argument('--pools', default='etf_3x,company_300,company_1000')
    ap.add_argument('--pkl', default=None, help='"池=路径" 覆盖默认数据源(可重复)')
    ap.add_argument('--start', default=START)
    ap.add_argument('--horizons', default=','.join(str(h) for h in cm_HORIZONS))
    ap.add_argument('--signals', default=None, help='逗号分隔, 只评估指定候选')
    args = ap.parse_args()

    pools = [x.strip() for x in args.pools.split(',') if x.strip()]
    for kv in (args.pkl or '').split(';'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            cm_POOLS[k.strip()] = v.strip()
    horizons = [int(x) for x in args.horizons.split(',')]

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output',
                           f'k_mine_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    longs = {}
    for pool in pools:
        if pool not in cm_POOLS or not os.path.exists(cm_POOLS[pool]):
            print(f'[ERROR] 池数据缺失: {pool} -> {cm_POOLS.get(pool)}')
            continue
        df = cm_eval_pool(pool, cm_POOLS[pool], out_dir, horizons, args.start)
        if not df.empty:
            longs[pool] = df
    if not longs:
        print('[ERROR] 无任何池产出')
        sys.exit(1)

    summ = build_summary(longs)
    if args.signals:
        keep = [x.strip() for x in args.signals.split(',') if x.strip()]
        summ = summ[summ['signal'].isin(keep)]
    res = summ.apply(classify_alpha, axis=1, result_type='expand')
    summ['tier'] = res[0]
    summ['score'] = res[1]
    summ = summ.sort_values(['score', 'exc_mean'], ascending=False)
    fp = os.path.join(out_dir, 'summary.csv')
    summ.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f'\n== 汇总 == {fp} ({len(summ)} 行)')

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 60)
    show = ['signal', 'h', 'group', 'tier', 'score', 'dir', 'n_pool', 'n_consist',
            'cons_ratio', 'exc_mean', 'exc_min', 'q_max', 'nw_t_min',
            'same_sign_half_n', 'same_sign_of', 'dyn_mean', 'top3_share_max', 'sig_ac1']
    show = [c for c in show if c in summ.columns]
    print('\n===== 分级计数 =====')
    print(summ['tier'].value_counts().to_string())
    for t in ['A强', 'B中', 'B中|无动态增量', 'C弱']:
        sub = summ[summ['tier'] == t]
        if len(sub):
            print(f'\n===== {t} ({len(sub)}) =====')
            print(sub[show].head(30).to_string(index=False))

    # 报告
    lines = [f'conditional_mining 报告(K_族: 条件/门控层) | pools={list(longs)} | run={run_id}',
             f'评估窗 {args.start}~, horizons={horizons}, top_k={TOP_K}, min_cs={MIN_CS}',
             '口径: 信号日收盘 -> 次日开盘入场 -> 持有h日开盘出场; '
             'NW-HAC(lag=h-1) on 日级超额; BH-FDR 族=每池每h全部候选; 分级=三池同号+FDR+分半+动态增量',
             '',
             '== 分级计数 ==',
             summ['tier'].value_counts().to_string(), '']
    for t in ['A强', 'B中', 'B中|无动态增量', 'C弱', 'D无效', 'S静态身份']:
        sub = summ[summ['tier'] == t]
        if len(sub):
            lines.append(f'== {t} ({len(sub)}) ==')
            lines.append(sub[show].to_string(index=False))
            lines.append('')
    for pool, df in longs.items():
        for h in horizons:
            sub = df[df['h'] == h].sort_values('excess_k', ascending=False)
            if len(sub):
                lines.append(f'== {pool} h={h} 全量(excess_k 降序) ==')
                cols = ['signal', 'group', 'excess_k', 'icir', 'nw_t', 'fdr_q', 'n_days']
                lines.append(sub[cols].to_string(index=False))
                lines.append('')
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')


# ==========================================================================
# ==== 源自 a_stat_mining.py ====  改名: main->asm_main, GROUPS->asm_GROUPS, POOLS->asm_POOLS, HORIZONS->asm_HORIZONS, eval_pool->asm_eval_pool, group_of->asm_group_of
# ==========================================================================
asm_POOLS = {
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data_research.pkl'),
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data_research.pkl'),
    'company_1000': os.path.join(_PKL_DIR, 'company_1000_day_ta_data_research.pkl'),
}

asm_HORIZONS = [5, 20, 60]

asm_GROUPS = {  # 前缀 -> 分族
    'R_res': '残差动量',
    'I_id': '信息离散度', 'I_iv': '信息离散度',
    'W_dd': '回撤时间', 'W_martin': '回撤时间', 'W_hi': '回撤时间',
}

def asm_group_of(name: str) -> str:
    for k, v in asm_GROUPS.items():
        if name.startswith(k):
            return v
    return '其他'

def build_a_cands(panel: pd.DataFrame) -> dict:
    """构建全部 R_/I_/W_ 候选(全历史构建, 因果安全; 评估窗截取在主流程做)."""
    def w(col):
        return panel[col].unstack('symbol').sort_index()

    close = w('Close')
    ret1 = close.pct_change()
    out = {}

    # ---- R_ 残差动量族(Blitz-Huij-Martens 2007) ----
    # 残差: 对池等权收益 pr 的滚动 OLS(60d) 中心化残差(含截距近似: cov/var beta + 均值中心化)
    pr = ret1.mean(axis=1)                     # 池等权收益(仅用 t 及之前, 与 G_ 族同口径)
    RB = 60
    beta = ret1.rolling(RB, min_periods=RB // 2).cov(pr) \
        .div(pr.rolling(RB, min_periods=RB // 2).var(), axis=0)
    resid = ret1.sub(ret1.rolling(RB, min_periods=RB // 2).mean()) \
        .sub(beta.mul(pr.sub(pr.rolling(RB, min_periods=RB // 2).mean()), axis=0))
    out['R_resmom121'] = resid.shift(21).rolling(231, min_periods=120).sum()   # 12-1 残差动量
    out['R_resmom60'] = resid.rolling(60, min_periods=30).sum()
    out['R_resmom20'] = resid.rolling(20, min_periods=10).sum()
    out['R_resvol121'] = out['R_resmom121'] \
        / resid.rolling(252, min_periods=120).std()                            # 波动调整版

    # ---- I_ 信息离散度族(Da-Gurun-Warachka 2014) ----
    # ID = sign(mom_q) x (跌天占比 - 涨天占比): 高离散 = 路径剧烈 -> 动量衰减快
    up = (ret1 > 0).where(ret1.notna()).astype(float)
    dn = (ret1 < 0).where(ret1.notna()).astype(float)
    for q in (20, 60, 121):
        upsh = up.rolling(q, min_periods=q // 2).mean()
        dnsh = dn.rolling(q, min_periods=q // 2).mean()
        mom = close.pct_change(q)
        out[f'I_id{q}'] = np.sign(mom) * (dnsh - upsh)
    # 特质波动/总波动(1-R^2 近似): 离散度的 IVOL 代理
    rstd = resid.rolling(60, min_periods=30).std()
    tstd = ret1.rolling(60, min_periods=30).std()
    out['I_ivshare60'] = rstd / tstd.replace(0, np.nan)

    # ---- W_ 回撤路径时间族 ----
    # 水下: 收盘 < 250 日滚动高点
    hi250 = close.rolling(250, min_periods=60).max()
    uw = close.lt(hi250)
    out['W_ddshare250'] = uw.astype(float).rolling(250, min_periods=60).mean()  # 水下时间占比
    # 最长单次连续水下段: rowno-ffill run-length(G_days_hi250 同技巧, 出水行=0)
    n = len(close)
    rowno = pd.DataFrame(np.repeat(np.arange(n, dtype=float)[:, None], close.shape[1], axis=1),
                         index=close.index, columns=close.columns)
    dur = rowno - rowno.where(~uw).ffill()     # 当前连续水下天数(出水=0)
    out['W_ddtime121'] = dur.rolling(121, min_periods=30).max()
    # Martin ratio: 均值收益 / Ulcer(回撤深度平方根)——回撤版 Sharpe
    dd60 = (close / close.rolling(60, min_periods=30).max() - 1.0).clip(upper=0.0)
    ulcer60 = dd60.pow(2).rolling(60, min_periods=30).mean().pow(0.5)
    out['W_martin60'] = ret1.rolling(60, min_periods=30).mean() \
        .div(ulcer60.replace(0, np.nan))
    # 60 日内创新高(250 日高点, 用 t-1 高点判定)次数——恢复能力/新新鲜度频率
    hi250p = hi250.shift(1)
    out['W_hihits60'] = close.gt(hi250p).astype(float).rolling(60, min_periods=10).sum()

    return {k: v for k, v in out.items()
            if isinstance(v, pd.DataFrame) and v.notna().sum().sum() > 0}

def asm_eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
    _, panel = load_panel(pkl, 'day')
    dts = panel.index.get_level_values('date')
    open_wide = panel['Open'].unstack('symbol').sort_index()
    cands = build_a_cands(panel)
    start_ts = pd.Timestamp(start)
    cands = {k: v.loc[v.index >= start_ts] for k, v in cands.items()}
    n_sym = len(panel.index.get_level_values('symbol').unique())
    print(f'\n===== {pool}: {n_sym} 标的, {dts.min().date()}~{dts.max().date()} | '
          f'候选 {len(cands)} 个, 评估窗 {start}~ =====', flush=True)

    fwds = {h: tradable_fwd(open_wide, h) for h in horizons}
    vrows, screen_rows = [], []
    for h in horizons:
        fwd = fwds[h]
        rows, pvals = [], []
        for name, sig in cands.items():
            try:
                rec = screen_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception as e:
                print(f'  [SKIP screen] {name}: {e}', flush=True)
                continue
            if not rec:
                continue
            exc, k = daily_excess(sig, fwd)
            t, p, nn = nw_tstat(exc.values, lag=max(h - 1, 1))
            rec.update({'signal': name, 'group': asm_group_of(name),
                        'nw_t': round(t, 2) if pd.notna(t) else np.nan,
                        'nw_p': round(p, 4) if pd.notna(p) else np.nan,
                        'nw_n': nn, 'exc_k_daily': k})
            rows.append(rec)
            pvals.append(p)
            try:
                v = validate_one(sig, fwd, top_k=TOP_K, min_cs=MIN_CS)
            except Exception:
                v = {}
            if v:
                v['signal'] = name
                v['h'] = h
                vrows.append(v)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df['fdr_q'] = bh_qvals(df['nw_p'].values)
        cols = ['signal', 'group', 'excess_k', 'topk_ret', 'pool_ret', 'ls_spread',
                'ic_mean', 'icir', 't_stat', 'ic_pos_rate', 'topk_turnover',
                'nw_t', 'nw_p', 'fdr_q', 'nw_n', 'n_days']
        df = df[cols].sort_values('excess_k', ascending=False)
        df.to_csv(os.path.join(out_dir, f'{pool}_h{h}_screen.csv'),
                  index=False, encoding='utf-8-sig')
        screen_rows.append(df)
        print(f'-- h={h}: {len(df)} 候选; q<=0.10 的 {int((df["fdr_q"] <= 0.10).sum())} 个; '
              f'q<=0.25 的 {int((df["fdr_q"] <= 0.25).sum())} 个', flush=True)
        print(df.head(10).to_string(index=False), flush=True)

    ac1 = {}
    for name, sig in cands.items():
        try:
            ac1[name] = sig_ac1(sig)
        except Exception:
            ac1[name] = np.nan
    if vrows:
        vdf = pd.DataFrame(vrows)
        vdf['sig_ac1'] = vdf['signal'].map(ac1)
        vdf.to_csv(os.path.join(out_dir, f'{pool}_validate.csv'),
                   index=False, encoding='utf-8-sig')

    long_rows = []
    for h, df in zip(horizons, screen_rows):
        vd = {r['signal']: r for r in vrows if r['h'] == h} if vrows else {}
        for _, r in df.iterrows():
            v = vd.get(r['signal'], {})
            long_rows.append({
                'pool': pool, 'h': h, 'signal': r['signal'], 'group': r['group'],
                'excess_k': r['excess_k'], 'topk_ret': r['topk_ret'], 'pool_ret': r['pool_ret'],
                'icir': r['icir'], 't_stat': r['t_stat'], 'topk_turnover': r['topk_turnover'],
                'nw_t': r['nw_t'], 'nw_p': r['nw_p'], 'fdr_q': r['fdr_q'], 'nw_n': r['nw_n'],
                'n_days': r['n_days'], 'sig_ac1': ac1.get(r['signal'], np.nan),
                'exc_half1': v.get('exc_half1', np.nan), 'exc_half2': v.get('exc_half2', np.nan),
                'dyn_minus_static': v.get('dyn_minus_static', np.nan),
                'top3_share': v.get('top3_share', np.nan),
                'n_eff_symbols': v.get('n_eff_symbols', np.nan),
            })
    return pd.DataFrame(long_rows)

def asm_main():
    ap = argparse.ArgumentParser(description='附录 A 类统计维度挖矿(R_/I_/W_族, 美股三池+FDR+分级)')
    ap.add_argument('--pools', default='etf_3x,company_300,company_1000')
    ap.add_argument('--pkl', default=None, help='"池=路径" 覆盖默认数据源(可重复)')
    ap.add_argument('--start', default=START)
    ap.add_argument('--horizons', default=','.join(str(h) for h in asm_HORIZONS))
    ap.add_argument('--signals', default=None, help='逗号分隔, 只评估指定候选')
    args = ap.parse_args()

    pools = [x.strip() for x in args.pools.split(',') if x.strip()]
    for kv in (args.pkl or '').split(';'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            asm_POOLS[k.strip()] = v.strip()
    horizons = [int(x) for x in args.horizons.split(',')]

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(HERE, 'output',
                           f'a_mine_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    longs = {}
    for pool in pools:
        if pool not in asm_POOLS or not os.path.exists(asm_POOLS[pool]):
            print(f'[ERROR] 池数据缺失: {pool} -> {asm_POOLS.get(pool)}')
            continue
        df = asm_eval_pool(pool, asm_POOLS[pool], out_dir, horizons, args.start)
        if not df.empty:
            longs[pool] = df
    if not longs:
        print('[ERROR] 无任何池产出')
        sys.exit(1)

    summ = build_summary(longs)
    if args.signals:
        keep = [x.strip() for x in args.signals.split(',') if x.strip()]
        summ = summ[summ['signal'].isin(keep)]
    res = summ.apply(classify_alpha, axis=1, result_type='expand')
    summ['tier'] = res[0]
    summ['score'] = res[1]
    summ = summ.sort_values(['score', 'exc_mean'], ascending=False)
    fp = os.path.join(out_dir, 'summary.csv')
    summ.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f'\n== 汇总 == {fp} ({len(summ)} 行)')

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 60)
    show = ['signal', 'h', 'group', 'tier', 'score', 'dir', 'n_pool', 'n_consist',
            'cons_ratio', 'exc_mean', 'exc_min', 'q_max', 'nw_t_min',
            'same_sign_half_n', 'same_sign_of', 'dyn_mean', 'top3_share_max', 'sig_ac1']
    show = [c for c in show if c in summ.columns]
    print('\n===== 分级计数 =====')
    print(summ['tier'].value_counts().to_string())
    for t in ['A强', 'B中', 'B中|无动态增量', 'C弱']:
        sub = summ[summ['tier'] == t]
        if len(sub):
            print(f'\n===== {t} ({len(sub)}) =====')
            print(sub[show].head(30).to_string(index=False))

    # 报告
    lines = [f'a_stat_mining 报告(R_/I_/W_族) | pools={list(longs)} | run={run_id}',
             f'评估窗 {args.start}~, horizons={horizons}, top_k={TOP_K}, min_cs={MIN_CS}',
             '口径: 信号日收盘 -> 次日开盘入场 -> 持有h日开盘出场; '
             'NW-HAC(lag=h-1) on 日级超额; BH-FDR 族=每池每h全部候选; 分级=三池同号+FDR+分半+动态增量',
             '',
             '== 分级计数 ==',
             summ['tier'].value_counts().to_string(), '']
    for t in ['A强', 'B中', 'B中|无动态增量', 'C弱', 'D无效', 'S静态身份']:
        sub = summ[summ['tier'] == t]
        if len(sub):
            lines.append(f'== {t} ({len(sub)}) ==')
            lines.append(sub[show].to_string(index=False))
            lines.append('')
    for pool, df in longs.items():
        for h in horizons:
            sub = df[df['h'] == h].sort_values('excess_k', ascending=False)
            if len(sub):
                lines.append(f'== {pool} h={h} 全量(excess_k 降序) ==')
                cols = ['signal', 'group', 'excess_k', 'icir', 'nw_t', 'fdr_q', 'n_days']
                lines.append(sub[cols].to_string(index=False))
                lines.append('')
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n== 完成 == 输出目录: {out_dir}')


# ==========================================================================
# ==== 源自 factor_signal.py ====  改名: main->fsg_main
# ==========================================================================
WEIGHT_PRESETS = {
    # 4 因子版(默认交付口径)
    'gold4': {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2, 'F_obv20': 0.1},
    # 3 因子版(etf_3x 2021-2026 回测报告 201620 口径: total_ret 6.30 / sharpe 0.98)
    'gold3': {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2},
}

def compute_composite(study: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """组合分数 = Σ w_i * 成分当日截面百分位排名(rank pct, 0~1).
    与 score_backtest.compute_composite 逐行同口径: 缺失(NaN)以中性 0.5 参与;
    单一成分整列缺失时该成分退化为常数, 无排序贡献."""
    total = float(sum(abs(w) for w in weights.values()))
    if total <= 0:
        raise ValueError('权重绝对值和为 0')
    comp = None
    for col, w in weights.items():
        if col not in study.columns:
            raise ValueError(f'组合分数成分列不存在: {col}')
        wide = pd.to_numeric(study[col], errors='coerce').unstack('symbol').sort_index()
        pct = wide.rank(axis=1, pct=True)  # 当日截面, NaN 保留
        part = (w / total) * pct
        comp = part if comp is None else comp.add(part, fill_value=0.0)
    return comp.fillna(0.5)

def build_factor_signal(pkl_path: str, interval: str, weights: dict):
    """读 pkl 计算因子与 composite.

    :returns: (close, factors, comp_wide)
      close: (date, symbol) 收盘宽表;
      factors: {factor: (date, symbol) 宽表};
      comp_wide: (date, symbol) composite 宽表
    """
    raw, panel = load_panel(pkl_path, interval)
    close = panel['Close'].unstack('symbol').sort_index()

    mined = build_mined(panel)
    missing = [f for f in weights if f not in mined]
    if missing:
        raise ValueError(f'build_mined 不产出因子: {missing}')
    factors = {f: mined[f].reindex(index=close.index, columns=close.columns) for f in weights}

    # 堆成长表后走 compute_composite(与回测同口径)
    study = pd.DataFrame({f: factors[f].stack() for f in weights})
    study.index.names = ['date', 'symbol']
    comp_wide = compute_composite(study, weights)
    return close, factors, comp_wide

def first_valid(wide: pd.DataFrame):
    """该因子首个非全 NaN 的日期; 全 NaN 返回 None."""
    ok = wide.notna().any(axis=1)
    return ok.idxmax() if bool(ok.any()) else None

def snapshot(close: pd.DataFrame, factors: dict, comp_wide: pd.DataFrame,
             weights: dict, as_of: pd.Timestamp, top_k: int, exit_rank: int) -> pd.DataFrame:
    """as_of 截面快照: 因子值 + composite + rank + zone 分区."""
    row = comp_wide.loc[as_of]
    order = row.rank(ascending=False, method='first')  # 与 run_engine 排名口径一致
    snap = pd.DataFrame({'composite': row, 'rank': order.astype(int)})
    for f in weights:
        snap[f] = factors[f].loc[as_of].reindex(snap.index)
    zone = np.where(order <= top_k, f'TOP(入场 rank<={top_k})',
                    np.where(order <= exit_rank, f'HOLD(rank<={exit_rank})', 'EXIT(退出区)'))
    snap['zone'] = zone
    return snap.sort_values('rank')

def fsg_main():
    ap = argparse.ArgumentParser(
        description='独立因子信号: 黄金组合因子 composite + 当日 top-K 快照(不修改现有系统)')
    ap.add_argument('--pool', default='etf_3x', help='池名, 默认 etf_3x')
    ap.add_argument('--interval', default='day', help='数据频率, 默认 day')
    ap.add_argument('--pkl-dir', default=os.path.join(os.path.expanduser('~'), 'quant'),
                    help='pkl 目录, 默认 ~/quant(生产管道 technical_analyst_parallel.py 输出目录)')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl 路径(优先)')
    ap.add_argument('--weights-preset', default='gold4', choices=sorted(WEIGHT_PRESETS),
                    help='权重预设, 默认 gold4(4 因子版)')
    ap.add_argument('--weights', default=None,
                    help='JSON 权重字典, 覆盖预设, 如 "{\\"F_mom121\\": 0.5, \\"F_er20\\": 0.3}"')
    ap.add_argument('--top-k', type=int, default=5, help='入场排名阈值, 默认 5(回测口径)')
    ap.add_argument('--exit-rank', type=int, default=12, help='退出排名阈值, 默认 12(回测口径)')
    ap.add_argument('--as-of', default=None, help='快照日期 YYYY-MM-DD, 默认最新交易日')
    ap.add_argument('--recent', type=int, default=3,
                    help='额外打印最近 N 日 top-K 组成(观察信号连续性), 默认 3')
    ap.add_argument('--out-csv', default=None,
                    help='可选: 导出全历史长表 csv(date, symbol, F_*, composite, rank)')
    args = ap.parse_args()

    weights = dict(WEIGHT_PRESETS[args.weights_preset])
    if args.weights:
        weights = {k: float(v) for k, v in json.loads(args.weights).items()}
        unknown = [k for k in weights if not k.startswith('F_')]
        if unknown:
            print(f'[WARN] 权重键非 F_* 因子命名: {unknown}(须为 build_mined 输出键)')

    pkl_path = args.pkl_path or os.path.join(args.pkl_dir, f'{args.pool}_{args.interval}_ta_data.pkl')
    if not os.path.exists(pkl_path):
        print(f'[ERROR] 找不到 pkl: {pkl_path}')
        sys.exit(1)

    close, factors, comp_wide = build_factor_signal(pkl_path, args.interval, weights)
    eff = {f: first_valid(factors[f]) for f in weights}          # 各因子首个生效日
    valid_eff = [d for d in eff.values() if d is not None]
    first_eff = min(valid_eff) if valid_eff else None            # 最早生效(截面最早有区分度)

    # as-of 日期: 默认数据最新交易日; 截断到因子有效期之后才有区分度
    as_of = pd.Timestamp(args.as_of) if args.as_of else close.index[-1]
    as_of = comp_wide.index[comp_wide.index <= as_of][-1]

    # ---------------------------------------------------------------- 输出 ---------------------------------------------------------------- #
    print('== 因子信号快照(独立脚本, 未修改现有系统) ==')
    print(f'  pkl      : {pkl_path}')
    print(f'  标的/日期: {comp_wide.shape[1]} 个, {close.index[0].date()} ~ {close.index[-1].date()}')
    print('  因子生效  : ' + ', '.join(f'{f} {d.date() if d is not None else "无值!"}'
                                       for f, d in eff.items()))
    print(f'  权重     : {weights}')
    print(f'  as-of    : {as_of.date()} (信号日收盘决策, 下一交易日开盘执行)')

    # warmup 防护: as-of 截面有效标的不足时警告(如 pkl 仅含约 1 年数据时 F_mom121 全 NaN,
    # compute_composite 的 fillna(0.5) 语义会让它静默退化为常数, composite 看似正常实则偏离回测口径)
    n_sym = int(comp_wide.shape[1])
    w_total = sum(abs(w) for w in weights.values())
    for f, w in weights.items():
        ok = int(factors[f].loc[as_of].notna().sum())
        if ok < n_sym:
            msg = f'  [WARN] {f}: as-of 截面仅 {ok}/{n_sym} 个标的有效(权重 {abs(w)/w_total:.0%})'
            msg += ' — 完全失效, 已退化为常数, 结果偏离回测口径!' if ok == 0 else ' — 部分失效'
            print(msg)

    snap = snapshot(close, factors, comp_wide, weights, as_of, args.top_k, args.exit_rank)
    cols = ['rank', 'composite'] + list(weights) + ['zone']
    print(f'\n-- {as_of.date()} 截面(按 composite 降序, rank 与 run_engine 口径一致) --')
    print(snap[cols].to_string(float_format=lambda x: f'{x:.4f}'))

    top_syms = snap.index[snap['rank'] <= args.top_k].tolist()
    print(f'\n  TOP-{args.top_k} 候选(入场区): {top_syms}')
    print(f'  注: zone 不含持仓状态; 引擎实际为滞回进出(已持有者 rank<={args.exit_rank} 不退出), '
          f'此处仅按当日截面分区作参考')

    if args.recent > 0:
        mask = comp_wide.index <= as_of
        if first_eff is not None:
            mask &= comp_wide.index >= first_eff
        recent_dates = comp_wide.index[mask][-args.recent:]
        print(f'\n-- 最近 {len(recent_dates)} 日 top-{args.top_k}(composite 值) --')
        for d in recent_dates:
            row = comp_wide.loc[d].sort_values(ascending=False).head(args.top_k)
            cells = '  '.join(f'{sym} {v:.3f}' for sym, v in row.items())
            print(f'  {d.date()}  {cells}')

    if args.out_csv:
        out_path = args.out_csv
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        long = pd.DataFrame({f: factors[f].stack() for f in weights})
        long['composite'] = comp_wide.stack()
        long['rank'] = comp_wide.rank(axis=1, ascending=False, method='first').stack()
        long.index.names = ['date', 'symbol']
        long = long.reset_index().sort_values(['date', 'rank'])
        long.to_csv(out_path, index=False, encoding='utf-8')
        print(f'\n  csv 导出: {out_path}({len(long)} 行; 全部因子生效日 {max(valid_eff).date() if valid_eff else "?"} '
              f'之前 warmup 段仅部分因子参与, 截面信息不完整)')


# ==========================================================================
# ==== 源自 export_pool_context.py ====  改名: main->epc_main
# ==========================================================================
CONTEXT_FACTORS = ['F_mom121', 'F_er20', 'F_obv20', 'F_idiovol60']

def export_pool_context(pkl_path: str, interval: str = 'day', out_path: str = None) -> str:
    """从研究 pkl 计算全池 4 因子 + 等权池收益, 导出池上下文 csv.

    :param pkl_path: 研究 pkl 路径({symbol}_{interval} -> df)
    :param interval: 数据频率, 'day' 为默认
    :param out_path: 输出 csv 路径, 默认 research/data/{pool}_context.csv(按 pkl 名推断池名)
    :returns: 实际写出的 csv 路径
    """
    raw, panel = load_panel(pkl_path, interval)
    close = panel['Close'].unstack('symbol').sort_index()
    ret1 = close.pct_change()

    # 等权池收益: 与 build_mined F 段完全同口径(当日池内有效列等权平均)
    pool_ret = ret1.mean(axis=1)

    # 全池因子(池口径, F_idiovol60 含 β 剥离)
    mined = build_mined(panel)
    factors = {f: mined[f].reindex(index=close.index, columns=close.columns) for f in CONTEXT_FACTORS}

    # 堆成长表: date, symbol, pool_ret, F_xxx
    stacked = pd.DataFrame({f: factors[f].stack() for f in CONTEXT_FACTORS})
    stacked.index.names = ['date', 'symbol']
    long = stacked.reset_index()
    long['pool_ret'] = long['date'].map(pool_ret)
    long = long.sort_values(['date', 'symbol'])

    if out_path is None:
        base = os.path.splitext(os.path.basename(pkl_path))[0]
        pool = base.replace(f'_{interval}_ta_data', '')
        out_path = os.path.join(HERE, 'data', f'{pool}_context.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    long.to_csv(out_path, index=False, encoding='utf-8')

    dts = long['date']
    print(f'== 池上下文导出 == {out_path}')
    print(f'  标的 {long["symbol"].nunique()} 个, {dts.min()} ~ {dts.max()}, {len(long)} 行')
    print(f'  因子: {CONTEXT_FACTORS} + pool_ret')
    return out_path

def epc_main():
    ap = argparse.ArgumentParser(description='导出池上下文 csv(全池 4 因子 + 等权池收益)')
    ap.add_argument('--pool', default='etf_3x', help='池名, 默认 etf_3x')
    ap.add_argument('--interval', default='day', help='数据频率, 默认 day')
    ap.add_argument('--pkl-dir', default=os.path.join(HERE, 'data'),
                    help='pkl 所在目录, 默认 research/data')
    ap.add_argument('--pkl-path', default=None, help='直接指定 pkl 路径(优先)')
    ap.add_argument('--out-path', default=None, help='输出 csv 路径, 默认 research/data/{pool}_context.csv')
    args = ap.parse_args()

    pkl_path = args.pkl_path or os.path.join(args.pkl_dir, f'{args.pool}_{args.interval}_ta_data.pkl')
    if not os.path.exists(pkl_path):
        print(f'[ERROR] 找不到 pkl: {pkl_path}')
        sys.exit(1)
    export_pool_context(pkl_path, interval=args.interval, out_path=args.out_path)


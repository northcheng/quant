# -*- coding: utf-8 -*-
"""
factor_research.py — 独立只读研究模块: 指标正确性审计 + 条件IC + 事件级研究

设计原则(确保不影响现有项目):
  1. 只读: 仅读取 {pkl_dir}/{pool}_{interval}_ta_data.pkl, 不写任何现有路径
  2. 独立: 不 import quant 包; 审计公式按 bc_technical_analysis.py 的实现逐条独立重写
     (审计的价值在于独立复现; 复用项目自身函数会变成循环验证)
  3. 隔离: 全部输出写入本脚本同目录下 output/{pool}_{run_id}/

三层内容:
  Layer 0  原始指标审计: 从 pkl 自身 OHLC 独立重算 rate/tr/atr/rsi/adx/adi/bb/
           ichimoku/kama, 与 pkl 列逐行比对. calculate_ta_feature 是"先切片再计算"
           (bc_technical_analysis.py L885), pkl 内指标就是从窗口首行起算的,
           因此独立重算理论上应与 pkl 精确一致, 不一致处即真实缺陷或自定义行为.
  Layer 0.5 分数聚合审计: 从 pkl 自身中间列复核 trend_score/trend_magnitude/
           trigger_score/pattern_score 及 *_alpha 归一化的聚合逻辑.
  Layer 1  条件IC: 多周期(默认1/5/10/20/60)Spearman秩IC, 全样本与门内样本分别统计,
           ICIR/t值/胜率, 以及按日的截面分位数单调性.
  Layer 2  事件研究: 以"门"(状态, 非事件)重放抽取合成交易(信号日收盘确认, 次日开盘
           成交), 统计胜率/盈亏比/持有天数/MFE/MAE, 以及事件级IC(同日入场分数与
           交易最终收益的截面秩相关).

防前视纪律: 因子值取信号日(收盘已知)的值; IC 的前瞻收益为 Close_t -> Close_{t+h};
事件的入场/出场均为信号日次日的开盘价.

用法:
  python factor_research.py --pool etf_3x
  python factor_research.py --pool etf_3x --pkl-dir "C:/Users/northcheng/quant"
  python factor_research.py --pool etf_3x --start 2020-01-01 --gate-col trend_magnitude_day
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Windows 控制台避免 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ================================================================ 参数基准 ================================================================ #
# 与 ta_config.json / bc_technical_analysis.py 默认签名一一对应
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
# add_support_resistance (bc_technical_analysis.py L2821) 对这些列统一 round(3) 后落盘:
# 审计时先把重算值 round(3) 再与 pkl 比对, 排除落盘精度截断的干扰(实测重算与项目原函数逐位一致)
ROUND3_COLS = {'kama_fast', 'kama_slow', 'tankan', 'kijun', 'candle_gap_top', 'candle_gap_bottom'}

DEFAULT_FACTOR_COLS = [
    'trend_magnitude', 'trend_magnitude_alpha', 'trend_magnitude_change',
    'pattern_score', 'pattern_score_alpha', 'pattern_score_change',
    'trigger_score', 'position_score', 'boundary_score', 'break_score',
    'trend_score', 'trend_score_change',
    'candle_position_score', 'candle_pattern_score',
    'adx_value', 'adx_strength', 'rsi', 'rate',
]


# ================================================================ 项目公式独立重写 ================================================================ #
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


# ---- 分数聚合(输入为 pkl 自身中间列) ----
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


# ================================================================ 通用工具 ================================================================ #
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


# ================================================================ Layer 0/0.5 审计 ================================================================ #
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


# ================================================================ Layer 1 条件IC ================================================================ #
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


# ================================================================ Layer 2 事件研究 ================================================================ #
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


# ================================================================ 主流程 ================================================================ #
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


def main():
    ap = argparse.ArgumentParser(description='独立只读研究: 指标审计 + 条件IC + 事件研究')
    ap.add_argument('--pool', default='etf_3x', help='池名, 默认 etf_3x')
    ap.add_argument('--interval', default='day', help='数据频率, 默认 day')
    ap.add_argument('--pkl-dir', default=r'C:\Users\northcheng\quant', help='pkl 所在目录')
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
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', f'{args.pool}_{run_id}')
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


if __name__ == '__main__':
    main()
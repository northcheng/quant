# -*- coding: utf-8 -*-
"""
alpha_mining.py — 独立只读研究模块: 「m_trend_score_alpha 式」信号挖掘(四池 + FDR + 分级)

定位: 在 signal_search(D_ 23 个) 与 factor_mining(F_ 33 个, 仅单池无 FDR) 之后,
以 m_trend_score_alpha = normalize_causal(|ichimoku_distance|, 252, 60) 为原型,
把「指标绝对值的因果归一化强度」这一构造模式推广到 panel 全部强度型指标,
并补齐验证缺口: 四池同号性 + Newey-West HAC + Benjamini-Hochberg FDR + 分级.

候选分族(共 ~46 个):
  H_ alpha 化   (21) 对 panel 强度/距离型指标做 nc(|x|, 252, 60) — 原型模式的直接推广
  S_ 方向 score (2)  sign(x) * nc(|x|) — m_trend_score 「方向×强度」模板的普适性验证
  X_ 新异象     (8)  MAX 效应/涨天数占比/低波 alpha 化/动量 alpha 化等 D_/F_ 未覆盖维度
  N_ 反向版     (6)  上轮 factor_mining 已证跨期负向稳健因子的取反(负向信号直接可用)
  C_ 秩组合     (2)  日度截面秩组合(质量动量等二阶组合)
  锚对照        (9)  H_ichimoku_alpha 精确重建 m_trend_score_alpha + pkl 原列 alpha
                     + 上轮 etf_3x 单池 Top 因子(验证其跨池稳健性)

统计口径(与既有工具完全一致, 无前视):
  信号日 t 收盘已知信号值 -> t+1 开盘入场 -> 持有 h 日 -> t+1+h 开盘出场
  挖掘防伪三坑:
    1) 截面相关 -> 日级超额序列(top-k 均值 - 池等权均值) 逐日聚合
    2) 窗口重叠 -> Newey-West HAC 标准误, lag = h-1 (复用 conditional_eval.nw_tstat)
    3) 多重检验 -> 每 (pool, h) 族内 Benjamini-Hochberg FDR (复用 conditional_eval.bh_qvals)
  真伪检验: 复用 signal_search.validate_one(静态身份/分半/静态退化对照)

全历史构建 -> 只在 START 之后评估(nc 的 252 窗口在评估前已 warmup, 因果安全).

输出: research/output/alpha_mine_{run_id}/
  {pool}_h{h}_screen.csv   每池每周期全候选(含 NW t / p / BH q)
  {pool}_validate.csv      真伪检验汇总(全候选 × 周期)
  summary.csv              四池汇总 + 同号性 + FDR + 分级(对标 eval_summary)
  report.txt

用法:
  python alpha_mining.py --pools etf_3x,company_300,hs300,a_etf_all
  python alpha_mining.py --pools etf_3x --horizons 5,20 --signals H_atr_alpha,H_bbw_alpha
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel, normalize_causal                # noqa: E402
from signal_search import tradable_fwd, screen_one, validate_one       # noqa: E402
from factor_mining import build_mined                                  # noqa: E402
from conditional_eval import nw_tstat, bh_qvals                        # noqa: E402
from indicator_eval import sig_ac1                                     # noqa: E402

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ================================================================ 参数 ================================================================ #
_PKL_DIR = os.path.join(os.path.expanduser('~'), 'quant')   # 本地数据目录(跨机器: 家目录下 quant)
# 研究一律优先 research 全史 pkl(生产 pkl 截短或滞后重建; 见 research_summary_20260918.md 数据规则).
# 生产桥 signal_bridge.py 不走本表默认值, 显式优先生产版 pkl(每日更新, research 快照滞后 1~2 天).
POOLS = {
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data_research.pkl'),  # 生产 pkl 截短 2025+, 用 research 全史
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data_research.pkl'),  # 全史; 生产版仅保留约近 2 年
    'company_1000': os.path.join(_PKL_DIR, 'company_1000_day_ta_data_research.pkl'),  # 2020+ 全史(无生产版)
    'hs300':       os.path.join(_PKL_DIR, 'hs300_day_ta_data_research.pkl'),  # 2020+ 全史
    'a_etf_all':   os.path.join(_PKL_DIR, 'a_etf_all_day_ta_data_research.pkl'),  # 全史
}
START = '2021-01-01'
HORIZONS = [5, 20, 60]
TOP_K = 5
MIN_CS = 10
NORM_WINDOW, NORM_MINP = 252, 60     # 与 bc_technical_analysis day 口径一致
STATIC_AC1 = 0.99                    # 截面排序几乎不随时间变化 => 固定选票

# 上轮 factor_mining(etf_3x 单池) Top 因子, 本轮跨池复验
F_ANCHORS = ['F_mom121', 'F_er20', 'F_updnvol20', 'F_volstab20',
             'F_obv20', 'F_alpha60', 'F_beta60']
# 上轮已证负向稳健因子, 取反成正向信号
F_NEG = ['F_idiovol60', 'F_cvcorr20', 'F_ulcer60', 'F_range20',
         'F_kurt60', 'F_dnvol20']


# ================================================================ 候选构建 ================================================================ #
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


def group_of(name: str) -> str:
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


# ================================================================ 日级序列 + NW/FDR ================================================================ #
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


# ================================================================ 单池评估 ================================================================ #
def eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
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
            rec.update({'signal': name, 'group': group_of(name),
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


# ================================================================ 四池汇总 + 分级 ================================================================ #
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


# ================================================================ 主流程 ================================================================ #
def main():
    ap = argparse.ArgumentParser(description='独立只读研究: m_trend_score_alpha 式信号挖掘(四池+FDR+分级)')
    ap.add_argument('--pools', default='etf_3x,company_300,hs300,a_etf_all')
    ap.add_argument('--pkl', default=None, help='"池=路径" 覆盖默认数据源(可重复)')
    ap.add_argument('--start', default=START)
    ap.add_argument('--horizons', default=','.join(str(h) for h in HORIZONS))
    ap.add_argument('--signals', default=None, help='逗号分隔, 只评估指定候选')
    args = ap.parse_args()

    pools = [x.strip() for x in args.pools.split(',') if x.strip()]
    for kv in (args.pkl or '').split(';'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            POOLS[k.strip()] = v.strip()
    horizons = [int(x) for x in args.horizons.split(',')]

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                           f'alpha_mine_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    longs = {}
    for pool in pools:
        if pool not in POOLS or not os.path.exists(POOLS[pool]):
            print(f'[ERROR] 池数据缺失: {pool} -> {POOLS.get(pool)}')
            continue
        df = eval_pool(pool, POOLS[pool], out_dir, horizons, args.start)
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


if __name__ == '__main__':
    main()

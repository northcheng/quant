# -*- coding: utf-8 -*-
"""
factor_mining2.py — 独立只读研究模块: 第二轮新因子挖矿(G_ 族, 美股三池 + FDR + 分级)

定位: 在 D_(23 经典)/F_(38 结构)/H_/S_/X_/N_/C_(46 alpha化) 三轮之后,
挖既有候选**未覆盖**的维度, 每个因子给出文献/逻辑出处:

  波动微观结构 (4)
    G_parkcc20   Parkinson(HL)波动 / Close-Close 波动比 — 跳空占比(隔夜vs日内波动结构, Alizadeh等)
    G_oviv20     隔夜波动/日内波动比 — 美股个股隔夜-日内微观结构差异(Lou-Polk-Skouloudakis式分解)
    G_semidn20   下行半方差占比 — 下行风险(Ang et al. 2006 的半方差版)
    G_extasym60  60日极值不对称 (max单日涨 + max单日跌, 带符号) — 尾部不对称
  时序结构 (5)
    G_ac1_60     个体60日自相关 — 该标的自身是趋势延续还是均值回归(Chan-Jegadeesh式)
    G_vr5        方差比 VR(5) — Lo-MacKinlay, >1 趋势 / <1 均值回归
    G_vr20       方差比 VR(20) — 期限结构版
    G_gaprev20   隔夜跳空×日内收益的20日相关 — 跳空回补倾向
    G_ac1mom     AC1 × sign(mom60) — 个体延续倾向×当前动量方向(个性化时序动量)
  量能不对称 (6)
    G_impupdn20  上涨/下跌单位收益所需量能比 — 买卖压力不对称(方向性冲击成本)
    G_volbeta60  成交量对池总量量的弹性 — 量能共动(拥挤度代理)
    G_amiasym20  Amihud(上涨日) - Amihud(下跌日) — 冲击成本不对称
    G_vwapdev20  收盘对20日VWAP偏离 — 典型价VWAP锚(机构成本基准偏离)
    G_upvshare20 上涨日量能占比 — 吸筹/派发方向版
    G_volconf20  mom20 × corr(ret, vol) — 量确认动量(量价共振才可信)
  路径时间形态 (3)
    G_days_hi250 距最近52周新高天数(取负=新鲜度) — 时间维度版近高效应(George-Hwang补充)
    G_dist_hi20  距20日高点距离 — Donchian 突破邻近度
    G_chop20     20日方向翻转次数(取负) — 噪声行情度(choppiness)
  风险结构 (3)
    G_dnbeta60   下行beta - 上行beta — 不对称beta(Ang-Chen-Xing 2006 downside risk)
    G_coskew60   池内coskewness — 系统性偏度(Harvey-Siddique 2000)
    G_poolcorr60 与池等权指数60日相关 — 区别于beta(方向敏感度vs共动强度)
  季节性 (2, Heston-Sadka 2013式, 仅价格数据可算)
    G_seas_dow   同星期几的历史平均收益(过去48周)
    G_seas_mon   同日历月的历史平均日收益(过去约3年同月)
  组合 (3)
    G_consist_sharpe  (mom60/vol60) × 涨天占比 — 风险调整动量×路径一致性
    G_mom_cons    sign(mom121) × 涨天占比 — Grinblatt-Moskowitz 动量一致性
    G_lowbabi     rank(-beta60) + rank(alpha60) — 池内低beta×高alpha(Betting-Against-Beta×动量)

口径与 alpha_mining 完全一致(无前视):
  信号日 t 收盘已知信号值 -> t+1 开盘入场 -> 持有 h 日 -> t+1+h 开盘出场
  日级超额逐日聚合 -> Newey-West HAC(lag=h-1) -> BH-FDR -> 三池同号性+分半+分级
  (复用 alpha_mining.daily_excess / build_summary / classify_alpha, 结果与上轮可直接对比)

输出: research/output/g_mine_{run_id}/
  {pool}_h{h}_screen.csv / {pool}_validate.csv / summary.csv / report.txt

用法:
  python factor_mining2.py --pools etf_3x,company_300,company_1000
  python factor_mining2.py --pools etf_3x --horizons 20 --signals G_parkcc20,G_seas_dow
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel                                # noqa: E402
from signal_search import tradable_fwd, screen_one, validate_one      # noqa: E402
from factor_mining import build_mined                                 # noqa: E402
from conditional_eval import nw_tstat, bh_qvals                       # noqa: E402
from indicator_eval import sig_ac1                                    # noqa: E402
from alpha_mining import daily_excess, build_summary, classify_alpha  # noqa: E402

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ================================================================ 参数 ================================================================ #
_PKL_DIR = os.path.join(os.path.expanduser('~'), 'quant')   # 研究一律优先 research 全史 pkl
POOLS = {
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data_research.pkl'),
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data_research.pkl'),
    'company_1000': os.path.join(_PKL_DIR, 'company_1000_day_ta_data_research.pkl'),
}
START = '2021-01-01'
HORIZONS = [5, 20, 60]
TOP_K = 5
MIN_CS = 10
GROUPS = {  # 前缀 -> 分族
    'G_park': '波动微观结构', 'G_oviv': '波动微观结构', 'G_semi': '波动微观结构', 'G_ext': '波动微观结构',
    'G_ac1': '时序结构', 'G_vr': '时序结构', 'G_gap': '时序结构',
    'G_imp': '量能不对称', 'G_volb': '量能不对称', 'G_ami': '量能不对称', 'G_vwap': '量能不对称',
    'G_upv': '量能不对称', 'G_volc': '量能不对称',
    'G_days': '路径时间形态', 'G_dist': '路径时间形态', 'G_chop': '路径时间形态',
    'G_dn': '风险结构', 'G_cosk': '风险结构', 'G_pool': '风险结构',
    'G_seas': '季节性',
    'G_consist': '组合', 'G_mom_': '组合', 'G_low': '组合',
}


def group_of(name: str) -> str:
    for k, v in GROUPS.items():
        if name.startswith(k):
            return v
    return '其他'


# ================================================================ 候选构建 ================================================================ #
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


# ================================================================ 单池评估(与 alpha_mining.eval_pool 同口径, 候选注入) ================================================================ #
def eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
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
            rec.update({'signal': name, 'group': group_of(name),
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


# ================================================================ 主流程 ================================================================ #
def main():
    ap = argparse.ArgumentParser(description='第二轮新因子挖矿(G_族, 美股三池+FDR+分级)')
    ap.add_argument('--pools', default='etf_3x,company_300,company_1000')
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
                           f'g_mine_{run_id}')
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


if __name__ == '__main__':
    main()

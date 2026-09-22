# -*- coding: utf-8 -*-
"""
a_share_mining.py — A 股专属新因子挖掘(X_ 族, hs300 / a_etf_all, FDR + 分级)

定位: R/I/W 族(a_stat_mining)、G 族(factor_mining2)、H/N/C 族(alpha_mining)、
F 族(factor_mining)、K 族(conditional_mining) 已在 A 股两池跑完后, 补上**A 股
微观结构/交易制度专属**、上述各轮均未覆盖的候选维度:

  彩票/极值 (Bali-Cakici-Whitelaw 2011 MAX 效应; A 股散户彩票偏好尤强)
    X_max1_20   过去 20 日单日最大收益(MAX)
    X_q80_20    过去 20 日日收益 80 分位(上尾幅度)
    X_skew20    20 日日收益偏度(右偏 = 彩票性)
    X_kurt60    60 日日收益峰度(尖峰 = 跳跃性)

  流动性/量能 (Amihud 2002 + 换手情绪)
    X_illiq20   Amihud 非流动性**水平** mean(|ret|/成交额)(区别 G_amiasym20 只做涨跌不对称)
    X_relvol20  相对成交量 mean20(V)/mean250(V)(异常放量)
    X_pvcorr20  量价相关 corr20(close, V)(价升量增 vs 价升量缩)
    X_amt20     log 日均成交额(规模/关注度代理)
    X_turnstd20 成交量增速波动 std20(V.pct_change())(量能不稳)

  隔夜/日内微观结构 (A 股 T+1 制度特有; G_oviv20 只做 std 比)
    X_ovnrev20  隔夜收益反转 -mean20(open/prev_close-1)
    X_intramom20 日内动量 mean20(close/open-1)
    X_ovnshare60 隔夜方差占比 var60(OVN)/(var60(OVN)+var60(ID))

  涨跌停/极端制度 (A 股 ±10%/±20% 限制)
    X_limit20   过去 20 日 |日收益|>=9.5% 天数占比(触板频率)

  短期反转 (A 股最稳健的横截面异象之一; 已有 D_rev3 为 3 日)
    X_rev1 / X_rev5 / X_rev10   过去 1/5/10 日收益取负

  位置族 (George-Hwang 52 周高点; 区别 W_ 只做水下时间)
    X_hi52prox  close / max250(close)(距 52 周高点接近度)
    X_low52dist close / min250(close) - 1(距 52 周低点距离)

  波动不对称 / 效率
    X_updnvol20 std(涨日收益) / std(跌日收益)(杠杆效应不对称)
    X_er5       5 日效率比 |P_t/P_{t-5}-1| / Σ|日收益|(区别 F_er20 为 20 日)

口径与 a_stat_mining/alpha_mining 完全一致(无前视):
  信号日 t 收盘已知 -> t+1 开盘入场 -> 持有 h 日 -> t+1+h 开盘出场
  日级超额逐日聚合 -> Newey-West HAC(lag=h-1) -> BH-FDR(每池每 h) -> 分级

输出: research/output/x_mine_{run_id}/
  {pool}_h{h}_screen.csv / {pool}_validate.csv / summary.csv / report.txt

用法:
  python a_share_mining.py --pools hs300,a_etf_all
  python a_share_mining.py --pools hs300 --horizons 20 --signals X_max1_20
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
    'hs300':     os.path.join(_PKL_DIR, 'hs300_day_ta_data_research.pkl'),
    'a_etf_all': os.path.join(_PKL_DIR, 'a_etf_all_day_ta_data_research.pkl'),
}
START = '2021-01-01'
HORIZONS = [5, 20, 60]
TOP_K = 5
MIN_CS = 10
LIMIT_TH = 0.095          # 涨跌停近似阈值(主板 10%/双创 20%, 取主板口径保守)
GROUPS = {                # 前缀 -> 分族(注意 startswith 顺序, 更长前缀在前)
    'X_max': '彩票极值', 'X_q80': '彩票极值', 'X_skew': '彩票极值', 'X_kurt': '彩票极值',
    'X_illiq': '流动性量能', 'X_relvol': '流动性量能', 'X_pvcorr': '流动性量能',
    'X_amt': '流动性量能', 'X_turnstd': '流动性量能',
    'X_ovn': '隔夜日内', 'X_intra': '隔夜日内',
    'X_limit': '涨跌停极端',
    'X_rev': '短期反转',
    'X_hi52': '位置', 'X_low52': '位置',
    'X_updnvol': '波动不对称', 'X_er': '效率',
}


def group_of(name: str) -> str:
    for k, v in GROUPS.items():
        if name.startswith(k):
            return v
    return '其他'


# ================================================================ 候选构建 ================================================================ #
def build_x_cands(panel: pd.DataFrame) -> dict:
    """构建全部 X_ 候选(全历史构建, 因果安全; 评估窗截取在主流程做)."""
    def w(col):
        return pd.to_numeric(panel[col], errors='coerce').unstack('symbol').sort_index()

    close = w('Close')
    open_ = w('Open')
    volume = w('Volume')
    ret1 = close.pct_change()
    amount = (close * volume).replace(0, np.nan)
    overnight = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    out = {}

    # ---- 彩票/极值 ----
    out['X_max1_20'] = ret1.rolling(20, min_periods=10).max()
    out['X_q80_20'] = ret1.rolling(20, min_periods=10).quantile(0.80)
    out['X_skew20'] = ret1.rolling(20, min_periods=10).skew()
    out['X_kurt60'] = ret1.rolling(60, min_periods=30).kurt()

    # ---- 流动性/量能 ----
    ir = ret1.abs() / amount
    out['X_illiq20'] = ir.where(ret1.notna()).rolling(20, min_periods=10).mean()
    out['X_relvol20'] = volume.rolling(20, min_periods=10).mean() \
        / volume.rolling(250, min_periods=60).mean().replace(0, np.nan)
    out['X_pvcorr20'] = close.rolling(20, min_periods=10).corr(volume)
    out['X_amt20'] = np.log(amount.rolling(20, min_periods=10).mean())
    out['X_turnstd20'] = volume.pct_change().replace([np.inf, -np.inf], np.nan) \
        .rolling(20, min_periods=10).std()

    # ---- 隔夜/日内微观结构 ----
    out['X_ovnrev20'] = -overnight.rolling(20, min_periods=10).mean()
    out['X_intramom20'] = intraday.rolling(20, min_periods=10).mean()
    ov, iv = overnight.rolling(60, min_periods=30).var(), intraday.rolling(60, min_periods=30).var()
    out['X_ovnshare60'] = ov / (ov + iv).replace(0, np.nan)

    # ---- 涨跌停/极端 ----
    out['X_limit20'] = (ret1.abs() >= LIMIT_TH).where(ret1.notna()).astype(float) \
        .rolling(20, min_periods=10).mean()

    # ---- 短期反转 ----
    out['X_rev1'] = -ret1
    out['X_rev5'] = -close.pct_change(5)
    out['X_rev10'] = -close.pct_change(10)

    # ---- 位置族 ----
    hi250 = close.rolling(250, min_periods=60).max()
    lo250 = close.rolling(250, min_periods=60).min()
    out['X_hi52prox'] = close / hi250.replace(0, np.nan)
    out['X_low52dist'] = close / lo250.replace(0, np.nan) - 1.0

    # ---- 波动不对称 / 效率 ----
    up_std = ret1.where(ret1 > 0).rolling(20, min_periods=5).std()
    dn_std = ret1.where(ret1 < 0).rolling(20, min_periods=5).std()
    out['X_updnvol20'] = up_std / dn_std.replace(0, np.nan)
    den = ret1.abs().rolling(5, min_periods=3).sum()
    out['X_er5'] = close.pct_change(5).abs() / den.replace(0, np.nan)

    return {k: v for k, v in out.items()
            if isinstance(v, pd.DataFrame) and v.notna().sum().sum() > 0}


# ================================================================ 单池评估(与 a_stat_mining.eval_pool 同口径) ================================================================ #
def eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
    _, panel = load_panel(pkl, 'day')
    dts = panel.index.get_level_values('date')
    open_wide = panel['Open'].unstack('symbol').sort_index()
    cands = build_x_cands(panel)
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
    ap = argparse.ArgumentParser(description='A 股专属新因子挖矿(X_ 族, hs300/a_etf_all + FDR + 分级)')
    ap.add_argument('--pools', default='hs300,a_etf_all')
    ap.add_argument('--pkl', default=None, help='"池=路径" 覆盖默认数据源(可重复)')
    ap.add_argument('--start', default=START)
    ap.add_argument('--horizons', default=','.join(str(h) for h in HORIZONS))
    ap.add_argument('--signals', default=None, help='逗号分隔, 只评估指定候选')
    ap.add_argument('--tag', default=None, help='输出目录后缀')
    args = ap.parse_args()

    pools = [x.strip() for x in args.pools.split(',') if x.strip()]
    for kv in (args.pkl or '').split(';'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            POOLS[k.strip()] = v.strip()
    horizons = [int(x) for x in args.horizons.split(',')]

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                           f'x_mine_{run_id}' + (f'_{args.tag}' if args.tag else ''))
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

    lines = [f'a_share_mining 报告(X_ 族) | pools={list(longs)} | run={run_id}',
             f'评估窗 {args.start}~, horizons={horizons}, top_k={TOP_K}, min_cs={MIN_CS}',
             '口径: 信号日收盘 -> 次日开盘入场 -> 持有h日开盘出场; '
             'NW-HAC(lag=h-1) on 日级超额; BH-FDR 族=每池每h全部候选; 分级=两池同号+FDR+分半+动态增量',
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

# -*- coding: utf-8 -*-
"""
a_stat_mining.py — 独立只读研究模块: 附录 A 类统计维度挖矿(R_/I_/W_ 族, 美股三池 + FDR + 分级)

定位: B 类(条件层/组合层)完成后, 按附录优先级执行 A 类——现有 OHLCV 日线内
四轮尚未覆盖的统计维度, 文献最扎实、实现成本最低的三类:

  残差动量 (4, Blitz-Huij-Martens 2007)
    R_resmom121  残差 12-1 动量(过去 252 日剔除最近 21 日, 剥离池 beta 后的特质动量)
    R_resmom60   残差 60 日动量(短窗版)
    R_resmom20   残差 20 日动量(反转区探测)
    R_resvol121  残差动量的波动调整版(除以残差 252 日波动)
    残差: 对池等权收益滚动 OLS(60d) 的中心化残差——区别于 F_alpha60(原始相对强度,
    未剥离 beta 后取动量)与 N_idiovol60(只有残差波动、无动量)

  信息离散度 (4, Da-Gurun-Warachka 2014「温水煮青蛙」)
    I_id20/60/121  ID = sign(mom_q) x (跌天占比 - 涨天占比): 路径剧烈(高离散)的动量
                   信息不连续 → 衰减快; 连续信息(低离散)的动量更持续
    I_ivshare60    特质波动/总波动(1-R^2 近似)——Da 文 IVOL 代理(离散度另一度量)
    区别于 X_upday20(只有涨天占比, 未与动量符号交互)

  回撤路径时间 (4, 附录「深度之外的时间维度」)
    W_ddshare250   过去 250 日水下天数占比(累计时间维度, 非当前状态)
    W_ddtime121    窗口内最长单次连续水下段长度(rowno-ffill run-length, G_days_hi250 同技巧)
    W_martin60     Martin ratio = 均值收益 / Ulcer(回撤深度平方根)——回撤版 Sharpe
    W_hihits60     60 日内创新高(250 日高点)次数——恢复能力/新新鲜度频率
    区别于 F_ulcer60/F_dd60(只有深度, 无持续时间)与 G_days_hi250(距新高距离, 无频率)

口径与 factor_mining2/alpha_mining 完全一致(无前视):
  信号日 t 收盘已知信号值 -> t+1 开盘入场 -> 持有 h 日 -> t+1+h 开盘出场
  日级超额逐日聚合 -> Newey-West HAC(lag=h-1) -> BH-FDR -> 三池同号性+分半+分级

输出: research/output/a_mine_{run_id}/
  {pool}_h{h}_screen.csv / {pool}_validate.csv / summary.csv / report.txt

用法:
  python a_stat_mining.py --pools etf_3x,company_300,company_1000
  python a_stat_mining.py --pools etf_3x --horizons 20 --signals I_id60
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
    'etf_3x':      os.path.join(_PKL_DIR, 'etf_3x_day_ta_data_research.pkl'),
    'company_300': os.path.join(_PKL_DIR, 'company_300_day_ta_data_research.pkl'),
    'company_1000': os.path.join(_PKL_DIR, 'company_1000_day_ta_data_research.pkl'),
}
START = '2021-01-01'
HORIZONS = [5, 20, 60]
TOP_K = 5
MIN_CS = 10
GROUPS = {  # 前缀 -> 分族
    'R_res': '残差动量',
    'I_id': '信息离散度', 'I_iv': '信息离散度',
    'W_dd': '回撤时间', 'W_martin': '回撤时间', 'W_hi': '回撤时间',
}


def group_of(name: str) -> str:
    for k, v in GROUPS.items():
        if name.startswith(k):
            return v
    return '其他'


# ================================================================ 候选构建 ================================================================ #
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


# ================================================================ 单池评估(与 factor_mining2.eval_pool 同口径, 候选注入) ================================================================ #
def eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
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
    ap = argparse.ArgumentParser(description='附录 A 类统计维度挖矿(R_/I_/W_族, 美股三池+FDR+分级)')
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
                           f'a_mine_{run_id}')
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


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
conditional_mining.py — 独立只读研究模块: 第五轮条件/门控层因子挖矿(K_ 族, 美股三池 + FDR + 分级)

定位(research_summary_20260920.md 附录 B 类第 1/2 条): 四轮裸因子挖掘边际递减(均无 A/B 级)后,
转向「条件/门控」构造——单因子弱不等于条件化后弱。K_ = Konditional(条件层)。
与 conditional_eval.py 的区别: 那是 pkl 自带 s/m_trend 列的「事件×状态」收益矩阵工具;
本轮是**因子级条件交互挖矿**, 产物是新因子列, 走与四轮完全相同的评估分级, 结果直接可比。

候选分族(19 个 K_ + 2 个裸动量锚):
  个股门控 (8)  个股质量状态调制动量的截面权重
    K_mom_lowvol   mom60 × (1−nc(σ20))      低波门控动量(低噪声环境动量更可信)
    K_mom_highvol  mom60 × nc(σ20)          高波门控动量(对照, 检验门控方向)
    K_mom_lquiet   mom60 × (1−nc(量比20))   低量比门控动量(Nagel: 低换手动量更持续)
    K_mom_hquiet   mom60 × nc(量比20)       高量比门控动量(对照)
    K_mom_er       mom60 × F_er20           趋势效率门控动量(ER 低 → 动量权重清零)
    K_mom_liq      mom60 × (1−nc(Amihud20)) 高流动性门控动量(退出成本可承受)
    K_mom_illiq    mom60 × nc(Amihud20)     低流动性门控动量(对照, 预期暴露流动性伪影)
    K_121_lowvol   mom121 × (1−nc(σ20))     12-1 动量低波版(检验能否打破 F_mom121 的 S 静态身份)
  池 regime 门控 (4)  ±1 时序门控: 正 gate 期排序=裸动量, 负 gate 期排序反转(无并列退化)
    K_mom_poolsgn  mom60 × sign(池mom20)    顺池动量/逆池反弹(池下行期选弱动量)
    K_mom_poolvolq mom60 × (1−2·池σ20分位)  低池波期动量/高池波期反转(Daniel-Moskowitz 动量崩溃防御)
    K_mom_pooltr   mom60 × sign(池价/MA60)  池趋势门控
    K_121_poolsgn  mom121 × sign(池mom60)   12-1 动量池趋势版
  多周期共振 (5)  连续强度 × sign 一致性权重
    K_reso3        mom60 × 三周期(5/20/60)sign 一致数/3
    K_reso2        mom60 × (2·(sign20=sign60)−1)
    K_reso_l5      mom20 × (2·(sign5=sign250)−1)   长趋势内短期同向
    K_reso_mag     mom60 × (2·(sign60=sign250)−1)  期限结构共振
    K_reso_cnt     mom60 × 4 周期对 sign20 的对齐数中心化
  强度×质量交叉 (2)
    K_tmq_z        z(mom121) × z(er20)      标准化乘积(vs C_tmqmom 的 rank 加法)
    K_tmqr         rank(mom121) × rank(er20) 秩乘积(乘法门控版质量动量)
  裸动量锚 (2)  同口径对照, 门控因子与其的差异 = 条件化的边际贡献
    F_mom60 / F_mom121

口径与 factor_mining2/alpha_mining 完全一致(无前视):
  信号日 t 收盘已知信号值 -> t+1 开盘入场 -> 持有 h 日 -> t+1+h 开盘出场
  日级超额 -> NW-HAC(lag=h-1) -> BH-FDR -> 三池同号+分半+动态增量+classify_alpha 分级

输出: research/output/k_mine_{run_id}/
  {pool}_h{h}_screen.csv / {pool}_validate.csv / summary.csv / report.txt

用法:
  python conditional_mining.py --pools etf_3x,company_300,company_1000
  python conditional_mining.py --pools etf_3x --horizons 20 --signals K_mom_er
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel, normalize_causal              # noqa: E402
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
GROUPS = {  # 前缀 -> 分族(注意 startswith 匹配顺序: 更长前缀在前)
    'K_mom_pool': '池regime门控', 'K_121_pool': '池regime门控',
    'K_mom_': '个股门控', 'K_121_': '个股门控',
    'K_reso': '多周期共振', 'K_tmq': '强度×质量交叉',
    'F_mom': '裸动量锚',
}


def group_of(name: str) -> str:
    for k, v in GROUPS.items():
        if name.startswith(k):
            return v
    return '其他'


# ================================================================ 候选构建 ================================================================ #
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


# ================================================================ 单池评估(与 factor_mining2.eval_pool 同口径, 候选注入) ================================================================ #
def eval_pool(pool: str, pkl: str, out_dir: str, horizons, start) -> pd.DataFrame:
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
    ap = argparse.ArgumentParser(description='第五轮条件/门控层因子挖矿(K_族, 美股三池+FDR+分级)')
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
                           f'k_mine_{run_id}')
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


if __name__ == '__main__':
    main()

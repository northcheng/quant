# -*- coding: utf-8 -*-
"""
factor_mining.py — 独立只读研究模块: 新因子挖矿(多 horizon IC 筛选)

定位: 在 signal_search.build_derived 的 23 个 D_ 经典因子之外, 挖一批结构性更强的候选因子,
集中在 D_ 族未覆盖的维度:
  A. 趋势效率/质量   — 效率比(ER)、回归 R²、斜率 t 值(区别于点动量: 度量路径质量而非端点涨幅)
  B. 隔夜/日内结构   — 隔夜收益 vs 日内收益的分解(美股盘中/隔夜微观结构差异的经典来源)
  C. 波动结构        — 波动期限结构、下行波动占比、肥尾、波动不稳、回撤深度、Ulcer、日内振幅
  D. 流动性/量能     — Amihud 冲击成本、量价相关性、归一化 OBV 流、上涨日量能占比、量能稳定性
  E. 价格路径形态    — 52 周区间位置、K 线实体占比、上下影线压力、连续同向天数
  F. 相对池强度      — 相对池等权指数的 alpha/beta/特异波动(池内相对强弱)
  G. 动量变体        — 12-1 动量(剥离短期反转)、动量加速、回撤调整动量

口径与 signal_search 完全一致(可交易口径, 无前视):
  信号日 t 收盘已知信号值 -> t+1 开盘入场 -> 持有 h 日 -> t+1+h 开盘出场
  复用 signal_search 的 screen_one / validate_one, 保证与既有 D_ 因子筛查结果可比.

输出: research/output/{pool}_mine_{run_id}/mine_h{h}.csv + validate_h{h}.csv + report.txt

用法:
  python factor_mining.py --pool etf_3x --pkl-path research/data/etf_3x_day_ta_data.pkl --start 2021-01-01
  python factor_mining.py --pool company_300 --pkl-path C:/Users/northcheng/quant/company_300_day_ta_data_research.pkl
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
from signal_search import tradable_fwd, screen_one, validate_one

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

EPS = 1e-12


# ================================================================ 挖矿因子库 ================================================================ #
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
    num = n * Sxy - Sx * Sy
    den = np.sqrt((n * Sxx - Sx ** 2) * (n * Syy - Sy ** 2))
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


# ================================================================ 主流程 ================================================================ #
def main():
    ap = argparse.ArgumentParser(description='独立只读研究: 新因子挖矿(多 horizon 可交易口径筛查)')
    ap.add_argument('--pool', default='etf_3x')
    ap.add_argument('--interval', default='day')
    ap.add_argument('--pkl-dir', default=r'C:\Users\northcheng\quant')
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
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
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


if __name__ == '__main__':
    main()

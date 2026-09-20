# -*- coding: utf-8 -*-
"""
factor_signal.py — 独立因子信号脚本(黄金组合因子 composite + 每日 top-K 快照)

定位: 完全独立运行, 不修改现有系统任何文件(bc_technical_analysis.py /
      technical_analyst_parallel.py / ta_config.json 均不动, 也不 import quant 包)。
      直接读取生产管道产出的 {pool}_{interval}_ta_data.pkl, 计算 factor_mining
      黄金组合因子, 截面 rank pct 加权合成 composite, 打印最新截面快照与 top-K 候选。

数据流:
  生产 pkl(~/quant/{pool}_{interval}_ta_data.pkl, technical_analyst_parallel.py 产出)
    -> load_panel -> build_mined(38 因子, 无前视) -> 取权重涉及因子
    -> 截面 rank pct 加权 composite -> 控制台快照(+ 可选 --out-csv 历史长表)

口径:
  - 因子: research/factor_mining.py build_mined 逐行同源, 仅用 t 及之前数据
  - composite: score_backtest.compute_composite 同口径 — 当日截面 rank pct(0~1),
    权重按绝对值和归一, 缺失(NaN)以中性 0.5 参与; 排名 method='first' 与 run_engine 一致
  - zone 分区仅基于当日截面(top_k / exit_rank), 不含持仓状态;
    run_engine 的滞回进出(已持有者 rank<=exit_rank 不动)与此不完全等价, 仅作参考
  - 决策时点: 信号日 s 收盘计算, 执行日 s+1 开盘(与回测引擎一致)

用法:
  python research/factor_signal.py                          # etf_3x, 生产 pkl, 最新日快照
  python research/factor_signal.py --weights-preset gold3   # 3 因子版(201620 回测口径)
  python research/factor_signal.py --as-of 2026-09-01 --recent 5
  python research/factor_signal.py --weights "{\"F_mom121\": 0.5, \"F_er20\": 0.3}"
  python research/factor_signal.py --out-csv research/data/etf_3x_composite.csv

零副作用: 默认只打印; 仅显式 --out-csv 时写一个文件。
warmup 说明: F_mom121 需 252 日, 其生效前(约前 1 年)composite 仅由已生效的部分因子
             参与计算(F_er20/F_obv20 窗口更短先生效), 截面信息不完整, 仅参考。
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel
from factor_mining import build_mined

try:  # Windows 控制台避免 UnicodeEncodeError
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 黄金组合权重预设(绝对值和为 1; 负号 = 反向使用, 买截面低值者)
WEIGHT_PRESETS = {
    # 4 因子版(默认交付口径)
    'gold4': {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2, 'F_obv20': 0.1},
    # 3 因子版(etf_3x 2021-2026 回测报告 201620 口径: total_ret 6.30 / sharpe 0.98)
    'gold3': {'F_mom121': 0.5, 'F_er20': 0.3, 'F_idiovol60': -0.2},
}


# ================================================================ 组合分数 ================================================================ #
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


# ================================================================ 主流程 ================================================================ #
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


def main():
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


if __name__ == '__main__':
    main()

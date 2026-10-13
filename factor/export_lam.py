# -*- coding: utf-8 -*-
"""
factor.export_lam — 把 PEAD 的倾斜强度 λ 升级为 **K 折 OOS 搜索**
================================================================================
背景(折中方案):
  生产 pathA 口径 = bc + λ·(PEAD_60 - 0.5), 其中
    bc   = BASE8 等权后逐日去均值(align/8 加权);
    λ    = PEAD 倾斜强度. 当前 0.75 来自一维手扫(--grid 扫 + 四窗净夏普不降门槛),
           非拟合, 阈值本身是人为定的.
  本脚本只把 λ 的确定方式换成与 export_cd_ic **同口径的 K 折 OOS**:
    - BASE8 固定等权 align/8, 不动 cd_ic 联合优化;
    - 只在 LAM_GRID 上搜 λ, 目标 = 6 折等分(仅 IS 内)生产引擎折外夏普均值;
    - 折外目标所用引擎取该池生产参数(top_k/exit_rank 读 factor_config.json).

权重编码: combine_signals 按 Σ|w| 归一化, 故 w = {BASE8: align/8, PEAD: align·λ}
  与 pathA 的横截面排序等价(逐日去均值与常数位移都不改排序), 可直接写入
  factor_config.json 的 schemes.cd_ic(PEAD 槽 = λ, BASE8 槽 = ±0.125).

自检:
  cd ~/git && python -u -m quant.factor.export_lam --pool us_500
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor import evaluate as ev
from quant.factor import export_cd_ic as eci
from quant.factor import prepare as prep
from quant.factor import repro_old_presets as repro
from quant.factor.data import load_pool

BASE8 = ['mom_12_1', 'C_tmqmom', 'G_oviv20', 'obv_20',
         'N_idiovol60', 'er_20', 'bias_20', 'H_ichimoku_alpha']
NEW = 'PEAD_60'
NF = 6                                            # 折外目标折数(与 export_cd_ic 一致)
CUR_LAM = 0.75                                    # 当前生产 λ(手扫结果), 作对照
LAM_GRID = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 3.0]

OUT = Path(__file__).resolve().parent / 'output' / 'pead'


def _weights(align: dict, lam: float) -> dict:
  """pathA 权重编码: BASE8 等权 align/8 + PEAD 倾斜 align·λ(λ=0 时 PEAD 自动被剔除)."""
  w = {n: round(align[n] / len(BASE8), 6) for n in BASE8}
  w[NEW] = round(align[NEW] * float(lam), 6)
  return w


def run(pool: str) -> dict:
  ds = load_pool(pool)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  names = BASE8 + [NEW]
  sigs = cmb.build_signals(ds, prep.DEFAULT_PREP, names, cfg.MIN_CS)
  p = eci.prod_engine(pool)
  mf = eci.prod_min_factors(pool)

  fwd = ev.forward_return(ds, h=1)
  is_s, is_e = cfg.resolve_windows(pool)['is']
  fwd_is = bt._win(fwd, is_s, is_e)
  scores = cmb.rank_ic_scores(sigs, fwd_is, names=names, min_cs=cfg.MIN_CS)
  align = {n: (1.0 if (pd.notna(scores.get(n)) and scores[n] >= 0) else -1.0)
           for n in names}

  folds = cmb.split_folds(ds.dates, is_s, is_e, NF)
  obj = bt.cached_fold_objective(open_w, close_w, p, folds, metric='sharpe', agg='mean')

  rows = []
  for lam in LAM_GRID:
    w = _weights(align, lam)
    comp = cmb.combine_signals(sigs, w, cfg.MIN_CS, mf)
    fo = float(obj(w, comp))
    rec = {'lam': lam, 'fold_obj': round(fo, 6)}
    for win, ws, we in repro.windows_for(pool):
      res = bt.run_engine(bt._win(open_w, ws, we), bt._win(close_w, ws, we),
                          bt._win(comp, ws, we), p)
      rec[win] = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])['sharpe']
    rows.append(rec)
  sweep = pd.DataFrame(rows)

  best = sweep.loc[sweep['fold_obj'].idxmax()]
  lam_star = float(best['lam'])
  return {'pool': pool, 'engine': f'top_k={p.top_k},exit_rank={p.exit_rank}',
          'align': align, 'scores': {n: round(float(v), 4) for n, v in scores.items()},
          'lam_star': lam_star, 'sweep': sweep,
          'weights_star': _weights(align, lam_star)}


def main():
  ap = argparse.ArgumentParser(description='PEAD 倾斜强度 λ 的 K 折 OOS 搜索')
  ap.add_argument('--pool', required=True)
  a = ap.parse_args()
  r = run(a.pool)

  OUT.mkdir(parents=True, exist_ok=True)
  csv = OUT / f'lam_search_{r["pool"]}.csv'
  r['sweep'].to_csv(csv, index=False, encoding='utf-8-sig')

  print(f"=== {r['pool']}  engine[{r['engine']}]  折数={NF}  目标=折外夏普均值 ===")
  print(f"  IS align : {json.dumps(r['align'], ensure_ascii=False)}")
  print(f"  IS RankIC: {json.dumps(r['scores'], ensure_ascii=False)}")
  print(f"\n{r['sweep'].to_string(index=False)}")
  print(f"\n  λ* (折外夏普最优) = {r['lam_star']}")
  print(f"  当前手扫 λ          = {CUR_LAM}")
  print(f"  λ* 权重(可写入 config schemes.cd_ic): "
        f"{json.dumps(r['weights_star'], ensure_ascii=False)}")
  print(f"  扫描明细 CSV        -> {csv}")


if __name__ == '__main__':
  main()
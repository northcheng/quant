# -*- coding: utf-8 -*-
"""
factor.wf_step4_export_weights — Step4 **临时脚本**: 导出 cd_ic+6段 的逐因子权重
================================================================================
Step3 结论: 折外目标口径取 **6 折等分 + agg=mean** 时, cd_ic(初值 = IS 期 IC 加权
  @ shrink=0.5, 去重 thresh=0.7, 坐标下降 3 轮)在缺口与全窗上同时最优.

本脚本把该规则在 5 个池上各重跑一遍, 打印**逐因子权重**(供生产写入 factor_config),
  并顺带复核 IS/OOS/FULL/YTD26 的 sharpe 与 fold_obj, 确认与 Step3 报表一致.

这是**临时脚本**: 只做一次性导出. 通用能力(ic 收缩 / 去重 / 坐标下降 / 折外目标)
  已落在 combine.py / backtest.py.

自检:
  cd ~/git && python -u -m quant.factor.wf_step4_export_weights
  cd ~/git && python -u -m quant.factor.wf_step4_export_weights --out out.json
"""
import argparse
import json

from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor import evaluate as ev
from quant.factor import repro_old_presets as repro
from quant.factor.data import load_pool

BASE_SHRINK = 0.5           # cd_ic 初值的 IC 收缩比例(与 Step3 冻结口径一致)
BASE_THRESH = 0.7           # 去重冗余阈值
NF = 6                      # 折外目标折数
AGG = 'mean'                # 折外目标聚合方式


def _ic_weights(sigs, fwd_is, names, shrink: float) -> dict:
  """ic 加权 + shrink 收缩(临时脚本内自实现, 因旧预设名不在 factor 注册表)."""
  w = cmb.ic_weights(sigs, fwd_is, names=names, shrink=0.0)
  n = len(w)
  if n == 0:
    return {}
  eq = 1.0 / n
  return {k: round((1.0 - shrink) * v + shrink * eq, 6) for k, v in w.items()}


def run_pool(pool: str, cd_passes: int = 3) -> dict:
  ds = load_pool(pool)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  sigs = repro.native_signals(ds)
  preset_w, preset_k = repro.PRESETS[pool]
  p = repro.engine_params(preset_k)
  fwd = ev.forward_return(ds, h=1)
  fwd_is = bt._win(fwd, cfg.TRADE_START, cfg.IS_END)

  scores = cmb.rank_ic_scores(sigs, fwd_is, min_cs=cfg.MIN_CS)
  kept, dropped = cmb.dedupe(sigs, scores, thresh=BASE_THRESH, min_cs=cfg.MIN_CS)
  cands = kept or list(sigs.keys())
  w0 = _ic_weights(sigs, fwd_is, cands, BASE_SHRINK)

  folds = cmb.split_folds(ds.dates, cfg.TRADE_START, cfg.IS_END, NF)
  obj = bt.make_fold_objective(open_w, close_w, p, folds, metric='sharpe', agg=AGG)
  w_cd, hist = cmb.coord_descent_weights(sigs, w0, obj, min_cs=cfg.MIN_CS,
                                         max_passes=cd_passes)

  comp = cmb.combine_signals(sigs, w_cd, min_cs=cfg.MIN_CS)
  stats = {}
  for win, ws, we in repro.WINDOWS:
    ow, cw = bt._win(open_w, ws, we), bt._win(close_w, ws, we)
    res = bt.run_engine(ow, cw, bt._win(comp, ws, we), p)
    s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    stats[win] = round(float(s['sharpe']), 4)

  return {'pool': pool, 'top_k': preset_k, 'n_factors': len(w_cd),
          'weights': w_cd, 'w0': w0, 'cands': cands,
          'dropped': [d['factor'] for d in dropped],
          'scores_is': {k: round(float(v), 4) for k, v in scores.items()},
          'stats': stats, 'fold_obj': round(float(obj(w_cd, comp)), 4),
          'n_cd_steps': int(len(hist))}


def main():
  ap = argparse.ArgumentParser(description='Step4 临时脚本: 导出 cd_ic+6段 逐因子权重')
  ap.add_argument('--pools', default=','.join(repro.PRESETS.keys()))
  ap.add_argument('--cd-passes', type=int, default=3)
  ap.add_argument('--out', default=None, help='可选: 权重 JSON 落盘路径')
  a = ap.parse_args()

  pools = [s.strip() for s in a.pools.split(',') if s.strip()]
  res = []
  for pool in pools:
    if pool not in repro.PRESETS:
      continue
    r = run_pool(pool, a.cd_passes)
    res.append(r)
    print(f"\n================ {pool}  (top_k={r['top_k']}, "
          f"{r['n_factors']} 因子, cd {r['n_cd_steps']} 步, "
          f"fold_obj={r['fold_obj']}) ================")
    print(f"  去重丢弃: {r['dropped'] or '无'}")
    print(f"  候选集  : {r['cands']}")
    print(f"  IS 期 IC : {r['scores_is']}")
    print(f"  初值 w0  : {r['w0']}")
    print(f"  最终权重 : {json.dumps(r['weights'], ensure_ascii=False)}")
    print(f"  窗口 sharpe: {r['stats']}")
    # 与权重同尺度: combine_signals 内部按 sum|w| 归一, 故这里给出归一化版本备查
    tot = sum(abs(v) for v in r['weights'].values())
    print(f"  归一化(+Σ|w|=1): "
          f"{json.dumps({k: round(v / tot, 6) for k, v in r['weights'].items()}, ensure_ascii=False)}")

  if a.out:
    with open(a.out, 'w', encoding='utf-8') as f:
      json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {a.out}")


if __name__ == '__main__':
  main()

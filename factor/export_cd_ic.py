# -*- coding: utf-8 -*-
"""
factor.export_cd_ic — 按**生产引擎口径**为指定池导出 cd_ic 逐因子权重
================================================================================
口径(与三池历史 cd_ic 一致的 recipe):
  候选 = repro.native_signals 的 10 个信号 -> IS 期 rank_ic -> 去重 thresh=0.7
  -> 初值 = IS 期 IC 加权 @ shrink=0.5 -> 折外目标(6 折等分, agg=mean, metric=sharpe)
  -> 坐标下降 max_passes=3, min_cs=8

与 wf_step4 的唯一差别: 折外目标所用引擎取**该池生产参数**
  (top_k/exit_rank 读 factor_config.json, sizing='equal', cap=0.30, cost/band 同全局),
  而非 repro 旧预设(top_k=12/sizing='tier').

输出: 权重键已映射到注册表名, 可直接粘进 factor_config.json 的 schemes.cd_ic.

自检:
  cd ~/git && python -u -m quant.factor.export_cd_ic --pool company_300
"""
import argparse
import json

from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor import evaluate as ev
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor import repro_old_presets as repro
from quant.factor.data import load_pool

# 旧预设名 -> 注册表名(注册表内已有等价因子; 其余 C_/N_/G_/H_ 同名直接复用)
RENAME = {'F_mom121': 'mom_12_1', 'D_ma20_dist': 'bias_20',
          'F_obv20': 'obv_20', 'F_er20': 'er_20'}

SHRINK = 0.5      # cd_ic 初值的 IC 收缩比例
THRESH = 0.7      # 去重冗余阈值
NF = 6            # 折外目标折数
AGG = 'mean'      # 折外目标聚合方式
PASSES = 3        # 坐标下降轮数


def _ic_weights(sigs, fwd_is, names, shrink: float) -> dict:
  w = cmb.ic_weights(sigs, fwd_is, names=names, shrink=0.0)
  n = len(w)
  if n == 0:
    return {}
  eq = 1.0 / n
  return {k: (1.0 - shrink) * v + shrink * eq for k, v in w.items()}


def prod_engine(pool: str) -> bt.EngineParams:
  conf = json.load(open(cfg.FACTOR_CONFIG, encoding='utf-8'))
  pr = conf['pools'][pool].get('params', {})
  return bt.EngineParams(int(pr.get('top_k', cfg.TOP_K)),
                         int(pr.get('exit_rank', cfg.EXIT_RANK)),
                         'equal', 1.0, 0.30, cfg.COST_BPS, 0.05)


def prod_min_factors(pool: str) -> int:
  """读该池生产 params.min_factors: 定权掩码必须与生产端应用的一致."""
  conf = json.load(open(cfg.FACTOR_CONFIG, encoding='utf-8'))
  return int(conf['pools'][pool].get('params', {}).get('min_factors', cfg.MIN_FACTORS))


def run(pool: str, extra: list = None) -> dict:
  ds = load_pool(pool)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  sigs = repro.native_signals(ds)
  if extra:                                    # 额外并入候选集(如 PEAD_60), 与 native 同 rank 口径
    pf = prep._make_prep(prep.DEFAULT_PREP, cfg.MIN_CS)
    for n in extra:
      sigs[n] = pf(fct.get_factor(n).compute(ds))
  p = prod_engine(pool)
  mf = prod_min_factors(pool)
  fwd = ev.forward_return(ds, h=1)
  is_s, is_e = cfg.resolve_windows(pool)['is']
  fwd_is = bt._win(fwd, is_s, is_e)

  scores = cmb.rank_ic_scores(sigs, fwd_is, min_cs=cfg.MIN_CS)
  kept, dropped = cmb.dedupe(sigs, scores, thresh=THRESH, min_cs=cfg.MIN_CS)
  cands = kept or list(sigs.keys())
  w0 = _ic_weights(sigs, fwd_is, cands, SHRINK)

  folds = cmb.split_folds(ds.dates, is_s, is_e, NF)
  obj = bt.cached_fold_objective(open_w, close_w, p, folds, metric='sharpe', agg=AGG)
  w_cd, hist = cmb.coord_descent_weights(sigs, w0, obj, min_cs=cfg.MIN_CS,
                                         max_passes=PASSES, min_factors=mf)

  comp = cmb.combine_signals(sigs, w_cd, min_cs=cfg.MIN_CS, min_factors=mf)
  stats = {}
  for win, ws, we in repro.windows_for(pool):
    res = bt.run_engine(bt._win(open_w, ws, we), bt._win(close_w, ws, we),
                        bt._win(comp, ws, we), p)
    s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
    stats[win] = {'sharpe': s['sharpe'], 'total_ret': s['total_ret'],
                  'turnover': s['ann_turnover']}
  return {'pool': pool, 'engine': f'top_k={p.top_k},exit_rank={p.exit_rank},'
          f'sizing={p.sizing}', 'cands': cands,
          'dropped': [d['factor'] for d in dropped], 'w0': w0,
          'weights_raw': w_cd,
          'weights_registry': {RENAME.get(k, k): round(float(v), 6)
                               for k, v in w_cd.items()},
          'stats': stats, 'fold_obj': round(float(obj(w_cd, comp)), 4),
          'n_cd_steps': int(len(hist))}


def main():
  ap = argparse.ArgumentParser(description='按生产引擎口径导出 cd_ic 权重')
  ap.add_argument('--pool', required=True)
  ap.add_argument('--extra', nargs='*', default=None,
                  help='额外并入候选集的注册表因子(同 rank 口径), 如 PEAD_60')
  a = ap.parse_args()
  r = run(a.pool, a.extra)
  print(f"=== {r['pool']}  engine[{r['engine']}]  {len(r['weights_registry'])} 因子 "
        f"cd {r['n_cd_steps']} 步 fold_obj={r['fold_obj']} ===")
  print(f"  候选集  : {r['cands']}")
  print(f"  去重丢弃: {r['dropped'] or '无'}")
  print(f"  初值 w0 : {json.dumps(r['w0'], ensure_ascii=False)}")
  print(f"  注册表名权重(可直接写入 config): "
        f"{json.dumps(r['weights_registry'], ensure_ascii=False)}")
  print(f"  窗口绩效: {json.dumps(r['stats'], ensure_ascii=False)}")


if __name__ == '__main__':
  main()
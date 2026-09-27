# -*- coding: utf-8 -*-
"""
factor.repro_old_presets — 用当前系统引擎复现旧系统 5 组预设权重(同权重对照)
==========================================================================
目的: 把旧系统(signal_bridge / bc_backtest / score_backtest)在旧报告
  research_2026-09/report/signal_bridge_pool_performance_20260923.md §8 里的 5 组预设,
  原样搬到当前 factor/ 引擎上重跑, 做一次**真正的同权重对照**.

对照逻辑:
  - old    : 旧报告 §8 的原始数值(基准, 逐字抄录).
  - legacy : 旧因子公式 + 旧 composite 口径(rank pct 加权 + fillna(0.5)) + **当前引擎**.
             -> 与 old 的差 = 纯"引擎/数据"差异(信号本身一致).
  - native : 当前系统口径(注册表因子 + prep('rank') + combine_signals) + 当前引擎, 权重不变.
             -> 与 legacy 的差 = 当前"因子/预处理/组合"层相对旧链路的差异.

口径(与旧报告 §1 一致):
  信号日 t 收盘决策 -> 执行日 t+1 开盘成交(open-to-open); sizing='tier'; exit_rank=12;
  max_exposure=1.0; per_symbol_cap=0.30; 单边成本 10bps; band=0.05.

自检:
  cd ~/git && python -m quant.factor.repro_old_presets
  cd ~/git && python -m quant.factor.repro_old_presets --pools etf_3x --variant legacy
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import backtest as bt
from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor import factor as fct
from quant.factor import prepare as prep
from quant.factor.data import load_pool

# ---------------------------- 旧系统 5 组预设(逐字照抄 signal_bridge.py L84-L96) ----------------------------
PRESETS = {
    'company_300':  ({'C_tmqmom': 1.0, 'F_er20': 1.0}, 12),
    'company_1000': ({'G_vr20': 1.0, 'H_ichimoku_alpha': 1.0}, 8),
    'etf_3x':       ({'F_mom121': 0.5, 'F_er20': 0.3, 'N_idiovol60': 0.2, 'F_obv20': 0.1}, 5),
    'hs300':        ({'N_range20': -1.0, 'D_ma20_dist': -1.0, 'C_tmqmom': 0.5}, 5),
    'a_etf_all':    ({'N_range20': 1.0, 'G_oviv20': 0.5}, 5),
}
EXIT_RANK = 12
COST_BPS = 10.0
BAND = 0.05
CAP = 0.30

# 旧报告 §8 基准数值(逐字抄录); 列序见 OLD_COLS
OLD_COLS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'vol',
            'ann_turnover', 'n_trades', 'win_rate', 'avg_days', 'n_days']
OLD_REPORT = {
    ('company_300', 'full'):  (21.760, 0.731, 1.34, -0.342, 2.14, 0.504, 161.6, 5086, 0.501, 3.4, 1431),
    ('company_300', 'is'):    (7.177, 0.694, 1.37, -0.325, 2.13, 0.461, 160.5, 3545, 0.500, 3.4, 1004),
    ('company_300', 'oos'):   (1.649, 0.773, 1.27, -0.342, 2.26, 0.596, 164.8, 1535, 0.504, 3.3, 426),
    ('company_300', 'ytd26'): (0.234, 0.353, 0.83, -0.287, 1.23, 0.542, 170.4, 641, 0.482, 3.2, 176),
    ('company_1000', 'full'):  (44.638, 0.955, 1.39, -0.481, 1.98, 0.609, 99.3, 1670, 0.519, 6.8, 1433),
    ('company_1000', 'is'):    (10.984, 0.865, 1.32, -0.481, 1.80, 0.601, 97.1, 1139, 0.513, 7.0, 1004),
    ('company_1000', 'oos'):   (2.571, 1.109, 1.50, -0.418, 2.65, 0.627, 104.5, 526, 0.532, 6.5, 428),
    ('company_1000', 'ytd26'): (0.605, 0.964, 2.22, -0.197, 4.88, 0.328, 102.9, 216, 0.565, 6.5, 178),
    ('etf_3x', 'full'):  (5.565, 0.391, 0.95, -0.425, 0.92, 0.458, 71.6, 336, 0.527, 21.0, 1432),
    ('etf_3x', 'is'):    (1.289, 0.231, 0.73, -0.425, 0.54, 0.387, 70.2, 221, 0.502, 21.1, 1004),
    ('etf_3x', 'oos'):   (1.869, 0.857, 1.35, -0.389, 2.21, 0.595, 76.0, 115, 0.583, 17.6, 427),
    ('etf_3x', 'ytd26'): (0.690, 1.120, 1.40, -0.385, 2.91, 0.723, 73.3, 41, 0.537, 18.8, 177),
    ('hs300', 'full'):  (9.860, 0.520, 1.39, -0.306, 1.70, 0.359, 76.1, 644, 0.615, 7.1, 1384),
    ('hs300', 'is'):    (4.603, 0.541, 1.48, -0.306, 1.77, 0.343, 54.9, 319, 0.589, 7.8, 968),
    ('hs300', 'oos'):   (0.984, 0.495, 1.26, -0.296, 1.67, 0.393, 125.5, 321, 0.651, 6.4, 415),
    ('hs300', 'ytd26'): (0.127, 0.187, 0.64, -0.296, 0.63, 0.399, 126.6, 136, 0.618, 6.2, 172),
    ('a_etf_all', 'full'):  (2.696, 0.258, 0.73, -0.227, 1.13, 0.406, 15.8, 34, 0.824, 190.9, 1384),
    ('a_etf_all', 'is'):    (1.359, 0.240, 0.64, -0.227, 1.06, 0.472, 12.3, 16, 0.688, 207.8, 968),
    ('a_etf_all', 'oos'):   (0.587, 0.312, 1.71, -0.158, 1.98, 0.173, 24.2, 20, 0.850, 82.2, 415),
    ('a_etf_all', 'ytd26'): (0.292, 0.445, 2.26, -0.112, 3.98, 0.174, 28.6, 10, 0.900, 56.7, 172),
}
# 等权买入持有基准(旧报告 §8 第二张表): total_ret / cagr / sharpe / max_dd / ann_turnover / n_days
OLD_BH_COLS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'ann_turnover', 'n_days']
OLD_BUYHOLD = {
    ('company_300', 'full'):  (6.994, 0.441, 1.35, -0.389, 1.5, 1431),
    ('company_300', 'ytd26'): (0.065, 0.095, 0.44, -0.204, 1.4, 176),
    ('company_1000', 'full'):  (2.570, 0.250, 1.25, -0.258, 1.3, 1433),
    ('company_1000', 'ytd26'): (0.141, 0.207, 1.28, -0.080, 2.7, 178),
    ('etf_3x', 'full'):  (1.279, 0.156, 0.60, -0.553, 0.7, 1432),
    ('etf_3x', 'ytd26'): (0.212, 0.316, 1.01, -0.168, 1.8, 177),
    ('hs300', 'full'):  (0.823, 0.111, 0.64, -0.239, 0.5, 1384),
    ('hs300', 'ytd26'): (-0.062, -0.088, -0.57, -0.109, 1.5, 172),
    ('a_etf_all', 'full'):  (0.020, 0.004, 0.12, -0.397, 0.9, 1384),
    ('a_etf_all', 'ytd26'): (-0.073, -0.103, -0.56, -0.143, 1.6, 172),
}

# 四窗口(与旧报告一致: IS / OOS / FULL / 2026 年初独立窗)
WINDOWS = (
    ('is', cfg.TRADE_START, cfg.IS_END),
    ('oos', cfg.OOS_START, None),
    ('full', cfg.TRADE_START, None),
    ('ytd26', '2026-01-01', None),
)


# ============================ 旧因子(逐字照抄 git HEAD 公式) ============================

def _rk(v: pd.DataFrame) -> pd.DataFrame:
  """旧 C_tmqmom 用的截面 pct 排名: rk = lambda v: v.rank(axis=1, pct=True)."""
  return v.rank(axis=1, pct=True)


def _range20(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
  return (h - l).rolling(20).mean() / c


def _idiovol60(c: pd.DataFrame) -> pd.DataFrame:
  """F_idiovol60: 对池等权收益回归取残差 -> 60 日残差 std(旧/当前同式)."""
  ret1 = c / c.shift(1) - 1.0
  pool_ret = ret1.mean(axis=1)
  var_pool = pool_ret.rolling(60).var().replace(0, np.nan)
  beta60 = ret1.rolling(60).cov(pool_ret).div(var_pool, axis=0)
  resid = ret1.sub(beta60.mul(pool_ret, axis=0))
  return resid.rolling(60).std()


def legacy_factors(ds) -> dict:
  """旧系统因子: 严格按 bc_factor_search / bc_combo_search 的定义复现."""
  o, h, l, c = ds.wide('open'), ds.wide('high'), ds.wide('low'), ds.wide('close')
  v = ds.wide('volume')
  ret1 = c / c.shift(1) - 1.0
  out = {}

  path = c.diff().abs().rolling(20).sum()                       # F_er20
  out['F_er20'] = (c - c.shift(20)).abs() / path.replace(0, np.nan)

  out['F_mom121'] = c.shift(21) / c.shift(252) - 1.0            # F_mom121

  obv_cs = (np.sign(ret1.fillna(0.0)) * v.fillna(0.0)).cumsum()  # F_obv20
  out['F_obv20'] = (obv_cs - obv_cs.shift(20)) / v.rolling(20).sum().replace(0, np.nan)

  rng = _range20(h, l, c)                                       # F_range20 / N_range20
  out['F_range20'] = rng
  out['N_range20'] = -rng

  iv = _idiovol60(c)                                            # F_idiovol60 / N_idiovol60
  out['F_idiovol60'] = iv
  out['N_idiovol60'] = -iv

  out['D_ma20_dist'] = c / c.rolling(20).mean() - 1.0           # D_ma20_dist

  out['C_tmqmom'] = _rk(out['F_mom121']) + _rk(out['F_er20'])   # C_tmqmom

  var20 = (c / c.shift(20) - 1.0).rolling(120, min_periods=60).var()   # G_vr20
  var1 = ret1.rolling(120, min_periods=60).var()
  out['G_vr20'] = var20 / (20.0 * var1.replace(0, np.nan))

  overnight = o / c.shift(1) - 1.0                              # G_oviv20
  intraday = c / o - 1.0
  out['G_oviv20'] = (overnight.rolling(20, min_periods=10).std()
                     / intraday.rolling(20, min_periods=10).std().replace(0, np.nan))

  out['H_ichimoku_alpha'] = fct._alphaize(ds.wide('ichimoku_distance'))   # H_ichimoku_alpha
  return out


# ============================ 旧 composite 口径(bc_backtest.compute_composite) ============================

def legacy_composite(facs: dict, weights: dict) -> pd.DataFrame:
  """Σ (w/Σ|w|) * rank(axis=1, pct=True); 缺成分以 0 参与, 全缺 -> fillna(0.5)."""
  total = float(sum(abs(w) for w in weights.values()))
  if total <= 0:
    raise ValueError('权重绝对值和为 0')
  comp = None
  for col, w in weights.items():
    if col not in facs:
      raise ValueError(f'组合分数成分列不存在: {col}')
    wide = facs[col].apply(pd.to_numeric, errors='coerce')   # 已是 dates x symbols 宽表
    pct = wide.rank(axis=1, pct=True)
    part = (w / total) * pct
    comp = part if comp is None else comp.add(part, fill_value=0.0)
  return comp.fillna(0.5)


# ============================ 当前系统口径(注册表因子 + prep(rank) + combine_signals) ============================

def native_signals(ds, min_cs: int = cfg.MIN_CS) -> dict:
  """当前系统信号: 键名沿用旧预设名, 以便与旧权重直接对齐同权合成."""
  pf = prep._make_prep(prep.DEFAULT_PREP, min_cs)
  c, h, l = ds.wide('close'), ds.wide('high'), ds.wide('low')
  reg = lambda n: fct.get_factor(n).compute(ds)                 # noqa: E731
  mom, er = reg('mom_12_1'), reg('er_20')
  raw = {
      'F_er20': er,                                             # er_20 == F_er20
      'F_mom121': mom,                                          # mom_12_1 ≈ F_mom121(shift 20 vs 21)
      'F_obv20': reg('obv_20'),
      'D_ma20_dist': reg('bias_20'),                            # bias_20 == D_ma20_dist
      'G_vr20': reg('G_vr20'),
      'H_ichimoku_alpha': reg('H_ichimoku_alpha'),
      'G_oviv20': reg('G_oviv20'),
      'C_tmqmom': _rk(mom) + _rk(er),                           # 由注册表动量/效率合成
      'N_range20': -_range20(h, l, c),                          # 注册表无 range20, 同式内联
      'N_idiovol60': -_idiovol60(c),                            # 注册表无 idiovol60, 同式内联
  }
  return {k: pf(v) for k, v in raw.items()}


# ============================ 回测 ============================

def engine_params(top_k: int) -> bt.EngineParams:
  return bt.EngineParams(top_k=top_k, exit_rank=EXIT_RANK, sizing='tier',
                         max_exposure=1.0, per_symbol_cap=CAP,
                         cost_bps=COST_BPS, band=BAND)


def run_pool(pool: str, variants: list, exclude: list = None) -> tuple:
  """跑一个池的全部窗口 x 变体; 返回 (结果行列表, 数据区间描述)."""
  weights, top_k = PRESETS[pool]
  ds = load_pool(pool, exclude=exclude)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  p = engine_params(top_k)

  comps = {}
  if 'legacy' in variants:
    comps['legacy'] = legacy_composite(legacy_factors(ds), weights)
  if 'native' in variants:
    comps['native'] = cmb.combine_signals(native_signals(ds), weights, min_cs=cfg.MIN_CS)

  rows = []
  for win, ws, we in WINDOWS:
    ow, cw = bt._win(open_w, ws, we), bt._win(close_w, ws, we)
    for name, comp in comps.items():
      res = bt.run_engine(ow, cw, bt._win(comp, ws, we), p)
      s = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      rows.append({'source': name, 'window': win,
                   **{k: s.get(k) for k in OLD_COLS}})
    # 等权买入持有基准(同一引擎)
    bh = bt.perf_stats(*_bh_args(bt.buyhold(ow, cw)))
    rows.append({'source': 'buyhold', 'window': win,
                 **{k: bh.get(k) for k in OLD_COLS}})
  actual = (f'{ds.dates.min():%Y-%m-%d}~{ds.dates.max():%Y-%m-%d}')
  return rows, actual, top_k


def _bh_args(res: dict) -> tuple:
  return res['equity'], res['ann_turnover'], res['trades']


# ============================ 输出 ============================

def _fmt_rows(pool: str, top_k: int, rows: list) -> pd.DataFrame:
  df = pd.DataFrame(rows)
  # 把旧报告基准拼进来(仅策略行有基准; buyhold 另表)
  old = []
  for r in rows:
    key = (pool, r['window'])
    if r['source'] in ('legacy', 'native') and key in OLD_REPORT:
      old.append(dict(zip(OLD_COLS, OLD_REPORT[key])))
    elif r['source'] == 'buyhold' and key in OLD_BUYHOLD:
      o = dict(zip(OLD_BH_COLS, OLD_BUYHOLD[key]))
      old.append({c: o.get(c, np.nan) for c in OLD_COLS})
    else:
      old.append({c: np.nan for c in OLD_COLS})
  df['top_k'] = top_k
  for c in OLD_COLS:
    df[f'old_{c}'] = [o.get(c, np.nan) for o in old]
  return df


def main():
  ap = argparse.ArgumentParser(description='用当前引擎复现旧系统 5 组预设权重(同权重对照)')
  ap.add_argument('--pools', default=','.join(PRESETS.keys()),
                  help='逗号分隔的池名')
  ap.add_argument('--variant', default='both', choices=('legacy', 'native', 'both'))
  ap.add_argument('--exclude', default=None,
                  help='剔除标的(逗号分隔; 旧报告未剔除任何标的)')
  ap.add_argument('--out', default=None, help='可选: 结果 CSV 输出路径')
  a = ap.parse_args()

  pools = [s.strip() for s in a.pools.split(',') if s.strip()]
  variants = ['legacy', 'native'] if a.variant == 'both' else [a.variant]
  exclude = [s.strip() for s in a.exclude.split(',')] if a.exclude else None

  log = cfg.get_logger('repro')
  frames = []
  for pool in pools:
    if pool not in PRESETS:
      log.warning(f'跳过未知池: {pool} (可选 {list(PRESETS)})')
      continue
    rows, actual, top_k = run_pool(pool, variants, exclude)
    for r in rows:
      r['pool'] = pool
      r['actual_range'] = actual
    t = _fmt_rows(pool, top_k, rows)
    frames.append(t)
    cols = ['source', 'window'] + OLD_COLS + [f'old_{c}' for c in OLD_COLS]
    print(f'\n===== {pool}  top_k={top_k}  weights={PRESETS[pool][0]}  '
          f'数据 {actual} =====')
    print(t[cols].to_string(index=False, float_format=lambda x: f'{x:g}'))

  table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

  if a.out and not table.empty:
    table.to_csv(a.out, index=False, encoding='utf-8-sig')
    log.info(f'结果已写入: {a.out}')

  _print_verdict(table, variants)


def _print_verdict(df: pd.DataFrame, variants: list):
  """同权重对照结论: 逐池逐窗看 legacy/native 与旧报告的偏离."""
  if df.empty:
    return
  print('\n===== 同权重对照小结(策略行, 关键指标 |new-old|) =====')
  main_v = 'legacy' if 'legacy' in variants else variants[0]
  sub = df[df['source'] == main_v].copy()
  for c in ('total_ret', 'sharpe', 'ann_turnover', 'n_days'):
    sub[f'd_{c}'] = sub[c] - sub[f'old_{c}']
  keep = ['pool', 'window', 'total_ret', 'old_total_ret', 'd_total_ret',
          'sharpe', 'old_sharpe', 'd_sharpe',
          'ann_turnover', 'old_ann_turnover', 'd_ann_turnover',
          'n_days', 'old_n_days', 'd_n_days']
  print(sub[keep].to_string(index=False, float_format=lambda x: f'{x:g}'))


if __name__ == '__main__':
  main()

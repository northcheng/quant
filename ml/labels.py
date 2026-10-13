# -*- coding: utf-8 -*-
"""
ml.labels — 标签层(第2步)
==========================
职责: 造"模型要预测的答案".

--- 标签方向与特征恰好相反: 特征禁未来, 标签必须用未来 ---
  特征(第1步)只在 t 时刻已知; 标签是"t 时刻决策之后真正发生的事情". 所以:
    信号日 t 收盘决策 -> t+1 开盘入场 -> t+1+h 开盘出场(可交易口径)
  这条口径直接复用 factor.evaluate.forward_return, 与 factor 的 IC/回测完全一致,
  保证 ml 与 factor 的结论可以放在同一把尺子上比.

--- 为什么把连续收益再转成截面 rank(默认) ---
  直接预测收益有几个麻烦: 收益重尾(几个大涨的票主导 MSE)、不同日期的收益量级不同、
  且与 factor 的 RankIC 口径不直接可比.
  改成**逐日截面 rank** -> [-1, 1] 之后:
    - 与 factor 的 IC(RankIC) 同构: 模型输出的截面分数与 rank 标签的截面相关就是
      模型版 RankIC, 可直接和因子对照;
    - 对极值免疫、跨日可比;
    - 学的是"这天里谁相对强", 正是选股真正需要的东西.
  (method='z' 提供逐日截面 zscore 作为备选, 但默认用 rank.
   method='raw' 直接学原始前向收益(仅逐日截面截尾), 用于"贴近收益目标"的对照实验.)

自检:
  cd ~/git && python -m quant.ml.labels --pool etf_3x
  cd ~/git && python -m quant.ml.labels --pool etf_3x --check
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor import config as fcfg
from quant.factor import evaluate as ev
from quant.factor import prepare as prep
from quant.factor.data import Dataset

from quant.ml import config as cfg
from quant.ml.data import load

# 标签方法 -> factor.prepare 的方法名(两者对本包而言是同一件事: 逐日截面变换)
_LABEL_TO_PREP = {'rank': 'rank', 'z': 'z'}

# 'raw' 标签: 直接学原始前向收益(仅逐日截面截尾压重尾, 不做 demean/标准化).
# 截尾分位 0.01 = 掐掉每日两头各 1% 的极值; 只削尾巴、不改量级, 保留"收益目标"的原始含义.
RAW_WINSOR_Q = 0.01


def forward_return(ds: Dataset, h: int = 1) -> pd.DataFrame:
  """
  可交易前向收益(直接复用 factor.evaluate 的口径).

  :param ds: Dataset
  :param h: 持有天数(开盘到开盘)
  :returns: (date × symbol) 宽表, 值 = 入场到出场简单收益
  """
  return ev.forward_return(ds, h)


def build_labels(ds: Dataset, h: int = cfg.DEFAULT_HORIZON,
                 method: str = cfg.DEFAULT_LABEL,
                 min_cs: int = fcfg.MIN_CS, log=None) -> pd.DataFrame:
  """
  构建标签宽表: 前向收益 -> 逐日截面变换(rank / z).

  :param ds: Dataset
  :param h: 持有天数(预测跨度)
  :param method: 'rank'(截面秩 -> [-1,1], 默认) / 'z'(截面 zscore) /
                 'raw'(原始前向收益, 仅逐日截面截尾)
  :param min_cs: 每日截面最小样本数(有效标的不足的日子整行作废)
  :param log: 可选 logger
  :returns: (date × symbol) 宽表, 模型预测目标
  """
  if method not in cfg.LABEL_METHODS:
    raise KeyError(f'未知标签方法: {method} (可选 {cfg.LABEL_METHODS})')
  fwd = forward_return(ds, h)
  if method == 'raw':
    # 先截尾压重尾, 再屏蔽薄截面(与其它方法共用同一 min_cs 口径)
    lab = prep.guard_rows(prep.winsorize(fwd, 'quantile', q=RAW_WINSOR_Q), min_cs)
  else:
    lab = prep.prepare(fwd, _LABEL_TO_PREP[method], min_cs)
  if log is not None:
    v = lab.to_numpy(dtype=float)
    log.info(f'[labels] h={h}, method={method}: 有效值占比 {np.isfinite(v).mean():.1%}, '
             f'取值范围 [{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]')
  return lab


# ============================ 防"反向前视": 标签必须吃到未来 ============================

def verify_label_forward(ds: Dataset, h: int = cfg.DEFAULT_HORIZON,
                         tol: float = 1e-12) -> dict:
  """
  标签口径自检: 确认标签**确实**用了未来数据(否则等于预测"已经发生的事").

  三条检查(全部通过才判定口径正确):
    1. 末尾 h+1 行必须为 NaN —— 因为 t+1+h 的开盘价还没出现, 拿不到答案;
    2. 手工重算一个样本: label_raw(t, s) == open(t+1+h, s)/open(t+1, s) - 1;
    3. 截断性: 把面板截到某日 cut 后, "未来越界"的样本必须变 NaN
       (证明标签真的依赖 t 之后的行情, 而不是当期已能算出的东西).

  :param ds: Dataset
  :param h: 持有天数
  :param tol: 逐值比对容差
  :returns: dict(ok, tail_nan_ok, manual_ok, truncation_ok, details...)
  """
  fwd = forward_return(ds, h)
  n = len(ds.dates)

  # 1. 末尾 h+1 行应为 NaN
  tail = fwd.tail(h + 1).to_numpy(dtype=float)
  tail_nan_ok = bool(np.isnan(tail).all())

  # 2. 手工重算: 取倒数第 (h+2) 个交易日(其未来完整可得)
  open_ = ds.wide('open')
  t_pos = n - (h + 2)
  manual_ok, manual_diff, probe = False, float('nan'), None
  if t_pos > 0:
    t = fwd.index[t_pos]
    sym = open_.iloc[t_pos].dropna().index
    if len(sym):
      s = sym[0]
      got = float(fwd.loc[t, s])
      entry = float(open_.iloc[t_pos + 1][s])
      exit_ = float(open_.iloc[t_pos + 1 + h][s])
      want = exit_ / entry - 1.0
      manual_diff = abs(got - want)
      manual_ok = bool(manual_diff <= 1e-9)
      probe = {'date': str(t.date()), 'symbol': s, 'label': round(got, 6),
               'manual': round(want, 6)}

  # 3. 截断性: 截到某日(留出至少 h+1 天)后, 边界样本应变 NaN
  cut = fwd.index[n - 2 * (h + 1)] if n > 3 * (h + 1) else fwd.index[max(n // 2, 1)]
  part = ds.slice(end=cut)
  fwd_part = forward_return(part, h)
  t_edge = fwd_part.index[-1] if len(fwd_part.index) else None
  truncation_ok = False
  if t_edge is not None:
    a = fwd.loc[t_edge].to_numpy(dtype=float)
    b = fwd_part.loc[t_edge].to_numpy(dtype=float)
    # 截断后该日应整体为 NaN(其未来被切断), 而全历史版本有效
    truncation_ok = bool(np.isnan(b).all() and np.isfinite(a).any())

  return {'ok': bool(tail_nan_ok and manual_ok and truncation_ok),
          'h': h, 'tail_nan_ok': tail_nan_ok, 'manual_ok': manual_ok,
          'truncation_ok': truncation_ok, 'manual_diff': manual_diff,
          'probe': probe, 'cut': str(cut.date())}


def main():
  ap = argparse.ArgumentParser(description='ml 标签层自检: 构建标签 + 前视口径检验')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  ap.add_argument('--check', action='store_true', help='做"标签必须吃未来"检验')
  a = ap.parse_args()

  log = cfg.get_logger('ml.labels')
  ds = load(a.pool, a.interval)
  log.info(f'池 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日, '
           f'区间 {ds.dates.min():%Y-%m-%d} ~ {ds.dates.max():%Y-%m-%d}')

  log.info('\n[各跨度标签覆盖] 有效值占比随 h 变化')
  rows = []
  for h in cfg.HORIZONS:
    fwd = forward_return(ds, h)
    rows.append({'h': h, 'valid_ratio': round(float(fwd.notna().mean().mean()), 3),
                 'expect_tail_nan': h + 1})
  log.info('\n' + pd.DataFrame(rows).to_string(index=False))

  lab = build_labels(ds, a.h, a.method, a.min_cs, log=log)
  log.info(f'\n[labels] h={a.h} 最近 5 日截面(前 5 列):\n'
           f'{lab.tail(5).iloc[:, :5].to_string()}')

  if a.check:
    res = verify_label_forward(ds, a.h)
    log.info(f'\n[标签口径检验] h={a.h}')
    log.info(f'  末尾 h+1 行全 NaN : {res["tail_nan_ok"]}')
    log.info(f'  手工重算一致      : {res["manual_ok"]} (diff={res["manual_diff"]:.2e})')
    if res['probe']:
      log.info(f'    探针 {res["probe"]}')
    log.info(f'  截断后边界样本失效: {res["truncation_ok"]} (cut={res["cut"]})')
    log.info('[结论] 标签口径正确(确实用了未来)' if res['ok']
             else '[结论] 标签口径异常, 需排查')


if __name__ == '__main__':
  main()